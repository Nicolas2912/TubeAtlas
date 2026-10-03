import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Segment } from '../../../shared/api.ts';
import { estimateTokens } from '../../../shared/units.ts';
import type { Db } from '../db.ts';
import { createOpenRouter, type EmbedResult } from '../integrations/openrouter.ts';
import { createTestApp } from '../testing.ts';
import { createRetrieval, type EmbeddingLog } from './retrieval.ts';
import { saveTranscript } from './transcripts.ts';

function transcript(db: Db, videoId: number, label: string, count = 5) {
  const segments: Segment[] = Array.from({ length: count }, (_, id) => ({
    id, start: id * 4, end: id * 4 + 3,
    text: `>> ${label}${id} ${'inventedcaptionword '.repeat(44).trim()}`,
  }));
  return saveTranscript(db, videoId, { source: 'youtube', language: 'en', timed: true, segments });
}
function video(db: Db, id: number) {
  db.prepare('INSERT INTO videos (id, youtube_id, title) VALUES (?, ?, ?)').run(id, `ExampleId${id}`, `Invented video ${id}`);
}
function fakeEmbedder() {
  const calls: { model: string; input: string[] }[] = [];
  const provider = { embed: async ({ model, input }: { model: string; input: string[] }): Promise<EmbedResult> => {
    calls.push({ model, input });
    return { model: 'provider/model-alias', usage: { promptTokens: 10, completionTokens: 0, reasoningTokens: 0, cost: 0.001 },
      vectors: input.map((text) => text === 'lantern query' || text.startsWith('lantern2 ') || text.startsWith('foreign') || text.startsWith('replacement') ? new Float32Array([3, 4]) : new Float32Array([4, -3])) };
  } };
  return { calls, provider };
}

test('retrieval scores only the requested revision, adds neighbours in order, and obeys whole-unit budgets', async (t) => {
  const app = createTestApp(); t.after(app.cleanup);
  video(app.db, 1); video(app.db, 2);
  const first = transcript(app.db, 1, 'lantern');
  const newer = transcript(app.db, 1, 'replacement');
  const other = transcript(app.db, 2, 'foreign');
  const { provider, calls } = fakeEmbedder(); const log: EmbeddingLog[] = [];
  const service = createRetrieval({ db: app.db, provider, model: 'configured/model', log: (entry) => log.push(entry) });
  await service.ensureChunks(other); await service.ensureChunks(newer);
  const result = await service.retrieve(first, 'lantern query', { topK: 1 });
  assert.deepEqual(result.map((u) => u.text.split(' ')[0]), ['lantern1', 'lantern2', 'lantern3']);
  assert.deepEqual(result.map((u) => u.start), [4, 8, 12]);
  assert.equal(new Set(result.map((u) => u.id)).size, 3);
  assert.deepEqual((await service.retrieve(first, 'lantern query')).map((u) => u.text.split(' ')[0]), ['lantern0', 'lantern1', 'lantern2', 'lantern3', 'lantern4']);
  const oneUnitBudget = estimateTokens(result[0]!.text);
  const limited = await service.retrieve(first, 'lantern query', { topK: 1, budgetTokens: oneUnitBudget });
  assert.deepEqual(limited, result.slice(0, 1));
  assert.equal(limited.reduce((n, u) => n + estimateTokens(u.text), 0), oneUnitBudget);
  assert.deepEqual(await service.retrieve(first, 'lantern query', { topK: 1, budgetTokens: oneUnitBudget - 1 }), []);
  const before = calls.length;
  assert.deepEqual(await service.retrieve(first, '   '), []);
  assert.deepEqual(await service.retrieve(first, 'lantern query', { budgetTokens: 0 }), []);
  assert.equal(calls.length, before);
  await assert.rejects(service.retrieve(first, 'query', { topK: -1 }), /non-negative integers/u);
  assert.ok(log.every((entry) => entry.cost === 0.001 && entry.requestedModel === 'configured/model' && entry.model === 'provider/model-alias'));
  assert.deepEqual(app.db.prepare('SELECT DISTINCT embedding_model FROM chunks').all(), [{ embedding_model: 'configured/model' }]);
});

test('chunks persist as normalized float32 bytes, reuse across service restarts, and rebuild on model or version changes', async (t) => {
  const app = createTestApp(); t.after(app.cleanup); video(app.db, 1);
  const id = transcript(app.db, 1, 'lantern'); const { provider, calls } = fakeEmbedder();
  const make = (model: string) => createRetrieval({ db: app.db, provider, model, log: () => {} });
  await make('model-one').ensureChunks(id);
  assert.equal(calls.length, 1);
  const rows = app.db.prepare('SELECT embedding, embedding_dim, start_seconds, end_seconds FROM chunks WHERE transcript_id = ? ORDER BY ord').all(id) as { embedding: Buffer; embedding_dim: number; start_seconds: number; end_seconds: number }[];
  assert.equal(rows.length, 5);
  assert.equal(rows[0]!.embedding_dim, 2);
  assert.equal(rows[0]!.embedding.length, 8);
  assert.ok(Math.abs(Math.hypot(rows[0]!.embedding.readFloatLE(0), rows[0]!.embedding.readFloatLE(4)) - 1) < 0.000001);
  assert.deepEqual([rows[0]!.start_seconds, rows[0]!.end_seconds], [0, 3]);
  await make('model-one').ensureChunks(id); assert.equal(calls.length, 1);
  await make('model-two').ensureChunks(id); assert.equal(calls.length, 2);
  assert.equal(calls.at(-1)!.model, 'model-two');
  assert.deepEqual(app.db.prepare('SELECT DISTINCT embedding_model FROM chunks WHERE transcript_id = ?').all(id), [{ embedding_model: 'model-two' }]);
  app.db.prepare('UPDATE chunks SET units_version = 0 WHERE transcript_id = ?').run(id);
  await make('model-two').ensureChunks(id); assert.equal(calls.length, 3);
  assert.deepEqual(app.db.prepare('SELECT DISTINCT units_version FROM chunks').all(), [{ units_version: 1 }]);
  app.db.prepare('DELETE FROM chunks WHERE transcript_id = ? AND ord = 2').run(id);
  await make('model-two').ensureChunks(id); assert.equal(calls.length, 4);
  const all = await make('model-two').retrieve(id, 'lantern query');
  assert.equal(all.length, 5); // Overlapping hit neighbourhoods do not duplicate units.
});

test('chunking packs consecutive units to 350 tokens and embeds batches of at most 64, sharing concurrent builds', async (t) => {
  const app = createTestApp(); t.after(app.cleanup); video(app.db, 1);
  const large = transcript(app.db, 1, 'lantern', 130); const { provider, calls } = fakeEmbedder();
  const service = createRetrieval({ db: app.db, provider, model: 'model', log: () => {} });
  const [chunks, same] = await Promise.all([service.ensureChunks(large), service.ensureChunks(large)]);
  assert.deepEqual(calls.map((c) => c.input.length), [64, 64, 2]);
  assert.equal(chunks, same);
  assert.ok(chunks.every((c) => estimateTokens(c.text) <= 350));
  const small = saveTranscript(app.db, 1, { source: 'youtube', language: 'en', timed: true,
    segments: Array.from({ length: 8 }, (_, id) => ({ id, start: id, end: id + 1, text: `>> short${id} text` })) });
  const packed = await service.ensureChunks(small);
  assert.equal(packed.length, 1);
  assert.deepEqual(packed[0]!.units.map((u) => u.id), ['u001', 'u002', 'u003', 'u004', 'u005', 'u006', 'u007', 'u008']);
});

test('failed or invalid embeddings preserve the previous cache and a later explicit attempt can succeed', async (t) => {
  const app = createTestApp(); t.after(app.cleanup); video(app.db, 1);
  const id = transcript(app.db, 1, 'lantern', 65); const { provider } = fakeEmbedder();
  await createRetrieval({ db: app.db, provider, model: 'old', log: () => {} }).ensureChunks(id);
  let fail = true, batches = 0;
  const service = createRetrieval({ db: app.db, model: 'new', log: () => {}, provider: { embed: async (args) => {
    if (fail && ++batches === 2) throw new Error('Invented provider failure');
    return provider.embed(args);
  } } });
  await assert.rejects(service.ensureChunks(id), /Invented provider failure/u);
  assert.deepEqual(app.db.prepare('SELECT DISTINCT embedding_model FROM chunks').all(), [{ embedding_model: 'old' }]);
  fail = false; await service.ensureChunks(id);
  assert.deepEqual(app.db.prepare('SELECT DISTINCT embedding_model FROM chunks').all(), [{ embedding_model: 'new' }]);
  for (const vectors of [[], [new Float32Array()], [new Float32Array([0, 0])], [new Float32Array([NaN, 1])]]) {
    const invalid = createRetrieval({ db: app.db, model: 'bad', log: () => {}, provider: { embed: async () => ({ vectors, model: 'bad', usage: null }) } });
    await assert.rejects(invalid.ensureChunks(id), /Embedding vectors/u);
    assert.deepEqual(app.db.prepare('SELECT DISTINCT embedding_model FROM chunks').all(), [{ embedding_model: 'new' }]);
  }
  await assert.rejects(service.ensureChunks(999), /no longer exists/u);
});

test('real provider adapter, SQLite persistence, and retrieval compose without any network calls', async (t) => {
  const app = createTestApp(); t.after(app.cleanup); video(app.db, 1);
  const id = transcript(app.db, 1, 'lantern'); let requests = 0; const logs: EmbeddingLog[] = [];
  const provider = createOpenRouter({ apiKey: 'invented-key', baseUrl: 'https://provider.invalid', fetch: async (_url, init) => {
    requests++;
    const body = JSON.parse(String(init?.body));
    assert.equal(body.model, 'configured/model');
    return Response.json({ model: 'model-alias', data: body.input.map((text: string, index: number) => ({ index, embedding: text.startsWith('lantern2 ') || text === 'lantern query' ? [3, 4] : [4, -3] })).reverse(), usage: { prompt_tokens: 10 } });
  } });
  const service = createRetrieval({ db: app.db, provider, model: 'configured/model', log: (entry) => logs.push(entry) });
  assert.deepEqual((await service.retrieve(id, 'lantern query', { topK: 1 })).map((u) => u.id), ['u002', 'u003', 'u004']);
  assert.equal(requests, 2);
  assert.deepEqual(logs.map((l) => [l.purpose, l.cost]), [['chunks', null], ['query', null]]);
  await service.retrieve(id, 'lantern query', { topK: 1 });
  assert.equal(requests, 3); // Query embedded again; chunks already durable.
});

test('query dimension mismatch is explicit, empty transcripts are free, and deleted transcripts are not re-indexed', async (t) => {
  const app = createTestApp(); t.after(app.cleanup); video(app.db, 1);
  const id = transcript(app.db, 1, 'lantern'); const { provider, calls } = fakeEmbedder();
  await createRetrieval({ db: app.db, provider, model: 'model', log: () => {} }).ensureChunks(id);
  const bad = createRetrieval({ db: app.db, model: 'model', log: () => {}, provider: { embed: async () => ({ model: 'model', usage: null, vectors: [new Float32Array([1, 0, 0])] }) } });
  await assert.rejects(bad.retrieve(id, 'query'), /incompatible dimensions/u);
  const empty = saveTranscript(app.db, 1, { source: 'paste_text', timed: false, language: null, segments: [] });
  const service = createRetrieval({ db: app.db, provider, model: 'model', log: () => {} });
  const before = calls.length;
  assert.deepEqual(await service.retrieve(empty, 'query'), []); assert.equal(calls.length, before);
  const deleting = createRetrieval({ db: app.db, model: 'other', log: () => {}, provider: { embed: async (args) => {
    app.db.prepare('DELETE FROM videos WHERE id = 1').run(); return provider.embed(args);
  } } });
  await assert.rejects(deleting.ensureChunks(id), /no longer exists/u);
  assert.deepEqual(app.db.prepare('SELECT count(*) AS n FROM chunks').get(), { n: 0 });
});

test('cancelling one shared build preserves the other caller; the last cancellation stops embedding without writing a cache', async (t) => {
  const app = createTestApp(); t.after(app.cleanup); video(app.db, 1); const id = transcript(app.db, 1, 'lantern');
  const requests: { signal: AbortSignal; resolve: (result: EmbedResult) => void }[] = [];
  const service = createRetrieval({ db: app.db, model: 'model', log: () => {}, provider: { embed: ({ input, signal }) => new Promise((resolve, reject) => {
    requests.push({ signal: signal!, resolve }); signal!.addEventListener('abort', () => reject(signal!.reason), { once: true });
  }) } });
  const first = new AbortController(), second = new AbortController();
  const a = service.ensureChunks(id, first.signal), b = service.ensureChunks(id, second.signal);
  first.abort(); await assert.rejects(a, { name: 'AbortError' }); assert.equal(requests[0]!.signal.aborted, false);
  requests[0]!.resolve({ model: 'model', usage: null, vectors: Array.from({ length: 5 }, () => new Float32Array([1, 0])) });
  assert.equal((await b).length, 5);
  app.db.prepare('DELETE FROM chunks').run();
  const last = new AbortController(); const c = service.ensureChunks(id, last.signal);
  last.abort(); await assert.rejects(c, { name: 'AbortError' }); assert.equal(requests[1]!.signal.aborted, true);
  assert.deepEqual(app.db.prepare('SELECT count(*) AS n FROM chunks').get(), { n: 0 });
  const retry = service.ensureChunks(id);
  requests[2]!.resolve({ model: 'model', usage: null, vectors: Array.from({ length: 5 }, () => new Float32Array([1, 0])) });
  assert.equal((await retry).length, 5);
});
