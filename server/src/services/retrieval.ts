import type { Segment } from '../../../shared/api.ts';
import { estimateTokens, UNITS_VERSION, type Unit } from '../../../shared/units.ts';
import { tx, type Db } from '../db.ts';
import { AppError } from '../errors.ts';
import type { OpenRouter, Usage } from '../integrations/openrouter.ts';
import { buildUnits } from './units.ts';

type Chunk = { ord: number; units: Unit[]; text: string; start: number | null; end: number | null; embedding: Float32Array };
type StoredChunk = {
  ord: number; units_version: number; unit_ids_json: string; text: string;
  embedding: Buffer; embedding_model: string; embedding_dim: number;
};
export type EmbeddingLog = {
  purpose: 'chunks' | 'query'; transcriptId: number; requestedModel: string; model: string;
  inputs: number; cost: number | null; usage: Usage | null;
};

/** On-demand, revision-scoped retrieval. Merely reading a transcript never calls the provider. */
export function createRetrieval(options: {
  db: Db; provider: Pick<OpenRouter, 'embed'>; model: string;
  log?: (entry: EmbeddingLog) => void;
}) {
  const { db, provider, model } = options;
  const log = options.log ?? ((entry: EmbeddingLog) => console.info('Embedding usage', entry));
  const building = new Map<number, { promise: Promise<Chunk[]>; controller: AbortController; users: number }>();

  async function embed(transcriptId: number, input: string[], purpose: EmbeddingLog['purpose'], signal?: AbortSignal) {
    signal?.throwIfAborted();
    const result = await provider.embed({ model, input, signal });
    // Record reported cost even if the returned vectors subsequently fail validation.
    log({ purpose, transcriptId, requestedModel: model, model: result.model, inputs: input.length, cost: result.usage?.cost ?? null, usage: result.usage });
    if (result.vectors.length !== input.length) badVectors();
    return result.vectors.map(normalizedVector);
  }

  async function build(transcriptId: number, signal: AbortSignal): Promise<Chunk[]> {
    const row = db.prepare('SELECT segments_json FROM transcripts WHERE id = ?').get(transcriptId) as { segments_json: string } | undefined;
    if (!row) throw new AppError(404, 'NO_TRANSCRIPT', 'That transcript revision no longer exists.');
    const { units } = buildUnits(JSON.parse(row.segments_json) as Segment[]);
    const chunks: Chunk[] = [];
    for (const unit of units) {
      let chunk = chunks.at(-1);
      if (!chunk || estimateTokens(`${chunk.text} ${unit.text}`) > 350) {
        chunk = { ord: chunks.length, units: [], text: '', start: unit.start, end: unit.end, embedding: new Float32Array() };
        chunks.push(chunk);
      }
      chunk.units.push(unit);
      chunk.text += `${chunk.text ? ' ' : ''}${unit.text}`;
      if (unit.end !== null) chunk.end = Math.max(chunk.end ?? 0, unit.end);
    }
    const stored = db.prepare('SELECT ord, units_version, unit_ids_json, text, embedding, embedding_model, embedding_dim FROM chunks WHERE transcript_id = ? ORDER BY ord').all(transcriptId) as StoredChunk[];
    if (stored.length === chunks.length && stored.every((s, i) =>
      s.ord === i && s.embedding_model === model && s.units_version === UNITS_VERSION &&
      s.text === chunks[i]!.text && s.unit_ids_json === JSON.stringify(chunks[i]!.units.map((u) => u.id)) &&
      s.embedding_dim > 0 && s.embedding.length === s.embedding_dim * 4 && s.embedding_dim === stored[0]!.embedding_dim)) {
      try {
        stored.forEach((s, i) => {
          const vector = new Float32Array(s.embedding_dim);
          new Uint8Array(vector.buffer).set(s.embedding);
          chunks[i]!.embedding = normalizedVector(vector);
        });
        return chunks;
      } catch (err) {
        if (!(err instanceof AppError) || err.code !== 'INVALID_EMBEDDING') throw err;
        // A corrupt cache is rebuilt through the same validated, atomic path.
      }
    }
    for (let offset = 0; offset < chunks.length; offset += 64) {
      const batch = chunks.slice(offset, offset + 64);
      const vectors = await embed(transcriptId, batch.map((c) => c.text), 'chunks', signal);
      vectors.forEach((vector, i) => { batch[i]!.embedding = vector; });
    }
    if (chunks.some((c) => c.embedding.length !== chunks[0]!.embedding.length)) badVectors();
    signal.throwIfAborted();
    // Keep the last usable cache until every batch succeeds. Never hold a DB transaction over HTTP.
    tx(db, () => {
      if (!db.prepare('SELECT id FROM transcripts WHERE id = ?').get(transcriptId)) {
        throw new AppError(404, 'NO_TRANSCRIPT', 'That transcript revision no longer exists.');
      }
      db.prepare('DELETE FROM chunks WHERE transcript_id = ?').run(transcriptId);
      const insert = db.prepare(`INSERT INTO chunks
        (transcript_id, ord, units_version, unit_ids_json, start_seconds, end_seconds, text, embedding, embedding_model, embedding_dim)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
      for (const chunk of chunks) insert.run(transcriptId, chunk.ord, UNITS_VERSION,
        JSON.stringify(chunk.units.map((u) => u.id)), chunk.start, chunk.end, chunk.text,
        Buffer.from(chunk.embedding.buffer), model, chunk.embedding.length);
    });
    return chunks;
  }

  async function ensureChunks(transcriptId: number, signal?: AbortSignal): Promise<Chunk[]> {
    signal?.throwIfAborted();
    let pending = building.get(transcriptId);
    if (!pending) {
      const controller = new AbortController();
      pending = { promise: build(transcriptId, controller.signal), controller, users: 0 };
      building.set(transcriptId, pending);
    }
    pending.users++;
    let abort: (() => void) | undefined;
    try {
      return await new Promise<Chunk[]>((resolve, reject) => {
        abort = () => reject(signal!.reason);
        signal?.addEventListener('abort', abort, { once: true });
        if (signal?.aborted) abort();
        pending!.promise.then(resolve, reject);
      });
    } finally {
      if (abort) signal?.removeEventListener('abort', abort);
      if (--pending.users === 0) {
        if (building.get(transcriptId) === pending) building.delete(transcriptId);
        pending.controller.abort(); // Last caller cancelled: stop paid source preparation too.
      }
    }
  }

  async function retrieve(transcriptId: number, query: string, { topK = 6, budgetTokens = 12000, signal }: { topK?: number; budgetTokens?: number; signal?: AbortSignal } = {}): Promise<Unit[]> {
    signal?.throwIfAborted();
    if (!Number.isSafeInteger(topK) || topK < 0 || !Number.isSafeInteger(budgetTokens) || budgetTokens < 0) {
      throw new AppError(400, 'INVALID_RETRIEVAL', 'Retrieval limits must be non-negative integers.');
    }
    if (!query.trim() || topK === 0 || budgetTokens === 0) return [];
    const chunks = await ensureChunks(transcriptId, signal);
    if (!chunks.length) return [];
    const [vector] = await embed(transcriptId, [query], 'query', signal);
    if (vector!.length !== chunks[0]!.embedding.length) badVectors();
    const scored = chunks.map((chunk) => ({ ord: chunk.ord, score: chunk.embedding.reduce((sum, value, i) => sum + value * vector![i]!, 0) }))
      .sort((a, b) => b.score - a.score || a.ord - b.ord);
    const selected = new Set<number>();
    for (const hit of scored.slice(0, topK)) {
      for (const ord of [hit.ord - 1, hit.ord, hit.ord + 1]) if (ord >= 0 && ord < chunks.length) selected.add(ord);
    }
    const candidates = [...selected].sort((a, b) => a - b).flatMap((ord) => chunks[ord]!.units)
      .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
    const result: Unit[] = [];
    let tokens = 0;
    for (const unit of candidates) {
      const cost = estimateTokens(unit.text);
      if (tokens + cost > budgetTokens) break; // Whole evidence units only; never manufacture partial quotes.
      result.push(unit);
      tokens += cost;
    }
    return result;
  }
  return { ensureChunks, retrieve };
}

function badVectors(): never {
  throw new AppError(502, 'INVALID_EMBEDDING', 'Embedding vectors are empty, invalid, or have incompatible dimensions.', true);
}

function normalizedVector(vector: Float32Array): Float32Array {
  if (!vector.length || vector.some((value) => !Number.isFinite(value))) badVectors();
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!norm) badVectors();
  return Float32Array.from(vector, (value) => value / norm);
}
