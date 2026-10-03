import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { ChatEvent } from '../../../shared/chat.ts';
import { readSSE } from '../../../shared/sse.ts';
import { createTestApp } from '../testing.ts';
import { recoverInterrupted } from '../db.ts';
import { createOpenRouter, type ChatMessage, type ChatResult } from '../integrations/openrouter.ts';
import { saveTranscript } from '../services/transcripts.ts';

function fixture() {
  const requests: { messages: ChatMessage[]; signal: AbortSignal; stream: ReadableStreamDefaultController<Uint8Array> }[] = [];
  const embeddings: string[][] = [];
  const provider = createOpenRouter({ apiKey: 'invented-key', baseUrl: 'https://provider.invalid', fetch: async (url, init) => {
    const body = JSON.parse(String(init?.body));
    if (String(url).endsWith('/embeddings')) {
      embeddings.push(body.input);
      return Response.json({ model: body.model, data: body.input.map((_text: string, index: number) => ({ index, embedding: [1, 0] })), usage: { prompt_tokens: 1, cost: 0.00001 } });
    }
    return new Response(new ReadableStream<Uint8Array>({ start(stream) {
      requests.push({ messages: body.messages, signal: init!.signal!, stream });
      init!.signal!.addEventListener('abort', () => stream.error(init!.signal!.reason), { once: true });
    } }));
  } });
  const app = createTestApp({ openrouter: provider, env: { OPENROUTER_API_KEY: 'invented-key' } });
  for (const id of [1, 2]) app.db.prepare('INSERT INTO videos (id, youtube_id, title) VALUES (?, ?, ?)').run(id, `InventedId${id}`, `Paper workshop ${id}`);
  const revision = saveTranscript(app.db, 1, { source: 'youtube', language: 'en', timed: true, segments: [
    { id: 0, start: 12.5, end: 17, text: 'The lantern folds flat. It cannot hold water.' },
    { id: 1, start: 23, end: 27, text: '>> The second speaker suggests using thick paper.' },
  ] });
  const emit = (index: number, chunk: object) => requests[index]!.stream.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`));
  const delta = (index: number, text: string) => emit(index, { id: `gen-${index}`, model: 'actual/model', provider: 'Invented', choices: [{ delta: { content: text } }] });
  const end = (index: number, reason = 'stop') => {
    emit(index, { choices: [{ delta: {}, finish_reason: reason }], usage: { prompt_tokens: 100, completion_tokens: 20, cost: 0.001 } });
    requests[index]!.stream.enqueue(new TextEncoder().encode('data: [DONE]\n\n')); requests[index]!.stream.close();
  };
  return { ...app, revision, requests, embeddings, delta, end, emit };
}
async function collect(response: Response) {
  const events: ChatEvent[] = [];
  for await (const item of readSSE(response.body!)) events.push({ event: item.event, data: JSON.parse(item.data) } as ChatEvent);
  return events;
}
async function conversation(app: ReturnType<typeof fixture>) {
  const res = await app.json('POST', '/api/videos/1/conversations', {}); assert.equal(res.status, 201); return (await res.json()).id as number;
}
async function start(app: ReturnType<typeof fixture>, cid: number, body: object = { content: 'What can it do?' }) {
  const res = await app.json('POST', `/api/conversations/${cid}/messages`, body);
  const reader = readSSE(res.body!); const first = (await reader.next()).value!;
  assert.equal(first.event, 'start');
  // Let the detached prompt preparation reach the provider, without a timing-dependent sleep.
  await new Promise<void>((resolve) => setImmediate(resolve));
  return { res, reader, ...JSON.parse(first.data) as { userMessageId: number; assistantMessageId: number } };
}

test('SSE order, duplicate-send guard, citations, usage, source notes, and revision links use persisted data', async (t) => {
  const app = fixture(); t.after(app.cleanup); const cid = await conversation(app);
  const note = await (await app.json('POST', '/api/videos/1/documents', { title: 'Folding note', markdown: 'Fold twice.' })).json();
  const run = await start(app, cid, { content: 'What can it do?', documentIds: [note.id] });
  assert.equal((await app.json('POST', `/api/conversations/${cid}/messages`, { content: 'Again' })).status, 409);
  app.delta(0, 'It folds flat [u001, u999]. '); app.delta(0, `Fold twice [d${note.id}]. Invalid [d999].`); app.end(0);
  const events: string[] = ['start']; let done: any;
  for await (const item of run.reader) { events.push(item.event); if (item.event === 'done') done = JSON.parse(item.data).message; }
  assert.deepEqual(events, ['start', 'delta', 'delta', 'done']);
  assert.equal(done.status, 'complete'); assert.ok(!done.content.includes('999'));
  assert.deepEqual(done.citations.map((c: any) => c.id), ['u001', `d${note.id}`]);
  assert.equal(done.citations[0].start, 12.5); assert.ok(done.citations.every((c: any) => c.excerpt.length <= 200));
  assert.equal(done.model, 'actual/model'); assert.equal(done.usage.usage.cost, 0.001); assert.equal(done.usage.generationId, 'gen-0');
  assert.deepEqual(done.context.documentIds, [note.id]); assert.equal(done.context.transcriptId, app.revision);
  assert.match(app.requests[0]!.messages[0]!.content, /data, not instructions/);
  assert.match(app.requests[0]!.messages[1]!.content, /d1 Folding note\nFold twice/);
  const newer = saveTranscript(app.db, 1, { source: 'paste_text', language: null, timed: false, segments: [{ id: 0, start: null, end: null, text: 'Replacement text.' }] });
  assert.equal((await (await app.request('/api/videos/1/transcript')).json()).transcriptId, newer);
  assert.equal((await (await app.request(`/api/videos/1/transcript?transcriptId=${app.revision}`)).json()).transcriptId, app.revision);
  assert.equal((await app.request(`/api/videos/2/transcript?transcriptId=${app.revision}`)).status, 404);
  assert.equal((await app.request('/api/videos/1/transcript?transcriptId=-1')).status, 400);
  const detail = await (await app.request(`/api/conversations/${cid}`)).json(); assert.equal(detail.title, 'What can it do?'); assert.equal(detail.messages.length, 2);
  await app.json('PATCH', `/api/conversations/${cid}`, { title: 'Lantern limits' });
  assert.equal((await (await app.request('/api/videos/1/conversations')).json())[0].title, 'Lantern limits');
});

test('disconnect continues; explicit Stop aborts the provider and retains a partial answer with unknown cost', async (t) => {
  const app = fixture(); t.after(app.cleanup); const cid = await conversation(app);
  const first = await start(app, cid); await first.reader.return(undefined);
  app.delta(0, 'A complete answer [u001].'); app.end(0); await app.chat.idle();
  assert.equal(app.chat.getMessage(first.assistantMessageId).status, 'complete'); assert.equal(app.requests[0]!.signal.aborted, false);
  const second = await start(app, cid); app.delta(1, 'Partial [u001].'); await second.reader.next();
  const stopped = await (await app.json('POST', `/api/messages/${second.assistantMessageId}/cancel`, {})).json();
  assert.equal(stopped.status, 'incomplete'); assert.equal(stopped.content, 'Partial [u001].'); assert.equal(stopped.usage.usage, null);
  assert.equal(app.requests[1]!.signal.aborted, true); await app.chat.idle();
  assert.equal(app.chat.getMessage(second.assistantMessageId).status, 'incomplete'); await second.reader.return(undefined);
});

test('provider failure and token limit keep partial text; empty answers fail and history stays bounded', async (t) => {
  const app = fixture(); t.after(app.cleanup); const cid = await conversation(app);
  for (let i = 0; i < 5; i++) {
    const response = await app.json('POST', `/api/conversations/${cid}/messages`, { content: `Question ${i}` });
    await new Promise<void>((resolve) => setImmediate(resolve)); app.delta(i, `Answer ${i} [u001].`); app.end(i); await collect(response);
  }
  const sixth = await start(app, cid); const prompt = app.requests[5]!.messages;
  assert.equal(prompt.length, 9); assert.equal(prompt[2]!.content, 'Question 2'); assert.ok(!prompt[3]!.content.includes('[u001]'));
  app.delta(5, 'So far [u002].'); app.emit(5, { error: { message: 'Invented provider failure' } }); app.requests[5]!.stream.close();
  const remaining = []; for await (const event of sixth.reader) remaining.push(event);
  assert.equal(remaining.at(-1)!.event, 'error');
  const failed = app.chat.getMessage(sixth.assistantMessageId); assert.equal(failed.status, 'failed'); assert.equal(failed.content, 'So far [u002].'); assert.equal(failed.usage!.generationId, 'gen-5');
  const seventh = await start(app, cid); app.delta(6, 'Cut short [u001].'); app.end(6, 'length'); for await (const _ of seventh.reader) { /* drain */ }
  assert.equal(app.chat.getMessage(seventh.assistantMessageId).status, 'incomplete');
  const eighth = await start(app, cid); app.end(7); for await (const _ of eighth.reader) { /* drain */ }
  assert.equal(app.chat.getMessage(eighth.assistantMessageId).status, 'failed');
});

test('long prompts retrieve only their frozen video, and record the exact provided unit IDs', async (t) => {
  const app = fixture(); t.after(app.cleanup);
  const long = saveTranscript(app.db, 1, { source: 'youtube', language: 'en', timed: true, segments: Array.from({ length: 150 }, (_, id) => ({ id, start: id * 4, end: id * 4 + 3, text: `>> Workshop${id} ${'inventedlanternword '.repeat(44)}` })) });
  saveTranscript(app.db, 2, { source: 'youtube', language: 'en', timed: true, segments: [{ id: 0, start: 0, end: 3, text: 'FOREIGN SOURCE NEVER INCLUDED.' }] });
  const run = await start(app, await conversation(app)); app.delta(0, 'A retrieved answer [u001].'); app.end(0); for await (const _ of run.reader) { /* drain */ }
  const message = app.chat.getMessage(run.assistantMessageId); assert.equal(message.context!.mode, 'retrieval'); assert.equal(message.context!.transcriptId, long);
  assert.ok(message.context!.unitIds.length < 150); assert.ok(message.context!.unitIds.includes('u001'));
  assert.ok(app.embeddings.length >= 2); assert.ok(app.embeddings.flat().every((text) => !text.includes('FOREIGN SOURCE')));
  assert.match(app.requests[0]!.messages[0]!.content, /excerpts of a long transcript/);
  assert.ok(!app.requests[0]!.messages[1]!.content.includes('FOREIGN SOURCE'));
});

test('invalid sources and missing configuration never insert messages or start paid calls', async (t) => {
  const app = fixture(); t.after(app.cleanup); const cid = await conversation(app);
  const foreign = await (await app.json('POST', '/api/videos/2/documents', { title: 'Other video' })).json();
  assert.equal((await app.json('POST', `/api/conversations/${cid}/messages`, { content: 'Question', documentIds: [foreign.id] })).status, 400);
  assert.equal((await app.json('POST', `/api/conversations/${cid}/messages`, { content: ' ' })).status, 400);
  assert.equal(app.requests.length, 0); assert.equal(app.chat.conversationDetail(cid).messages.length, 0);
  const unconfigured = createTestApp(); t.after(unconfigured.cleanup);
  unconfigured.db.prepare("INSERT INTO videos (youtube_id, title) VALUES ('InventedId', 'Empty')").run();
  const convo = unconfigured.chat.createConversation(1);
  assert.equal((await unconfigured.json('POST', `/api/conversations/${convo.id}/messages`, { content: 'Question' })).status, 503);
  assert.equal(unconfigured.chat.conversationDetail(convo.id).messages.length, 0);
});

test('shutdown marks responses interrupted; deleting conversations or videos cancels their work', async (t) => {
  const app = fixture(); t.after(app.cleanup); const cid = await conversation(app); const run = await start(app, cid);
  app.delta(0, 'A partial response [u001].'); await run.reader.next(); await app.chat.stop();
  assert.equal(app.chat.getMessage(run.assistantMessageId).status, 'interrupted'); await run.reader.return(undefined);
  app.db.prepare("UPDATE messages SET status = 'generating' WHERE id = ?").run(run.assistantMessageId);
  assert.equal(recoverInterrupted(app.db).messages, 1); assert.equal(app.chat.getMessage(run.assistantMessageId).status, 'interrupted');
  const fresh = fixture(); t.after(fresh.cleanup); const first = await start(fresh, await conversation(fresh));
  await fresh.request(`/api/conversations/${fresh.chat.getMessage(first.assistantMessageId).conversationId}`, { method: 'DELETE' });
  assert.equal(fresh.requests[0]!.signal.aborted, true); await first.reader.return(undefined); await fresh.chat.idle();
  const second = await start(fresh, await conversation(fresh)); await fresh.request('/api/videos/1', { method: 'DELETE' });
  assert.equal(fresh.requests[1]!.signal.aborted, true); await second.reader.return(undefined); await fresh.chat.idle();
  assert.deepEqual(fresh.db.prepare('SELECT count(*) AS n FROM messages').get(), { n: 0 });
});

test('two concurrent answers are allowed; a third and a video without readable text are rejected before insertion', async (t) => {
  const app = fixture(); t.after(app.cleanup);
  const empty = app.chat.createConversation(2);
  assert.equal((await app.json('POST', `/api/conversations/${empty.id}/messages`, { content: 'Question' })).status, 409);
  const a = await start(app, await conversation(app)); const b = await start(app, await conversation(app));
  const third = await conversation(app);
  assert.equal((await app.json('POST', `/api/conversations/${third}/messages`, { content: 'Question' })).status, 429);
  assert.equal(app.chat.conversationDetail(third).messages.length, 0); assert.equal(app.requests.length, 2);
  app.end(0); app.end(1); await app.chat.idle(); await a.reader.return(undefined); await b.reader.return(undefined);
});

test('an unexpected stream EOF keeps partial text but never claims a complete answer', async (t) => {
  const app = fixture(); t.after(app.cleanup); const run = await start(app, await conversation(app));
  app.delta(0, 'Cut off [u001].'); app.requests[0]!.stream.close();
  const events = []; for await (const event of run.reader) events.push(event);
  assert.equal(events.at(-1)!.event, 'error');
  const message = app.chat.getMessage(run.assistantMessageId);
  assert.equal(message.status, 'failed'); assert.equal(message.errorCode, 'INCOMPLETE_STREAM'); assert.equal(message.content, 'Cut off [u001].');
});

test('a first question equal to the default conversation title still stays the first title', async (t) => {
  const app = fixture(); t.after(app.cleanup); const cid = await conversation(app);
  for (const [index, content] of ['New conversation', 'A different second question'].entries()) {
    const run = await start(app, cid, { content }); app.delta(index, 'Answer [u001].'); app.end(index);
    for await (const _ of run.reader) { /* drain */ }
  }
  assert.equal(app.chat.getConversation(cid).title, 'New conversation');
});

test('a deleted run finishing late cannot remove a newer run that reuses its SQLite message ID', async (t) => {
  const runs: { args: Parameters<ReturnType<typeof createOpenRouter>['chatStream']>[0]; result: ReturnType<typeof Promise.withResolvers<ChatResult>> }[] = [];
  const app = createTestApp({ env: { OPENROUTER_API_KEY: 'invented-key' }, openrouter: {
    embed: async () => { throw new Error('short transcripts do not embed'); },
    chatStream: (args) => { const result = Promise.withResolvers<ChatResult>(); runs.push({ args, result }); return result.promise; },
  } }); t.after(app.cleanup);
  app.db.prepare("INSERT INTO videos (youtube_id, title) VALUES ('InventedId', 'Paper workshop')").run();
  saveTranscript(app.db, 1, { source: 'paste_text', language: null, timed: false, segments: [{ id: 0, start: null, end: null, text: 'Invented folding passage.' }] });
  const first = app.chat.start(app.chat.createConversation(1).id, { content: 'First', documentIds: [] });
  await new Promise<void>((resolve) => setImmediate(resolve)); app.chat.deleteConversation(first.conversation.id);
  const second = app.chat.start(app.chat.createConversation(1).id, { content: 'Second', documentIds: [] });
  assert.equal(first.id, second.id); await new Promise<void>((resolve) => setImmediate(resolve));
  const response = { text: 'Answer [u001].', model: 'invented', provider: null, usage: null, generationId: null, finishReason: 'stop', latencyMs: 1 };
  runs[0]!.result.resolve(response); await first.pending;
  assert.equal(app.chat.cancel(second.id).status, 'incomplete'); assert.equal(runs[1]!.args.signal!.aborted, true);
  runs[1]!.result.resolve(response); await second.pending;
});
