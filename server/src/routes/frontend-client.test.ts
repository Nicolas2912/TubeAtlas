import { test } from 'node:test';
import assert from 'node:assert/strict';
import { api, ApiError, send, unwrap } from '../../../frontend/src/api.ts';
import { createTestApp } from '../testing.ts';

test('the frontend client reaches the API sub-app, including topics, filtering, jobs, and settings', async (t) => {
  const app = createTestApp({ youtube: {
    fetchMetadata: async (youtubeId) => ({ youtubeId, title: 'The paper lantern workshop', channel: 'Fold & Glow', durationSeconds: 90, thumbnailUrl: null }),
    fetchTranscript: async () => ({ status: 'ready', language: 'en', segments: [{ id: 0, start: 0, end: 5, text: 'Fold the paper into a square.' }] }),
  } });
  t.after(app.cleanup);
  const paths: string[] = [];
  t.mock.method(globalThis, 'fetch', (input: string, init?: RequestInit) => {
    paths.push(input);
    return app.request(input, init);
  });

  const settings = await unwrap(api.settings.$get());
  assert.equal(settings.aiConfigured, false);
  const topic = await unwrap(api.topics.$post({ json: { name: 'Paper crafts' } }));
  const { video, job } = await unwrap(api.videos.import.$post({ json: { url: 'AbCdEfGhIjK' } }));
  assert.ok(job);
  await app.jobs.idle();
  assert.equal((await unwrap(api.jobs[':id'].$get({ param: { id: String(job.id) } }))).status, 'succeeded');
  const transcript = await unwrap(api.videos[':id'].transcript.$get({ param: { id: String(video.id) }, query: {} }));
  assert.equal(transcript.timed, true);
  assert.equal(transcript.segments[0]!.start, 0);
  assert.equal(transcript.unitsVersion, 1);
  assert.equal(transcript.units[0]!.id, 'u001');
  assert.deepEqual(transcript.units[0]!.segmentIds, [0]);
  await unwrap(api.videos[':id'].$patch({ param: { id: String(video.id) }, json: { playbackSeconds: 42.5 } }));
  assert.equal((await unwrap(api.videos[':id'].$get({ param: { id: String(video.id) } }))).playbackSeconds, 42.5);
  const note = await unwrap(api.videos[':id'].documents.$post({ param: { id: String(video.id) }, json: { title: 'Paper notes' } }));
  await unwrap(api.documents[':id'].$patch({ param: { id: String(note.id) }, json: { markdown: 'Fold once.' } }));
  await unwrap(api.documents[':id'].$patch({ param: { id: String(note.id) }, json: { appendMarkdown: 'Keep corners even.' } }));
  assert.equal((await unwrap(api.documents[':id'].$get({ param: { id: String(note.id) } }))).markdown, 'Fold once.\n\nKeep corners even.');
  const uploaded = await unwrap(api.videos[':id'].assets.$post({ param: { id: String(video.id) }, form: { file: new File(['Paper notes.'], 'notes.txt') } }));
  assert.equal(uploaded.kind, 'note');
  assert.equal((await unwrap(api.videos[':id'].documents.$get({ param: { id: String(video.id) } }))).length, 2);
  await unwrap(api.videos[':id'].$patch({ param: { id: String(video.id) }, json: { topicIds: [topic.id] } }));
  const filtered = await unwrap(api.videos.$get({ query: { topicId: String(topic.id) } }));
  assert.deepEqual(filtered.map((v) => v.id), [video.id]);
  assert.equal(filtered[0]!.transcriptStatus, 'ready');
  await unwrap(api.topics[':id'].$patch({ param: { id: String(topic.id) }, json: { name: 'Lanterns' } }));
  assert.equal((await unwrap(api.topics.$get()))[0]!.name, 'Lanterns');
  await send(api.topics[':id'].$delete({ param: { id: String(topic.id) } }));
  assert.equal((await unwrap(api.videos[':id'].$get({ param: { id: String(video.id) } }))).topics.length, 0);
  assert.ok(paths.every((path) => path.startsWith('/api/')), paths.join('\n'));
});

test('the frontend displays API errors with their message and code', async (t) => {
  const app = createTestApp();
  t.after(app.cleanup);
  t.mock.method(globalThis, 'fetch', (input: string, init?: RequestInit) => app.request(input, init));
  await assert.rejects(unwrap(api.videos.import.$post({ json: { url: 'not a video' } })), (err: unknown) =>
    err instanceof ApiError && err.status === 400 && err.code === 'INVALID_URL' && err.message.length > 0,
  );
});

test('the frontend turns an unreachable server into a useful connection error', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('fetch failed'); });
  await assert.rejects(unwrap(api.settings.$get()), (err: unknown) =>
    err instanceof ApiError && err.code === 'NETWORK' && err.message.includes('Is the server running?'),
  );
});
