import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Segment } from '../../../shared/api.ts';
import { AppError } from '../errors.ts';
import type { TranscriptFetchResult, YouTube } from '../integrations/youtube.ts';
import { createTestApp } from '../testing.ts';

const ID = 'AbCdEfGhIjK';
const URL_ = `https://www.youtube.com/watch?v=${ID}`;

// Invented transcript content.
const SEGMENTS: Segment[] = [
  { id: 0, start: 0.32, end: 5.04, text: 'Welcome to the bakery.' },
  { id: 1, start: 2.6, end: 7, text: 'Today we bake rye bread.' },
  { id: 2, start: 5.04, end: 9.2, text: 'It needs a sourdough starter.' },
];

function fakeYouTube(transcript: (signal?: AbortSignal) => Promise<TranscriptFetchResult> = async () => ({ status: 'ready', language: 'en', segments: SEGMENTS })) {
  const calls = { metadata: 0, transcript: 0 };
  const youtube: YouTube = {
    fetchMetadata: async (youtubeId) => {
      calls.metadata++;
      return { youtubeId, title: 'Baking rye bread', channel: 'Crumb & Crust', durationSeconds: 600, thumbnailUrl: 'https://i.ytimg.com/x.jpg' };
    },
    fetchTranscript: async (_id, signal) => {
      calls.transcript++;
      return transcript(signal);
    },
  };
  return { youtube, calls };
}

async function setup(t: { after: (fn: () => Promise<void>) => void }, youtube = fakeYouTube().youtube) {
  const app = createTestApp({ youtube });
  t.after(app.cleanup);
  return app;
}

test('import creates the video and a transcript job; the job stores timed segments that reload identically', async (t) => {
  const { youtube, calls } = fakeYouTube();
  const { json, request, jobs } = await setup(t, youtube);

  const res = await json('POST', '/api/videos/import', { url: URL_ });
  assert.equal(res.status, 201);
  const { video, job } = await res.json();
  assert.equal(video.youtubeId, ID);
  assert.equal(video.title, 'Baking rye bread');
  assert.equal(video.durationSeconds, 600);
  assert.equal(video.transcriptStatus, 'pending');
  assert.equal(job.kind, 'transcript');
  assert.equal(video.activeJob.id, job.id);

  await jobs.idle();
  const done = await (await request(`/api/jobs/${job.id}`)).json();
  assert.equal(done.status, 'succeeded');
  assert.deepEqual(done.result, { status: 'ready', segments: 3, language: 'en' });

  const transcript = await (await request(`/api/videos/${video.id}/transcript`)).json();
  assert.deepEqual(transcript, { transcriptId: transcript.transcriptId, revision: 1, language: 'en', source: 'youtube', timed: true, segments: SEGMENTS });
  const after = await (await request(`/api/videos/${video.id}`)).json();
  assert.equal(after.transcriptStatus, 'ready');
  assert.equal(after.activeJob, null);
  assert.deepEqual(calls, { metadata: 1, transcript: 1 });
});

test('caption retry can recover a blocked outcome, reuses active work, and protects a ready transcript', async (t) => {
  let blocked = true;
  const { youtube, calls } = fakeYouTube(async () => blocked ? { status: 'blocked', message: 'Caption access was blocked.' } : { status: 'ready', language: 'en', segments: SEGMENTS });
  const { json, request, jobs } = await setup(t, youtube);
  const { video, job } = await (await json('POST', '/api/videos/import', { url: URL_ })).json();
  await jobs.idle();
  assert.equal(jobs.get(job.id)?.status, 'succeeded');
  assert.equal((await (await request(`/api/videos/${video.id}`)).json()).transcriptStatus, 'blocked');
  blocked = false;
  const retried = await json('POST', `/api/videos/${video.id}/transcript/retry`, {});
  assert.equal(retried.status, 201);
  const next = await retried.json();
  const duplicate = await (await json('POST', `/api/videos/${video.id}/transcript/retry`, {})).json();
  assert.equal(next.id, duplicate.id);
  assert.notEqual(next.id, job.id);
  await jobs.idle();
  assert.equal((await (await request(`/api/videos/${video.id}`)).json()).transcriptStatus, 'ready');
  const ready = await json('POST', `/api/videos/${video.id}/transcript/retry`, {});
  assert.equal(ready.status, 409);
  assert.equal((await ready.json()).error.code, 'TRANSCRIPT_READY');
  assert.equal(calls.transcript, 2);
  assert.equal((await json('POST', '/api/videos/999/transcript/retry', {})).status, 404);
});

test('importing the same video again returns the existing one (200) without new work', async (t) => {
  const { youtube, calls } = fakeYouTube();
  const { json, jobs } = await setup(t, youtube);
  const first = await (await json('POST', '/api/videos/import', { url: URL_ })).json();
  await jobs.idle();

  const again = await json('POST', '/api/videos/import', { url: `https://youtu.be/${ID}?t=5` });
  assert.equal(again.status, 200);
  const body = await again.json();
  assert.equal(body.video.id, first.video.id);
  assert.equal(body.job, null);
  assert.deepEqual(calls, { metadata: 1, transcript: 1 });
});

test('import validates the URL and topic, and reports YouTube errors', async (t) => {
  const youtube: YouTube = {
    fetchMetadata: async () => {
      throw new AppError(404, 'VIDEO_NOT_FOUND', 'YouTube has no video with that id.');
    },
    fetchTranscript: async () => assert.fail('no transcript fetch expected'),
  };
  const { json } = await setup(t, youtube);
  const codes = async (body: unknown) => {
    const res = await json('POST', '/api/videos/import', body);
    return [res.status, (await res.json()).error.code];
  };
  assert.deepEqual(await codes({ url: 'https://vimeo.com/1' }), [400, 'INVALID_URL']);
  assert.deepEqual(await codes({}), [400, 'VALIDATION_FAILED']);
  assert.deepEqual(await codes({ url: URL_, topicId: 99 }), [400, 'UNKNOWN_TOPIC']);
  assert.deepEqual(await codes({ url: URL_ }), [404, 'VIDEO_NOT_FOUND']);
});

test('import with a topic links it; re-importing with another topic adds that one too', async (t) => {
  const { json, jobs } = await setup(t);
  const ml = await (await json('POST', '/api/topics', { name: 'Baking' })).json();
  const other = await (await json('POST', '/api/topics', { name: 'Food' })).json();
  const { video } = await (await json('POST', '/api/videos/import', { url: URL_, topicId: ml.id })).json();
  assert.deepEqual(video.topics, [{ id: ml.id, name: 'Baking' }]);
  const again = await (await json('POST', '/api/videos/import', { url: URL_, topicId: other.id })).json();
  assert.deepEqual(again.video.topics.map((t: { name: string }) => t.name), ['Baking', 'Food']);
  await jobs.idle();
});

test('no captions and blocked are outcomes: the status is set and the job succeeds', async (t) => {
  for (const status of ['no_captions', 'blocked'] as const) {
    const { youtube } = fakeYouTube(async () => ({ status, message: `msg ${status}` }));
    const { json, request, jobs } = await setup(t, youtube);
    const { video, job } = await (await json('POST', '/api/videos/import', { url: URL_ })).json();
    await jobs.idle();
    const after = await (await request(`/api/videos/${video.id}`)).json();
    assert.equal(after.transcriptStatus, status);
    assert.equal(after.transcriptError, `msg ${status}`);
    const done = await (await request(`/api/jobs/${job.id}`)).json();
    assert.equal(done.status, 'succeeded');
    assert.deepEqual(done.result, { status, message: `msg ${status}` });
    const missing = await request(`/api/videos/${video.id}/transcript`);
    assert.equal(missing.status, 404);
    assert.equal((await missing.json()).error.code, 'NO_TRANSCRIPT');
  }
});

test('an unexpected failure fails the job; retry creates a new job that can succeed', async (t) => {
  let fail = true;
  const { youtube } = fakeYouTube(async () => {
    if (fail) throw new AppError(502, 'UNKNOWN_CAPTION_FORMAT', 'YouTube returned captions in an unrecognized format.');
    return { status: 'ready', language: 'en', segments: SEGMENTS };
  });
  const { json, request, jobs } = await setup(t, youtube);
  const { video, job } = await (await json('POST', '/api/videos/import', { url: URL_ })).json();
  await jobs.idle();
  const failed = await (await request(`/api/jobs/${job.id}`)).json();
  assert.equal(failed.status, 'failed');
  assert.deepEqual(failed.error, { code: 'UNKNOWN_CAPTION_FORMAT', message: 'YouTube returned captions in an unrecognized format.' });

  fail = false;
  const retried = await (await request(`/api/jobs/${job.id}/retry`, { method: 'POST' })).json();
  assert.notEqual(retried.id, job.id);
  await jobs.idle();
  assert.equal((await (await request(`/api/jobs/${retried.id}`)).json()).status, 'succeeded');
  assert.equal((await (await request(`/api/videos/${video.id}`)).json()).transcriptStatus, 'ready');

  const again = await request(`/api/jobs/${retried.id}/retry`, { method: 'POST' });
  assert.equal(again.status, 409);
  assert.equal((await again.json()).error.code, 'JOB_NOT_RETRYABLE');
});

test('a duplicate enqueue returns the active job; cancel works for queued and running jobs', async (t) => {
  const { youtube } = fakeYouTube((signal) => new Promise((_, reject) => signal?.addEventListener('abort', () => reject(signal.reason))));
  const { json, request, jobs, db } = await setup(t, youtube);
  const { video, job } = await (await json('POST', '/api/videos/import', { url: URL_ })).json();
  await new Promise((resolve) => setTimeout(resolve, 20)); // let the runner pick it up
  assert.equal(jobs.get(job.id)!.status, 'running');
  assert.equal(jobs.get(job.id)!.stage, 'fetching captions');
  assert.equal(jobs.enqueue('transcript', video.id).id, job.id, 'same active job');

  // A queued job of another video waits behind the running one and can be cancelled directly.
  const otherVideo = Number(db.prepare(`INSERT INTO videos (youtube_id, title) VALUES ('ZZZZZZZZZZZ', 'Other')`).run().lastInsertRowid);
  const queued = jobs.enqueue('transcript', otherVideo);
  assert.equal(queued.status, 'queued');
  assert.equal((await (await request(`/api/jobs/${queued.id}/cancel`, { method: 'POST' })).json()).status, 'cancelled');

  await request(`/api/jobs/${job.id}/cancel`, { method: 'POST' });
  await jobs.idle();
  assert.equal(jobs.get(job.id)!.status, 'cancelled');
});

test('manual transcripts (text, VTT, SRT) create new revisions; only the newest is current', async (t) => {
  const { youtube } = fakeYouTube(async () => ({ status: 'no_captions', message: 'none' }));
  const { json, request, jobs, db } = await setup(t, youtube);
  const { video } = await (await json('POST', '/api/videos/import', { url: URL_ })).json();
  await jobs.idle();

  const text = await json('POST', `/api/videos/${video.id}/transcript`, {
    format: 'text',
    content: 'First paragraph\nstill first.\n\n\nSecond paragraph.',
    language: 'en',
  });
  assert.equal(text.status, 201);
  const t1 = await text.json();
  assert.equal(t1.revision, 1);
  assert.equal(t1.timed, false);
  assert.equal(t1.source, 'paste_text');
  assert.deepEqual(t1.segments, [
    { id: 0, start: null, end: null, text: 'First paragraph still first.' },
    { id: 1, start: null, end: null, text: 'Second paragraph.' },
  ]);
  assert.equal((await (await request(`/api/videos/${video.id}`)).json()).transcriptStatus, 'ready');

  const vtt = `WEBVTT\nKind: captions\n\nNOTE invented example\n\n00:00:01.500 --> 00:00:04.000 align:start position:0%\n<c.colorE5E5E5>Knead the</c> <00:00:02.000><c>dough</c>\n\n1:02:03.250 --> 1:02:05.000\nLet it &amp; rest\n\n00:10.000 --> 00:12.000\n\n`;
  const t2 = await (await json('POST', `/api/videos/${video.id}/transcript`, { format: 'vtt', content: vtt })).json();
  assert.equal(t2.revision, 2);
  assert.equal(t2.timed, true);
  assert.equal(t2.source, 'upload_timed');
  assert.deepEqual(t2.segments, [
    { id: 0, start: 1.5, end: 4, text: 'Knead the dough' },
    { id: 1, start: 3723.25, end: 3725, text: 'Let it & rest' },
  ]);

  const srt = '﻿1\r\n00:00:01,000 --> 00:00:02,500\r\nPreheat the oven\r\nto 250 degrees\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\n<i>Bake</i>\r\n';
  const t3 = await (await json('POST', `/api/videos/${video.id}/transcript`, { format: 'srt', content: srt })).json();
  assert.equal(t3.revision, 3);
  assert.deepEqual(t3.segments, [
    { id: 0, start: 1, end: 2.5, text: 'Preheat the oven to 250 degrees' },
    { id: 1, start: 3, end: 4, text: 'Bake' },
  ]);

  const rows = db.prepare('SELECT revision, is_current FROM transcripts WHERE video_id = ? ORDER BY revision').all(video.id);
  assert.deepEqual(rows.map((r: any) => [r.revision, r.is_current]), [[1, 0], [2, 0], [3, 1]]);
  assert.deepEqual((await (await request(`/api/videos/${video.id}/transcript`)).json()).segments, t3.segments);

  const empty = await json('POST', `/api/videos/${video.id}/transcript`, { format: 'vtt', content: 'WEBVTT\n\nno cues here' });
  assert.equal(empty.status, 400);
  assert.equal((await empty.json()).error.code, 'NO_TRANSCRIPT_TEXT');
});

test('manual transcripts may exceed the 1 MB API limit up to 5 MB', async (t) => {
  const { json, request, jobs, db } = await setup(t);
  const videoId = Number(db.prepare(`INSERT INTO videos (youtube_id, title) VALUES (?, 'Long')`).run(ID).lastInsertRowid);
  const paragraph = 'Flour water salt and time. '.repeat(40);
  const big = Array.from({ length: 1200 }, () => paragraph).join('\n\n'); // about 1.3 MB
  assert.ok(big.length > 1024 * 1024);
  const ok = await json('POST', `/api/videos/${videoId}/transcript`, { format: 'text', content: big });
  assert.equal(ok.status, 201);
  assert.equal((await ok.json()).segments.length, 1200);

  const tooBig = await json('POST', `/api/videos/${videoId}/transcript`, { format: 'text', content: 'x'.repeat(5 * 1024 * 1024 + 10) });
  assert.equal(tooBig.status, 413);
  const otherRoute = await request('/api/topics', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name: 'x'.repeat(1024 * 1024 + 10) }) });
  assert.equal(otherRoute.status, 413);
  await jobs.idle();
});

test('video list, topic filter, PATCH of playback position and topics', async (t) => {
  const { json, request, jobs } = await setup(t);
  const topic = await (await json('POST', '/api/topics', { name: 'Baking' })).json();
  const { video } = await (await json('POST', '/api/videos/import', { url: URL_ })).json();
  await json('POST', '/api/videos/import', { url: 'https://youtu.be/ZZZZZZZZZZZ' });
  await jobs.idle();

  assert.equal((await (await request('/api/videos')).json()).length, 2);
  assert.equal((await (await request(`/api/videos?topicId=${topic.id}`)).json()).length, 0);

  const patched = await (await json('PATCH', `/api/videos/${video.id}`, { playbackSeconds: 125.5, topicIds: [topic.id, topic.id] })).json();
  assert.equal(patched.playbackSeconds, 125.5);
  assert.deepEqual(patched.topics, [{ id: topic.id, name: 'Baking' }]);
  const filtered = await (await request(`/api/videos?topicId=${topic.id}`)).json();
  assert.deepEqual(filtered.map((v: { id: number }) => v.id), [video.id]);

  assert.equal((await json('PATCH', `/api/videos/${video.id}`, { playbackSeconds: -1 })).status, 400);
  assert.equal((await json('PATCH', `/api/videos/${video.id}`, { topicIds: [999] })).status, 400);
  assert.equal((await json('PATCH', `/api/videos/${video.id}`, { title: 'nope' })).status, 400);
  assert.equal((await request('/api/videos/999')).status, 404);
  assert.equal((await request('/api/videos/abc')).status, 404);
});

test('topics: create, list with counts, rename, duplicate names, delete keeps videos', async (t) => {
  const { json, request, jobs } = await setup(t);
  const baking = await (await json('POST', '/api/topics', { name: '  Baking ' })).json();
  assert.equal(baking.name, 'Baking');
  const dup = await json('POST', '/api/topics', { name: 'baking' });
  assert.equal(dup.status, 409);
  assert.equal((await dup.json()).error.code, 'TOPIC_EXISTS');

  const { video } = await (await json('POST', '/api/videos/import', { url: URL_, topicId: baking.id })).json();
  await jobs.idle();
  const bread = await (await json('POST', '/api/topics', { name: 'Bread' })).json();
  assert.deepEqual(await (await request('/api/topics')).json(), [
    { id: baking.id, name: 'Baking', videoCount: 1 },
    { id: bread.id, name: 'Bread', videoCount: 0 },
  ]);
  assert.equal((await json('PATCH', `/api/topics/${bread.id}`, { name: 'BAKING' })).status, 409);
  assert.equal((await (await json('PATCH', `/api/topics/${bread.id}`, { name: 'Sourdough' })).json()).name, 'Sourdough');
  assert.equal((await json('POST', '/api/topics', { name: '' })).status, 400);

  assert.equal((await request(`/api/topics/${baking.id}`, { method: 'DELETE' })).status, 204);
  assert.equal((await request(`/api/topics/${baking.id}`, { method: 'DELETE' })).status, 404);
  const kept = await (await request(`/api/videos/${video.id}`)).json();
  assert.deepEqual(kept.topics, []);
});

test('deleting a video removes its rows and its stored files, and cancels its active job', async (t) => {
  const { youtube } = fakeYouTube((signal) => new Promise((_, reject) => signal?.addEventListener('abort', () => reject(signal.reason))));
  const { json, request, jobs, db, dataDir } = await setup(t, youtube);
  const { video, job } = await (await json('POST', '/api/videos/import', { url: URL_ })).json();
  const file = join(dataDir, 'files', 'invented.pdf');
  writeFileSync(file, '%PDF-1.4 invented');
  db.prepare(`INSERT INTO assets (video_id, storage_name, original_name, media_type, size_bytes) VALUES (?, 'invented.pdf', 'notes.pdf', 'application/pdf', 17)`).run(video.id);
  await new Promise((resolve) => setTimeout(resolve, 20));

  assert.equal((await request(`/api/videos/${video.id}`, { method: 'DELETE' })).status, 204);
  assert.ok(!existsSync(file));
  assert.equal((await request(`/api/videos/${video.id}`)).status, 404);
  await jobs.idle();
  assert.equal(jobs.get(job.id), null, 'job row went with the video');
  assert.equal((await request(`/api/videos/${video.id}`, { method: 'DELETE' })).status, 404);
});

test('unknown job ids are 404', async (t) => {
  const { request } = await setup(t);
  assert.equal((await request('/api/jobs/12345')).status, 404);
  assert.equal((await request('/api/jobs/12345/cancel', { method: 'POST' })).status, 404);
  assert.equal((await request('/api/jobs/12345/retry', { method: 'POST' })).status, 404);
});
