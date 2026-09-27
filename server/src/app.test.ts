import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AppError } from './errors.ts';
import { createTestApp } from './testing.ts';

test('health reports which integrations are configured, never key values', async (t) => {
  const bare = createTestApp();
  t.after(bare.cleanup);
  assert.deepEqual(await (await bare.request('/api/health')).json(), { ok: true, aiConfigured: false, youtubeKey: false });

  const keyed = createTestApp({ env: { OPENROUTER_API_KEY: 'sk-or-secret', GOOGLE_API_KEY: 'g-secret' } });
  t.after(keyed.cleanup);
  const text = await (await keyed.request('/api/health')).text();
  assert.deepEqual(JSON.parse(text), { ok: true, aiConfigured: true, youtubeKey: true });
  assert.ok(!text.includes('secret'));
});

test('unknown API paths are JSON 404s for every method, not the frontend page', async (t) => {
  const { request, cleanup } = createTestApp();
  t.after(cleanup);
  for (const method of ['GET', 'POST']) {
    const res = await request('/api/does-not-exist', { method });
    assert.equal(res.status, 404);
    assert.equal((await res.json()).error.code, 'NOT_FOUND');
  }
});

test('requests for a foreign host are refused (DNS rebinding)', async (t) => {
  const { app, cleanup } = createTestApp();
  t.after(cleanup);
  for (const url of ['http://evil.example/api/health', 'http://evil.example:5170/', 'http://127.0.0.1:9999/api/health']) {
    const res = await app.request(url);
    assert.equal(res.status, 403, url);
    assert.equal((await res.json()).error.code, 'BAD_HOST');
  }
  for (const url of ['http://localhost:5170/api/health', 'http://127.0.0.1:5171/api/health']) {
    assert.equal((await app.request(url)).status, 200, url);
  }
});

test('state-changing requests from a foreign origin are refused', async (t) => {
  const { request, cleanup } = createTestApp();
  t.after(cleanup);
  const foreign = await request('/api/health', { method: 'POST', headers: { origin: 'https://evil.example' } });
  assert.equal(foreign.status, 403);
  assert.equal((await foreign.json()).error.code, 'BAD_ORIGIN');
  // Same-origin or no Origin header passes the guard (and then 404s: there is no POST route).
  assert.equal((await request('/api/health', { method: 'POST', headers: { origin: 'http://127.0.0.1:5171' } })).status, 404);
  assert.equal((await request('/api/health', { method: 'POST' })).status, 404);
  // Reads are never blocked by Origin.
  assert.equal((await request('/api/health', { headers: { origin: 'https://evil.example' } })).status, 200);
});

test('errors: AppError keeps its code; unknown errors become a generic 500', async (t) => {
  const { app, request, cleanup } = createTestApp();
  t.after(cleanup);
  app.get('/api/test-app-error', () => {
    throw new AppError(409, 'TOPIC_EXISTS', 'A topic with that name exists.');
  });
  app.get('/api/test-crash', () => {
    throw new Error('secret internal detail');
  });

  const known = await request('/api/test-app-error');
  assert.equal(known.status, 409);
  assert.deepEqual(await known.json(), { error: { code: 'TOPIC_EXISTS', message: 'A topic with that name exists.', retryable: false } });

  const originalConsoleError = console.error;
  console.error = () => {}; // the handler logs the crash; keep test output clean
  t.after(() => (console.error = originalConsoleError));
  const crash = await request('/api/test-crash');
  assert.equal(crash.status, 500);
  const body = await crash.text();
  assert.equal(JSON.parse(body).error.code, 'INTERNAL');
  assert.ok(!body.includes('secret internal detail'));
});

test('JSON bodies over 1 MB are refused', async (t) => {
  const { request, cleanup } = createTestApp();
  t.after(cleanup);
  const res = await request('/api/health', { method: 'POST', body: 'x'.repeat(1024 * 1024 + 1), headers: { 'content-type': 'application/json' } });
  assert.equal(res.status, 413);
  assert.equal((await res.json()).error.code, 'PAYLOAD_TOO_LARGE');
});

test('settings show models, data directory, and key presence without key values', async (t) => {
  const { request, config, cleanup } = createTestApp({ env: { OPENROUTER_API_KEY: 'sk-or-secret' } });
  t.after(cleanup);
  const text = await (await request('/api/settings')).text();
  assert.deepEqual(JSON.parse(text), {
    aiConfigured: true,
    youtubeKey: false,
    models: { chat: 'openai/gpt-4.1-mini', embedding: 'openai/text-embedding-3-small', kg: 'openai/gpt-6-astra', kgReasoningEffort: 'low' },
    dataDir: config.dataDir,
  });
  assert.ok(!text.includes('secret'));
});
