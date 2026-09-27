import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createApp } from './app.ts';

test('health answers ok', async () => {
  const res = await createApp().app.request('/api/health');
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { ok: true });
});

test('unknown API paths are JSON 404s, not the frontend page', async () => {
  const res = await createApp().app.request('/api/does-not-exist');
  assert.equal(res.status, 404);
  assert.equal((await res.json()).error.code, 'NOT_FOUND');
});
