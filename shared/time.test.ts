import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatTime, watchLink } from './time.ts';

test('formatTime uses m:ss below an hour and h:mm:ss above', () => {
  assert.equal(formatTime(0), '0:00');
  assert.equal(formatTime(522.9), '8:42');
  assert.equal(formatTime(3599), '59:59');
  assert.equal(formatTime(9592.64), '2:39:52');
  assert.equal(formatTime(-3), '0:00');
});

test('watchLink floors seconds and never goes negative', () => {
  assert.equal(watchLink(12, 522.9), '/videos/12/watch?t=522');
  assert.equal(watchLink(12, -1), '/videos/12/watch?t=0');
});
