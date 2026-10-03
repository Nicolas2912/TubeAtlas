import { test } from 'node:test';
import assert from 'node:assert/strict';
import { activePassage, clampPlayback, exportTranscript, findTextMatches, groupPassages, playbackTarget, prepareSearch } from './watch.ts';

test('passages break at sixty words or a real pause, preserving source ids and overlapping timings', () => {
  const passages = groupPassages([
    { id: 4, start: 0, end: 4, text: 'paper '.repeat(30).trim() },
    { id: 5, start: 3, end: 6, text: 'lantern '.repeat(30).trim() },
    { id: 6, start: 5, end: 7, text: 'Fold once.' },
    { id: 7, start: 10, end: 12, text: 'Light the lantern.' },
  ], true);
  assert.deepEqual(passages.map(({ segmentIds, start, end }) => ({ segmentIds, start, end })), [
    { segmentIds: [4, 5], start: 0, end: 6 }, { segmentIds: [6], start: 5, end: 7 }, { segmentIds: [7], start: 10, end: 12 },
  ]);
  assert.equal(activePassage(passages, 5.5), 1);
  assert.equal(activePassage(passages, 8), -1);
  assert.equal(activePassage(passages, 12), -1);
  assert.equal(activePassage([], 0), -1);
});

test('untimed paragraphs stay separate and never acquire fabricated timestamps', () => {
  const passages = groupPassages([{ id: 0, start: null, end: null, text: 'First paragraph.' }, { id: 1, start: null, end: null, text: 'Second paragraph.' }], false);
  assert.equal(passages.length, 2);
  assert.equal(activePassage(passages, 1), -1);
  assert.equal(exportTranscript(passages, 7, true), 'First paragraph.\n\nSecond paragraph.\n');
});

test('linked and saved playback positions are validated and clamped without modifying captions', () => {
  assert.equal(playbackTarget('0', 40, 90), 0);
  assert.equal(playbackTarget('300', 40, 90), 90);
  for (const value of [null, '', 'bad', '-1', 'Infinity']) assert.equal(playbackTarget(value, 40, 90), 40);
  assert.equal(playbackTarget('300.5', 40, null), 300.5);
  assert.equal(clampPlayback(NaN, null), 0);
  assert.equal(clampPlayback(-3, 90), 0);
});

test('search is case and diacritic insensitive while highlights preserve original Unicode offsets', () => {
  const text = 'Über den grünen Hügel: KI und ki. 😀 Café, Cafe\u0301.';
  const index = prepareSearch(text);
  assert.deepEqual(findTextMatches(index, 'ki').map((m) => text.slice(m.start, m.end)), ['KI', 'ki']);
  assert.deepEqual(findTextMatches(index, 'uber').map((m) => text.slice(m.start, m.end)), ['Über']);
  assert.deepEqual(findTextMatches(index, 'cafe').map((m) => text.slice(m.start, m.end)), ['Café', 'Cafe\u0301']);
  assert.deepEqual(findTextMatches(index, '😀').map((m) => text.slice(m.start, m.end)), ['😀']);
  assert.deepEqual(findTextMatches(index, '  '), []);
  assert.deepEqual(findTextMatches(index, '<script>'), []);
});

test('exports preserve all passages and timestamps link to the reader', () => {
  const passages = groupPassages([{ id: 0, start: 62.3, end: 65, text: 'Fold the blue paper.' }, { id: 1, start: 70, end: 75, text: 'Add a warm light.' }], true);
  assert.equal(exportTranscript(passages, 9, false), 'Fold the blue paper.\n\nAdd a warm light.\n');
  assert.equal(exportTranscript(passages, 9, true), '[1:02](/videos/9/watch?t=62) Fold the blue paper.\n\n[1:10](/videos/9/watch?t=70) Add a warm light.\n');
});
