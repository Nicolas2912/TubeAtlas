import { test } from 'node:test';
import assert from 'node:assert/strict';
import { activeUnit, clampPlayback, exportTranscript, findTextMatches, groupPassages, playbackTarget, prepareSearch } from './watch.ts';
import type { Unit } from './units.ts';

const unit = (id: number, text: string, start: number | null, end: number | null, turnStart = false): Unit => ({
  id: `u${id}`, segmentIds: [id], text, start, end, turnStart, annotations: [],
});

test('paragraphs group whole units at sixty words and always break on speaker turns', () => {
  const units = [unit(4, 'paper '.repeat(30).trim(), 0, 4), unit(5, 'lantern '.repeat(30).trim(), 3, 6),
    unit(6, 'Fold once.', 5, 7), unit(7, 'Light the lantern.', 6, 12, true)];
  const passages = groupPassages(units);
  assert.deepEqual(passages.map((p) => p.units.map((u) => u.id)), [['u4', 'u5'], ['u6'], ['u7']]);
  assert.equal(activeUnit(units, 5.5), 2);
  assert.equal(activeUnit(units, 12), -1);
  assert.equal(activeUnit([], 0), -1);
  assert.equal(activeUnit([unit(0, 'Long caption', 0, 10), unit(1, 'Nested caption', 1, 2)], 5), 0);
  assert.equal(activeUnit([unit(0, 'Before gap', 0, 1), unit(1, 'After gap', 3, 4)], 2), -1);
  const split = [unit(0, 'First voice', 0, 2), unit(1, 'Second voice', 0, 2, true)];
  assert.equal(activeUnit(split, 1), 1); // No invented time inside a shared caption.
});

test('untimed paragraphs stay separate and never acquire fabricated timestamps', () => {
  const units = [unit(0, 'First paragraph.', null, null), unit(1, 'Second paragraph.', null, null)];
  const passages = groupPassages(units);
  assert.equal(passages.length, 2);
  assert.equal(activeUnit(units, 1), -1);
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
  const punctuation = '“blue—paper”';
  assert.deepEqual(findTextMatches(prepareSearch(punctuation), '"blue-paper"').map((m) => punctuation.slice(m.start, m.end)), [punctuation]);
});

test('exports preserve all passages and timestamps link to the reader', () => {
  const passages = groupPassages([unit(0, 'Fold the blue paper.', 62.3, 65), unit(1, 'Add a warm light.', 70, 75, true)]);
  assert.equal(exportTranscript(passages, 9, false), 'Fold the blue paper.\n\nAdd a warm light.\n');
  assert.equal(exportTranscript(passages, 9, true), '[1:02](/videos/9/watch?t=62) Fold the blue paper.\n\n[1:10](/videos/9/watch?t=70) Add a warm light.\n');
});
