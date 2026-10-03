import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Segment } from '../../../shared/api.ts';
import { estimateTokens, normalizeForMatching } from '../../../shared/units.ts';
import { buildUnits } from './units.ts';

const caption = (id: number, text: string, start = id, end = start + 2): Segment => ({ id, text, start, end });
const words = (n: number) => Array.from({ length: n }, (_, i) => `word${i + 1}`).join(' ');

test('units preserve nested overlaps, source IDs, and the immutable input', () => {
  const input = [caption(7, 'Fold the paper.', 2, 20), caption(8, 'Light the lamp.', 3, 5)];
  const before = structuredClone(input);
  const result = buildUnits(input);
  assert.deepEqual(result.units, [{ id: 'u001', segmentIds: [7, 8], start: 2, end: 20, text: 'Fold the paper. Light the lamp.', turnStart: false, annotations: [] }]);
  assert.deepEqual(input, before);
  assert.deepEqual(buildUnits(input), result);
});

test('every speaker marker splits units, including markers inside a segment or word', () => {
  const { units } = buildUnits([caption(0, 'First voice>>Second voice >> Third voice'), caption(1, '>> Final voice')]);
  assert.deepEqual(units.map((u) => [u.text, u.turnStart, u.segmentIds]), [
    ['First voice', false, [0]], ['Second voice', true, [0]], ['Third voice', true, [0]], ['Final voice', true, [1]],
  ]);
  assert.deepEqual(units.map((u) => [u.start, u.end]), [[0, 2], [0, 2], [0, 2], [1, 3]]);
  assert.equal(buildUnits([caption(0, '>> >> Hello')]).units.length, 1);
});

test('non-speech tags become annotations, including standalone and trailing tags', () => {
  const { units } = buildUnits([caption(0, '[soft music] Fold [laughter] the paper.'), caption(1, '[applause]')]);
  assert.equal(units[0]!.text, 'Fold the paper.');
  assert.deepEqual(units[0]!.annotations, ['[soft music]', '[laughter]', '[applause]']);
  assert.deepEqual(units[0]!.segmentIds, [0, 1]);
  const trailing = buildUnits([caption(0, `${words(15)}. [music]`)]).units;
  assert.deepEqual(trailing[0]!.annotations, ['[music]']);
  assert.deepEqual(buildUnits([caption(0, '[music]')]).units, []);
});

test('rolling duplicate prefixes have provenance; ordinary repetition and speaker changes stay intact', () => {
  const result = buildUnits([caption(0, 'We fold blue paper'), caption(1, 'blue paper into a lantern')]);
  assert.equal(result.units[0]!.text, 'We fold blue paper into a lantern');
  assert.deepEqual(result.deduplications, [{ segmentId: 1, previousSegmentId: 0, text: 'blue paper' }]);
  const ordinary = [caption(0, 'Turn the wheel'), caption(1, 'wheel again.'), caption(2, 'again and again.'), caption(3, 'Try once more.')];
  assert.equal(buildUnits(ordinary).units.map((u) => u.text).join(' '), ordinary.map((s) => s.text).join(' '));
  assert.equal(buildUnits([caption(0, 'blue paper'), caption(1, 'blue paper', 5)]).deduplications.length, 0);
  assert.equal(buildUnits([caption(0, 'blue paper'), caption(1, '>> blue paper')]).deduplications.length, 0);
  assert.equal(buildUnits([caption(0, 'paperwork'), caption(1, 'paper work')]).deduplications.length, 0);
  const full = buildUnits([caption(0, 'blue paper'), caption(1, 'blue paper'), caption(2, 'blue paper lantern')]);
  assert.equal(full.units[0]!.text, 'blue paper lantern');
  assert.equal(full.deduplications.length, 2);
});

test('sentence boundaries wait for fifteen words; long sentences close at caption boundaries', () => {
  for (const punctuation of ['.', '?', '!', '…']) {
    const { units } = buildUnits([caption(0, `A short sentence. ${words(12)}${punctuation} ${words(16)}${punctuation}`)]);
    assert.deepEqual(units.map((u) => u.text.split(' ').length), [15, 16]);
    assert.deepEqual(units.map((u) => u.segmentIds), [[0], [0]]);
  }
  const { units } = buildUnits([caption(0, words(30)), caption(1, `more ${words(15)}`), caption(2, 'The remainder')]);
  assert.deepEqual(units.map((u) => u.text.split(' ').length), [46, 2]);
  assert.deepEqual(units.map((u) => u.segmentIds), [[0, 1], [2]]);
});

test('normalization keeps display punctuation while matching unifies it, and untimed paragraphs stay untimed', () => {
  const input: Segment[] = [
    { id: 0, start: null, end: null, text: '  Cafe\u0301\n “blue—paper”   ' },
    { id: 1, start: null, end: null, text: 'Second paragraph.' },
  ];
  const result = buildUnits(input);
  assert.deepEqual(result.units.map((u) => [u.text, u.start, u.end]), [['Café “blue—paper”', null, null], ['Second paragraph.', null, null]]);
  assert.equal(normalizeForMatching(result.units[0]!.text), 'Café "blue-paper"');
  assert.equal(estimateTokens(''), 0);
  assert.equal(estimateTokens('12345678'), 3);
  assert.deepEqual(buildUnits([]), { unitsVersion: 1, units: [], deduplications: [] });
});

test('bad times and duplicate source IDs are rejected; overlaps are accepted', () => {
  for (const [start, end] of [[NaN, 2], [0, Infinity], [-1, 2], [3, 2], [null, 2], [0, null]] as [number | null, number | null][]) {
    assert.throws(() => buildUnits([{ id: 0, text: 'Hello', start, end }]), /valid, consistently timed/u);
  }
  assert.throws(() => buildUnits([caption(0, 'One'), caption(0, 'Two')]), /unique IDs/u);
  assert.throws(() => buildUnits([caption(0, 'One'), { id: 1, text: 'Two', start: null, end: null }]), /consistently timed/u);
  assert.doesNotThrow(() => buildUnits([caption(0, 'One', 0, 10), caption(1, 'Two', 1, 2)]));
});

test('unit IDs stay unique beyond three digits', () => {
  const { units } = buildUnits(Array.from({ length: 1234 }, (_, id) => caption(id, `>> voice${id}`)));
  assert.equal(units[0]!.id, 'u001');
  assert.equal(units[998]!.id, 'u999');
  assert.equal(units[999]!.id, 'u1000');
  assert.equal(units.at(-1)!.id, 'u1234');
});
