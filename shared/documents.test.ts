import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatMarkdown, transcriptQuote } from './documents.ts';

test('formatting wraps selected text, prefixes whole lines, and retains text outside the selection', () => {
  assert.deepEqual(formatMarkdown('Fold the paper.', 5, 8, 'bold'), { text: 'Fold **the** paper.', start: 7, end: 10 });
  assert.equal(formatMarkdown('One\nTwo\nThree', 5, 6, 'h2').text, 'One\n## Two\nThree');
  assert.equal(formatMarkdown('One\nTwo\nThree', 0, 8, 'numbered').text, '1. One\n2. Two\nThree');
  assert.equal(formatMarkdown('## Lantern', 0, 10, 'paragraph').text, 'Lantern');
  assert.equal(formatMarkdown('Paper', 0, 5, 'link', '/videos/7/watch?t=12').text, '[Paper](/videos/7/watch?t=12)');
  assert.equal(formatMarkdown('Paper', 5, 5, 'italic').text, 'Paper*text*');
});
test('transcript quotes retain every selected line and omit source times for untimed text', () => {
  assert.equal(transcriptQuote('Fold once.\nKeep corners even.', 4, 62.4), '> "Fold once.\n> Keep corners even."\n> — [1:02](/videos/4/watch?t=62)');
  assert.equal(transcriptQuote('Fold once.', 4, null), '> "Fold once."');
});
