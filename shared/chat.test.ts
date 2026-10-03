import { test } from 'node:test';
import assert from 'node:assert/strict';
import { answerMarkdown, askAbout, savedAnswer, validateCitations, type Message } from './chat.ts';
import { readSSE } from './sse.ts';

test('validated timed, untimed and document citations keep exact revision links in saved answers', () => {
  const sources = new Map([
    ['u001', { id: 'u001', kind: 'unit' as const, start: 62.4, excerpt: 'Fold here.' }],
    ['u002', { id: 'u002', kind: 'unit' as const, start: null, excerpt: 'Untimed.' }],
    ['d3', { id: 'd3', kind: 'doc' as const, start: null, excerpt: 'Note.' }],
  ]);
  const validated = validateCitations('Fold [u001, u999, u001]. More [u002, d3]. Unknown [d99].', sources);
  assert.deepEqual(validated.invalid, ['u999', 'd99']); assert.equal(validated.citations.length, 3);
  const message = { ...validated, context: { transcriptId: 8 }, status: 'incomplete' } as unknown as Message;
  const markdown = answerMarkdown(message, 2);
  assert.match(markdown, /\[1:02\]\(\/videos\/2\/watch\?t=62&transcriptId=8&unit=u001\)/);
  assert.match(markdown, /\[passage\]\(\/videos\/2\/watch\?transcriptId=8&unit=u002\)/);
  assert.match(markdown, /\[Note 3\]\(\/videos\/2\/documents\/3\)/);
  assert.match(savedAnswer('Why?', message, 2), /Incomplete answer/);
  assert.equal(askAbout('Selected text', null), 'About this passage "Selected text": ');
  assert.equal(askAbout('Selected text', 62), 'About [1:02] "Selected text": ');
});

test('SSE framing preserves event types, split UTF-8, CRLF, multiline data, final events and cancellation', async () => {
  const bytes = new TextEncoder().encode(': heartbeat\r\n\r\nevent: delta\r\ndata: Über\r\ndata: words\r\n\r\nevent: done\ndata: tail');
  const body = new ReadableStream<Uint8Array>({ start(c) { for (const byte of bytes) c.enqueue(new Uint8Array([byte])); c.close(); } });
  const events = []; for await (const event of readSSE(body)) events.push(event);
  assert.deepEqual(events, [{ event: 'delta', data: 'Über\nwords' }, { event: 'done', data: 'tail' }]);
  let cancelled = false;
  const open = new ReadableStream<Uint8Array>({ start(c) { c.enqueue(new TextEncoder().encode('data: first\n\n')); }, cancel() { cancelled = true; } });
  const reader = readSSE(open); await reader.next(); await reader.return(undefined); assert.equal(cancelled, true);
});
