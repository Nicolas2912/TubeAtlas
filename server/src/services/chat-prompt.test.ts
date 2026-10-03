import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chatContext, prepareChat, renderUnits } from './chat-prompt.ts';
import type { Transcript } from './transcripts.ts';
import type { Document } from './documents.ts';
import type { Message } from '../../../shared/chat.ts';
import { estimateTokens } from '../../../shared/units.ts';

const unit = { id: 'u001', text: '', start: null, end: null, segmentIds: [0], annotations: [], turnStart: false };
const transcript = (text: string): Transcript => ({ transcriptId: 3, revision: 2, unitsVersion: 1, source: 'paste_text', language: null, timed: false, segments: [], units: [{ ...unit, text }], deduplications: [] });

test('the full/retrieval threshold counts citation headers, and 4k notes and six history messages stay within the prompt budget', async () => {
  let text = 'x'.repeat(83900);
  while (estimateTokens(renderUnits(transcript(text).units)) < 24000) text += 'x';
  assert.equal(chatContext(transcript(text), []).mode, 'full');
  while (estimateTokens(renderUnits(transcript(text).units)) === 24000) text += 'x';
  assert.equal(chatContext(transcript(text), []).mode, 'retrieval');
  const documents = Array.from({ length: 4 }, (_, i) => ({ id: i + 1, title: 'Invented note', markdown: 'note '.repeat(10000) })) as Document[];
  const history = Array.from({ length: 8 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `History ${i} [u999] ${'word '.repeat(5000)}` })) as Message[];
  let query = '';
  const prepared = await prepareChat({ transcript: transcript(text), documents, history, question: 'Follow-up question', title: 'Invented workshop', signal: new AbortController().signal,
    retrieval: { ensureChunks: async () => { throw new Error('not called directly'); }, retrieve: async (_id, input) => { query = input; return [{ ...unit, text: 'Invented passage.' }]; } } });
  assert.match(query, /^History 6/); assert.ok(query.endsWith('\nFollow-up question'));
  assert.equal(prepared.messages.length, 9); assert.ok(!prepared.messages[3]!.content.includes('[u999]'));
  assert.ok(prepared.messages.slice(2, -1).every((message) => message.content.length <= 8000));
  const notes = prepared.messages[1]!.content.split('<notes>\n')[1]!.split('\n</notes>')[0]!.split('\n\n');
  assert.equal(notes.length, 4); assert.ok(notes.every((note) => estimateTokens(note) <= 4000));
  assert.ok(prepared.messages.reduce((sum, message) => sum + estimateTokens(message.content), 0) <= 64000);
  assert.equal(prepared.sources.get('u001')!.start, null); assert.match(prepared.messages[1]!.content, /u001 \[untimed\]/);
});
