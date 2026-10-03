import type { Unit } from './units.ts';
import { formatTime, watchLink } from './time.ts';

export type Citation = { id: string; kind: 'unit' | 'doc'; start: number | null; excerpt: string };
export type ChatContext = {
  transcriptId: number; revision: number; unitsVersion: number; mode: 'full' | 'retrieval';
  documentIds: number[]; unitIds: string[];
};
export type Message = {
  id: number; conversationId: number; role: 'user' | 'assistant'; content: string;
  status: 'generating' | 'complete' | 'incomplete' | 'interrupted' | 'failed';
  citations: Citation[]; context: ChatContext | null; model: string | null;
  usage: { usage: { promptTokens: number; completionTokens: number; reasoningTokens: number; cost: number | null } | null;
    provider: string | null; generationId: string | null; latencyMs: number } | null;
  errorCode: string | null; createdAt: string;
};
export type Conversation = { id: number; videoId: number; title: string; createdAt: string; updatedAt: string };
export type ChatEvent =
  | { event: 'start'; data: { userMessageId: number; assistantMessageId: number } }
  | { event: 'delta'; data: { text: string } }
  | { event: 'done'; data: { message: Message } }
  | { event: 'error'; data: { code: string; message: string } };

export const CITATION_PATTERN = /\[([ud]\d+(?:\s*,\s*[ud]\d+)*)\]/g;
export const clipText = (text: string, characters: number): string => text.slice(0, characters).replace(/[\uD800-\uDBFF]$/u, '');

/** Context membership establishes provenance; it does not prove that a claim is entailed. */
export function validateCitations(text: string, sources: Map<string, Citation>) {
  const used = new Map<string, Citation>();
  const invalid = new Set<string>();
  const content = text.replace(CITATION_PATTERN, (_match, group: string) => {
    const ids = [...new Set(group.split(',').map((id) => id.trim()))];
    const valid = ids.filter((id) => {
      const source = sources.get(id);
      if (source) { used.set(id, source); return true; }
      invalid.add(id); return false;
    });
    return valid.length ? `[${valid.join(', ')}]` : '';
  });
  return { content: content.trim(), citations: [...used.values()], invalid: [...invalid] };
}

export function unitCitation(unit: Unit): Citation {
  return { id: unit.id, kind: 'unit', start: unit.start, excerpt: clipText(unit.text, 200) };
}

/** A revision and unit target keep old citations accurate, including untimed or split captions. */
export function citationLink(videoId: number, citation: Citation, transcriptId: number): string {
  if (citation.kind === 'doc') return `/videos/${videoId}/documents/${citation.id.slice(1)}`;
  const base = citation.start === null ? `/videos/${videoId}/watch?` : `${watchLink(videoId, citation.start)}&`;
  return `${base}transcriptId=${transcriptId}&unit=${encodeURIComponent(citation.id)}`;
}

export function answerMarkdown(message: Message, videoId: number): string {
  const citations = new Map(message.citations.map((citation) => [citation.id, citation]));
  return message.content.replace(CITATION_PATTERN, (_match, group: string) => group.split(',').map((id) => {
    const citation = citations.get(id.trim());
    if (!citation || !message.context) return '';
    const label = citation.kind === 'doc' ? `Note ${citation.id.slice(1)}` : citation.start === null ? 'passage' : formatTime(citation.start);
    return `[${label}](${citationLink(videoId, citation, message.context.transcriptId)})`;
  }).filter(Boolean).join(' '));
}

export function savedAnswer(question: string, message: Message, videoId: number): string {
  const status = message.status === 'complete' ? '' : `\n\n_${message.status === 'incomplete' ? 'Incomplete answer' : message.status === 'interrupted' ? 'Interrupted before completion' : 'Response failed; partial answer'}._`;
  return `## Question\n\n${question}\n\n## Answer\n\n${answerMarkdown(message, videoId)}${status}`;
}

export function askAbout(text: string, start: number | null): string {
  return `About ${start === null ? 'this passage' : `[${formatTime(start)}]`} "${text}": `;
}
