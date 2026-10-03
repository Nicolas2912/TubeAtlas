// Request contracts shared by server (validation) and frontend (types), plus core domain types.
import { z } from 'zod';

export type Segment = { id: number; start: number | null; end: number | null; text: string };

export type TranscriptStatus = 'pending' | 'ready' | 'no_captions' | 'blocked' | 'failed';
export type JobKind = 'transcript' | 'graph';
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';

export { MAX_MANUAL_TRANSCRIPT_BYTES } from './limits.ts';

export const ImportVideoBody = z.strictObject({
  url: z.string().trim().min(1).max(2000),
  topicId: z.number().int().positive().optional(),
});

export const UpdateVideoBody = z.strictObject({
  playbackSeconds: z.number().finite().min(0).optional(),
  topicIds: z.array(z.number().int().positive()).max(100).optional(),
});

export const ManualTranscriptBody = z.strictObject({
  format: z.enum(['text', 'vtt', 'srt']),
  content: z.string().min(1),
  language: z.string().trim().min(2).max(20).optional(),
});

export const TopicBody = z.strictObject({ name: z.string().trim().min(1).max(80) });

export const VideoListQuery = z.object({ topicId: z.coerce.number().int().positive().optional() });

export const IdParam = z.object({ id: z.coerce.number().int().positive() });

const TextKind = z.enum(['note', 'summary', 'study_guide', 'qa']);
const DocumentTitle = z.string().trim().min(1).max(200);
export const CreateDocumentBody = z.strictObject({
  title: DocumentTitle,
  kind: TextKind.default('note'),
  markdown: z.string().default(''),
});
export const UpdateDocumentBody = z.strictObject({
  title: DocumentTitle.optional(),
  kind: TextKind.optional(),
  markdown: z.string().optional(),
  appendMarkdown: z.string().min(1).optional(),
}).refine((body) => Object.keys(body).length > 0, 'Provide a document change.')
  .refine((body) => body.markdown === undefined || body.appendMarkdown === undefined, 'Replace or append Markdown, not both.');
