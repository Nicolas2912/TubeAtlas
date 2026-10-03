import type { z } from 'zod';
import type { ChatMessageBody } from '../../../shared/api.ts';
import { clipText, validateCitations, type ChatContext, type ChatEvent, type Citation, type Conversation, type Message } from '../../../shared/chat.ts';
import type { Config } from '../config.ts';
import { tx, type Db } from '../db.ts';
import { AppError } from '../errors.ts';
import { createOpenRouter, type ChatMetadata, type OpenRouter } from '../integrations/openrouter.ts';
import { getDocument, type Document } from './documents.ts';
import { getCurrentTranscript, type Transcript } from './transcripts.ts';
import { getVideo } from './videos.ts';
import { createRetrieval } from './retrieval.ts';
import { chatContext, prepareChat } from './chat-prompt.ts';

export type ChatProvider = Pick<OpenRouter, 'chatStream' | 'embed'>;
type MessageRow = Omit<Message, 'citations' | 'context' | 'usage'> & { citationsJson: string; contextJson: string | null; usageJson: string | null };
const messageSelect = `SELECT id, conversation_id AS conversationId, role, content, status, citations_json AS citationsJson,
  context_json AS contextJson, model, usage_json AS usageJson, error_code AS errorCode, created_at AS createdAt FROM messages`;
const conversationSelect = 'SELECT id, video_id AS videoId, title, created_at AS createdAt, updated_at AS updatedAt FROM conversations';
const NOW = "strftime('%Y-%m-%dT%H:%M:%fZ','now')";
type Runtime = {
  id: number; userId: number; conversation: Conversation; controller: AbortController; content: string; context: ChatContext;
  sources: Map<string, Citation>; metadata: ChatMetadata; events: ChatEvent[]; finished: boolean;
  changed: ReturnType<typeof Promise.withResolvers<void>>; pending: Promise<void>; startedAt: number; lastFlush: number;
};
export type ChatService = ReturnType<typeof createChatService>;

/** Each run outlives its HTTP subscriber. Only explicit Stop/deletion/shutdown abort paid work. */
export function createChatService({ db, config, provider: supplied }: { db: Db; config: Config; provider?: ChatProvider }) {
  const provider = supplied ?? (config.openrouterApiKey ? createOpenRouter({ apiKey: config.openrouterApiKey, baseUrl: config.openrouterBaseUrl }) : undefined);
  const retrieval = provider ? createRetrieval({ db, provider, model: config.embeddingModel }) : undefined;
  const running = new Map<number, Runtime>();
  let stopped = false;

  function decode(row: MessageRow): Message {
    const { citationsJson, contextJson, usageJson, ...message } = row;
    return { ...message, citations: JSON.parse(citationsJson), context: contextJson ? JSON.parse(contextJson) : null, usage: usageJson ? JSON.parse(usageJson) : null };
  }
  function getMessage(id: number): Message {
    const row = db.prepare(`${messageSelect} WHERE id = ?`).get(id) as MessageRow | undefined;
    if (!row) throw new AppError(404, 'NOT_FOUND', 'Message not found.');
    const message = decode(row);
    const run = running.get(id);
    if (message.status === 'generating' && run) return { ...message, content: run.content, context: run.context };
    return message;
  }
  function getConversation(id: number): Conversation {
    const conversation = db.prepare(`${conversationSelect} WHERE id = ?`).get(id) as Conversation | undefined;
    if (!conversation) throw new AppError(404, 'NOT_FOUND', 'Conversation not found.');
    return conversation;
  }
  function listConversations(videoId: number): Conversation[] {
    getVideo(db, videoId);
    return db.prepare(`${conversationSelect} WHERE video_id = ? ORDER BY updated_at DESC, id DESC`).all(videoId) as Conversation[];
  }
  function conversationDetail(id: number) {
    return { ...getConversation(id), messages: (db.prepare(`${messageSelect} WHERE conversation_id = ? ORDER BY id`).all(id) as MessageRow[]).map((row) => getMessage(row.id)) };
  }
  function createConversation(videoId: number, title = 'New conversation'): Conversation {
    getVideo(db, videoId);
    const id = Number(db.prepare('INSERT INTO conversations (video_id, title) VALUES (?, ?)').run(videoId, title).lastInsertRowid);
    return getConversation(id);
  }
  function renameConversation(id: number, title: string): Conversation {
    getConversation(id);
    db.prepare(`UPDATE conversations SET title = ?, updated_at = ${NOW} WHERE id = ?`).run(title, id);
    return getConversation(id);
  }
  function publish(run: Runtime, event: ChatEvent) {
    run.events.push(event);
    run.changed.resolve();
    run.changed = Promise.withResolvers<void>();
  }
  function finish(run: Runtime, status: Message['status'], error?: { code: string; message: string }) {
    if (run.finished) return;
    const validated = validateCitations(run.content, run.sources);
    if (validated.invalid.length) console.warn('Dropped chat citation IDs', { messageId: run.id, ids: validated.invalid });
    const accounting: Message['usage'] = { usage: run.metadata.usage, provider: run.metadata.provider,
      generationId: run.metadata.generationId, latencyMs: Date.now() - run.startedAt };
    tx(db, () => {
      db.prepare(`UPDATE messages SET content = ?, status = ?, citations_json = ?, context_json = ?, model = ?, usage_json = ?, error_code = ?
        WHERE id = ? AND status = 'generating'`).run(validated.content, status, JSON.stringify(validated.citations), JSON.stringify(run.context),
          run.metadata.model, JSON.stringify(accounting), error?.code ?? null, run.id);
      db.prepare(`UPDATE conversations SET updated_at = ${NOW} WHERE id = ?`).run(run.conversation.id);
    });
    run.finished = true;
    console.info('Chat usage', { messageId: run.id, transcriptId: run.context.transcriptId, status,
      model: run.metadata.model, ...accounting, cost: accounting.usage?.cost ?? null });
    if (error) publish(run, { event: 'error', data: error });
    else publish(run, { event: 'done', data: { message: getMessage(run.id) } });
  }
  async function generate(run: Runtime, input: { transcript: Transcript; documents: Document[]; history: Message[]; question: string; title: string }) {
    try {
      const prepared = await prepareChat({ ...input, retrieval: retrieval!, signal: run.controller.signal });
      if (run.finished) return;
      run.context = prepared.context; run.sources = prepared.sources;
      db.prepare('UPDATE messages SET context_json = ? WHERE id = ? OR id = ?').run(JSON.stringify(run.context), run.id, run.userId);
      const result = await provider!.chatStream({ model: config.chatModel, messages: prepared.messages, maxTokens: 2000, signal: run.controller.signal,
        onMetadata: (metadata) => { run.metadata = metadata; },
        onDelta: (text) => {
          if (run.finished) return;
          run.content += text;
          if (Date.now() - run.lastFlush >= 500) {
            const partial = validateCitations(run.content, run.sources);
            db.prepare("UPDATE messages SET content = ?, citations_json = ?, model = ? WHERE id = ? AND status = 'generating'")
              .run(partial.content, JSON.stringify(partial.citations), run.metadata.model, run.id);
            run.lastFlush = Date.now();
          }
          publish(run, { event: 'delta', data: { text } });
        },
      });
      if (!run.finished) {
        run.metadata = result;
        run.content = result.text;
        if (result.finishReason === 'length') finish(run, 'incomplete');
        else if (result.finishReason === 'content_filter' || !result.text.trim()) {
          finish(run, 'failed', { code: 'EMPTY_OR_FILTERED_RESPONSE', message: 'The provider did not return a usable answer. Your question is saved.' });
        } else if (result.finishReason !== 'stop') {
          finish(run, 'failed', { code: 'INCOMPLETE_STREAM', message: 'The provider stream ended without confirming completion. The partial answer is saved.' });
        } else finish(run, 'complete');
      }
    } catch (error) {
      if (!run.finished) {
        if (run.controller.signal.aborted) finish(run, 'incomplete');
        else {
          const known = error instanceof AppError;
          if (!known) console.error('Chat generation failed', { messageId: run.id, error });
          finish(run, 'failed', { code: known ? error.code : 'CHAT_FAILED', message: known ? error.message : 'Could not finish the answer. Your question and partial response are saved.' });
        }
      }
    } finally { if (running.get(run.id) === run) running.delete(run.id); }
  }
  function start(id: number, body: z.infer<typeof ChatMessageBody>) {
    const conversation = getConversation(id);
    if (!provider || !config.openrouterApiKey) throw new AppError(503, 'AI_NOT_CONFIGURED', 'AI is not configured. Check Settings to enable chat.');
    if (stopped) throw new AppError(503, 'SHUTTING_DOWN', 'TubeAtlas is shutting down.');
    if (db.prepare("SELECT 1 FROM messages WHERE conversation_id = ? AND status = 'generating'").get(id)) {
      throw new AppError(409, 'ALREADY_GENERATING', 'This conversation already has an answer in progress.');
    }
    if (running.size >= 2) throw new AppError(429, 'AI_BUSY', 'Two answers are already in progress. Try again when one finishes.');
    const transcript = getCurrentTranscript(db, conversation.videoId);
    if (!transcript?.units.length) throw new AppError(409, 'NO_TRANSCRIPT', 'Add a transcript with readable text before asking a question.');
    const documents = [...new Set(body.documentIds)].map((documentId) => {
      const doc = getDocument(db, documentId);
      if (doc.videoId !== conversation.videoId || doc.kind === 'attachment') throw new AppError(400, 'INVALID_DOCUMENT_SOURCE', 'Choose text documents belonging to this video. Attachments cannot be used as sources.');
      return doc;
    });
    const history = (db.prepare(`${messageSelect} WHERE conversation_id = ? AND status IN ('complete','incomplete') ORDER BY id DESC LIMIT 6`).all(id) as MessageRow[]).reverse().map(decode);
    const context = chatContext(transcript, documents);
    const firstQuestion = !db.prepare("SELECT 1 FROM messages WHERE conversation_id = ? AND role = 'user'").get(id);
    const { userMessageId, assistantMessageId } = tx(db, () => {
      const insert = db.prepare('INSERT INTO messages (conversation_id, role, content, status, context_json) VALUES (?, ?, ?, ?, ?)');
      const userMessageId = Number(insert.run(id, 'user', body.content, 'complete', JSON.stringify(context)).lastInsertRowid);
      const assistantMessageId = Number(insert.run(id, 'assistant', '', 'generating', JSON.stringify(context)).lastInsertRowid);
      db.prepare(`UPDATE conversations SET title = CASE WHEN title = 'New conversation' AND ? = 1 THEN ? ELSE title END, updated_at = ${NOW} WHERE id = ?`)
        .run(firstQuestion ? 1 : 0, clipText(body.content, 60), id);
      return { userMessageId, assistantMessageId };
    });
    const run: Runtime = { id: assistantMessageId, userId: userMessageId, conversation, controller: new AbortController(), content: '', context, sources: new Map(),
      metadata: { model: config.chatModel, provider: null, usage: null, generationId: null, finishReason: null, latencyMs: 0 },
      events: [{ event: 'start', data: { userMessageId, assistantMessageId } }], changed: Promise.withResolvers<void>(), finished: false,
      pending: Promise.resolve(), startedAt: Date.now(), lastFlush: 0 };
    running.set(run.id, run);
    run.pending = generate(run, { transcript, documents, history, question: body.content, title: getVideo(db, conversation.videoId).title }).catch((error) => {
      // A persistence failure must not become an unhandled rejection or a fabricated saved answer.
      console.error('Could not persist chat response', { messageId: run.id, error });
      run.finished = true;
      publish(run, { event: 'error', data: { code: 'CHAT_PERSIST_FAILED', message: 'The answer could not be saved. Check the server and reopen this conversation.' } });
    });
    // The route subscribes to this buffer; disconnection never passes its abort signal to generate.
    return run;
  }
  function cancel(id: number): Message {
    const message = getMessage(id);
    if (message.role !== 'assistant') throw new AppError(400, 'NOT_ASSISTANT', 'Only an assistant response can be stopped.');
    const run = running.get(id);
    if (run && !run.finished) { run.controller.abort(); finish(run, 'incomplete'); }
    else if (message.status === 'generating') db.prepare("UPDATE messages SET status = 'interrupted' WHERE id = ?").run(id);
    return getMessage(id);
  }
  function cancelForVideo(videoId: number) {
    for (const run of running.values()) if (run.conversation.videoId === videoId) cancel(run.id);
  }
  function deleteConversation(id: number) {
    getConversation(id);
    for (const run of running.values()) if (run.conversation.id === id) cancel(run.id);
    db.prepare('DELETE FROM conversations WHERE id = ?').run(id);
  }
  async function stop() {
    stopped = true;
    const runs = [...running.values()];
    for (const run of runs) { run.controller.abort(); finish(run, 'interrupted'); }
    await Promise.all(runs.map((run) => run.pending));
  }
  return { getMessage, getConversation, conversationDetail, listConversations, createConversation, renameConversation,
    deleteConversation, start, cancel, cancelForVideo, stop, idle: () => Promise.all([...running.values()].map((run) => run.pending)) };
}
