import { Hono } from 'hono';
import { streamSSE } from 'hono/streaming';
import { ChatMessageBody, ConversationBody, IdParam, RenameConversationBody } from '../../../shared/api.ts';
import type { ChatService } from '../services/chat.ts';
import { validate } from '../validate.ts';

export function videoConversationRoutes(chat: ChatService) {
  return new Hono()
    .get('/:id/conversations', validate('param', IdParam), (c) => c.json(chat.listConversations(c.req.valid('param').id)))
    .post('/:id/conversations', validate('param', IdParam), validate('json', ConversationBody), (c) => c.json(chat.createConversation(c.req.valid('param').id, c.req.valid('json').title), 201));
}
export function conversationRoutes(chat: ChatService) {
  return new Hono()
    .get('/:id', validate('param', IdParam), (c) => c.json(chat.conversationDetail(c.req.valid('param').id)))
    .patch('/:id', validate('param', IdParam), validate('json', RenameConversationBody), (c) => c.json(chat.renameConversation(c.req.valid('param').id, c.req.valid('json').title)))
    .delete('/:id', validate('param', IdParam), (c) => { chat.deleteConversation(c.req.valid('param').id); return c.body(null, 204); })
    .post('/:id/messages', validate('param', IdParam), validate('json', ChatMessageBody), (c) => {
      const run = chat.start(c.req.valid('param').id, c.req.valid('json'));
      return streamSSE(c, async (stream) => {
        const disconnected = Promise.withResolvers<void>();
        stream.onAbort(() => disconnected.resolve());
        let cursor = 0;
        while (!stream.aborted) {
          while (cursor < run.events.length && !stream.aborted) {
            const event = run.events[cursor++]!;
            await stream.writeSSE({ event: event.event, data: JSON.stringify(event.data) });
          }
          if (run.finished || stream.aborted) return;
          await Promise.race([run.changed.promise, disconnected.promise]);
        }
      });
    });
}
export function messageRoutes(chat: ChatService) {
  return new Hono()
    .get('/:id', validate('param', IdParam), (c) => c.json(chat.getMessage(c.req.valid('param').id)))
    .post('/:id/cancel', validate('param', IdParam), (c) => c.json(chat.cancel(c.req.valid('param').id)));
}
