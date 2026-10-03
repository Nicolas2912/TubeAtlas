import { useCallback, useEffect, useRef, useState } from 'react';
import { api, checked, errorMessage, unwrap } from '../../api.ts';
import { readSSE } from '../../../../shared/sse.ts';
import { clipText, type Conversation, type Message } from '../../../../shared/chat.ts';

type Detail = Conversation & { messages: Message[] };
export function useChat(id: string, videoId: number, onChanged: () => void) {
  const [detail, setDetail] = useState<Detail>();
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const connection = useRef<AbortController | null>(null);
  const alive = useRef(true);
  const changed = useRef(onChanged); changed.current = onChanged;
  const upsert = useCallback((message: Message) => setDetail((current) => current && ({ ...current, messages: [...current.messages.filter((item) => item.id !== message.id), message].sort((a, b) => a.id - b.id) })), []);
  const reload = useCallback(async () => {
    try {
      const next = await unwrap(api.conversations[':id'].$get({ param: { id } }));
      if (next.videoId !== videoId) throw new Error('This conversation belongs to another video.');
      if (alive.current) { setDetail(next); setError(null); }
    } catch (err) { if (alive.current) setError(errorMessage(err)); }
  }, [id, videoId]);
  useEffect(() => {
    alive.current = true; void reload();
    return () => { alive.current = false; connection.current?.abort(); };
  }, [reload]);
  const generatingId = detail?.messages.find((message) => message.status === 'generating')?.id;
  useEffect(() => {
    if (!generatingId || sending) return;
    let active = true; let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const message = await unwrap(api.messages[':id'].$get({ param: { id: String(generatingId) } }));
        if (!active) return;
        upsert(message); setError(null);
        if (message.status !== 'generating') { changed.current(); return; }
      } catch (err) { if (active) setError(errorMessage(err)); }
      if (active) timer = setTimeout(poll, 1500);
    }
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [generatingId, sending, upsert]);

  async function ask(content: string, documentIds: number[], acknowledged: () => void) {
    if (connection.current || generatingId) return;
    const controller = new AbortController(); connection.current = controller;
    setSending(true); setError(null);
    let assistantId: number | undefined; let terminal = false;
    try {
      const response = await checked(api.conversations[':id'].messages.$post({ param: { id }, json: { content, documentIds } }, { init: { signal: controller.signal } }));
      if (!response.body) throw new Error('The answer stream is unavailable.');
      for await (const event of readSSE(response.body)) {
        if (!alive.current) break;
        const data = JSON.parse(event.data);
        if (event.event === 'start') {
          assistantId = data.assistantMessageId;
          const base = { conversationId: Number(id), citations: [], context: null, model: null, usage: null, errorCode: null, createdAt: new Date().toISOString() };
          setDetail((current) => current && ({ ...current, title: current.title === 'New conversation' && current.messages.length === 0 ? clipText(content, 60) : current.title,
            messages: [...current.messages, { ...base, id: data.userMessageId, role: 'user', content, status: 'complete' }, { ...base, id: data.assistantMessageId, role: 'assistant', content: '', status: 'generating' }] }));
          acknowledged(); changed.current();
        } else if (event.event === 'delta') {
          setDetail((current) => current && ({ ...current, messages: current.messages.map((message) => message.id === assistantId ? { ...message, content: message.content + data.text } : message) }));
        } else if (event.event === 'done') { upsert(data.message); terminal = true; changed.current(); }
        else if (event.event === 'error') {
          if (assistantId) upsert(await unwrap(api.messages[':id'].$get({ param: { id: String(assistantId) } })));
          setError(data.message); terminal = true; changed.current();
        }
      }
      if (!terminal && !controller.signal.aborted) throw new Error('The connection closed. Checking the saved answer…');
    } catch (err) {
      if (alive.current && !controller.signal.aborted) {
        // A request may have reached the server even if its acknowledgement was lost. Never retry it automatically.
        await reload(); setError(errorMessage(err));
      }
    } finally {
      if (connection.current === controller) connection.current = null;
      if (alive.current) setSending(false);
    }
  }
  async function stop() {
    if (!generatingId) return;
    try {
      const message = await unwrap(api.messages[':id'].cancel.$post({ param: { id: String(generatingId) } }));
      if (alive.current) { upsert(message); setError(null); }
      connection.current?.abort(); changed.current();
    } catch (err) { if (alive.current) setError(errorMessage(err)); }
  }
  return { detail, error, sending, generatingId, ask, stop, reload };
}
