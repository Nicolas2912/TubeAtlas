import { useEffect, useRef, useState } from 'react';
import { Link, NavLink, useLocation, useNavigate, useParams } from 'react-router';
import { api, errorMessage, send, unwrap, useApi } from '../../api.ts';
import { useVideoContext } from '../../app/VideoLayout.tsx';
import { Markdown } from '../../app/Markdown.tsx';
import { LoadError } from '../../components/LoadError.tsx';
import { answerMarkdown } from '../../../../shared/chat.ts';
import { useChat } from './useChat.ts';
import { AnswerActions } from './AnswerActions.tsx';

const chips = ['Explain more simply', 'Test my understanding', 'Give an example'];
function ConversationView({ id, onChanged }: { id: string; onChanged: () => void }) {
  const { video } = useVideoContext();
  const location = useLocation();
  const chat = useChat(id, video.id, onChanged);
  const documents = useApi(() => unwrap(api.videos[':id'].documents.$get({ param: { id: String(video.id) } })), [video.id]);
  const health = useApi(() => unwrap(api.health.$get()), []);
  const [draft, setDraft] = useState<string>(() => typeof location.state?.prefill === 'string' ? location.state.prefill : '');
  const [selectedNotes, setSelectedNotes] = useState<number[]>([]);
  const [rename, setRename] = useState(false);
  const [title, setTitle] = useState('');
  const [titleError, setTitleError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const composer = useRef<HTMLTextAreaElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const prefill = useRef(location.state?.prefill);
  useEffect(() => {
    if (chat.detail && prefill.current) { composer.current?.focus(); composer.current?.setSelectionRange(draft.length, draft.length); prefill.current = null; }
  }, [Boolean(chat.detail)]);
  const last = chat.detail?.messages.at(-1);
  useEffect(() => { if (stick.current && scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight; }, [last?.content, last?.status, chat.detail?.messages.length]);
  async function renameConversation() {
    setRenaming(true); setTitleError(null);
    try { await unwrap(api.conversations[':id'].$patch({ param: { id }, json: { title } })); setRename(false); await chat.reload(); onChanged(); }
    catch (err) { setTitleError(errorMessage(err)); }
    finally { setRenaming(false); }
  }
  if (!chat.detail) return chat.error ? <LoadError error={chat.error} retry={() => void chat.reload()} /> : <p role="status">Loading conversation…</p>;
  const textDocuments = documents.data?.filter((doc) => doc.kind !== 'attachment') ?? [];
  const availableIds = new Set(textDocuments.map((doc) => doc.id));
  const sourceIds = selectedNotes.filter((noteId) => availableIds.has(noteId));
  const source = chat.detail.messages.findLast((message) => message.role === 'assistant')?.context;
  const busy = chat.sending || !!chat.generatingId;
  return <section className="conversation" aria-label="Conversation">
    <header className="conversation-header">
      {rename ? <form className="inline-form" onSubmit={(e) => { e.preventDefault(); void renameConversation(); }}>
        <input className="input" aria-label="Conversation title" autoFocus value={title} maxLength={200} required onChange={(e) => setTitle(e.target.value)} />
        <button className="button small" disabled={renaming || !title.trim()}>Save title</button><button className="button small" type="button" onClick={() => setRename(false)}>Cancel</button>
        {titleError && <p className="error" role="alert">{titleError}</p>}
      </form> : <div className="conversation-title"><h2>{chat.detail.title}</h2><button className="link-button" onClick={() => { setTitle(chat.detail!.title); setRename(true); }}>Rename</button></div>}
      <span className="chip source-label">Source: video transcript{source?.mode === 'retrieval' ? ' excerpts' : ''}{source?.documentIds.length ? ` + ${source.documentIds.length} ${source.documentIds.length === 1 ? 'note' : 'notes'}` : ''}</span>
    </header>
    <div ref={scroller} className="chat-messages" role="region" aria-label="Messages" tabIndex={0} onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60; }}>
      {!chat.detail.messages.length && <div className="chat-intro"><h3>Start with a question.</h3><p className="muted">Explore an idea, clarify a passage, or check your understanding. Answers link back to their sources.</p></div>}
      {chat.detail.messages.map((message, index, messages) => <article key={message.id} className={`chat-message ${message.role}`} aria-label={message.role === 'user' ? 'Your question' : 'Assistant answer'}>
        <span className="chat-avatar" aria-hidden="true">{message.role === 'user' ? 'U' : 'A'}</span><div className="chat-message-body">
          {message.role === 'user' ? <p className="question-text">{message.content}</p> : <>
            <Markdown>{message.status === 'generating' ? message.content : answerMarkdown(message, video.id)}</Markdown>
            {message.status === 'generating' ? <p className="muted" role="status">{message.content ? 'Writing…' : 'Preparing an answer…'} You can leave this view and return.</p> : <>
              {message.status !== 'complete' && <p className={message.status === 'failed' ? 'error' : 'muted'} role="status">{message.status === 'incomplete' ? 'Answer ended before completion. The partial answer is saved.' : message.status === 'interrupted' ? 'Interrupted when the server stopped. The partial answer is saved.' : 'The answer could not be finished. Your question and any partial answer are saved.'}</p>}
              {!message.citations.length && <p className="muted no-citations">No sources cited</p>}
              {!!message.content && <AnswerActions question={messages[index - 1]?.content ?? ''} message={message} videoId={video.id} documents={textDocuments} refresh={documents.reload} />}
            </>}
          </>}
        </div>
      </article>)}
    </div>
    <div className="chat-composer">
      {chat.error && <p className="error" role="alert">{chat.error}</p>}
      {health.error !== undefined && <LoadError error={health.error} retry={health.reload} />}
      {health.data && !health.data.aiConfigured && <p className="notice">AI is not configured. <Link to="/settings">Check Settings</Link> to enable chat.</p>}
      {video.transcriptStatus !== 'ready' && <p className="notice"><Link to={`/videos/${video.id}/watch`}>Add a transcript</Link> before asking a question.</p>}
      <form onSubmit={(e) => { e.preventDefault(); stick.current = true; const question = draft.trim(); void chat.ask(question, sourceIds, () => setDraft((current) => current.trim() === question ? '' : current)); }}>
        <label htmlFor="chat-question">Your question</label>
        <textarea id="chat-question" ref={composer} className="input" placeholder="Ask about this video…" rows={3} value={draft} maxLength={8000} onChange={(e) => setDraft(e.target.value)} />
        <div className="composer-controls"><details className="chat-source-picker"><summary className="button small">Sources{sourceIds.length ? ` + ${sourceIds.length} ${sourceIds.length === 1 ? 'note' : 'notes'}` : ''}</summary>
          <div className="source-options"><strong>Sources for your next question</strong><p className="muted">Transcript always included. Choose up to 4 notes; long notes are included as excerpts.</p>
            {documents.error !== undefined ? <LoadError error={documents.error} retry={documents.reload} /> : documents.loading ? <p role="status">Loading notes…</p> : <>
              <label><input type="checkbox" checked disabled /> Video transcript</label>
              {textDocuments.map((doc) => <label key={doc.id}><input type="checkbox" checked={sourceIds.includes(doc.id)} disabled={busy || (!sourceIds.includes(doc.id) && sourceIds.length >= 4)} onChange={(e) => setSelectedNotes((current) => e.target.checked ? [...current, doc.id] : current.filter((noteId) => noteId !== doc.id))} /> {doc.title}</label>)}
              {!textDocuments.length && <p className="muted">No text documents yet.</p>}
            </>}
          </div></details><span className="muted">{draft.length}/8,000</span>
          {busy ? <button type="button" className="button" disabled={!chat.generatingId} onClick={() => void chat.stop()}>Stop</button> : <button className="button primary" disabled={!draft.trim() || !health.data?.aiConfigured || video.transcriptStatus !== 'ready' || documents.loading || documents.error !== undefined}>Send</button>}
        </div>
      </form>
      <div className="prompt-chips">{chips.map((chip) => <button key={chip} className="button small" disabled={busy} onClick={() => { setDraft(`${chip}.`); composer.current?.focus(); }}>{chip}</button>)}</div>
    </div>
  </section>;
}

export default function ChatPage() {
  const { video } = useVideoContext();
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const conversations = useApi(() => unwrap(api.videos[':id'].conversations.$get({ param: { id: String(video.id) } })), [video.id]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  async function create() {
    setBusy(true); setError(null);
    try { const conversation = await unwrap(api.videos[':id'].conversations.$post({ param: { id: String(video.id) }, json: {} })); conversations.reload(); navigate(`/videos/${video.id}/chat/${conversation.id}`); }
    catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  }
  async function remove() {
    if (!conversationId) return;
    setBusy(true); setError(null);
    try { await send(api.conversations[':id'].$delete({ param: { id: conversationId } })); setDeleting(false); conversations.reload(); navigate(`/videos/${video.id}/chat`, { replace: true }); }
    catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  }
  useEffect(() => { setDeleting(false); }, [conversationId]);
  return <div className="chat-layout">
    <aside className="conversation-list" aria-label="Conversations">
      <div className="conversation-list-heading"><h2>Conversations</h2><button className="button small" disabled={busy} onClick={() => void create()}>New conversation</button></div>
      {conversations.error !== undefined && <LoadError error={conversations.error} retry={conversations.reload} />}
      {conversations.loading && !conversations.data && <p role="status">Loading conversations…</p>}
      <nav aria-label="Saved conversations">{conversations.data?.map((conversation) => <NavLink key={conversation.id} to={`/videos/${video.id}/chat/${conversation.id}`}><span>{conversation.title}</span><small className="muted">{new Date(conversation.updatedAt).toLocaleDateString()}</small></NavLink>)}</nav>
      {conversations.data?.length === 0 && <p className="muted">Your conversations will be kept here.</p>}
      {conversationId && <div className="conversation-delete">{deleting ? <><p>Delete this conversation and stop any answer in progress?</p><button className="button small danger" disabled={busy} onClick={() => void remove()}>Delete conversation</button> <button className="button small" onClick={() => setDeleting(false)}>Keep conversation</button></> : <button className="link-button" onClick={() => setDeleting(true)}>Delete conversation…</button>}</div>}
      {error && <p className="error" role="alert">{error}</p>}
    </aside>
    {conversationId ? <ConversationView key={`${video.id}-${conversationId}`} id={conversationId} onChanged={conversations.reload} /> : <section className="chat-empty"><h2>A closer look at the video.</h2><p className="muted">Start a conversation or reopen one from the list.</p><button className="button primary" disabled={busy} onClick={() => void create()}>New conversation</button></section>}
  </div>;
}
