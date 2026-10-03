import { useState } from 'react';
import { Link } from 'react-router';
import { api, errorMessage, unwrap, type Document } from '../../api.ts';
import { answerMarkdown, clipText, savedAnswer, type Message } from '../../../../shared/chat.ts';

export function AnswerActions({ question, message, videoId, documents, refresh }: { question: string; message: Message; videoId: number; documents: Document[]; refresh: () => void }) {
  const [append, setAppend] = useState(false);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [saved, setSaved] = useState<Document>();
  const [error, setError] = useState<string | null>(null);
  async function save(id?: number) {
    setBusy(true); setError(null); setFeedback(null);
    try {
      const markdown = savedAnswer(question, message, videoId);
      const document = id === undefined ? await unwrap(api.videos[':id'].documents.$post({ param: { id: String(videoId) }, json: { title: clipText(question.trim(), 200) || 'Saved answer', kind: 'qa', markdown } })) :
        await unwrap(api.documents[':id'].$patch({ param: { id: String(id) }, json: { appendMarkdown: markdown } }));
      setSaved(document); setAppend(false); refresh();
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  }
  async function copy() {
    setError(null);
    try { await navigator.clipboard.writeText(answerMarkdown(message, videoId)); setFeedback('Copied.'); }
    catch { setError('Could not copy the answer. Try again or select its text.'); }
  }
  return <div className="answer-actions">
    <div className="chat-actions"><button className="link-button" disabled={busy} onClick={() => void save()}>Save as document</button>
      <button className="link-button" disabled={busy} aria-expanded={append} onClick={() => setAppend(!append)}>Add to document…</button>
      <button className="link-button" disabled={busy} onClick={() => void copy()}>Copy</button></div>
    {append && <section className="append-targets" aria-label="Choose a document to append to">
      <p className="muted">Choose a text document</p>
      {documents.length ? documents.map((doc) => <button key={doc.id} className="button small" disabled={busy} onClick={() => void save(doc.id)}>{doc.title}</button>) : <p>No text documents yet. Save this answer as a new document.</p>}
      <button className="button small" onClick={() => setAppend(false)}>Close</button>
    </section>}
    {busy && <p role="status">Saving answer…</p>}
    {saved && <p role="status">Saved. <Link to={`/videos/${videoId}/documents/${saved.id}`}>Open {saved.title}</Link></p>}
    {feedback && <p role="status">{feedback}</p>}{error && <p role="alert" className="error">{error}</p>}
  </div>;
}
