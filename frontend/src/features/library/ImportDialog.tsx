import { useEffect, useRef, useState, type FormEvent } from 'react';
import { useMatch, useNavigate } from 'react-router';
import { api, errorMessage, unwrap, type Topic } from '../../api.ts';
import { trapDialogFocus } from '../../components/dialog.ts';
import { ensureTopic } from '../topics/api.ts';

const NEW_TOPIC = 'new';

export function ImportDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const navigate = useNavigate();
  const topicRoute = useMatch('/topics/:topicId');
  const [url, setUrl] = useState('');
  const [topics, setTopics] = useState<Topic[]>([]);
  const [topicChoice, setTopicChoice] = useState('');
  const [newTopic, setNewTopic] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [topicsLoading, setTopicsLoading] = useState(false);
  const [topicsError, setTopicsError] = useState<string | null>(null);
  const attempt = useRef(0);

  useEffect(() => {
    const el = dialog.current;
    if (!el) return;
    if (open && !el.open) {
      setUrl('');
      setNewTopic('');
      setError(null);
      setBusy(false);
      setTopics([]);
      setTopicsError(null);
      setTopicsLoading(true);
      setTopicChoice(topicRoute?.params.topicId ?? '');
      el.showModal();
      const current = ++attempt.current;
      unwrap(api.topics.$get()).then(
        (loaded) => { if (attempt.current === current) { setTopics(loaded); setTopicsLoading(false); } },
        (err) => { if (attempt.current === current) { setTopicsError(errorMessage(err)); setTopicsLoading(false); } },
      );
    } else if (!open && el.open) {
      attempt.current++;
      el.close();
    }
  }, [open, topicRoute?.params.topicId]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const current = attempt.current;
    try {
      let topicId: number | undefined;
      if (topicChoice === NEW_TOPIC) {
        if (!newTopic.trim()) throw new Error('Enter a name for the new topic.');
        topicId = (await ensureTopic(newTopic)).id;
      } else if (topicChoice) {
        topicId = Number(topicChoice);
      }
      const { video } = await unwrap(api.videos.import.$post({ json: { url, topicId } }));
      if (attempt.current !== current) return;
      onClose();
      navigate(`/videos/${video.id}/watch`);
    } catch (err) {
      if (attempt.current === current) setError(errorMessage(err));
    } finally {
      if (attempt.current === current) setBusy(false);
    }
  }

  return (
    <dialog ref={dialog} onKeyDown={trapDialogFocus} onClose={() => { attempt.current++; onClose(); }} aria-labelledby="import-title">
      <h2 id="import-title">Import video</h2>
      <form className="form" onSubmit={submit}>
        <div className="field">
          <label htmlFor="import-url">YouTube link</label>
          <input
            id="import-url"
            className="input"
            type="text"
            inputMode="url"
            placeholder="https://www.youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            required
            disabled={busy}
            autoFocus
            aria-describedby={error ? 'import-error' : undefined}
          />
        </div>
        <div className="field">
          <label htmlFor="import-topic">Topic (optional)</label>
          <select id="import-topic" className="input" value={topicChoice} disabled={busy || topicsLoading || topicsError !== null} onChange={(e) => setTopicChoice(e.target.value)}>
            <option value="">No topic</option>
            {topics.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
            <option value={NEW_TOPIC}>New topic…</option>
          </select>
        </div>
        {topicChoice === NEW_TOPIC && (
          <div className="field">
            <label htmlFor="import-new-topic">New topic name</label>
            <input id="import-new-topic" className="input" value={newTopic} maxLength={80} disabled={busy} required onChange={(e) => setNewTopic(e.target.value)} />
          </div>
        )}
        {topicsLoading && <p className="muted" role="status">Loading topics…</p>}
        {topicsError && <p className="error" role="alert">Could not load topics. {topicsError} Close and reopen this dialog to try again.</p>}
        {error && (
          <p id="import-error" className="error" role="alert">
            {error}
          </p>
        )}
        <div className="form-actions">
          <button type="button" className="button" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="button primary" disabled={busy || topicsLoading || (topicsError !== null && topicChoice !== '')}>
            {busy ? 'Importing…' : 'Import'}
          </button>
        </div>
      </form>
    </dialog>
  );
}
