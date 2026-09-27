import { useEffect, useRef, useState, type FormEvent } from 'react';
import { useMatch, useNavigate } from 'react-router';
import { api, ApiError, errorMessage, unwrap, type Topic } from '../../api.ts';

const NEW_TOPIC = 'new';

/** Finds or creates a topic by name (case-insensitive, like the server). */
export async function ensureTopic(name: string): Promise<Topic> {
  try {
    return await unwrap(api.topics.$post({ json: { name } }));
  } catch (err) {
    if (!(err instanceof ApiError) || err.code !== 'TOPIC_EXISTS') throw err;
    const topics = await unwrap(api.topics.$get());
    const existing = topics.find((t) => t.name.toLowerCase() === name.trim().toLowerCase());
    if (!existing) throw err;
    return existing;
  }
}

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

  useEffect(() => {
    const el = dialog.current;
    if (!el) return;
    if (open && !el.open) {
      setUrl('');
      setNewTopic('');
      setError(null);
      setTopicChoice(topicRoute?.params.topicId ?? '');
      el.showModal();
      unwrap(api.topics.$get()).then(setTopics, () => setTopics([]));
    } else if (!open && el.open) {
      el.close();
    }
  }, [open, topicRoute?.params.topicId]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      let topicId: number | undefined;
      if (topicChoice === NEW_TOPIC) {
        if (!newTopic.trim()) throw new Error('Enter a name for the new topic.');
        topicId = (await ensureTopic(newTopic)).id;
      } else if (topicChoice) {
        topicId = Number(topicChoice);
      }
      const { video } = await unwrap(api.videos.import.$post({ json: { url, topicId } }));
      onClose();
      navigate(`/videos/${video.id}/watch`);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <dialog ref={dialog} onClose={onClose} aria-labelledby="import-title">
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
            autoFocus
            aria-describedby={error ? 'import-error' : undefined}
          />
        </div>
        <div className="field">
          <label htmlFor="import-topic">Topic (optional)</label>
          <select id="import-topic" className="input" value={topicChoice} onChange={(e) => setTopicChoice(e.target.value)}>
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
            <input id="import-new-topic" className="input" value={newTopic} maxLength={80} onChange={(e) => setNewTopic(e.target.value)} />
          </div>
        )}
        {error && (
          <p id="import-error" className="error" role="alert">
            {error}
          </p>
        )}
        <div className="form-actions">
          <button type="button" className="button" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="button primary" disabled={busy}>
            {busy ? 'Importing…' : 'Import'}
          </button>
        </div>
      </form>
    </dialog>
  );
}
