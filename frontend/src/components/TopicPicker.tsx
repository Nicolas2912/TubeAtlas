import { useEffect, useState, type FormEvent } from 'react';
import { api, errorMessage, unwrap, useApi, type VideoSummary } from '../api.ts';
import { ensureTopic } from '../features/topics/api.ts';
import { LoadError } from './LoadError.tsx';

/** "Topics" button with a popover of checkboxes; changes are saved immediately. */
export function TopicPicker({ video, onChanged }: { video: VideoSummary; onChanged: () => void }) {
  const topics = useApi(() => unwrap(api.topics.$get()), []);
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [assigned, setAssigned] = useState(video.topics);
  useEffect(() => setAssigned(video.topics), [video.topics]);
  const id = `topics-${video.id}`;
  const current = new Set(assigned.map((t) => t.id));

  async function save(topicIds: number[]) {
    setError(null);
    try {
      const updated = await unwrap(api.videos[':id'].$patch({ param: { id: String(video.id) }, json: { topicIds } }));
      setAssigned(updated.topics);
      onChanged();
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function toggle(topicIds: number[]) {
    setBusy(true);
    try { await save(topicIds); }
    finally { setBusy(false); }
  }

  async function add(event: FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const topic = await ensureTopic(name);
      setName('');
      topics.reload();
      await save([...current, topic.id]);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button className="button small" popoverTarget={id}>
        Topics{assigned.length ? ` (${assigned.length})` : ''}
      </button>
      <div id={id} popover="auto" aria-label="Topics for this video">
        {topics.loading && !topics.data && <p role="status">Loading topics…</p>}
        {topics.error !== undefined && <LoadError error={topics.error} retry={topics.reload} />}
        <ul className="checklist">
          {topics.data?.map((topic) => (
            <li key={topic.id}>
              <label>
                <input
                  type="checkbox"
                  checked={current.has(topic.id)}
                  disabled={busy}
                  onChange={(e) => toggle(e.target.checked ? [...current, topic.id] : [...current].filter((t) => t !== topic.id))}
                />
                {topic.name}
              </label>
            </li>
          ))}
          {topics.data?.length === 0 && <li className="muted">No topics yet.</li>}
        </ul>
        <form className="inline-form" onSubmit={add}>
          <input className="input" placeholder="New topic" aria-label="New topic name" value={name} maxLength={80} disabled={busy} onChange={(e) => setName(e.target.value)} />
          <button className="button small" type="submit" disabled={busy || !name.trim()}>
            Add
          </button>
        </form>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </div>
    </>
  );
}
