import { useState, type FormEvent } from 'react';
import { api, errorMessage, send, unwrap, useApi, type VideoSummary } from '../api.ts';
import { ensureTopic } from '../features/library/ImportDialog.tsx';

/** "Topics" button with a popover of checkboxes; changes are saved immediately. */
export function TopicPicker({ video, onChanged }: { video: VideoSummary; onChanged: () => void }) {
  const topics = useApi(() => unwrap(api.topics.$get()), []);
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const id = `topics-${video.id}`;
  const current = new Set(video.topics.map((t) => t.id));

  async function save(topicIds: number[]) {
    setError(null);
    try {
      await send(api.videos[':id'].$patch({ param: { id: String(video.id) }, json: { topicIds } }));
      onChanged();
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function add(event: FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    try {
      const topic = await ensureTopic(name);
      setName('');
      topics.reload();
      await save([...current, topic.id]);
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  return (
    <>
      <button className="button small" popoverTarget={id}>
        Topics{video.topics.length ? ` (${video.topics.length})` : ''}
      </button>
      <div id={id} popover="auto" aria-label="Topics for this video">
        <ul className="checklist">
          {topics.data?.map((topic) => (
            <li key={topic.id}>
              <label>
                <input
                  type="checkbox"
                  checked={current.has(topic.id)}
                  onChange={(e) => save(e.target.checked ? [...current, topic.id] : [...current].filter((t) => t !== topic.id))}
                />
                {topic.name}
              </label>
            </li>
          ))}
          {topics.data?.length === 0 && <li className="muted">No topics yet.</li>}
        </ul>
        <form className="inline-form" onSubmit={add}>
          <input className="input" placeholder="New topic" aria-label="New topic name" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} />
          <button className="button small" type="submit">
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
