import { useState, type FormEvent } from 'react';
import { Link, NavLink, useNavigate, useParams } from 'react-router';
import { api, errorMessage, send, unwrap, useApi, type Topic } from '../../api.ts';
import { LoadError } from '../../components/LoadError.tsx';
import LibraryPage from '../library/LibraryPage.tsx';

function TopicActions({ topic, onChanged }: { topic: Topic; onChanged: () => void }) {
  const navigate = useNavigate();
  const [editing, setEditing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [name, setName] = useState(topic.name);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function rename(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (!name.trim()) throw new Error('Enter a topic name.');
      await unwrap(api.topics[':id'].$patch({ param: { id: String(topic.id) }, json: { name: name.trim() } }));
      setEditing(false);
      onChanged();
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  }

  async function remove() {
    setBusy(true);
    setError(null);
    try {
      await send(api.topics[':id'].$delete({ param: { id: String(topic.id) } }));
      navigate('/topics');
      onChanged();
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  }

  return (
    <div className="topic-actions">
      {editing ? <form className="inline-form" onSubmit={rename}>
        <input className="input" aria-label="Topic name" value={name} maxLength={80} required autoFocus disabled={busy} onChange={(e) => setName(e.target.value)} />
        <button className="button small" disabled={busy}>Save name</button>
        <button className="button small" type="button" disabled={busy} onClick={() => { setEditing(false); setError(null); }}>Cancel</button>
      </form> : <div className="video-actions">
        <button className="button small" onClick={() => { setName(topic.name); setEditing(true); setDeleting(false); setError(null); }}>Rename topic</button>
        <button className="button small danger" onClick={() => { setDeleting(true); setError(null); }}>Delete topic</button>
      </div>}
      {deleting && <div className="notice" role="alert">
        <p>Delete “{topic.name}”? Its videos will stay in your library.</p>
        <div className="video-actions">
          <button className="button small" disabled={busy} onClick={() => setDeleting(false)}>Keep topic</button>
          <button className="button small danger" disabled={busy} onClick={remove}>{busy ? 'Deleting…' : 'Confirm deletion'}</button>
        </div>
      </div>}
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}

export default function TopicsPage() {
  const { topicId } = useParams();
  const navigate = useNavigate();
  const topics = useApi(() => unwrap(api.topics.$get()), []);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const topic = topics.data?.find((t) => String(t.id) === topicId);

  async function create(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (!name.trim()) throw new Error('Enter a topic name.');
      const created = await unwrap(api.topics.$post({ json: { name: name.trim() } }));
      setName('');
      topics.reload();
      navigate(`/topics/${created.id}`);
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  }

  return (
    <div className="split">
      <section className="topic-sidebar" aria-label="Topic management">
        <h2>Your topics</h2>
        {topics.loading && !topics.data && <p role="status">Loading topics…</p>}
        {topics.error !== undefined && <LoadError error={topics.error} retry={topics.reload} />}
        <ul className="topic-list">
          <li><NavLink to="/topics" end>All videos</NavLink></li>
          {topics.data?.map((t) => <li key={t.id}><NavLink to={`/topics/${t.id}`}><span>{t.name}</span><span className="muted" aria-label={`${t.videoCount} ${t.videoCount === 1 ? 'video' : 'videos'}`}>{t.videoCount}</span></NavLink></li>)}
        </ul>
        {topics.data?.length === 0 && <p className="muted">Create a topic to organize your videos.</p>}
        <form className="form" onSubmit={create}>
          <div className="field"><label htmlFor="new-topic">New topic</label><input id="new-topic" className="input" value={name} maxLength={80} required disabled={busy} onChange={(e) => setName(e.target.value)} /></div>
          <button className="button" disabled={busy}>{busy ? 'Creating…' : 'Create topic'}</button>
        </form>
        {error && <p className="error" role="alert">{error}</p>}
      </section>
      <section className="topic-content">
        {!topicId && <LibraryPage title="Topics" />}
        {topic && <><TopicActions key={topic.id} topic={topic} onChanged={topics.reload} /><LibraryPage topicId={topicId} title={topic.name} /></>}
        {topicId && !topic && topics.loading && <p role="status">Loading topic…</p>}
        {topicId && !topic && !topics.loading && topics.data && <div className="empty"><h1>Topic not found</h1><p className="muted">This topic may have been deleted.</p><Link to="/topics" className="button">Back to topics</Link></div>}
      </section>
    </div>
  );
}
