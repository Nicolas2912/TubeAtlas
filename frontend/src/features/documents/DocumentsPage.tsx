import { useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';
import { useVideoContext } from '../../app/VideoLayout.tsx';
import { api, errorMessage, unwrap, useApi } from '../../api.ts';
import { LoadError } from '../../components/LoadError.tsx';
import { kindLabels } from '../../../../shared/documents.ts';
import { DocumentEditor } from './DocumentEditor.tsx';

export default function DocumentsPage() {
  const { video } = useVideoContext();
  const { documentId } = useParams();
  const navigate = useNavigate();
  const files = useRef<HTMLInputElement>(null);
  const list = useApi(() => unwrap(api.videos[':id'].documents.$get({ param: { id: String(video.id) } })), [video.id]);
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function create() {
    setBusy(true); setError(null);
    try { const doc = await unwrap(api.videos[':id'].documents.$post({ param: { id: String(video.id) }, json: { title: 'Untitled note' } })); list.reload(); navigate(`/videos/${video.id}/documents/${doc.id}`); }
    catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  }
  async function upload(file: File | undefined) {
    if (!file) return;
    setBusy(true); setError(null);
    try {
      if (file.size > 25 * 1024 * 1024) throw new Error('Files must be 25 MB or smaller.');
      const doc = await unwrap(api.videos[':id'].assets.$post({ param: { id: String(video.id) }, form: { file } })); list.reload(); navigate(`/videos/${video.id}/documents/${doc.id}`);
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); if (files.current) files.current.value = ''; }
  }
  const documents = list.data ?? [];
  const visible = documents.filter((doc) => doc.title.toLocaleLowerCase().includes(query.toLocaleLowerCase()) && (filter === 'all' || (filter === 'files') === (doc.kind === 'attachment')));
  return <div className="documents-layout">
    <section className="documents-list panel" aria-labelledby="documents-title">
      <div className="documents-heading"><h2 id="documents-title">Video documents</h2><button className="button small" disabled={busy} onClick={() => void create()}>New</button><button className="button small" disabled={busy} onClick={() => files.current?.click()}>Upload</button></div>
      <input ref={files} className="file-input" aria-label="Upload document" type="file" accept=".pdf,.png,.jpg,.jpeg,.webp,.docx,.xlsx,.pptx,.md,.txt" onChange={(e) => void upload(e.target.files?.[0])} />
      <input className="input" type="search" aria-label="Find a document" placeholder="Find a document…" value={query} onChange={(e) => setQuery(e.target.value)} />
      <div className="document-filters" role="group" aria-label="Document filters">{['all', 'notes', 'files'].map((value) => <button key={value} className="button small" aria-pressed={filter === value} onClick={() => setFilter(value)}>{value[0]!.toUpperCase() + value.slice(1)}</button>)}</div>
      {busy && <p role="status">Saving document…</p>}{error && <p className="error" role="alert">{error}</p>}
      {list.error !== undefined && <LoadError error={list.error} retry={list.reload} />}
      <div className="document-rows">{list.loading && !list.data ? <p role="status">Loading documents…</p> : visible.length ? visible.map((doc) => <Link key={doc.id} to={`/videos/${video.id}/documents/${doc.id}`} className={`document-row${String(doc.id) === documentId ? ' active' : ''}`} aria-current={String(doc.id) === documentId ? 'page' : undefined}><span>{doc.title}</span><span className="chip">{kindLabels[doc.kind]}</span></Link>) : <p className="muted">{documents.length ? 'No documents match this filter.' : 'Create a note or upload a file for this video.'}</p>}</div>
      <p className="document-count muted">{documents.length} {documents.length === 1 ? 'document' : 'documents'}</p>
    </section>
    {documentId ? <DocumentEditor key={documentId} id={documentId} video={video} onSaved={list.reload} onDeleted={list.reload} /> : <section className="panel document-empty"><h2>Your notes, beside the source</h2><p className="muted">Choose a document, start a note, or upload a file.</p><p className="muted">PDF and images preview here. Office files download in their original format; Markdown and text files become editable notes.</p></section>}
  </div>;
}
