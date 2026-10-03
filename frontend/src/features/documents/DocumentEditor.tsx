import { useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { api, errorMessage, send, unwrap, useApi, type Document, type VideoSummary } from '../../api.ts';
import { LoadError } from '../../components/LoadError.tsx';
import { Markdown } from '../../app/Markdown.tsx';
import { kindLabels } from '../../../../shared/documents.ts';
import { useDocumentSave } from './useDocumentSave.ts';
import { FormattingToolbar } from './FormattingToolbar.tsx';

function Editor({ document, video, onSaved, onDeleted }: { document: Document; video: VideoSummary; onSaved: (doc: Document) => void; onDeleted: () => void }) {
  const state = useDocumentSave(document, onSaved);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const title = useRef<HTMLInputElement>(null);
  const menu = useRef<HTMLDetailsElement>(null);
  const navigate = useNavigate();
  const [preview, setPreview] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileError, setFileError] = useState(false);
  const { draft } = state;
  const asset = document.asset;
  const contentUrl = asset ? `/api/assets/${asset.id}/content` : '';
  async function remove() {
    if (!window.confirm(`Delete “${draft.title}”?${asset ? ' Its file will also be deleted.' : ''}${state.dirty ? ' Unsaved changes will be discarded.' : ''}`)) return;
    setDeleting(true); setError(null);
    try { await send(api.documents[':id'].$delete({ param: { id: String(document.id) } })); state.discard(); onDeleted(); navigate(`/videos/${video.id}/documents`); }
    catch (err) { setError(errorMessage(err)); }
    finally { setDeleting(false); }
  }
  async function exportMarkdown() {
    if (await state.save()) { const link = window.document.createElement('a'); link.href = `/api/documents/${document.id}/export`; link.click(); }
  }
  return <section className="document-editor panel" aria-label="Document editor">
    <div className="editor-actions">
      <span role="status" className={state.error ? 'error' : 'muted'}>{kindLabels[draft.kind]} · {state.saving ? 'Saving…' : state.error ? 'Save failed' : state.dirty ? 'Unsaved changes' : 'Saved'}</span>
      {!asset && <div className="view-toggle" role="group" aria-label="Document view"><button className="button small" aria-pressed={!preview} onClick={() => setPreview(false)}>Edit</button><button className="button small" aria-pressed={preview} onClick={() => setPreview(true)}>Preview</button></div>}
      {asset ? <a className="button small" href={`${contentUrl}?download=1`}>Export</a> : <button className="button small" disabled={state.saving || deleting} onClick={() => void exportMarkdown()}>Export</button>}
      <details ref={menu} className="document-menu export-menu"><summary className="button small" aria-label="Document options">•••</summary><div className="export-options">
        <button className="link-button" onClick={() => { menu.current!.open = false; title.current?.focus(); title.current?.select(); }}>Rename</button>
        {!asset && <label className="field">Change kind<select className="input" aria-label="Document kind" value={draft.kind} onChange={(e) => state.edit({ kind: e.target.value as Document['kind'] })}>{Object.entries(kindLabels).filter(([kind]) => kind !== 'attachment').map(([kind, label]) => <option key={kind} value={kind}>{label}</option>)}</select></label>}
        <button className="link-button error" disabled={deleting || state.saving} onClick={() => void remove()}>{deleting ? 'Deleting…' : 'Delete'}</button>
      </div></details>
    </div>
    {state.error && <p className="error" role="alert">{state.error} <button className="link-button" onClick={() => void state.save()}>Retry save</button></p>}
    {error && <p className="error" role="alert">{error}</p>}
    {!asset && !preview && <FormattingToolbar textarea={textarea} text={draft.markdown!} onChange={(markdown) => state.edit({ markdown })} />}
    <input ref={title} className="document-title" aria-label="Document title" value={draft.title} maxLength={200} onChange={(e) => state.edit({ title: e.target.value })} />
    <div className="document-content">
      {asset ? <div className="attachment-preview">
        <p className="muted">{asset.originalName} · {(asset.sizeBytes / 1024).toFixed(1)} KB · File attachment</p>
        {asset.mediaType.startsWith('image/') && !fileError ? <img src={contentUrl} alt={draft.title} onError={() => setFileError(true)} /> : asset.mediaType === 'application/pdf' ? <iframe title={`PDF preview: ${draft.title}`} src={contentUrl} /> : <p>{fileError ? 'The image could not be previewed. Try downloading the file.' : 'This file is available for download. It cannot be edited or used as an AI text source.'}</p>}
        <a className="button" href={`${contentUrl}?download=1`}>Download original file</a>
      </div> : preview ? <Markdown>{draft.markdown || 'This note is empty.'}</Markdown> : <textarea ref={textarea} className="document-textarea" aria-label="Document Markdown" spellCheck value={draft.markdown!} onChange={(e) => state.edit({ markdown: e.target.value })} />}
    </div>
    <footer className="document-footer"><span>Linked to: <Link to={`/videos/${video.id}/watch`}>{video.title}</Link></span><span>Last edited {new Date(state.updatedAt).toLocaleString()}</span></footer>
  </section>;
}

export function DocumentEditor({ id, video, onSaved, onDeleted }: { id: string; video: VideoSummary; onSaved: (doc: Document) => void; onDeleted: () => void }) {
  const result = useApi(() => unwrap(api.documents[':id'].$get({ param: { id } })), [id]);
  if (!result.data) return result.error ? <LoadError error={result.error} retry={result.reload} /> : <p role="status">Loading document…</p>;
  if (result.data.videoId !== video.id) return <p className="notice error-box" role="alert">This document belongs to another video.</p>;
  return <Editor key={result.data.id} document={result.data} video={video} onSaved={onSaved} onDeleted={onDeleted} />;
}
