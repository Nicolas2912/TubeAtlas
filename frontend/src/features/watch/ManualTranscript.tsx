import { useState, type FormEvent } from 'react';
import { MAX_MANUAL_TRANSCRIPT_BYTES } from '../../../../shared/limits.ts';
import { api, errorMessage, unwrap } from '../../api.ts';

export function ManualTranscript({ videoId, onSaved }: { videoId: number; onSaved: () => void }) {
  const [content, setContent] = useState('');
  const [format, setFormat] = useState<'text' | 'vtt' | 'srt'>('text');
  const [language, setLanguage] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload(file: File | undefined) {
    if (!file) return;
    const extension = file.name.split('.').pop()?.toLowerCase();
    if (extension !== 'vtt' && extension !== 'srt') { setError('Choose a .vtt or .srt caption file.'); return; }
    if (file.size > MAX_MANUAL_TRANSCRIPT_BYTES) { setError('The caption file must be 5 MB or smaller.'); return; }
    setBusy(true);
    setError(null);
    try { setContent(await file.text()); setFormat(extension); }
    catch { setError('Could not read that file.'); }
    finally { setBusy(false); }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (!content.trim()) throw new Error('Paste transcript text or choose a caption file.');
      if (new TextEncoder().encode(content).byteLength > MAX_MANUAL_TRANSCRIPT_BYTES) throw new Error('The transcript must be 5 MB or smaller.');
      await unwrap(api.videos[':id'].transcript.$post({ param: { id: String(videoId) }, json: { content, format, ...(language.trim() ? { language: language.trim() } : {}) } }));
      onSaved();
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  }

  return <section className="manual-transcript" aria-labelledby="manual-title">
    <h3 id="manual-title">Add transcript</h3>
    <p className="muted">Paste text, or upload timed captions. Plain text stays readable without timestamps.</p>
    <form className="form" onSubmit={submit}>
      <div className="field"><label htmlFor="caption-file">Caption file (.vtt or .srt, up to 5 MB)</label><input id="caption-file" type="file" accept=".vtt,.srt" disabled={busy} onChange={(e) => void upload(e.target.files?.[0])} /></div>
      <div className="field"><label htmlFor="transcript-format">Format</label><select id="transcript-format" className="input" value={format} disabled={busy} onChange={(e) => setFormat(e.target.value as typeof format)}><option value="text">Plain text</option><option value="vtt">WebVTT</option><option value="srt">SRT</option></select></div>
      <div className="field"><label htmlFor="manual-content">Transcript text or captions</label><textarea id="manual-content" className="input" rows={8} value={content} disabled={busy} onChange={(e) => setContent(e.target.value)} required /></div>
      <div className="field"><label htmlFor="transcript-language">Language (optional, e.g. en or de)</label><input id="transcript-language" className="input" value={language} minLength={2} maxLength={20} disabled={busy} onChange={(e) => setLanguage(e.target.value)} /></div>
      {error && <p className="error" role="alert">{error}</p>}
      <button className="button primary" disabled={busy}>{busy ? 'Saving transcript…' : 'Save transcript'}</button>
    </form>
  </section>;
}
