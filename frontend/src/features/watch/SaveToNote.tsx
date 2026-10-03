import { useEffect, useRef, useState, type RefObject } from 'react';
import { Link } from 'react-router';
import { api, errorMessage, unwrap, useApi, type Document } from '../../api.ts';
import { LoadError } from '../../components/LoadError.tsx';
import { transcriptQuote } from '../../../../shared/documents.ts';
import type { Unit } from '../../../../shared/units.ts';

type Quote = { text: string; start: number | null; left: number; top: number };

export function SaveToNote({ scroller, units, videoId }: { scroller: RefObject<HTMLDivElement | null>; units: Unit[]; videoId: number }) {
  const [quote, setQuote] = useState<Quote | null>(null);
  const [open, setOpen] = useState(false);
  const frozen = useRef(false);
  const panel = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<Document | null>(null);
  const notes = useApi(async () => open ? unwrap(api.videos[':id'].documents.$get({ param: { id: String(videoId) } })) : [], [videoId, open]);

  useEffect(() => {
    function selection() {
      if (frozen.current) return;
      const selected = window.getSelection();
      const root = scroller.current;
      if (!selected?.rangeCount || selected.isCollapsed || !root || !root.contains(selected.anchorNode) || !root.contains(selected.focusNode)) { setQuote(null); return; }
      const range = selected.getRangeAt(0);
      const pieces: string[] = [];
      let start: number | null = null;
      for (const paragraph of root.querySelectorAll<HTMLElement>('.passage p')) {
        if (!range.intersectsNode(paragraph)) continue;
        const part = window.document.createRange();
        part.selectNodeContents(paragraph);
        if (paragraph.contains(range.startContainer)) part.setStart(range.startContainer, range.startOffset);
        if (paragraph.contains(range.endContainer)) part.setEnd(range.endContainer, range.endOffset);
        const text = part.toString().trim();
        if (!text) continue;
        if (!pieces.length) {
          for (const unit of paragraph.querySelectorAll<HTMLElement>('[data-unit-index]')) {
            if (!range.intersectsNode(unit)) continue;
            const selectedUnit = window.document.createRange();
            selectedUnit.selectNodeContents(unit);
            if (unit.contains(range.startContainer)) selectedUnit.setStart(range.startContainer, range.startOffset);
            if (unit.contains(range.endContainer)) selectedUnit.setEnd(range.endContainer, range.endOffset);
            if (selectedUnit.toString().trim()) { start = units[Number(unit.dataset.unitIndex)]!.start; break; }
          }
        }
        pieces.push(text);
      }
      if (!pieces.length) { setQuote(null); return; }
      const rect = range.getBoundingClientRect();
      setQuote({ text: pieces.join('\n\n'), start, left: Math.max(16, Math.min(rect.left, window.innerWidth - 336)), top: Math.max(16, Math.min(rect.bottom + 8, window.innerHeight - 310)) });
      setError(null); setSaved(null);
    }
    window.document.addEventListener('selectionchange', selection);
    return () => window.document.removeEventListener('selectionchange', selection);
  }, [scroller, units]);

  function close(restoreFocus = true) { frozen.current = false; setOpen(false); setQuote(null); if (restoreFocus) scroller.current?.focus({ preventScroll: true }); }
  useEffect(() => { if (open) panel.current?.querySelector<HTMLButtonElement>('button')?.focus(); }, [open]);
  useEffect(() => { if (saved) panel.current?.querySelector<HTMLAnchorElement>('a')?.focus(); }, [saved]);
  useEffect(() => {
    if (!quote) return;
    function outside(event: PointerEvent) { if (!panel.current?.contains(event.target as Node)) close(false); }
    function escape(event: KeyboardEvent) { if (event.key === 'Escape') close(); }
    window.document.addEventListener('pointerdown', outside);
    window.document.addEventListener('keydown', escape);
    return () => { window.document.removeEventListener('pointerdown', outside); window.document.removeEventListener('keydown', escape); };
  }, [quote]);

  async function append(id?: number) {
    if (!quote) return;
    setBusy(true); setError(null);
    const markdown = transcriptQuote(quote.text, videoId, quote.start);
    try {
      const doc = id === undefined ? await unwrap(api.videos[':id'].documents.$post({ param: { id: String(videoId) }, json: { title: 'Transcript notes', markdown } })) :
        await unwrap(api.documents[':id'].$patch({ param: { id: String(id) }, json: { appendMarkdown: markdown } }));
      setSaved(doc);
      window.getSelection()?.removeAllRanges();
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  }
  if (!quote) return null;
  return <div ref={panel} className="selection-note" style={{ left: quote.left, top: quote.top }}>
    {!open ? <button className="button" onMouseDown={(e) => e.preventDefault()} onClick={() => { frozen.current = true; setOpen(true); }}>Save to note</button> : <section aria-label="Save selected transcript to note">
      <div className="selection-heading"><strong>Save to note</strong><button className="button small" disabled={busy} onClick={() => close()}>Close</button></div>
      {saved ? <p role="status">Saved. <Link to={`/videos/${videoId}/documents/${saved.id}`}>Open {saved.title}</Link></p> : <>
        <p className="muted">{quote.start === null ? 'Selected text · no timestamp' : 'Selected text with a source link'}</p>
        {notes.error !== undefined ? <LoadError error={notes.error} retry={notes.reload} /> : notes.loading ? <p role="status">Loading notes…</p> : <div className="note-targets">
          <button className="button" disabled={busy} onClick={() => void append()}>New note</button>
          {notes.data?.filter((doc) => doc.kind !== 'attachment').map((doc) => <button key={doc.id} className="button" disabled={busy} onClick={() => void append(doc.id)}>{doc.title}</button>)}
        </div>}
        {busy && <p role="status">Saving quote…</p>}{error && <p className="error" role="alert">{error}</p>}
      </>}
    </section>}
  </div>;
}
