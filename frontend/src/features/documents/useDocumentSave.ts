import { useCallback, useEffect, useRef, useState } from 'react';
import { useBeforeUnload, useBlocker } from 'react-router';
import { api, errorMessage, unwrap, type Document } from '../../api.ts';

type Draft = Pick<Document, 'title' | 'kind' | 'markdown'>;
const same = (a: Draft, b: Draft) => a.title === b.title && a.kind === b.kind && a.markdown === b.markdown;

/** One save at a time; changes made during a request follow immediately after it. */
export function useDocumentSave(document: Document, onSaved: (doc: Document) => void) {
  const [draft, setDraft] = useState<Draft>({ title: document.title, kind: document.kind, markdown: document.markdown });
  const current = useRef(draft);
  const saved = useRef(draft);
  const flight = useRef<Promise<boolean> | null>(null);
  const alive = useRef(true);
  const allowLeave = useRef(false);
  const callback = useRef(onSaved);
  callback.current = onSaved;
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState(document.updatedAt);
  const dirty = !same(draft, saved.current);

  const save = useCallback((): Promise<boolean> => {
    if (flight.current) return flight.current;
    const run = async () => {
      setSaving(true);
      setError(null);
      while (!same(current.current, saved.current)) {
        const snapshot = { ...current.current };
        try {
          const doc = await unwrap(api.documents[':id'].$patch({ param: { id: String(document.id) }, json: snapshot.kind === 'attachment' ? { title: snapshot.title } : { title: snapshot.title, kind: snapshot.kind, markdown: snapshot.markdown! } }));
          saved.current = snapshot;
          if (alive.current) { setUpdatedAt(doc.updatedAt); callback.current(doc); }
        } catch (err) { if (alive.current) setError(errorMessage(err)); return false; }
        if (!alive.current) return true;
      }
      return true;
    };
    flight.current = run().finally(() => { flight.current = null; if (alive.current) setSaving(false); });
    return flight.current;
  }, [document.id]);

  useEffect(() => {
    if (!same(draft, saved.current)) { const timer = setTimeout(() => void save(), 800); return () => clearTimeout(timer); }
  }, [draft, save]);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useBeforeUnload(useCallback((event) => { if (!same(current.current, saved.current) && !allowLeave.current) { event.preventDefault(); event.returnValue = ''; } }, []));
  const blocker = useBlocker(() => !allowLeave.current && !same(current.current, saved.current));
  useEffect(() => {
    if (blocker.state !== 'blocked') return;
    if (window.confirm('This document has unsaved changes. Leave and discard those changes?')) blocker.proceed();
    else blocker.reset();
  }, [blocker]);

  function edit(change: Partial<Draft>) { current.current = { ...current.current, ...change }; setDraft(current.current); }
  return { draft, edit, save, dirty, saving, error, updatedAt, discard: () => { allowLeave.current = true; } };
}
