import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { activeUnit, exportTranscript, findTextMatches, groupPassages, prepareSearch, type TextMatch } from '../../../../shared/watch.ts';
import { formatTime } from '../../../../shared/time.ts';
import type { Transcript, VideoSummary } from '../../api.ts';
import { SaveToNote } from './SaveToNote.tsx';

function highlighted(text: string, matches: TextMatch[], selected: TextMatch | undefined): ReactNode[] {
  const parts: ReactNode[] = [];
  let cursor = 0;
  for (const match of matches) {
    parts.push(text.slice(cursor, match.start), <mark key={match.start} className={match === selected ? 'selected-match' : undefined}>{text.slice(match.start, match.end)}</mark>);
    cursor = match.end;
  }
  parts.push(text.slice(cursor));
  return parts;
}

export function TranscriptPanel({ transcript, video, time, ready, seek }: { transcript: Transcript; video: VideoSummary; time: number; ready: boolean; seek: (seconds: number) => void }) {
  const units = transcript.units;
  const passages = useMemo(() => groupPassages(units), [units]);
  const unitIndexes = useMemo(() => new Map(units.map((unit, index) => [unit.id, index])), [units]);
  const indexes = useMemo(() => units.map((unit) => prepareSearch(unit.text)), [units]);
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [follow, setFollow] = useState(true);
  const scroller = useRef<HTMLDivElement>(null);
  const manualScroll = useRef(0);
  const matches = useMemo(() => indexes.map((index) => findTextMatches(index, query)), [indexes, query]);
  const results = useMemo(() => matches.flatMap((list, unitIndex) => list.map((match) => ({ unitIndex, match }))), [matches]);
  const selected = results[selectedIndex];
  const active = transcript.timed ? activeUnit(units, time) : -1;
  const searching = query.trim().length > 0;

  function center(index: number) {
    const list = scroller.current;
    const element = list?.querySelector<HTMLElement>(`[data-unit-index="${index}"]`);
    if (!list || !element) return;
    const rect = element.getBoundingClientRect();
    list.scrollTop += rect.top - list.getBoundingClientRect().top - (list.clientHeight - rect.height) / 2;
  }

  useEffect(() => {
    if (follow && !searching && active >= 0 && Date.now() - manualScroll.current >= 5000) center(active);
  }, [active, follow, searching]);
  useEffect(() => { if (selected) center(selected.unitIndex); }, [selected]);

  function moveMatch(delta: number) { setSelectedIndex((index) => (index + delta + results.length) % results.length); }

  function download(timestamps: boolean) {
    const blob = new Blob([exportTranscript(passages, video.id, timestamps)], { type: timestamps ? 'text/markdown;charset=utf-8' : 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${video.youtubeId}-transcript.${timestamps ? 'md' : 'txt'}`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const rows = useMemo(() => passages.map((passage, index) => {
    const visible = passage.units.filter((unit) => !searching || matches[unitIndexes.get(unit.id)!]!.length);
    if (!visible.length) return null;
    const start = visible[0]!.start;
    return <div key={passage.id} data-passage-index={index} className="passage">
      {start !== null && <button className="timestamp" disabled={!ready} aria-label={`Seek to ${formatTime(start)}`} onClick={() => { manualScroll.current = 0; seek(start); }}>{formatTime(start)}</button>}
      <p>{visible.map((unit, position) => {
        const unitIndex = unitIndexes.get(unit.id)!;
        return <span key={unit.id}>{position > 0 && ' '}<span data-unit-id={unit.id} data-unit-index={unitIndex} data-start={unit.start ?? undefined}
          className={`evidence-unit${unitIndex === active ? ' active-unit' : ''}`} aria-current={unitIndex === active ? 'true' : undefined}>
          {highlighted(unit.text, matches[unitIndex]!, selected?.unitIndex === unitIndex ? selected.match : undefined)}
        </span></span>;
      })}</p>
    </div>;
  }), [passages, unitIndexes, searching, matches, selected, active, ready, seek]);

  return <section className="transcript-panel panel" aria-labelledby="transcript-title">
    <div className="transcript-heading">
      <h2 id="transcript-title">Transcript</h2>
      <label className="follow-toggle"><input type="checkbox" checked={follow && transcript.timed} disabled={!transcript.timed} onChange={(e) => setFollow(e.target.checked)} /> Follow playback</label>
      <details className="export-menu"><summary className="button small">Export</summary><div className="export-options">
        <button className="link-button" onClick={() => download(false)}>Transcript as text (.txt)</button>
        <button className="link-button" disabled={!transcript.timed} onClick={() => download(true)}>With timestamps (.md)</button>
      </div></details>
    </div>
    {!transcript.timed && <p className="muted untimed-notice">This transcript has no timestamps. Seeking and follow playback are unavailable.</p>}
    <input className="input" type="search" aria-label="Search transcript" placeholder="Search transcript…" value={query} onChange={(e) => { setQuery(e.target.value); setSelectedIndex(0); }} />
    {searching && <div className="match-navigation">
      <span role="status">{results.length ? `${selectedIndex + 1} of ${results.length}` : 'No matches'}</span>
      <button className="button small" aria-label="Previous match" disabled={!results.length} onClick={() => moveMatch(-1)}>Previous</button>
      <button className="button small" aria-label="Next match" disabled={!results.length} onClick={() => moveMatch(1)}>Next</button>
    </div>}
    <div ref={scroller} className={`transcript-scroll${transcript.timed ? '' : ' untimed'}`} tabIndex={0} role="region" aria-label="Transcript passages"
      onWheel={() => { manualScroll.current = Date.now(); }} onPointerDown={() => { manualScroll.current = Date.now(); }}
      onKeyDown={(e) => { if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(e.key)) manualScroll.current = Date.now(); }}>
      {rows}
      {searching && !results.length && <p className="muted">Try another word or clear the search to read the whole transcript.</p>}
    </div>
    <SaveToNote scroller={scroller} units={units} videoId={video.id} />
  </section>;
}
