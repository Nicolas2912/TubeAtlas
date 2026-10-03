import { normalizeForMatching, type Unit } from './units.ts';
import { formatTime, watchLink } from './time.ts';

export type Passage = { id: string; units: Unit[]; start: number | null; text: string };
export type TextMatch = { start: number; end: number };

/** Paragraphs group consecutive evidence units, respecting turns and user-supplied paragraphs. */
export function groupPassages(units: Unit[]): Passage[] {
  const passages: Passage[] = [];
  let current: Passage | undefined;
  let words = 0;
  for (const unit of units) {
    const previous = current?.units.at(-1);
    const paragraphBreak = unit.start === null && previous && !previous.segmentIds.includes(unit.segmentIds[0]!);
    if (!current || unit.turnStart || words >= 60 || paragraphBreak) {
      current = { id: unit.id, units: [unit], start: unit.start, text: unit.text };
      passages.push(current);
      words = 0;
    } else {
      current.text += ` ${unit.text}`;
      current.units.push(unit);
    }
    words += unit.text.split(/\s+/u).length;
  }
  return passages;
}

/** Last covering unit wins, including split captions. Nested overlaps can resume an earlier unit. */
export function activeUnit(units: Unit[], seconds: number): number {
  for (let index = units.length - 1; index >= 0; index--) {
    const unit = units[index]!;
    if (unit.start !== null && unit.end !== null && unit.start <= seconds && seconds < unit.end) return index;
  }
  return -1;
}

export function clampPlayback(seconds: number, duration: number | null): number {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return duration !== null && duration > 0 ? Math.min(safe, duration) : safe;
}

/** A valid link timestamp takes precedence over the saved position, including t=0. */
export function playbackTarget(value: string | null, saved: number, duration: number | null): number {
  const requested = value !== null && value.trim() !== '' ? Number(value) : NaN;
  return clampPlayback(Number.isFinite(requested) && requested >= 0 ? requested : saved, duration);
}

const fold = (text: string) => normalizeForMatching(text).normalize('NFD').replace(/\p{M}/gu, '').toLowerCase();

/** Cache a search index, mapping normalized text back to the original Unicode characters. */
export function prepareSearch(text: string) {
  let normalized = '';
  const starts: number[] = [];
  const ends: number[] = [];
  let offset = 0;
  for (const character of text) {
    const folded = fold(character);
    normalized += folded;
    for (let i = 0; i < folded.length; i++) { starts.push(offset); ends.push(offset + character.length); }
    if (!folded && ends.length) ends[ends.length - 1] = offset + character.length;
    offset += character.length;
  }
  return { normalized, starts, ends };
}

export function findTextMatches(index: ReturnType<typeof prepareSearch>, query: string): TextMatch[] {
  const needle = fold(query.trim());
  if (!needle) return [];
  const matches: TextMatch[] = [];
  let cursor = 0;
  while ((cursor = index.normalized.indexOf(needle, cursor)) !== -1) {
    matches.push({ start: index.starts[cursor]!, end: index.ends[cursor + needle.length - 1]! });
    cursor += needle.length;
  }
  return matches;
}

export function exportTranscript(passages: Passage[], videoId: number, timestamps: boolean): string {
  return passages.map((p) => timestamps && p.start !== null ? `[${formatTime(p.start)}](${watchLink(videoId, p.start)}) ${p.text}` : p.text).join('\n\n') + '\n';
}
