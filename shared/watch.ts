import type { Segment } from './api.ts';
import { formatTime, watchLink } from './time.ts';

export type Passage = { id: number; segmentIds: number[]; start: number | null; end: number | null; text: string };
export type TextMatch = { start: number; end: number };

/** Preserve caption timings; group at about 60 words or a pause longer than two seconds. */
export function groupPassages(segments: Segment[], timed: boolean): Passage[] {
  const passages: Passage[] = [];
  let current: Passage | undefined;
  let words = 0;
  for (const segment of segments) {
    if (!segment.text.trim()) continue;
    const gap = current?.end !== null && current?.end !== undefined && segment.start !== null ? segment.start - current.end : 0;
    if (!timed || segment.start === null || !current || words >= 60 || gap > 2) {
      current = { id: segment.id, segmentIds: [segment.id], start: timed ? segment.start : null, end: timed ? segment.end : null, text: segment.text };
      passages.push(current);
      words = 0;
    } else {
      current.text += ` ${segment.text}`;
      current.segmentIds.push(segment.id);
      if (segment.end !== null) current.end = Math.max(current.end ?? 0, segment.end);
    }
    words += segment.text.trim().split(/\s+/u).length;
  }
  return passages;
}

/** Latest starting passage wins when rolling captions overlap. Gaps have no active passage. */
export function activePassage(passages: Passage[], seconds: number): number {
  let low = 0;
  let high = passages.length - 1;
  let index = -1;
  while (low <= high) {
    const mid = (low + high) >>> 1;
    const start = passages[mid]!.start;
    if (start !== null && start <= seconds) { index = mid; low = mid + 1; }
    else high = mid - 1;
  }
  const end = passages[index]?.end;
  return end !== null && end !== undefined && seconds < end ? index : -1;
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

const fold = (text: string) => text.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase();

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
