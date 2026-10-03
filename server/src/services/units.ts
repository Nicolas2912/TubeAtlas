import type { Segment } from '../../../shared/api.ts';
import { normalizeForMatching, UNITS_VERSION, type EvidenceUnits, type Unit } from '../../../shared/units.ts';
import { AppError } from '../errors.ts';

/** Raw captions stay immutable. IDs and all derived boundaries depend only on these segments. */
export function buildUnits(segments: Segment[]): EvidenceUnits {
  const units: Unit[] = [];
  const deduplications: EvidenceUnits['deduplications'] = [];
  let current: Unit = emptyUnit();
  let words = 0;
  let previous: { segment: Segment; text: string } | undefined;

  function emptyUnit(): Unit {
    return { id: '', segmentIds: [], start: null, end: null, text: '', turnStart: false, annotations: [] };
  }
  function include(segment: Segment) {
    if (!current.segmentIds.length) current.start = segment.start;
    if (!current.segmentIds.includes(segment.id)) current.segmentIds.push(segment.id);
    if (segment.end !== null) current.end = Math.max(current.end ?? 0, segment.end);
  }
  function close() {
    if (!current.text) return;
    current.id = `u${String(units.length + 1).padStart(3, '0')}`;
    units.push(current);
    current = emptyUnit();
    words = 0;
  }

  const ids = new Set<number>();
  for (const segment of segments) {
    const { start, end } = segment;
    if ((start === null) !== (end === null) || (start !== null && end !== null &&
      (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start)) ||
      !Number.isInteger(segment.id) || ids.has(segment.id) ||
      (segments.length > 0 && (start === null) !== (segments[0]!.start === null))) {
      throw new AppError(400, 'INVALID_TRANSCRIPT', 'Transcript captions need unique IDs and valid, consistently timed or untimed ranges.');
    }
    ids.add(segment.id);
    const text = segment.text.normalize('NFC').replace(/\s+/gu, ' ').trim();
    let derived = text;
    // Require a multiword overlap in time: a repeated single word or an untimed paragraph is not
    // evidence of rolling captions. Never match across a speaker marker or a non-speech tag.
    if (previous && start !== null && previous.segment.end !== null && start < previous.segment.end) {
      const before = previous.text.split('>>').at(-1)!.split(' ');
      const after = text.split(' ');
      for (let count = Math.min(before.length, after.length); count >= 2; count--) {
        const prefix = after.slice(0, count).join(' ');
        if (/>>|\[|\]/u.test(prefix)) continue;
        if (normalizeForMatching(before.slice(-count).join(' ')) === normalizeForMatching(prefix)) {
          derived = after.slice(count).join(' ');
          deduplications.push({ segmentId: segment.id, previousSegmentId: previous.segment.id, text: prefix });
          break;
        }
      }
    }
    previous = { segment, text };
    for (const part of derived.split(/(>>|\[[^\]]*\])/u)) {
      if (part === '>>') {
        close();
        current.turnStart = true;
      } else if (part.startsWith('[') && part.endsWith(']')) {
        include(segment);
        current.annotations.push(part);
      } else {
        for (const word of part.trim().split(/\s+/u).filter(Boolean)) {
          include(segment);
          current.text += `${current.text ? ' ' : ''}${word}`;
          words++;
          if (words >= 15 && /[.?!…]["'”’»)\]}]*$/u.test(word)) close();
        }
      }
    }
    // Untimed source segments are paragraphs supplied by the user, rather than caption fragments.
    if (words >= 45 || start === null) close();
  }
  close();
  if (current.annotations.length && units.length) {
    const last = units.at(-1)!;
    last.annotations.push(...current.annotations);
    last.segmentIds = [...new Set([...last.segmentIds, ...current.segmentIds])];
    if (current.end !== null) last.end = Math.max(last.end ?? 0, current.end);
  }
  return { unitsVersion: UNITS_VERSION, units, deduplications };
}
