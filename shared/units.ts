export type Unit = {
  id: string;
  segmentIds: number[];
  start: number | null;
  end: number | null;
  text: string;
  turnStart: boolean;
  annotations: string[];
};

export type EvidenceUnits = {
  unitsVersion: 1;
  units: Unit[];
  /** Removed text remains traceable to both immutable source captions. */
  deduplications: { segmentId: number; previousSegmentId: number; text: string }[];
};

export const UNITS_VERSION = 1;
export const estimateTokens = (text: string): number => Math.ceil(text.length / 3.5);

/** For matching only: display text retains its quote and dash characters. */
export const normalizeForMatching = (text: string): string => text.normalize('NFC')
  .replace(/[‘’‚‛]/gu, "'").replace(/[“”„‟]/gu, '"').replace(/[‐‑‒–—−]/gu, '-')
  .replace(/\s+/gu, ' ');
