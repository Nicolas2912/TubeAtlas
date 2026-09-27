import { createHash } from 'node:crypto';
import type { Segment } from '../../../shared/api.ts';
import { tx, type Db } from '../db.ts';
import { AppError } from '../errors.ts';

export type TranscriptSource = 'youtube' | 'upload_timed' | 'paste_text';

export type Transcript = {
  transcriptId: number;
  revision: number;
  language: string | null;
  source: TranscriptSource;
  timed: boolean;
  segments: Segment[];
};

/** Stores a new current revision (the previous one stays for old citations) and marks the video ready. */
export function saveTranscript(
  db: Db,
  videoId: number,
  input: { source: TranscriptSource; language: string | null; timed: boolean; segments: Segment[] },
): number {
  const segmentsJson = JSON.stringify(input.segments);
  const sha256 = createHash('sha256').update(segmentsJson).digest('hex');
  const plainText = input.segments.map((s) => s.text).join(' ');
  return tx(db, () => {
    const { next } = db.prepare('SELECT coalesce(max(revision), 0) + 1 AS next FROM transcripts WHERE video_id = ?').get(videoId) as {
      next: number;
    };
    db.prepare('UPDATE transcripts SET is_current = 0 WHERE video_id = ? AND is_current = 1').run(videoId);
    const id = Number(
      db
        .prepare(
          `INSERT INTO transcripts (video_id, revision, is_current, source, language, timed, sha256, segments_json, plain_text)
           VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)`,
        )
        .run(videoId, next, input.source, input.language, input.timed ? 1 : 0, sha256, segmentsJson, plainText).lastInsertRowid,
    );
    db.prepare(`UPDATE videos SET transcript_status = 'ready', transcript_error = NULL WHERE id = ?`).run(videoId);
    return id;
  });
}

export function getCurrentTranscript(db: Db, videoId: number): Transcript | null {
  const row = db
    .prepare('SELECT id, revision, language, source, timed, segments_json FROM transcripts WHERE video_id = ? AND is_current = 1')
    .get(videoId) as
    | { id: number; revision: number; language: string | null; source: TranscriptSource; timed: number; segments_json: string }
    | undefined;
  if (!row) return null;
  return {
    transcriptId: row.id,
    revision: row.revision,
    language: row.language,
    source: row.source,
    timed: row.timed === 1,
    segments: JSON.parse(row.segments_json) as Segment[],
  };
}

/** Pasted text: one untimed segment per paragraph (blank-line separated; if there are none, per line). */
export function segmentsFromText(content: string): Segment[] {
  const normalized = content.replace(/\r\n?/g, '\n');
  const blocks = /\n\s*\n/.test(normalized) ? normalized.split(/\n\s*\n/) : normalized.split('\n');
  return blocks
    .map((block) => block.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .map((text, id) => ({ id, start: null, end: null, text }));
}

const CUE_TIME = /((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})/;

/** "01:02:03.500", "02:03,5", or "2:03.500" → seconds. */
function cueSeconds(value: string): number {
  const [clock = '', fraction = '0'] = value.replace(',', '.').split('.');
  const parts = clock.split(':').map(Number);
  const [h = 0, m = 0, s = 0] = parts.length === 3 ? parts : [0, ...parts];
  return Math.round((h * 3600 + m * 60 + s + Number(`0.${fraction}`)) * 1000) / 1000;
}

/** WebVTT and SRT: timed cues; tags, cue settings, and empty cues are dropped. */
export function segmentsFromCues(content: string): Segment[] {
  const segments: Segment[] = [];
  for (const block of content.replace(/^﻿/, '').replace(/\r\n?/g, '\n').split(/\n\s*\n/)) {
    const lines = block.split('\n');
    const timeLine = lines.findIndex((line) => CUE_TIME.test(line));
    if (timeLine === -1) continue; // WEBVTT header, NOTE, STYLE, REGION, or junk
    const [, from = '', to = ''] = CUE_TIME.exec(lines[timeLine]!)!;
    const text = lines
      .slice(timeLine + 1)
      .join(' ')
      .replace(/<[^>]*>/g, '')
      .replace(/&nbsp;/g, ' ')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&amp;/g, '&')
      .replace(/\s+/g, ' ')
      .trim();
    if (!text) continue;
    segments.push({ id: segments.length, start: cueSeconds(from), end: cueSeconds(to), text });
  }
  return segments;
}

export function saveManualTranscript(db: Db, videoId: number, body: { format: 'text' | 'vtt' | 'srt'; content: string; language?: string }) {
  const timed = body.format !== 'text';
  const segments = timed ? segmentsFromCues(body.content) : segmentsFromText(body.content);
  if (segments.length === 0) {
    throw new AppError(400, 'NO_TRANSCRIPT_TEXT', timed ? 'No captions found in that file.' : 'The transcript is empty.');
  }
  return saveTranscript(db, videoId, { source: timed ? 'upload_timed' : 'paste_text', language: body.language ?? null, timed, segments });
}
