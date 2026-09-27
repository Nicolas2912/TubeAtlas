import { unlinkSync } from 'node:fs';
import { join } from 'node:path';
import type { JobKind, JobStatus, TranscriptStatus } from '../../../shared/api.ts';
import { tx, type Db } from '../db.ts';
import { AppError } from '../errors.ts';

export type VideoSummary = {
  id: number;
  youtubeId: string;
  title: string;
  channel: string | null;
  durationSeconds: number | null;
  thumbnailUrl: string | null;
  playbackSeconds: number;
  transcriptStatus: TranscriptStatus;
  transcriptError: string | null;
  topics: { id: number; name: string }[];
  activeJob: { id: number; kind: JobKind; status: JobStatus; stage: string | null } | null;
  createdAt: string;
  updatedAt: string;
};

type VideoRow = {
  id: number;
  youtube_id: string;
  title: string;
  channel: string | null;
  duration_seconds: number | null;
  thumbnail_url: string | null;
  playback_seconds: number;
  transcript_status: TranscriptStatus;
  transcript_error: string | null;
  created_at: string;
  updated_at: string;
};

function summarize(db: Db, rows: VideoRow[]): VideoSummary[] {
  if (rows.length === 0) return [];
  const ids = rows.map((r) => r.id);
  const marks = ids.map(() => '?').join(',');
  const topics = db
    .prepare(`SELECT vt.video_id, t.id, t.name FROM video_topics vt JOIN topics t ON t.id = vt.topic_id WHERE vt.video_id IN (${marks}) ORDER BY t.name`)
    .all(...ids) as { video_id: number; id: number; name: string }[];
  const jobs = db
    .prepare(`SELECT id, kind, status, stage, video_id FROM jobs WHERE status IN ('queued','running') AND video_id IN (${marks}) ORDER BY id`)
    .all(...ids) as { id: number; kind: JobKind; status: JobStatus; stage: string | null; video_id: number }[];
  return rows.map((r) => {
    const job = jobs.find((j) => j.video_id === r.id);
    return {
      id: r.id,
      youtubeId: r.youtube_id,
      title: r.title,
      channel: r.channel,
      durationSeconds: r.duration_seconds,
      thumbnailUrl: r.thumbnail_url,
      playbackSeconds: r.playback_seconds,
      transcriptStatus: r.transcript_status,
      transcriptError: r.transcript_error,
      topics: topics.filter((t) => t.video_id === r.id).map(({ id, name }) => ({ id, name })),
      activeJob: job ? { id: job.id, kind: job.kind, status: job.status, stage: job.stage } : null,
      createdAt: r.created_at,
      updatedAt: r.updated_at,
    };
  });
}

export function listVideos(db: Db, topicId?: number): VideoSummary[] {
  const rows = (
    topicId === undefined
      ? db.prepare('SELECT * FROM videos ORDER BY created_at DESC, id DESC').all()
      : db
          .prepare('SELECT v.* FROM videos v JOIN video_topics vt ON vt.video_id = v.id WHERE vt.topic_id = ? ORDER BY v.created_at DESC, v.id DESC')
          .all(topicId)
  ) as VideoRow[];
  return summarize(db, rows);
}

export function getVideo(db: Db, id: number): VideoSummary {
  const row = db.prepare('SELECT * FROM videos WHERE id = ?').get(id) as VideoRow | undefined;
  if (!row) throw new AppError(404, 'NOT_FOUND', 'Video not found.');
  return summarize(db, [row])[0]!;
}

export function findVideoByYoutubeId(db: Db, youtubeId: string): VideoSummary | null {
  const row = db.prepare('SELECT * FROM videos WHERE youtube_id = ?').get(youtubeId) as VideoRow | undefined;
  return row ? summarize(db, [row])[0]! : null;
}

export function assertTopicsExist(db: Db, topicIds: number[]) {
  const unique = [...new Set(topicIds)];
  if (unique.length === 0) return;
  const { n } = db.prepare(`SELECT count(*) AS n FROM topics WHERE id IN (${unique.map(() => '?').join(',')})`).get(...unique) as { n: number };
  if (n !== unique.length) throw new AppError(400, 'UNKNOWN_TOPIC', 'One of the topics does not exist.');
}

export function updateVideo(db: Db, id: number, patch: { playbackSeconds?: number; topicIds?: number[] }): VideoSummary {
  getVideo(db, id);
  if (patch.topicIds) assertTopicsExist(db, patch.topicIds);
  tx(db, () => {
    if (patch.playbackSeconds !== undefined) {
      db.prepare('UPDATE videos SET playback_seconds = ? WHERE id = ?').run(patch.playbackSeconds, id);
    }
    if (patch.topicIds) {
      db.prepare('DELETE FROM video_topics WHERE video_id = ?').run(id);
      const add = db.prepare('INSERT INTO video_topics (video_id, topic_id) VALUES (?, ?)');
      for (const topicId of new Set(patch.topicIds)) add.run(id, topicId);
      db.prepare(`UPDATE videos SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?`).run(id);
    }
  });
  return getVideo(db, id);
}

/** Deletes the video with everything that belongs to it (via cascade), then its stored files. */
export function deleteVideo(db: Db, dataDir: string, id: number) {
  getVideo(db, id);
  const files = db.prepare('SELECT storage_name FROM assets WHERE video_id = ?').all(id) as { storage_name: string }[];
  db.prepare('DELETE FROM videos WHERE id = ?').run(id);
  for (const { storage_name } of files) {
    try {
      unlinkSync(join(dataDir, 'files', storage_name));
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') console.error(`Could not delete file ${storage_name}:`, err);
    }
  }
}
