import type { Db } from '../db.ts';
import { tx } from '../db.ts';
import { parseYouTubeId, type YouTube } from '../integrations/youtube.ts';
import type { Job, JobHandler, JobRunner } from '../jobs.ts';
import { saveTranscript } from './transcripts.ts';
import { assertTopicsExist, findVideoByYoutubeId, getVideo, type VideoSummary } from './videos.ts';

/**
 * Imports a video by URL: metadata now (inside the request), captions through a job.
 * An existing video is returned as is (with the topic added, if one was given).
 */
export async function importVideo(
  deps: { db: Db; youtube: YouTube; jobs: JobRunner },
  input: { url: string; topicId?: number },
): Promise<{ created: boolean; video: VideoSummary; job: Job | null }> {
  const { db, youtube, jobs } = deps;
  const youtubeId = parseYouTubeId(input.url);
  if (input.topicId !== undefined) assertTopicsExist(db, [input.topicId]);

  const addTopic = (videoId: number) => {
    if (input.topicId !== undefined) db.prepare('INSERT OR IGNORE INTO video_topics (video_id, topic_id) VALUES (?, ?)').run(videoId, input.topicId);
  };

  const existing = findVideoByYoutubeId(db, youtubeId);
  if (existing) {
    addTopic(existing.id);
    return { created: false, video: getVideo(db, existing.id), job: jobs.activeFor('transcript', existing.id) };
  }

  const meta = await youtube.fetchMetadata(youtubeId);
  let videoId: number;
  try {
    videoId = tx(db, () => {
      const id = Number(
        db
          .prepare('INSERT INTO videos (youtube_id, title, channel, duration_seconds, thumbnail_url) VALUES (?, ?, ?, ?, ?)')
          .run(youtubeId, meta.title, meta.channel, meta.durationSeconds, meta.thumbnailUrl).lastInsertRowid,
      );
      addTopic(id);
      return id;
    });
  } catch (err) {
    // Two imports of the same video at once: the second one returns the first.
    const raced = findVideoByYoutubeId(db, youtubeId);
    if (raced) return { created: false, video: raced, job: jobs.activeFor('transcript', raced.id) };
    throw err;
  }
  const job = jobs.enqueue('transcript', videoId);
  return { created: true, video: getVideo(db, videoId), job };
}

/** Transcript job: fetch captions; a missing or blocked transcript is an outcome, not a failure. */
export function transcriptJobHandler(youtube: YouTube): JobHandler {
  return async (job, { db, signal, setStage }) => {
    const video = db.prepare('SELECT youtube_id, transcript_status FROM videos WHERE id = ?').get(job.videoId) as
      | { youtube_id: string; transcript_status: string }
      | undefined;
    if (!video) return { status: 'video_deleted' };
    if (video.transcript_status !== 'ready') {
      db.prepare(`UPDATE videos SET transcript_status = 'pending', transcript_error = NULL WHERE id = ?`).run(job.videoId);
    }
    setStage('fetching captions');
    const result = await youtube.fetchTranscript(video.youtube_id, signal);
    signal.throwIfAborted();
    if (result.status === 'ready') {
      setStage('saving');
      saveTranscript(db, job.videoId, { source: 'youtube', language: result.language, timed: true, segments: result.segments });
      return { status: 'ready', segments: result.segments.length, language: result.language };
    }
    // Keep an existing transcript (e.g. pasted manually) if a later fetch finds nothing.
    db.prepare(
      `UPDATE videos SET transcript_status = ?, transcript_error = ? WHERE id = ? AND NOT EXISTS
         (SELECT 1 FROM transcripts WHERE video_id = videos.id AND is_current = 1)`,
    ).run(result.status, result.message, job.videoId);
    return { status: result.status, message: result.message };
  };
}
