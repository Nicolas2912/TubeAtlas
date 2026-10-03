import { Hono } from 'hono';
import { IdParam, ManualTranscriptBody, TranscriptQuery } from '../../../shared/api.ts';
import type { AppDeps } from '../app.ts';
import { AppError } from '../errors.ts';
import { getCurrentTranscript, getTranscript, saveManualTranscript } from '../services/transcripts.ts';
import { getVideo } from '../services/videos.ts';
import { validate } from '../validate.ts';

export function transcriptRoutes({ db, jobs }: AppDeps) {
  return new Hono()
    .get('/:id/transcript', validate('param', IdParam), validate('query', TranscriptQuery), (c) => {
      const { id } = c.req.valid('param');
      getVideo(db, id);
      const transcript = getTranscript(db, id, c.req.valid('query').transcriptId);
      if (!transcript) throw new AppError(404, 'NO_TRANSCRIPT', c.req.valid('query').transcriptId ? 'This source transcript revision is no longer available.' : 'This video has no transcript yet.');
      return c.json(transcript);
    })
    .post('/:id/transcript/retry', validate('param', IdParam), (c) => {
      const { id } = c.req.valid('param');
      const video = getVideo(db, id);
      if (video.transcriptStatus === 'ready') throw new AppError(409, 'TRANSCRIPT_READY', 'This video already has a transcript.');
      return c.json(jobs.enqueue('transcript', id), 201);
    })
    .post('/:id/transcript', validate('param', IdParam), validate('json', ManualTranscriptBody), (c) => {
      const { id } = c.req.valid('param');
      getVideo(db, id);
      saveManualTranscript(db, id, c.req.valid('json'));
      return c.json(getCurrentTranscript(db, id)!, 201);
    });
}
