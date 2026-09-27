import { Hono } from 'hono';
import { IdParam, ManualTranscriptBody } from '../../../shared/api.ts';
import type { AppDeps } from '../app.ts';
import { AppError } from '../errors.ts';
import { getCurrentTranscript, saveManualTranscript } from '../services/transcripts.ts';
import { getVideo } from '../services/videos.ts';
import { validate } from '../validate.ts';

export function transcriptRoutes({ db }: AppDeps) {
  return new Hono()
    .get('/:id/transcript', validate('param', IdParam), (c) => {
      const { id } = c.req.valid('param');
      getVideo(db, id);
      const transcript = getCurrentTranscript(db, id);
      if (!transcript) throw new AppError(404, 'NO_TRANSCRIPT', 'This video has no transcript yet.');
      return c.json(transcript);
    })
    .post('/:id/transcript', validate('param', IdParam), validate('json', ManualTranscriptBody), (c) => {
      const { id } = c.req.valid('param');
      getVideo(db, id);
      saveManualTranscript(db, id, c.req.valid('json'));
      return c.json(getCurrentTranscript(db, id)!, 201);
    });
}
