import { Hono } from 'hono';
import { IdParam } from '../../../shared/api.ts';
import type { AppDeps } from '../app.ts';
import { AppError } from '../errors.ts';
import { validate } from '../validate.ts';

export function jobRoutes({ jobs }: AppDeps) {
  return new Hono()
    .get('/:id', validate('param', IdParam), (c) => {
      const job = jobs.get(c.req.valid('param').id);
      if (!job) throw new AppError(404, 'NOT_FOUND', 'Job not found.');
      return c.json(job);
    })
    .post('/:id/cancel', validate('param', IdParam), (c) => c.json(jobs.cancel(c.req.valid('param').id)))
    .post('/:id/retry', validate('param', IdParam), (c) => c.json(jobs.retry(c.req.valid('param').id), 201));
}
