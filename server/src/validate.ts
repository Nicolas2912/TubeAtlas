import { zValidator } from '@hono/zod-validator';
import type { ValidationTargets } from 'hono';
import type { ZodType } from 'zod';
import { AppError } from './errors.ts';

/** zValidator that reports failures in the app's error format (400 VALIDATION_FAILED; 404 for bad path ids). */
export function validate<T extends ZodType, Target extends keyof ValidationTargets>(target: Target, schema: T) {
  return zValidator(target, schema, (result) => {
    if (result.success) return;
    if (target === 'param') throw new AppError(404, 'NOT_FOUND', 'Not found.');
    const issue = result.error.issues[0];
    const where = issue?.path.length ? `${issue.path.join('.')}: ` : '';
    throw new AppError(400, 'VALIDATION_FAILED', `${where}${issue?.message ?? 'Invalid request.'}`);
  });
}
