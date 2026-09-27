import type { Context } from 'hono';
import type { ContentfulStatusCode } from 'hono/utils/http-status';

/** An expected failure with a stable code; serialized as the API error body. */
export class AppError extends Error {
  readonly status: ContentfulStatusCode;
  readonly code: string;
  readonly retryable: boolean;

  constructor(status: ContentfulStatusCode, code: string, message: string, retryable = false) {
    super(message);
    this.name = 'AppError';
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

export function errorBody(code: string, message: string, retryable = false) {
  return { error: { code, message, retryable } };
}

/** Hono onError handler: AppErrors keep their code; anything else becomes a generic 500. */
export function handleError(err: Error, c: Context) {
  if (err instanceof AppError) return c.json(errorBody(err.code, err.message, err.retryable), err.status);
  console.error(`Unhandled error on ${c.req.method} ${new URL(c.req.url).pathname}:`, err);
  return c.json(errorBody('INTERNAL', 'Something went wrong.', true), 500);
}
