// Persisted, sequential job runner: one job at a time in id order, in this process.
import type { JobKind, JobStatus } from '../../shared/api.ts';
import type { Db } from './db.ts';
import { AppError } from './errors.ts';

export type Job = {
  id: number;
  kind: JobKind;
  videoId: number;
  status: JobStatus;
  stage: string | null;
  input: unknown;
  result: unknown;
  error: { code: string; message: string } | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
};

export type JobContext = { db: Db; signal: AbortSignal; setStage: (stage: string) => void };

/** Writes its outputs (in one transaction) and returns a small result object. Throwing fails the job. */
export type JobHandler = (job: Job, ctx: JobContext) => Promise<unknown>;

export type JobRunner = ReturnType<typeof createJobRunner>;

type Row = {
  id: number;
  kind: JobKind;
  video_id: number;
  status: JobStatus;
  stage: string | null;
  input_json: string;
  result_json: string | null;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

const NOW = `strftime('%Y-%m-%dT%H:%M:%fZ','now')`;

function toJob(row: Row): Job {
  return {
    id: row.id,
    kind: row.kind,
    videoId: row.video_id,
    status: row.status,
    stage: row.stage,
    input: JSON.parse(row.input_json),
    result: row.result_json === null ? null : JSON.parse(row.result_json),
    error: row.error_code === null ? null : { code: row.error_code, message: row.error_message ?? '' },
    createdAt: row.created_at,
    startedAt: row.started_at,
    finishedAt: row.finished_at,
  };
}

export function createJobRunner(db: Db, handlers: Partial<Record<JobKind, JobHandler>>) {
  const running = new Map<number, AbortController>();
  let started = false;
  let looping = false;
  let idleWaiters: (() => void)[] = [];

  const get = (id: number): Job | null => {
    const row = db.prepare('SELECT * FROM jobs WHERE id = ?').get(id) as Row | undefined;
    return row ? toJob(row) : null;
  };

  const activeFor = (kind: JobKind, videoId: number): Job | null => {
    const row = db
      .prepare(`SELECT * FROM jobs WHERE kind = ? AND video_id = ? AND status IN ('queued','running') ORDER BY id DESC LIMIT 1`)
      .get(kind, videoId) as Row | undefined;
    return row ? toJob(row) : null;
  };

  /** Queues a job, or returns the one already queued/running for the same kind and video. */
  function enqueue(kind: JobKind, videoId: number, input: unknown = {}): Job {
    const existing = activeFor(kind, videoId);
    if (existing) return existing;
    let id: number;
    try {
      id = Number(
        db.prepare(`INSERT INTO jobs (kind, video_id, status, input_json) VALUES (?, ?, 'queued', ?)`).run(kind, videoId, JSON.stringify(input))
          .lastInsertRowid,
      );
    } catch (err) {
      const raced = activeFor(kind, videoId); // the partial unique index enforces one active job
      if (raced) return raced;
      throw err;
    }
    wake();
    return get(id)!;
  }

  function cancel(id: number): Job {
    const job = get(id);
    if (!job) throw new AppError(404, 'NOT_FOUND', 'Job not found.');
    if (job.status === 'queued') {
      db.prepare(`UPDATE jobs SET status = 'cancelled', finished_at = ${NOW} WHERE id = ? AND status = 'queued'`).run(id);
    } else if (job.status === 'running') {
      running.get(id)?.abort(); // becomes 'cancelled' when the handler exits
    }
    return get(id)!;
  }

  function retry(id: number): Job {
    const job = get(id);
    if (!job) throw new AppError(404, 'NOT_FOUND', 'Job not found.');
    if (!['failed', 'cancelled', 'interrupted'].includes(job.status)) {
      throw new AppError(409, 'JOB_NOT_RETRYABLE', `A ${job.status} job cannot be retried.`);
    }
    return enqueue(job.kind, job.videoId, job.input);
  }

  /** Aborts and cancels every active job of a video (used before deleting it). */
  function cancelForVideo(videoId: number) {
    const ids = db.prepare(`SELECT id FROM jobs WHERE video_id = ? AND status IN ('queued','running')`).all(videoId) as { id: number }[];
    for (const { id } of ids) cancel(id);
  }

  function start() {
    started = true;
    wake();
  }

  function wake() {
    if (started && !looping) setImmediate(loop);
  }

  async function loop() {
    if (looping) return;
    looping = true;
    try {
      while (started) {
        const row = db.prepare(`SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1`).get() as Row | undefined;
        if (!row) break;
        await run(toJob(row));
      }
    } finally {
      looping = false;
      const waiters = idleWaiters;
      idleWaiters = [];
      for (const resolve of waiters) resolve();
    }
  }

  async function run(job: Job) {
    const claimed = db.prepare(`UPDATE jobs SET status = 'running', started_at = ${NOW} WHERE id = ? AND status = 'queued'`).run(job.id);
    if (claimed.changes === 0) return;
    const controller = new AbortController();
    running.set(job.id, controller);
    const ctx: JobContext = {
      db,
      signal: controller.signal,
      setStage: (stage) => db.prepare('UPDATE jobs SET stage = ? WHERE id = ?').run(stage, job.id),
    };
    try {
      const handler = handlers[job.kind];
      if (!handler) throw new AppError(500, 'NO_HANDLER', `No handler for ${job.kind} jobs.`);
      const result = await handler({ ...job, status: 'running' }, ctx);
      db.prepare(`UPDATE jobs SET status = 'succeeded', result_json = ?, finished_at = ${NOW} WHERE id = ? AND status = 'running'`).run(
        JSON.stringify(result ?? null),
        job.id,
      );
    } catch (err) {
      if (controller.signal.aborted) {
        db.prepare(`UPDATE jobs SET status = 'cancelled', finished_at = ${NOW} WHERE id = ? AND status = 'running'`).run(job.id);
      } else {
        const known = err instanceof AppError;
        if (!known) console.error(`Job ${job.id} (${job.kind}) failed:`, err);
        db.prepare(
          `UPDATE jobs SET status = 'failed', error_code = ?, error_message = ?, finished_at = ${NOW} WHERE id = ? AND status = 'running'`,
        ).run(known ? err.code : 'JOB_FAILED', known ? err.message : 'The job failed unexpectedly.', job.id);
      }
    } finally {
      running.delete(job.id);
    }
  }

  /**
   * Stops taking new jobs and resolves when the current one has finished. A job still running when the
   * process exits stays 'running' in the database and becomes 'interrupted' on the next start.
   */
  function stop(): Promise<void> {
    started = false;
    if (!looping) return Promise.resolve();
    return new Promise((resolve) => idleWaiters.push(resolve));
  }

  /** Resolves once no job is queued or running (tests and scripts). */
  function idle(): Promise<void> {
    const pending = db.prepare(`SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1`).get();
    if (!pending && !looping) return Promise.resolve();
    return new Promise((resolve) => {
      idleWaiters.push(resolve);
      wake();
    });
  }

  return { enqueue, cancel, retry, cancelForVideo, get, activeFor, start, stop, idle };
}
