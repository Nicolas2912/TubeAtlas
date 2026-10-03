import { useCallback, useEffect, useRef, useState } from 'react';
import { hc } from 'hono/client';
import type { AppType } from '../../server/src/app.ts';
import type { Job } from '../../server/src/jobs.ts';

export type { VideoSummary } from '../../server/src/services/videos.ts';
export type { Topic } from '../../server/src/services/topics.ts';
export type { Transcript } from '../../server/src/services/transcripts.ts';
export type { Document, TextKind } from '../../server/src/services/documents.ts';
export type { Job };

export const api = hc<AppType>('/api');

/** An API failure in the server's error format (or a network failure, code NETWORK). */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;
  constructor(status: number, code: string, message: string, retryable: boolean) {
    super(message);
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

export async function checked<R extends Response>(request: Promise<R>): Promise<R> {
  let res: R;
  try {
    res = await request;
  } catch {
    throw new ApiError(0, 'NETWORK', 'Could not reach TubeAtlas. Is the server running?', true);
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { error?: { code?: string; message?: string; retryable?: boolean } } | null;
    throw new ApiError(res.status, body?.error?.code ?? `HTTP_${res.status}`, body?.error?.message ?? `Request failed (${res.status}).`, body?.error?.retryable ?? false);
  }
  return res;
}

/** Awaits a typed client call and returns its JSON body, throwing ApiError on failure. */
export async function unwrap<R extends Response & { json(): Promise<unknown> }>(request: Promise<R>): Promise<Awaited<ReturnType<R['json']>>> {
  const res = await checked(request);
  return (await res.json()) as Awaited<ReturnType<R['json']>>;
}

/** For calls without a response body (204). */
export async function send(request: Promise<Response>): Promise<void> {
  await checked(request);
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : 'Something went wrong.';
}

/** Loads data when deps change; keeps the previous data while reloading. */
export function useApi<T>(load: () => Promise<T>, deps: unknown[]) {
  const key = JSON.stringify(deps);
  const [state, setState] = useState<{ key: string; data?: T; error?: unknown; loading: boolean }>({ key, loading: true });
  const [version, setVersion] = useState(0);
  useEffect(() => {
    let alive = true;
    setState((s) => s.key === key ? { ...s, error: undefined, loading: true } : { key, loading: true });
    load().then(
      (data) => alive && setState({ key, data, loading: false }),
      (error: unknown) => alive && setState((s) => ({ key, data: s.data, error, loading: false })),
    );
    return () => {
      alive = false;
    };
  }, [key, version]);
  const reload = useCallback(() => setVersion((v) => v + 1), []);
  const current = state.key === key;
  return { data: current ? state.data : undefined, error: current ? state.error : undefined, loading: current ? state.loading : true, reload };
}

const ACTIVE = new Set(['queued', 'running']);

/** Polls a job every 2 s while it is queued or running; calls onFinish once when it ends. */
export function useJob(jobId: number | null | undefined, onFinish?: (job: Job) => void): Job | null {
  const [job, setJob] = useState<Job | null>(null);
  const finish = useRef(onFinish);
  finish.current = onFinish;
  useEffect(() => {
    setJob(null);
    if (!jobId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const next = await unwrap(api.jobs[':id'].$get({ param: { id: String(jobId) } }));
        if (!alive) return;
        setJob(next);
        if (ACTIVE.has(next.status)) timer = setTimeout(poll, 2000);
        else finish.current?.(next);
      } catch {
        if (alive) timer = setTimeout(poll, 2000); // keep trying while the server restarts
      }
    };
    void poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [jobId]);
  return job;
}
