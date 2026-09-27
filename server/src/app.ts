import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { Hono, type MiddlewareHandler } from 'hono';
import { bodyLimit } from 'hono/body-limit';
import { serveStatic } from '@hono/node-server/serve-static';
import { MAX_MANUAL_TRANSCRIPT_BYTES } from '../../shared/api.ts';
import type { Config } from './config.ts';
import type { Db } from './db.ts';
import { errorBody, handleError } from './errors.ts';
import type { YouTube } from './integrations/youtube.ts';
import type { JobRunner } from './jobs.ts';
import { jobRoutes } from './routes/jobs.ts';
import { topicRoutes } from './routes/topics.ts';
import { transcriptRoutes } from './routes/transcripts.ts';
import { videoRoutes } from './routes/videos.ts';
import { DEV_WEB_PORT } from '../../shared/ports.ts';

export type AppDeps = { db: Db; config: Config; youtube: YouTube; jobs: JobRunner };

const frontendDist = fileURLToPath(new URL('../../frontend/dist', import.meta.url));

const isApiPath = (path: string) => path === '/api' || path.startsWith('/api/');

/**
 * Only loopback hosts on our ports may talk to the server. This rejects DNS-rebinding
 * (a foreign domain resolving to 127.0.0.1) and cross-site state-changing requests.
 */
function localOnly(port: number): MiddlewareHandler {
  const hosts = new Set([port, DEV_WEB_PORT].flatMap((p) => [`127.0.0.1:${p}`, `localhost:${p}`]));
  const origins = new Set([...hosts].map((h) => `http://${h}`));
  return async (c, next) => {
    if (!hosts.has(new URL(c.req.url).host)) return c.json(errorBody('BAD_HOST', 'Unexpected host.'), 403);
    const origin = c.req.header('origin');
    if (!['GET', 'HEAD', 'OPTIONS'].includes(c.req.method) && origin !== undefined && !origins.has(origin)) {
      return c.json(errorBody('BAD_ORIGIN', 'Cross-origin request refused.'), 403);
    }
    await next();
  };
}

const MB = 1024 * 1024;
const limit = (maxSize: number) =>
  bodyLimit({ maxSize, onError: (c) => c.json(errorBody('PAYLOAD_TOO_LARGE', `Request body is larger than ${maxSize / MB} MB.`), 413) });

/** 1 MB for API requests, except the few routes that accept large bodies. */
function bodyLimits(): MiddlewareHandler {
  const standard = limit(MB);
  const large: [method: string, path: RegExp, handler: MiddlewareHandler][] = [
    ['POST', /^\/api\/videos\/\d+\/transcript$/, limit(MAX_MANUAL_TRANSCRIPT_BYTES)],
  ];
  return (c, next) => {
    const match = large.find(([method, path]) => c.req.method === method && path.test(c.req.path));
    return (match ? match[2] : standard)(c, next);
  };
}

export function createApp(deps: AppDeps) {
  const { config } = deps;
  const api = new Hono()
    .use(bodyLimits())
    .get('/health', (c) =>
      c.json({ ok: true, aiConfigured: config.openrouterApiKey !== undefined, youtubeKey: config.youtubeApiKey !== undefined }),
    )
    .get('/settings', (c) =>
      c.json({
        aiConfigured: config.openrouterApiKey !== undefined,
        youtubeKey: config.youtubeApiKey !== undefined,
        models: { chat: config.chatModel, embedding: config.embeddingModel, kg: config.kgModel, kgReasoningEffort: config.kgReasoningEffort },
        dataDir: config.dataDir,
      }),
    )
    .route('/videos', videoRoutes(deps))
    .route('/videos', transcriptRoutes(deps))
    .route('/topics', topicRoutes(deps))
    .route('/jobs', jobRoutes(deps));
  api.onError(handleError);

  const app = new Hono();
  app.onError(handleError);
  app.notFound((c) =>
    isApiPath(c.req.path) ? c.json(errorBody('NOT_FOUND', 'Not found.'), 404) : c.text('Not found', 404),
  );
  app.use('*', localOnly(config.port));
  app.route('/api', api);

  // Built frontend (npm run build); in development Vite serves it instead. API paths never fall through to it.
  const assets = serveStatic({ root: frontendDist });
  app.use('*', (c, next) => (isApiPath(c.req.path) ? next() : assets(c, next)));
  app.get('*', async (c, next) => {
    if (isApiPath(c.req.path)) return next();
    const html = await readFile(`${frontendDist}/index.html`, 'utf8').catch(() => null);
    return html === null ? c.text('Frontend not built. Run npm run build, or use npm run dev.', 404) : c.html(html);
  });

  return { app, api };
}

export type AppType = ReturnType<typeof createApp>['api'];
