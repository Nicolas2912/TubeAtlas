import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { Hono, type MiddlewareHandler } from 'hono';
import { bodyLimit } from 'hono/body-limit';
import { serveStatic } from '@hono/node-server/serve-static';
import type { Config } from './config.ts';
import type { Db } from './db.ts';
import { errorBody, handleError } from './errors.ts';
import { DEV_WEB_PORT } from '../../shared/ports.ts';

export type AppDeps = { db: Db; config: Config };

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

export function createApp({ config }: AppDeps) {
  const api = new Hono()
    .use(
      bodyLimit({
        maxSize: 1024 * 1024,
        onError: (c) => c.json(errorBody('PAYLOAD_TOO_LARGE', 'Request body is larger than 1 MB.'), 413),
      }),
    )
    .get('/health', (c) =>
      c.json({ ok: true, aiConfigured: config.openrouterApiKey !== undefined, youtubeKey: config.youtubeApiKey !== undefined }),
    );
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
