import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { Hono } from 'hono';
import { serveStatic } from '@hono/node-server/serve-static';

const frontendDist = fileURLToPath(new URL('../../frontend/dist', import.meta.url));

export function createApp() {
  const api = new Hono().get('/health', (c) => c.json({ ok: true }));

  const app = new Hono().route('/api', api);
  // Unknown API paths are real 404s, never the frontend page.
  app.all('/api/*', (c) => c.json({ error: { code: 'NOT_FOUND', message: 'Not found', retryable: false } }, 404));

  // Built frontend (npm run build). In development Vite serves it instead.
  app.use('*', serveStatic({ root: frontendDist }));
  app.get('*', async (c) => {
    const html = await readFile(`${frontendDist}/index.html`, 'utf8').catch(() => null);
    return html === null ? c.text('Frontend not built. Run npm run build, or use npm run dev.', 404) : c.html(html);
  });

  return { app, api };
}

export type AppType = ReturnType<typeof createApp>['api'];
