import { serve } from '@hono/node-server';
import { createApp } from './app.ts';

const port = Number(process.env.PORT ?? 5170);
const { app } = createApp();

serve({ fetch: app.fetch, hostname: '127.0.0.1', port }, (info) => {
  console.log(`TubeAtlas listening on http://127.0.0.1:${info.port}`);
});
