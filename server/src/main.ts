import { serve } from '@hono/node-server';
import { createApp } from './app.ts';
import { loadConfig } from './config.ts';
import { migrate, open, recoverInterrupted } from './db.ts';
import { createYouTube } from './integrations/youtube.ts';
import { createJobRunner } from './jobs.ts';
import { transcriptJobHandler } from './services/import.ts';

const config = loadConfig();
const db = open(config.dataDir);
const applied = migrate(db);
if (applied.length) console.log(`Applied migrations: ${applied.join(', ')}`);

// Before anything can run: whatever was running when the process stopped is now interrupted.
const recovered = recoverInterrupted(db);
if (recovered.jobs || recovered.messages) {
  console.log(`Marked interrupted after restart: ${recovered.jobs} job(s), ${recovered.messages} message(s).`);
}

const youtube = createYouTube({ apiKey: config.youtubeApiKey });
const jobs = createJobRunner(db, { transcript: transcriptJobHandler(youtube) });
const { app } = createApp({ db, config, youtube, jobs });
jobs.start();
const server = serve({ fetch: app.fetch, hostname: '127.0.0.1', port: config.port }, (info) => {
  console.log(`TubeAtlas listening on http://127.0.0.1:${info.port} (data: ${config.dataDir})`);
});

function shutdown() {
  server.close(async () => {
    // Give a running job a moment to finish; anything still running becomes 'interrupted' on the next start.
    await Promise.race([jobs.stop(), new Promise((resolve) => setTimeout(resolve, 3000))]);
    db.close();
    process.exit(0);
  });
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
