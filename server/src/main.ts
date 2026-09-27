import { serve } from '@hono/node-server';
import { createApp } from './app.ts';
import { loadConfig } from './config.ts';
import { migrate, open, recoverInterrupted } from './db.ts';

const config = loadConfig();
const db = open(config.dataDir);
const applied = migrate(db);
if (applied.length) console.log(`Applied migrations: ${applied.join(', ')}`);

// Before anything can run: whatever was running when the process stopped is now interrupted.
const recovered = recoverInterrupted(db);
if (recovered.jobs || recovered.messages) {
  console.log(`Marked interrupted after restart: ${recovered.jobs} job(s), ${recovered.messages} message(s).`);
}

const { app } = createApp({ db, config });
const server = serve({ fetch: app.fetch, hostname: '127.0.0.1', port: config.port }, (info) => {
  console.log(`TubeAtlas listening on http://127.0.0.1:${info.port} (data: ${config.dataDir})`);
});

function shutdown() {
  server.close(() => {
    db.close();
    process.exit(0);
  });
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
