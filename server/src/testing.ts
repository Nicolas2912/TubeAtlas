// Test helper: a real app on a fresh temporary data directory (no mocks of our own modules).
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { ChatProvider } from './services/chat.ts';
import { createApp } from './app.ts';
import { loadConfig } from './config.ts';
import { migrate, open } from './db.ts';
import type { YouTube } from './integrations/youtube.ts';
import { createJobRunner } from './jobs.ts';
import { transcriptJobHandler } from './services/import.ts';

export const TEST_ORIGIN = 'http://127.0.0.1:5170';

/** A YouTube stand-in that fails loudly; tests override the methods they use. */
export const unusedYouTube: YouTube = {
  fetchMetadata: async () => {
    throw new Error('fetchMetadata not expected in this test');
  },
  fetchTranscript: async () => {
    throw new Error('fetchTranscript not expected in this test');
  },
};

export function createTestApp(options: { env?: Record<string, string>; youtube?: YouTube; openrouter?: ChatProvider } = {}) {
  const dataDir = mkdtempSync(join(tmpdir(), 'tubeatlas-test-'));
  const config = loadConfig({ DATA_DIR: dataDir, PORT: '5170', ...options.env });
  const db = open(dataDir);
  migrate(db);
  const youtube = options.youtube ?? unusedYouTube;
  const jobs = createJobRunner(db, { transcript: transcriptJobHandler(youtube) });
  jobs.start();
  const { app, chat } = createApp({ db, config, youtube, jobs, openrouter: options.openrouter });
  return {
    app,
    db,
    config,
    dataDir,
    jobs,
    chat,
    /** Sends a request as the local browser would (loopback host). */
    request: (path: string, init?: RequestInit) => app.request(`${TEST_ORIGIN}${path}`, init),
    /** Sends a JSON body as the local browser would. */
    json: (method: string, path: string, body: unknown) =>
      app.request(`${TEST_ORIGIN}${path}`, { method, headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
    cleanup: async () => {
      await Promise.all([jobs.stop(), chat.stop()]);
      db.close();
      rmSync(dataDir, { recursive: true, force: true });
    },
  };
}
