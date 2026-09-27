// Test helper: a real app on a fresh temporary data directory (no mocks of our own modules).
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createApp } from './app.ts';
import { loadConfig } from './config.ts';
import { migrate, open } from './db.ts';

export const TEST_ORIGIN = 'http://127.0.0.1:5170';

export function createTestApp(env: Record<string, string> = {}) {
  const dataDir = mkdtempSync(join(tmpdir(), 'tubeatlas-test-'));
  const config = loadConfig({ DATA_DIR: dataDir, PORT: '5170', ...env });
  const db = open(dataDir);
  migrate(db);
  const { app } = createApp({ db, config });
  return {
    app,
    db,
    config,
    dataDir,
    /** Sends a request as the local browser would (loopback host). */
    request: (path: string, init?: RequestInit) => app.request(`${TEST_ORIGIN}${path}`, init),
    cleanup: () => {
      db.close();
      rmSync(dataDir, { recursive: true, force: true });
    },
  };
}
