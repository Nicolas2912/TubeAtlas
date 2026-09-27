import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';

export type Db = Database.Database;

export const MIGRATIONS_DIR = fileURLToPath(new URL('../migrations', import.meta.url));

/** Opens (creating if needed) the database in dataDir, plus the managed files directory. */
export function open(dataDir: string): Db {
  mkdirSync(join(dataDir, 'files'), { recursive: true });
  const db = new Database(join(dataDir, 'tubeatlas.sqlite'));
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
  db.pragma('busy_timeout = 5000');
  db.pragma('synchronous = NORMAL');
  return db;
}

/** Runs fn in a transaction; it commits on return and rolls back on throw. */
export function tx<T>(db: Db, fn: () => T): T {
  return db.transaction(fn)();
}

/** Applies new `NNN_name.sql` files in order; refuses edited or missing applied migrations. Returns applied names. */
export function migrate(db: Db, dir = MIGRATIONS_DIR): string[] {
  db.exec(`CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT NOT NULL, applied_at TEXT NOT NULL)`);

  const files = readdirSync(dir)
    .filter((name) => name.endsWith('.sql'))
    .map((name) => {
      const match = /^(\d{3})_[a-z0-9_]+\.sql$/.exec(name);
      if (!match) throw new Error(`MIGRATION_BAD_NAME: ${name} (expected NNN_name.sql)`);
      const sql = readFileSync(join(dir, name), 'utf8');
      return { version: Number(match[1]), name, sql, sha256: createHash('sha256').update(sql).digest('hex') };
    })
    .sort((a, b) => a.version - b.version);

  const seen = new Set<number>();
  for (const f of files) {
    if (seen.has(f.version)) throw new Error(`MIGRATION_DUPLICATE_VERSION: ${f.version}`);
    seen.add(f.version);
  }

  const applied = db.prepare('SELECT version, name, sha256 FROM schema_migrations').all() as { version: number; name: string; sha256: string }[];
  for (const a of applied) {
    const file = files.find((f) => f.version === a.version);
    if (!file) throw new Error(`MIGRATION_MISSING: ${a.name} was applied but its file is gone`);
    if (file.sha256 !== a.sha256) throw new Error(`MIGRATION_EDITED: ${a.name} changed after it was applied`);
  }

  const done = new Set(applied.map((a) => a.version));
  const record = db.prepare('INSERT INTO schema_migrations (version, name, sha256, applied_at) VALUES (?, ?, ?, ?)');
  const newlyApplied: string[] = [];
  for (const f of files) {
    if (done.has(f.version)) continue;
    tx(db, () => {
      db.exec(f.sql);
      record.run(f.version, f.name, f.sha256, new Date().toISOString());
    });
    newlyApplied.push(f.name);
  }
  return newlyApplied;
}

/** After a restart nothing is still running: mark leftovers interrupted so the user can retry them explicitly. */
export function recoverInterrupted(db: Db): { jobs: number; messages: number } {
  return tx(db, () => ({
    jobs: db
      .prepare(`UPDATE jobs SET status = 'interrupted', finished_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE status = 'running'`)
      .run().changes,
    messages: db.prepare(`UPDATE messages SET status = 'interrupted' WHERE status = 'generating'`).run().changes,
  }));
}

if (import.meta.main && process.argv.includes('--migrate')) {
  const { loadConfig } = await import('./config.ts');
  const db = open(loadConfig().dataDir);
  const applied = migrate(db);
  console.log(applied.length ? `Applied: ${applied.join(', ')}` : 'Database is up to date.');
  db.close();
}
