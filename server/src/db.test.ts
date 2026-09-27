import { test } from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync, appendFileSync, unlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { MIGRATIONS_DIR, migrate, open, recoverInterrupted, type Db } from './db.ts';

function tempDir(t: { after: (fn: () => void) => void }) {
  const dir = mkdtempSync(join(tmpdir(), 'tubeatlas-db-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
}

function freshDb(t: { after: (fn: () => void) => void }): Db {
  const db = open(tempDir(t));
  t.after(() => db.close());
  migrate(db);
  return db;
}

const addVideo = (db: Db, youtubeId = 'abcdefghijk') =>
  Number(db.prepare('INSERT INTO videos (youtube_id, title) VALUES (?, ?)').run(youtubeId, 'A video').lastInsertRowid);

const addTranscript = (db: Db, videoId: number, revision: number, isCurrent: 0 | 1) =>
  Number(
    db
      .prepare(
        `INSERT INTO transcripts (video_id, revision, is_current, source, timed, sha256, segments_json, plain_text)
         VALUES (?, ?, ?, 'youtube', 1, 'h', '[]', '')`,
      )
      .run(videoId, revision, isCurrent).lastInsertRowid,
  );

const count = (db: Db, table: string) => (db.prepare(`SELECT count(*) AS n FROM ${table}`).get() as { n: number }).n;

test('open creates the data directory, files directory, and a WAL database with foreign keys on', (t) => {
  const dir = join(tempDir(t), 'nested', 'data');
  const db = open(dir);
  t.after(() => db.close());
  assert.ok(existsSync(join(dir, 'tubeatlas.sqlite')));
  assert.ok(existsSync(join(dir, 'files')));
  assert.equal(db.pragma('journal_mode', { simple: true }), 'wal');
  assert.equal(db.pragma('foreign_keys', { simple: true }), 1);
  assert.equal(db.pragma('busy_timeout', { simple: true }), 5000);
});

test('a fresh database migrates; a second run is a no-op', (t) => {
  const db = open(tempDir(t));
  t.after(() => db.close());
  assert.deepEqual(migrate(db), ['001_init.sql']);
  const tables = (db.prepare(`SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name`).all() as { name: string }[]).map((r) => r.name);
  for (const table of ['videos', 'transcripts', 'chunks', 'topics', 'video_topics', 'assets', 'documents', 'conversations', 'messages', 'graphs', 'jobs']) {
    assert.ok(tables.includes(table), table);
  }
  assert.deepEqual(migrate(db), []);
  assert.equal(count(db, 'schema_migrations'), 1);
});

test('an edited or missing applied migration stops startup', (t) => {
  const migrations = tempDir(t);
  for (const name of readdirSync(MIGRATIONS_DIR)) copyFileSync(join(MIGRATIONS_DIR, name), join(migrations, name));
  const db = open(tempDir(t));
  t.after(() => db.close());
  migrate(db, migrations);

  appendFileSync(join(migrations, '001_init.sql'), '\n-- edited later\n');
  assert.throws(() => migrate(db, migrations), /MIGRATION_EDITED: 001_init\.sql/);

  unlinkSync(join(migrations, '001_init.sql'));
  assert.throws(() => migrate(db, migrations), /MIGRATION_MISSING/);
});

test('new migrations apply in order, each atomically', (t) => {
  const migrations = tempDir(t);
  copyFileSync(join(MIGRATIONS_DIR, '001_init.sql'), join(migrations, '001_init.sql'));
  writeFileSync(join(migrations, '002_ok.sql'), 'CREATE TABLE extra (id INTEGER PRIMARY KEY);');
  writeFileSync(join(migrations, '003_broken.sql'), 'CREATE TABLE half (id INTEGER); THIS IS NOT SQL;');
  const db = open(tempDir(t));
  t.after(() => db.close());
  assert.throws(() => migrate(db, migrations));
  const applied = (db.prepare('SELECT name FROM schema_migrations ORDER BY version').all() as { name: string }[]).map((r) => r.name);
  assert.deepEqual(applied, ['001_init.sql', '002_ok.sql']);
  assert.equal(db.prepare(`SELECT name FROM sqlite_master WHERE name = 'half'`).get(), undefined, 'broken migration rolled back');
});

test('deleting a video cascades to everything that belongs to it', (t) => {
  const db = freshDb(t);
  const videoId = addVideo(db);
  const transcriptId = addTranscript(db, videoId, 1, 1);
  const topicId = Number(db.prepare(`INSERT INTO topics (name) VALUES ('ML')`).run().lastInsertRowid);
  db.prepare('INSERT INTO video_topics (video_id, topic_id) VALUES (?, ?)').run(videoId, topicId);
  db.prepare(`INSERT INTO chunks (transcript_id, ord, units_version, unit_ids_json, text, embedding, embedding_model, embedding_dim)
              VALUES (?, 0, 1, '[]', 't', x'00', 'm', 1)`).run(transcriptId);
  const assetId = Number(db.prepare(`INSERT INTO assets (video_id, storage_name, original_name, media_type, size_bytes)
                                     VALUES (?, 'a.pdf', 'a.pdf', 'application/pdf', 1)`).run(videoId).lastInsertRowid);
  db.prepare(`INSERT INTO documents (video_id, title, kind, markdown) VALUES (?, 'Note', 'note', '# hi')`).run(videoId);
  db.prepare(`INSERT INTO documents (video_id, title, kind, asset_id) VALUES (?, 'File', 'attachment', ?)`).run(videoId, assetId);
  const conversationId = Number(db.prepare(`INSERT INTO conversations (video_id, title) VALUES (?, 'Q')`).run(videoId).lastInsertRowid);
  db.prepare(`INSERT INTO messages (conversation_id, role, content, status) VALUES (?, 'user', 'hi', 'complete')`).run(conversationId);
  db.prepare(`INSERT INTO graphs (video_id, transcript_id, graph_json, diagnostics_json, meta_json) VALUES (?, ?, '{}', '{}', '{}')`).run(videoId, transcriptId);
  db.prepare(`INSERT INTO jobs (kind, video_id, status) VALUES ('graph', ?, 'succeeded')`).run(videoId);

  db.prepare('DELETE FROM videos WHERE id = ?').run(videoId);
  for (const table of ['transcripts', 'chunks', 'video_topics', 'assets', 'documents', 'conversations', 'messages', 'graphs', 'jobs']) {
    assert.equal(count(db, table), 0, table);
  }
  assert.equal(count(db, 'topics'), 1, 'topics themselves survive');
});

test('only one current transcript, active job per kind, and generating message at a time', (t) => {
  const db = freshDb(t);
  const videoId = addVideo(db);
  addTranscript(db, videoId, 1, 1);
  assert.throws(() => addTranscript(db, videoId, 2, 1), /UNIQUE/);
  addTranscript(db, videoId, 2, 0); // an older, non-current revision is fine

  const job = db.prepare(`INSERT INTO jobs (kind, video_id, status) VALUES (?, ?, ?)`);
  job.run('graph', videoId, 'queued');
  assert.throws(() => job.run('graph', videoId, 'running'), /UNIQUE/);
  job.run('transcript', videoId, 'running'); // a different kind may run
  job.run('graph', videoId, 'failed'); // finished jobs don't count

  const conversationId = Number(db.prepare(`INSERT INTO conversations (video_id, title) VALUES (?, 'Q')`).run(videoId).lastInsertRowid);
  const message = db.prepare(`INSERT INTO messages (conversation_id, role, status) VALUES (?, 'assistant', ?)`);
  message.run(conversationId, 'generating');
  assert.throws(() => message.run(conversationId, 'generating'), /UNIQUE/);
  message.run(conversationId, 'complete');
});

test('documents: attachments need an asset, other kinds need Markdown', (t) => {
  const db = freshDb(t);
  const videoId = addVideo(db);
  const assetId = Number(db.prepare(`INSERT INTO assets (video_id, storage_name, original_name, media_type, size_bytes)
                                     VALUES (?, 'b.png', 'b.png', 'image/png', 1)`).run(videoId).lastInsertRowid);
  const insert = db.prepare('INSERT INTO documents (video_id, title, kind, markdown, asset_id) VALUES (?, ?, ?, ?, ?)');
  insert.run(videoId, 'ok note', 'note', 'text', null);
  insert.run(videoId, 'ok file', 'attachment', null, assetId);
  assert.throws(() => insert.run(videoId, 'attachment without asset', 'attachment', null, null), /CHECK/);
  assert.throws(() => insert.run(videoId, 'note with asset', 'note', 'text', assetId), /CHECK/);
  assert.throws(() => insert.run(videoId, 'note without text', 'note', null, null), /CHECK/);
  assert.throws(() => insert.run(videoId, 'bad kind', 'poem', 'text', null), /CHECK/);
});

test('recovery marks running jobs and generating messages interrupted, and nothing else', (t) => {
  const db = freshDb(t);
  const videoId = addVideo(db);
  const job = db.prepare(`INSERT INTO jobs (kind, video_id, status) VALUES (?, ?, ?)`);
  job.run('graph', videoId, 'running');
  job.run('transcript', videoId, 'queued');
  const conversationId = Number(db.prepare(`INSERT INTO conversations (video_id, title) VALUES (?, 'Q')`).run(videoId).lastInsertRowid);
  db.prepare(`INSERT INTO messages (conversation_id, role, status, content) VALUES (?, 'assistant', 'generating', 'partial')`).run(conversationId);

  assert.deepEqual(recoverInterrupted(db), { jobs: 1, messages: 1 });
  const jobs = db.prepare('SELECT kind, status, finished_at FROM jobs ORDER BY id').all() as { kind: string; status: string; finished_at: string | null }[];
  assert.equal(jobs[0]!.status, 'interrupted');
  assert.ok(jobs[0]!.finished_at);
  assert.equal(jobs[1]!.status, 'queued', 'queued jobs will simply run');
  assert.deepEqual(db.prepare('SELECT status, content FROM messages').get(), { status: 'interrupted', content: 'partial' });
  assert.deepEqual(recoverInterrupted(db), { jobs: 0, messages: 0 });
});
