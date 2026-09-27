CREATE TABLE videos (
  id                INTEGER PRIMARY KEY,
  youtube_id        TEXT NOT NULL UNIQUE,
  title             TEXT NOT NULL,
  channel           TEXT,
  duration_seconds  REAL,
  thumbnail_url     TEXT,
  playback_seconds  REAL NOT NULL DEFAULT 0,
  transcript_status TEXT NOT NULL DEFAULT 'pending'
    CHECK (transcript_status IN ('pending','ready','no_captions','blocked','failed')),
  transcript_error  TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE transcripts (
  id            INTEGER PRIMARY KEY,
  video_id      INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  revision      INTEGER NOT NULL,
  is_current    INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0,1)),
  source        TEXT NOT NULL CHECK (source IN ('youtube','upload_timed','paste_text')),
  language      TEXT,
  timed         INTEGER NOT NULL CHECK (timed IN (0,1)),
  sha256        TEXT NOT NULL,
  segments_json TEXT NOT NULL,          -- [{id,start,end,text}]; start/end null when untimed
  plain_text    TEXT NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE (video_id, revision)
);
CREATE UNIQUE INDEX transcripts_one_current ON transcripts(video_id) WHERE is_current = 1;

CREATE TABLE chunks (
  id              INTEGER PRIMARY KEY,
  transcript_id   INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
  ord             INTEGER NOT NULL,
  units_version   INTEGER NOT NULL,
  unit_ids_json   TEXT NOT NULL,
  start_seconds   REAL,
  end_seconds     REAL,
  text            TEXT NOT NULL,
  embedding       BLOB NOT NULL,        -- Float32Array bytes, L2-normalized
  embedding_model TEXT NOT NULL,
  embedding_dim   INTEGER NOT NULL,
  UNIQUE (transcript_id, ord)
);

CREATE TABLE topics (
  id         INTEGER PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE video_topics (
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
  PRIMARY KEY (video_id, topic_id)
);
CREATE INDEX video_topics_topic ON video_topics(topic_id);

CREATE TABLE assets (
  id            INTEGER PRIMARY KEY,
  video_id      INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  storage_name  TEXT NOT NULL UNIQUE,    -- '<uuid>.<ext>' in DATA_DIR/files
  original_name TEXT NOT NULL,
  media_type    TEXT NOT NULL,
  size_bytes    INTEGER NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE documents (
  id         INTEGER PRIMARY KEY,
  video_id   INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  title      TEXT NOT NULL,
  kind       TEXT NOT NULL CHECK (kind IN ('note','summary','study_guide','qa','attachment')),
  markdown   TEXT,
  asset_id   INTEGER REFERENCES assets(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  CHECK ((kind = 'attachment') = (asset_id IS NOT NULL)),
  CHECK ((kind = 'attachment') OR markdown IS NOT NULL)
);
CREATE INDEX documents_video ON documents(video_id, updated_at DESC);
CREATE INDEX documents_updated ON documents(updated_at DESC);

CREATE TABLE conversations (
  id         INTEGER PRIMARY KEY,
  video_id   INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  title      TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX conversations_video ON conversations(video_id, updated_at DESC);

CREATE TABLE messages (
  id              INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role            TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content         TEXT NOT NULL DEFAULT '',
  status          TEXT NOT NULL CHECK (status IN ('generating','complete','incomplete','interrupted','failed')),
  citations_json  TEXT NOT NULL DEFAULT '[]',
  context_json    TEXT,                 -- transcript id/revision, document ids, retrieval mode
  model           TEXT,
  usage_json      TEXT,
  error_code      TEXT,
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX messages_conversation ON messages(conversation_id, id);
CREATE UNIQUE INDEX messages_one_generating ON messages(conversation_id) WHERE status = 'generating';

CREATE TABLE graphs (
  id               INTEGER PRIMARY KEY,
  video_id         INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  transcript_id    INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
  is_current       INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0,1)),
  graph_json       TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL,
  meta_json        TEXT NOT NULL,
  layout_json      TEXT NOT NULL DEFAULT '{}',
  created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE UNIQUE INDEX graphs_one_current ON graphs(video_id) WHERE is_current = 1;

CREATE TABLE jobs (
  id            INTEGER PRIMARY KEY,
  kind          TEXT NOT NULL CHECK (kind IN ('transcript','graph')),
  video_id      INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  status        TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed','cancelled','interrupted')),
  stage         TEXT,
  input_json    TEXT NOT NULL DEFAULT '{}',
  result_json   TEXT,
  error_code    TEXT,
  error_message TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  started_at    TEXT,
  finished_at   TEXT
);
CREATE UNIQUE INDEX jobs_one_active ON jobs(kind, video_id) WHERE status IN ('queued','running');
CREATE INDEX jobs_queue ON jobs(status, id);
