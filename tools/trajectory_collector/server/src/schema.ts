/**
 * Central DDL for the collector database.
 *
 * All CREATE TABLE / CREATE INDEX statements live here and are executed
 * exclusively by migrations.ts. Request handlers must never execute DDL.
 */

export const DDL_TABLES = `
CREATE TABLE IF NOT EXISTS machines (
  machine_id TEXT PRIMARY KEY,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  editor TEXT,
  editor_version TEXT,
  plugin_version TEXT,
  platform TEXT,
  hostname_hash TEXT
);

CREATE TABLE IF NOT EXISTS repositories (
  repo_id TEXT PRIMARY KEY,
  root_name TEXT,
  root_identity_hash TEXT,
  origin_url TEXT,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  machine_id TEXT NOT NULL REFERENCES machines(machine_id) ON UPDATE CASCADE,
  repo_id TEXT REFERENCES repositories(repo_id) ON UPDATE CASCADE,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  cwd_name TEXT,
  git_head_at_start TEXT,
  git_branch_at_start TEXT,
  editor TEXT,
  editor_version TEXT,
  plugin_version TEXT
);

CREATE TABLE IF NOT EXISTS files (
  file_id INTEGER PRIMARY KEY AUTOINCREMENT,
  repo_id TEXT NOT NULL REFERENCES repositories(repo_id) ON UPDATE CASCADE ON DELETE CASCADE,
  relative_path TEXT NOT NULL,
  extension TEXT,
  language TEXT,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  UNIQUE(repo_id, relative_path)
);

CREATE TABLE IF NOT EXISTS blobs (
  sha256 TEXT PRIMARY KEY,
  original_bytes INTEGER NOT NULL,
  stored_bytes INTEGER NOT NULL,
  compression TEXT NOT NULL,
  content BLOB NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS file_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id INTEGER NOT NULL REFERENCES files(file_id) ON UPDATE CASCADE ON DELETE CASCADE,
  -- Logical FK to blobs(sha256). Intentionally NOT a hard FOREIGN KEY so
  -- snapshots can be ingested before the corresponding blob bytes arrive
  -- (client uploads blobs separately)._queries join against blobs explicitly.
  sha256 TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  source TEXT NOT NULL,
  UNIQUE(file_id, sha256)
);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON UPDATE CASCADE ON DELETE CASCADE,
  sequence_number INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  -- Logical reference to files(file_id); intentionally no hard FK so events
  -- can arrive before the repository snapshot that registers the file.
  file_id INTEGER,
  changedtick INTEGER,
  cursor_row INTEGER,
  cursor_col INTEGER,
  mode TEXT,
  payload_json TEXT,
  UNIQUE(session_id, sequence_number)
);
`;

export const DDL_INDEXES = `
CREATE INDEX IF NOT EXISTS idx_sessions_machine ON sessions(machine_id);
CREATE INDEX IF NOT EXISTS idx_sessions_repo ON sessions(repo_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_files_repo_path ON files(repo_id, relative_path);
CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);
CREATE INDEX IF NOT EXISTS idx_file_versions_file_observed ON file_versions(file_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_file_versions_sha ON file_versions(sha256);
CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events(session_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_file ON events(file_id);
CREATE INDEX IF NOT EXISTS idx_events_session_type ON events(session_id, event_type);
`;
