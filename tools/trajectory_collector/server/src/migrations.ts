import type { Database } from "bun:sqlite";
import { DDL_INDEXES, DDL_TABLES } from "./schema";
import { rebuildAllPredictions } from "./projection";

export const CURRENT_SCHEMA_VERSION = 2;

interface Migration {
  version: number;
  description: string;
  up: (db: Database) => void;
}

export const MIGRATIONS: Migration[] = [
  {
    version: 1,
    description: "initial collector schema",
    up: (db: Database) => {
      db.exec("PRAGMA foreign_keys=OFF;");
      try {
        db.exec(DDL_TABLES);
        db.exec(DDL_INDEXES);
      } finally {
        db.exec("PRAGMA foreign_keys=ON;");
      }
    },
  },
  {
    version: 2,
    description: "rebuildable prediction outcomes in the existing collector database",
    up: (db: Database) => {
      db.exec(`
        CREATE TABLE IF NOT EXISTS prediction_projection (
          session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
          prediction_id TEXT NOT NULL,
          request_event_id TEXT,
          shown_event_id TEXT,
          last_outcome_event_id TEXT,
          file_identity TEXT,
          resolved_file_id INTEGER,
          model_revision TEXT,
          model_gguf_sha256 TEXT,
          runtime_config_hash TEXT,
          context_policy_version TEXT,
          pre_state_hash TEXT,
          pre_state_sequence INTEGER,
          context_blob_hash TEXT,
          action_blob_hash TEXT,
          requested_at INTEGER,
          shown_at INTEGER,
          closed_at INTEGER,
          outcome TEXT,
          outcome_source TEXT,
          projection_version INTEGER NOT NULL,
          updated_through_sequence INTEGER NOT NULL,
          PRIMARY KEY (session_id, prediction_id)
        );
        CREATE INDEX IF NOT EXISTS idx_prediction_projection_session_time
          ON prediction_projection(session_id, requested_at);
        CREATE INDEX IF NOT EXISTS idx_prediction_projection_outcome
          ON prediction_projection(outcome, closed_at);
        CREATE INDEX IF NOT EXISTS idx_prediction_projection_prediction_id
          ON prediction_projection(prediction_id);
        CREATE INDEX IF NOT EXISTS idx_events_prediction_identity
          ON events(session_id, json_extract(payload_json, '$.prediction_id'))
          WHERE json_valid(payload_json)
            AND (event_type LIKE 'prediction_%' OR event_type = 'heartbeat');
      `);
      rebuildAllPredictions(db);
    },
  },
];

export function getAppliedVersions(db: Database): number[] {
  db.exec(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      version INTEGER PRIMARY KEY,
      applied_at TEXT NOT NULL
    );
  `);
  const rows = db.query<{ version: number }, []>("SELECT version FROM schema_migrations ORDER BY version ASC").all();
  return rows.map((r) => r.version);
}

/** Apply all pending migrations. Idempotent: re-running is a no-op. */
export function migrate(db: Database): { applied: number[]; alreadyAt: number[] } {
  const appliedBefore = new Set(getAppliedVersions(db));
  const applied: number[] = [];
  const insert = db.prepare("INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)");
  for (const m of MIGRATIONS) {
    if (appliedBefore.has(m.version)) continue;
    m.up(db);
    insert.run(m.version, new Date().toISOString());
    applied.push(m.version);
  }
  return { applied, alreadyAt: [...appliedBefore] };
}
