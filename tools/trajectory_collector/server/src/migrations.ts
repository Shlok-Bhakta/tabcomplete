import type { Database } from "bun:sqlite";
import { DDL_INDEXES, DDL_TABLES } from "./schema";

export const CURRENT_SCHEMA_VERSION = 1;

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
