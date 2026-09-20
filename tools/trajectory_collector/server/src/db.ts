import { Database } from "bun:sqlite";
import { migrate } from "./migrations";

export function resolveDbPath(): string {
  const fromEnv = process.env["TABCOMPLETE_COLLECTOR_DB"];
  if (fromEnv !== undefined && fromEnv.trim() !== "") return fromEnv;
  return "/data/collector.sqlite";
}

export function openDatabase(dbPath?: string): Database {
  const path = dbPath ?? resolveDbPath();
  const db = new Database(path, { create: true });
  // Required durability / concurrency pragmas on every startup.
  db.exec("PRAGMA journal_mode=WAL;");
  db.exec("PRAGMA synchronous=NORMAL;");
  db.exec("PRAGMA foreign_keys=ON;");
  db.exec("PRAGMA busy_timeout=5000;");
  migrate(db);
  return db;
}
