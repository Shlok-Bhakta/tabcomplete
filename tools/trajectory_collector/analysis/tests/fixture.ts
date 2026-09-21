import { Database } from "bun:sqlite";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";

export const REL_PATH = "exercises/07_structs/structs1.rs";
export const REPO_A = "repo:rustlings";
export const REPO_B = "repo:other";
export const ABS_PATH = "/home/user/rustlings/exercises/07_structs/structs1.rs";
export const FILE2_REL = "exercises/07_structs/structs2.rs";
export const FILE2_ABS = "/home/user/rustlings/exercises/07_structs/structs2.rs";
export const SESSION_ID = "sess-fixture-0001";
export const MACHINE_ID = "machine-fixture-1";
export const BASE_TS = 1700000000000;

export function sha256HexText(s: string): string {
  return createHash("sha256").update(s, "utf8").digest("hex");
}

const DDL = `
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
  file_id INTEGER,
  changedtick INTEGER,
  cursor_row INTEGER,
  cursor_col INTEGER,
  mode TEXT,
  payload_json TEXT,
  UNIQUE(session_id, sequence_number)
);
`;

export interface DeltaSpec {
  start_row: number;
  old_end_row: number;
  new_end_row: number;
  deleted_text: string;
  inserted_text: string;
  timestamp_ms: number;
  sequence_number: number;
  cursor_after: { row: number; col: number } | null;
  path: string;
}

export interface Fixture {
  dir: string;
  dbPath: string;
  /** all file1 contents in order: [open, write1, write2] */
  contents: string[];
  hashes: string[];
  anchorTs: number[];
  deltas: DeltaSpec[];
  file2Hash: string;
  file2Ts: number[];
  grossInserted: number;
  grossDeleted: number;
  cleanup: () => void;
}

function storeBlob(
  db: Database,
  text: string,
  compression: "gzip" | "raw",
): { sha: string; stored: Uint8Array } {
  const sha = sha256HexText(text);
  const raw = Buffer.from(text, "utf8");
  const stored: Uint8Array =
    compression === "gzip" ? (gzipSync(raw) as unknown as Uint8Array) : new Uint8Array(raw);
  db.query(
    "INSERT OR IGNORE INTO blobs (sha256, original_bytes, stored_bytes, compression, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
  ).run(sha, raw.length, (stored as Uint8Array).length, compression, stored, "2023-11-14T22:13:20.000Z");
  return { sha, stored };
}

export function buildFixture(): Fixture {
  const dir = mkdtempSync(join(tmpdir(), "analysis-test-"));
  const dbPath = join(dir, "fixture.sqlite");
  const db = new Database(dbPath, { create: true });
  db.exec(DDL);

  const T0 = "2023-11-14T22:13:20.000Z";

  db.query(
    "INSERT INTO machines (machine_id, first_seen_at, last_seen_at, editor, platform) VALUES (?, ?, ?, ?, ?)",
  ).run(MACHINE_ID, T0, T0, "nvim", "linux");
  for (const [repoId, root] of [
    [REPO_A, "rustlings"],
    [REPO_B, "other-root"],
  ] as const) {
    db.query(
      "INSERT INTO repositories (repo_id, root_name, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?)",
    ).run(repoId, root, T0, T0);
  }
  db.query(
    "INSERT INTO sessions (session_id, machine_id, repo_id, started_at, ended_at, cwd_name, git_branch_at_start) VALUES (?, ?, ?, ?, ?, ?, ?)",
  ).run(SESSION_ID, MACHINE_ID, REPO_A, T0, null, "rustlings", "main");

  const fileInsert = db.query(
    "INSERT INTO files (repo_id, relative_path, extension, language, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?)",
  );
  fileInsert.run(REPO_A, REL_PATH, ".rs", "rust", T0, T0);
  const fileIdA = Number(
    (db.query<{ id: number }, []>("SELECT last_insert_rowid() AS id").get() as { id: number }).id,
  );
  // Duplicate path in second repo (for ambiguous-resolution coverage). No events attached.
  fileInsert.run(REPO_B, REL_PATH, ".rs", "rust", T0, T0);
  fileInsert.run(REPO_A, FILE2_REL, ".rs", "rust", T0, T0);
  const fileId2 = Number(
    (db.query<{ id: number }, []>("SELECT last_insert_rowid() AS id").get() as { id: number }).id,
  );

  // ---- synthetic structs-like content (20 lines, trailing newline) ----
  const initLines: string[] = [
    "// structs1 fixture (synthetic)",
    "struct Point {",
    "    x: i32,",
    "    y: i32,",
    "}",
    "",
    "fn main() {",
    "    let p = Point { x: 1, y: 2 };",
    '    println!("x={}", p.x);',
    "}",
    "",
    "fn dist(a: Point, b: Point) -> i32 {",
    "    (a.x - b.x).abs() + (a.y - b.y).abs()",
    "}",
    "",
    "// section alpha",
    "// section beta",
    "// section gamma",
    "// section delta",
    "// section epsilon",
  ];

  // Simulate edits on a working copy; record exact triples.
  // Edits are content-addressed (find marker rows) so row numbers stay
  // correct as the buffer evolves. Fixed initial content + fixed sequence
  // => fully deterministic.
  const joinText = (ls: string[]) => ls.join("\n") + "\n";
  const openText = joinText(initLines);
  const work: string[] = [...initLines];
  const idxOf = (needle: string): number => {
    const i = work.indexOf(needle);
    if (i < 0) throw new Error(`fixture marker missing: ${JSON.stringify(needle)} in ${JSON.stringify(work)}`);
    return i;
  };
  const firstEmpty = (): number => {
    const i = work.indexOf("");
    if (i < 0) throw new Error("fixture has no empty line");
    return i;
  };
  type RawOp = { start: number; oldEnd: number; ins: string[]; cursor: { row: number; col: number } | null; abs?: boolean };
  const steps: (() => RawOp)[] = [
    // 0 pure insertion: new println after the x println
    () => {
      const r = idxOf('    println!("x={}", p.x);') + 1;
      return { start: r, oldEnd: r, ins: ['    println!("y={}", p.y);'], cursor: { row: r, col: 4 } };
    },
    // 1 pure deletion: footer epsilon
    () => {
      const r = idxOf("// section epsilon");
      return { start: r, oldEnd: r + 1, ins: [], cursor: null };
    },
    // 2 replace x field
    () => {
      const r = idxOf("    x: i32,");
      return { start: r, oldEnd: r + 1, ins: ["    x: i64,"], cursor: { row: r, col: 10 } };
    },
    // 3 empty-line delete: real zero-text triple (deleted "" joins to "")
    () => {
      const r = firstEmpty();
      return { start: r, oldEnd: r + 1, ins: [], cursor: null };
    },
    // 4 undo-shaped replace: revert step 2
    () => {
      const r = idxOf("    x: i64,");
      return { start: r, oldEnd: r + 1, ins: ["    x: i32,"], cursor: { row: r, col: 10 } };
    },
    // 5 re-replace to final value
    () => {
      const r = idxOf("    x: i32,");
      return { start: r, oldEnd: r + 1, ins: ["    x: u32,"], cursor: { row: r, col: 10 } };
    },
    // 6 replace middle let line (will be undone next)
    () => {
      const r = idxOf("    let p = Point { x: 1, y: 2 };");
      return { start: r, oldEnd: r + 1, ins: ["    let p = Point { x: 1, y: 3 };"], cursor: { row: r, col: 8 } };
    },
    // 7 undo-shaped: revert step 6
    () => {
      const r = idxOf("    let p = Point { x: 1, y: 3 };");
      return { start: r, oldEnd: r + 1, ins: ["    let p = Point { x: 1, y: 2 };"], cursor: { row: r, col: 8 } };
    },
    // 8 insert a note line after the dist fn open (non-empty insert)
    () => {
      const r = idxOf("fn dist(a: Point, b: Point) -> i32 {") + 1;
      return { start: r, oldEnd: r, ins: ["    // dist note"], cursor: { row: r, col: 0 } };
    },
    // 9 delete the just-inserted note (pure deletion undoing step 8)
    () => {
      const r = idxOf("    // dist note");
      return { start: r, oldEnd: r + 1, ins: [], cursor: { row: r, col: 0 } };
    },
    // 10 replace footer beta
    () => {
      const r = idxOf("// section beta");
      return { start: r, oldEnd: r + 1, ins: ["// section beta (edited)"], cursor: { row: r, col: 5 } };
    },
    // 11 insert footer zeta at end
    () => {
      const r = work.length;
      return { start: r, oldEnd: r, ins: ["// section zeta"], cursor: { row: r, col: 3 } };
    },
    // 12 delete footer alpha
    () => {
      const r = idxOf("// section alpha");
      return { start: r, oldEnd: r + 1, ins: [], cursor: null, abs: true };
    },
    // 13 undo the step-0 insert (pure deletion in the middle; keeps middle net-zero)
    () => {
      const r = idxOf('    println!("y={}", p.y);');
      return { start: r, oldEnd: r + 1, ins: [], cursor: { row: r, col: 0 } };
    },
    // 14 replace y field (same struct hunk as step 5)
    () => {
      const r = idxOf("    y: i32,");
      return { start: r, oldEnd: r + 1, ins: ["    y: u32,"], cursor: { row: r, col: 10 } };
    },
    // 15 insert extra note after gamma (same footer hunk)
    () => {
      const r = idxOf("// section gamma") + 1;
      return { start: r, oldEnd: r, ins: ["// extra note"], cursor: { row: r, col: 2 } };
    },
  ];
  const recorded: { deleted: string; inserted: string; op: RawOp }[] = [];
  let midSnapshot = "";
  for (let i = 0; i < steps.length; i++) {
    const op = steps[i]!();
    const deleted = work.slice(op.start, op.oldEnd).join("\n");
    const inserted = op.ins.join("\n");
    work.splice(op.start, op.oldEnd - op.start, ...op.ins);
    recorded.push({ deleted, inserted, op });
    if (i === 9) midSnapshot = joinText(work);
  }
  const finalText = joinText(work);

  const deltaSpecs: DeltaSpec[] = [];
  let tick = 100;
  // timestamps: open=BASE, deltas0-9, write1, deltas10-15, write2
  const openTs = BASE_TS;
  const deltasTs: number[] = [];
  {
    let ts = BASE_TS;
    for (let i = 0; i < recorded.length; i++) {
      ts += 1000;
      if (i === 10) ts += 1000; // room for middle anchor (write1)
      deltasTs.push(ts);
    }
  }
  const write1Ts = BASE_TS + 11000;
  const write2Ts = deltasTs[deltasTs.length - 1]! + 1000;

  const contents = [openText, midSnapshot, finalText];
  const hashes: string[] = [];
  hashes.push(storeBlob(db, openText, "raw").sha);
  hashes.push(storeBlob(db, midSnapshot, "gzip").sha);
  hashes.push(storeBlob(db, finalText, "gzip").sha);

  const anchorTs = [openTs, write1Ts, write2Ts];

  // file_versions for file A
  const fv = db.query("INSERT INTO file_versions (file_id, sha256, observed_at, source) VALUES (?, ?, ?, ?)");
  for (const h of hashes) fv.run(fileIdA, h, T0, "anchor");

  // events: seq order = timestamp order
  let seq = 0;
  const insertEvent = (
    eventId: string,
    type: string,
    timestamp: number,
    payload: unknown,
    extra?: { changedtick?: number; cursor_row?: number | null; cursor_col?: number | null },
  ) => {
    db.query(
      "INSERT INTO events (event_id, session_id, sequence_number, event_type, timestamp_ms, file_id, changedtick, cursor_row, cursor_col, mode, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    ).run(
      eventId,
      SESSION_ID,
      seq++,
      type,
      timestamp,
      null,
      extra?.changedtick ?? null,
      extra?.cursor_row ?? null,
      extra?.cursor_col ?? null,
      "n",
      JSON.stringify(payload),
    );
  };

  // anchor open uses absolute path spelling; write anchors use relative (mix covers both)
  insertEvent("evt-anchor-open", "buffer_open", openTs, { path: ABS_PATH, content_hash: hashes[0] });
  for (let i = 0; i < recorded.length; i++) {
    const r = recorded[i]!;
    const op = r.op;
    const newEnd = op.start + op.ins.length;
    const path = op.abs ? ABS_PATH : REL_PATH;
    const spec: DeltaSpec = {
      start_row: op.start,
      old_end_row: op.oldEnd,
      new_end_row: newEnd,
      deleted_text: r.deleted,
      inserted_text: r.inserted,
      timestamp_ms: deltasTs[i]!,
      sequence_number: seq,
      cursor_after: op.cursor,
      path,
    };
    deltaSpecs.push(spec);
    insertEvent(`evt-delta-${i}`, "edit_delta", deltasTs[i]!, {
      path,
      start_row: op.start,
      old_end_row: op.oldEnd,
      new_end_row: newEnd,
      deleted_text: r.deleted,
      inserted_text: r.inserted,
      cursor_after: op.cursor,
      changedtick: tick++,
      bytecount: r.inserted.length,
    });
    if (i === 9) {
      insertEvent("evt-anchor-write1", "buffer_write", write1Ts, {
        path: REL_PATH,
        content_hash: hashes[1],
      });
    }
  }
  insertEvent("evt-anchor-write2", "buffer_write", write2Ts, { path: REL_PATH, content_hash: hashes[2] });

  // ---- second file: two anchors, no deltas, identical content (empty window) ----
  const file2Text = "// structs2 fixture (synthetic)\nstruct Empty {}\n";
  const file2Hash = storeBlob(db, file2Text, "raw").sha;
  fv.run(fileId2, file2Hash, T0, "anchor");
  // Note: these reuse seq numbers after file1 events; timestamps overlap but paths differ.
  insertEvent("evt-f2-open", "buffer_open", openTs, { path: FILE2_REL, content_hash: file2Hash });
  insertEvent("evt-f2-write", "buffer_write", write2Ts, { path: FILE2_ABS, content_hash: file2Hash });

  let grossInserted = 0;
  let grossDeleted = 0;
  for (const d of deltaSpecs) {
    grossInserted += d.inserted_text.length;
    grossDeleted += d.deleted_text.length;
  }

  db.close();

  return {
    dir,
    dbPath,
    contents,
    hashes,
    anchorTs,
    deltas: deltaSpecs,
    file2Hash,
    file2Ts: [openTs, write2Ts],
    grossInserted,
    grossDeleted,
    cleanup: () => rmSync(dir, { recursive: true, force: true }),
  };
}

const HERE = dirname(fileURLToPath(import.meta.url));
export const ANALYSIS_DIR = join(HERE, "..");

export interface RunResult {
  exit: number;
  stdout: string;
  stderr: string;
}

export async function runScript(script: string, args: string[]): Promise<RunResult> {
  const proc = Bun.spawn(["bun", join(ANALYSIS_DIR, script), ...args], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exit] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { exit, stdout, stderr };
}
