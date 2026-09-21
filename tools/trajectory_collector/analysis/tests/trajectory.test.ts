import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { Database } from "bun:sqlite";
import { SESSION_ID, buildFixture, runScript, type Fixture } from "./fixture.ts";

let f: Fixture;
beforeEach(() => {
  f = buildFixture();
});
afterEach(() => {
  f.cleanup();
});

function addSession(dbPath: string, sessionId: string, startedAt: string, endedAt: string | null) {
  const db = new Database(dbPath);
  db.query(
    "INSERT INTO sessions (session_id, machine_id, repo_id, started_at, ended_at, cwd_name, git_branch_at_start) VALUES (?, ?, ?, ?, ?, ?, ?)",
  ).run(sessionId, "machine-fixture-1", "repo:rustlings", startedAt, endedAt, "rustlings", "main");
  db.close();
}

describe("trajectory", () => {
  test("unique prefix resolves; counts and span math match", async () => {
    const r = await runScript("trajectory.ts", ["--session", SESSION_ID, "--db", f.dbPath]);
    expect(r.exit).toBe(0);
    expect(r.stdout).toContain(`# session: ${SESSION_ID}`);
    // counts match inserted data: 16 deltas, 2 opens, 3 writes in this session
    expect(r.stdout).toContain("edit_delta: 16");
    expect(r.stdout).toContain("buffer_open: 2");
    expect(r.stdout).toContain("buffer_write: 3");
    // span math: lo = BASE open ts, hi = final write ts
    const lo = f.anchorTs[0]!;
    const hi = f.anchorTs[f.anchorTs.length - 1]!;
    const exp = `# span: ${new Date(lo).toISOString()} -> ${new Date(hi).toISOString()} (${((hi - lo) / 1000).toFixed(1)}s)`;
    expect(r.stdout).toContain(exp);
    // short unique prefix also resolves
    const short = await runScript("trajectory.ts", [
      "--session",
      SESSION_ID.slice(0, 12),
      "--db",
      f.dbPath,
    ]);
    expect(short.exit).toBe(0);
    expect(short.stdout).toContain(`# session: ${SESSION_ID}`);
  });

  test("ambiguous prefix errors", async () => {
    addSession(f.dbPath, "sess-fixture-0002", "2023-11-14T23:00:00.000Z", null);
    const r = await runScript("trajectory.ts", ["--session", "sess-fixture", "--db", f.dbPath]);
    expect(r.exit).not.toBe(0);
    expect(r.stderr).toContain("matched 2 sessions");
  });

  test("unknown prefix errors", async () => {
    const r = await runScript("trajectory.ts", ["--session", "zzz-nope", "--db", f.dbPath]);
    expect(r.exit).not.toBe(0);
    expect(r.stderr).toContain("matched 0 sessions");
  });

  test("--live picks the open session (latest open)", async () => {
    addSession(f.dbPath, "sess-fixture-0002", "2023-11-14T23:00:00.000Z", null);
    // NOTE: --live must come last: parseArgs consumes the next token as the
    // flag value, so `--live --db PATH` mis-parses (reported as analysis bug).
    const r = await runScript("trajectory.ts", ["--db", f.dbPath, "--live"]);
    expect(r.exit).toBe(0);
    expect(r.stdout).toContain("# session: sess-fixture-0002");
  });
});
