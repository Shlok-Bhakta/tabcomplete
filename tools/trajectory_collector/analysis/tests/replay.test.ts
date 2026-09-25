import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { Database } from "bun:sqlite";
import { REL_PATH, REPO_A, buildFixture, runScript, type Fixture } from "./fixture.ts";

let f: Fixture;
beforeEach(() => {
  f = buildFixture();
});
afterEach(() => {
  f.cleanup();
});

describe("replay", () => {
  test("REPLAY-OK on fixture (exit 0)", async () => {
    const r = await runScript("replay.ts", [
      "--file", REL_PATH, "--repo", REPO_A, "--session", "sess-fixture-0001",
      "--db", f.dbPath,
    ]);
    expect(r.exit).toBe(0);
    expect(r.stdout).toContain("REPLAY-OK");
    expect(r.stdout).toContain("unanchored: 0");
  });

  test("planted phantom zero-text deletion -> REPLAY-FAIL with suspect ts/seq", async () => {
    // Plant a zero-text deletion where the live buffer really holds "" so
    // replay accepts it and splices, shifting every later row (the README
    // phantom signature). Timestamp slots between deltas[5] and deltas[6].
    const plantTs = f.deltas[5]!.timestamp_ms + 500;
    const plantSeq = 1000;
    const wb = new Database(f.dbPath);
    // Buffer row 10 is "" at that point in history (verified in fixture sim).
    wb.query(
      "INSERT INTO events (event_id, session_id, sequence_number, event_type, timestamp_ms, file_id, changedtick, cursor_row, cursor_col, mode, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    ).run(
      "evt-phantom",
      "sess-fixture-0001",
      plantSeq,
      "edit_delta",
      plantTs,
      null,
      999,
      null,
      null,
      "n",
      JSON.stringify({
        path: REL_PATH,
        start_row: 10,
        old_end_row: 11,
        new_end_row: 10,
        deleted_text: "",
        inserted_text: "",
        cursor_after: null,
        changedtick: 999,
        bytecount: 0,
      }),
    );
    wb.close();
    const r = await runScript("replay.ts", ["--file", REL_PATH, "--repo", REPO_A, "--db", f.dbPath]);
    expect(r.exit).toBe(1);
    expect(r.stdout).toContain("REPLAY-FAIL");
    expect(r.stdout).toContain(`ts=${plantTs}`);
    expect(r.stdout).toContain(`seq=${plantSeq}`);
  });

  test("corrupted deleted_text -> REPLAY-FAIL", async () => {
    const wb = new Database(f.dbPath);
    const row = wb
      .query("SELECT payload_json AS p FROM events WHERE event_id = 'evt-delta-2'")
      .get() as { p: string };
    const payload = JSON.parse(row.p);
    payload.deleted_text = "CORRUPTED-NEVER-IN-BUFFER";
    wb.query("UPDATE events SET payload_json = ? WHERE event_id = 'evt-delta-2'").run(
      JSON.stringify(payload),
    );
    wb.close();
    const r = await runScript("replay.ts", ["--file", REL_PATH, "--repo", REPO_A, "--db", f.dbPath]);
    expect(r.exit).toBe(1);
    expect(r.stdout).toContain("REPLAY-FAIL");
  });
});
