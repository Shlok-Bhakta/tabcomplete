import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { gunzipSync } from "node:zlib";
import { blobText, openReadOnly } from "../lib.ts";
import {
  FILE2_REL,
  REL_PATH,
  REPO_A,
  buildFixture,
  runScript,
  type Fixture,
} from "./fixture.ts";

let f: Fixture;
beforeEach(() => {
  f = buildFixture();
});
afterEach(() => {
  f.cleanup();
});

interface ExampleRecord {
  protocol: string;
  file: string;
  repo_id: string;
  window: {
    from: { ts: number; anchor: string; sha256: string };
    to: { ts: number; anchor: string; sha256: string };
    span_s: number;
  };
  state: { content: string; cursor: { row: number; col: number } | null };
  recent_history: {
    ts: number;
    start_row: number;
    old_end_row: number;
    new_end_row: number;
    deleted_text: string;
    inserted_text: string;
    cursor_after: { row: number; col: number } | null;
  }[];
  action: {
    hunks: { aStart: number; aCount: number; bStart: number; bCount: number; lines: { t: string; line: string }[] }[];
  };
  stats: {
    gross_deltas: number;
    gross_inserted_chars: number;
    gross_deleted_chars: number;
    net_bytes_from: number;
    net_bytes_to: number;
  };
}

/** Reconstruct b-side content by walking state lines through ordered hunks. */
function applyHunks(stateContent: string, hunks: ExampleRecord["action"]["hunks"]): string {
  const trailing = stateContent.endsWith("\n");
  const aLines = trailing ? stateContent.slice(0, -1).split("\n") : stateContent.split("\n");
  if (stateContent === "") return "";
  const out: string[] = [];
  let aPos = 1; // 1-based next unconsumed a line
  const sorted = [...hunks].sort((x, y) => x.aStart - y.aStart);
  for (const h of sorted) {
    for (let n = aPos; n < h.aStart; n++) out.push(aLines[n - 1]!);
    for (const l of h.lines) if (l.t !== "-") out.push(l.line);
    aPos = h.aStart + h.aCount;
  }
  for (let n = aPos; n <= aLines.length; n++) out.push(aLines[n - 1]!);
  return out.join("\n") + (trailing ? "\n" : "");
}

describe("examples", () => {
  test("window count == anchors-1 minus empty windows", async () => {
    const r = await runScript("examples.ts", ["--file", REL_PATH, "--repo", REPO_A, "--db", f.dbPath]);
    expect(r.exit).toBe(0);
    const records = r.stdout.trim().split("\n").filter(Boolean).map((l) => JSON.parse(l) as ExampleRecord);
    // 3 anchors, both windows non-empty (verified hunks > 0) => 2 records
    expect(records.length).toBe(2);
    // second file: 2 identical anchors, no deltas => empty window skipped => 0 records
    const r2 = await runScript("examples.ts", ["--file", FILE2_REL, "--db", f.dbPath]);
    expect(r2.exit).toBe(0);
    expect(r2.stdout.trim()).toBe("");
  });

  test("records parse with required keys; stats exact; hunks reconstruct end", async () => {
    const r = await runScript("examples.ts", ["--file", REL_PATH, "--repo", REPO_A, "--db", f.dbPath]);
    expect(r.exit).toBe(0);
    const records = r.stdout.trim().split("\n").filter(Boolean).map((l) => JSON.parse(l) as ExampleRecord);
    const db = openReadOnly(f.dbPath);
    try {
      for (const rec of records) {
        expect(rec.protocol).toBe("trajectory-example/v1");
        expect(rec.file).toBe(REL_PATH);
        expect(rec.repo_id).toBe(REPO_A);
        expect(typeof rec.window.from.ts).toBe("number");
        expect(typeof rec.window.to.ts).toBe("number");
        expect(typeof rec.window.span_s).toBe("number");
        expect(typeof rec.state.content).toBe("string");
        expect(Array.isArray(rec.recent_history)).toBe(true);
        expect(Array.isArray(rec.action.hunks)).toBe(true);
        expect(rec.action.hunks.length).toBeGreaterThan(0);
        // stats: hand-computed from fixture triples in this window
        const inWin = f.deltas.filter((d) => d.timestamp_ms > rec.window.from.ts && d.timestamp_ms <= rec.window.to.ts);
        let expIns = 0;
        let expDel = 0;
        for (const d of inWin) {
          expIns += d.inserted_text.length;
          expDel += d.deleted_text.length;
        }
        expect(rec.stats.gross_deltas).toBe(inWin.length);
        expect(rec.stats.gross_inserted_chars).toBe(expIns);
        expect(rec.stats.gross_deleted_chars).toBe(expDel);
        expect(rec.stats.net_bytes_from).toBe(rec.state.content.length);
        // end content via blob lookup
        const row = db.query("SELECT content, compression FROM blobs WHERE sha256 = ?").get(rec.window.to.sha256) as {
          content: Uint8Array;
          compression: string;
        } | null;
        expect(row).not.toBeNull();
        const raw = row!.content as unknown as Buffer;
        const endContent =
          row!.compression === "gzip" ? gunzipSync(raw).toString("utf8") : Buffer.from(raw).toString("utf8");
        // cross-check lib decoder agrees
        expect(blobText(db, rec.window.to.sha256)).toBe(endContent);
        expect(rec.stats.net_bytes_to).toBe(endContent.length);
        // action hunks reconstruct end content byte-exact
        expect(applyHunks(rec.state.content, rec.action.hunks)).toBe(endContent);
        // span math
        expect(rec.window.span_s).toBe(+((rec.window.to.ts - rec.window.from.ts) / 1000).toFixed(1));
      }
    } finally {
      db.close();
    }
  });

  test("history bound respected; cursor from last pre-window delta", async () => {
    const r = await runScript("examples.ts", [
      "--file",
      REL_PATH,
      "--repo",
      REPO_A,
      "--db",
      f.dbPath,
      "--history",
      "3",
    ]);
    expect(r.exit).toBe(0);
    const records = r.stdout.trim().split("\n").filter(Boolean).map((l) => JSON.parse(l) as ExampleRecord);
    expect(records.length).toBe(2);
    for (const rec of records) expect(rec.recent_history.length).toBeLessThanOrEqual(3);
    // window 2 has 10 pre-window deltas -> truncated to exactly 3
    expect(records[1]!.recent_history.length).toBe(3);
    // cursor = last pre-window delta carrying cursor_after (or null)
    for (const rec of records) {
      const pre = f.deltas.filter((d) => d.timestamp_ms <= rec.window.from.ts);
      const src = [...pre].reverse().find((d) => d.cursor_after);
      expect(rec.state.cursor).toEqual(src ? src.cursor_after : null);
    }
    // first window has no history -> null cursor
    expect(records[0]!.recent_history).toEqual([]);
    expect(records[0]!.state.cursor).toBeNull();
  });
});
