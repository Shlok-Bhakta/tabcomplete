import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { REL_PATH, REPO_A, buildFixture, runScript, type Fixture } from "./fixture.ts";

let f: Fixture;
beforeEach(() => {
  f = buildFixture();
});
afterEach(() => {
  f.cleanup();
});

describe("netdiff", () => {
  test("2-hunk output with exact gross-vs-net stats", async () => {
    const r = await runScript("netdiff.ts", ["--file", REL_PATH, "--repo", REPO_A, "--db", f.dbPath]);
    expect(r.exit).toBe(0);
    // all hunks present
    const atCount = (r.stdout.match(/^@@ /gm) ?? []).length;
    expect(atCount).toBe(2);
    expect(r.stdout).toContain("# hunks: 2");
    // hand-computed gross stats from the fixture triples
    let expIns = 0;
    let expDel = 0;
    for (const d of f.deltas) {
      expIns += d.inserted_text.length;
      expDel += d.deleted_text.length;
    }
    expect(expIns).toBe(f.grossInserted);
    expect(expDel).toBe(f.grossDeleted);
    expect(r.stdout).toContain(
      `gross inserted chars: ${expIns}  gross deleted chars: ${expDel}`,
    );
    const beforeLen = f.contents[0]!.length;
    const afterLen = f.contents[f.contents.length - 1]!.length;
    const net = afterLen - beforeLen;
    expect(r.stdout).toContain(`# net bytes: ${beforeLen} -> ${afterLen} (${net >= 0 ? "+" : ""}${net})`);
    expect(r.stdout).toContain(`--- a/${REL_PATH}`);
    expect(r.stdout).toContain(`+++ b/${REL_PATH}`);
  });

  test("missing file exits nonzero", async () => {
    const r = await runScript("netdiff.ts", [
      "--file",
      "nope/missing.rs",
      "--repo",
      REPO_A,
      "--db",
      f.dbPath,
    ]);
    expect(r.exit).not.toBe(0);
  });
});
