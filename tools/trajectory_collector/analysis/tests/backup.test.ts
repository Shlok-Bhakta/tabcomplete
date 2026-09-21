import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { copyFileSync, existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, basename } from "node:path";
import { buildFixture, runScript, type Fixture } from "./fixture.ts";

let f: Fixture;
let scratch = "";
beforeEach(() => {
  f = buildFixture();
  scratch = mkdtempSync(join(tmpdir(), "backup-test-"));
});
afterEach(() => {
  try {
    f.cleanup();
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
});

describe("backup (WRITE script)", () => {
  test("copies sqlite, prints BACKUP-OK + WRITE banner, skips absent sidecars", async () => {
    const destRoot = join(scratch, "backups");
    const r = await runScript("backup.ts", ["--db", f.dbPath, "--dest", destRoot]);
    expect(r.exit).toBe(0);
    expect(r.stdout).toContain("*** WRITE SCRIPT");
    expect(r.stdout).toContain("BACKUP-OK");
    expect(r.stdout).toContain("skip (absent)");
    const dirs = readdirSync(destRoot);
    expect(dirs.length).toBe(1);
    const dest = join(destRoot, dirs[0]!);
    const got = readdirSync(dest);
    expect(got).toContain(basename(f.dbPath));
    // byte-identical copy of the main db file
    const srcBytes = Bun.file(f.dbPath).size;
    const dstBytes = Bun.file(join(dest, basename(f.dbPath))).size;
    expect(dstBytes).toBe(srcBytes);
  });

  test("copies wal+shm sidecars when present", async () => {
    // Fixture DB is never production: copy it aside and add fake sidecars.
    const dbCopy = join(scratch, "copy.sqlite");
    copyFileSync(f.dbPath, dbCopy);
    writeFileSync(dbCopy + "-wal", "fake-wal-bytes");
    writeFileSync(dbCopy + "-shm", "fake-shm-bytes");
    const destRoot = join(scratch, "backups2");
    const r = await runScript("backup.ts", ["--db", dbCopy, "--dest", destRoot]);
    expect(r.exit).toBe(0);
    expect(r.stdout).toContain("BACKUP-OK");
    const dirs = readdirSync(destRoot);
    const dest = join(destRoot, dirs[0]!);
    const got = readdirSync(dest).sort();
    expect(got).toEqual(["copy.sqlite", "copy.sqlite-shm", "copy.sqlite-wal"].sort());
    expect(r.stdout).not.toContain("skip (absent)");
    expect(existsSync(join(dest, "copy.sqlite-wal"))).toBe(true);
    expect(existsSync(join(dest, "copy.sqlite-shm"))).toBe(true);
  });
});
