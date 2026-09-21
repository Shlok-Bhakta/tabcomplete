import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { Database } from "bun:sqlite";
import {
  blobBytes,
  blobText,
  fileAnchors,
  fileDeltas,
  lineOps,
  openReadOnly,
  opsToHunks,
  resolveFile,
  splitLines,
} from "../lib.ts";
import {
  ABS_PATH,
  FILE2_ABS,
  FILE2_REL,
  REL_PATH,
  REPO_A,
  REPO_B,
  buildFixture,
  type Fixture,
} from "./fixture.ts";

let f: Fixture;
let db: Database;

beforeEach(() => {
  f = buildFixture();
  db = openReadOnly(f.dbPath);
});

afterEach(() => {
  try {
    db.close();
  } finally {
    f.cleanup();
  }
});

describe("splitLines", () => {
  test('"" -> []', () => {
    expect(splitLines("")).toEqual([]);
  });
  test('"\\n" -> [""]', () => {
    expect(splitLines("\n")).toEqual([""]);
  });
  test("trailing newline stripped", () => {
    expect(splitLines("a\n")).toEqual(["a"]);
    expect(splitLines("a\nb\n")).toEqual(["a", "b"]);
  });
  test("no trailing newline kept whole", () => {
    expect(splitLines("a")).toEqual(["a"]);
    expect(splitLines("a\nb")).toEqual(["a", "b"]);
  });
  test("interior empty lines preserved", () => {
    expect(splitLines("a\n\nb\n")).toEqual(["a", "", "b"]);
  });
});

describe("lineOps/opsToHunks", () => {
  test("identical files -> 0 hunks", () => {
    const a = ["x", "y", "z"];
    expect(opsToHunks(lineOps(a, [...a]))).toEqual([]);
  });
  test("identical empty -> 0 hunks", () => {
    expect(opsToHunks(lineOps([], []))).toEqual([]);
  });
  test("pure insertion -> 1 hunk with + lines", () => {
    const hunks = opsToHunks(lineOps(["x"], ["x", "y"]));
    expect(hunks.length).toBe(1);
    expect(hunks[0]!.body.some((o) => o.t === "+")).toBe(true);
    expect(hunks[0]!.body.filter((o) => o.t === "-").length).toBe(0);
  });
  test("pure deletion -> 1 hunk with - lines", () => {
    const hunks = opsToHunks(lineOps(["x", "y"], ["x"]));
    expect(hunks.length).toBe(1);
    expect(hunks[0]!.body.some((o) => o.t === "-")).toBe(true);
    expect(hunks[0]!.body.filter((o) => o.t === "+").length).toBe(0);
  });
  test("far-apart changes -> 2 hunks; close changes merge -> 1 hunk", () => {
    const base = Array.from({ length: 20 }, (_, i) => `line ${i}`);
    const far = [...base];
    far[1] = "line 1 EDITED";
    far[18] = "line 18 EDITED";
    expect(opsToHunks(lineOps(base, far)).length).toBe(2);
    const near = [...base];
    near[5] = "line 5 EDITED";
    near[8] = "line 8 EDITED";
    expect(opsToHunks(lineOps(base, near)).length).toBe(1);
  });
});

describe("blobBytes", () => {
  test("refuses short prefixes", () => {
    expect(() => blobBytes(db, f.hashes[0]!.slice(0, 12))).toThrow();
    expect(() => blobBytes(db, "abc")).toThrow();
  });
  test("missing hash throws", () => {
    expect(() => blobBytes(db, "0".repeat(64))).toThrow();
  });
  test("raw + gzip roundtrip to fixture contents", () => {
    expect(blobText(db, f.hashes[0])).toBe(f.contents[0]);
    expect(blobText(db, f.hashes[1])).toBe(f.contents[1]);
    expect(blobText(db, f.hashes[2])).toBe(f.contents[2]);
    expect(blobText(db, f.file2Hash)).toBe("// structs2 fixture (synthetic)\nstruct Empty {}\n");
  });
});

describe("resolveFile", () => {
  test("ambiguous without --repo throws", () => {
    expect(() => resolveFile(db, { file: REL_PATH })).toThrow(/ambiguous/);
  });
  test("with --repo resolves exact repo", () => {
    const a = resolveFile(db, { file: REL_PATH, repo: REPO_A });
    expect(a.file.repo_id).toBe(REPO_A);
    expect(a.file.relative_path).toBe(REL_PATH);
    const b = resolveFile(db, { file: REL_PATH, repo: REPO_B });
    expect(b.file.repo_id).toBe(REPO_B);
  });
  test("absolute path resolves (unique file)", () => {
    const r = resolveFile(db, { file: FILE2_ABS });
    expect(r.file.relative_path).toBe(FILE2_REL);
  });
  test("unknown file throws", () => {
    expect(() => resolveFile(db, { file: "nope/missing.rs", repo: REPO_A })).toThrow(/no such file/);
  });
  test("prefers payload spellings for absPaths", () => {
    const r = resolveFile(db, { file: REL_PATH, repo: REPO_A });
    expect(r.absPaths).toContain(ABS_PATH);
    expect(r.absPaths).toContain(REL_PATH);
  });
});

describe("fileAnchors/fileDeltas", () => {
  test("ordering ascending and completeness (no truncation)", () => {
    const { file, absPaths } = resolveFile(db, { file: REL_PATH, repo: REPO_A });
    const anchors = fileAnchors(db, file.relative_path, absPaths);
    const deltas = fileDeltas(db, file.relative_path, absPaths);
    expect(anchors.length).toBe(3);
    expect(deltas.length).toBe(f.deltas.length);
    expect(deltas.length).toBe(16);
    for (let i = 1; i < anchors.length; i++) {
      expect(anchors[i]!.timestamp_ms).toBeGreaterThanOrEqual(anchors[i - 1]!.timestamp_ms);
    }
    for (let i = 1; i < deltas.length; i++) {
      const prev = deltas[i - 1]!;
      const cur = deltas[i]!;
      expect(
        cur.timestamp_ms > prev.timestamp_ms ||
          (cur.timestamp_ms === prev.timestamp_ms && cur.sequence_number > prev.sequence_number),
      ).toBe(true);
    }
    expect(anchors[0]!.event_type).toBe("buffer_open");
    expect(anchors[anchors.length - 1]!.event_type).toBe("buffer_write");
    // every inserted delta timestamp present exactly once
    expect(deltas.map((d) => d.timestamp_ms)).toEqual(f.deltas.map((d) => d.timestamp_ms));
  });
});
