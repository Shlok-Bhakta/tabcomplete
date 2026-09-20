import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { gunzipSync } from "node:zlib";
import { createApp, type CollectorApp } from "../src/index";
import { migrate } from "../src/migrations";
import { MAX_BODY_BYTES } from "../src/types";

function sha256HexText(s: string): string {
  return createHash("sha256").update(s, "utf8").digest("hex");
}

let dir = "";
let app: CollectorApp | null = null;

function freshApp(): CollectorApp {
  dir = mkdtempSync(join(tmpdir(), "collector-test-"));
  const dbPath = join(dir, "test.sqlite");
  return createApp(dbPath);
}

beforeEach(() => {
  app = freshApp();
});

afterEach(() => {
  try {
    app?.close();
  } finally {
    app = null;
    if (dir !== "") {
      rmSync(dir, { recursive: true, force: true });
      dir = "";
    }
  }
});

function post(path: string, body: unknown, extraHeaders?: Record<string, string>): Promise<Response> {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return app!.fetch(
    new Request(`http://localhost${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", ...(extraHeaders ?? {}) },
      body: text,
    }),
  );
}

function get(path: string): Promise<Response> {
  return app!.fetch(new Request(`http://localhost${path}`, { method: "GET" }));
}

async function bodyJson(res: Response): Promise<any> {
  return (await res.json()) as any;
}

const SESSION_BASE = {
  protocol_version: 1,
  session_id: "sess-1",
  machine_id: "machine-1",
  started_at: "2026-09-20T00:00:00.000Z",
  cwd_name: "tabcomplete",
  editor: "nvim",
  editor_version: "0.11",
  plugin_version: "0.1.0",
  platform: "linux",
  hostname_hash: "abc123",
};

function evt(seq: number, overrides: Record<string, unknown> = {}) {
  return {
    protocol_version: 1,
    event_id: `evt-${seq}`,
    session_id: "sess-1",
    sequence_number: seq,
    timestamp_ms: 1000 + seq,
    event_type: "cursor_move",
    cursor: { row: seq, col: 0 },
    mode: "n",
    payload: {},
    ...overrides,
  };
}

describe("healthz + stats", () => {
  test("healthz returns ok", async () => {
    const res = await get("/healthz");
    expect(res.status).toBe(200);
    const body: any = await bodyJson(res);
    expect(body.status).toBe("ok");
    expect(body.protocol_version).toBe(1);
  });

  test("stats returns zeroed counters on fresh db", async () => {
    const res = await get("/v1/stats");
    expect(res.status).toBe(200);
    const s: any = await bodyJson(res);
    expect(s.event_count).toBe(0);
    expect(s.session_count).toBe(0);
    expect(s.repository_count).toBe(0);
    expect(s.file_count).toBe(0);
    expect(s.file_version_count).toBe(0);
    expect(s.blob_count).toBe(0);
    expect(typeof s.db_bytes).toBe("number");
    expect(typeof s.wal_bytes).toBe("number");
    expect(s.compression_ratio).toBeNull();
  });
});

describe("sessions", () => {
  test("session create and end (missing ended_at tolerated)", async () => {
    const start = await post("/v1/session/start", SESSION_BASE);
    expect([200, 201]).toContain(start.status);
    const started: any = await bodyJson(start);
    expect(started.session_id).toBe("sess-1");

    const end = await post("/v1/session/end", { protocol_version: 1, session_id: "sess-1" });
    expect(end.status).toBe(200);
    const ended: any = await bodyJson(end);
    expect(ended.session_id).toBe("sess-1");
    expect(typeof ended.ended_at).toBe("string");

    const row = app!.db
      .query<{ ended_at: string }, [string]>("SELECT ended_at FROM sessions WHERE session_id = ?")
      .get("sess-1");
    expect(row?.ended_at).toBe(ended.ended_at);
  });

  test("session end with explicit ended_at", async () => {
    await post("/v1/session/start", SESSION_BASE);
    const end = await post("/v1/session/end", {
      protocol_version: 1,
      session_id: "sess-1",
      ended_at: "2026-09-20T01:00:00.000Z",
    });
    expect(end.status).toBe(200);
    expect((await bodyJson(end)).ended_at).toBe("2026-09-20T01:00:00.000Z");
  });

  test("malformed session start rejected", async () => {
    const missingId = await post("/v1/session/start", { protocol_version: 1 });
    expect(missingId.status).toBe(400);

    const badVersion = await post("/v1/session/start", { ...SESSION_BASE, protocol_version: 2 });
    expect(badVersion.status).toBe(400);

    const badJson = await app!.fetch(
      new Request("http://localhost/v1/session/start", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{not json",
      }),
    );
    expect(badJson.status).toBe(400);
  });

  test("session end for unknown session is 404", async () => {
    const res = await post("/v1/session/end", { protocol_version: 1, session_id: "nope" });
    expect(res.status).toBe(404);
  });
});

describe("events batch", () => {
  test("ingest preserves sequence order", async () => {
    await post("/v1/session/start", SESSION_BASE);
    const batch = [evt(0), evt(1), evt(2)];
    const res = await post("/v1/events/batch", { protocol_version: 1, events: batch });
    expect(res.status).toBe(200);
    const body: any = await bodyJson(res);
    expect(body.ingested).toBe(3);
    expect(body.total).toBe(3);

    const rows = app!.db
      .query<{ sequence_number: number }, [string]>(
        "SELECT sequence_number FROM events WHERE session_id = ? ORDER BY sequence_number ASC",
      )
      .all("sess-1")
      .map((r) => r.sequence_number);
    expect(rows).toEqual([0, 1, 2]);
  });

  test("duplicate batch is idempotent", async () => {
    await post("/v1/session/start", SESSION_BASE);
    const batch = { protocol_version: 1, events: [evt(0), evt(1)] };
    const first = await post("/v1/events/batch", batch);
    expect((await bodyJson(first)).ingested).toBe(2);
    const second = await post("/v1/events/batch", batch);
    const secondBody: any = await bodyJson(second);
    expect(second.status).toBe(200);
    expect(secondBody.ingested).toBe(0);
    expect(secondBody.skipped_duplicate).toBe(2);

    const count = app!.db.query<{ n: number }, []>("SELECT COUNT(*) AS n FROM events").get();
    expect(count?.n).toBe(2);
  });

  test("edit_delta payload validated (zero-based ranges)", async () => {
    await post("/v1/session/start", SESSION_BASE);
    const good = evt(0, {
      event_id: "evt-delta",
      event_type: "edit_delta",
      payload: {
        start_row: 0,
        old_end_row: 2,
        new_end_row: 3,
        deleted_text: "a\nb\n",
        inserted_text: "a\nb\nc\n",
        cursor_after: { row: 3, col: 0 },
      },
    });
    const okRes = await post("/v1/events/batch", { protocol_version: 1, events: [good] });
    expect(okRes.status).toBe(200);

    const bad = evt(1, {
      event_id: "evt-bad",
      event_type: "edit_delta",
      payload: { start_row: -1, old_end_row: 0, new_end_row: 0 },
    });
    const badRes = await post("/v1/events/batch", { protocol_version: 1, events: [bad] });
    expect(badRes.status).toBe(400);
  });

  test("batch for unknown session is 404", async () => {
    const res = await post("/v1/events/batch", { protocol_version: 1, events: [evt(0)] });
    expect(res.status).toBe(404);
  });

  test("malformed event payload is 4xx", async () => {
    await post("/v1/session/start", SESSION_BASE);
    const res = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [{ ...evt(0), event_type: "not_a_real_type" }],
    });
    expect(res.status).toBe(400);
  });
});

describe("blobs", () => {
  test("check reports missing, upload stores, duplicate stored once + gzip roundtrip", async () => {
    const text = "hello trajectory collector\n".repeat(100);
    const hash = sha256HexText(text);
    const b64 = Buffer.from(text, "utf8").toString("base64");

    const check1 = await post("/v1/blobs/check", { hashes: [hash] });
    expect(check1.status).toBe(200);
    expect((await bodyJson(check1)).missing).toEqual([hash]);

    const up1 = await post("/v1/blobs/upload", { sha256: hash, content_base64: b64 });
    expect(up1.status).toBe(200);
    const up1Body: any = await bodyJson(up1);
    expect(up1Body.sha256).toBe(hash);
    expect(up1Body.deduped).toBe(false);
    expect(up1Body.compression).toBe("gzip");
    expect(up1Body.original_bytes).toBe(Buffer.byteLength(text));
    expect(up1Body.stored_bytes).toBeLessThan(up1Body.original_bytes);

    // Gzip roundtrip straight from SQLite.
    const row = app!.db
      .query<{ content: Uint8Array; compression: string }, [string]>(
        "SELECT content, compression FROM blobs WHERE sha256 = ?",
      )
      .get(hash);
    expect(row?.compression).toBe("gzip");
    const decoded = gunzipSync(row!.content as unknown as Buffer).toString("utf8");
    expect(decoded).toBe(text);

    // Duplicate upload: stored once.
    const up2 = await post("/v1/blobs/upload", { sha256: hash, content_base64: b64 });
    const up2Body: any = await bodyJson(up2);
    expect(up2Body.deduped).toBe(true);
    expect(up2Body.stored_bytes).toBe(up1Body.stored_bytes);
    const count = app!.db.query<{ n: number }, []>("SELECT COUNT(*) AS n FROM blobs").get();
    expect(count?.n).toBe(1);

    const check2 = await post("/v1/blobs/check", { hashes: [hash] });
    expect((await bodyJson(check2)).missing).toEqual([]);
  });

  test("already-compressed bytes stored raw", async () => {
    // Minimal PNG header + filler.
    const png = Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      Buffer.alloc(64, 7),
    ]);
    const hash = createHash("sha256").update(png).digest("hex");
    const res = await post("/v1/blobs/upload", {
      sha256: hash,
      content_base64: png.toString("base64"),
    });
    expect(res.status).toBe(200);
    expect((await bodyJson(res)).compression).toBe("raw");
  });

  test("sha mismatch rejected", async () => {
    const res = await post("/v1/blobs/upload", {
      sha256: "0".repeat(64),
      content_base64: Buffer.from("x").toString("base64"),
    });
    expect(res.status).toBe(400);
  });
});

describe("repository snapshot", () => {
  test("inserts files + versions, dedups identical versions, sanitizes origin", async () => {
    const content = "print('hi')\n";
    const hash = sha256HexText(content);
    const snap = {
      protocol_version: 1,
      repo_id: "repo-1",
      root_name: "tabcomplete",
      origin_url: "https://user:s3cret@github.com/example/tabcomplete.git",
      files: [
        { relative_path: "src/a.py", extension: ".py", language: "python", sha256: hash, source: "repo_scan" },
        { relative_path: "src/b.py", extension: ".py", language: "python", sha256: hash, source: "repo_scan" },
      ],
    };
    const res = await post("/v1/repository/snapshot", snap);
    expect(res.status).toBe(200);
    const body: any = await bodyJson(res);
    expect(body.files_upserted).toBe(2);
    expect(body.versions_inserted).toBe(2);

    const repo = app!.db
      .query<{ origin_url: string }, [string]>("SELECT origin_url FROM repositories WHERE repo_id = ?")
      .get("repo-1");
    expect(repo?.origin_url).not.toContain("s3cret");
    expect(repo?.origin_url).not.toContain("user@");
    expect(repo?.origin_url).toContain("github.com");

    const fileCount = app!.db.query<{ n: number }, []>("SELECT COUNT(*) AS n FROM files").get();
    expect(fileCount?.n).toBe(2);

    // Re-snapshot identical content: versions skipped as duplicates.
    const res2 = await post("/v1/repository/snapshot", snap);
    const body2: any = await bodyJson(res2);
    expect(body2.versions_inserted).toBe(0);
    expect(body2.versions_skipped_duplicate).toBe(2);
    const versionCount = app!.db
      .query<{ n: number }, []>("SELECT COUNT(*) AS n FROM file_versions")
      .get();
    expect(versionCount?.n).toBe(2);
  });

  test("malformed snapshot is 4xx", async () => {
    const res = await post("/v1/repository/snapshot", { protocol_version: 1, files: [] });
    expect(res.status).toBe(400);
  });
});

describe("limits + migrations", () => {
  test("oversized body returns 413", async () => {
    await post("/v1/session/start", SESSION_BASE);
    const big = "x".repeat(MAX_BODY_BYTES + 1024);
    const res = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [{ ...evt(0), payload: { blob: big } }],
    });
    expect(res.status).toBe(413);
  });

  test("migrations are clean and idempotent + pragmas set", async () => {
    const first = migrate(app!.db);
    expect(first.applied).toEqual([]);
    const second = migrate(app!.db);
    expect(second.applied).toEqual([]);

    const journal = app!.db.query<{ journal_mode: string }, []>("PRAGMA journal_mode").get();
    expect(journal?.journal_mode.toLowerCase()).toBe("wal");
    const fk = app!.db.query<{ foreign_keys: number }, []>("PRAGMA foreign_keys").get();
    expect(fk?.foreign_keys).toBe(1);

    const tables = app!.db
      .query<{ name: string }, []>(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
      )
      .all()
      .map((r) => r.name);
    for (const t of ["machines", "repositories", "sessions", "files", "blobs", "file_versions", "events"]) {
      expect(tables).toContain(t);
    }
  });

  test("unknown route is 404 with structured error", async () => {
    const res = await get("/nope");
    expect(res.status).toBe(404);
    const body: any = await bodyJson(res);
    expect(typeof body.error).toBe("string");
  });

  test("string file_id accepted and stored as null (legacy client ids)", async () => {
    await post("/v1/session/start", SESSION_BASE);
    const res = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [evt(0, { event_id: "evt-str-fid", file_id: "repo:tabcomplete:src/a.py" })],
    });
    expect(res.status).toBe(200);
    expect((await bodyJson(res)).ingested).toBe(1);
    const row = app!.db
      .query<{ file_id: number | null }, [string]>("SELECT file_id FROM events WHERE event_id = ?")
      .get("evt-str-fid");
    expect(row?.file_id).toBeNull();
  });

  test("integer file_id stored verbatim; batch without protocol_version is 400", async () => {
    await post("/v1/session/start", SESSION_BASE);
    const res = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [evt(0, { event_id: "evt-int-fid", file_id: 7 })],
    });
    expect(res.status).toBe(200);
    const row = app!.db
      .query<{ file_id: number | null }, [string]>("SELECT file_id FROM events WHERE event_id = ?")
      .get("evt-int-fid");
    expect(row?.file_id).toBe(7);

    const noVer = await post("/v1/events/batch", { events: [evt(1, { event_id: "evt-no-ver" })] });
    expect(noVer.status).toBe(400);
  });
});
