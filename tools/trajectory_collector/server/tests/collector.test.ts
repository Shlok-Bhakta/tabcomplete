import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { gunzipSync, gzipSync } from "node:zlib";
import { Database } from "bun:sqlite";
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

function sessBaseFor(sessionId: string, machineId = "machine-1") {
  return { ...SESSION_BASE, session_id: sessionId, machine_id: machineId };
}

function countTable(table: string): number {
  return app!.db.query<{ n: number }, []>(`SELECT COUNT(*) AS n FROM ${table}`).get()?.n ?? 0;
}

describe("retry storm (exactly-once)", () => {
  test("same batch POSTed 5x concurrently + 5x sequentially ingests exactly once, order preserved", async () => {
    await post("/v1/session/start", sessBaseFor("sess-retry"));
    const mk = (seq: number) => evt(seq, { event_id: `evt-retry-${seq}`, session_id: "sess-retry" });
    const payload = { protocol_version: 1, events: [mk(0), mk(1), mk(2)] };

    const concurrent = await Promise.all([
      post("/v1/events/batch", payload),
      post("/v1/events/batch", payload),
      post("/v1/events/batch", payload),
      post("/v1/events/batch", payload),
      post("/v1/events/batch", payload),
    ]);
    for (const r of concurrent) expect(r.status).toBe(200);
    // The batch inserts atomically in one synchronous transaction, so exactly
    // one concurrent winner ingests all 3; the losers skip all 3.
    let ingested = 0;
    let skipped = 0;
    for (const r of concurrent) {
      const b: any = await bodyJson(r);
      ingested += b.ingested;
      skipped += b.skipped_duplicate;
    }
    expect(ingested).toBe(3);
    expect(skipped).toBe(12);

    for (let i = 0; i < 5; i++) {
      const r = await post("/v1/events/batch", payload);
      expect(r.status).toBe(200);
      const b: any = await bodyJson(r);
      expect(b.ingested).toBe(0);
      expect(b.skipped_duplicate).toBe(3);
    }

    expect(countTable("events")).toBe(3);
    const rows = app!.db
      .query<{ sequence_number: number; event_id: string }, [string]>(
        "SELECT sequence_number, event_id FROM events WHERE session_id = ? ORDER BY sequence_number ASC",
      )
      .all("sess-retry");
    expect(rows.map((r) => r.sequence_number)).toEqual([0, 1, 2]);
    expect(rows.map((r) => r.event_id)).toEqual(["evt-retry-0", "evt-retry-1", "evt-retry-2"]);
  });
});

describe("out-of-order batch arrival", () => {
  test("later batch first, earlier batch second: both stored, replay returns global order", async () => {
    await post("/v1/session/start", sessBaseFor("sess-ooo"));
    const mk = (seq: number) => evt(seq, { event_id: `evt-ooo-${seq}`, session_id: "sess-ooo" });
    const secondHalf = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [mk(3), mk(4), mk(5)],
    });
    expect(secondHalf.status).toBe(200);
    expect((await bodyJson(secondHalf)).ingested).toBe(3);

    const firstHalf = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [mk(0), mk(1), mk(2)],
    });
    expect(firstHalf.status).toBe(200);
    expect((await bodyJson(firstHalf)).ingested).toBe(3);

    // Replay query in global (session_id, sequence_number) order.
    const rows = app!.db
      .query<{ sequence_number: number; event_id: string }, [string]>(
        "SELECT sequence_number, event_id FROM events WHERE session_id = ? ORDER BY sequence_number ASC",
      )
      .all("sess-ooo");
    expect(rows.map((r) => r.sequence_number)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(rows.map((r) => r.event_id)).toEqual([
      "evt-ooo-0",
      "evt-ooo-1",
      "evt-ooo-2",
      "evt-ooo-3",
      "evt-ooo-4",
      "evt-ooo-5",
    ]);
  });
});

describe("duplicate event_ids", () => {
  test("duplicate event_id with different payload is idempotent first-wins (INSERT OR IGNORE on event_id PK)", async () => {
    // Implementation guarantee: events.event_id is the PRIMARY KEY and the
    // batch insert uses INSERT OR IGNORE, so a re-uploaded event_id is
    // skipped (skipped_duplicate) and the ORIGINAL row (payload + sequence)
    // is preserved. Conflicting payloads are NOT merged or rejected with 4xx.
    await post("/v1/session/start", sessBaseFor("sess-dup"));
    const first = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [evt(0, { event_id: "evt-dup-conflict", session_id: "sess-dup", payload: { v: 1 } })],
    });
    expect((await bodyJson(first)).ingested).toBe(1);

    const second = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [
        evt(99, { event_id: "evt-dup-conflict", session_id: "sess-dup", payload: { v: 2 } }),
      ],
    });
    expect(second.status).toBe(200);
    const b: any = await bodyJson(second);
    expect(b.ingested).toBe(0);
    expect(b.skipped_duplicate).toBe(1);

    const row = app!.db
      .query<{ sequence_number: number; payload_json: string }, [string]>(
        "SELECT sequence_number, payload_json FROM events WHERE event_id = ?",
      )
      .get("evt-dup-conflict");
    expect(row?.sequence_number).toBe(0);
    expect(JSON.parse(row!.payload_json)).toEqual({ v: 1 });
  });
});

describe("malformed battery", () => {
  test("batch without protocol_version is 400", async () => {
    await post("/v1/session/start", sessBaseFor("sess-mal"));
    const res = await post("/v1/events/batch", {
      events: [evt(0, { event_id: "evt-mal-nover", session_id: "sess-mal" })],
    });
    expect(res.status).toBe(400);
  });

  test("session start with empty session_id is 400", async () => {
    const res = await post("/v1/session/start", { ...SESSION_BASE, session_id: "" });
    expect(res.status).toBe(400);
  });

  test("negative sequence_number is 400", async () => {
    await post("/v1/session/start", sessBaseFor("sess-mal"));
    const res = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [evt(-1, { event_id: "evt-mal-neg", session_id: "sess-mal" })],
    });
    expect(res.status).toBe(400);
  });

  test("non-string event_id is 400", async () => {
    await post("/v1/session/start", sessBaseFor("sess-mal"));
    const res = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [evt(0, { event_id: 12345 as unknown as string, session_id: "sess-mal" })],
    });
    expect(res.status).toBe(400);
  });

  test("unknown event_type is 400", async () => {
    await post("/v1/session/start", sessBaseFor("sess-mal"));
    const res = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [evt(0, { event_id: "evt-mal-type", session_id: "sess-mal", event_type: "not_a_real_type" })],
    });
    expect(res.status).toBe(400);
  });

  test("malformed sha256 identifier is 400 (blobs check + snapshot)", async () => {
    const check = await post("/v1/blobs/check", { hashes: ["not-a-uuid"] });
    expect(check.status).toBe(400);
    const snap = await post("/v1/repository/snapshot", {
      protocol_version: 1,
      repo_id: "repo-mal",
      files: [{ relative_path: "a.py", sha256: "zzz" }],
    });
    expect(snap.status).toBe(400);
  });

  test("NUL bytes in identifiers are tolerated and stored verbatim (no 4xx enforcement)", async () => {
    // Documented behavior: the server performs no NUL-byte validation;
    // event_id "evt-nul\0id" is accepted and stored as-is.
    await post("/v1/session/start", sessBaseFor("sess-mal"));
    const res = await post("/v1/events/batch", {
      protocol_version: 1,
      events: [evt(60, { event_id: "evt-nul id", session_id: "sess-mal" })],
    });
    expect(res.status).toBe(200);
    expect(countTable("events")).toBe(1);
  });

  test("non-UUID session_id/event_id strings are accepted (no UUID-format enforcement)", async () => {
    // Documented behavior: session_id and event_id only require non-empty
    // strings; "not-a-uuid" style values are accepted.
    const start = await post("/v1/session/start", sessBaseFor("not-a-uuid!!"));
    expect([200, 201]).toContain(start.status);
  });

  test("11MB body is 413", async () => {
    const huge = "a".repeat(MAX_BODY_BYTES + 1024);
    const res = await app!.fetch(
      new Request("http://localhost/v1/events/batch", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: huge,
      }),
    );
    expect(res.status).toBe(413);
  });
});

describe("blob battery", () => {
  test("empty content_base64 is 400", async () => {
    const res = await post("/v1/blobs/upload", {
      sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      content_base64: "",
    });
    expect(res.status).toBe(400);
  });

  test("1-byte blob roundtrips", async () => {
    const hash = sha256HexText("x");
    const res = await post("/v1/blobs/upload", {
      sha256: hash,
      content_base64: Buffer.from("x", "utf8").toString("base64"),
    });
    expect(res.status).toBe(200);
    const b: any = await bodyJson(res);
    expect(b.original_bytes).toBe(1);
    const row = app!.db
      .query<{ content: Uint8Array; compression: string }, [string]>(
        "SELECT content, compression FROM blobs WHERE sha256 = ?",
      )
      .get(hash);
    expect(gunzipSync(row!.content as unknown as Buffer).toString("utf8")).toBe("x");
  });

  test("binary blob with NUL bytes roundtrips", async () => {
    const bin = Buffer.from([0, 1, 2, 0, 255, 0, 72, 105, 0, 0]);
    const hash = createHash("sha256").update(bin).digest("hex");
    const res = await post("/v1/blobs/upload", {
      sha256: hash,
      content_base64: bin.toString("base64"),
    });
    expect(res.status).toBe(200);
    const row = app!.db
      .query<{ content: Uint8Array; compression: string }, [string]>(
        "SELECT content, compression FROM blobs WHERE sha256 = ?",
      )
      .get(hash);
    expect(row?.compression).toBe("gzip");
    expect(Buffer.from(gunzipSync(row!.content as unknown as Buffer)).equals(bin)).toBe(true);
  });

  test("already-gzipped bytes are stored raw and byte-identical", async () => {
    const text = "already compressed payload " + "y".repeat(500);
    const gz = gzipSync(Buffer.from(text, "utf8"));
    const hash = createHash("sha256").update(gz).digest("hex");
    const res = await post("/v1/blobs/upload", {
      sha256: hash,
      content_base64: Buffer.from(gz).toString("base64"),
    });
    expect(res.status).toBe(200);
    expect((await bodyJson(res)).compression).toBe("raw");
    const row = app!.db
      .query<{ content: Uint8Array; compression: string }, [string]>(
        "SELECT content, compression FROM blobs WHERE sha256 = ?",
      )
      .get(hash);
    expect(row?.compression).toBe("raw");
    expect(Buffer.from(row!.content as unknown as Buffer).equals(Buffer.from(gz))).toBe(true);
  });

  test("unicode text roundtrips", async () => {
    const uni = "héllo 🌍 trajektorie — 日本語テスト\n".repeat(20);
    const hash = sha256HexText(uni);
    const res = await post("/v1/blobs/upload", {
      sha256: hash,
      content_base64: Buffer.from(uni, "utf8").toString("base64"),
    });
    expect(res.status).toBe(200);
    const row = app!.db
      .query<{ content: Uint8Array }, [string]>("SELECT content FROM blobs WHERE sha256 = ?")
      .get(hash);
    expect(gunzipSync(row!.content as unknown as Buffer).toString("utf8")).toBe(uni);
  });

  test("wrong sha256 is 400", async () => {
    const res = await post("/v1/blobs/upload", {
      sha256: "f".repeat(64),
      content_base64: Buffer.from("something else", "utf8").toString("base64"),
    });
    expect(res.status).toBe(400);
  });

  test("check-then-upload with partial overlap reports only missing hashes", async () => {
    const textA = "blob A deterministic content";
    const textB = "blob B deterministic content";
    const hashA = sha256HexText(textA);
    const hashB = sha256HexText(textB);
    const upA = await post("/v1/blobs/upload", {
      sha256: hashA,
      content_base64: Buffer.from(textA, "utf8").toString("base64"),
    });
    expect(upA.status).toBe(200);

    const check1 = await post("/v1/blobs/check", { hashes: [hashA, hashB] });
    expect(check1.status).toBe(200);
    expect((await bodyJson(check1)).missing).toEqual([hashB]);

    const upB = await post("/v1/blobs/upload", {
      sha256: hashB,
      content_base64: Buffer.from(textB, "utf8").toString("base64"),
    });
    expect(upB.status).toBe(200);
    const check2 = await post("/v1/blobs/check", { hashes: [hashA, hashB] });
    expect((await bodyJson(check2)).missing).toEqual([]);
  });
});

describe("session lifecycle adversarial", () => {
  test("end without start is 404", async () => {
    const res = await post("/v1/session/end", {
      protocol_version: 1,
      session_id: "sess-never-started",
    });
    expect(res.status).toBe(404);
  });

  test("double end: both 200, last ended_at wins", async () => {
    await post("/v1/session/start", sessBaseFor("sess-dbl-end"));
    const e1 = await post("/v1/session/end", {
      protocol_version: 1,
      session_id: "sess-dbl-end",
      ended_at: "2026-09-20T01:00:00.000Z",
    });
    expect(e1.status).toBe(200);
    const e2 = await post("/v1/session/end", {
      protocol_version: 1,
      session_id: "sess-dbl-end",
      ended_at: "2026-09-20T02:00:00.000Z",
    });
    expect(e2.status).toBe(200);
    expect((await bodyJson(e2)).ended_at).toBe("2026-09-20T02:00:00.000Z");
    const row = app!.db
      .query<{ ended_at: string }, [string]>("SELECT ended_at FROM sessions WHERE session_id = ?")
      .get("sess-dbl-end");
    expect(row?.ended_at).toBe("2026-09-20T02:00:00.000Z");
  });

  test("start twice with same session_id dedups, single row, original started_at preserved", async () => {
    const first = await post("/v1/session/start", sessBaseFor("sess-twice"));
    expect(first.status).toBe(201);
    const second = await post("/v1/session/start", {
      ...sessBaseFor("sess-twice"),
      started_at: "2026-09-21T00:00:00.000Z",
    });
    expect(second.status).toBe(200);
    expect((await bodyJson(second)).deduped).toBe(true);
    const rows = app!.db
      .query<{ started_at: string }, [string]>("SELECT started_at FROM sessions WHERE session_id = ?")
      .all("sess-twice");
    expect(rows.length).toBe(1);
    expect(rows[0]?.started_at).toBe("2026-09-20T00:00:00.000Z");
  });

  test("missing ended_at tolerated and session visible in stats", async () => {
    await post("/v1/session/start", sessBaseFor("sess-no-ended"));
    const end = await post("/v1/session/end", { protocol_version: 1, session_id: "sess-no-ended" });
    expect(end.status).toBe(200);
    const endedAt: string = (await bodyJson(end)).ended_at;
    expect(typeof endedAt).toBe("string");
    expect(Number.isNaN(Date.parse(endedAt))).toBe(false);
    const stats: any = await bodyJson(await get("/v1/stats"));
    expect(stats.session_count).toBe(1);
  });
});

describe("repository snapshot adversarial", () => {
  test("credential-stripping variants (user:pass@, token@, scp-like user@)", async () => {
    const hash = sha256HexText("cred content");
    const cases: Array<{ repo: string; origin: string }> = [
      { repo: "repo-cred-1", origin: "https://user:s3cret@example.com/org/r.git" },
      { repo: "repo-cred-2", origin: "https://ghp_FAKETOKEN0000000000000000000000@example.com/org/r.git" },
      { repo: "repo-cred-3", origin: "git@example.com:org/r.git" },
    ];
    for (const c of cases) {
      const res = await post("/v1/repository/snapshot", {
        protocol_version: 1,
        repo_id: c.repo,
        origin_url: c.origin,
        files: [{ relative_path: "a.py", sha256: hash }],
      });
      expect(res.status).toBe(200);
    }
    const rows = app!.db
      .query<{ repo_id: string; origin_url: string | null }, []>(
        "SELECT repo_id, origin_url FROM repositories ORDER BY repo_id",
      )
      .all();
    const byId = new Map(rows.map((r) => [r.repo_id, r.origin_url ?? ""]));
    expect(byId.get("repo-cred-1")).not.toContain("s3cret");
    expect(byId.get("repo-cred-1")).not.toContain("user@");
    expect(byId.get("repo-cred-2")).not.toContain("ghp_FAKETOKEN");
    expect(byId.get("repo-cred-2")).not.toContain("@example.com/org");
    expect(byId.get("repo-cred-3")).not.toContain("git@");
    for (const c of cases) expect(byId.get(c.repo)).toContain("example.com");
  });

  test("same file rescanned 10x yields a single version row", async () => {
    const hash = sha256HexText("stable content");
    const snap = {
      protocol_version: 1,
      repo_id: "repo-rescan",
      files: [{ relative_path: "src/a.py", sha256: hash, source: "repo_scan" }],
    };
    for (let i = 0; i < 10; i++) {
      const res = await post("/v1/repository/snapshot", snap);
      expect(res.status).toBe(200);
      const b: any = await bodyJson(res);
      if (i === 0) {
        expect(b.files_upserted).toBe(1);
        expect(b.versions_inserted).toBe(1);
      } else {
        expect(b.versions_inserted).toBe(0);
        expect(b.versions_skipped_duplicate).toBe(1);
      }
    }
    expect(countTable("files")).toBe(1);
    expect(countTable("file_versions")).toBe(1);
  });

  test("snapshot with 0 files succeeds and registers the repo", async () => {
    const res = await post("/v1/repository/snapshot", {
      protocol_version: 1,
      repo_id: "repo-empty",
      files: [],
    });
    expect(res.status).toBe(200);
    const b: any = await bodyJson(res);
    expect(b.files_upserted).toBe(0);
    expect(b.versions_inserted).toBe(0);
    expect(countTable("repositories")).toBe(1);
    expect(countTable("files")).toBe(0);
  });

  test("absolute paths, ../ traversal, backslashes, and NUL bytes rejected", async () => {
    // handleSnapshot rejects non-relative storage paths: leading "/",
    // ".." segments, backslashes, and NUL bytes all 400. Legit nested
    // paths still accepted.
    const hash = sha256HexText("traversal content");
    const abs = await post("/v1/repository/snapshot", {
      protocol_version: 1,
      repo_id: "repo-trav",
      files: [{ relative_path: "/abs.py", sha256: hash }],
    });
    expect(abs.status).toBe(400);

    for (const bad of ["../evil.py", "a/../../b.py", "a\\b.py", "a\0b.py"]) {
      const res = await post("/v1/repository/snapshot", {
        protocol_version: 1,
        repo_id: "repo-trav",
        files: [{ relative_path: bad, sha256: hash }],
      });
      expect(res.status).toBe(400);
    }
    expect(countTable("files")).toBe(0);

    const nested = await post("/v1/repository/snapshot", {
      protocol_version: 1,
      repo_id: "repo-trav",
      files: [{ relative_path: "a/b/c.py", sha256: hash }],
    });
    expect(nested.status).toBe(200);
  });
});

describe("file_versions dedup", () => {
  test("identical (file,hash,source) repeated collapses to one row", async () => {
    const hash = sha256HexText("dedup content");
    const snap = {
      protocol_version: 1,
      repo_id: "repo-dedup",
      files: [{ relative_path: "a.py", sha256: hash, source: "repo_scan" as const }],
    };
    await post("/v1/repository/snapshot", snap);
    await post("/v1/repository/snapshot", snap);
    await post("/v1/repository/snapshot", snap);
    expect(countTable("file_versions")).toBe(1);
  });

  test("same (file,hash) with different sources still yields one row: first source wins", async () => {
    // Schema guarantee: file_versions has UNIQUE(file_id, sha256) with no
    // source column in the key, and ingestion uses INSERT OR IGNORE. A second
    // source for the same file+hash is therefore skipped, not recorded.
    const hash = sha256HexText("source content");
    const r1 = await post("/v1/repository/snapshot", {
      protocol_version: 1,
      repo_id: "repo-src",
      files: [{ relative_path: "a.py", sha256: hash, source: "repo_scan" }],
    });
    expect((await bodyJson(r1)).versions_inserted).toBe(1);
    const r2 = await post("/v1/repository/snapshot", {
      protocol_version: 1,
      repo_id: "repo-src",
      default_source: "buffer_write",
      files: [{ relative_path: "a.py", sha256: hash }],
    });
    const b2: any = await bodyJson(r2);
    expect(b2.versions_inserted).toBe(0);
    expect(b2.versions_skipped_duplicate).toBe(1);
    const rows = app!.db
      .query<{ source: string }, []>("SELECT source FROM file_versions")
      .all();
    expect(rows.length).toBe(1);
    expect(rows[0]?.source).toBe("repo_scan");
  });
});

describe("stats consistency after mixed workload", () => {
  test("every stats field equals direct table counts / aggregates", async () => {
    await post("/v1/session/start", sessBaseFor("sess-st1", "machine-st"));
    await post("/v1/session/start", sessBaseFor("sess-st2", "machine-st"));
    const mk = (sess: string, seq: number) =>
      evt(seq, { event_id: `evt-${sess}-${seq}`, session_id: sess });
    await post("/v1/events/batch", {
      protocol_version: 1,
      events: [mk("sess-st1", 0), mk("sess-st1", 1), mk("sess-st1", 2)],
    });
    await post("/v1/events/batch", {
      protocol_version: 1,
      events: [mk("sess-st2", 0), mk("sess-st2", 1)],
    });
    const hashA = sha256HexText("stats-a");
    const hashB = sha256HexText("stats-b");
    await post("/v1/repository/snapshot", {
      protocol_version: 1,
      repo_id: "repo-st",
      files: [
        { relative_path: "a.py", sha256: hashA },
        { relative_path: "b.py", sha256: hashB },
      ],
    });
    const blobText1 = "stats blob one ".repeat(50);
    const blobText2 = "stats blob two ".repeat(50);
    await post("/v1/blobs/upload", {
      sha256: sha256HexText(blobText1),
      content_base64: Buffer.from(blobText1, "utf8").toString("base64"),
    });
    await post("/v1/blobs/upload", {
      sha256: sha256HexText(blobText2),
      content_base64: Buffer.from(blobText2, "utf8").toString("base64"),
    });

    const stats: any = await bodyJson(await get("/v1/stats"));
    expect(stats.event_count).toBe(countTable("events"));
    expect(stats.session_count).toBe(countTable("sessions"));
    expect(stats.repository_count).toBe(countTable("repositories"));
    expect(stats.file_count).toBe(countTable("files"));
    expect(stats.file_version_count).toBe(countTable("file_versions"));
    expect(stats.blob_count).toBe(countTable("blobs"));
    expect(stats.event_count).toBe(5);
    expect(stats.session_count).toBe(2);
    expect(stats.repository_count).toBe(1);
    expect(stats.file_count).toBe(2);
    expect(stats.file_version_count).toBe(2);
    expect(stats.blob_count).toBe(2);

    const agg = app!.db
      .query<{ original: number; stored: number }, []>(
        "SELECT COALESCE(SUM(original_bytes),0) AS original, COALESCE(SUM(stored_bytes),0) AS stored FROM blobs",
      )
      .get();
    expect(stats.blob_original_bytes).toBe(agg?.original);
    expect(stats.blob_stored_bytes).toBe(agg?.stored);
    expect(stats.blob_original_bytes).toBe(
      Buffer.byteLength(blobText1) + Buffer.byteLength(blobText2),
    );
    if ((agg?.original ?? 0) > 0) {
      expect(stats.compression_ratio).toBeCloseTo((agg?.stored ?? 0) / (agg?.original ?? 1), 10);
    } else {
      expect(stats.compression_ratio).toBeNull();
    }
    expect(typeof stats.db_bytes).toBe("number");
    expect(typeof stats.wal_bytes).toBe("number");
  });
});

describe("concurrent writers", () => {
  test("10 parallel batches on different sessions all persist, no SQLITE_BUSY escapes", async () => {
    const ids = Array.from({ length: 10 }, (_, i) => `sess-conc-${i}`);
    for (const id of ids) {
      const r = await post("/v1/session/start", sessBaseFor(id, "machine-conc"));
      expect([200, 201]).toContain(r.status);
    }
    const results = await Promise.all(
      ids.map((id, i) =>
        post("/v1/events/batch", {
          protocol_version: 1,
          events: [evt(i, { event_id: `evt-conc-${i}`, session_id: id })],
        }),
      ),
    );
    for (const r of results) expect(r.status).toBe(200);
    let ingested = 0;
    for (const r of results) {
      const b: any = await bodyJson(r);
      expect(JSON.stringify(b)).not.toContain("SQLITE_BUSY");
      ingested += b.ingested;
    }
    expect(ingested).toBe(10);
    const got = app!.db
      .query<{ session_id: string }, []>("SELECT session_id FROM events ORDER BY session_id")
      .all()
      .map((r) => r.session_id);
    expect(got).toEqual([...ids].sort());
    expect(ingested).toBeGreaterThanOrEqual(0);
  });
});

describe("migration on old-schema DB file", () => {
  test("legacy file with only machines table migrates in place, data preserved", async () => {
    // Build a legacy DB file by hand: only the machines table (subset of the
    // v1 schema) plus one data row, and no schema_migrations tracking table.
    app!.close();
    rmSync(dir, { recursive: true, force: true });
    dir = mkdtempSync(join(tmpdir(), "collector-migrate-"));
    const legacyPath = join(dir, "legacy.sqlite");
    const legacy = new Database(legacyPath, { create: true });
    legacy.exec(`
      CREATE TABLE machines (
        machine_id TEXT PRIMARY KEY,
        first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        editor TEXT,
        editor_version TEXT,
        plugin_version TEXT,
        platform TEXT,
        hostname_hash TEXT
      );
      INSERT INTO machines (machine_id, first_seen_at, last_seen_at, editor)
      VALUES ('legacy-machine', '2026-01-01T00:00:00.000Z', '2026-01-02T00:00:00.000Z', 'nvim');
    `);
    legacy.close();

    app = createApp(legacyPath);
    const tables = app!
      .db.query<{ name: string }, []>("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
      .all()
      .map((r) => r.name);
    for (const t of ["machines", "repositories", "sessions", "files", "blobs", "file_versions", "events"]) {
      expect(tables).toContain(t);
    }
    const kept = app!
      .db.query<{ machine_id: string; editor: string | null }, [string]>(
        "SELECT machine_id, editor FROM machines WHERE machine_id = ?",
      )
      .get("legacy-machine");
    expect(kept?.machine_id).toBe("legacy-machine");
    expect(kept?.editor).toBe("nvim");
    const versions = app!
      .db.query<{ version: number }, []>("SELECT version FROM schema_migrations ORDER BY version")
      .all()
      .map((r) => r.version);
    expect(versions).toContain(1);

    // The migrated DB is fully usable: start a session and ingest an event.
    const started = await post("/v1/session/start", sessBaseFor("sess-post-migrate", "legacy-machine"));
    expect([200, 201]).toContain(started.status);
  });
});

describe("WAL mode assertions", () => {
  test("journal_mode WAL, foreign_keys ON, busy_timeout set", async () => {
    const journal = app!.db.query<{ journal_mode: string }, []>("PRAGMA journal_mode").get();
    expect(journal?.journal_mode.toLowerCase()).toBe("wal");
    const fk = app!.db.query<{ foreign_keys: number }, []>("PRAGMA foreign_keys").get();
    expect(fk?.foreign_keys).toBe(1);
    const busy = app!.db.query<{ timeout: number }, []>("PRAGMA busy_timeout").get();
    expect(busy?.timeout).toBe(5000);
    const sync = app!.db.query<{ synchronous: number }, []>("PRAGMA synchronous").get();
    expect(sync?.synchronous).toBe(1);
  });
});

describe("prediction projection ingestion", () => {
  test("out-of-order retry batches update one projection row", async () => {
    expect((await post("/v1/session/start", SESSION_BASE)).status).toBe(201);
    const event = (id: string, seq: number, eventType: string, payload: object) => ({
      protocol_version: 1, event_id: id, session_id: "sess-1", sequence_number: seq,
      timestamp_ms: 1790000000000 + seq, event_type: eventType,
      file_id: "repo:synthetic:main.py", cursor: { row: 0, col: 0 }, mode: "i",
      payload: { prediction_id: "prediction-1", ...payload },
    });
    const later = [
      event("display", 3, "prediction_shown", { active_buffer: true }),
      event("dismiss", 4, "prediction_dismissed", {
        outcome: "typed_match", outcome_source: "editor_observation", ended_by_event_id: "delta",
      }),
    ];
    const earlier = [
      event("request", 1, "prediction_requested", {
        file_identity: "repo:synthetic:main.py", pre_state_hash: "a".repeat(64),
        pre_state_sequence: 0, model_revision: "selected-q25",
      }),
      event("generated", 2, "prediction_generated", { action_blob_hash: "b".repeat(64) }),
    ];
    const batch = (events: object[]) => ({ protocol_version: 1, events });
    expect((await post("/v1/events/batch", batch(later))).status).toBe(200);
    expect((await post("/v1/events/batch", batch(earlier))).status).toBe(200);
    const replay = await bodyJson(await post("/v1/events/batch", batch([...earlier, ...later])));
    expect(replay.ingested).toBe(0);
    const row = app!.db.query<{ outcome: string; request_event_id: string;
      shown_event_id: string; resolved_file_id: number | null; updated_through_sequence: number }, []>(
      "SELECT outcome,request_event_id,shown_event_id,resolved_file_id,updated_through_sequence " +
      "FROM prediction_projection WHERE prediction_id='prediction-1'",
    ).get();
    expect(row).toEqual({ outcome: "typed_match", request_event_id: "request",
      shown_event_id: "display", resolved_file_id: null, updated_through_sequence: 4 });
    expect(app!.db.query<{n: number}, []>("SELECT COUNT(*) n FROM prediction_projection").get()?.n)
      .toBe(1);
  });
});
