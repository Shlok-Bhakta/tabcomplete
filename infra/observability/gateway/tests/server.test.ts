import { describe, expect, test } from "bun:test";
import {
  artifactPath,
  createHandler,
  escapeFilterValue,
  extractRows,
  parseSince,
} from "../src/server";

const options = {
  signozUrl: "http://signoz:8080",
  apiKey: "test-key",
  artifactRoot: "/artifacts",
  allowedHosts: new Set(["kiwi", "kiwi.tail.test", "crabcake.tail.test"]),
  fetchImpl: async () => Response.json({ data: { result: [] } }),
};

describe("gateway request bounds", () => {
  test("rejects unknown Host values", async () => {
    const response = await createHandler(options)(
      new Request("http://evil.test/api/v1/runs", { headers: { Host: "evil.test" } }),
    );
    expect(response.status).toBe(421);
  });

  test("rejects unsafe cross-origin requests", async () => {
    const response = await createHandler(options)(
      new Request("http://kiwi/api/v1/runs", {
        headers: { Host: "kiwi", Origin: "https://evil.test" },
      }),
    );
    expect(response.status).toBe(403);
  });

  test("bounds time windows and pagination", () => {
    expect(parseSince("24h")).toBe(24 * 60 * 60 * 1000);
    expect(() => parseSince("91d")).toThrow("90 days");
    expect(() => parseSince("forever")).toThrow("invalid");
  });

  test("escapes SigNoz filter values", () => {
    expect(escapeFilterValue("run' OR true")).toBe("run\\' OR true");
  });

  test("blocks artifact path traversal", () => {
    expect(() => artifactPath("/artifacts", "../../etc/passwd")).toThrow("hash");
    expect(artifactPath("/artifacts", "a".repeat(64))).toBe(
      `/artifacts/aa/${"a".repeat(64)}.txt.gz`,
    );
  });
});

describe("gateway response contract", () => {
  test("returns a stable empty-state schema", async () => {
    const response = await createHandler(options)(
      new Request("http://kiwi/api/v1/runs?since=24h&limit=20", {
        headers: { Host: "kiwi" },
      }),
    );
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.schema_version).toBe(1);
    expect(body.data).toEqual([]);
    expect(body.page.limit).toBe(20);
    expect(body.state).toBe("no_runs");
  });

  test("extracts raw table rows without evaluating model text", () => {
    const dangerous = "<img src=x onerror=alert(1)>";
    const rows = extractRows({ data: { result: [{ table: { rows: [{ response: dangerous }] } }] } });
    expect(rows).toEqual([{ response: dangerous }]);
  });

  test("flattens SigNoz v5 raw row wrappers", () => {
    const rows = extractRows({
      data: {
        result: [{ table: { rows: [{ data: { name: "run.summary", "tabcomplete.run_id": "run-1" }, timestamp: "2026-09-22T18:58:39Z" }] } }],
      },
    });
    expect(rows).toEqual([
      { name: "run.summary", "tabcomplete.run_id": "run-1", timestamp: "2026-09-22T18:58:39Z" },
    ]);
  });

  test("has no arbitrary query or file endpoint", async () => {
    for (const path of ["/api/v1/sql", "/api/v1/files", "/api/v1/shell"]) {
      const response = await createHandler(options)(
        new Request(`http://kiwi${path}`, { headers: { Host: "kiwi" } }),
      );
      expect(response.status).toBe(404);
    }
  });
});
