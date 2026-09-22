import { join, normalize } from "node:path";
import { mkdir, stat, rename, statfs, utimes, unlink } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { createHash, timingSafeEqual } from "node:crypto";

type Fetch = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

export type GatewayOptions = {
  signozUrl: string;
  apiKey: string;
  artifactRoot: string;
  allowedHosts: Set<string>;
  fetchImpl?: Fetch;
  uploadToken?: string;
  artifactCapBytes?: number;
  stateRoot?: string;
  minimumFreeBytes?: number;
};

const MAX_WINDOW_MS = 90 * 24 * 60 * 60 * 1000;
const MAX_LIMIT = 200;
const ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const HASH = /^[0-9a-f]{64}$/;
let artifactWrite: Promise<void> = Promise.resolve();
async function withArtifactLock<T>(action: () => Promise<T>): Promise<T> {
  const previous=artifactWrite;
  let release!:()=>void;
  artifactWrite=new Promise<void>(resolve=>{release=resolve;});
  await previous;
  try{return await action();}finally{release();}
}

function flattenRow(row: Record<string, unknown>): Record<string, unknown> {
  const nested = row.data;
  if (typeof nested === "object" && nested !== null && !Array.isArray(nested)) {
    return { ...(nested as Record<string, unknown>), ...Object.fromEntries(Object.entries(row).filter(([key]) => key !== "data")) };
  }
  return row;
}

const securityHeaders = {
  "Cache-Control": "no-store",
  "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
};

export function escapeFilterValue(value: string): string {
  return value.replaceAll("\\", "\\\\").replaceAll("'", "\\'");
}

export function parseSince(value: string | null): number {
  const raw = value ?? "24h";
  const match = /^(\d+)(m|h|d)$/.exec(raw);
  if (!match) throw new Error("invalid since value; use Nm, Nh, or Nd");
  const amount = Number(match[1]);
  const multiplier = { m: 60_000, h: 3_600_000, d: 86_400_000 }[match[2]]!;
  const duration = amount * multiplier;
  if (duration <= 0 || duration > MAX_WINDOW_MS) throw new Error("since cannot exceed 90 days");
  return duration;
}

export function artifactPath(root: string, hash: string): string {
  if (!HASH.test(hash)) throw new Error("artifact hash must be 64 lowercase hexadecimal characters");
  const path = normalize(join(root, hash.slice(0, 2), `${hash}.txt.gz`));
  const normalizedRoot = normalize(root + "/");
  if (!path.startsWith(normalizedRoot)) throw new Error("artifact path is outside storage root");
  return path;
}

export async function pruneArtifacts(root: string, now = Date.now(), retentionDays = 30) {
  let removed = 0;
  for await (const path of new Bun.Glob("*/*.txt.gz").scan({cwd:root,absolute:true})) {
    const name = path.split('/').pop()!.replace('.txt.gz','');
    if (!HASH.test(name)) continue;
    if (now - (await stat(path)).mtimeMs > retentionDays * 86400000) {
      await unlink(path); removed++;
    }
  }
  return removed;
}

export function extractRows(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) {
    const direct = value.filter(
      (item): item is Record<string, unknown> =>
        typeof item === "object" && item !== null && !Array.isArray(item),
    );
    const wrapperKeys = new Set(["data", "result", "table", "results", "rows"]);
    if (
      direct.length === value.length &&
      direct.length > 0 &&
      direct.every((item) => !Object.keys(item).some((key) => wrapperKeys.has(key)))
    ) return direct;
    for (const item of value) {
      const nested = extractRows(item);
      if (nested.length) return nested;
    }
  }
  if (typeof value === "object" && value !== null) {
    const object = value as Record<string, unknown>;
    if (Array.isArray(object.rows)) {
      return (object.rows as Record<string, unknown>[]).map(flattenRow);
    }
    for (const key of ["data", "result", "table", "results"]) {
      if (key in object) {
        const nested = extractRows(object[key]);
        if (nested.length) return nested;
      }
    }
  }
  return [];
}

function json(status: number, value: unknown): Response {
  return Response.json(value, { status, headers: securityHeaders });
}

function acceptedRequest(request: Request, allowedHosts: Set<string>): Response | null {
  const host = (request.headers.get("host") ?? "").split(":")[0].toLowerCase();
  if (!allowedHosts.has(host)) return json(421, { error: "host_not_allowed" });
  const origin = request.headers.get("origin");
  if (origin) {
    let originHost: string;
    try {
      originHost = new URL(origin).hostname.toLowerCase();
    } catch {
      return json(403, { error: "origin_not_allowed" });
    }
    if (!allowedHosts.has(originHost) || originHost !== host || new URL(origin).host !== request.headers.get("host")) {
      return json(403, { error: "origin_not_allowed" });
    }
  }
  return null;
}

const selectFields = [
  "name",
  "timestamp",
  "duration_nano",
  "trace_id",
  "span_id",
  "service.name",
  "tabcomplete.campaign_id",
  "tabcomplete.run_id",
  "tabcomplete.run_attempt_id",
  "tabcomplete.case_id",
  "tabcomplete.case_attempt_id",
  "tabcomplete.request_id",
  "tabcomplete.run.state",
  "tabcomplete.phase",
  "tabcomplete.next_edit.predicted_action",
  "tabcomplete.next_edit.action_correct",
  "tabcomplete.next_edit.deletion_success",
  "tabcomplete.next_edit.valid_patch",
  "tabcomplete.next_edit.functional_success",
  "tabcomplete.model.revision",
  "gen_ai.request.model",
  "gen_ai.provider.name",
  "gen_ai.request.max_tokens",
  "gen_ai.usage.input_tokens",
  "gen_ai.usage.output_tokens",
  "gen_ai.response.finish_reasons",
  "tabcomplete.timing.total_ms",
  "tabcomplete.timing.first_output_ms",
  "tabcomplete.output.truncated",
  "tabcomplete.quality.parse",
  "tabcomplete.quality.compile",
  "tabcomplete.quality.execute",
  "tabcomplete.quality.functional",
  "tabcomplete.failure.type",
  "tabcomplete.failure.message",
  "tabcomplete.artifact.input.sha256",
  "tabcomplete.artifact.input.status",
  "tabcomplete.artifact.input.preview",
  "tabcomplete.artifact.output.sha256",
  "tabcomplete.artifact.output.status",
  "tabcomplete.artifact.output.preview",
  "tabcomplete.cases.planned",
  "tabcomplete.cases.completed",
  "tabcomplete.cases.failed",
  "tabcomplete.artifact.input.source_ref",
  "tabcomplete.artifact.output.source_ref",
  "host.name",
  "tabcomplete.suite.sha256", "tabcomplete.protocol", "tabcomplete.decoding",
  "tabcomplete.device", "tabcomplete.request.context_tokens", "tabcomplete.historical",
  "tabcomplete.terminal_event_id", "tabcomplete.outcome", "tabcomplete.training.loss",
  "tabcomplete.training.validation_nll", "tabcomplete.training.lr",
  "tabcomplete.training.gradient_norm", "tabcomplete.training.input_tokens",
  "tabcomplete.training.scored_tokens", "tabcomplete.training.successful_updates",
  "tabcomplete.training.skipped_updates", "tabcomplete.storage.free_bytes",
  "tabcomplete.training.event", "tabcomplete.training.consecutive_scaler_events",
  "tabcomplete.timestamp.kind",
  "tabcomplete.storage.healthy", "tabcomplete.storage.artifact_bytes",
];

export function queryPayload(start: number, end: number, filter: string, limit: number, offset: number) {
  return {
    start,
    end,
    requestType: "raw",
    variables: {},
    compositeQuery: {
      queries: [
        {
          type: "builder_query",
          spec: {
            name: "A",
            signal: "traces",
            filter: { expression: filter },
            selectFields: selectFields.map((name) => ({
              name,
              fieldContext: name === "service.name" || name === "host.name" ? "resource" : "span",
            })),
            order: [{ key: { name: "timestamp" }, direction: "desc" }],
            limit,
            offset,
            disabled: false,
          },
        },
      ],
    },
  };
}

let activeQueries = 0;
async function queryRows(
  options: GatewayOptions,
  start: number,
  end: number,
  filter: string,
  limit: number,
  offset: number,
  signal: "traces" | "logs" = "traces",
): Promise<Record<string, unknown>[]> {
  const fetcher = options.fetchImpl ?? fetch;
  if (activeQueries >= 8) throw new Error("SigNoz query failed: concurrency limit");
  activeQueries++;
  try {
  const payload = queryPayload(start, end, filter, limit, offset);
  if (signal === "logs") {
    const spec = payload.compositeQuery.queries[0].spec;
    spec.signal = "logs";
    spec.selectFields = ["timestamp", "body", "trace_id", "span_id", "tabcomplete.run_id", "tabcomplete.case_id", "tabcomplete.request_id", "tabcomplete.failure.type", "tabcomplete.failure.message", "tabcomplete.failure.stage"].map(name => ({name,fieldContext:"log"}));
  }
  const response = await fetcher(`${options.signozUrl}/api/v5/query_range`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "SIGNOZ-API-KEY": options.apiKey },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(8_000),
  });
  if (!response.ok) throw new Error(`SigNoz query failed with HTTP ${response.status}`);
  return extractRows(await response.json());
  } finally { activeQueries--; }
}

function page(url: URL) {
  const limit = Math.min(MAX_LIMIT, Math.max(1, Number(url.searchParams.get("limit") ?? "50")));
  const offset = Math.min(10_000, Math.max(0, Number(url.searchParams.get("cursor") ?? "0")));
  if (!Number.isInteger(limit) || !Number.isInteger(offset)) throw new Error("invalid pagination");
  return { limit, offset };
}

function identifier(value: string, kind: string): string {
  if (!ID.test(value)) throw new Error(`invalid ${kind}`);
  return escapeFilterValue(value);
}

function envelope(rows: Record<string, unknown>[], limit: number, offset: number) {
  const timestamps = rows
    .map((row) => {
      const value = row.timestamp ?? 0;
      return typeof value === "string" ? Date.parse(value) : Number(value);
    })
    .filter((value) => Number.isFinite(value) && value > 0);
  return {
    schema_version: 1,
    state: rows.length ? "ok" : "no_runs",
    data: rows,
    page: { limit, next_cursor: rows.length === limit ? String(offset + limit) : null },
    // This is event time, not backend arrival time. Offline events retain their timestamp.
    last_event_at: timestamps.length ? Math.max(...timestamps) : null,
    last_ingested_at: null,
    queried_at: Date.now(),
  };
}

const collectorMetricNames = ["otelcol_exporter_queue_size","otelcol_exporter_queue_capacity","otelcol_exporter_send_failed_spans_total","otelcol_exporter_enqueue_failed_spans_total","otelcol_receiver_refused_spans_total","otelcol_process_memory_rss"];
async function collectorMetrics(options: GatewayOptions) {
  const end=Date.now();
  const response=await (options.fetchImpl ?? fetch)(options.signozUrl+'/api/v5/query_range',{
    method:'POST',headers:{'Content-Type':'application/json','SIGNOZ-API-KEY':options.apiKey},signal:AbortSignal.timeout(8000),
    body:JSON.stringify({start:end-300000,end,requestType:'scalar',compositeQuery:{queries:collectorMetricNames.map((metricName,i)=>({type:'builder_query',spec:{name:'M'+i,signal:'metrics',aggregations:[{metricName,temporality:'Unspecified',timeAggregation:'latest',spaceAggregation:'sum',reduceTo:'last'}],limit:10,order:[{key:{name:'__result'},direction:'desc'}]}}))}})
  });
  if (!response.ok) return {state:'unavailable',metrics:[]};
  const body=await response.json();
  const results=body.data?.data?.results ?? body.data?.result ?? [];
  return {state:'queried',window_seconds:300,metrics:collectorMetricNames.map((name,i)=>({name,value:results.find((r:any)=>r.queryName==='M'+i)?.data?.[0]?.[0] ?? null}))};
}

async function api(request: Request, options: GatewayOptions, url: URL): Promise<Response> {
  if (request.method !== "GET") return json(405, { error: "method_not_allowed" });
  if (url.pathname === "/api/v1/infrastructure") {
    const collectors=await collectorMetrics(options).catch(()=>({state:'unavailable',metrics:[]}));
    const file = Bun.file(join(options.stateRoot ?? "/state", "health.json"));
    if (!await file.exists()) return json(200,{schema_version:1,collectors,storage:{state:"unknown",reason:"health_record_missing"}});
    const health = await file.json();
    return json(200,{schema_version:1,collectors,storage:{...health,state:Date.now()-Number(health.checked_at)>90000 ? "stale" : health.ok ? "healthy" : "unhealthy"}});
  }
  if (url.pathname === "/api/v1/status") {
    const fetcher = options.fetchImpl ?? fetch;
    const response = await fetcher(`${options.signozUrl}/api/v2/healthz`, {
      signal: AbortSignal.timeout(3_000),
    });
    return json(response.ok ? 200 : 503, {
      schema_version: 1,
      status: response.ok ? "healthy" : "unhealthy",
      checked_at: Date.now(),
    });
  }
  const { limit, offset } = page(url);
  const end = Date.now();
  const start = end - parseSince(url.searchParams.get("since"));
  let filter = "service.name = 'tabcomplete'";
  const runMatch = /^\/api\/v1\/runs\/([^/]+)$/.exec(url.pathname);
  const failuresMatch = /^\/api\/v1\/runs\/([^/]+)\/failures$/.exec(url.pathname);
  const traceMatch = /^\/api\/v1\/traces\/([^/]+)$/.exec(url.pathname);
  const requestMatch = /^\/api\/v1\/requests\/([^/]+)$/.exec(url.pathname);
  const logsMatch = /^\/api\/v1\/runs\/([^/]+)\/logs$/.exec(url.pathname);
  const compareMatch = /^\/api\/v1\/compare\/([^/]+)\/([^/]+)$/.exec(url.pathname);
  if (logsMatch) {
    const rows = await queryRows(options,start,end,filter + ` AND tabcomplete.run_id = '${identifier(logsMatch[1], "run ID")}'`,limit,offset,"logs");
    return json(200,envelope(rows,limit,offset));
  } else if (url.pathname === "/api/v1/runs") {
    filter += " AND tabcomplete.run_id EXISTS AND name IN ('run.start','run.heartbeat','run.summary')";
  } else if (url.pathname === "/api/v1/events") {
    filter += " AND tabcomplete.run_id EXISTS";
  } else if (failuresMatch) {
    filter += ` AND tabcomplete.run_id = '${identifier(failuresMatch[1], "run ID")}' AND (tabcomplete.failure.type EXISTS OR tabcomplete.outcome = 'failed')`;
  } else if (runMatch) {
    filter += ` AND tabcomplete.run_id = '${identifier(runMatch[1], "run ID")}'`;
  } else if (traceMatch) {
    const traceId = traceMatch[1];
    if (!/^[0-9a-f]{32}$/.test(traceId)) throw new Error("invalid trace ID");
    filter = `trace_id = '${traceId}'`;
  } else if (requestMatch) {
    filter += ` AND tabcomplete.request_id = '${identifier(requestMatch[1], "request ID")}'`;
  } else if (compareMatch) {
    const left = await queryRows(
      options,
      start,
      end,
      `service.name = 'tabcomplete' AND tabcomplete.run_id = '${identifier(compareMatch[1], "run ID")}'`,
      limit,
      0,
    );
    const right = await queryRows(
      options,
      start,
      end,
      `service.name = 'tabcomplete' AND tabcomplete.run_id = '${identifier(compareMatch[2], "run ID")}'`,
      limit,
      0,
    );
    const keys = [
      "tabcomplete.suite.sha256",
      "tabcomplete.protocol",
      "tabcomplete.device",
      "gen_ai.request.model",
      "tabcomplete.request.context_tokens",
      "tabcomplete.decoding",
    ];
    return json(200, {
      schema_version: 1,
      runs: [left, right],
      incompatibilities: keys.filter((key) => {
        const values = (rows: Record<string, unknown>[]) => [...new Set(rows.map(row => row[key]).filter(v => v != null && v !== ""))].sort();
        return JSON.stringify(values(left)) !== JSON.stringify(values(right));
      }),
      unknown_comparisons: keys.filter(key => !left.some(row => row[key] != null && row[key] !== "") || !right.some(row => row[key] != null && row[key] !== "")),
      partial: left.length === limit || right.length === limit,
      queried_at: Date.now(),
    });
  } else {
    return json(404, { error: "not_found" });
  }
  const rows = await queryRows(options, start, end, filter, limit, offset);
  return json(200, envelope(rows, limit, offset));
}

export function createHandler(options: GatewayOptions) {
  return async (request: Request): Promise<Response> => {
    const rejected = acceptedRequest(request, options.allowedHosts);
    if (rejected) return rejected;
    const url = new URL(request.url);
    try {
      if (url.pathname.startsWith("/internal/v1/artifacts/")) {
        if (request.method !== "PUT") return json(405, { error: "method_not_allowed" });
        const supplied = Buffer.from(request.headers.get("authorization") ?? "");
        const expected = Buffer.from("Bearer " + (options.uploadToken ?? ""));
        if (!options.uploadToken || supplied.length !== expected.length || !timingSafeEqual(supplied, expected)) {
          return json(401, { error: "unauthorized" });
        }
        const hash = url.pathname.slice("/internal/v1/artifacts/".length);
        const destination = artifactPath(options.artifactRoot, hash);
        if (Number(request.headers.get("content-length") ?? "0") > 8 * 2 ** 20) return json(413, { error: "payload_too_large" });
        const chunks: Uint8Array[] = [];
        let size = 0;
        if (!request.body) return json(400, { error: "missing_body" });
        for await (const chunk of request.body) {
          size += chunk.byteLength;
          if (size > 8 * 2 ** 20) return json(413, { error: "payload_too_large" });
          chunks.push(chunk);
        }
        const payload = Buffer.concat(chunks);
        if (createHash("sha256").update(payload).digest("hex") !== hash) return json(400, { error: "hash_mismatch" });
        return await withArtifactLock(async()=>{
        const existing = Bun.file(destination);
        if (await existing.exists()) {
          await utimes(destination,new Date(),new Date());
          return json(200, { sha256: hash, deduplicated: true });
        }
        const filesystem = await statfs(options.artifactRoot);
        if (filesystem.bavail * filesystem.bsize < (options.minimumFreeBytes ?? 100 * 2 ** 30)) return json(507,{error:"storage_pressure"});
        let total = 0;
        const glob = new Bun.Glob("*/*.txt.gz");
        for await (const path of glob.scan({ cwd: options.artifactRoot, absolute: true })) total += (await stat(path)).size;
        if (total + size > (options.artifactCapBytes ?? 20 * 2 ** 30)) return json(507, { error: "artifact_capacity" });
        await mkdir(join(options.artifactRoot, hash.slice(0, 2)), { recursive: true, mode: 0o750 });
        const temporary = destination + "." + crypto.randomUUID() + ".tmp";
        await Bun.write(temporary, Bun.gzipSync(payload));
        await rename(temporary, destination);
        return json(201, { sha256: hash, deduplicated: false, bytes: size });
        });
      }
      if (url.pathname.startsWith("/api/v1/artifacts/")) {
        if (request.method !== "GET") return json(405, { error: "method_not_allowed" });
        const hash = url.pathname.slice("/api/v1/artifacts/".length);
        const file = Bun.file(artifactPath(options.artifactRoot, hash));
        if (!(await file.exists())) return json(404, { error: "artifact_not_found" });
        if (file.size > 9 * 2 ** 20) return json(413, { error: "artifact_too_large" });
        const payload = gunzipSync(Buffer.from(await file.arrayBuffer()), { maxOutputLength: 8 * 2 ** 20 });
        if (createHash("sha256").update(payload).digest("hex") !== hash) return json(500, { error: "artifact_integrity" });
        return new Response(payload, {
          headers: {
            ...securityHeaders,
            "Content-Type": "text/plain; charset=utf-8",
          },
        });
      }
      if (url.pathname.startsWith("/api/")) return await api(request, options, url);
      const publicRoot = normalize(join(import.meta.dir, "../public/"));
      const relative = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
      const filePath = normalize(join(publicRoot, relative));
      if (!filePath.startsWith(publicRoot)) return json(404, { error: "not_found" });
      const file = Bun.file(filePath);
      if (!(await file.exists())) return json(404, { error: "not_found" });
      return new Response(file, { headers: securityHeaders });
    } catch (error) {
      const message = error instanceof Error ? error.message : "request failed";
      const status = message.startsWith("SigNoz query failed") ? 502 : 400;
      return json(status, { error: status === 502 ? "backend_unavailable" : "invalid_request", message });
    }
  };
}

if (import.meta.main) {
  const allowedHosts = new Set(
    (process.env.TABCOMPLETE_ALLOWED_HOSTS ?? "kiwi,localhost,127.0.0.1")
      .split(",")
      .map((value) => value.trim().toLowerCase())
      .filter(Boolean),
  );
  const options: GatewayOptions = {
    signozUrl: process.env.SIGNOZ_URL ?? "http://signoz-signoz-0:8080",
    apiKey: process.env.SIGNOZ_API_KEY ?? "",
    artifactRoot: process.env.TABCOMPLETE_ARTIFACT_ROOT ?? "/artifacts",
    allowedHosts,
    uploadToken: process.env.TABCOMPLETE_ARTIFACT_UPLOAD_TOKEN,
    artifactCapBytes: Number(process.env.TABCOMPLETE_ARTIFACT_CAP_BYTES ?? 20 * 2 ** 30),
    stateRoot: process.env.TABCOMPLETE_STATE_ROOT ?? "/state",
  };
  if (!options.apiKey) throw new Error("SIGNOZ_API_KEY is required");
  const prune = () => pruneArtifacts(options.artifactRoot,Date.now(),Number(process.env.TABCOMPLETE_ARTIFACT_RETENTION_DAYS ?? 30)).catch(()=>undefined);
  void prune();
  setInterval(prune,3600000);
  Bun.serve({
    hostname: "0.0.0.0",
    port: Number(process.env.PORT ?? "9090"),
    fetch: createHandler(options),
  });
}
