/** Rebuildable, compact prediction query rows. Raw events remain authoritative. */
import type { Database } from "bun:sqlite";

const VERSION = 1;
type Row = {
  event_id: string; session_id: string; sequence_number: number;
  event_type: string; timestamp_ms: number; file_id: number | null;
  payload_json: string | null;
};
function object(raw: string | null): Record<string, unknown> {
  if (!raw) return {};
  try {
    const value: unknown = JSON.parse(raw);
    return value && typeof value === "object" && !Array.isArray(value)
      ? value as Record<string, unknown> : {};
  } catch { return {}; }
}
function string(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}
function integer(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) ? value : null;
}

export function rebuildPrediction(db: Database, sessionId: string, predictionId: string): void {
  const rows = db.query<Row, [string, string]>(
    `SELECT event_id, session_id, sequence_number, event_type, timestamp_ms, file_id, payload_json
     FROM events WHERE session_id = ?
       AND (event_type LIKE 'prediction_%' OR event_type = 'heartbeat')
       AND json_valid(payload_json)
       AND json_extract(payload_json, '$.prediction_id') = ?
     ORDER BY sequence_number, event_id`,
  ).all(sessionId, predictionId);
  if (rows.length === 0) return;
  let request: Row | null = null;
  let shown: Row | null = null;
  let lastOutcome: Row | null = null;
  let action: Row | null = null;
  let outcome: string | null = null;
  let source: string | null = null;
  let terminalRecorded = false;
  for (const row of rows) {
    const payload = object(row.payload_json);
    if (row.event_type === "prediction_requested" && !request) request = row;
    if (row.event_type === "prediction_generated") action = row;
    if (row.event_type === "prediction_shown" && !shown) shown = row;
    if (row.event_type === "prediction_accepted" && !terminalRecorded) {
      outcome = "accepted"; source = "explicit_acceptance"; lastOutcome = row;
      terminalRecorded = true;
    } else if (row.event_type === "prediction_partially_accepted" && !terminalRecorded) {
      outcome = "partially_accepted"; source = "explicit_partial_acceptance"; lastOutcome = row;
      terminalRecorded = true;
    } else if (row.event_type === "prediction_dismissed" && !terminalRecorded) {
      outcome = string(payload["outcome"]) ?? "dismissed_unknown";
      source = string(payload["outcome_source"]) ?? "editor_observation";
      lastOutcome = row;
      terminalRecorded = true;
    } else if (row.event_type === "prediction_rejected" && !terminalRecorded) {
      outcome = "rejected_explicit"; source = "explicit_rejection"; lastOutcome = row;
      terminalRecorded = true;
    } else if (row.event_type === "heartbeat") {
      const lifecycle = string(payload["prediction_lifecycle"]);
      if (!terminalRecorded && lifecycle && ["cancelled_unseen", "invalidated_unseen", "shadow_only",
        "model_no_edit", "request_failed", "invalid_output", "automatic_multiline_suppressed",
        "expired"].includes(lifecycle)) {
        outcome = lifecycle; source = "request_lifecycle"; lastOutcome = row;
        terminalRecorded = true;
      }
    }
  }
  const req = request ? object(request.payload_json) : {};
  const display = shown ? object(shown.payload_json) : {};
  const generated = action ? object(action.payload_json) : {};
  // The editor's `repo:<root-name>:<relative-path>` is a client identity,
  // while files.file_id is an integer. Resolve through the session's repo;
  // never reinterpret the text stored in events.file_id as an integer FK.
  const fileIdentity = string(req["file_identity"]) ?? string(req["file"])
    ?? string(display["file"]);
  const resolved = fileIdentity ? db.query<{ file_id: number }, [string, string]>(
    `SELECT f.file_id FROM files f
       JOIN repositories r ON r.repo_id = f.repo_id
       JOIN sessions s ON s.repo_id = f.repo_id
      WHERE s.session_id = ?
        AND ('repo:' || r.root_name || ':' || f.relative_path) = ?
      LIMIT 1`,
  ).get(sessionId, fileIdentity)?.file_id ?? null : null;
  const lastSeq = rows[rows.length - 1]!.sequence_number;
  db.prepare(
    `INSERT INTO prediction_projection (
       session_id,prediction_id,request_event_id,shown_event_id,last_outcome_event_id,
       file_identity,resolved_file_id,model_revision,model_gguf_sha256,
       runtime_config_hash,context_policy_version,pre_state_hash,pre_state_sequence,
       context_blob_hash,action_blob_hash,requested_at,shown_at,closed_at,
       outcome,outcome_source,projection_version,updated_through_sequence)
     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
     ON CONFLICT(session_id,prediction_id) DO UPDATE SET
       request_event_id=excluded.request_event_id,shown_event_id=excluded.shown_event_id,
       last_outcome_event_id=excluded.last_outcome_event_id,file_identity=excluded.file_identity,
       resolved_file_id=excluded.resolved_file_id,model_revision=excluded.model_revision,
       model_gguf_sha256=excluded.model_gguf_sha256,runtime_config_hash=excluded.runtime_config_hash,
       context_policy_version=excluded.context_policy_version,pre_state_hash=excluded.pre_state_hash,
       pre_state_sequence=excluded.pre_state_sequence,context_blob_hash=excluded.context_blob_hash,
       action_blob_hash=excluded.action_blob_hash,requested_at=excluded.requested_at,
       shown_at=excluded.shown_at,closed_at=excluded.closed_at,outcome=excluded.outcome,
       outcome_source=excluded.outcome_source,projection_version=excluded.projection_version,
       updated_through_sequence=excluded.updated_through_sequence`,
  ).run(
    sessionId, predictionId, request?.event_id ?? null, shown?.event_id ?? null,
    lastOutcome?.event_id ?? null, fileIdentity, resolved,
    string(req["model_revision"]), string(req["model_gguf_sha256"]),
    string(req["runtime_config_hash"]), string(req["context_policy_version"]),
    string(req["pre_state_hash"]), integer(req["pre_state_sequence"]),
    string(req["context_blob_hash"]), string(generated["action_blob_hash"])
      ?? string(display["action_blob_hash"]), request?.timestamp_ms ?? null,
    shown?.timestamp_ms ?? null, lastOutcome?.timestamp_ms ?? null,
    outcome, source, VERSION, lastSeq,
  );
}

export function rebuildAllPredictions(db: Database): number {
  const keys = db.query<{ session_id: string; prediction_id: string }, []>(
    `SELECT DISTINCT session_id, json_extract(payload_json, '$.prediction_id') AS prediction_id
     FROM events WHERE (event_type LIKE 'prediction_%' OR event_type = 'heartbeat')
       AND json_valid(payload_json)
       AND typeof(json_extract(payload_json, '$.prediction_id')) = 'text'`,
  ).all();
  for (const row of keys) rebuildPrediction(db, row.session_id, row.prediction_id);
  return keys.length;
}
