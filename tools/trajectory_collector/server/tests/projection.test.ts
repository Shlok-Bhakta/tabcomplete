import { describe, expect, test } from "bun:test";
import { openDatabase } from "../src/db";
import { migrate } from "../src/migrations";
import { rebuildAllPredictions, rebuildPrediction } from "../src/projection";

describe("prediction projection", () => {
  test("rebuilds out-of-order and duplicate raw events without inventing a second outcome", () => {
    const db = openDatabase(":memory:");
    const stamp = "2026-09-25T00:00:00Z";
    db.prepare("INSERT INTO machines(machine_id,first_seen_at,last_seen_at) VALUES (?,?,?)")
      .run("machine", stamp, stamp);
    db.prepare("INSERT INTO sessions(session_id,machine_id,started_at) VALUES (?,?,?)")
      .run("session", "machine", stamp);
    const insert = db.prepare(`INSERT OR IGNORE INTO events
      (event_id,session_id,sequence_number,event_type,timestamp_ms,payload_json)
      VALUES (?,?,?,?,?,?)`);
    const add = (id: string, seq: number, type: string, payload: object) =>
      insert.run(id, "session", seq, type, seq * 1000, JSON.stringify({ prediction_id: "p", ...payload }));
    add("dismiss", 4, "prediction_dismissed", {
      outcome: "rejected_implicit_typing", outcome_source: "editor_observation", ended_by_event_id: "delta",
    });
    add("shown", 3, "prediction_shown", { file: "/synthetic/main.py", action_blob_hash: "a".repeat(64) });
    add("generated", 2, "prediction_generated", { action_blob_hash: "a".repeat(64) });
    add("request", 1, "prediction_requested", {
      file_identity: "repo:test:main.py", model_revision: "q25",
      pre_state_hash: "b".repeat(64), pre_state_sequence: 0,
    });
    add("dismiss", 4, "prediction_dismissed", { outcome: "rejected_explicit" });
    expect(rebuildAllPredictions(db)).toBe(1);
    const row = db.query<{outcome: string;request_event_id: string;shown_event_id: string;
      last_outcome_event_id: string;updated_through_sequence: number}, []>(
      "SELECT outcome,request_event_id,shown_event_id,last_outcome_event_id,updated_through_sequence FROM prediction_projection",
    ).get();
    expect(row).toEqual({ outcome: "rejected_implicit_typing", request_event_id: "request",
      shown_event_id: "shown", last_outcome_event_id: "dismiss", updated_through_sequence: 4 });
    expect(rebuildAllPredictions(db)).toBe(1);
    expect(migrate(db).applied).toEqual([]);
    add("accepted", 5, "prediction_accepted", {});
    rebuildPrediction(db, "session", "p");
    expect(db.query<{outcome: string}, []>("SELECT outcome FROM prediction_projection").get()?.outcome)
      .toBe("rejected_implicit_typing");
    db.close();
  });
});
