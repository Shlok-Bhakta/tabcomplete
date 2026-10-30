#!/bin/sh
# End-to-end: run telemetry.lua against a scratch collector server (the real
# server code, a temporary database, 127.0.0.1 only) and check that every
# event the client queued was stored and none was rejected.
# Run from the repository root: sh tools/nes_engine/nvim/tests/collector_e2e.sh
set -eu
ROOT=$(pwd)
SERVER=$ROOT/tools/trajectory_collector/server
W=$(mktemp -d)
trap 'kill $PID 2>/dev/null; rm -rf "$W"' EXIT
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
cat > "$W/serve.ts" <<EOF
import { createApp } from "$SERVER/src/index";
const app = createApp("$W/collector.sqlite");
Bun.serve({ port: $PORT, hostname: "127.0.0.1", fetch: app.fetch });
console.log("ready");
EOF
bun "$W/serve.ts" > "$W/server.log" 2>&1 &
PID=$!
for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break; sleep 0.1; done
COLLECTOR_URL="http://127.0.0.1:$PORT" E2E_OUT="$W/out.json" \
  NES_CLIENT=tools/nes_engine/nvim COLLECTOR_CLIENT=tools/trajectory_collector/nvim \
  nvim --headless --clean -l tools/nes_engine/nvim/tests/telemetry.lua
python3 - "$W/collector.sqlite" "$W/out.json" <<'EOF'
import json, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
out = json.load(open(sys.argv[2]))
sid = out["session_id"]
stored = dict(db.execute("SELECT event_type, count(*) FROM events WHERE session_id = ? GROUP BY 1", (sid,)).fetchall())
want = out["counts"]
missing = {k: (v, stored.get(k, 0)) for k, v in want.items() if stored.get(k, 0) < v}
assert not missing, f"events not stored (queued, stored): {missing}"
print("stored events:", json.dumps(stored, sort_keys=True))
# Prediction payloads survive verbatim.
gen = [json.loads(r[0]) for r in db.execute(
    "SELECT payload_json FROM events WHERE session_id = ? AND event_type = 'prediction_generated'", (sid,))]
w = [g for g in gen if g.get("withheld_reason") == "low_confidence"][0]
assert w["withheld_edits"][0]["replaced_text"] == "foo(" and w["gate_threshold"] == -1.3, w
assert any(g.get("status") == "cancelled" for g in gen)
assert all("engine" in g for g in gen if g.get("status") in ("ok", "empty")), "engine identity stored"
blobs = {r[0] for r in db.execute("SELECT sha256 FROM blobs")}
req = [json.loads(r[0]) for r in db.execute(
    "SELECT payload_json FROM events WHERE session_id = ? AND event_type = 'prediction_requested'", (sid,))]
diag = [r["diagnostics_hash"] for r in req if r.get("diagnostics_hash")]
assert diag and all(h in blobs for h in diag), "diagnostics blob stored"
assert all(r["clipboard_hash"] in blobs for r in req if r.get("clipboard_hash")), "clipboard blob stored"
# Server-side projection reads the outcome labels.
rows = db.execute("SELECT outcome, outcome_source, count(*) FROM prediction_projection "
                  "WHERE session_id = ? AND shown_event_id IS NOT NULL GROUP BY 1, 2", (sid,)).fetchall()
print("projection outcomes:", rows)
assert all(o is not None and o != "dismissed_unknown" for o, _, _ in rows), rows
assert db.execute("SELECT count(*) FROM sessions WHERE session_id = ? AND ended_at IS NOT NULL", (sid,)).fetchone()[0] == 1
print("collector e2e: ok")
EOF
