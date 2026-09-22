import { summarize } from './summary.js';
const byId = (id) => document.getElementById(id);

function text(element, value) {
  element.textContent = value == null ? "unknown" : String(value);
}

function value(row, key) {
  return row[key] ?? row.attributes?.[key] ?? null;
}

function table(rows, columns, onSelect) {
  const root = document.createElement("div");
  root.className = "table-wrap";
  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const [label] of columns) {
    const cell = document.createElement("th");
    text(cell, label);
    head.append(cell);
  }
  table.append(head);
  for (const row of rows) {
    const line = document.createElement("tr");
    if (onSelect) {
      line.tabIndex = 0;
      line.addEventListener("click", () => onSelect(row));
    }
    for (const [, key] of columns) {
      const cell = document.createElement("td");
      text(cell, value(row, key));
      line.append(cell);
    }
    table.append(line);
  }
  root.append(table);
  return root;
}

async function showDetail(row) {
  const id = value(row, 'tabcomplete.request_id') || value(row, 'tabcomplete.run_id');
  if (id) {
    const kind = value(row, 'tabcomplete.request_id') ? 'requests' : 'runs';
    const response = await fetch(`/api/v1/${kind}/${encodeURIComponent(id)}?since=14d&limit=200`);
    const result = await response.json();
    const related = result.data || [];
    row = Object.assign({}, ...related.reverse(), row);
    row.related_events = related;
    const run = value(row,'tabcomplete.run_id');
    if (run) {
      const logs = await fetch(`/api/v1/runs/${encodeURIComponent(run)}/logs?since=14d&limit=50`);
      row.related_logs = logs.ok ? (await logs.json()).data : 'Logs query unavailable';
    }
  }
  const root = byId("detail");
  root.replaceChildren();
  root.classList.remove("empty");
  const fields = [
    "tabcomplete.run_id", "tabcomplete.case_id", "tabcomplete.request_id", "trace_id",
    "gen_ai.request.model", "gen_ai.request.max_tokens", "tabcomplete.failure.type",
    "tabcomplete.failure.message", "tabcomplete.artifact.input.preview",
    "tabcomplete.artifact.output.preview", "tabcomplete.decoding", "tabcomplete.protocol",
    "tabcomplete.suite.sha256", "related_events", "related_logs",
  ];
  for (const key of fields) {
    const line = document.createElement("div");
    const label = document.createElement("strong");
    text(label, key);
    const content = document.createElement("pre");
    const field = value(row,key);
    text(content, typeof field === 'object' && field ? JSON.stringify(field,null,2) : field);
    line.append(label, content);
    root.append(line);
  }
  for (const kind of ["input", "output"]) {
    const hash = value(row, `tabcomplete.artifact.${kind}.sha256`);
    if (hash) {
      const link = document.createElement("a");
      link.href = `/api/v1/artifacts/${encodeURIComponent(hash)}`;
      text(link, `Open full ${kind}`);
      root.append(link);
    }
  }
}

async function refresh() {
  try {
    const [statusResponse, runsResponse] = await Promise.all([
      fetch("/api/v1/status"),
      fetch("/api/v1/events?since=24h&limit=200"),
    ]);
    const status = await statusResponse.json();
    const runs = await runsResponse.json();
    if (!runsResponse.ok) throw new Error('query failed');
    text(byId("health"), status.status === "healthy" ? "SigNoz reachable; workload freshness below" : "Telemetry unavailable");
    text(byId("refreshed"), `Refreshed ${new Date().toLocaleTimeString()} · Latest event ${runs.last_event_at ? new Date(runs.last_event_at).toLocaleString() : 'unknown'} · Ingestion time unavailable${runs.page?.next_cursor ? ' · PARTIAL: latest 200 events' : ''}`);
    const summary = summarize(runs.data || []);
    const root = byId("runs");
    root.replaceChildren();
    if (!runs.data?.length) {
      root.className = "panel empty";
      text(root, "No runs have been ingested in the last 24 hours.");
      return;
    }
    root.className = "panel";
    root.append(table(summary.runs, [
      ["Run", "tabcomplete.run_id"], ["State", "tabcomplete.run.state"],
      ["Model", "gen_ai.request.model"], ["Host", "host.name"],
      ["Completed", "tabcomplete.cases.completed"], ["Planned", "tabcomplete.cases.planned"],
      ["Last event", "timestamp"], ["Phase", "tabcomplete.phase"], ["Failures", "tabcomplete.cases.failed"],
      ["Freshness", "freshness"], ["Started", "started_at"], ["Elapsed (s)", "elapsed_seconds"],
    ], showDetail));
    for (const [id, rows, columns, empty] of [
      ['models', summary.models, [['Model / suite / protocol','model_suite_protocol'],['Calls','calls'],['p50 ms','p50_ms'],['p95 ms','p95_ms'],['p99 ms','p99_ms'],['First output p50 ms','first_output_p50_ms'],['Known input tokens','input_tokens'],['Known output tokens','output_tokens'],['Truncated','truncation']], 'No model calls in this event window.'],
      ['quality', summary.quality, [['Suite / protocol','suite_protocol'],['Observed cases','cases'],['Parse pass / observed','parse'],['Compile pass / observed','compile'],['Functional pass / observed','functional']], 'No quality records in this window. Scientific result files remain authoritative.'],
      ['training', summary.training, [['Run','tabcomplete.run_id'],['Event','name'],['Loss','tabcomplete.training.loss'],['Validation NLL','tabcomplete.training.validation_nll'],['LR','tabcomplete.training.lr'],['Gradient norm','tabcomplete.training.gradient_norm'],['Input tokens','tabcomplete.training.input_tokens'],['Updates','tabcomplete.training.successful_updates']], 'No training telemetry in this window.'],
      ['infrastructure', summary.infrastructure, [['Host','host.name'],['Storage healthy','tabcomplete.storage.healthy'],['Free bytes','tabcomplete.storage.free_bytes'],['Artifact bytes','tabcomplete.storage.artifact_bytes']], 'Storage and collector signals not yet observed; backend availability alone is not end-to-end health.'],
    ]) {
      const target = byId(id); target.replaceChildren(); target.className = 'panel';
      if (rows.length) target.append(table(rows,columns,showDetail)); else text(target,empty);
    }
    const failures = summary.failures;
    if (summary.nextEdits.length) byId('quality').append(table(summary.nextEdits, [
      ['Run','tabcomplete.run_id'],['Case','tabcomplete.case_id'],
      ['Suite','tabcomplete.suite.sha256'],['Protocol','tabcomplete.protocol'],
      ['Action','tabcomplete.next_edit.predicted_action'],['Correct','tabcomplete.next_edit.action_correct'],
      ['Deletion success','tabcomplete.next_edit.deletion_success'],['Valid patch','tabcomplete.next_edit.valid_patch'],
    ], showDetail));
    const infraResponse = await fetch('/api/v1/infrastructure');
    if (infraResponse.ok) {
      const infra = await infraResponse.json();
      const target=byId('infrastructure');target.replaceChildren();
      target.append(table([infra.storage],[['Volume state','state'],['Checked','checked_at'],['Free bytes','available_bytes'],['Database bytes','clickhouse_bytes'],['Artifact bytes','artifact_bytes'],['Collector WAL bytes','collector_wal_bytes'],['100 GiB warning','high_water_warning']]));
      if (infra.collectors?.metrics?.length) target.append(table(infra.collectors.metrics,[['Collector metric, latest in 5m','name'],['Value, unknown if absent','value']]));
    }
    const failureRoot = byId("failures");
    failureRoot.replaceChildren();
    if (failures.length) {
      failureRoot.className = "panel";
      failureRoot.append(table(failures, [
        ["Run", "tabcomplete.run_id"], ["Case", "tabcomplete.case_id"],
        ["Type", "tabcomplete.failure.type"], ["Message", "tabcomplete.failure.message"],
      ], showDetail));
    }
  } catch {
    text(byId("health"), "Telemetry query failed");
  }
}

refresh();
setInterval(refresh, 5000);
