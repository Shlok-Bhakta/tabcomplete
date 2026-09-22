// Bounded event-window summaries, never scientific benchmark totals.
export function summarize(rows, now = Date.now()) {
  const unique = new Map();
  for (const row of rows) {
    const key = row['tabcomplete.terminal_event_id'] || `${row.trace_id}:${row.span_id}`;
    if (!unique.has(key)) unique.set(key, row);
  }
  const events = [...unique.values()].sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
  const runs = new Map(), models = new Map(), quality = new Map();
  const nextEdits = [];
  for (const row of events) {
    const id = row['tabcomplete.run_id'];
    if (row.name === 'eval.next_edit') nextEdits.push(row);
    if (row.name?.startsWith('run.')) {
      const previous = runs.get(id) || {};
      const merged = {...previous, ...Object.fromEntries(Object.entries(row).filter(([,v]) => v !== '' && v != null))};
      merged.started_at = previous.started_at || row.timestamp;
      merged.freshness = row['tabcomplete.historical'] ? 'historical' : now - Date.parse(row.timestamp) > 45000 ? 'stale / unknown' : 'fresh';
      if (!row['tabcomplete.historical'] && ['completed', 'failed'].includes(merged['tabcomplete.run.state'])) merged.freshness = 'terminal record';
      merged.elapsed_seconds = Math.round(((merged.freshness === 'terminal record' ? Date.parse(row.timestamp) : now) - Date.parse(merged.started_at))/1000);
      if (row['tabcomplete.historical']) merged.elapsed_seconds = null;
      runs.set(id, merged);
    }
    if (row.name === 'model.generate') {
      const key = [row['gen_ai.request.model'], row['tabcomplete.suite.sha256'], row['tabcomplete.protocol']].join(' / ');
      const group = models.get(key) || {model_suite_protocol:key, calls:0, durations:[], first:[], input_tokens:null, output_tokens:null, truncated:0};
      group.calls++;
      const duration = Number(row['tabcomplete.timing.total_ms']);
      if (row['tabcomplete.timing.total_ms'] !== '' && Number.isFinite(duration) && duration > 0) group.durations.push(duration);
      for (const kind of ['input', 'output']) {
        const v = row[`gen_ai.usage.${kind}_tokens`];
        if (v != null && v !== '') group[`${kind}_tokens`] = (group[`${kind}_tokens`] || 0) + Number(v);
      }
      const first = row['tabcomplete.timing.first_output_ms'];
      if (first != null && first !== '') group.first.push(Number(first));
      if (row['tabcomplete.output.truncated'] === true) group.truncated++;
      models.set(key, group);
    }
    if (row.name === 'eval.case' && ['parse','compile','functional'].some(k => row[`tabcomplete.quality.${k}`] != null && row[`tabcomplete.quality.${k}`] !== '')) {
      const key = [row['tabcomplete.suite.sha256'] || 'unknown suite', row['tabcomplete.protocol'] || 'unknown protocol'].join(' / ');
      const group = quality.get(key) || {suite_protocol:key, cases:0, parse:0, compile:0, functional:0, parse_n:0, compile_n:0, functional_n:0};
      group.cases++;
      for (const kind of ['parse','compile','functional']) {
        const v = row[`tabcomplete.quality.${kind}`];
        if (v !== undefined && v !== null && v !== '') { group[`${kind}_n`]++; if (v === true || v === 'passed' || v === 'pass') group[kind]++; }
      }
      quality.set(key, group);
    }
  }
  const percentile = (a,p) => a.length ? [...a].sort((x,y)=>x-y)[Math.max(0, Math.ceil(a.length*p)-1)].toFixed(2) : null;
  return {
    runs:[...runs.values()].reverse(),
    models:[...models.values()].map(g=>({...g,p50_ms:percentile(g.durations,.5),p95_ms:percentile(g.durations,.95),p99_ms:percentile(g.durations,.99),first_output_p50_ms:percentile(g.first,.5),truncation:`${g.truncated}/${g.calls}`})),
    quality:[...quality.values()].map(g=>({...g,parse:`${g.parse}/${g.parse_n}`,compile:`${g.compile}/${g.compile_n}`,functional:`${g.functional}/${g.functional_n}`})),
    nextEdits,
    failures:events.filter(r=>r['tabcomplete.failure.type'] || r['tabcomplete.outcome']==='failed').reverse(),
    training:events.filter(r=>r.name?.startsWith('training.') || r.name?.startsWith('checkpoint.') || (r['tabcomplete.historical'] && r['tabcomplete.training.input_tokens'] != null)).reverse(),
    infrastructure:events.filter(r=>r.name==='storage.health').reverse(),
  };
}
