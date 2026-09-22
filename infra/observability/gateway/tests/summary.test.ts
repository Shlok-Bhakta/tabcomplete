import { expect, test } from 'bun:test';
import { summarize } from '../public/summary.js';

test('SigNoz null fields do not count generation-only spans as quality cases',()=>{
  const row={name:'eval.case',timestamp:new Date().toISOString(),trace_id:'trace',span_id:'generation',
    'tabcomplete.suite.sha256':'suite','tabcomplete.protocol':'v1',
    'tabcomplete.quality.parse':null,'tabcomplete.quality.compile':null,'tabcomplete.quality.functional':null};
  const result=summarize([row,{...row,span_id:'quality','tabcomplete.quality.functional':'pass'}]);
  expect(result.quality[0].cases).toBe(1);
  expect(result.quality[0].functional).toBe('1/1');
});

test('deduplicates terminal cases and separates incompatible suites',()=>{
  const row = {name:'eval.case',timestamp:new Date().toISOString(),trace_id:'trace',span_id:'span', 'tabcomplete.terminal_event_id':'case-terminal','tabcomplete.suite.sha256':'suite-a','tabcomplete.protocol':'v1','tabcomplete.quality.functional':'pass'};
  const summary = summarize([row,row,{...row,'tabcomplete.terminal_event_id':'second','tabcomplete.suite.sha256':'suite-b'}]);
  expect(summary.quality).toHaveLength(2);
  expect(summary.quality[0].cases).toBe(1);
});
test('missing usage and first output stay unknown',()=>{
  const summary = summarize([{name:'model.generate',trace_id:'t',span_id:'s',timestamp:new Date().toISOString()}]);
  expect(summary.models[0].input_tokens).toBeNull();
  expect(summary.models[0].first_output_p50_ms).toBeNull();
});
test('absent heartbeat is stale, not failed',()=>{
  const summary = summarize([{name:'run.start',trace_id:'t',span_id:'s',timestamp:'2020-01-01T00:00:00Z','tabcomplete.run_id':'r','tabcomplete.run.state':'started'}]);
  expect(summary.runs[0].freshness).toBe('stale / unknown');
  expect(summary.runs[0]['tabcomplete.run.state']).toBe('started');
});
test('historical summaries never imply live elapsed training duration',()=>{
  const result=summarize([{name:'run.summary',trace_id:'t',span_id:'s',timestamp:new Date().toISOString(),
    'tabcomplete.run_id':'history','tabcomplete.run.state':'completed','tabcomplete.historical':true,
    'tabcomplete.training.input_tokens':100}]);
  expect(result.runs[0].freshness).toBe('historical');
  expect(result.runs[0].elapsed_seconds).toBeNull();
  expect(result.training).toHaveLength(1);
});
