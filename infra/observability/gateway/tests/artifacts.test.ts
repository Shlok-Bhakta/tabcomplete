import {expect,test} from 'bun:test';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {createHandler,pruneArtifacts} from '../src/server';

test('authenticated artifact upload deduplicates, preserves text, and expires only copies',async()=>{
 const root=await mkdtemp(join(tmpdir(),'tabcomplete-artifact-test-'));
 try {
  const handler=createHandler({signozUrl:'http://unused',apiKey:'fixture',allowedHosts:new Set(['kiwi']),artifactRoot:root,uploadToken:'fixture-upload',minimumFreeBytes:0});
  const text='  <script>neverExecute()</script>\n\treturn 1\n';
  const hash=createHash('sha256').update(text).digest('hex');
  const upload=()=>handler(new Request(`http://kiwi/internal/v1/artifacts/${hash}`,{method:'PUT',headers:{Host:'kiwi',Authorization:'Bearer fixture-upload'},body:text}));
  expect((await upload()).status).toBe(201);
  expect((await (await upload()).json()).deduplicated).toBe(true);
  const response=await handler(new Request(`http://kiwi/api/v1/artifacts/${hash}`,{headers:{Host:'kiwi'}}));
  expect(response.headers.get('content-type')).toContain('text/plain');
  expect(await response.text()).toBe(text);
  const bad=await handler(new Request(`http://kiwi/internal/v1/artifacts/${hash}`,{method:'PUT',headers:{Host:'kiwi'},body:text}));
  expect(bad.status).toBe(401);
  expect(await pruneArtifacts(root,Date.now()+31*86400000)).toBe(1);
 } finally {await rm(root,{recursive:true,force:true});}
});

test('rejects cross-port browser origins',async()=>{
 const response=await createHandler({signozUrl:'http://unused',apiKey:'fixture',allowedHosts:new Set(['kiwi']),artifactRoot:'/unused'})(new Request('http://kiwi/api/v1/status',{headers:{Host:'kiwi:9090',Origin:'http://kiwi:8080'}}));
 expect(response.status).toBe(403);
});
