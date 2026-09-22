// Fixed, tailnet-only reverse proxy. No credentials and no telemetry database.
const upstream = new URL(process.env.TABCOMPLETE_ALIAS_UPSTREAM!);
const bind = process.env.TABCOMPLETE_ALIAS_BIND!;
const hosts = new Set((process.env.TABCOMPLETE_ALIAS_HOSTS || '').split(','));
if (!bind || !hosts.size || upstream.protocol !== 'http:') throw new Error('Invalid alias configuration');
Bun.serve({hostname:bind,port:Number(process.env.PORT || 9090),async fetch(request) {
  const host = request.headers.get('host') || '';
  if (!hosts.has(host.split(':')[0])) return new Response('Host not allowed',{status:421});
  const origin = request.headers.get('origin');
  if (origin && new URL(origin).host !== host) return new Response('Origin not allowed',{status:403});
  const path = new URL(request.url);
  if (request.method !== 'GET' || path.pathname.startsWith('/internal/')) return new Response('Method not allowed',{status:405});
  const target = new URL(upstream); target.pathname=path.pathname; target.search=path.search;
  try {
    return await fetch(target,{headers:{Host:upstream.host},redirect:'manual',signal:AbortSignal.timeout(12000)});
  } catch { return new Response('Monitoring gateway unavailable',{status:502}); }
}});
