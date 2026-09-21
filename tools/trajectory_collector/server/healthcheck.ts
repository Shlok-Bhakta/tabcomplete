// Container healthcheck: exit 0 only when /healthz reports ok.
// Kept in a file (not `bun -e`) because podman-compose mangles multi-arg
// CMD arrays through /bin/sh, which chokes on JS parentheses.
const res = await fetch("http://127.0.0.1:8787/healthz");
if (!res.ok) process.exit(1);
const body = (await res.json()) as { status?: string };
if (body?.status !== "ok") process.exit(1);
