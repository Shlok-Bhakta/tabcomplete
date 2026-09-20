FROM docker.io/oven/bun:1.3.14-slim

WORKDIR /app

COPY package.json bun.lock* ./
RUN bun install --production --frozen-lockfile || bun install --production

COPY src ./src
COPY tsconfig.json ./

ENV TABCOMPLETE_COLLECTOR_DB=/data/collector.sqlite
ENV TABCOMPLETE_COLLECTOR_PORT=8787

EXPOSE 8787

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD bun -e "const r = await fetch('http://127.0.0.1:8787/healthz'); if (!r.ok) process.exit(1)"

CMD ["bun", "src/index.ts"]
