FROM docker.io/oven/bun:1.3.14-slim

WORKDIR /app

COPY package.json bun.lock* ./
RUN bun install --production --frozen-lockfile || bun install --production

COPY src ./src
COPY healthcheck.ts ./
COPY tsconfig.json ./

ENV TABCOMPLETE_COLLECTOR_DB=/data/collector.sqlite
ENV TABCOMPLETE_COLLECTOR_PORT=8787

EXPOSE 8787

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD bun ./healthcheck.ts

CMD ["bun", "src/index.ts"]
