FROM oven/bun:1.3.14-slim@sha256:621f249399228db47cf34611ee662585e77e015250ed29d5d0932b2d3282f0b0

WORKDIR /app
COPY package.json ./
COPY src ./src
COPY public ./public
USER bun
EXPOSE 9090
CMD ["bun", "src/server.ts"]
