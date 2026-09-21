#!/usr/bin/env bun
/**
 * *** WRITE SCRIPT *** — backup.ts copies the live collector database
 * (main + WAL + SHM sidecars) to a timestamped directory. Run this BEFORE
 * any other write operation against production data.
 *
 * It never opens the database (pure file copy), so it cannot corrupt it,
 * but it DOES write to the filesystem — hence the WRITE label.
 *
 * Usage:
 *   bun analysis/backup.ts [--db /path/collector.sqlite] [--dest /mnt/ssd/backups]
 */
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { DEFAULT_DB } from "./lib.ts";

console.log("*** WRITE SCRIPT: backup.ts — copies DB files to a timestamped directory ***");

function arg(name: string): string | undefined {
  const raw = Bun.argv.slice(2);
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] === `--${name}`) return raw[i + 1];
    if (raw[i]!.startsWith(`--${name}=`)) return raw[i]!.slice(name.length + 3);
  }
  return undefined;
}

const dbPath = arg("db") ?? DEFAULT_DB;
const destRoot = arg("dest") ?? "/mnt/ssd/backups";
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const dest = join(destRoot, `collector-backup-${stamp}`);
mkdirSync(dest, { recursive: true });

for (const suffix of ["", "-wal", "-shm"]) {
  const src = dbPath + suffix;
  try {
    statSync(src);
  } catch {
    console.log(`skip (absent): ${src}`);
    continue;
  }
  const base = src.split("/").pop()!;
  copyFileSync(src, join(dest, base));
  const bytes = statSync(join(dest, base)).size;
  console.log(`copied ${src} -> ${dest}/${base} (${bytes} bytes)`);
}
console.log(`BACKUP-OK: ${dest}`);
console.log(`restore with: cp ${dest}/collector.sqlite* "$(dirname ${dbPath})/"  (stop the collector first)`);
