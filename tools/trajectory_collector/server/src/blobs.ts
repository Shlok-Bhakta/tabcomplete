import { createHash, timingSafeEqual } from "node:crypto";
import { gzipSync, gunzipSync } from "node:zlib";
import type { Database } from "bun:sqlite";

export function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

/** Magic-byte prefixes that indicate bytes are already compressed/encoded media. */
const ALREADY_COMPRESSED_PREFIXES: Uint8Array[] = [
  Uint8Array.from([0x1f, 0x8b]), // gzip
  Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), // png
  Uint8Array.from([0xff, 0xd8, 0xff]), // jpeg
  Uint8Array.from([0x50, 0x4b, 0x03, 0x04]), // zip/jar/docx
  Uint8Array.from([0x50, 0x4b, 0x05, 0x06]), // zip empty
  Uint8Array.from([0x50, 0x4b, 0x07, 0x08]), // zip spanned
  Uint8Array.from([0x42, 0x5a, 0x68]), // bzip2
  Uint8Array.from([0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00]), // xz
  Uint8Array.from([0x28, 0xb5, 0x2f, 0xfd]), // zstd
  Uint8Array.from([0x47, 0x49, 0x46, 0x38]), // gif
  Uint8Array.from([0x52, 0x49, 0x46, 0x46]), // webp (RIFF....WEBP)
  Uint8Array.from([0x25, 0x50, 0x44, 0x46]), // pdf
  Uint8Array.from([0x00, 0x00, 0x01, 0x00]), // ico
];

function startsWith(haystack: Uint8Array, needle: Uint8Array): boolean {
  if (haystack.length < needle.length) return false;
  for (let i = 0; i < needle.length; i++) {
    if (haystack[i] !== needle[i]) return false;
  }
  return true;
}

export function looksAlreadyCompressed(bytes: Uint8Array): boolean {
  for (const prefix of ALREADY_COMPRESSED_PREFIXES) {
    if (startsWith(bytes, prefix)) return true;
  }
  return false;
}

export type StoredCompression = "gzip" | "raw";

export function encodeForStorage(
  original: Uint8Array,
  opts?: { alreadyCompressed?: boolean | null; mimeType?: string | null },
): { stored: Uint8Array; compression: StoredCompression } {
  const hint = opts?.alreadyCompressed === true;
  const mime = (opts?.mimeType ?? "").toLowerCase();
  const mimeSaysCompressed =
    mime.includes("gzip") ||
    mime.includes("zip") ||
    mime.includes("image/") ||
    mime.includes("video/") ||
    mime.includes("audio/") ||
    mime.includes("octet-stream");
  if (hint || mimeSaysCompressed || looksAlreadyCompressed(original)) {
    return { stored: original, compression: "raw" };
  }
  const gzipped = gzipSync(original);
  return { stored: new Uint8Array(gzipped), compression: "gzip" };
}

export function decodeFromStorage(stored: Uint8Array, compression: string): Uint8Array {
  if (compression === "gzip") {
    return new Uint8Array(gunzipSync(stored));
  }
  return stored;
}

export interface BlobUploadResult {
  sha256: string;
  original_bytes: number;
  stored_bytes: number;
  compression: StoredCompression;
  deduped: boolean;
}

function nowIso(): string {
  return new Date().toISOString();
}

function constantTimeHexEqual(a: string, b: string): boolean {
  const al = a.toLowerCase();
  const bl = b.toLowerCase();
  if (al.length !== bl.length) return false;
  const ab = Buffer.from(al, "utf8");
  const bb = Buffer.from(bl, "utf8");
  if (ab.length !== bb.length) return false;
  return timingSafeEqual(ab, bb);
}

export function storeBlob(
  db: Database,
  expectedSha256: string,
  original: Uint8Array,
  opts?: { alreadyCompressed?: boolean | null; mimeType?: string | null },
): BlobUploadResult {
  const actual = sha256Hex(original);
  if (!constantTimeHexEqual(actual, expectedSha256)) {
    throw Object.assign(new Error(`sha256 mismatch: claimed ${expectedSha256} but content hashes to ${actual}`), {
      code: "sha_mismatch",
      status: 400,
    });
  }
  const normalized = expectedSha256.toLowerCase();
  const existing = db
    .query<{ sha256: string; original_bytes: number; stored_bytes: number; compression: string }, [string]>
    ("SELECT sha256, original_bytes, stored_bytes, compression FROM blobs WHERE sha256 = ?")
    .get(normalized);
  if (existing) {
    return {
      sha256: normalized,
      original_bytes: existing.original_bytes,
      stored_bytes: existing.stored_bytes,
      compression: existing.compression as StoredCompression,
      deduped: true,
    };
  }
  const { stored, compression } = encodeForStorage(original, opts);
  db.prepare(
    `INSERT OR IGNORE INTO blobs (sha256, original_bytes, stored_bytes, compression, content, created_at)
     VALUES (?, ?, ?, ?, ?, ?)`,
  ).run(normalized, original.length, stored.length, compression, stored, nowIso());
  const after = db
    .query<{ original_bytes: number; stored_bytes: number; compression: string }, [string]>
    ("SELECT original_bytes, stored_bytes, compression FROM blobs WHERE sha256 = ?")
    .get(normalized);
  // INSERT OR IGNORE lost a race: treat as deduped.
  if (!after) {
    throw new Error("failed to store blob");
  }
  const wasRace = after.stored_bytes !== stored.length || after.compression !== compression;
  return {
    sha256: normalized,
    original_bytes: after.original_bytes,
    stored_bytes: after.stored_bytes,
    compression: after.compression as StoredCompression,
    deduped: wasRace,
  };
}

export function findMissingHashes(db: Database, hashes: string[]): string[] {
  const normalized = hashes.map((h) => h.toLowerCase());
  if (normalized.length === 0) return [];
  const placeholders = normalized.map(() => "?").join(",");
  const rows = db
    .query<{ sha256: string }, string[]>(
      `SELECT sha256 FROM blobs WHERE sha256 IN (${placeholders})`,
    )
    .all(...normalized);
  const present = new Set(rows.map((r) => r.sha256.toLowerCase()));
  return normalized.filter((h) => !present.has(h));
}
