"""Convert the pinned Continue Instinct TypeScript train shard.

Only local preparation is performed here. The native test shard is never read.
Rows remain marked as unreviewed for inferability and source-file licensing.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    "/mnt/ssd/tabcomplete-one-line-gpu-pilot-r1/raw/instinct-train-typescript-e557e3e.parquet"
)
DEFAULT_OUTPUT = Path("/mnt/ssd/tabcomplete-one-line-gpu-pilot-r1/prepared/instinct-v1")
DEFAULT_TOKENIZER = Path(
    "/home/crabcake/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B"
    "/snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)

DATASET_ID = "continuedev/instinct-data"
DATASET_REVISION = "e557e3ed1ea2c28b9c6b7cc46d670aa3cd451c29"
DATASET_LICENSE = "Apache-2.0"
SOURCE_SHA256 = "29b21939a3f1abc242a4b461105ba2444e310547755211117a82c397c53f3c3d"
TOKENIZER_REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
SCHEMA = "one-line-instinct-pilot-v1"
SOURCE_TYPE = "continue_instinct_observed"
MAX_INPUT_TOKENS = 1024
MAX_RESPONSE_TOKENS = 64
MAX_TOTAL_TOKENS = 2048

EXCERPT_HEADING = "### User Excerpt:"
EDITS_HEADING = "### User Edits:"
START = "<|editable_region_start|>"
END = "<|editable_region_end|>"
CURSOR = "<|user_cursor_is_here|>"

_SECRET_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b",
        r"\bsk-[A-Za-z0-9_-]{20,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bAIza[0-9A-Za-z_-]{30,}\b",
        r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b",
        r"\b(?:api[_-]?key|client[_-]?secret|password|passwd|secret[_-]?token)"
        r"\s*[:=]\s*[\"'][A-Za-z0-9/+_=-]{20,}[\"']",
    )
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_ts_path(value: str) -> bool:
    if not value or "\\" in value or "://" in value or "?" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and path.as_posix() == value
        and value.endswith((".ts", ".tsx"))
    )


def _path_line_to_value(line: str) -> str | None:
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _extract_excerpt(user_text: str) -> tuple[str, str] | None:
    if user_text.count(EXCERPT_HEADING) != 1:
        return None
    remainder = user_text.split(EXCERPT_HEADING, 1)[1]
    remainder = remainder.lstrip("\r\n")
    path_line, separator, rest = remainder.partition("\n")
    if not separator:
        return None
    path_line = path_line.removesuffix("\r")
    path = _path_line_to_value(path_line)
    if path is None or not _safe_relative_ts_path(path):
        return None
    # The blank line between the quoted path and source excerpt is framing.
    if rest.startswith("\r\n"):
        body = rest[2:]
    elif rest.startswith("\n"):
        body = rest[1:]
    else:
        body = rest
    return path, body


def _standalone_marker_lines(text: str, marker: str) -> list[tuple[int, int, int]]:
    """Return (marker start, marker end, full line end) for exact marker lines."""
    found: list[tuple[int, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.removesuffix("\n").removesuffix("\r")
        if content == marker:
            marker_start = offset
            marker_end = offset + len(marker)
            found.append((marker_start, marker_end, offset + len(line)))
        offset += len(line)
    return found


def _unmark_region(body: str) -> tuple[str, str, int, int] | None:
    starts = _standalone_marker_lines(body, START)
    ends = _standalone_marker_lines(body, END)
    candidates: list[tuple[str, str, int, int]] = []
    for start, _start_end, start_line_end in starts:
        for end, _end_end, end_line_end in ends:
            if start_line_end > end:
                continue
            marked_region = body[start_line_end:end]
            if marked_region.count(CURSOR) != 1:
                continue
            cursor_position = marked_region.index(CURSOR)
            region = (
                marked_region[:cursor_position] + marked_region[cursor_position + len(CURSOR) :]
            )
            prefix = body[:start]
            suffix = body[end_line_end:]
            cursor_byte_offset = len(prefix.encode("utf-8")) + len(
                marked_region[:cursor_position].encode("utf-8")
            )
            source = prefix + region + suffix
            candidates.append((source, region, cursor_byte_offset, len(prefix)))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _split_source_lines(source: str) -> list[tuple[str, str]]:
    """Return exact physical line content and terminator without normalization."""
    raw = source.encode("utf-8")
    if b"\r" in raw.replace(b"\r\n", b""):
        raise ValueError("lone CR is outside the one-line pilot contract")
    result: list[tuple[str, str]] = []
    start = 0
    index = 0
    while index < len(raw):
        if raw[index] == 10:
            crlf = index > start and raw[index - 1] == 13
            stop = index - 1 if crlf else index
            result.append(
                (
                    raw[start:stop].decode("utf-8"),
                    "\r\n" if crlf else "\n",
                )
            )
            start = index + 1
        index += 1
    if start < len(raw):
        result.append((raw[start:].decode("utf-8"), ""))
    return result


def _cursor_position(source: str, cursor_byte_offset: int) -> tuple[int, int]:
    raw = source.encode("utf-8")
    if not 0 <= cursor_byte_offset <= len(raw):
        raise ValueError("cursor offset outside source")
    preceding = raw[:cursor_byte_offset]
    row = preceding.count(b"\n")
    last_newline = preceding.rfind(b"\n")
    column = len(preceding) if last_newline < 0 else len(preceding) - last_newline - 1
    # A cursor after CRLF is measured from the next line; before LF, CR is not
    # a valid content column and will be rejected by EditState validation.
    return row, column


def _parse_diff_lines(diff_text: str) -> tuple[str, str] | None:
    additions: list[str] = []
    removals: list[str] = []
    in_hunk = False
    hunk_count = 0
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            hunk_count += 1
            continue
        if not in_hunk:
            continue
        if line.startswith("\\"):
            continue
        if line.startswith("+"):
            additions.append(line[1:])
        elif line.startswith("-"):
            removals.append(line[1:])
        elif line.startswith(" "):
            continue
        else:
            return None
    if hunk_count < 1 or len(additions) != 1 or len(removals) > 1:
        return None
    if not removals and not additions:
        return None
    return (removals[0] if removals else ""), additions[0]


def _history_entries(
    user_text: str, source_path: str, source: str
) -> tuple[list[dict[str, Any]], int]:
    if user_text.count(EDITS_HEADING) != 1:
        return [], 0
    section = user_text.split(EDITS_HEADING, 1)[1].split(EXCERPT_HEADING, 1)[0]
    records: list[tuple[str, str]] = []
    lines = section.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        prefix = "User edited file "
        if not line.startswith(prefix):
            index += 1
            continue
        edited_path = _path_line_to_value(line[len(prefix) :])
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines) or lines[index] != "```diff":
            continue
        index += 1
        diff_lines: list[str] = []
        while index < len(lines) and lines[index] != "```":
            diff_lines.append(lines[index])
            index += 1
        if index < len(lines):
            index += 1
        if isinstance(edited_path, str):
            records.append((edited_path, "\n".join(diff_lines)))

    source_lines = _split_source_lines(source)
    candidates: list[dict[str, Any]] = []
    for edited_path, diff_text in records:  # Prompt orders edits newest first.
        if edited_path != source_path:
            continue
        if not _safe_relative_ts_path(edited_path):
            continue
        change = _parse_diff_lines(diff_text)
        if change is None:
            continue
        old_text, new_text = change
        matches = [
            row for row, (content, _terminator) in enumerate(source_lines) if content == new_text
        ]
        if len(matches) != 1:
            continue
        candidates.append({"row": matches[0], "old_text": old_text, "new_text": new_text})
        if len(candidates) == 5:
            break
    # Instinct lists newest edits first. The TabComplete serializer expects
    # oldest-to-newest so it can retain the most recent history at the end.
    return list(reversed(candidates)), len(candidates)


def _action_at_cursor(source: str, after_source: str, target_row: int) -> dict[str, Any] | None:
    old_lines = _split_source_lines(source)
    new_lines = _split_source_lines(after_source)
    old_raw = [content + terminator for content, terminator in old_lines]
    new_raw = [content + terminator for content, terminator in new_lines]
    opcodes = [
        opcode
        for opcode in difflib.SequenceMatcher(None, old_raw, new_raw, autojunk=False).get_opcodes()
        if opcode[0] != "equal"
    ]
    if len(opcodes) != 1:
        return None
    tag, a0, a1, b0, b1 = opcodes[0]
    if tag == "replace" and a1 - a0 == b1 - b0 == 1 and a0 == target_row:
        return {"kind": "replace_line", "text": new_lines[b0][0]}
    if tag == "delete" and a1 - a0 == 1 and b0 == b1 and a0 == target_row:
        return {"kind": "delete_line", "text": None}
    if tag == "insert" and a0 == a1 and b1 - b0 == 1 and a0 == target_row:
        return {"kind": "insert_before", "text": new_lines[b0][0]}
    return None


def split_for_path(path: str) -> str:
    bucket = hashlib.sha256(("pilot-v1:" + path).encode("utf-8")).digest()[0]
    return "development" if bucket < 64 else "train"


def _secret_like(values: list[str]) -> bool:
    joined = "\n".join(values)
    return any(pattern.search(joined) is not None for pattern in _SECRET_PATTERNS)


def convert_messages(
    messages: Any, row_index: int, tokenizer: Any | None = None
) -> tuple[dict[str, Any] | None, str, dict[str, int]]:
    """Convert one row; return (row, outcome, token totals)."""
    counters: Counter[str] = Counter()
    if not isinstance(messages, list) or len(messages) != 3:
        return None, "invalid_message_shape", dict(counters)
    if [item.get("role") for item in messages if isinstance(item, dict)] != [
        "system",
        "user",
        "assistant",
    ]:
        return None, "invalid_message_roles", dict(counters)
    user_text = messages[1].get("content")
    assistant_text = messages[2].get("content")
    if not isinstance(user_text, str) or not isinstance(assistant_text, str):
        return None, "invalid_message_content", dict(counters)

    excerpt = _extract_excerpt(user_text)
    if excerpt is None:
        return None, "invalid_or_unsupported_path", dict(counters)
    file_path, body = excerpt
    unmarked = _unmark_region(body)
    if unmarked is None:
        return None, "invalid_or_ambiguous_markers", dict(counters)
    source, region, cursor_byte_offset, region_start = unmarked
    try:
        target_row, cursor_col = _cursor_position(source, cursor_byte_offset)
        # Construction validates the byte cursor and the complete excerpt.
        from tinycomplete.one_line.contract import EditState, RecentEdit

        history, history_count = _history_entries(user_text, file_path, source)
        state = EditState(
            file_id=file_path,
            filetype="typescript",
            source=source,
            target_row=target_row,
            cursor_col=cursor_col,
            history=tuple(RecentEdit(**entry) for entry in history),
        )
    except (UnicodeError, TypeError, ValueError):
        return None, "invalid_source_or_cursor", dict(counters)

    # The standalone end-marker line is framing. Preserve its preceding source
    # terminator outside the editable text and remove at most one matching
    # terminator from the assistant region. Do not strip any other whitespace.
    boundary = "\r\n" if region.endswith("\r\n") else "\n" if region.endswith("\n") else ""
    new_core = assistant_text
    if not assistant_text and boundary:
        # An empty assistant region represents removing the target's physical
        # line, including its terminator. A whitespace line instead arrives as
        # an explicit LF/CRLF and keeps the boundary below.
        boundary = ""
    elif boundary and new_core.endswith(boundary):
        new_core = new_core[: -len(boundary)]
    region_end = region_start + len(region)
    if source[region_start:region_end] != region:
        return None, "region_offset_mismatch", dict(counters)
    after_source = source[:region_start] + new_core + boundary + source[region_end:]
    action_raw = _action_at_cursor(source, after_source, target_row)
    if action_raw is None:
        return None, "not_one_line_at_cursor", dict(counters)

    from tinycomplete.one_line.contract import EditAction, apply_action

    try:
        action = EditAction(**action_raw)
        replayed = apply_action(state, action)
    except (TypeError, ValueError):
        return None, "invalid_action", dict(counters)
    if replayed != after_source:
        return None, "replay_mismatch", dict(counters)

    secret_scan = [source, assistant_text]
    secret_scan.extend(text for entry in history for text in (entry["old_text"], entry["new_text"]))
    if _secret_like(secret_scan):
        return None, "secret_like_material", dict(counters)
    if history_count == 0:
        return None, "no_unique_same_file_history", dict(counters)

    split = split_for_path(file_path)
    from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION

    row_id = hashlib.sha256(f"{DATASET_REVISION}:{row_index}:{file_path}".encode()).hexdigest()[:20]
    row: dict[str, Any] = {
        "id": f"continue-instinct-ts-{row_index:05d}-{row_id}",
        "split": split,
        "state": {
            "file_id": state.file_id,
            "filetype": state.filetype,
            "source": state.source,
            "target_row": state.target_row,
            "cursor_col": state.cursor_col,
            "history": [
                {"row": item.row, "old_text": item.old_text, "new_text": item.new_text}
                for item in state.history
            ],
            "relevant": [],
        },
        "action": action_raw,
        "after_source": after_source,
        "source_type": SOURCE_TYPE,
        "source_repo": (
            "continuedev/continue (publisher-declared project; row identity unavailable)"
        ),
        "source_revision": DATASET_REVISION,
        "source_license": DATASET_LICENSE,
        "license_provenance": (
            "Apache-2.0 is declared by the pinned dataset repository. Continue describes the "
            "original TypeScript rows as team edits to its open-source project; per-file "
            "path/revision and license scope are unverified."
        ),
        "session_or_commit": "unknown",
        "mechanism": "unreviewed_observed_edit",
        "generator_family": "Continue Instinct original TypeScript",
        "template_id": "continue-instinct-e557e3e",
        "provenance": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_row_index": row_index,
            "dataset_split": "train_typescript",
            "published_as_real_team_edit": True,
            "row_level_repository_commit_or_session": None,
        },
        "validation": {
            "replay_verified": True,
            "inferability_reviewed": False,
            "source_file_license_verified": False,
            "cursor_aligned": True,
            "one_line_action_verified": True,
            "context_policy_version": CONTEXT_POLICY_VERSION,
        },
    }
    token_totals: dict[str, int] = {}
    if tokenizer is not None:
        from tinycomplete.one_line.context import serialize_state_bounded
        from tinycomplete.one_line.train import encode_training_row

        try:
            serialized = serialize_state_bounded(
                state, tokenizer, max_input_tokens=MAX_INPUT_TOKENS
            )
            if serialized.included_history < 1:
                return None, "token_budget_dropped_history", dict(counters)
            encoded = encode_training_row(
                tokenizer,
                state,
                action,
                max_input_tokens=MAX_INPUT_TOKENS,
                max_total_tokens=MAX_TOTAL_TOKENS,
                max_action_tokens=MAX_RESPONSE_TOKENS,
            )
        except ValueError as error:
            message = str(error)
            if "input token budget" in message or "input" in message and "token" in message:
                return None, "input_token_limit", dict(counters)
            if "action" in message or "EOS" in message or "sequence" in message:
                return None, "response_or_total_token_limit", dict(counters)
            return None, "tokenization_error", dict(counters)
        token_totals = {
            "prompt_tokens": encoded.prompt_tokens,
            "response_tokens_including_eos": encoded.response_tokens,
            "total_tokens": encoded.total_tokens,
            "retained_history_count": serialized.included_history,
        }
        row["validation"]["token_counts"] = token_totals
    return row, "accepted", token_totals


def load_tokenizer(tokenizer_dir: Path) -> Any:
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError("local tokenizer.json is missing")
    if sha256_file(tokenizer_json) != TOKENIZER_SHA256:
        raise ValueError("local tokenizer SHA-256 does not match the pinned q25 tokenizer")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False
    )


def _iter_parquet(path: Path) -> Any:
    import pyarrow
    import pyarrow.parquet as parquet

    if pyarrow.__version__ != "25.0.1":
        raise ValueError("converter requires pyarrow==25.0.1")
    reader = parquet.ParquetFile(path)
    if reader.metadata.num_rows != 4371:
        raise ValueError("pinned train shard row count mismatch")
    for batch in reader.iter_batches(batch_size=128, columns=["messages"]):
        yield from batch.to_pylist()


def build(
    input_path: Path,
    output_dir: Path,
    tokenizer: Any,
    *,
    write: bool,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"local Instinct train shard not found: {input_path}")
    source_hash = sha256_file(input_path)
    if source_hash != SOURCE_SHA256:
        raise ValueError("pinned Continue Instinct train shard SHA-256 mismatch")
    counters: Counter[str] = Counter()
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "development": []}
    token_totals: dict[str, Counter[str]] = {
        "train": Counter(),
        "development": Counter(),
    }
    for row_index, example in enumerate(_iter_parquet(input_path)):
        counters["raw_train_rows"] += 1
        row, reason, tokens = convert_messages(example.get("messages"), row_index, tokenizer)
        counters[reason] += 1
        if row is None:
            continue
        split = row["split"]
        split_rows[split].append(row)
        token_totals[split].update(tokens)

    for split in split_rows:
        split_rows[split].sort(key=lambda row: row["id"])
    train_paths = {row["state"]["file_id"] for row in split_rows["train"]}
    dev_paths = {row["state"]["file_id"] for row in split_rows["development"]}
    disjoint = not train_paths.intersection(dev_paths)
    if not disjoint:
        raise ValueError("file path appears in both train and development")

    train_bytes = _jsonl_bytes(split_rows["train"])
    dev_bytes = _jsonl_bytes(split_rows["development"])
    converter_hash = sha256_file(Path(__file__))
    manifest = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "source_file_license_status": "unverified",
        "source_file": input_path.name,
        "source_sha256": source_hash,
        "converter_sha256": converter_hash,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "split_policy_id": "pilot-v1-sha256-path-byte0-lt-64",
        "context_policy_version": "single-line-context-v2",
        "input_token_limit": MAX_INPUT_TOKENS,
        "response_token_limit_including_eos": MAX_RESPONSE_TOKENS,
        "maximum_total_tokens": MAX_TOTAL_TOKENS,
        "train_sha256": sha256_bytes(train_bytes),
        "dev_sha256": sha256_bytes(dev_bytes),
        "development_sha256": sha256_bytes(dev_bytes),
        "train_count": len(split_rows["train"]),
        "dev_count": len(split_rows["development"]),
        "file_groups_disjoint": disjoint,
        "file_group_counts": {
            "train": len(train_paths),
            "development": len(dev_paths),
            "intersection": 0,
        },
        "split_policy": (
            "For relative file path p, SHA256(UTF-8('pilot-v1:' + p))[0] < 64 assigns "
            "development; all other files assign train. Native Instinct test split is excluded."
        ),
        "filtering_counters": dict(sorted(counters.items())),
        "token_totals": {key: dict(value) for key, value in token_totals.items()},
        "inferability_reviewed_count": 0,
        "source_file_license_verified_count": 0,
        "train_artifact": "train.jsonl",
        "dev_artifact": "dev.jsonl",
    }
    if write:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_exclusive(output_dir / "train.jsonl", train_bytes)
        _write_exclusive(output_dir / "dev.jsonl", dev_bytes)
        _write_exclusive(
            output_dir / "manifest.json",
            (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
    return manifest


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--build",
        action="store_true",
        help="write deterministic train/dev JSONL and manifest; otherwise print counts only",
    )
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    manifest = build(args.input, args.output_dir, tokenizer, write=args.build)
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
