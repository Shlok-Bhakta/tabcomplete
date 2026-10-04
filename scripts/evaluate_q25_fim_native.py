"""Verify the selected FIM model through the real Rust context and SSE route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from evaluate_q25_fim import decoded_completion, development_case
from measure_r2_local import NativeProvider
from prepare_q25_fim import (
    _load_pinned_inputs,
    _token_ids,
    _truncate_prefix,
    _truncate_suffix,
)

from tinycomplete.observability.context import RunContext
from tinycomplete.observability.runs import run_scope
from tinycomplete.observability.spans import operation

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "q25-fim-native-evaluation-v1"
PROTOCOL = "q25-fim-line-completion-v1"
MODE = "remaining_logical_line_after_utf8_cursor"
TOKENIZER_ID = "Qwen/Qwen2.5-Coder-0.5B"
TOKENIZER_REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
TOKENIZER_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
EOS_TOKEN_ID = 151643
FIM_TOKEN_IDS = {
    "eos": EOS_TOKEN_ID,
    "fim_prefix": 151659,
    "fim_suffix": 151661,
    "fim_middle": 151660,
}
FIM_TOKEN_SPELLINGS = {
    EOS_TOKEN_ID: "<|endoftext|>",
    151659: "<|fim_prefix|>",
    151661: "<|fim_suffix|>",
    151660: "<|fim_middle|>",
}
FIM_ACTION_POLICY = "q25-fim-completion-v1"
CONTEXT_POLICY_VERSION = "q25-fim-psm-cursor-to-line-end-bounded640-256-v2"
CONTEXT_LAYOUT = "q25-fim-psm-bounded-v2"
PREFIX_CONTEXT_TOKEN_LIMIT = 640
SUFFIX_CONTEXT_TOKEN_LIMIT = 256
FIM_QUALITY_CODES = frozenset(
    {
        "fim_missing_eos",
        "fim_terminal_not_eos",
        "fim_output_cap_exceeded",
        "fim_token_id_outside_vocabulary",
        "fim_control_token_in_body",
        "fim_nul_in_body",
        "fim_literal_control_spelling",
        "fim_line_ending_mismatch",
        "fim_multiline_completion",
    }
)
PROC_ROOT = Path("/proc")
SOURCE_FILES = (
    "scripts/evaluate_q25_fim_native.py",
    "scripts/evaluate_q25_fim.py",
    "scripts/install_small_model_lazyvim.py",
    "scripts/measure_r2_local.py",
    "tools/tabcomplete_engine/Cargo.lock",
    "tools/tabcomplete_engine/src/main.rs",
    "tools/tabcomplete_engine/src/fim_v1.rs",
    "tools/tabcomplete_engine/src/context.rs",
)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise ValueError("native evaluation JSONL contains a blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError("native evaluation JSONL contains invalid JSON") from None
        if not isinstance(row, dict):
            raise ValueError("native evaluation JSONL rows must be objects")
        rows.append(row)
    return rows


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _socket_inodes_for_port(path: Path, port: int) -> set[str]:
    try:
        rows = path.read_text(encoding="ascii").splitlines()[1:]
    except OSError:
        raise ValueError("native service process socket table is unavailable") from None
    inodes: set[str] = set()
    for row in rows:
        fields = row.split()
        if len(fields) < 10 or fields[3] != "0A":
            continue
        try:
            local_port = int(fields[1].rsplit(":", 1)[1], 16)
        except (IndexError, ValueError):
            continue
        if local_port == port:
            inodes.add(fields[9])
    return inodes


def _process_socket_inodes(process: Path) -> set[str]:
    descriptors = process / "fd"
    try:
        entries = list(descriptors.iterdir())
    except OSError:
        raise ValueError("native service process descriptors are unavailable") from None
    inodes = set()
    for entry in entries:
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        match = re.fullmatch(r"socket:\[(\d+)\]", target)
        if match:
            inodes.add(match.group(1))
    return inodes


def _process_start_ticks(process: Path) -> int:
    try:
        stat = (process / "stat").read_text(encoding="ascii")
        fields = stat[stat.rfind(")") + 2 :].split()
        value = int(fields[19])
    except (OSError, ValueError, IndexError):
        raise ValueError("native service process start identity is unavailable") from None
    if value < 0:
        raise ValueError("native service process start identity is invalid")
    return value


def attest_process(
    pid: int, binary_sha256: str, port: int, *, proc_root: Path = PROC_ROOT
) -> dict:
    process = proc_root / str(pid)
    if pid <= 0 or not 1 <= port <= 65535 or sha(process / "exe") != binary_sha256:
        raise ValueError("running native executable differs from the frozen binary")
    listening = _socket_inodes_for_port(process / "net" / "tcp", port)
    listening |= _socket_inodes_for_port(process / "net" / "tcp6", port)
    owned = listening & _process_socket_inodes(process)
    if not owned:
        raise ValueError("native service PID does not own the configured listening port")
    return {
        "pid": pid,
        "binary_sha256": binary_sha256,
        "executable": str((process / "exe").resolve()),
        "process_start_ticks": _process_start_ticks(process),
        "listening_port": port,
        "listening_socket_inodes": sorted(owned, key=int),
    }


def memory_snapshot(pid: int) -> dict:
    process = Path("/proc") / str(pid)
    values = {}
    for name, fields in (
        ("smaps_rollup", {"Rss", "Pss", "Pss_Anon", "Pss_File", "Anonymous", "Swap"}),
        ("status", {"VmHWM", "VmRSS", "VmSwap"}),
    ):
        for line in (process / name).read_text().splitlines():
            key, _, value = line.partition(":")
            if key in fields:
                values[key + "_bytes"] = int(value.split()[0]) * 1024
    return values


def load_tokenizer(directory: Path):
    from transformers import AutoTokenizer

    if directory.name != TOKENIZER_REVISION or sha(directory / "tokenizer.json") != TOKENIZER_SHA:
        raise ValueError("native evaluation tokenizer identity differs")
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id != EOS_TOKEN_ID:
        raise ValueError("native evaluation tokenizer EOS identity differs")
    for token_id, spelling in FIM_TOKEN_SPELLINGS.items():
        if tokenizer.encode(spelling, add_special_tokens=False) != [token_id]:
            raise ValueError("native evaluation tokenizer FIM identity differs")
    return tokenizer


def tokenizer_added_token_inventory(directory: Path) -> list[dict]:
    """Read the pinned tokenizer's complete added-token spelling inventory."""
    try:
        payload = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        entries = payload["added_tokens"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError, UnicodeError):
        raise ValueError("pinned tokenizer added-token inventory is unreadable") from None
    if not isinstance(entries, list):
        raise ValueError("pinned tokenizer added-token inventory is invalid")
    inventory = []
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not _is_nonnegative_integer(entry.get("id"))
            or not isinstance(entry.get("content"), str)
            or not entry["content"]
            or not entry["content"].isascii()
            or any(char in entry["content"] for char in "\n\r\t")
        ):
            raise ValueError("pinned tokenizer added-token inventory is invalid")
        inventory.append({"id": entry["id"], "spelling": entry["content"]})
    inventory.sort(key=lambda item: item["id"])
    ids = [item["id"] for item in inventory]
    spellings = [item["spelling"] for item in inventory]
    if len(ids) != len(set(ids)) or len(spellings) != len(set(spellings)):
        raise ValueError("pinned tokenizer added-token inventory is ambiguous")
    return inventory


def tokenizer_contract_sha256(profile_tokenizer: dict) -> str:
    inventory = profile_tokenizer.get("special_tokens")
    if not isinstance(inventory, list):
        raise ValueError("native tokenizer special-token inventory is invalid")
    ordered = sorted(inventory, key=lambda item: item.get("id", -1))
    fields = (
        profile_tokenizer.get("tokenizer_id"),
        profile_tokenizer.get("tokenizer_revision"),
        str(profile_tokenizer.get("tokenizer_sha256", "")).lower(),
        profile_tokenizer.get("eos_id"),
        profile_tokenizer.get("fim_prefix_id"),
        profile_tokenizer.get("fim_suffix_id"),
        profile_tokenizer.get("fim_middle_id"),
        profile_tokenizer.get("completion_mode"),
        profile_tokenizer.get("tokenizer_vocab_size"),
        str(profile_tokenizer.get("tokenizer_vocab_ids_sha256", "")).lower(),
    )
    material = "q25-fim-tokenizer-contract-v1\n" + "".join(f"{value}\n" for value in fields)
    for entry in ordered:
        if (
            not isinstance(entry, dict)
            or not _is_nonnegative_integer(entry.get("id"))
            or not isinstance(entry.get("spelling"), str)
        ):
            raise ValueError("native tokenizer special-token inventory is invalid")
        material += f"{entry['id']}\t{entry['spelling']}\n"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def tokenizer_vocab_ids(tokenizer) -> list[int]:
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("native evaluation tokenizer vocabulary is invalid")
    ids = []
    for value in vocabulary.values():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("native evaluation tokenizer vocabulary IDs are invalid")
        ids.append(value)
    return sorted(set(ids))


def tokenizer_vocab_ids_sha256(ids: list[int]) -> str:
    digest = hashlib.sha256()
    previous = -1
    for value in ids:
        if not isinstance(value, int) or isinstance(value, bool) or value <= previous:
            raise ValueError("native evaluation tokenizer vocabulary IDs are invalid")
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
        previous = value
    return digest.hexdigest()


def _same_json_identity(actual: object, expected: object) -> bool:
    """Compare JSON identity while preventing bool/int equality aliases."""
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == expected
        )
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same_json_identity(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same_json_identity(actual[key], value) for key, value in expected.items())
        )
    return actual is expected


def validate_native_tokenizer_identity(
    conversion: dict,
    health: dict,
    tokenizer_payload: dict,
    tokenizer,
    expected_special_tokens: list[dict],
) -> dict:
    """Bind service tokenizer bytes, IDs, and artifact provenance to Qwen FIM."""
    expected_conversion_tokenizer = {
        "model_id": TOKENIZER_ID,
        "revision": TOKENIZER_REVISION,
        "sha256": TOKENIZER_SHA,
        "eos_token_id": EOS_TOKEN_ID,
        "fim_marker_ids": {
            "fim_prefix": FIM_TOKEN_IDS["fim_prefix"],
            "fim_middle": FIM_TOKEN_IDS["fim_middle"],
            "fim_suffix": FIM_TOKEN_IDS["fim_suffix"],
        },
    }
    if not _same_json_identity(conversion.get("tokenizer"), expected_conversion_tokenizer):
        raise ValueError("selected conversion tokenizer identity differs")
    profile = health.get("fim_profile")
    if not isinstance(profile, dict):
        raise ValueError("native service lacks a frozen FIM profile")
    export_manifest_sha = conversion.get("source_export_manifest_sha256")
    if not isinstance(export_manifest_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", export_manifest_sha
    ):
        raise ValueError("selected FIM export manifest identity is invalid")
    profile_tokenizer = profile.get("tokenizer")
    if not isinstance(profile_tokenizer, dict):
        raise ValueError("native service lacks its FIM tokenizer profile")
    expected_profile = {
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA,
        "eos_id": EOS_TOKEN_ID,
        "fim_prefix_id": FIM_TOKEN_IDS["fim_prefix"],
        "fim_middle_id": FIM_TOKEN_IDS["fim_middle"],
        "fim_suffix_id": FIM_TOKEN_IDS["fim_suffix"],
        "completion_mode": MODE,
    }
    if (
        profile.get("artifact_manifest_sha256") != export_manifest_sha
        or health.get("tokenizer_sha256") != TOKENIZER_SHA
        or health.get("tokenizer_contract_sha256")
        != profile_tokenizer.get("tokenizer_contract_sha256")
        or any(
            not _same_json_identity(profile_tokenizer.get(key), value)
            for key, value in expected_profile.items()
        )
    ):
        raise ValueError("native service tokenizer or export identity differs")
    contract_sha = profile_tokenizer.get("tokenizer_contract_sha256")
    vocab_sha = profile_tokenizer.get("tokenizer_vocab_ids_sha256")
    vocab_size = profile_tokenizer.get("tokenizer_vocab_size")
    if (
        not isinstance(contract_sha, str)
        or len(contract_sha) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in contract_sha)
        or not isinstance(vocab_sha, str)
        or len(vocab_sha) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in vocab_sha)
        or not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size <= 0
    ):
        raise ValueError("native service tokenizer contract is invalid")
    flattened_identity = {
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA,
        "tokenizer_contract_sha256": contract_sha,
        "tokenizer_vocab_size": vocab_size,
        "tokenizer_vocab_ids_sha256": vocab_sha,
    }
    if any(
        not _same_json_identity(health.get(key), value)
        for key, value in flattened_identity.items()
    ):
        raise ValueError("native health tokenizer identity differs from its FIM profile")
    special_tokens = profile_tokenizer.get("special_tokens")
    if not isinstance(special_tokens, list) or not _same_json_identity(
        special_tokens, expected_special_tokens
    ):
        raise ValueError("native service special-token inventory is invalid")
    observed_specials: dict[int, str] = {}
    for item in special_tokens:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), int)
            or isinstance(item.get("id"), bool)
            or not isinstance(item.get("spelling"), str)
            or item["id"] in observed_specials
        ):
            raise ValueError("native service special-token inventory is invalid")
        observed_specials[item["id"]] = item["spelling"]
    if any(
        observed_specials.get(token_id) != spelling
        for token_id, spelling in FIM_TOKEN_SPELLINGS.items()
    ):
        raise ValueError("native service FIM control-token inventory differs")
    if tokenizer_contract_sha256(profile_tokenizer) != contract_sha:
        raise ValueError("native service tokenizer contract digest is invalid")

    expected_ids = tokenizer_vocab_ids(tokenizer)
    expected_vocab_sha = tokenizer_vocab_ids_sha256(expected_ids)
    native_ids = tokenizer_payload.get("tokenizer_vocab_ids")
    if (
        not isinstance(native_ids, list)
        or any(not _is_nonnegative_integer(value) for value in native_ids)
        or tokenizer_payload.get("model_sha256") != health.get("model_sha256")
        or tokenizer_payload.get("artifact_manifest_sha256") != export_manifest_sha
        or not _same_json_identity(tokenizer_payload.get("tokenizer"), profile_tokenizer)
        or native_ids != expected_ids
        or vocab_size != len(expected_ids)
        or vocab_sha != expected_vocab_sha
    ):
        raise ValueError("native service vocabulary IDs differ from the pinned tokenizer")
    return {
        "artifact_manifest_sha256": export_manifest_sha,
        "tokenizer_contract_sha256": contract_sha,
        "tokenizer_vocab_size": vocab_size,
        "tokenizer_vocab_ids_sha256": vocab_sha,
    }


def expected_context(case: dict, tokenizer) -> dict:
    """Rebuild the bounded PSM context and byte ranges from original source."""
    state = case.get("state")
    if not isinstance(state, dict) or not isinstance(state.get("source"), str):
        raise ValueError("native case source state is invalid")
    source = state["source"]
    raw = source.encode("utf-8")
    row = state.get("target_row")
    cursor_col = state.get("cursor_col")
    if (
        not isinstance(row, int)
        or isinstance(row, bool)
        or row < 0
        or not isinstance(cursor_col, int)
        or isinstance(cursor_col, bool)
        or cursor_col < 0
    ):
        raise ValueError("native case cursor identity is invalid")
    starts = [0]
    starts.extend(index + 1 for index, byte in enumerate(raw) if byte == 10)
    if not raw:
        if row != 0 or cursor_col != 0:
            raise ValueError("native case cursor is outside the empty file")
        line_start = content_end = line_end = 0
        ending = "EOF"
    else:
        if row >= len(starts) or (row == len(starts) - 1 and raw.endswith(b"\n")):
            raise ValueError("native case target row is outside the Rust physical-line model")
        line_start = starts[row]
        next_lf = raw.find(b"\n", line_start)
        line_end = next_lf + 1 if next_lf >= 0 else len(raw)
        if next_lf >= 0 and next_lf > line_start and raw[next_lf - 1] == 13:
            content_end = next_lf - 1
            ending = "CRLF"
        elif next_lf >= 0:
            content_end = next_lf
            ending = "LF"
        else:
            content_end = line_end
            ending = "EOF"
        if any(
            byte == 13 and (index + 1 >= len(raw) or raw[index + 1] != 10)
            for index, byte in enumerate(raw)
        ):
            raise ValueError("native case contains an unsupported lone-CR source")
    cursor_byte = line_start + cursor_col
    if cursor_byte > content_end:
        raise ValueError("native case cursor exceeds its physical line")
    try:
        full_prefix = raw[:cursor_byte].decode("utf-8")
        raw[cursor_byte:content_end].decode("utf-8")
        full_suffix = raw[line_end:].decode("utf-8")
    except UnicodeError:
        raise ValueError("native case cursor splits UTF-8") from None
    try:
        prefix = _truncate_prefix(tokenizer, full_prefix, PREFIX_CONTEXT_TOKEN_LIMIT)
        suffix = _truncate_suffix(tokenizer, full_suffix, SUFFIX_CONTEXT_TOKEN_LIMIT)
        prefix_token_count = len(_token_ids(tokenizer, prefix))
        suffix_token_count = len(_token_ids(tokenizer, suffix))
    except Exception:
        raise ValueError("native bounded context cannot be reproduced") from None
    prefix_start = cursor_byte - len(prefix.encode("utf-8"))
    suffix_end = line_end + len(suffix.encode("utf-8"))
    prompt = (
        "<|fim_prefix|>"
        + prefix
        + "<|fim_suffix|>"
        + suffix
        + "<|fim_middle|>"
    )
    return {
        "prompt": prompt,
        "target_row": row,
        "cursor_col": cursor_col,
        "line_ending": ending,
        "model_hole_range": {
            "start_byte": cursor_byte,
            "end_byte": line_end,
            "end_exclusive": True,
        },
        "prefix_range": {
            "start_byte": prefix_start,
            "end_byte": cursor_byte,
            "end_exclusive": True,
        },
        "prefix_token_count": prefix_token_count,
        "suffix_range": {
            "start_byte": line_end,
            "end_byte": suffix_end,
            "end_exclusive": True,
        },
        "suffix_token_count": suffix_token_count,
        "apply_range": {
            "start_byte": line_start,
            "end_byte": content_end,
            "end_exclusive": True,
        },
    }


def context_digest(
    request_id: str, case: dict, tokenizer_contract_sha256: str, tokenizer
) -> str:
    state = case["state"]
    source = state["source"].encode("utf-8")
    expected = expected_context(case, tokenizer)
    if case.get("prompt") != expected["prompt"]:
        raise ValueError("native case prompt differs from its bounded source context")
    prompt = expected["prompt"].encode("utf-8")
    canonical = (
        f"q25-fim-context-v2\n{request_id}\n{hashlib.sha256(source).hexdigest()}\n"
        f"{state['target_row']}\n{state['cursor_col']}\n"
        f"{hashlib.sha256(prompt).hexdigest()}\n"
        f"{expected['prefix_range']['start_byte']}\n"
        f"{expected['prefix_range']['end_byte']}\n"
        f"{expected['prefix_token_count']}\n"
        f"{expected['suffix_range']['start_byte']}\n"
        f"{expected['suffix_range']['end_byte']}\n"
        f"{expected['suffix_token_count']}\n"
        f"{CONTEXT_POLICY_VERSION}\n{CONTEXT_LAYOUT}\n"
        f"{tokenizer_contract_sha256.lower()}\n"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def source_state(row: dict, source: str, target: str) -> dict:
    """Reconstruct the original byte range, including its original line ending."""
    raw = source.encode()
    start, end = row["region_start"], row["region_end"]
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start <= end <= len(raw)
    ):
        raise ValueError("native case has an invalid source range")
    try:
        before = raw[:start].decode()
        actual = raw[start:end].decode()
        raw[end:].decode()
    except UnicodeError:
        raise ValueError("native case range splits UTF-8") from None
    if actual != target or hashlib.sha256(raw).hexdigest() != row["source_content_sha256"]:
        raise ValueError("native case does not reconstruct its public source")
    last_lf = raw.rfind(b"\n", 0, start)
    return {
        "file_id": row["source_path"],
        "filetype": row["language"],
        "source": source,
        "target_row": before.count("\n"),
        "cursor_col": start - last_lf - 1,
        "history": [],
        "relevant": [],
    }


def prepare(args: argparse.Namespace) -> dict:
    plan = json.loads(args.training_plan.read_text())
    expected = plan["data"]["development"]
    if (
        sha(args.development) != expected["sha256"]
        or args.development.stat().st_size != expected["bytes"]
    ):
        raise ValueError("native cases differ from frozen development data")
    preparation = ROOT / "reports/research/q25_code_cpt_r2/fim_preparation_plan.json"
    if sha(preparation) != plan["preparation_plan_sha256"]:
        raise ValueError("native case preparation identity differs")
    _, _, _, documents = _load_pinned_inputs(preparation)
    source_by_identity = {
        (doc.content_sha256, doc.repository_identity_sha256): doc.content for doc in documents
    }
    tokenizer = load_tokenizer(args.tokenizer)
    inputs = read_rows(args.development)
    if len(inputs) != expected["row_count"]:
        raise ValueError("native development count differs")
    cases = []
    for row in inputs:
        decoded = development_case(row, tokenizer)
        source = source_by_identity.get(
            (row["source_content_sha256"], row["repository_identity_sha256"])
        )
        if source is None:
            raise ValueError("native case lacks its exact original source")
        case = {
            "case_id": decoded["id"],
            "repository": decoded["repository"],
            "state": source_state(row, source, decoded["target"]),
            "prompt": decoded["prompt"],
            "target": decoded["target"],
            "prompt_token_ids": row["input_ids"][: row["prompt_tokens"]],
            "context_sha256": row["prompt_sha256"],
            "synthetic": True,
            "source_completion_mode": row["mode"],
        }
        if (
            expected_context(case, tokenizer)["prompt"] != case["prompt"]
            or hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest()
            != case["context_sha256"]
        ):
            raise ValueError("native case PSM prompt differs from the exact source cursor")
        cases.append(case)
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("native cases have duplicate identities")
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases)
    if args.cases.exists() and args.cases.read_text() != content:
        raise ValueError("native cases already exist with another identity")
    args.cases.write_text(content)
    return {"cases": len(cases), "cases_sha256": sha(args.cases), "synthetic": True}


def freeze(args: argparse.Namespace) -> dict:
    conversion = json.loads(args.conversion.read_text())
    if (
        conversion.get("status") != "complete"
        or sha(args.model) != conversion["q4_export"]["sha256"]
        or conversion.get("training_plan_sha256") != sha(args.training_plan)
    ):
        raise ValueError("native plan requires the actual verified selected Q4 artifact")
    if urlsplit(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("native freeze requires a configured local owned service")
    response = httpx.get(args.url + "/health", timeout=10)
    response.raise_for_status()
    health = response.json()
    port = urlsplit(args.url).port
    if port is None:
        raise ValueError("native evaluation URL must specify its service port")
    process = attest_process(args.pid, sha(args.binary), port)
    if health.get("model_sha256") != conversion["q4_export"]["sha256"]:
        raise ValueError("native freeze service does not use the selected model")
    tokenizer = load_tokenizer(args.tokenizer)
    tokenizer_response = httpx.get(args.url + "/v1/fim-tokenizer", timeout=10)
    tokenizer_response.raise_for_status()
    tokenizer_profile = validate_native_tokenizer_identity(
        conversion,
        health,
        tokenizer_response.json(),
        tokenizer,
        tokenizer_added_token_inventory(args.tokenizer),
    )
    runtime_fields = (
        "model_protocol",
        "output_tokens",
        "active_slots",
        "saved_contexts",
        "backend",
        "runtime_config_hash",
        "context_layout",
        "context_size",
        "input_tokens",
        "threads",
        "prompt_threads",
        "batch_size",
        "microbatch_size",
        "cache_type",
        "syntax_validation",
        "tokenizer_contract_sha256",
        "tokenizer_sha256",
        "fim_profile",
    )
    if any(key not in health for key in runtime_fields) or (
        health.get("model_protocol") != PROTOCOL
        or health.get("output_tokens") != 96
        or health.get("active_slots") != 1
        or health.get("saved_contexts") != 0
        or health.get("syntax_validation") is not False
    ):
        raise ValueError("native freeze runtime contract is incomplete or unsafe")
    if memory_snapshot(args.pid)["VmRSS_bytes"] > 1536 * 1024**2:
        raise ValueError("native predictor exceeds the resident memory safeguard")
    plan = {
        "schema": SCHEMA,
        "training_plan_sha256": sha(args.training_plan),
        "cases_sha256": sha(args.cases),
        "tokenizer_sha256": TOKENIZER_SHA,
        "model_sha256": sha(args.model),
        "binary_sha256": sha(args.binary),
        "conversion_sha256": sha(args.conversion),
        "precision": "Q4_K_M",
        "comparison": "HF FP16 versus native Q4 changes runtime and precision",
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "health": {key: health[key] for key in runtime_fields},
        "tokenizer_profile": tokenizer_profile,
        "process": process,
        "decoding": "greedy, observed EOS only, 96 sampled tokens including EOS",
        "case_timeout_seconds": 30,
        "session_seconds": 1800,
        "quality_caveat": "Synthetic completion, not human next-edit intent or acceptance.",
    }
    if args.plan.exists() and json.loads(args.plan.read_text()) != plan:
        raise ValueError("native evaluation plan already exists with another identity")
    save(args.plan, plan)
    save(args.plan.with_suffix(".process.json"), process)
    return plan


def validate_terminal(terminal: dict, health: dict, binding: dict) -> bool:
    profile = health.get("fim_profile")
    profile_tokenizer = profile.get("tokenizer") if isinstance(profile, dict) else None
    if not isinstance(profile_tokenizer, dict):
        return False
    expected = {
        **binding,
        "model_protocol": PROTOCOL,
        "model_sha256": health["model_sha256"],
        "context_layout": health["context_layout"],
        "tokenizer_sha256": TOKENIZER_SHA,
        "tokenizer_contract_sha256": health["tokenizer_contract_sha256"],
        "fim_token_ids": FIM_TOKEN_IDS,
        "artifact_manifest_sha256": profile.get("artifact_manifest_sha256"),
        "tokenizer_id": profile_tokenizer.get("tokenizer_id"),
        "tokenizer_revision": profile_tokenizer.get("tokenizer_revision"),
        "tokenizer_vocab_size": profile_tokenizer.get("tokenizer_vocab_size"),
        "tokenizer_vocab_ids_sha256": profile_tokenizer.get("tokenizer_vocab_ids_sha256"),
    }
    return (
        terminal.get("stop") is True
        and terminal.get("stop_type") in {"eos", "control", "limit"}
        and all(_same_json_identity(terminal.get(k), v) for k, v in expected.items())
    )


def expected_canonical_action(state: dict, raw_text: str) -> dict | None:
    """Verify the native edit independently without trimming or unescaping code."""
    source = state["source"]
    if source == "":
        return (
            None
            if "\n" in raw_text or "\r" in raw_text
            else {"kind": "insert_before", "text": raw_text}
        )
    pieces = source.split("\n")
    line = pieces[state["target_row"]]
    if state["target_row"] < len(pieces) - 1:
        line += "\n"
    ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
    body = raw_text
    if ending:
        if not body.endswith(ending):
            return None
        body = body[: -len(ending)]
    if "\r" in body or "\n" in body:
        return None
    prefix = line.encode()[: state["cursor_col"]].decode()
    return {"kind": "replace_line", "text": prefix + body}


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def score_terminal(case: dict, terminal: dict, raw_text: str, tokenizer) -> dict:
    """Re-decode bounded native token evidence without repairing model text."""
    body_ids = terminal.get("sampled_token_ids")
    terminal_id = terminal.get("terminal_token_id")
    stop_type = terminal.get("stop_type")
    if (
        not isinstance(body_ids, list)
        or any(not _is_nonnegative_integer(token_id) for token_id in body_ids)
        or len(body_ids) > 96
        or not _is_nonnegative_integer(terminal.get("tokens_predicted"))
        or terminal["tokens_predicted"] != len(body_ids)
    ):
        raise ValueError("native terminal sampled-token evidence is invalid")
    if stop_type == "eos":
        if terminal_id != EOS_TOKEN_ID or len(body_ids) > 95:
            raise ValueError("native EOS terminal evidence is inconsistent")
        all_ids = body_ids + [EOS_TOKEN_ID]
        decoder_ids = all_ids
    elif stop_type == "control":
        if (
            not _is_nonnegative_integer(terminal_id)
            or terminal_id == EOS_TOKEN_ID
            or len(body_ids) > 95
        ):
            raise ValueError("native non-EOS terminal evidence is inconsistent")
        all_ids = body_ids + [terminal_id]
        decoder_ids = body_ids
    elif stop_type == "limit":
        if terminal_id is not None or len(body_ids) != 96:
            raise ValueError("native output-limit terminal evidence is inconsistent")
        all_ids = body_ids
        decoder_ids = body_ids
    else:
        raise ValueError("native terminal has an unsupported stop type")
    if len(all_ids) > 96:
        raise ValueError("native terminal exceeds the EOS-inclusive output cap")
    vocab = tokenizer.get_vocab()
    known_ids = set(vocab.values())
    unknown_ids = [token_id for token_id in body_ids if token_id not in known_ids]
    control_ids = (
        set(getattr(tokenizer, "all_special_ids", []))
        | set(getattr(tokenizer, "added_tokens_decoder", {}))
        | {
            vocab[spelling]
            for spelling in FIM_TOKEN_SPELLINGS.values()
            if spelling in vocab
        }
    )
    validation = terminal.get("action_validation")
    action = terminal.get("canonical_action")
    if unknown_ids:
        first_failure = next(
            (
                "fim_token_id_outside_vocabulary"
                if token_id not in known_ids
                else "fim_control_token_in_body"
                for token_id in body_ids
                if token_id not in known_ids or token_id in control_ids
            ),
            None,
        )
        if (
            stop_type != "eos"
            or not isinstance(validation, dict)
            or validation.get("policy") != FIM_ACTION_POLICY
            or validation.get("status") != "invalid"
            or validation.get("code") != "fim_token_id_outside_vocabulary"
            or first_failure != "fim_token_id_outside_vocabulary"
            or action is not None
        ):
            raise ValueError("unknown native token lacks matching bound codec evidence")
        unexpected = [token_id for token_id in body_ids if token_id in control_ids]
        evidence = {
            "output_token_ids": all_ids,
            "ended_by_eos": True,
            "unexpected_special_token_ids": unexpected,
            "unexpected_control_token_ids": unexpected,
            "unknown_token_ids": unknown_ids,
            "reached_token_ceiling": len(all_ids) >= 96,
            "truncated": False,
        }
        return {
            "decoded": None,
            "decode_reason": "invalid_control_or_vocabulary",
            "evidence": evidence,
            "all_ids": all_ids,
            "terminal_id": terminal_id,
            "terminated": False,
            "quality_error_code": "fim_token_id_outside_vocabulary",
            "canonical_action": None,
        }
    if stop_type == "control" and terminal_id not in known_ids:
        raise ValueError("native terminal control token is outside the pinned vocabulary")
    decoded, decode_reason, evidence = decoded_completion(
        tokenizer, decoder_ids, ceiling=96, newline_stop=False
    )
    if decoded != raw_text:
        raise ValueError("native output bytes differ from the selected tokenizer")
    evidence["output_token_ids"] = all_ids
    evidence["ended_by_eos"] = stop_type == "eos"
    evidence["reached_token_ceiling"] = len(all_ids) >= 96
    evidence["truncated"] = stop_type == "limit"

    if not isinstance(validation, dict) or validation.get("policy") != FIM_ACTION_POLICY:
        raise ValueError("native terminal lacks the FIM codec validation result")
    if validation.get("status") == "not_applicable":
        expected_action = expected_canonical_action(case["state"], raw_text)
        if (
            stop_type != "eos"
            or decode_reason != "eos"
            or expected_action is None
            or action != expected_action
        ):
            raise ValueError("native valid action does not match the decoded FIM completion")
        terminated = True
        quality_error_code = None
    elif validation.get("status") == "invalid":
        quality_error_code = validation.get("code")
        if quality_error_code not in FIM_QUALITY_CODES or action is not None:
            raise ValueError("native invalid-completion evidence is inconsistent")
        terminated = False
    else:
        raise ValueError("native FIM codec validation status is invalid")
    return {
        "decoded": decoded,
        "decode_reason": decode_reason,
        "evidence": evidence,
        "all_ids": all_ids,
        "terminal_id": terminal_id,
        "terminated": terminated,
        "quality_error_code": quality_error_code,
        "canonical_action": action,
    }


def validate_timing_record(row: dict, expected_input_tokens: int, predicted_tokens: int) -> None:
    latency = row.get("latency_seconds")
    first_token = row.get("first_token_seconds")
    if not _is_finite_nonnegative(latency) or (
        first_token is not None
        and (not _is_finite_nonnegative(first_token) or first_token > latency)
    ):
        raise ValueError("native evaluation latency is non-finite or inconsistent")
    timings = row.get("server_timings")
    if not isinstance(timings, dict):
        raise ValueError("native server timings are missing")
    for key in ("cache_n", "prompt_n", "predicted_n"):
        if not _is_nonnegative_integer(timings.get(key)):
            raise ValueError("native server timing counts are invalid")
    for key in ("prompt_ms", "predicted_ms", "total_ms"):
        if not _is_finite_nonnegative(timings.get(key)):
            raise ValueError("native server timings are non-finite")
    if (
        timings["prompt_n"] != expected_input_tokens
        or timings["predicted_n"] != predicted_tokens
        or row.get("input_tokens") != expected_input_tokens
        or not _is_nonnegative_integer(row.get("output_tokens_including_terminal"))
        or row["output_tokens_including_terminal"] < predicted_tokens
    ):
        raise ValueError("native server token counts differ from the frozen case")
    memory = row.get("post_request_process_memory")
    if (
        not isinstance(memory, dict)
        or not _is_nonnegative_integer(memory.get("VmRSS_bytes"))
        or memory["VmRSS_bytes"] > 1536 * 1024**2
        or any(not _is_nonnegative_integer(value) for value in memory.values())
    ):
        raise ValueError("native process memory evidence is invalid")


def validate_result_row(row: dict, case: dict, tokenizer, health: dict, process: dict) -> None:
    """Reject stale, edited, or numerically invalid resume rows."""
    if (
        row.get("case_id") != case.get("case_id")
        or row.get("repository") != case.get("repository")
        or row.get("context_sha256") != case.get("context_sha256")
        or row.get("process_identity") != process
        or not isinstance(row.get("request_id"), str)
        or not row["request_id"].startswith("request-")
        or row.get("completion_sha256")
        != hashlib.sha256(str(row.get("raw_response", "")).encode("utf-8")).hexdigest()
    ):
        raise ValueError("native result row identity differs from its frozen case")
    terminal = row.get("terminal_event")
    if not isinstance(terminal, dict):
        raise ValueError("native result row lacks its terminal event")
    binding = {
        "request_id": row["request_id"],
        "context_hash": context_digest(
            row["request_id"], case, health["tokenizer_contract_sha256"], tokenizer
        ),
        "completion_mode": MODE,
    }
    if terminal.get("context_hash") != binding["context_hash"] or not validate_terminal(
        terminal, health, binding
    ):
        raise ValueError("native result row terminal identity is stale")
    if not isinstance(row.get("case_attempt_id"), str) or not row["case_attempt_id"].startswith(
        "case-attempt-"
    ):
        raise ValueError("native result row case-attempt identity is invalid")
    for key, prefix in (
        ("campaign_id", "campaign-"),
        ("run_id", "run-"),
        ("run_attempt_id", "attempt-"),
    ):
        if not isinstance(row.get(key), str) or not row[key].startswith(prefix):
            raise ValueError("native result row run identity is invalid")
    if row.get("raw_response") is None or not isinstance(row["raw_response"], str):
        raise ValueError("native result row text is missing")
    scored = score_terminal(case, terminal, row["raw_response"], tokenizer)
    exact = scored["decoded"] == case["target"]
    expected_values = {
        "exact": exact,
        "exact_and_terminated": exact and scored["terminated"],
        "ended_by_eos": terminal["stop_type"] == "eos",
        "terminated": scored["terminated"],
        "finish_reason": (
            "terminal_control"
            if terminal["stop_type"] == "control"
            else scored["decode_reason"]
        ),
        "output_tokens_including_terminal": len(scored["all_ids"]),
        "canonical_action": scored["canonical_action"],
        "quality_error_code": scored["quality_error_code"],
        **scored["evidence"],
    }
    if any(not _same_json_identity(row.get(key), value) for key, value in expected_values.items()):
        raise ValueError("native result row does not match re-decoded token evidence")
    if row.get("server_timings") != terminal.get("timings"):
        raise ValueError("native result row server timings differ from its terminal event")
    if row.get("native_stop_type") != terminal.get("stop_type"):
        raise ValueError("native result row finish evidence differs from its terminal event")
    validate_timing_record(row, len(case["prompt_token_ids"]), len(terminal["sampled_token_ids"]))


def prepare_request(
    client, url: str, case: dict, request_id: str, health: dict, tokenizer
) -> dict:
    # Tokenize invalidates the backend's one-shot prepared context. Check the
    # fixed prompt first, then prepare and immediately submit generation.
    token_response = client.post(
        url + "/tokenize", json={"content": case["prompt"], "add_special": False}
    )
    token_response.raise_for_status()
    if token_response.json().get("tokens") != case["prompt_token_ids"]:
        raise ValueError("native token IDs differ from the frozen training input")
    response = client.post(
        url + "/v1/editor/context",
        json={"state": case["state"], "repository_identity": case["repository"],
              "request_id": request_id, "completion_mode": MODE},
    )
    response.raise_for_status()
    prepared = response.json()
    expected = expected_context(case, tokenizer)
    identity = prepared.get("model_identity")
    if (
        any(prepared.get(key) != value for key, value in expected.items())
        or prepared.get("prompt") != case["prompt"]
        or prepared.get("request_id") != request_id
        or prepared.get("completion_mode") != MODE
        or prepared.get("model_protocol") != PROTOCOL
        or prepared.get("context_policy_version") != CONTEXT_POLICY_VERSION
        or prepared.get("context_layout") != CONTEXT_LAYOUT
        or prepared.get("tokenizer_sha256") != TOKENIZER_SHA
        or prepared.get("tokenizer_contract_sha256") != health["tokenizer_contract_sha256"]
        or not isinstance(prepared.get("prompt_tokens"), int)
        or isinstance(prepared.get("prompt_tokens"), bool)
        or prepared["prompt_tokens"] != len(case["prompt_token_ids"])
        or prepared["prompt_tokens"]
        != expected["prefix_token_count"] + expected["suffix_token_count"] + 3
        or not isinstance(identity, dict)
        or identity.get("model_sha256") != health["model_sha256"]
        or identity.get("model_protocol") != PROTOCOL
        or identity.get("tokenizer_sha256") != TOKENIZER_SHA
        or identity.get("tokenizer_contract_sha256") != health["tokenizer_contract_sha256"]
        or prepared.get("context_hash")
        != context_digest(request_id, case, health["tokenizer_contract_sha256"], tokenizer)
    ):
        raise ValueError("native context differs from the frozen training prompt")
    return prepared


def evaluate(args: argparse.Namespace) -> dict:
    if urlsplit(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("native evaluation requires the configured local owned service")
    plan = json.loads(args.plan.read_text())
    identities = {
        "training_plan_sha256": sha(args.training_plan),
        "cases_sha256": sha(args.cases),
        "model_sha256": sha(args.model),
        "binary_sha256": sha(args.binary),
        "conversion_sha256": sha(args.conversion),
        "tokenizer_sha256": TOKENIZER_SHA,
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
    }
    if plan.get("schema") != SCHEMA or any(plan.get(k) != v for k, v in identities.items()):
        raise ValueError("native evaluation fingerprints differ from frozen plan")
    args.output.mkdir(parents=True, exist_ok=True)
    record_path = args.output / "results.jsonl"
    metadata_path = args.output / "metadata.json"
    fingerprint = sha(args.plan)
    existing = read_rows(record_path) if record_path.exists() else []
    if metadata_path.exists():
        if (
            not args.resume
            or json.loads(metadata_path.read_text()).get("plan_sha256") != fingerprint
        ):
            raise ValueError("native evaluation requires explicit matching resume")
    elif existing:
        raise ValueError("native evaluation records lack their metadata")
    cases = read_rows(args.cases)
    by_id = {row.get("case_id"): row for row in cases}
    completed = {row.get("case_id") for row in existing}
    if (
        len(by_id) != len(cases)
        or len(completed) != len(existing)
        or not completed <= by_id.keys()
    ):
        raise ValueError("native evaluation case identities are duplicate or invalid")
    tokenizer = load_tokenizer(args.tokenizer)
    port = urlsplit(args.url).port
    if port is None:
        raise ValueError("native evaluation URL must specify its service port")
    process = attest_process(args.pid, plan["binary_sha256"], port)
    if process != plan.get("process"):
        raise ValueError("native evaluator process differs from the frozen plan")
    process_path = args.plan.with_suffix(".process.json")
    if not process_path.exists() or json.loads(process_path.read_text()) != plan["process"]:
        raise ValueError("native evaluator process sidecar differs from the frozen plan")
    for case in cases:
        if (
            not isinstance(case.get("case_id"), str)
            or not isinstance(case.get("repository"), str)
            or not isinstance(case.get("context_sha256"), str)
            or not isinstance(case.get("target"), str)
            or not isinstance(case.get("prompt_token_ids"), list)
            or any(not _is_nonnegative_integer(token_id) for token_id in case["prompt_token_ids"])
            or expected_context(case, tokenizer)["prompt"] != case.get("prompt")
            or hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest()
            != case["context_sha256"]
            or tokenizer.encode(case["prompt"], add_special_tokens=False)
            != case["prompt_token_ids"]
        ):
            raise ValueError("native case prompt or token evidence differs from the frozen source")
    if existing and any(row.get("case_id") not in by_id for row in existing):
        raise ValueError("native resume contains an unknown case identity")
    provider = NativeProvider(args.url, "q25-fim")
    provider.timeout_seconds = plan["case_timeout_seconds"]
    provider.cache = False
    started = time.monotonic()
    with httpx.Client(timeout=30) as client:
        health_response = client.get(args.url + "/health")
        health_response.raise_for_status()
        health = health_response.json()
        expected = {**plan["health"], "model_sha256": plan["model_sha256"], "status": "ok"}
        if any(not _same_json_identity(health.get(k), v) for k, v in expected.items()):
            raise ValueError("native service does not match the selected plan")
        tokenizer_response = client.get(args.url + "/v1/fim-tokenizer")
        tokenizer_response.raise_for_status()
        conversion = json.loads(args.conversion.read_text())
        tokenizer_profile = validate_native_tokenizer_identity(
            conversion,
            health,
            tokenizer_response.json(),
            tokenizer,
            tokenizer_added_token_inventory(args.tokenizer),
        )
        if not _same_json_identity(tokenizer_profile, plan.get("tokenizer_profile")):
            raise ValueError("native tokenizer profile differs from the frozen plan")
        if metadata_path.exists():
            prior_metadata = json.loads(metadata_path.read_text())
            if (
                prior_metadata.get("plan_sha256") != fingerprint
                or not _same_json_identity(prior_metadata.get("process"), process)
                or not _same_json_identity(prior_metadata.get("health"), health)
                or not _same_json_identity(
                    prior_metadata.get("tokenizer_profile"), tokenizer_profile
                )
            ):
                raise ValueError("native resume metadata differs from the frozen process")
        else:
            save(
                metadata_path,
                {
                    "plan_sha256": fingerprint,
                    "health": health,
                    "process": process,
                    "tokenizer_profile": tokenizer_profile,
                    "synthetic": True,
                },
            )
        for row in existing:
            validate_result_row(row, by_id[row["case_id"]], tokenizer, health, process)
        with run_scope(args.output / "observability-run.json", "q25-fim-native") as run:
            for case in cases:
                if case["case_id"] in completed:
                    continue
                if time.monotonic() - started + provider.timeout_seconds > plan["session_seconds"]:
                    raise TimeoutError("native evaluation deadline reserve reached")
                context = (
                    run.for_case(case["case_id"])
                    if isinstance(run, RunContext)
                    else RunContext.new().for_case(case["case_id"])
                )
                request_id = context.request_id
                with context.activate(), operation(
                    "eval.case", attributes={"tabcomplete.case_kind": "synthetic_fim"}
                ):
                    prepared = prepare_request(
                        client, args.url, case, request_id, health, tokenizer
                    )
                    binding = {
                        "request_id": request_id,
                        "context_hash": prepared["context_hash"],
                        "completion_mode": MODE,
                    }
                    provider.editor_request_binding = binding
                    provider.repository_identity = case["repository"]
                    result = provider.generate_detailed(prepared["prompt"], 96)
                    memory = memory_snapshot(args.pid)
                    if memory.get("VmRSS_bytes", 0) > 1536 * 1024**2:
                        raise ValueError("native predictor exceeded its resident memory safeguard")
                    terminal = provider.last.get("terminal_event")
                    if not isinstance(terminal, dict) or not validate_terminal(
                        terminal, health, binding
                    ):
                        raise ValueError("native completion terminal identity is missing or stale")
                    ids = terminal.get("sampled_token_ids")
                    if not isinstance(ids, list) or ids != provider.last.get("token_ids"):
                        raise ValueError(
                            "native sampled-token evidence differs from streamed output"
                        )
                    with operation(
                        "model.score", attributes={"tabcomplete.metric": "fim_exact_completion"}
                    ) as score_span:
                        scored = score_terminal(case, terminal, result.text, tokenizer)
                        exact = scored["decoded"] == case["target"]
                        score_span.set_attribute("tabcomplete.score.exact", exact)
                        score_span.set_attribute(
                            "tabcomplete.score.terminated", scored["terminated"]
                        )
                    timings = provider.last.get("server_timings")
                    row = {
                        "case_id": case["case_id"],
                        "repository": case["repository"],
                        "context_sha256": case["context_sha256"],
                        "raw_response": result.text,
                        "completion_sha256": hashlib.sha256(
                            result.text.encode("utf-8")
                        ).hexdigest(),
                        "exact": exact,
                        "exact_and_terminated": exact and scored["terminated"],
                        "ended_by_eos": terminal["stop_type"] == "eos",
                        "terminated": scored["terminated"],
                        "finish_reason": (
                            "terminal_control"
                            if terminal["stop_type"] == "control"
                            else scored["decode_reason"]
                        ),
                        "native_stop_type": terminal["stop_type"],
                        "quality_error_code": scored["quality_error_code"],
                        "input_tokens": len(case["prompt_token_ids"]),
                        "output_tokens_including_terminal": len(scored["all_ids"]),
                        "latency_seconds": provider.last.get("total_seconds"),
                        "first_token_seconds": provider.last.get("token_arrival_seconds", {}).get(
                            "1"
                        ),
                        "server_timings": timings,
                        "canonical_action": scored["canonical_action"],
                        "post_request_process_memory": memory,
                        "request_id": request_id,
                        "campaign_id": context.campaign_id,
                        "run_id": context.run_id,
                        "run_attempt_id": context.run_attempt_id,
                        "case_attempt_id": context.case_attempt_id,
                        "process_identity": process,
                        "terminal_event": terminal,
                        **scored["evidence"],
                    }
                    validate_result_row(row, case, tokenizer, health, process)
                with record_path.open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                existing.append(row)
    if len(existing) != len(cases):
        raise ValueError("native evaluation ended without every frozen development case")
    summary = {
        "schema": SCHEMA,
        "plan_sha256": fingerprint,
        "cases": len(existing),
        "exact": sum(row["exact"] for row in existing),
        "exact_and_terminated": sum(row["exact_and_terminated"] for row in existing),
        "terminated": sum(row["terminated"] for row in existing),
        "token_id_parity_verified_cases": len(existing),
        "precision": "Q4_K_M",
        "runtime_precision_joint_comparison": True,
        "quality_caveat": plan["quality_caveat"],
    }
    save(args.output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "freeze", "evaluate"), required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--conversion", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--url", default="http://127.0.0.1:19104")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    required = {
        "prepare": ("development",),
        "freeze": ("plan", "model", "binary", "conversion", "pid"),
        "evaluate": ("plan", "model", "binary", "conversion", "output", "pid"),
    }[args.stage]
    if any(getattr(args, key) is None for key in required):
        parser.error("selected stage lacks its required paths")
    result = {"prepare": prepare, "freeze": freeze, "evaluate": evaluate}[args.stage](args)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, sort_keys=True))


if __name__ == "__main__":
    main()
