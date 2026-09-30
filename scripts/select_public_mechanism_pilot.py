#!/usr/bin/env python3
"""Select a small, offline source-evidence pilot from the pinned public bank.

This command only reads the existing JSONL bank and writes a frozen, private
selection plan. It does not access the network, call a model provider, or create
labels. Selected files are source anchors for a later authoring and blind-solver
pilot; selection is not evidence that any edit is inferable or training-ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SELECTOR_VERSION = "public-mechanism-source-preflight-v2"
EXPECTED_BANK_SHA256 = "f1320155723d450454484999ebb3605614e251c223e1801f7e9808d1d3bde933"
DEFAULT_BANK = Path("artifacts/research/one_line_r1/public_source_authoring_100.jsonl")
DEFAULT_OUTPUT = Path("/mnt/ssd/tabcomplete-public-mechanism-pilot-r1-v2")
MAX_SELECTED = 24
PER_FAMILY_CAP = 8
PER_LANGUAGE_FAMILY_CAP = 2
PER_REPOSITORY_CAP = 1

# These selectors name visible source structures only. They do not assert that
# an edit is correct, inferable, licensed for training, or appropriate to show
# a student. The canonical JSON for these rules is hashed into the plan before
# any later provider call can be made.
SELECTION_RULES: dict[str, Any] = {
    "version": "public-mechanism-source-rules-v1",
    "families": [
        {
            "id": "local_helper_reuse",
            "description": (
                "A local function declaration and at least two distinct in-file call sites."
            ),
            "evidence": "one declaration line plus two call lines; all are copied from source",
            "limits": {"min_call_sites": 2, "max_source_line_span": 40},
        },
        {
            "id": "sort_call_with_ordering_signal",
            "description": (
                "A visible sort-like call followed within twelve lines by a "
                "comparator or key signal."
            ),
            "evidence": "sort call line plus the nearest following comparison/key line",
            "limits": {"max_following_line_distance": 12},
        },
        {
            "id": "qualified_type_member_overlap",
            "description": (
                "Two Rust qualified type/member spellings occur under the "
                "same pair of type qualifiers."
            ),
            "evidence": (
                "two repeated member names, each with the exact source lines "
                "for both type qualifiers"
            ),
            "limits": {"minimum_distinct_members": 2, "max_pair_line_span": 80},
        },
    ],
    "candidate_limits": {
        "total": MAX_SELECTED,
        "per_family_queue": PER_FAMILY_CAP,
        "per_language_in_family_queue": PER_LANGUAGE_FAMILY_CAP,
        "per_repository": PER_REPOSITORY_CAP,
    },
    "allocation": (
        "scarce family queues first, then stable SHA-256 ranking and "
        "round-robin; each source/repository selected at most once"
    ),
    "not_used": ["author focus tags", "candidate split tags", "source filename as a family label"],
}

FAMILY_ORDER = tuple(family["id"] for family in SELECTION_RULES["families"])
SPDX_RE = re.compile(
    r"^\s*(?:(?://|#|/\*+|\*)\s*)?SPDX-License-Identifier\s*:\s*(.*?)\s*(?:\*/)?\s*$",
    re.IGNORECASE,
)
IDENTIFIER_RE = r"[A-Za-z_$][A-Za-z0-9_$]*"
SORT_CALL_RE = re.compile(
    r"(?:\.|::)?\b(?:sort|sorted|sort_by|sort_unstable|sort_unstable_by|Sort|SortFunc|SortSlice|order_by|orderBy)\s*\("
)
ORDER_SIGNAL_RE = re.compile(r"(?:<|>|\.cmp\s*\(|\bcmp\s*\(|\bcompare\s*\(|\bkey\s*[:=])")
QUALIFIED_MEMBER_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)::([A-Z][A-Za-z0-9_]*)\b")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rules_sha256() -> str:
    return sha256_bytes(canonical_json(SELECTION_RULES))


def _mask_non_code(source: str, language: str) -> str:
    """Mask strings and comments while preserving offsets and line breaks.

    This deliberately small lexer is used only to avoid treating comment or
    literal text as structural evidence. It does not parse or validate code.
    """

    python = language == "python"
    out = list(source)
    i = 0
    state = "code"
    quote = ""
    triple = False
    block_depth = 0
    while i < len(source):
        char = source[i]
        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                out[i] = " "
            i += 1
            continue
        if state == "block_comment":
            if not python and source.startswith("/*", i):
                block_depth += 1
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if not python and source.startswith("*/", i):
                block_depth -= 1
                out[i] = out[i + 1] = " "
                i += 2
                if block_depth == 0:
                    state = "code"
                continue
            if char != "\n":
                out[i] = " "
            i += 1
            continue
        if state == "string":
            delimiter = quote * (3 if triple else 1)
            if char == "\\":
                out[i] = " " if char != "\n" else char
                if i + 1 < len(source):
                    out[i + 1] = " " if source[i + 1] != "\n" else "\n"
                i += 2
                continue
            if source.startswith(delimiter, i):
                for j in range(i, min(len(source), i + len(delimiter))):
                    if source[j] != "\n":
                        out[j] = " "
                i += len(delimiter)
                state = "code"
                continue
            if char != "\n":
                out[i] = " "
            i += 1
            continue

        if source.startswith("//", i) and not python:
            out[i] = out[i + 1] = " "
            state = "line_comment"
            i += 2
            continue
        if source.startswith("/*", i) and not python:
            out[i] = out[i + 1] = " "
            state = "block_comment"
            block_depth = 1
            i += 2
            continue
        if python and char == "#":
            out[i] = " "
            state = "line_comment"
            i += 1
            continue

        # Rust lifetimes use apostrophes without a closing quote. Do not let
        # one hide the rest of a declaration; char literals still get masked.
        if language == "rust" and char == "'":
            if i + 2 >= len(source) or source[i + 2] != "'":
                i += 1
                continue

        if char in ("'", '"', "`"):
            triple = python and source.startswith(char * 3, i)
            delimiter = char * (3 if triple else 1)
            for j in range(i, min(len(source), i + len(delimiter))):
                if source[j] != "\n":
                    out[j] = " "
            state = "string"
            quote = char
            i += len(delimiter)
            continue

        i += 1
    return "".join(out)


def _definition_pattern(language: str) -> re.Pattern[str] | None:
    if language == "python":
        return re.compile(rf"^\s*(?:async\s+)?def\s+({IDENTIFIER_RE})\s*\(")
    if language == "go":
        return re.compile(rf"^\s*func\s+(?:\([^)]*\)\s*)?({IDENTIFIER_RE})\s*\(")
    if language == "rust":
        return re.compile(
            rf"^\s*(?:(?:pub(?:\([^)]*\))?|async|unsafe|const)\s+)*fn\s+({IDENTIFIER_RE})\s*\("
        )
    if language == "typescript":
        return re.compile(
            rf"^\s*(?:(?:export|default|async|declare)\s+)*function\s+({IDENTIFIER_RE})\s*\("
        )
    return None


def _evidence(
    family: str, lines: list[str], line_numbers: list[int], roles: list[str]
) -> dict[str, Any]:
    entries = []
    for number, role in zip(line_numbers, roles, strict=True):
        raw = lines[number - 1]
        entries.append(
            {
                "line": number,
                "role": role,
                "text": raw,
                "line_sha256": sha256_bytes(raw.encode("utf-8")),
            }
        )
    return {"family": family, "evidence": entries}


def detect_source_evidence(source: str, language: str) -> dict[str, dict[str, Any]]:
    """Return directly supported structural observations, never inferred labels."""

    lines = source.splitlines()
    code_lines = _mask_non_code(source, language).splitlines()
    if len(code_lines) != len(lines):
        # splitlines() omits a final empty line in both representations; this
        # guard makes offset drift a hard failure instead of bad evidence.
        raise ValueError("source line masking changed line count")
    found: dict[str, dict[str, Any]] = {}

    definition_re = _definition_pattern(language)
    if definition_re:
        definitions: list[tuple[str, int]] = []
        for index, line in enumerate(code_lines):
            match = definition_re.search(line)
            if match:
                definitions.append((match.group(1), index))
        for name, def_index in definitions:
            call_re = re.compile(rf"(?<![\w$]){re.escape(name)}\s*\(")
            call_indexes = [
                i for i, line in enumerate(code_lines) if i != def_index and call_re.search(line)
            ]
            eligible_calls = [i for i in call_indexes if abs(i - def_index) <= 40]
            if len(eligible_calls) >= 2:
                chosen = sorted(eligible_calls, key=lambda i: (abs(i - def_index), i))[:2]
                all_numbers = [def_index + 1, *(i + 1 for i in chosen)]
                found["local_helper_reuse"] = _evidence(
                    "local_helper_reuse",
                    lines,
                    all_numbers,
                    ["function declaration", "call site", "call site"],
                )
                break

    sort_lines = [i for i, line in enumerate(code_lines) if SORT_CALL_RE.search(line)]
    for sort_index in sort_lines:
        signals = [
            i
            for i in range(sort_index, min(len(code_lines), sort_index + 13))
            if ORDER_SIGNAL_RE.search(code_lines[i])
        ]
        if signals:
            signal_index = min(signals, key=lambda i: (i - sort_index, i))
            found["sort_call_with_ordering_signal"] = _evidence(
                "sort_call_with_ordering_signal",
                lines,
                [sort_index + 1, signal_index + 1],
                ["sort-like call", "nearby ordering/comparator/key signal"],
            )
            break

    if language == "rust":
        members_by_type: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i, line in enumerate(code_lines):
            for match in QUALIFIED_MEMBER_RE.finditer(line):
                members_by_type[match.group(1)][match.group(2)].append(i)
        type_pairs: list[tuple[int, str, str, list[tuple[str, int, int]]]] = []
        type_names = sorted(members_by_type)
        for left_index, left_type in enumerate(type_names):
            for right_type in type_names[left_index + 1 :]:
                overlaps: list[tuple[str, int, int]] = []
                for member in sorted(
                    set(members_by_type[left_type]) & set(members_by_type[right_type])
                ):
                    left_lines = members_by_type[left_type][member]
                    right_lines = members_by_type[right_type][member]
                    pair = min(
                        ((left, right) for left in left_lines for right in right_lines),
                        key=lambda item: (abs(item[0] - item[1]), item),
                    )
                    if abs(pair[0] - pair[1]) <= 80:
                        overlaps.append((member, pair[0], pair[1]))
                if len(overlaps) >= 2:
                    chosen_overlaps = overlaps[:2]
                    span = max(max(pair[1], pair[2]) for pair in chosen_overlaps) - min(
                        min(pair[1], pair[2]) for pair in chosen_overlaps
                    )
                    type_pairs.append((span, left_type, right_type, overlaps))
        if type_pairs:
            _, left_type, right_type, overlaps = min(
                type_pairs, key=lambda item: (item[0], item[1], item[2])
            )
            support_lines: list[int] = []
            roles: list[str] = []
            for member, left_line, right_line in overlaps[:2]:
                support_lines.extend([left_line + 1, right_line + 1])
                roles.extend([f"{left_type}::{member}", f"{right_type}::{member}"])
            found["qualified_type_member_overlap"] = _evidence(
                "qualified_type_member_overlap", lines, support_lines, roles
            )
    return found


def _file_spdx_notices(source: str) -> list[str]:
    notices: list[str] = []
    for line in source.splitlines():
        match = SPDX_RE.match(line)
        if match:
            value = match.group(1).strip().strip("*/ ")
            if value:
                notices.append(value)
    return notices


def _license_review(record: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    metadata = record.get("authoring_metadata")
    seed = record.get("student_state_seed")
    if not isinstance(metadata, dict) or not isinstance(seed, dict):
        return False, {}, "missing_source_metadata"
    source = seed.get("source")
    if not isinstance(source, str):
        return False, {}, "missing_source_text"
    notices = _file_spdx_notices(source)
    metadata_notice = metadata.get("file_spdx_notice")
    if isinstance(metadata_notice, str) and metadata_notice.strip():
        notices.append(metadata_notice.strip())
    unique_notices = sorted(set(notices))
    common = {
        "repository_license": metadata.get("source_license"),
        "repository_license_path": metadata.get("license_path"),
        "repository_license_sha256": metadata.get("license_sha256"),
        "file_spdx_notices": unique_notices,
    }
    if metadata.get("source_license") != "MIT":
        return False, common, "repository_license_not_exactly_mit"
    if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("license_sha256", ""))):
        return False, common, "missing_repository_license_hash"
    if len(unique_notices) > 1:
        return False, common, "file_spdx_metadata_conflict"
    if unique_notices and unique_notices[0] != "MIT":
        return False, common, "file_spdx_conflicts_with_repository_mit"
    common["file_scope"] = (
        "explicit_mit_notice" if unique_notices else "unverified_without_file_notice"
    )
    return True, common, None


def _validate_source_record(record: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
    metadata = record.get("authoring_metadata")
    seed = record.get("student_state_seed")
    if not isinstance(metadata, dict) or not isinstance(seed, dict):
        return False, "missing_source_metadata", {}
    source = seed.get("source")
    language = seed.get("filetype")
    source_id = record.get("id")
    if not isinstance(source, str) or language not in {"python", "go", "rust", "typescript"}:
        return False, "unsupported_or_missing_source", {}
    source_hash = sha256_bytes(source.encode("utf-8"))
    # source_sha256 is the hash of the frozen source body in this bank. The
    # r2_discovery_content_sha256 field records an earlier discovery object and
    # can differ for rows deliberately refetched at the immutable source rev.
    if source_hash != metadata.get("source_sha256"):
        return False, "source_content_hash_mismatch", {}
    if metadata.get("source_provenance_verified") is not True:
        return False, "source_provenance_not_verified", {}
    if not isinstance(source_id, str) or not source_id.startswith("public-source/"):
        return False, "invalid_source_id", {}
    revision = metadata.get("source_revision")
    source_url = metadata.get("source_url")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        return False, "source_revision_not_immutable", {}
    if not isinstance(source_url, str) or revision not in source_url:
        return False, "source_url_revision_mismatch", {}
    return True, None, {"source_sha256": source_hash, "language": language}


def _load_bank(raw: bytes) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on source-bank line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"source-bank line {line_number} is not an object")
        source_id = record.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source-bank record has an invalid source ID")
        if source_id in seen:
            raise ValueError("duplicate source ID in pinned bank")
        seen.add(source_id)
        records.append(record)
    return records


def build_selection(
    raw: bytes, selector_bytes: bytes
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bank_hash = sha256_bytes(raw)
    if bank_hash != EXPECTED_BANK_SHA256:
        raise ValueError("pinned source bank hash mismatch; refusing selection")
    rows = _load_bank(raw)
    rejects: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    all_family_counts: Counter[str] = Counter()
    license_scope_counts: Counter[str] = Counter()
    passed_identity_license = 0

    for record in rows:
        valid, reason, identity = _validate_source_record(record)
        if not valid:
            rejects[reason or "invalid_source"] += 1
            continue
        licensed, license_review, reason = _license_review(record)
        if not licensed:
            rejects[reason or "license_review_failed"] += 1
            continue
        passed_identity_license += 1
        seed = record["student_state_seed"]
        matches = detect_source_evidence(seed["source"], seed["filetype"])
        if not matches:
            rejects["no_declared_source_evidence"] += 1
            continue
        for family in matches:
            all_family_counts[family] += 1
        license_scope_counts[license_review["file_scope"]] += 1
        provenance = record["authoring_metadata"]
        candidates.append(
            {
                "seed_id": record["id"],
                "student_state_seed": {
                    "file_id": seed.get("file_id"),
                    "filetype": seed["filetype"],
                    "source": seed["source"],
                },
                "source_provenance": {
                    "source_repo": provenance.get("source_repo"),
                    "source_path": provenance.get("source_path"),
                    "source_revision": provenance.get("source_revision"),
                    "source_sha256": identity["source_sha256"],
                    "source_url": provenance.get("source_url"),
                    "repository_license": license_review["repository_license"],
                    "repository_license_path": license_review["repository_license_path"],
                    "repository_license_sha256": license_review["repository_license_sha256"],
                    "file_spdx_notices": license_review["file_spdx_notices"],
                    "file_license_scope": license_review["file_scope"],
                },
                "detected_source_evidence": [
                    matches[family] for family in FAMILY_ORDER if family in matches
                ],
                "authoring_status": "source_anchor_only_not_an_action_or_training_example",
                "training_eligible": False,
            }
        )

    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        for observation in candidate["detected_source_evidence"]:
            family = observation["family"]
            rank = sha256_bytes(f"{rules_sha256()}\0{family}\0{candidate['seed_id']}".encode())
            queues[family].append({"rank": rank, "candidate": candidate})
    for queue in queues.values():
        queue.sort(key=lambda item: (item["rank"], item["candidate"]["seed_id"]))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_repos: set[str] = set()
    family_slots: Counter[str] = Counter()
    family_language_slots: Counter[tuple[str, str]] = Counter()
    selected_by_queue: Counter[str] = Counter()
    # Scarce queues run first so a single supported family does not disappear
    # behind common structures. Round robin then fills each bounded queue. The
    # queue is an allocation aid, not a ground-truth label.
    allocation_order = sorted(
        FAMILY_ORDER,
        key=lambda family: (
            len({item["candidate"]["seed_id"] for item in queues.get(family, [])}),
            FAMILY_ORDER.index(family),
        ),
    )
    while len(selected) < MAX_SELECTED:
        progressed = False
        for family in allocation_order:
            if family_slots[family] >= PER_FAMILY_CAP:
                continue
            for item in queues.get(family, []):
                candidate = item["candidate"]
                source_id = candidate["seed_id"]
                repo = candidate["source_provenance"]["source_repo"]
                language = candidate["student_state_seed"]["filetype"]
                if source_id in selected_ids or repo in selected_repos:
                    continue
                if family_language_slots[(family, language)] >= PER_LANGUAGE_FAMILY_CAP:
                    continue
                selected_candidate = dict(candidate)
                selected_candidate["allocation_queue"] = family
                selected.append(selected_candidate)
                selected_ids.add(source_id)
                selected_repos.add(repo)
                family_slots[family] += 1
                family_language_slots[(family, language)] += 1
                selected_by_queue[family] += 1
                progressed = True
                break
            if len(selected) >= MAX_SELECTED:
                break
        if not progressed:
            break

    seed_lines = b"".join(canonical_json(item) + b"\n" for item in selected)
    plan_without_hash: dict[str, Any] = {
        "schema": "public-mechanism-source-preflight-v1",
        "selector_version": SELECTOR_VERSION,
        "status": "frozen_source_preflight_no_provider_calls",
        "source_bank_sha256": bank_hash,
        "source_bank_rows": len(rows),
        "selector_sha256": sha256_bytes(selector_bytes),
        "selection_rules": SELECTION_RULES,
        "selection_rules_sha256": rules_sha256(),
        "allocation_family_order": allocation_order,
        "selection_limits": {
            "max_selected": MAX_SELECTED,
            "per_family_queue": PER_FAMILY_CAP,
            "per_language_per_family_queue": PER_LANGUAGE_FAMILY_CAP,
            "per_repository": PER_REPOSITORY_CAP,
        },
        "selector_inputs": {
            "authoring_focus_fields_read": False,
            "candidate_split_fields_read": False,
            "source_downloads": 0,
            "provider_calls": 0,
        },
        "results": {
            "source_rows": len(rows),
            "source_rows_passing_provenance_and_license_screen": passed_identity_license,
            "structural_candidates": len(candidates),
            "selected_source_seeds": len(selected),
            "selected_seed_ids": [item["seed_id"] for item in selected],
            "detected_evidence_family_counts": {
                family: all_family_counts[family] for family in FAMILY_ORDER
            },
            "selected_allocation_queue_counts": {
                family: selected_by_queue[family] for family in FAMILY_ORDER
            },
            "file_license_scope_counts_among_structural_candidates": dict(
                sorted(license_scope_counts.items())
            ),
            "rejected_rows_by_reason": dict(sorted(rejects.items())),
            "remaining_missing_evidence": {
                "blind_authoring_and_solver_result": "not run",
                "independent_objective_check": "not run",
                "file_level_license_scope_for_unnoted_files": "unverified",
                "accepted_training_labels": 0,
            },
        },
        "selected_seed_lines_sha256": sha256_bytes(seed_lines),
        "teacher_call_gate": (
            "no call from this script; this plan and seed file must be "
            "reviewed before a separate authoring action"
        ),
    }
    plan_without_hash["plan_content_sha256"] = sha256_bytes(canonical_json(plan_without_hash))
    return plan_without_hash, selected


def _private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("refusing symlink output directory")
    if not path.exists():
        path.mkdir(mode=0o700, parents=False)
    st = path.stat()
    if not stat.S_ISDIR(st.st_mode):
        raise ValueError("output path is not a directory")
    if st.st_uid != os.getuid():
        raise ValueError("output directory is not owned by the current user")
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise ValueError("output directory must be owner-only (mode 0700)")


def _write_new_or_identical(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlink output file: {path.name}")
    if path.exists():
        st = path.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            raise ValueError(f"unsafe existing output file: {path.name}")
        if stat.S_IMODE(st.st_mode) & 0o077:
            raise ValueError(f"existing output file is not private: {path.name}")
        if path.read_bytes() != data:
            raise ValueError(f"frozen output differs from existing file: {path.name}")
        return
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temp_path = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temp_path, 0o600)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_outputs(
    output_dir: Path, plan: dict[str, Any], selected: list[dict[str, Any]]
) -> tuple[Path, Path]:
    _private_directory(output_dir)
    plan_path = output_dir / "selection_plan.json"
    seeds_path = output_dir / "selected_seeds.jsonl"
    plan_bytes = (
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    seed_bytes = b"".join(canonical_json(item) + b"\n" for item in selected)
    if sha256_bytes(seed_bytes) != plan["selected_seed_lines_sha256"]:
        raise ValueError("selected seed serialization differs from frozen plan")
    _write_new_or_identical(plan_path, plan_bytes)
    _write_new_or_identical(seeds_path, seed_bytes)
    return plan_path, seeds_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execute", action="store_true", help="write the frozen offline preflight outputs"
    )
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("pass --execute to write the offline, provider-free selection")
    try:
        raw = args.source_bank.read_bytes()
        script_bytes = Path(__file__).read_bytes()
        plan, selected = build_selection(raw, script_bytes)
        plan_path, seeds_path = write_outputs(args.output_dir, plan, selected)
    except (OSError, ValueError) as error:
        print(f"selection failed: {error}", file=sys.stderr)
        return 2
    result = plan["results"]
    print(
        json.dumps(
            {
                "status": plan["status"],
                "source_bank_rows": result["source_rows"],
                "structural_candidates": result["structural_candidates"],
                "selected_source_seeds": result["selected_source_seeds"],
                "detected_evidence_family_counts": result["detected_evidence_family_counts"],
                "selected_allocation_queue_counts": result["selected_allocation_queue_counts"],
                "rejected_rows_by_reason": result["rejected_rows_by_reason"],
                "plan_path": str(plan_path),
                "seeds_path": str(seeds_path),
                "provider_calls": 0,
                "source_downloads": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
