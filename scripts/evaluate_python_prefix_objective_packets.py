#!/usr/bin/env python3
"""Freeze and check eight Python-only public-source next-edit objectives.

This is a bounded diagnostic. Source tasks use synthetic typed-append history;
the isolated function harness is a transformation, not a whole-project check.
Generated function code is parsed/compiled/run only in the existing pinned,
network-disabled OCI evaluator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import textwrap
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinycomplete.eval.code_benchmark import (
    BenchmarkCase,
    CheckSpec,
    Prediction,
    evaluate_prediction,
)
from tinycomplete.observability.context import RunContext
from tinycomplete.one_line.contract import (
    MAX_ACTION_TOKENS,
    EditAction,
    EditState,
    apply_action,
    encode_action,
    physical_lines,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/prototype/product_r2"
PACKAGE_ROOT = Path("/mnt/ssd/tabcomplete-product-r2/commitpackft")
MATERIALIZED = PACKAGE_ROOT / "verified-parent-prefix-materialized-v3"
EXPANSION = PACKAGE_ROOT / "source-verification-expansion-v3"
TREE_OUTPUT = PACKAGE_ROOT / "python-prefix-parent-tree-v1"
PLAN_PATH = REPORT / "python_prefix_objective_oracle_plan_v2.json"
OUTPUT_DIR = PACKAGE_ROOT / "python-prefix-objective-oracle-v2"
TOKENIZER_DIR = Path(
    "/home/crabcake/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B"
    "/snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
TOKENIZER_REVISION = "Qwen/Qwen2.5-Coder-0.5B@8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
MAX_INPUT_TOKENS = 1024
MAX_TOTAL_TOKENS = 2048
MAX_ROWS = 8
MAX_OUTPUT_BYTES = 16 * 1024**2
MAX_CHECKS = MAX_ROWS * 3
MAX_WALL_SECONDS = 15 * 60
PYTHON_IMAGE = (
    "docker.io/library/python:3.12-slim@sha256:"
    "2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9"
)
PYTHON_RUNTIME_ID = {
    "kind": "sandbox_container",
    "identity": PYTHON_IMAGE,
    "identity_sha256": hashlib.sha256(PYTHON_IMAGE.encode()).hexdigest(),
}
PYTHON_RUNTIME_SHA256 = hashlib.sha256(
    json.dumps(PYTHON_RUNTIME_ID, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

OBJECTIVE_IDS = (
    "commit-sequence/1008cf90b10cfe58c9bf9886/return-prefix-00",
    "commit-sequence/27ee64f11bb4e7c355ba5f1c/return-prefix-01",
    "commit-sequence/56cd12609c93cfef18ad6048/return-prefix-00",
    "commit-sequence/84d05fc0a76d009cc8968ad5/return-prefix-00",
    "commit-sequence/9a7745914da0d7ac44bcf10f/return-prefix-00",
    "commit-sequence/b5931eca4a4bb5bb4503d539/return-prefix-00",
    "commit-sequence/f14a31f007558fbc6334bc42/return-prefix-00",
    "commit-sequence/91e9864d9a5e9d92ebde3c35/return-prefix-01",
)

# Test code is independent of the hidden target line and remains answer-side.
OBJECTIVES: dict[str, dict[str, str]] = {
    OBJECTIVE_IDS[0]: {
        "contract": "preserve_base_context_and_add_nonempty_active_product_list",
        "intent_basis": "adjacent_source_comment",
        "wrong_expression": "{}",
        "test": """
import types
import solution

class Products:
    def __init__(self, rows):
        self.rows = rows
    def filter(self, **kwargs):
        assert kwargs == {"active": True}
        return self.rows

class Base:
    def get_context_data(self, **kwargs):
        return {"base": kwargs.get("base", "kept")}

class View(Base):
    pass

solution.CatalogDetailView = View
View.get_context_data = solution.get_context_data
items = [object()]
view = View()
view.object = types.SimpleNamespace(products=Products(items))
assert view.get_context_data(base="kept") == {"base": "kept", "product_list": items}
view.object = types.SimpleNamespace(products=Products([]))
assert view.get_context_data(base="kept") == {"base": "kept"}
""",
    },
    OBJECTIVE_IDS[1]: {
        "contract": "return_true_exactly_for_decimal_digit_palindromes",
        "intent_basis": "function_docstring",
        "wrong_expression": "True",
        "test": """
import solution

for value, expected in ((121, True), (123, False), (0, True), (10, False), (-121, False)):
    assert solution.isPalindrome(value) is expected
""",
    },
    OBJECTIVE_IDS[2]: {
        "contract": "validate_lists_and_every_member_with_supplied_validator",
        "intent_basis": "function_docstring",
        "wrong_expression": "lambda value: True",
        "test": """
import solution

class Validators:
    @staticmethod
    def instance_of(kind):
        return lambda value: isinstance(value, kind)
    @staticmethod
    def deep_iterable(member_validator, iterable_validator):
        def check(value):
            return iterable_validator(value) and all(member_validator(item) for item in value)
        return check

solution.validators = Validators()
check = solution.list_of(lambda item: isinstance(item, int) and not isinstance(item, bool))
assert check([1, 2, 3])
assert check([])
assert not check([1, "bad"])
assert not check("123")
""",
    },
    OBJECTIVE_IDS[3]: {
        "contract": "integrate_the_piecewise_linear_curve_by_the_trapezoidal_rule",
        "intent_basis": "function_docstring",
        "wrong_expression": "0.0",
        "test": """
import solution

class Vec(list):
    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            return Vec(super().__getitem__(i) for i in index)
        return super().__getitem__(index)

class NumpyStub:
    @staticmethod
    def concatenate(parts):
        return Vec(item for part in parts for item in part)
    @staticmethod
    def argsort(values):
        return sorted(range(len(values)), key=values.__getitem__)

solution.np = NumpyStub()
solution.array = Vec
assert abs(solution.trapz(Vec([0.75, 0.25, 0.5]), Vec([0.75, 0.25, 0.5])) - 0.5) < 1e-12
""",
    },
    OBJECTIVE_IDS[4]: {
        "contract": "return_successful_scss_compiler_output_and_raise_on_failure",
        "intent_basis": "function_docstring",
        "wrong_expression": "None",
        "test": """
import types
import solution

solution.os = types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda _path: True))
solution.safe_join = lambda root, path: root + "/" + path
solution.settings = types.SimpleNamespace(SMATIC_SCSS_PATH="/scss", SASS_BIN="sass")
solution.getstatusoutput = lambda _command: (0, "compiled css")
assert solution.scss("style.scss") == "compiled css"
solution.getstatusoutput = lambda _command: (1, "compiler error")
try:
    solution.scss("style.scss")
except Exception as error:
    assert str(error) == "compiler error"
else:
    raise AssertionError("nonzero compiler status must raise")
""",
    },
    OBJECTIVE_IDS[5]: {
        "contract": "return_only_posts_accepted_by_the_predicate",
        "intent_basis": "function_docstring",
        "wrong_expression": "self",
        "test": """
import solution

class Posts(list):
    pass

solution.Posts = Posts
posts = Posts([1, 2, 3, 4])
result = solution.filter(posts, lambda post: post % 2 == 0)
assert list(result) == [2, 4]
""",
    },
    OBJECTIVE_IDS[6]: {
        "contract": "report_true_only_for_a_parseable_nonempty_musicbrainz_uuid",
        "intent_basis": "function_docstring",
        "wrong_expression": "False",
        "test": """
import types
import uuid
import solution

solution.uuid = uuid
def track(value):
    return types.SimpleNamespace(mbid=types.SimpleNamespace(value=value))

assert solution.has_mbid(track("550e8400-e29b-41d4-a716-446655440000")) is True
assert solution.has_mbid(track("not-a-uuid")) is False
assert solution.has_mbid(track("")) is False
""",
    },
    OBJECTIVE_IDS[7]: {
        "contract": "return_fixed_width_timestamp_plus_two_random_bytes_as_hex",
        "intent_basis": "function_docstring",
        "wrong_expression": "''",
        "test": """
import binascii
import datetime
import types
import solution

class FixedDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2024, 1, 2, 3, 4, 5, 678901)

solution.datetime = types.SimpleNamespace(datetime=FixedDateTime)
solution.binascii = binascii
def randombytes(size):
    return b"\\xab\\xcd" if size == 2 else b""
solution.libnacl = types.SimpleNamespace(randombytes=randombytes)
assert solution.time_nonce() == "20240102030405678901abcd"
""",
    },
}


class ObjectivePacketError(ValueError):
    """Sanitized error code for objective packet preparation."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _canonical_sha(value: Any) -> str:
    return _sha(_canonical(value))


def _artifact_descriptor(provenance: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "path": provenance[key + "_path"],
        "sha256": provenance[key + "_sha256"],
        "bytes": provenance[key + "_bytes"],
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_private(path: Path, payload: bytes, *, exclusive: bool = True) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def _append_private_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(fd, "ab") as stream:
        stream.write(_canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_execution_state(path: Path, value: dict[str, Any]) -> None:
    _write_private(path, _canonical(value) + b"\n", exclusive=not path.exists())


def _load_tokenizer() -> tuple[Any, dict[str, Any]]:
    if not (TOKENIZER_DIR / "tokenizer.json").is_file():
        raise ObjectivePacketError("pinned_q25_tokenizer_missing")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_DIR, local_files_only=True, trust_remote_code=False
        )
    except Exception:
        raise ObjectivePacketError("pinned_q25_tokenizer_load_failed") from None
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    files = {}
    for name in names:
        path = TOKENIZER_DIR / name
        if path.is_file():
            payload = path.read_bytes()
            files[name] = {"bytes": len(payload), "sha256": _sha(payload)}
    return tokenizer, {
        "source": TOKENIZER_REVISION,
        "files": files,
        "eos_token_id": tokenizer.eos_token_id,
        "transformers_version": importlib.metadata.version("transformers"),
    }


def _load_bound_artifact(descriptor: dict[str, Any]) -> bytes:
    relative = Path(str(descriptor.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ObjectivePacketError("source_artifact_path_invalid")
    candidate = PACKAGE_ROOT / relative
    if candidate.is_symlink():
        raise ObjectivePacketError("source_artifact_not_regular")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(PACKAGE_ROOT.resolve(strict=True))
    except ValueError:
        raise ObjectivePacketError("source_artifact_path_escape") from None
    if path.is_symlink() or not path.is_file():
        raise ObjectivePacketError("source_artifact_not_regular")
    payload = path.read_bytes()
    if len(payload) != descriptor.get("bytes") or _sha(payload) != descriptor.get("sha256"):
        raise ObjectivePacketError("source_artifact_identity_mismatch")
    return payload


def _bound_state_action(
    row: dict[str, Any], provenance: dict[str, Any]
) -> tuple[EditState, EditAction, str, bytes]:
    try:
        state = EditState.from_mapping(row["state"])
        source = _load_bound_artifact(
            {
                "path": provenance["source_artifact_path"],
                "sha256": provenance["source_artifact_sha256"],
                "bytes": provenance["source_artifact_bytes"],
            }
        )
        source_lines = physical_lines(source)
        current_lines = physical_lines(state.source.encode("utf-8"))
        target_row = state.target_row
        if target_row >= len(source_lines) or target_row >= len(current_lines):
            raise ObjectivePacketError("target_row_outside_parent_source")
        expected_prefix = b"".join(line.raw for line in source_lines[:target_row])
        current_prefix = b"".join(line.raw for line in current_lines[:target_row])
        target_source_line = source_lines[target_row]
        target_state_line = current_lines[target_row]
        if (
            current_prefix != expected_prefix
            or target_state_line.content != target_source_line.content[: state.cursor_col]
            or target_state_line.terminator != target_source_line.terminator
            or len(current_lines) != target_row + 1
        ):
            raise ObjectivePacketError("source_to_cursor_replay_mismatch")
        action = EditAction("replace_line", target_source_line.content.decode("utf-8", "strict"))
        after = apply_action(state, action)
        return state, action, after, source
    except (KeyError, UnicodeError, TypeError, ValueError) as error:
        if isinstance(error, ObjectivePacketError):
            raise
        raise ObjectivePacketError("source_state_or_action_invalid") from None


def _extract_function(after_source: str, target_row: int) -> tuple[str, dict[str, Any]]:
    try:
        tree = ast.parse(after_source)
    except SyntaxError:
        raise ObjectivePacketError("completed_source_not_python_syntax") from None
    target_line = target_row + 1
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and isinstance(node.end_lineno, int)
        and node.lineno <= target_line <= node.end_lineno
    ]
    if not candidates:
        raise ObjectivePacketError("target_not_inside_function")
    node = min(candidates, key=lambda item: (item.end_lineno or item.lineno) - item.lineno)
    end_line = node.end_lineno
    if end_line is None:
        raise ObjectivePacketError("function_end_line_missing")
    lines = after_source.splitlines(keepends=True)
    start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    segment = textwrap.dedent("".join(lines[start - 1 : end_line]))
    if not segment.endswith("\n"):
        segment += "\n"
    descriptor = {
        "kind": "python_function_slice_v1",
        "function_name": node.name,
        "start_line": start,
        "end_line": end_line,
        "source_sha256": _sha(segment.encode("utf-8")),
    }
    return segment, descriptor


def _objective_slug(candidate_id: str) -> str:
    return _sha(candidate_id.encode())[:20]


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in indexed:
            raise ObjectivePacketError("candidate_index_identity_invalid")
        indexed[candidate_id] = row
    return indexed


def _rows() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    inputs = _jsonl(MATERIALIZED / "review_inputs.jsonl")
    provenance = _jsonl(MATERIALIZED / "provenance_private.jsonl")
    tree_rows = _jsonl(TREE_OUTPUT / "parent_trees.jsonl")
    expansion_rows = _jsonl(EXPANSION / "candidate_results.jsonl")
    input_by_id = _index_rows(inputs)
    provenance_by_id = _index_rows(provenance)
    tree_by_id = _index_rows(tree_rows)
    expansion_by_id = _index_rows(expansion_rows)
    if (
        set(OBJECTIVE_IDS) - set(input_by_id)
        or set(OBJECTIVE_IDS) - set(provenance_by_id)
        or set(OBJECTIVE_IDS) - set(tree_by_id)
    ):
        raise ObjectivePacketError("frozen_candidate_records_incomplete")
    return input_by_id, provenance_by_id, tree_by_id, expansion_by_id


def _replace_history_line(source: str, state: EditState, row: int, text: str) -> str:
    lines = physical_lines(source.encode("utf-8"))
    if row < 0 or row >= len(lines):
        raise ObjectivePacketError("history_row_outside_state")
    target = EditState(state.file_id, state.filetype, source, row, 0)
    return apply_action(target, EditAction("replace_line", text))


def _verify_history_replay(state: EditState) -> None:
    """Check that synthetic events reconstruct the exact state in both directions."""
    replay = state.source
    for edit in reversed(state.history):
        lines = physical_lines(replay.encode("utf-8"))
        if edit.row >= len(lines) or lines[edit.row].content.decode("utf-8") != edit.new_text:
            raise ObjectivePacketError("history_reverse_replay_mismatch")
        replay = _replace_history_line(replay, state, edit.row, edit.old_text)
    for edit in state.history:
        lines = physical_lines(replay.encode("utf-8"))
        if edit.row >= len(lines) or lines[edit.row].content.decode("utf-8") != edit.old_text:
            raise ObjectivePacketError("history_forward_replay_mismatch")
        replay = _replace_history_line(replay, state, edit.row, edit.new_text)
    if replay.encode("utf-8") != state.source.encode("utf-8"):
        raise ObjectivePacketError("history_does_not_reconstruct_exact_state")


def _build_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer, tokenizer_identity = _load_tokenizer()
    input_by_id, provenance_by_id, tree_by_id, expansion_by_id = _rows()
    if len(OBJECTIVE_IDS) > MAX_ROWS or len(set(OBJECTIVE_IDS)) != len(OBJECTIVE_IDS):
        raise ObjectivePacketError("candidate_count_or_identity_invalid")
    seen_groups: set[str] = set()
    seen_near: set[str] = set()
    cases = []
    bound_artifact_bytes = 0
    for candidate_id in OBJECTIVE_IDS:
        row = input_by_id[candidate_id]
        provenance = provenance_by_id[candidate_id]
        tree = tree_by_id[candidate_id]
        expansion = expansion_by_id.get(str(provenance.get("license_scope_candidate_id")))
        if (
            expansion is None
            or provenance.get("split") != "train"
            or row.get("seed_id") != candidate_id
            or row.get("context_sha256") != _sha(str(row.get("prompt", "")).encode("utf-8"))
            or provenance.get("source_group_id") in seen_groups
            or provenance.get("near_duplicate_key") in seen_near
            or provenance.get("accepted_training") is not False
            or provenance.get("human_chronology_observed") is not False
            or provenance.get("inferability_reviewed") is not False
            or provenance.get("objective_verified") is not False
            or tree.get("source_tree_sha") is None
            or tree.get("source_sha256") != provenance.get("source_artifact_sha256")
            or tree.get("parent_commit") != provenance.get("session_or_commit")
            or tree.get("source_path") != provenance.get("source_path")
            or expansion.get("parent_commit") != tree.get("parent_commit")
            or expansion.get("source_sha256") != provenance.get("source_artifact_sha256")
            or expansion.get("source_path") != provenance.get("source_path")
            or expansion.get("source_revision") != provenance.get("source_revision")
        ):
            raise ObjectivePacketError("candidate_group_or_provenance_invalid")
        seen_groups.add(str(provenance["source_group_id"]))
        seen_near.add(str(provenance["near_duplicate_key"]))
        state, gold_action, gold_after, source_bytes = _bound_state_action(row, provenance)
        _verify_history_replay(state)
        objective = OBJECTIVES[candidate_id]
        current_lines = physical_lines(state.source.encode("utf-8"))
        indent_bytes = current_lines[state.target_row].content
        indent = indent_bytes[: len(indent_bytes) - len(indent_bytes.lstrip(b" \t"))].decode()
        wrong_action = EditAction(
            "replace_line", indent + "return " + objective["wrong_expression"]
        )
        if wrong_action == gold_action or apply_action(state, wrong_action) == state.source:
            raise ObjectivePacketError("wrong_control_is_not_a_distinct_edit")
        prompt_tokens = len(tokenizer.encode(row["prompt"], add_special_tokens=True))
        response_tokens = (
            len(tokenizer.encode(encode_action(gold_action), add_special_tokens=False)) + 1
        )
        if (
            prompt_tokens > MAX_INPUT_TOKENS
            or response_tokens > MAX_ACTION_TOKENS
            or prompt_tokens + response_tokens > MAX_TOTAL_TOKENS
        ):
            raise ObjectivePacketError("candidate_exceeds_frozen_token_budget")
        fixture_source = objective["test"].lstrip("\n")
        fixture_payload = fixture_source.encode("utf-8")
        slug = _objective_slug(candidate_id)
        fixture_path = f"artifacts/fixtures/{slug}.py"
        state_raw = row["state"]
        transform = {
            "kind": "synthetic_typed_return_prefix_v1",
            "parent_source_sha256": provenance["source_artifact_sha256"],
            "target_physical_row": state.target_row,
            "cursor_byte_column": state.cursor_col,
            "state_reconstructed_from_exact_parent_prefix": True,
            "history_origin": "synthetic_editor_typing",
            "history_sha256": _canonical_sha(state_raw.get("history", [])),
            "source_suffix_after_cursor_in_input": False,
        }

        source_descriptor = _artifact_descriptor(provenance, "source_artifact")
        path_license_descriptor = _artifact_descriptor(provenance, "path_license_artifact")
        root_license_descriptor = _artifact_descriptor(provenance, "root_license_artifact")
        scope_descriptor = _artifact_descriptor(provenance, "license_scope_artifact")
        for artifact in (
            source_descriptor,
            path_license_descriptor,
            root_license_descriptor,
            scope_descriptor,
        ):
            bound_artifact_bytes += len(_load_bound_artifact(artifact))
        if bound_artifact_bytes > MAX_OUTPUT_BYTES:
            raise ObjectivePacketError("bound_public_artifacts_exceed_storage_cap")
        tree_sha = tree["source_tree_sha"]
        canonical_gold = asdict(gold_action)
        canonical_wrong = asdict(wrong_action)
        action_context = {
            "candidate_id": candidate_id,
            "seed_id": candidate_id,
            "split": "train",
            "source_group_id": provenance["source_group_id"],
            "state_sha256": _canonical_sha(state_raw),
            "context_sha256": row["context_sha256"],
            "history_before_sha256": _canonical_sha(state_raw.get("history", [])),
            "action_sha256": _canonical_sha(canonical_gold),
            "after_source_sha256": _sha(gold_after.encode("utf-8")),
            "wrong_action_sha256": _canonical_sha(canonical_wrong),
            "source_sha256": provenance["source_artifact_sha256"],
            "source_tree_sha": tree_sha,
            "source_bytes": len(source_bytes),
            "transform_sha256": _canonical_sha(transform),
            "fixture_sha256": _sha(fixture_payload),
            "fixture_bytes": len(fixture_payload),
            "fixture_path": fixture_path,
            "fixture_contract": objective["contract"],
            "visible_intent_basis": objective["intent_basis"],
            "input_tokens": prompt_tokens,
            "response_tokens_including_eos": response_tokens,
            "function_extraction": _extract_function(gold_after, state.target_row)[1],
        }
        cases.append(
            {
                "candidate_id": candidate_id,
                "source_group_id": provenance["source_group_id"],
                "near_duplicate_key": provenance["near_duplicate_key"],
                "source_verification_candidate_id": provenance["license_scope_candidate_id"],
                "source_repo": provenance["source_repo"],
                "source_revision": provenance["source_revision"],
                "parent_commit": tree["parent_commit"],
                "source_path": provenance["source_path"],
                "source_tree_sha": tree_sha,
                "source_sha256": provenance["source_artifact_sha256"],
                "state_sha256": action_context["state_sha256"],
                "context_sha256": row["context_sha256"],
                "history_before_sha256": action_context["history_before_sha256"],
                "gold_action_sha256": action_context["action_sha256"],
                "wrong_action_sha256": action_context["wrong_action_sha256"],
                "after_source_sha256": action_context["after_source_sha256"],
                "transform": transform,
                "transform_sha256": action_context["transform_sha256"],
                "fixture": {
                    "path": fixture_path,
                    "sha256": action_context["fixture_sha256"],
                    "bytes": action_context["fixture_bytes"],
                },
                "runtime": PYTHON_RUNTIME_ID,
                "runtime_sha256": PYTHON_RUNTIME_SHA256,
                "objective_contract": objective["contract"],
                "visible_intent_basis": objective["intent_basis"],
                "artifact_descriptors": {
                    "source_artifact": source_descriptor,
                    "path_license_artifact": path_license_descriptor,
                    "root_license_artifact": root_license_descriptor,
                    "license_scope_artifact": scope_descriptor,
                },
                "input_tokens": prompt_tokens,
                "response_tokens_including_eos": response_tokens,
                "function_extraction": action_context["function_extraction"],
                "state_object": state_raw,
                "action_object": canonical_gold,
                "wrong_action_object": canonical_wrong,
                "after_source": gold_after,
                "fixture_source": fixture_source,
            }
        )
    if len({case["source_group_id"] for case in cases}) != len(cases):
        raise ObjectivePacketError("selected_groups_not_disjoint")
    return cases, tokenizer_identity


def build_plan() -> dict[str, Any]:
    cases, tokenizer_identity = _build_cases()
    identity_inputs = {
        "materialization_manifest": MATERIALIZED / "manifest.json",
        "review_inputs": MATERIALIZED / "review_inputs.jsonl",
        "provenance_private": MATERIALIZED / "provenance_private.jsonl",
        "expansion_manifest": EXPANSION / "manifest.json",
        "candidate_results": EXPANSION / "candidate_results.jsonl",
        "tree_manifest": TREE_OUTPUT / "manifest.json",
        "tree_rows": TREE_OUTPUT / "parent_trees.jsonl",
    }
    return {
        "schema": "python-prefix-objective-oracle-plan-v2",
        "purpose": "bounded synthetic task diagnostics; not human chronology or training approval",
        "inputs": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha(path.read_bytes()),
            }
            for name, path in identity_inputs.items()
        },
        "source_rows": [
            {
                key: case[key]
                for key in (
                    "candidate_id",
                    "source_verification_candidate_id",
                    "source_group_id",
                    "source_repo",
                    "source_revision",
                    "parent_commit",
                    "source_path",
                    "source_tree_sha",
                    "source_sha256",
                    "state_sha256",
                    "context_sha256",
                    "history_before_sha256",
                    "gold_action_sha256",
                    "wrong_action_sha256",
                    "after_source_sha256",
                    "transform_sha256",
                    "fixture",
                    "runtime_sha256",
                    "objective_contract",
                    "visible_intent_basis",
                    "input_tokens",
                    "response_tokens_including_eos",
                    "function_extraction",
                )
            }
            for case in cases
        ],
        "tokenizer": tokenizer_identity,
        "task_contract": {
            "codec": "single-line-edit-v1",
            "maximum_input_tokens": MAX_INPUT_TOKENS,
            "maximum_response_tokens_including_eos": MAX_ACTION_TOKENS,
            "maximum_total_tokens": MAX_TOTAL_TOKENS,
            "all_targets_are_replacements": True,
            "synthetic_history": True,
            "human_chronology_observed": False,
            "no_idle_or_netzero_no_edit_labels": True,
        },
        "oracle": {
            "backend": "container",
            "network": "none",
            "image": PYTHON_IMAGE,
            "runtime_identity": PYTHON_RUNTIME_ID,
            "runtime_sha256": PYTHON_RUNTIME_SHA256,
            "compile": ["python", "-m", "py_compile", "solution.py"],
            "test": ["python", "oracle.py"],
            "expected": {
                "gold": {"parse_status": "pass", "compile_status": "pass", "test_status": "pass"},
                "before": {
                    "parse_status": "fail",
                    "compile_status": "fail",
                    "test_status": "not_run",
                },
                "behavior_breaking": {
                    "parse_status": "pass",
                    "compile_status": "pass",
                    "test_status": "fail",
                },
            },
            "cases": MAX_CHECKS,
            "timeout_seconds_per_check": 10,
            "maximum_wall_seconds": MAX_WALL_SECONDS,
        },
        "review_protocol": {
            "model_input_fields": ["candidate_id", "seed_id", "state", "prompt", "context_sha256"],
            "answer_oracle_provenance_in_model_input": False,
            "provider_calls": 0,
            "model_runs": 0,
            "training_tokens": 0,
            "accepted_training": 0,
            "objective_verified_before_execution": False,
            "inferability_reviewed": False,
            "quality_evidence": False,
        },
        "output_directory": str(OUTPUT_DIR),
        "builder_sha256": _sha(Path(__file__).read_bytes()),
        "evaluator_sha256": _sha(Path(evaluate_prediction.__code__.co_filename).read_bytes()),
        "contract_sha256": _sha((ROOT / "src/tinycomplete/one_line/contract.py").read_bytes()),
    }


def freeze_plan() -> dict[str, Any]:
    if PLAN_PATH.exists() or OUTPUT_DIR.exists():
        raise ObjectivePacketError("objective_plan_or_output_already_exists")
    plan = build_plan()
    plan["plan_identity_sha256"] = _canonical_sha(plan)
    payload = (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(PLAN_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    PLAN_PATH.chmod(0o600)
    return {"status": "frozen", "plan_raw_sha256": _sha(payload), **plan}


def _evaluate(
    *,
    candidate_id: str,
    variant: str,
    code: str,
    fixture_source: str,
    run_context: RunContext,
    case_dir: Path,
) -> dict[str, Any]:
    case_id = f"{candidate_id}/{variant}"
    request_id = uuid.uuid4().hex
    attempt_id = uuid.uuid4().hex
    check = CheckSpec(
        compile=["python", "-m", "py_compile", "solution.py"],
        test=["python", "oracle.py"],
        files={"oracle.py": fixture_source},
        timeout_seconds=10,
        container_image=PYTHON_IMAGE,
    )
    case = BenchmarkCase(
        id=case_id,
        language="python",
        path="solution.py",
        prefix="",
        expected="objective fixture only",
        check=check,
        category="synthetic-public-source-function-oracle",
        repository_context=False,
    )
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(case_dir, 0o700)
    started = time.monotonic()
    with run_context.for_case(case_id, request_id=request_id).activate():
        result = evaluate_prediction(
            case,
            Prediction(case_id=case_id, completion=code),
            work_root=case_dir,
            execution_backend="container",
        )
    checks = {
        "parse": {
            "status": result.parse.status,
            "returncode": result.parse.returncode,
            "elapsed_seconds": result.parse.seconds,
            "stdout": result.parse.stdout,
            "stderr": result.parse.stderr,
        },
        "compile": {
            "status": result.compile.status,
            "returncode": result.compile.returncode,
            "elapsed_seconds": result.compile.seconds,
            "stdout": result.compile.stdout,
            "stderr": result.compile.stderr,
        },
        "test": {
            "status": result.test.status,
            "returncode": result.test.returncode,
            "elapsed_seconds": result.test.seconds,
            "stdout": result.test.stdout,
            "stderr": result.test.stderr,
        },
    }
    return {
        "candidate_id": candidate_id,
        "variant": variant,
        "case_id": case_id,
        "case_attempt_id": attempt_id,
        "request_id": request_id,
        "parse_status": result.parse.status,
        "compile_status": result.compile.status,
        "test_status": result.test.status,
        "compile_configured": True,
        "test_configured": True,
        "working_tree_sha256": result.working_tree_sha256,
        "elapsed_seconds": time.monotonic() - started,
        "execution_backend": "container",
        "network_access": "none",
        "checks": checks,
    }


def _copy_artifact(source: Path, destination: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    resolved = source.resolve(strict=True)
    body = resolved.read_bytes()
    if len(body) != descriptor["bytes"] or _sha(body) != descriptor["sha256"]:
        raise ObjectivePacketError("source_license_artifact_changed")
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != body:
            raise ObjectivePacketError("packaged_artifact_collision")
    else:
        _write_private(destination, body)
    return {"path": destination.relative_to(OUTPUT_DIR).as_posix(), **descriptor}


def _package_case_artifacts(case: dict[str, Any]) -> dict[str, Any]:
    result = {}
    names = {
        "source_artifact": "source",
        "path_license_artifact": "path_license",
        "root_license_artifact": "root_license",
        "license_scope_artifact": "license_scope",
    }
    for descriptor_name, slug in names.items():
        descriptor = case["artifact_descriptors"][descriptor_name]
        src = PACKAGE_ROOT / descriptor["path"]
        suffix = (
            ".src"
            if descriptor_name == "source_artifact"
            else ".json"
            if "license_scope" in descriptor_name or "path_license" in descriptor_name
            else ".txt"
        )
        destination = OUTPUT_DIR / "artifacts" / slug / f"{descriptor['sha256']}{suffix}"
        ref = _copy_artifact(src, destination, descriptor)
        result[descriptor_name] = ref
    fixture_destination = OUTPUT_DIR / case["fixture"]["path"]
    fixture_descriptor = {
        "sha256": case["fixture"]["sha256"],
        "bytes": case["fixture"]["bytes"],
    }
    fixture_ref = _copy_artifact(
        OUTPUT_DIR / ".staged_fixtures" / f"{_objective_slug(case['candidate_id'])}.py",
        fixture_destination,
        fixture_descriptor,
    )
    result["fixture"] = fixture_ref
    return result


def execute() -> dict[str, Any]:
    raw_plan = PLAN_PATH.read_bytes()
    frozen_plan = json.loads(raw_plan)
    plan = dict(frozen_plan)
    identity = plan.pop("plan_identity_sha256", None)
    if identity != _canonical_sha(plan) or build_plan() != plan:
        raise ObjectivePacketError("frozen_objective_plan_or_inputs_changed")
    if OUTPUT_DIR.exists():
        raise ObjectivePacketError("objective_output_already_exists")
    cases, _tokenizer_identity = _build_cases()
    by_id = {case["candidate_id"]: case for case in cases}
    OUTPUT_DIR.mkdir(mode=0o700, parents=True)
    os.chmod(OUTPUT_DIR, 0o700)
    staged_fixtures = OUTPUT_DIR / ".staged_fixtures"
    staged_fixtures.mkdir(mode=0o700)
    for case in cases:
        _write_private(
            staged_fixtures / f"{_objective_slug(case['candidate_id'])}.py",
            case["fixture_source"].encode("utf-8"),
        )
    for case in cases:
        case["package_artifacts"] = _package_case_artifacts(case)
    shutil.rmtree(staged_fixtures)

    model_inputs = []
    answer_rows = []
    provenance_rows = []
    fixture_rows = []
    specs_by_id = OBJECTIVES
    input_by_id, provenance_by_id, _tree_by_id, _expansion_by_id = _rows()
    for candidate_id in OBJECTIVE_IDS:
        case = by_id[candidate_id]
        source_input = input_by_id[candidate_id]
        model_input = {
            key: source_input[key]
            for key in ("candidate_id", "seed_id", "state", "prompt", "context_sha256")
        }
        model_inputs.append(model_input)
        action = case["action_object"]
        provenance_row = provenance_by_id[candidate_id]
        answer_rows.append(
            {
                "candidate_id": candidate_id,
                "seed_id": candidate_id,
                "split": "train",
                "source_type": "synthetic_public_source_task",
                "action": action,
                "action_sha256": case["gold_action_sha256"],
                "state_sha256": case["state_sha256"],
                "context_sha256": case["context_sha256"],
                "history_before_sha256": case["history_before_sha256"],
                "after_source_sha256": case["after_source_sha256"],
                "human_chronology_observed": False,
                "accepted_training": False,
                "inferability_reviewed": False,
                "objective_verified": False,
                "wrong_action_controls_rejected": False,
                "history_leakage_check": True,
                "input_tokens": case["input_tokens"],
                "response_tokens_including_eos": case["response_tokens_including_eos"],
            }
        )
        authoring_metadata = {
            "license_scope_candidate_id": provenance_row["license_scope_candidate_id"],
            "source_repo": provenance_row["source_repo"],
            "source_revision": provenance_row["source_revision"],
            "parent_commit": case["parent_commit"],
            "source_tree_sha": case["source_tree_sha"],
            "source_path": provenance_row["source_path"],
            "source_sha256": provenance_row["source_artifact_sha256"],
            "source_group_id": provenance_row["source_group_id"],
            "session_or_commit": provenance_row["session_or_commit"],
            "task_family_id": provenance_row["task_family_id"],
            "template_id": provenance_row["template_id"],
            "near_duplicate_key": provenance_row["near_duplicate_key"],
            "license_spdx": provenance_row["license_spdx"],
            "human_chronology_observed": False,
            "history_origin": "synthetic_editor_typing",
            "visible_request_location": "state.source@cursor",
            "visible_intent_basis": case["visible_intent_basis"],
            "transform": case["transform"],
            "transform_sha256": case["transform_sha256"],
        }
        for source_key, out_key in (
            ("source_artifact", "source_artifact"),
            ("path_license_artifact", "path_license_artifact"),
            ("root_license_artifact", "root_license_artifact"),
            ("license_scope_artifact", "license_scope_artifact"),
        ):
            ref = case["package_artifacts"][source_key]
            authoring_metadata[out_key + "_path"] = ref["path"]
            authoring_metadata[out_key + "_sha256"] = ref["sha256"]
            authoring_metadata[out_key + "_bytes"] = ref["bytes"]
        provenance_rows.append(
            {
                "candidate_id": candidate_id,
                "seed_id": candidate_id,
                "split": "train",
                "source_type": "synthetic_public_source_task",
                "source_group_id": provenance_row["source_group_id"],
                "state_sha256": case["state_sha256"],
                "action_sha256": case["gold_action_sha256"],
                "context_sha256": case["context_sha256"],
                "history_before_sha256": case["history_before_sha256"],
                "after_source_sha256": case["after_source_sha256"],
                "accepted_training": False,
                "inferability_reviewed": False,
                "objective_verified": False,
                "wrong_action_controls_rejected": False,
                "history_leakage_check": True,
                "authoring_metadata": authoring_metadata,
            }
        )
        fixture_rows.append(
            {
                "candidate_id": candidate_id,
                "seed_id": candidate_id,
                "state_sha256": case["state_sha256"],
                "action_sha256": case["gold_action_sha256"],
                "context_sha256": case["context_sha256"],
                "fixture_path": case["package_artifacts"]["fixture"]["path"],
                "fixture_sha256": case["fixture"]["sha256"],
                "fixture_bytes": case["fixture"]["bytes"],
                "contract": case["objective_contract"],
                "wrong_action": case["wrong_action_object"],
            }
        )

    preflight_files = {
        "plan.json": raw_plan,
        "source_only_inputs.jsonl": b"".join(_canonical(row) + b"\n" for row in model_inputs),
        "answers_private.jsonl": b"".join(_canonical(row) + b"\n" for row in answer_rows),
        "provenance_private.jsonl": b"".join(_canonical(row) + b"\n" for row in provenance_rows),
        "fixture_bindings_private.jsonl": b"".join(_canonical(row) + b"\n" for row in fixture_rows),
    }
    for name, payload in preflight_files.items():
        _write_private(OUTPUT_DIR / name, payload)
    diagnostics_path = OUTPUT_DIR / "oracle_diagnostics.jsonl"
    oracle_path = OUTPUT_DIR / "oracle_results.jsonl"
    execution_state_path = OUTPUT_DIR / "execution_state.json"
    _write_private(diagnostics_path, b"")
    _write_private(oracle_path, b"")
    completed_ids: list[str] = []
    _write_execution_state(
        execution_state_path,
        {"status": "running", "completed_candidate_ids": completed_ids, "diagnostic_rows": 0},
    )

    offline_path = OUTPUT_DIR / "observability.jsonl"
    os.environ["TABCOMPLETE_OBSERVABILITY_ENABLED"] = "1"
    os.environ["TABCOMPLETE_OBSERVABILITY_MODE"] = "offline"
    os.environ["TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE"] = str(offline_path)
    os.environ["TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT"] = "0"
    run_context = RunContext.new(campaign_id="python-prefix-objective-oracle-v2")
    run_id = run_context.run_id
    diagnostics: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    diagnostics_count = 0
    work_root = Path(tempfile.mkdtemp(prefix="tabcomplete-objective-v2-"))
    os.chmod(work_root, 0o700)
    start = time.monotonic()
    for candidate_id in OBJECTIVE_IDS:
        case = by_id[candidate_id]
        state = EditState.from_mapping(case["state_object"])
        gold = EditAction(**case["action_object"])
        wrong = EditAction(**case["wrong_action_object"])
        expected_actions = {
            "gold": gold,
            "before": EditAction("keep"),
            "behavior_breaking": wrong,
        }
        variant_results = {}
        for variant, action in expected_actions.items():
            if variant == "before":
                code = state.source
            else:
                after = apply_action(state, action)
                code, extraction = _extract_function(after, state.target_row)
                if variant == "gold" and _canonical_sha(asdict(gold)) != case["gold_action_sha256"]:
                    raise ObjectivePacketError("gold_action_hash_changed")
                if variant == "behavior_breaking" and action == gold:
                    raise ObjectivePacketError("behavior_control_matches_gold")
            diag = _evaluate(
                candidate_id=candidate_id,
                variant=variant,
                code=code,
                fixture_source=specs_by_id[candidate_id]["test"].lstrip("\n"),
                run_context=run_context,
                case_dir=work_root / _objective_slug(candidate_id) / variant,
            )
            diagnostics.append(diag)
            diagnostics_count += 1
            _append_private_jsonl(diagnostics_path, diag)
            variant_results[variant] = {
                "action": asdict(action),
                "action_sha256": _canonical_sha(asdict(action)),
                "after_source_sha256": _sha(
                    (state.source if variant == "before" else apply_action(state, action)).encode(
                        "utf-8"
                    )
                ),
                "functional_expected": "pass" if variant == "gold" else "fail",
                "functional_status": (
                    "pass"
                    if variant == "gold"
                    and diag["parse_status"] == "pass"
                    and diag["compile_status"] == "pass"
                    and diag["test_status"] == "pass"
                    else "fail"
                    if variant == "behavior_breaking"
                    and diag["parse_status"] == "pass"
                    and diag["compile_status"] == "pass"
                    and diag["test_status"] == "fail"
                    else "invalid_pre_state"
                    if variant == "before" and diag["parse_status"] == "fail"
                    else "unverified"
                ),
                "parse_status": diag["parse_status"],
                "compile_status": diag["compile_status"],
                "test_status": diag["test_status"],
                "diagnostic_sha256": _canonical_sha(diag),
            }
        case_result = {
            "candidate_id": candidate_id,
            "seed_id": candidate_id,
            "split": "train",
            "source_type": "synthetic_public_source_task",
            "source_group_id": case["source_group_id"],
            "state_sha256": case["state_sha256"],
            "action_sha256": case["gold_action_sha256"],
            "context_sha256": case["context_sha256"],
            "history_before_sha256": case["history_before_sha256"],
            "fixture_sha256": case["fixture"]["sha256"],
            "fixture_artifact_path": case["fixture"]["path"],
            "fixture_artifact_bytes": case["fixture"]["bytes"],
            "evaluator_sha256": plan["evaluator_sha256"],
            "runtime_sha256": case["runtime_sha256"],
            "backend": "container",
            "network": "none",
            "variants": variant_results,
            "objective_verified": all(
                variant_results[name]["functional_status"] == expected
                for name, expected in (
                    ("gold", "pass"),
                    ("before", "invalid_pre_state"),
                    ("behavior_breaking", "fail"),
                )
            ),
            "accepted_training": False,
            "inferability_reviewed": False,
        }
        oracle_rows.append(case_result)
        _append_private_jsonl(oracle_path, case_result)
        completed_ids.append(candidate_id)
        _write_execution_state(
            execution_state_path,
            {
                "status": "running",
                "completed_candidate_ids": completed_ids,
                "diagnostic_rows": diagnostics_count,
            },
        )
        if time.monotonic() - start > MAX_WALL_SECONDS:
            raise ObjectivePacketError("frozen_oracle_wall_limit_exceeded")
    shutil.rmtree(work_root)
    _write_execution_state(
        execution_state_path,
        {
            "status": "complete",
            "completed_candidate_ids": completed_ids,
            "diagnostic_rows": diagnostics_count,
        },
    )

    # Only student-visible fields enter this file.
    if offline_path.exists():
        os.chmod(offline_path, 0o600)
    else:
        _write_private(offline_path, b"")
    source_tree_bytes = (TREE_OUTPUT / "parent_trees.jsonl").read_bytes()
    _write_private(OUTPUT_DIR / "artifacts/provenance/parent_trees.jsonl", source_tree_bytes)
    manifest = {
        "schema": "python-prefix-objective-oracle-package-v2",
        "status": "complete",
        "plan_raw_sha256": _sha(raw_plan),
        "plan_identity_sha256": identity,
        "artifact_root": ".",
        "candidate_count": len(oracle_rows),
        "candidate_split_counts": {"train": len(oracle_rows)},
        "source_group_count": len({row["source_group_id"] for row in oracle_rows}),
        "action_counts": {"replace_line": len(oracle_rows)},
        "all_source_type": "synthetic_public_source_task",
        "human_chronology_observed": False,
        "accepted_training": 0,
        "objective_verified_count": sum(bool(row["objective_verified"]) for row in oracle_rows),
        "inferability_reviewed": False,
        "quality_evidence": False,
        "provider_calls": 0,
        "model_runs": 0,
        "training_tokens": 0,
        "gpu_hours": 0,
        "wall_seconds": time.monotonic() - start,
        "execution_backend": "container",
        "network_access": "none",
        "campaign_id": run_context.campaign_id,
        "run_id": run_id,
        "run_attempt_id": run_context.run_attempt_id,
        "runtime_sha256": PYTHON_RUNTIME_SHA256,
        "evaluator_sha256": plan["evaluator_sha256"],
        "files": {},
        "parent_tree_artifact": {
            "path": "artifacts/provenance/parent_trees.jsonl",
            "bytes": len(source_tree_bytes),
            "sha256": _sha(source_tree_bytes),
        },
        "diagnostic_counts": {
            "gold_pass": sum(
                row["variants"]["gold"]["functional_status"] == "pass" for row in oracle_rows
            ),
            "before_invalid": sum(
                row["variants"]["before"]["functional_status"] == "invalid_pre_state"
                for row in oracle_rows
            ),
            "behavior_breaking_fail": sum(
                row["variants"]["behavior_breaking"]["functional_status"] == "fail"
                for row in oracle_rows
            ),
        },
    }
    manifest["files"] = {
        path.relative_to(OUTPUT_DIR).as_posix(): {
            "path": path.relative_to(OUTPUT_DIR).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path.read_bytes()),
        }
        for path in sorted(OUTPUT_DIR.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if sum(item["bytes"] for item in manifest["files"].values()) > MAX_OUTPUT_BYTES:
        raise ObjectivePacketError("objective_package_exceeds_storage_cap")
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    _write_private(OUTPUT_DIR / "manifest.json", manifest_bytes)
    manifest["manifest_sha256"] = _sha(manifest_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = freeze_plan() if args.freeze_plan else execute()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
