#!/usr/bin/env python3
"""Freeze and audit exact-parent license scope for a small public-source seed set.

This is CPU-only preparation. It does not read child source, call a model or
provider, execute candidate code, or create training rows. Public source and
license bodies remain in the existing private SSD artifact tree.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path("/mnt/ssd/tabcomplete-product-r2/commitpackft/source-verification-v1")
PRIVATE_DIR = Path("/mnt/ssd/tabcomplete-product-r2/public-source-oracle-seeds-r1")
SPEC_PATH = PRIVATE_DIR / "seed_spec_v1.json"
PLAN_PATH_V1 = PRIVATE_DIR / "plan_v1.json"
PLAN_PATH = PRIVATE_DIR / "plan_v2.json"
SCOPE_PATH = PRIVATE_DIR / "license_scope_v2.json"
LICENSE_DIR = PRIVATE_DIR / "license_blobs"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
SOURCE_ROWS = SOURCE_DIR / "candidate_results.jsonl"
SOURCE_MANIFEST_SHA = "3b3f5de211e8836b4e7fed067af7f794c51182f746267d6f4da09c4c8bbc1b0e"
SOURCE_ROWS_SHA = "fe993904622cb44090c07c42775d621ea82050959435c6004b85e720e2e1efd9"
TOKENIZER = Path(
    "/home/crabcake/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B"
    "/snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301/tokenizer.json"
)

SCHEMA = "public-source-oracle-seed-plan-v2"
SPEC_SCHEMA = "public-source-oracle-seed-spec-v1"
SCOPE_SCHEMA = "exact-parent-license-scope-v2"
MAX_ROWS = 8
MAX_NETWORK_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_WALL_SECONDS = 20 * 60
MAX_PARENT_BYTES = 16 * 1024
MAX_CONTEXT_TOKENS = 1024
MAX_ACTION_TOKENS = 64
ALLOWED_ACTIONS = {"keep", "replace_line", "insert_before", "delete_line"}
ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"}
SANDBOX_IMAGES = {
    "go": (
        "docker.io/library/golang:1.24-bookworm@sha256:"
        "1a6d4452c65dea36aac2e2d606b01b4a029ec90cc1ae53890540ce6173ea77ac"
    ),
    "python": (
        "docker.io/library/python:3.12-slim@sha256:"
        "2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9"
    ),
    "rust": (
        "docker.io/library/rust:1.85-slim@sha256:"
        "9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36"
    ),
    "typescript": (
        "localhost/tabcomplete-typescript-bench:5.9.2@sha256:"
        "3cc808896e2be1342f6fc82b12be5f6371b9b13b6e70ef92489ebaa34eff5b73"
    ),
}
LICENSE_NAME = re.compile(r"^(?:licen[cs]e|copying|unlicense)(?:[._ -].*)?$", re.IGNORECASE)
SPDX = re.compile(rb"SPDX-License-Identifier\s*:\s*([^\r\n]+)", re.IGNORECASE)
SPDX_SIMPLE = re.compile(r"^[a-z0-9.+-]+$")
NOTICE_OR_COPYRIGHT = re.compile(r"^(?:notice|copyright)(?:[._ -].*)?$", re.IGNORECASE)

# These identities are the only rows this preparation tool may inspect.
# Values are taken from source-verification-v1's verified parent snapshots.
# No child revision or child blob identity is carried into this plan.
PARENT_IDENTITIES: dict[str, dict[str, Any]] = {
    "commit-sequence/68a81d31e1ad9a6f66d642b3": {
        "source_group_id": "a43b333dc7e00e024306d533a2066d7376cdc193b4072067e33f8fd9e18eb956",
        "repository": "cassava/repoctl",
        "parent_commit": "06351ed65a23d9dfd59344d2a8fbf10e56e43069",
        "parent_tree_sha": "ecf85ac3b5e4885ddb6c7dded524695185234983",
        "file_path": "alpm/format.go",
        "parent_source_sha256": "a8e43df0db3bba191a2916568ad68a0483c9830d3b964dea6a20329077a45ca4",
        "parent_private_path": (
            "source_pairs/b3f19c65ac9a612fa2d4c29cc4daa348de6b359df0a000c1a8bb4b3ff0e8ebc2/"
            "parent-a8e43df0db3bba191a2916568ad68a0483c9830d3b964dea6a20329077a45ca4.src"
        ),
        "root_license_path": "LICENSE",
        "root_license_git_blob_sha": "530058f891f2bc17581b35c49890260f0fbd761e",
        "root_license_sha256": "80d871c1f63b3c729f72e21374401b871c47ab951613b1dc6c7793361b1186f1",
        "root_spdx": ["mit"],
    },
    "commit-sequence/2fc499c2c241a0e4c93eaf56": {
        "source_group_id": "6689ff0aecf8fa7ebf8526a4abfb7fc20b832656af29113a0ab1f456e08d927f",
        "repository": "Becklyn/mojave",
        "parent_commit": "e9995a6d3e834534ebed81ec4f60256f4445a9cb",
        "parent_tree_sha": "fa979525b62c137d07496c43833c0c469f910fd0",
        "file_path": "json.ts",
        "parent_source_sha256": "3827f6ea060dbf48b32394e078736097d3b4c4a1d3ba4eaec9aad4a71d1021e0",
        "parent_private_path": (
            "source_pairs/ad3ec8ff1e301f006c13b3d446e803377c91e3e407f609a7036ac3bf19e08300/"
            "parent-3827f6ea060dbf48b32394e078736097d3b4c4a1d3ba4eaec9aad4a71d1021e0.src"
        ),
        "root_license_path": "LICENSE",
        "root_license_git_blob_sha": "a5184d48640868286bbbad07020601648a969477",
        "root_license_sha256": "cd81479c1bc8fdc8a4e1977bd24e7102d6b705dd6a4c3121a54a9bd2ae46fccd",
        "root_spdx": ["bsd-3-clause"],
    },
    "commit-sequence/a6eb3efc9d322f7f59ec69fa": {
        "source_group_id": "5b1d2c08cb282ecc63ab6d9c49dd690668269a8660467a59d8fea0c56a629aa4",
        "repository": "omise/omise-go",
        "parent_commit": "e98032d4c2e5b2a12e75f0cf8523927909599aa2",
        "parent_tree_sha": "117ee0575ac1280b384f1b121c8e2fb7bfa3f1c5",
        "file_path": "operations/list.go",
        "parent_source_sha256": "c3bf18fa2ac6383c2b44ba42155f2fff15ea89934c64f95e0192a7843d1f5848",
        "parent_private_path": (
            "source_pairs/f744dabb21bace02d44889094c265f0311d1e5676c5969bc77643a02d064ab22/"
            "parent-c3bf18fa2ac6383c2b44ba42155f2fff15ea89934c64f95e0192a7843d1f5848.src"
        ),
        "root_license_path": "LICENSE",
        "root_license_git_blob_sha": "e0ccc701cdcb2d87d268dedafc1c140d9adc3c03",
        "root_license_sha256": "5a9a79f0e8b686cb1346c7203038c22518e4c76e30784b9429089762bd0737dc",
        "root_spdx": ["mit"],
    },
    "commit-sequence/b1709088ff3b111101fd24a1": {
        "source_group_id": "935d93542b94460e80c531d27bfb8976ee150db47a01fb35642859fe9b5be32d",
        "repository": "DestinyItemManager/DIM",
        "parent_commit": "39a6c005ddded2083777f83bccdc3d21772c3447",
        "parent_tree_sha": "93c5e85a0b21d3369914f4bdc9d61b495a4d6c3d",
        "file_path": "src/app/inventory/store/item-index.ts",
        "parent_source_sha256": "bd0006c6d989821da59bd197de2e28f09341c158c3abfc40bfa6c61be3b4c9b8",
        "parent_private_path": (
            "source_pairs/c8835cab05eb351f00b77096fce096efe9922f1b5b009d486a1c360c88a4f137/"
            "parent-bd0006c6d989821da59bd197de2e28f09341c158c3abfc40bfa6c61be3b4c9b8.src"
        ),
        "root_license_path": "LICENSE.md",
        "root_license_git_blob_sha": "10494a5b38cf84d38253b0f138361d4266399a76",
        "root_license_sha256": "174146cd88f80d8a3e9f0c3ded3af1f65757a534e3a397709d169463a95dbb5c",
        "root_spdx": ["mit"],
    },
    "commit-sequence/ecec19afc8be49798907c2c6": {
        "source_group_id": "d129b35ddbcb3607c931ad5f2e07e6e1d421bf711f7ea48a720b207388ad2ae9",
        "repository": "nodeswork/sbase",
        "parent_commit": "fd8cfc7dffbcdaa11ff2dc2c61ec2b62df660d8b",
        "parent_tree_sha": "ab22bd7a681f6fcc4521fd27cf7612e0af7f1e8d",
        "file_path": "src/koa/utils.ts",
        "parent_source_sha256": "5396e15eee01cee0d343d3b85b04f78a41cac45430ef58788c9045fc5d5619a7",
        "parent_private_path": (
            "source_pairs/2efaed92af305fdf9a78ef8f344df93009421f4997f7ea1c9844713207133180/"
            "parent-5396e15eee01cee0d343d3b85b04f78a41cac45430ef58788c9045fc5d5619a7.src"
        ),
        "root_license_path": "LICENSE",
        "root_license_git_blob_sha": "8dada3edaf50dbc082c9a125058f25def75e625a",
        "root_license_sha256": "b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1",
        "root_spdx": ["apache-2.0"],
    },
    "commit-sequence/4c6f19d20f253d59172465b6": {
        "source_group_id": "8e99bbe7861814812f62a4da79cc8828fd9765aa1e70f3052e97df8e8e6f2fa4",
        "repository": "jonathanslenders/python-prompt-toolkit",
        "parent_commit": "073f0ee9c73111502e8eb6fb692802dd41855dbe",
        "parent_tree_sha": "59148ca1c128d0ca19871bf5ab7e791e6e359597",
        "file_path": "prompt_toolkit/filters/utils.py",
        "parent_source_sha256": "85a851d0ad626bada3b64e347769971585e76e7e72aaa9f985a26d057f4d7a0d",
        "parent_private_path": (
            "source_pairs/b4d19d925e3bad52d9ffbb6d8aa2944f34518dbf11ce6823412fcdcfa5ca9e88/"
            "parent-85a851d0ad626bada3b64e347769971585e76e7e72aaa9f985a26d057f4d7a0d.src"
        ),
        "root_license_path": "LICENSE",
        "root_license_git_blob_sha": "e1720e0fb70684043842e94ede48622e6bffc62d",
        "root_license_sha256": "303574d9bdd85c757d6025017942bf17baeedf2778f62bd7f425d07d880f4c4a",
        "root_spdx": ["bsd-3-clause"],
    },
    "commit-sequence/dc7b6133ce548947d9d120fb": {
        "source_group_id": "5f19cfded5ae5065aee7157c9034f79df910baa985284ba7c3047d2ee10b19c0",
        "repository": "bbrks/wrap",
        "parent_commit": "7b7958ba4830d201c1b31275f449d3d03d77b47a",
        "parent_tree_sha": "84b33c6b88dbbb68657a7fc8baa5b0711786b653",
        "file_path": "wrap.go",
        "parent_source_sha256": "f804d603542627e94a5ea124800915f544696da75dadf73c40850dd9671e39fd",
        "parent_private_path": (
            "source_pairs/6e55e30ba48ad360547759cf435c09860846a9a9a1aacc355eb74b8c527e7fa1/"
            "parent-f804d603542627e94a5ea124800915f544696da75dadf73c40850dd9671e39fd.src"
        ),
        "root_license_path": "LICENSE",
        "root_license_git_blob_sha": "b3ff1184918688eace4fb0da087922de3ef41c10",
        "root_license_sha256": "f6ad3257edb2bed711971e0e4ed91d5e7784fa6c1ff5af92cc308c5c019e8b86",
        "root_spdx": ["mit"],
    },
    "commit-sequence/f23a85517771485aea1ae128": {
        "source_group_id": "70bb50cfe9e52c778ef61a14ec4e399516a2b9ac59b02cc44cf93486326a94cc",
        "repository": "mneumann/lindenmayer-system",
        "parent_commit": "ecd16757238cc0b204b6a5136f89eb1be3e73f7e",
        "parent_tree_sha": "aab65834fd5a58368d5c6147ee7f217e8746588e",
        "file_path": "src/lib.rs",
        "parent_source_sha256": "b8fc9504f28aa84d0fe5e0b253bed4b318feb29b14aea0e46648381b75d43523",
        "parent_private_path": (
            "source_pairs/fcebe93b099efad10479c8ea132bff771c80bec8ede2cac497d753e3a0e18fec/"
            "parent-b8fc9504f28aa84d0fe5e0b253bed4b318feb29b14aea0e46648381b75d43523.src"
        ),
        "root_license_path": "LICENSE",
        "root_license_git_blob_sha": "bd92d22e4fcce9f01798354954381e99e2428f85",
        "root_license_sha256": "2ed77474952bd007d8892e1f7105f1c5f30970fb26b6312b28f1d987f7a54750",
        "root_spdx": ["mit"],
    },
}


class ApiClient(Protocol):
    """Minimal interface that also makes the path walk unit-testable."""

    bytes_received: int

    def json(self, repository: str, endpoint: str) -> dict[str, Any]: ...


class AuditError(ValueError):
    """Sanitized failure status; never includes source, credentials, or API body."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: dict[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _safe_relpath(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in value
        and "\x00" not in value
    )


def _load_verified_parent_rows(
    *, source_dir: Path, ids: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Read only pinned train-row parent fields; child fields are never consulted."""
    if sha256_file(source_dir / "manifest.json") != SOURCE_MANIFEST_SHA:
        raise AuditError("source_manifest_identity_changed")
    if sha256_file(source_dir / "candidate_results.jsonl") != SOURCE_ROWS_SHA:
        raise AuditError("candidate_index_identity_changed")
    wanted = ids or set(PARENT_IDENTITIES)
    if not wanted.issubset(PARENT_IDENTITIES):
        raise AuditError("candidate_not_allowlisted")
    found: dict[str, dict[str, Any]] = {}
    with (source_dir / "candidate_results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            # The file contains only sanitized metadata. Use the row's split and
            # parent identity; do not read child commit/license/source fields.
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            if candidate_id not in wanted:
                continue
            expected = PARENT_IDENTITIES[candidate_id]
            source_pair = row.get("source_pair")
            parent_license = row.get("parent_license")
            if not isinstance(source_pair, dict) or not isinstance(parent_license, dict):
                raise AuditError("parent_identity_missing")
            closest = parent_license.get("closest_license")
            root_license = parent_license.get("root_license")
            observed = {
                "candidate_id": candidate_id,
                "split": row.get("split"),
                "status": row.get("status"),
                "source_group_id": row.get("source_group_id"),
                "repository": row.get("repository"),
                "parent_commit": row.get("parent_commit"),
                "file_path": row.get("file_path"),
                "parent_source_sha256": source_pair.get("parent_sha256"),
                "parent_private_path": source_pair.get("parent_private_path"),
                "parent_tree_sha": parent_license.get("tree_sha"),
                "root_license_path": (
                    root_license.get("path") if isinstance(root_license, dict) else None
                ),
                "root_license_git_blob_sha": (
                    root_license.get("git_blob_sha") if isinstance(root_license, dict) else None
                ),
                "root_license_sha256": (
                    root_license.get("sha256") if isinstance(root_license, dict) else None
                ),
                "root_spdx": parent_license.get("root_spdx"),
                "closest_parent_license_path": (
                    closest.get("path") if isinstance(closest, dict) else None
                ),
                "closest_parent_license_scope": (
                    closest.get("root_or_closest") if isinstance(closest, dict) else None
                ),
            }
            for key, value in expected.items():
                if observed.get(key) != value:
                    raise AuditError("pinned_parent_field_mismatch")
            if (
                row.get("split") != "train"
                or row.get("status") != "source_and_license_verified_for_human_review"
            ):
                raise AuditError("candidate_not_verified_train")
            private_path = expected["parent_private_path"]
            if not _safe_relpath(private_path):
                raise AuditError("unsafe_parent_path")
            parent_path = source_dir / private_path
            if not parent_path.is_file() or parent_path.is_symlink():
                raise AuditError("parent_snapshot_missing")
            if parent_path.stat().st_size > MAX_PARENT_BYTES:
                raise AuditError("parent_snapshot_size_limit")
            if sha256_file(parent_path) != expected["parent_source_sha256"]:
                raise AuditError("parent_snapshot_hash_mismatch")
            if candidate_id in found:
                raise AuditError("duplicate_candidate_id")
            found[candidate_id] = observed
    if set(found) != wanted:
        raise AuditError("pinned_candidate_missing")
    return found


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema") != SPEC_SCHEMA or spec.get("revision") != 1:
        raise AuditError("spec_schema_mismatch")
    rows = spec.get("seeds")
    if not isinstance(rows, list) or len(rows) != MAX_ROWS:
        raise AuditError("seed_count_mismatch")
    seen_ids: set[str] = set()
    seen_seed_ids: set[str] = set()
    seen_groups: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise AuditError("seed_row_shape")
        candidate_id = row.get("candidate_id")
        if candidate_id not in PARENT_IDENTITIES or candidate_id in seen_ids:
            raise AuditError("candidate_id_not_unique_and_allowlisted")
        seen_ids.add(candidate_id)
        seed_id = row.get("seed_id")
        if not isinstance(seed_id, str) or not seed_id or seed_id in seen_seed_ids:
            raise AuditError("seed_id_not_unique")
        seen_seed_ids.add(seed_id)
        group = PARENT_IDENTITIES[candidate_id]["source_group_id"]
        if group in seen_groups:
            raise AuditError("parent_source_group_reused")
        seen_groups.add(group)
        if row.get("split") != "train":
            raise AuditError("nontraining_seed_forbidden")
        request = row.get("student_request")
        if not isinstance(request, str) or not request.strip() or len(request.encode()) > 1200:
            raise AuditError("student_request_invalid")
        if row.get("expected_action_kind") not in ALLOWED_ACTIONS:
            raise AuditError("canonical_action_kind_invalid")
        if not isinstance(row.get("execution_scope"), str) or not row["execution_scope"].strip():
            raise AuditError("execution_scope_missing")
        if not isinstance(row.get("objective_contract"), list) or not row["objective_contract"]:
            raise AuditError("objective_contract_missing")
        if not isinstance(row.get("control_contract"), list) or not row["control_contract"]:
            raise AuditError("semantic_control_contract_missing")
        if row.get("history_policy") != "empty_synthetic_history":
            raise AuditError("history_policy_invalid")
        if row.get("source_type") != "synthetic_public_source_task":
            raise AuditError("source_type_invalid")
        if row.get("human_chronology_observed") is not False:
            raise AuditError("synthetic_chronology_mislabelled")
        if row.get("whole_project_claim") is not False:
            raise AuditError("whole_project_claim_forbidden")
    if seen_ids != set(PARENT_IDENTITIES):
        raise AuditError("seed_identity_set_mismatch")
    forbidden = {"source", "after_source", "gold_action", "child_diff", "child_source"}
    if any(forbidden.intersection(row) for row in rows):
        raise AuditError("spec_contains_source_or_gold_body")


def make_plan(
    *,
    spec: dict[str, Any],
    source_dir: Path,
    script_path: Path,
    test_path: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    validate_spec(spec)
    rows = _load_verified_parent_rows(source_dir=source_dir)
    if not tokenizer_path.is_file():
        raise AuditError("local_tokenizer_missing")
    selected = []
    for seed in spec["seeds"]:
        identity = rows[seed["candidate_id"]]
        parent = PARENT_IDENTITIES[seed["candidate_id"]]
        selected.append(
            {
                "seed_id": seed["seed_id"],
                "candidate_id": seed["candidate_id"],
                "split": "train",
                "source_type": "synthetic_public_source_task",
                "source_group_id": identity["source_group_id"],
                "repository": identity["repository"],
                "parent_commit": identity["parent_commit"],
                "file_path": identity["file_path"],
                "parent_source_sha256": identity["parent_source_sha256"],
                "parent_source_bytes": (source_dir / parent["parent_private_path"]).stat().st_size,
                "root_license_path": identity["root_license_path"],
                "root_license_git_blob_sha": identity["root_license_git_blob_sha"],
                "root_license_sha256": identity["root_license_sha256"],
                "root_spdx": identity["root_spdx"],
                "student_request": seed["student_request"],
                "expected_action_kind": seed["expected_action_kind"],
                "objective_contract": seed["objective_contract"],
                "control_contract": seed["control_contract"],
                "history_policy": seed["history_policy"],
                "execution_scope": seed["execution_scope"],
                "whole_project_claim": False,
                "human_chronology_observed": False,
            }
        )
    return {
        "schema": SCHEMA,
        "revision": 2,
        "campaign": "public-source-oracle-seeds-r1",
        "input_artifacts": {
            "source_verification_manifest_sha256": sha256_file(source_dir / "manifest.json"),
            "source_verification_candidate_rows_sha256": sha256_file(
                source_dir / "candidate_results.jsonl"
            ),
            "seed_spec_sha256": sha256_bytes(
                json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ),
            "tokenizer_revision": (
                "Qwen/Qwen2.5-Coder-0.5B@8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
            ),
            "tokenizer_json_sha256": sha256_file(tokenizer_path),
            "contract_module_sha256": sha256_file(ROOT / "src/tinycomplete/one_line/contract.py"),
            "context_module_sha256": sha256_file(ROOT / "src/tinycomplete/one_line/context.py"),
            "training_encoder_sha256": sha256_file(ROOT / "src/tinycomplete/one_line/train.py"),
            "sandbox_evaluator_sha256": sha256_file(
                ROOT / "src/tinycomplete/eval/code_benchmark.py"
            ),
            "builder_sha256": sha256_file(script_path),
            "test_sha256": sha256_file(test_path),
        },
        "protocol": {
            "action_wire": "single-line-edit-v1",
            "action_ceiling_tokens_including_eos": MAX_ACTION_TOKENS,
            "input_context_policy": "single-line-context-v2",
            "input_context_token_ceiling": MAX_CONTEXT_TOKENS,
            "task_request_transport": (
                "A short synthetic request is stored as one relevant text item and serialized "
                "by the existing canonical context builder; it states behavior, not an action line."
            ),
            "history": (
                "only explicit synthetic or verified prior edits; byte-exact replay required"
            ),
            "code_execution": "pinned local SandboxEvaluator only; network=none; no host execution",
            "sandbox_images": SANDBOX_IMAGES,
            "tokenizer": "existing local Qwen2.5 tokenizer; local_files_only; no downloads",
        },
        "budgets": {
            "network_bytes_total": MAX_NETWORK_BYTES,
            "network_response_bytes": MAX_RESPONSE_BYTES,
            "wall_seconds": MAX_WALL_SECONDS,
            "parent_source_bytes_each": MAX_PARENT_BYTES,
            "selected_parent_rows": MAX_ROWS,
            "provider_calls": 0,
            "model_calls": 0,
            "gpu_hours": 0,
            "training_tokens": 0,
        },
        "selection_policy": {
            "source_rows": "the eight pinned train rows only",
            "splits": "preserve every source group as train; no dev/test reassignment",
            "label_origin": "explicit synthetic request plus independently executed objective",
            "child_source_or_diff_accessed": False,
            "source_gold_storage": "private SSD only",
            "root_spdx_and_path_scope_are_separate_fields": True,
            "unsupported_or_ambiguous": "skip; do not patch around the task contract",
            "quality_evidence": False,
            "training_ready": False,
        },
        "seeds": selected,
    }


def write_frozen_json(
    path: Path, value: dict[str, Any], *, identity_field: str = "artifact_identity_sha256"
) -> str:
    if path.exists() or path.is_symlink():
        raise AuditError("frozen_artifact_exists")
    body = dict(value)
    body[identity_field] = canonical_sha(value)
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(payload)


def load_frozen_plan(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    identity = value.pop("plan_identity_sha256", None)
    if identity != canonical_sha(value):
        raise AuditError("plan_identity_mismatch")
    return value, sha256_bytes(raw)


def _license_candidate(name: str, entry_type: str) -> bool:
    return entry_type == "blob" and bool(LICENSE_NAME.match(name))


def _spdx_evidence(data: bytes) -> dict[str, Any]:
    header = b"\n".join(data.splitlines()[:80])
    expressions = []
    ids = set()
    unparsed = False
    for match in SPDX.finditer(header):
        expression = match.group(1).strip().strip(b"*/# ").decode("ascii", "ignore").lower()
        expressions.append(expression)
        if SPDX_SIMPLE.fullmatch(expression):
            ids.add(expression)
        else:
            unparsed = True
    return {
        "present": bool(expressions),
        "expression_count": len(expressions),
        "ids": sorted(ids),
        "ambiguous": unparsed or len(expressions) > 1,
    }


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def path_directories(file_path: str) -> tuple[str, ...]:
    if not _safe_relpath(file_path):
        raise AuditError("unsafe_source_file_path")
    components = PurePosixPath(file_path).parts
    return ("",) + tuple("/".join(components[:depth]) for depth in range(1, len(components)))


def _decode_github_blob(record: dict[str, Any]) -> bytes:
    if record.get("encoding") != "base64" or not isinstance(record.get("content"), str):
        raise AuditError("license_blob_encoding")
    try:
        return base64.b64decode(record["content"], validate=False)
    except (ValueError, TypeError):
        raise AuditError("license_blob_encoding") from None


def audit_parent_license_scope(
    client: ApiClient,
    *,
    repository: str,
    parent_commit: str,
    expected_tree_sha: str,
    source_path: str,
    root_license_path: str,
    root_license_git_blob_sha: str,
    root_license_sha256: str,
    root_spdx: list[str],
    parent_source_bytes: bytes,
    private_license_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve the nearest license in exact parent-tree ancestor directories."""
    network_start = client.bytes_received
    commit = client.json(repository, f"git/commits/{parent_commit}")
    tree_record = commit.get("tree")
    tree_sha = tree_record.get("sha") if isinstance(tree_record, dict) else None
    if commit.get("sha") != parent_commit or tree_sha != expected_tree_sha:
        raise AuditError("parent_tree_identity_mismatch")
    directories = path_directories(source_path)
    current_tree = expected_tree_sha
    found: list[tuple[int, str, str]] = []
    additional_terms: list[dict[str, str]] = []
    components = PurePosixPath(source_path).parts
    final_entries: list[dict[str, Any]] = []
    reuse_trees: list[tuple[str, str]] = []
    for depth in range(len(directories)):
        tree = client.json(repository, f"git/trees/{current_tree}")
        if tree.get("truncated") is True:
            raise AuditError("parent_tree_truncated")
        entries = tree.get("tree")
        if not isinstance(entries, list):
            raise AuditError("parent_tree_shape")
        final_entries = [entry for entry in entries if isinstance(entry, dict)]
        current_directory = directories[depth]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_name = str(entry.get("path", ""))
            full = f"{current_directory}/{entry_name}" if current_directory else entry_name
            if _license_candidate(entry_name, str(entry.get("type", ""))):
                found.append((depth, full, str(entry.get("sha", ""))))
            elif str(entry.get("type", "")) == "blob" and NOTICE_OR_COPYRIGHT.match(entry_name):
                additional_terms.append({"path": full, "git_blob_sha": str(entry.get("sha", ""))})
            if entry_name == ".reuse" and entry.get("type") == "tree":
                reuse_path = f"{current_directory}/.reuse" if current_directory else ".reuse"
                reuse_trees.append((reuse_path, str(entry.get("sha", ""))))
        if depth + 1 < len(directories):
            component = components[depth]
            next_sha = next(
                (
                    str(entry.get("sha"))
                    for entry in entries
                    if isinstance(entry, dict)
                    and entry.get("path") == component
                    and entry.get("type") == "tree"
                ),
                None,
            )
            if not next_sha:
                raise AuditError("parent_source_directory_missing")
            current_tree = next_sha
    source_file_name = components[-1]
    source_entry = next(
        (
            entry
            for entry in final_entries
            if entry.get("path") == source_file_name and entry.get("type") == "blob"
        ),
        None,
    )
    if source_entry is None or source_entry.get("sha") != git_blob_sha1(parent_source_bytes):
        raise AuditError("parent_source_tree_identity_mismatch")
    reuse_dep5: list[dict[str, str]] = []
    for reuse_path, reuse_tree_sha in reuse_trees:
        reuse_tree = client.json(repository, f"git/trees/{reuse_tree_sha}")
        if reuse_tree.get("truncated") is True:
            raise AuditError("reuse_tree_truncated")
        reuse_entries = reuse_tree.get("tree")
        if not isinstance(reuse_entries, list):
            raise AuditError("reuse_tree_shape")
        dep5 = next(
            (
                entry
                for entry in reuse_entries
                if isinstance(entry, dict)
                and entry.get("path") == "dep5"
                and entry.get("type") == "blob"
            ),
            None,
        )
        if dep5 is not None:
            reuse_dep5.append(
                {
                    "path": f"{reuse_path}/dep5",
                    "git_blob_sha": str(dep5.get("sha", "")),
                }
            )
    if not found:
        raise AuditError("no_ancestor_license")
    deepest = max(depth for depth, _, _ in found)
    closest = sorted((path, blob) for depth, path, blob in found if depth == deepest)
    if len(closest) != 1:
        raise AuditError("ambiguous_closest_license")
    license_path, license_blob_sha = closest[0]
    if len(license_blob_sha) != 40 or not re.fullmatch(r"[0-9a-f]{40}", license_blob_sha):
        raise AuditError("license_blob_identity_invalid")
    blob_record = client.json(repository, f"git/blobs/{license_blob_sha}")
    license_bytes = _decode_github_blob(blob_record)
    if len(license_bytes) > MAX_RESPONSE_BYTES:
        raise AuditError("license_blob_size_limit")
    license_sha256 = sha256_bytes(license_bytes)
    if private_license_dir is not None:
        _write_private_blob(private_license_dir / f"{license_sha256}.txt", license_bytes)
    root_ids = sorted(set(item.lower() for item in root_spdx))
    path_evidence = _spdx_evidence(license_bytes)
    path_ids = path_evidence["ids"]
    if license_path == root_license_path:
        if license_blob_sha != root_license_git_blob_sha:
            raise AuditError("root_license_tree_identity_mismatch")
        if license_sha256 != root_license_sha256:
            raise AuditError("root_license_content_mismatch")
        if path_evidence["ambiguous"]:
            scope_source = "exact_parent_tree_root_license_ambiguous_spdx_expression"
        elif path_evidence["present"]:
            scope_source = "exact_parent_tree_root_license_spdx_header"
        else:
            path_ids = root_ids
            scope_source = "exact_parent_tree_root_license_plus_frozen_root_api_spdx"
    else:
        scope_source = "exact_parent_ancestor_license_spdx_header"
    header_evidence = _spdx_evidence(parent_source_bytes)
    header_ids = header_evidence["ids"]
    if additional_terms or reuse_dep5:
        status = "skip_additional_license_notice_requires_review"
    elif path_evidence["ambiguous"] or header_evidence["ambiguous"]:
        status = "skip_ambiguous_spdx_expression"
    elif path_evidence["present"] and path_ids != root_ids and license_path == root_license_path:
        status = "skip_root_license_spdx_mismatch"
    elif header_evidence["present"] and header_ids != path_ids:
        status = "skip_source_header_scope_mismatch"
    elif len(path_ids) != 1 or path_ids[0] not in ALLOWED_LICENSES:
        status = "skip_unclassified_or_multi_license_scope"
    elif path_ids != root_ids:
        status = "skip_path_scope_differs_from_root_spdx"
    else:
        status = "verified_path_scope"
    return {
        "schema": SCOPE_SCHEMA,
        "status": status,
        "repository": repository,
        "parent_commit": parent_commit,
        "source_path": source_path,
        "source_sha256": sha256_bytes(parent_source_bytes),
        "root_license": {
            "path": root_license_path,
            "git_blob_sha": root_license_git_blob_sha,
            "sha256": root_license_sha256,
            "root_spdx": root_ids,
        },
        "path_scope": {
            "scope": "root" if deepest == 0 else "closest_ancestor",
            "license_path": license_path,
            "git_blob_sha": license_blob_sha,
            "sha256": license_sha256,
            "spdx": path_ids,
            "source": scope_source,
            "spdx_expression_count": path_evidence["expression_count"],
            "spdx_ambiguous": path_evidence["ambiguous"],
        },
        "source_header_spdx": header_ids,
        "source_header_spdx_expression_count": header_evidence["expression_count"],
        "source_header_spdx_ambiguous": header_evidence["ambiguous"],
        "additional_license_references": additional_terms,
        "reuse_dep5_references": reuse_dep5,
        "network_bytes": client.bytes_received - network_start,
    }


def _write_private_blob(path: Path, value: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_file() and sha256_file(path) == sha256_bytes(value):
            return
        raise AuditError("private_license_blob_collision")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


class BoundedGhApi:
    """Credential-safe GitHub API reader with strict transfer and time caps."""

    def __init__(self, *, total_bytes: int = MAX_NETWORK_BYTES, timeout: float = MAX_WALL_SECONDS):
        self.total_cap = total_bytes
        self.response_cap = min(MAX_RESPONSE_BYTES, total_bytes)
        self.deadline = time.monotonic() + timeout
        self.bytes_received = 0
        self.cache: dict[tuple[str, str], dict[str, Any]] = {}

    def json(self, repository: str, endpoint: str) -> dict[str, Any]:
        cache_key = (repository, endpoint)
        if cache_key in self.cache:
            return self.cache[cache_key]
        remaining = self.total_cap - self.bytes_received
        if remaining <= 0 or time.monotonic() >= self.deadline:
            raise AuditError("network_or_wall_budget_exhausted")
        safe_repo = quote(repository, safe="/")
        args = ["gh", "api", "--hostname", "github.com", f"repos/{safe_repo}/{endpoint}"]
        env = dict(os.environ)
        env.update({"GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1"})
        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
        except OSError:
            raise AuditError("github_cli_unavailable") from None
        assert proc.stdout is not None and proc.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)
        chunks = {"stdout": bytearray(), "stderr": bytearray()}
        try:
            while selector.get_map():
                if time.monotonic() >= self.deadline:
                    raise AuditError("wall_budget_exhausted")
                ready = selector.select(min(0.2, self.deadline - time.monotonic()))
                if not ready and proc.poll() is not None:
                    ready = [
                        (selector_key, selectors.EVENT_READ)
                        for selector_key in selector.get_map().values()
                    ]
                for selector_key, _ in ready:
                    remaining = self.total_cap - self.bytes_received
                    if remaining <= 0:
                        raise AuditError("network_byte_budget_exhausted")
                    try:
                        block = os.read(selector_key.fd, min(64 * 1024, remaining))
                    except BlockingIOError:
                        continue
                    if not block:
                        selector.unregister(selector_key.fileobj)
                        continue
                    self.bytes_received += len(block)
                    output = chunks[selector_key.data]
                    output.extend(block)
                    if len(output) > self.response_cap:
                        raise AuditError("single_response_byte_limit")
            code = proc.wait(timeout=max(0.1, self.deadline - time.monotonic()))
        except (AuditError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise AuditError("network_request_failed_or_budget_exhausted") from None
        finally:
            selector.close()
            proc.stdout.close()
            proc.stderr.close()
        if code != 0:
            raise AuditError("github_api_request_failed")
        try:
            value = json.loads(chunks["stdout"])
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise AuditError("github_api_json_invalid") from None
        if not isinstance(value, dict):
            raise AuditError("github_api_json_shape_invalid")
        self.cache[cache_key] = value
        return value


def audit_plan(plan_path: Path = PLAN_PATH) -> dict[str, Any]:
    plan, plan_raw_sha = load_frozen_plan(plan_path)
    if plan.get("schema") != SCHEMA or plan.get("revision") != 2:
        raise AuditError("plan_schema_mismatch")
    source_rows = _load_verified_parent_rows(source_dir=SOURCE_DIR)
    api = BoundedGhApi()
    results = []
    for seed in plan["seeds"]:
        candidate_id = seed["candidate_id"]
        pinned = PARENT_IDENTITIES[candidate_id]
        parent_file = SOURCE_DIR / pinned["parent_private_path"]
        parent_bytes = parent_file.read_bytes()
        result = audit_parent_license_scope(
            api,
            repository=pinned["repository"],
            parent_commit=pinned["parent_commit"],
            expected_tree_sha=pinned["parent_tree_sha"],
            source_path=pinned["file_path"],
            root_license_path=pinned["root_license_path"],
            root_license_git_blob_sha=pinned["root_license_git_blob_sha"],
            root_license_sha256=pinned["root_license_sha256"],
            root_spdx=pinned["root_spdx"],
            parent_source_bytes=parent_bytes,
            private_license_dir=LICENSE_DIR,
        )
        result.update(
            {
                "seed_id": seed["seed_id"],
                "candidate_id": candidate_id,
                "source_group_id": source_rows[candidate_id]["source_group_id"],
            }
        )
        results.append(result)
    document = {
        "schema": SCOPE_SCHEMA,
        "revision": 2,
        "plan_raw_sha256": plan_raw_sha,
        "network_bytes_total": api.bytes_received,
        "network_cap_bytes": MAX_NETWORK_BYTES,
        "wall_cap_seconds": MAX_WALL_SECONDS,
        "results": results,
        "training_ready": False,
        "child_source_accessed": False,
        "provider_calls": 0,
        "model_calls": 0,
        "gpu_hours": 0,
    }
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--audit-parent-license-scope", action="store_true")
    parser.add_argument("--private-dir", type=Path, default=PRIVATE_DIR)
    args = parser.parse_args()
    if args.freeze_plan:
        private_dir = args.private_dir
        spec_path = private_dir / SPEC_PATH.name
        if not spec_path.is_file() or spec_path.is_symlink():
            raise SystemExit("private seed specification is unavailable")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        plan = make_plan(
            spec=spec,
            source_dir=SOURCE_DIR,
            script_path=Path(__file__).resolve(),
            test_path=ROOT / "tests/test_prepare_public_source_oracle_seeds.py",
            tokenizer_path=TOKENIZER,
        )
        path = private_dir / PLAN_PATH.name
        raw_sha = write_frozen_json(path, plan, identity_field="plan_identity_sha256")
        print(
            json.dumps(
                {"plan_path": str(path), "plan_raw_sha256": raw_sha, "seeds": len(plan["seeds"])}
            )
        )
        return
    audit = audit_plan(args.private_dir / PLAN_PATH.name)
    target = args.private_dir / SCOPE_PATH.name
    raw_sha = write_frozen_json(target, audit, identity_field="artifact_identity_sha256")
    print(
        json.dumps(
            {
                "scope_path": str(target),
                "scope_raw_sha256": raw_sha,
                "candidate_count": len(audit["results"]),
                "network_bytes": audit["network_bytes_total"],
            }
        )
    )


if __name__ == "__main__":
    main()
