"""Verify public Git source snapshots and license evidence for a frozen queue.

The command uses the user's existing ``gh`` authentication, stores raw public
source and license text only on a private SSD directory, and emits sanitized
qualification metadata. It never calls a model/provider or changes queue splits.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

from tinycomplete.one_line.data import _SENSITIVE_TEXT

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/prototype/product_r2"
QUEUE_PATH = Path(
    "/mnt/ssd/tabcomplete-product-r2/commitpackft/authoring-queue-v4/review_index.jsonl"
)
QUEUE_MANIFEST = QUEUE_PATH.parent / "manifest.json"
QUEUE_PLAN = REPORT / "commit_sequence_authoring_queue_plan_v4.json"
PLAN_PATH = REPORT / "commit_sequence_source_verification_plan_v2.json"
PRIOR_PLAN_PATH = REPORT / "commit_sequence_source_verification_plan_v1.json"
OUTPUT_DIR = Path("/mnt/ssd/tabcomplete-product-r2/commitpackft/source-verification-v1")
SCRIPT_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests/test_commit_sequence_source_verification.py"

SCHEMA = "commit-sequence-exact-source-verification-v2"
MAX_ROWS = 128
MAX_TRANSFER_BYTES = 20 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_WALL_SECONDS = 60 * 60
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"}
LICENSE_NAME = re.compile(r"^(?:licen[cs]e|copying|unlicense)(?:[._ -].*)?$", re.IGNORECASE)
SPDX = re.compile(r"SPDX-License-Identifier\s*:\s*([^\r\n]+)", re.IGNORECASE)
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class VerificationError(Exception):
    """A sanitized, machine-readable verification failure."""

    def __init__(self, status: str, *, stop: bool = False) -> None:
        super().__init__(status)
        self.status = status
        self.stop = stop


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _queue_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with QUEUE_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") not in {"train", "development"}:
                raise ValueError("queue contains a non-authorized split")
            rows.append(row)
    if len(rows) != MAX_ROWS or len({row["candidate_id"] for row in rows}) != MAX_ROWS:
        raise ValueError("frozen queue row count or identity changed")
    if Counter(row["split"] for row in rows) != Counter({"train": 96, "development": 32}):
        raise ValueError("frozen train/development split assignment changed")
    return rows


def make_plan() -> dict[str, Any]:
    queue_manifest = json.loads(QUEUE_MANIFEST.read_text(encoding="utf-8"))
    queue_plan = json.loads(QUEUE_PLAN.read_text(encoding="utf-8"))
    queue_digest = sha256_file(QUEUE_PATH)
    manifest_digest = sha256_file(QUEUE_MANIFEST)
    plan_digest = sha256_file(QUEUE_PLAN)
    if queue_digest != queue_manifest.get("private_index_sha256"):
        raise ValueError("frozen queue digest does not match its manifest")
    if plan_digest != queue_manifest.get("plan_file_sha256"):
        raise ValueError("frozen queue plan digest does not match its manifest")
    if queue_plan.get("plan_identity_sha256") != queue_manifest.get("plan_identity_sha256"):
        raise ValueError("frozen queue plan identity does not match its manifest")
    gh_version = (
        subprocess.run(["gh", "--version"], capture_output=True, check=True, timeout=10)
        .stdout.decode("utf-8", "replace")
        .splitlines()[0]
    )
    file_sha = lambda path: sha256_file(path)  # noqa: E731
    selected_rows = _queue_rows()
    selection_commitment = [
        {
            "candidate_id": row["candidate_id"],
            "source_group_id": row["source_group_id"],
            "split": row["split"],
            "repository": row["source_repo"],
            "child_commit": row["source_revision"],
            "file_path": row["file_path"],
            "expected_parent_file_sha256": row["source_before_sha256"],
            "expected_child_file_sha256": row["committed_child_file_sha256"],
            "dataset_license_claim": row["source_license_claim"],
        }
        for row in selected_rows
    ]
    return {
        "schema": SCHEMA,
        "revision": 2,
        "supersedes": {
            "plan_path": PRIOR_PLAN_PATH.name,
            "plan_raw_sha256": sha256_file(PRIOR_PLAN_PATH),
            "reason": (
                "v1 validation included its embedded self-identity in the frozen-input "
                "comparison; execution stopped before authentication, network access, or output"
            ),
        },
        "selector_plan_v4_raw_sha256": plan_digest,
        "selector_plan_v4_identity_sha256": queue_plan["plan_identity_sha256"],
        "queue_manifest_sha256": manifest_digest,
        "queue_index_sha256": queue_digest,
        "queue_index_bytes": QUEUE_PATH.stat().st_size,
        "selected_rows_commitment_sha256": canonical_sha({"rows": selection_commitment}),
        "selected_row_count": len(selected_rows),
        "selected_split_counts": {"train": 96, "development": 32},
        "selector_sha256": file_sha(ROOT / "scripts/select_commit_sequence_review_queue.py"),
        "reconstruction_module_sha256": file_sha(
            ROOT / "src/tinycomplete/one_line/commit_sequences.py"
        ),
        "grouping_module_sha256": file_sha(ROOT / "src/tinycomplete/one_line/data.py"),
        "verifier_sha256": file_sha(SCRIPT_PATH),
        "test_sha256": file_sha(TEST_PATH),
        "github_cli_version": gh_version,
        "github_auth": "existing gh credential store only; token never read or placed in argv",
        "caps": {
            "rows_total": MAX_ROWS,
            "train_rows": 96,
            "development_rows": 32,
            "reserved_test_rows": 0,
            "response_body_bytes": MAX_TRANSFER_BYTES,
            "response_body_accounting": (
                "all gh stdout and stderr bytes; HTTP framing not exposed by gh"
            ),
            "single_response_bytes": MAX_RESPONSE_BYTES,
            "wall_seconds": MAX_WALL_SECONDS,
            "private_output_bytes": MAX_OUTPUT_BYTES,
            "fresh_train_seed_candidates": 5,
        },
        "verification_policy": {
            "source": "GitHub REST API through authorized gh CLI",
            "base": (
                "only the single first parent of the exact queued child commit; "
                "merge commits are skipped"
            ),
            "source_check": (
                "fetch exact parent and child path bytes; require both SHA-256 values "
                "to match frozen queue identities"
            ),
            "license_check": (
                "at each revision inspect exact root license metadata, path-ancestor "
                "license files and source SPDX headers; require one consistent "
                "allowlisted SPDX identifier matching the dataset claim"
            ),
            "privacy": (
                "run existing sensitive-text filter on exact parent and child source; "
                "persist bodies only on private SSD and never in repository outputs"
            ),
            "uncertain_or_mismatched": "skip without weakening queue identity or split assignments",
            "training_labels": (
                "none; source verification does not establish edit chronology, "
                "inferability, or objective correctness"
            ),
            "provider_calls": 0,
            "model_downloads": 0,
            "reserved_evaluation_access": False,
        },
        "output_directory": str(OUTPUT_DIR),
    }


def freeze_plan() -> str:
    if PLAN_PATH.exists() or OUTPUT_DIR.exists():
        raise FileExistsError(
            "verification plan or output already exists; preserve it and revise explicitly"
        )
    identity = make_plan()
    identity["plan_identity_sha256"] = canonical_sha(identity)
    os.umask(0o077)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLAN_PATH.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(PLAN_PATH, 0o600)
    return sha256_file(PLAN_PATH)


def load_plan() -> tuple[dict[str, Any], str]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    supplied = plan.pop("plan_identity_sha256", None)
    if supplied != canonical_sha(plan):
        raise ValueError("frozen verification plan identity mismatch")
    current = make_plan()
    if current != plan:
        raise ValueError("frozen verification inputs, tools, queue, or runtime changed")
    plan["plan_identity_sha256"] = supplied
    return plan, sha256_file(PLAN_PATH)


class GitHubClient:
    """Bounded gh subprocess client; API bodies never reach stdout or logs."""

    def __init__(self, *, byte_cap: int, deadline: float) -> None:
        self.byte_cap = byte_cap
        self.deadline = deadline
        self.bytes_received = 0
        self.cache: dict[tuple[str, str, str], Any] = {}

    def request(self, endpoint: str, *, raw: bool = False) -> bytes:
        if time.monotonic() >= self.deadline:
            raise VerificationError("wall_deadline", stop=True)
        remaining = self.byte_cap - self.bytes_received
        if remaining <= 0:
            raise VerificationError("transfer_byte_cap", stop=True)
        args = ["gh", "api", "--hostname", "github.com"]
        if raw:
            args.extend(["-H", "Accept: application/vnd.github.raw"])
        args.append(endpoint)
        env = dict(os.environ)
        env["GH_PROMPT_DISABLED"] = "1"
        env["GH_NO_UPDATE_NOTIFIER"] = "1"
        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise VerificationError("github_cli_unavailable", stop=True) from exc
        assert process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        outputs = {"stdout": bytearray(), "stderr": bytearray()}
        try:
            while selector.get_map():
                time_left = self.deadline - time.monotonic()
                if time_left <= 0:
                    raise VerificationError("wall_deadline", stop=True)
                ready = selector.select(min(0.25, time_left))
                if not ready and process.poll() is not None:
                    # Drain both descriptors until EOF; select reports closed pipes.
                    ready = [
                        (key, selectors.EVENT_READ) for key in list(selector.get_map().values())
                    ]
                for key, _ in ready:
                    stream_name = key.data
                    remaining = self.byte_cap - self.bytes_received
                    if remaining <= 0:
                        raise VerificationError("transfer_byte_cap", stop=True)
                    try:
                        chunk = os.read(key.fd, min(64 * 1024, remaining))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    self.bytes_received += len(chunk)
                    outputs[stream_name].extend(chunk)
                    if len(outputs["stdout"]) + len(outputs["stderr"]) > MAX_RESPONSE_BYTES:
                        raise VerificationError("single_response_cap", stop=False)
            try:
                exit_code = process.wait(timeout=max(0.1, self.deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise VerificationError("wall_deadline", stop=True) from None
        except VerificationError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()
        body = bytes(outputs["stdout"])
        error = bytes(outputs["stderr"])
        if exit_code != 0:
            status = "api_error"
            if re.search(rb"HTTP\s+404", error):
                status = "not_found"
            elif re.search(rb"HTTP\s+401", error):
                raise VerificationError("authentication_failed", stop=True)
            elif re.search(rb"HTTP\s+(?:403|429)", error):
                raise VerificationError("rate_or_permission_limit", stop=True)
            raise VerificationError(status)
        return body

    def json(self, repo: str, endpoint: str) -> dict[str, Any]:
        key = (repo, endpoint, "json")
        if key not in self.cache:
            encoded_repo = quote(repo, safe="/")
            self.cache[key] = json.loads(self.request(f"repos/{encoded_repo}/{endpoint}"))
        value = self.cache[key]
        if not isinstance(value, dict):
            raise VerificationError("unexpected_api_shape")
        return value

    def raw(self, repo: str, path: str, revision: str) -> bytes:
        key = (repo, f"{path}?ref={revision}", "raw")
        if key not in self.cache:
            endpoint = (
                f"repos/{quote(repo, safe='/')}/contents/{quote(path, safe='/')}"
                f"?ref={quote(revision, safe='')}"
            )
            self.cache[key] = self.request(endpoint, raw=True)
        value = self.cache[key]
        if not isinstance(value, bytes):
            raise VerificationError("unexpected_api_shape")
        return value


def _license_candidate(name: str, entry_type: str) -> bool:
    return entry_type == "blob" and bool(LICENSE_NAME.match(name))


def _spdx_from_bytes(value: bytes) -> set[str]:
    header = b"\n".join(value.splitlines()[:80]).decode("utf-8", "ignore")
    identifiers = set()
    for match in SPDX.finditer(header):
        expression = match.group(1).strip().strip("*/# ").lower()
        if re.fullmatch(r"[a-z0-9.+-]+", expression):
            identifiers.add(expression)
    return identifiers


def _decode_github_blob(record: dict[str, Any]) -> bytes:
    if record.get("encoding") != "base64" or not isinstance(record.get("content"), str):
        raise VerificationError("license_blob_encoding")
    try:
        return base64.b64decode(record["content"], validate=False)
    except (ValueError, TypeError):
        raise VerificationError("license_blob_encoding") from None


def _content_hash_path(path: Path, data: bytes) -> None:
    if path.exists():
        if sha256_file(path) != sha256_bytes(data):
            raise VerificationError("private_blob_collision", stop=True)
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _walk_license_evidence(
    client: GitHubClient,
    repo: str,
    revision: str,
    tree_sha: str,
    file_path: str,
) -> dict[str, Any]:
    directories = file_path.split("/")[:-1]
    current_tree = tree_sha
    found: list[tuple[int, str, str]] = []
    for depth in range(len(directories) + 1):
        tree = client.json(repo, f"git/trees/{current_tree}")
        entries = tree.get("tree")
        if not isinstance(entries, list):
            raise VerificationError("tree_shape")
        for entry in entries:
            if isinstance(entry, dict) and _license_candidate(
                str(entry.get("path", "")), str(entry.get("type", ""))
            ):
                found.append((depth, str(entry["path"]), str(entry["sha"])))
        if depth == len(directories):
            break
        component = directories[depth]
        next_tree = next(
            (
                str(entry["sha"])
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("path") == component
                and entry.get("type") == "tree"
            ),
            None,
        )
        if next_tree is None:
            raise VerificationError("source_directory_missing")
        current_tree = next_tree
    if not found:
        return {"status": "no_license_file_in_ancestor_directories", "root_or_closest": None}
    max_depth = max(row[0] for row in found)
    closest = sorted((path, sha) for depth, path, sha in found if depth == max_depth)
    if len(closest) != 1:
        raise VerificationError("ambiguous_closest_license")
    path, blob_sha = closest[0]
    record = client.json(repo, f"git/blobs/{blob_sha}")
    data = _decode_github_blob(record)
    license_digest = sha256_bytes(data)
    _content_hash_path(OUTPUT_DIR / "license_blobs" / f"{license_digest}.txt", data)
    markers = _spdx_from_bytes(data)
    return {
        "status": "found",
        "root_or_closest": "root" if max_depth == 0 else "closest_ancestor",
        "path": path if max_depth else path,
        "git_blob_sha": blob_sha,
        "sha256": license_digest,
        "spdx_identifiers": sorted(markers),
        "bytes": len(data),
        "license_text_blob": f"license_blobs/{license_digest}.txt",
    }


def _revision_license(
    client: GitHubClient,
    repo: str,
    revision: str,
    file_path: str,
) -> dict[str, Any]:
    commit = client.json(repo, f"git/commits/{revision}")
    parents = commit.get("parents")
    tree = commit.get("tree")
    if (
        not isinstance(parents, list)
        or not isinstance(tree, dict)
        or not isinstance(tree.get("sha"), str)
    ):
        raise VerificationError("commit_shape")
    root_api: dict[str, Any] | None = None
    try:
        root_api = client.json(repo, f"license?ref={quote(revision, safe='')}")
    except VerificationError as exc:
        if exc.status not in {"not_found", "api_error"}:
            raise
    closest = _walk_license_evidence(client, repo, revision, tree["sha"], file_path)
    root_spdx: set[str] = set()
    root_evidence: dict[str, Any] | None = None
    if root_api is not None:
        license_meta = root_api.get("license")
        content = _decode_github_blob(root_api)
        digest = sha256_bytes(content)
        _content_hash_path(OUTPUT_DIR / "license_blobs" / f"{digest}.txt", content)
        if isinstance(license_meta, dict) and isinstance(license_meta.get("spdx_id"), str):
            root_spdx.add(license_meta["spdx_id"].lower())
        root_spdx |= _spdx_from_bytes(content)
        root_evidence = {
            "path": root_api.get("path"),
            "git_blob_sha": root_api.get("sha"),
            "sha256": digest,
            "spdx_identifiers": sorted(root_spdx),
            "bytes": len(content),
            "license_text_blob": f"license_blobs/{digest}.txt",
        }
        if closest.get("status") == "found" and closest.get("root_or_closest") == "root":
            if closest.get("path") != root_api.get("path") or closest.get(
                "git_blob_sha"
            ) != root_api.get("sha"):
                raise VerificationError("root_license_tree_mismatch")
    return {
        "tree_sha": tree["sha"],
        "parent_shas": [str(row.get("sha", "")) for row in parents if isinstance(row, dict)],
        "root_license": root_evidence,
        "closest_license": closest,
        "root_spdx": sorted(root_spdx),
    }


def _effective_spdx(license_record: dict[str, Any], source: bytes) -> tuple[set[str], str]:
    source_ids = _spdx_from_bytes(source)
    closest = license_record["closest_license"]
    closest_ids = (
        set(closest.get("spdx_identifiers", [])) if closest.get("status") == "found" else set()
    )
    root_ids = set(license_record.get("root_spdx", []))
    if closest.get("status") == "found" and closest.get("root_or_closest") == "closest_ancestor":
        evidence = closest_ids
        source_kind = "closest_license_file"
    elif closest_ids:
        evidence = closest_ids
        source_kind = "root_license_file"
    else:
        evidence = root_ids
        source_kind = "root_license_api"
    if not evidence:
        evidence = source_ids
        source_kind = "source_spdx_header"
    combined = evidence | source_ids
    return combined, source_kind


def _license_claim_matches(claim: str, identifiers: set[str]) -> bool:
    normalized = claim.strip().lower()
    return bool(normalized in ALLOWED_LICENSES and identifiers == {normalized})


def _store_source_pair(candidate_id: str, before: bytes, after: bytes) -> dict[str, str]:
    safe_id = hashlib.sha256(candidate_id.encode()).hexdigest()
    base = OUTPUT_DIR / "source_pairs" / safe_id
    before_sha = sha256_bytes(before)
    after_sha = sha256_bytes(after)
    _content_hash_path(base / f"parent-{before_sha}.src", before)
    _content_hash_path(base / f"child-{after_sha}.src", after)
    return {
        "parent_sha256": before_sha,
        "child_sha256": after_sha,
        "parent_private_path": str((base / f"parent-{before_sha}.src").relative_to(OUTPUT_DIR)),
        "child_private_path": str((base / f"child-{after_sha}.src").relative_to(OUTPUT_DIR)),
    }


def verify_candidate(row: dict[str, Any], client: GitHubClient) -> dict[str, Any]:
    repo = row.get("source_repo")
    commit_sha = row.get("source_revision")
    path = row.get("file_path")
    candidate_id = row.get("candidate_id")
    if (
        not isinstance(repo, str)
        or not REPOSITORY.fullmatch(repo)
        or not isinstance(commit_sha, str)
        or not SHA1.fullmatch(commit_sha)
        or not isinstance(path, str)
        or path.startswith("/")
        or ".." in path.split("/")
        or not isinstance(candidate_id, str)
    ):
        return {
            "candidate_id": candidate_id,
            "split": row.get("split"),
            "status": "invalid_frozen_identity",
        }
    try:
        commit = client.json(repo, f"git/commits/{commit_sha}")
        parents = commit.get("parents")
        if not isinstance(parents, list) or len(parents) != 1 or not isinstance(parents[0], dict):
            raise VerificationError("non_single_parent_commit")
        parent_sha = parents[0].get("sha")
        if not isinstance(parent_sha, str) or not SHA1.fullmatch(parent_sha):
            raise VerificationError("invalid_parent_identity")
        parent_bytes = client.raw(repo, path, parent_sha)
        child_bytes = client.raw(repo, path, commit_sha)
        parent_hash, child_hash = sha256_bytes(parent_bytes), sha256_bytes(child_bytes)
        if parent_hash != row.get("source_before_sha256"):
            raise VerificationError("parent_source_hash_mismatch")
        if child_hash != row.get("committed_child_file_sha256"):
            raise VerificationError("child_source_hash_mismatch")
        if _SENSITIVE_TEXT.search(
            parent_bytes.decode("utf-8", "ignore") + "\n" + child_bytes.decode("utf-8", "ignore")
        ):
            raise VerificationError("sensitive_text_screen")
        parent_license = _revision_license(client, repo, parent_sha, path)
        child_license = _revision_license(client, repo, commit_sha, path)
        parent_ids, parent_evidence = _effective_spdx(parent_license, parent_bytes)
        child_ids, child_evidence = _effective_spdx(child_license, child_bytes)
        claim = str(row.get("source_license_claim", "")).lower()
        if not _license_claim_matches(claim, parent_ids) or not _license_claim_matches(
            claim, child_ids
        ):
            raise VerificationError("license_unknown_or_mismatch")
        stored = _store_source_pair(candidate_id, parent_bytes, child_bytes)
        return {
            "candidate_id": candidate_id,
            "source_group_id": row["source_group_id"],
            "split": row["split"],
            "repository": repo,
            "file_path": path,
            "child_commit": commit_sha,
            "parent_commit": parent_sha,
            "source_pair": stored,
            "license_claim": claim,
            "parent_license": parent_license,
            "child_license": child_license,
            "parent_license_evidence_kind": parent_evidence,
            "child_license_evidence_kind": child_evidence,
            "status": "source_and_license_verified_for_human_review",
            "accepted_training": False,
            "chronology_observed": False,
            "inferability_reviewed": False,
        }
    except VerificationError as exc:
        if exc.stop:
            raise
        return {
            "candidate_id": candidate_id,
            "source_group_id": row.get("source_group_id"),
            "split": row.get("split"),
            "status": exc.status,
            "accepted_training": False,
        }


def _verify_authentication() -> None:
    result = subprocess.run(
        ["gh", "auth", "status", "--hostname", "github.com"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1"},
        check=False,
        timeout=15,
    )
    if result.returncode:
        raise VerificationError("authentication_unavailable", stop=True)


def execute() -> dict[str, Any]:
    plan, plan_raw_sha = load_plan()
    if OUTPUT_DIR.exists():
        raise FileExistsError("source verification output already exists; never overwrite it")
    usage = shutil.disk_usage(OUTPUT_DIR.parent)
    if usage.free < 2 * 1024**3:
        raise VerificationError("private_ssd_reserve_low", stop=True)
    _verify_authentication()
    os.umask(0o077)
    OUTPUT_DIR.mkdir(mode=0o700, parents=True)
    os.chmod(OUTPUT_DIR, 0o700)
    results_path = OUTPUT_DIR / "candidate_results.jsonl"
    started = time.monotonic()
    client = GitHubClient(byte_cap=MAX_TRANSFER_BYTES, deadline=started + MAX_WALL_SECONDS)
    rows = _queue_rows()
    counts: Counter[str] = Counter()
    result_rows: list[dict[str, Any]] = []
    stopped_status: str | None = None
    train_seed_count = 0
    with results_path.open("x", encoding="utf-8") as results:
        os.chmod(results_path, 0o600)
        for row in rows:
            if time.monotonic() >= client.deadline:
                stopped_status = "wall_deadline"
                break
            try:
                record = verify_candidate(row, client)
            except VerificationError as exc:
                record = {
                    "candidate_id": row.get("candidate_id"),
                    "source_group_id": row.get("source_group_id"),
                    "split": row.get("split"),
                    "status": exc.status,
                    "accepted_training": False,
                }
                if exc.stop:
                    stopped_status = exc.status
            if record["status"] == "source_and_license_verified_for_human_review":
                if row["split"] == "train" and train_seed_count < 5:
                    record["next_stage_seed_eligible"] = True
                    train_seed_count += 1
                else:
                    record["next_stage_seed_eligible"] = False
            else:
                record["next_stage_seed_eligible"] = False
            result_rows.append(record)
            counts[str(record["status"])] += 1
            results.write(json.dumps(record, sort_keys=True) + "\n")
            results.flush()
            os.fsync(results.fileno())
            if stopped_status:
                break
    unprocessed = len(rows) - len(result_rows)
    if unprocessed:
        counts["unprocessed_unknown"] = unprocessed
    output_bytes = sum(path.stat().st_size for path in OUTPUT_DIR.rglob("*") if path.is_file())
    if output_bytes > MAX_OUTPUT_BYTES:
        stopped_status = "private_output_cap"
    manifest = {
        "schema": "commit-sequence-source-verification-result-v1",
        "plan_raw_sha256": plan_raw_sha,
        "plan_identity_sha256": plan["plan_identity_sha256"],
        "status": "partial" if stopped_status or unprocessed else "complete",
        "stopped_status": stopped_status,
        "input_queue_sha256": plan["queue_index_sha256"],
        "rows_selected": len(rows),
        "rows_processed": len(result_rows),
        "rows_unprocessed_unknown": unprocessed,
        "preserved_split_counts": {"train": 96, "development": 32, "reserved": 0},
        "result_counts": dict(sorted(counts.items())),
        "eligible_train_seed_count_max5": train_seed_count,
        "response_body_bytes_counted": client.bytes_received,
        "response_body_budget_bytes": MAX_TRANSFER_BYTES,
        "wall_seconds": round(time.monotonic() - started, 3),
        "private_output_bytes": output_bytes,
        "candidate_results_sha256": sha256_file(results_path),
        "source_bodies_persisted_only_for_exact_hash_and_privacy_screen_pass": True,
        "license_text_persisted_private": True,
        "source_or_license_text_exported_to_repository": False,
        "accepted_training": 0,
        "chronology_observed": False,
        "inferability_reviewed": False,
        "teacher_or_provider_calls": 0,
        "model_downloads": 0,
        "reserved_evaluation_access": False,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(manifest_path, 0o600)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.freeze_plan:
            print(json.dumps({"status": "plan_frozen", "plan_sha256": freeze_plan()}))
        else:
            print(json.dumps(execute(), sort_keys=True))
    except VerificationError as exc:
        print(json.dumps({"status": "stopped", "reason": exc.status}), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        # Keep raw API/source text and exception contents out of logs.
        print('{"status":"failed","reason":"identity_or_io_error"}', file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
