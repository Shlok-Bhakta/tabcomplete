"""Bounded, unreviewed chronology yield pilot over pinned public Git seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import cast

from tinycomplete.one_line.chronology import (
    MineLimits,
    PinnedChronologySource,
    mine_cross_commit_candidates,
)

EXPECTED_SEEDS_SHA256 = "f1320155723d450454484999ebb3605614e251c223e1801f7e9808d1d3bde933"
LANGUAGES = ("python", "typescript", "rust", "go")


class TransferBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self.lock = threading.Lock()
        self.exhausted = threading.Event()

    def take(self, data: bytes) -> bytes:
        with self.lock:
            remaining = self.limit - self.used
            if remaining <= 0:
                self.exhausted.set()
                return b""
            allowed = data[:remaining]
            self.used += len(allowed)
            if self.used >= self.limit:
                self.exhausted.set()
            return allowed


class CountingConnectProxy(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        budget: TransferBudget = self.server.budget  # type: ignore[attr-defined]
        self.request.settimeout(10)
        try:
            header = bytearray()
            while b"\r\n\r\n" not in header and len(header) < 8192:
                chunk = self.request.recv(1024)
                if not chunk:
                    return
                header.extend(chunk)
            first = header.split(b"\r\n", 1)[0]
            if first != b"CONNECT github.com:443 HTTP/1.1":
                self.request.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return
            remote = socket.create_connection(("github.com", 443), timeout=10)
            with remote:
                self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                self.request.settimeout(15)
                remote.settimeout(15)
                sel = selectors.DefaultSelector()
                with sel:
                    sel.register(self.request, selectors.EVENT_READ, remote)
                    sel.register(remote, selectors.EVENT_READ, self.request)
                    while not budget.exhausted.is_set():
                        ready = sel.select(timeout=15)
                        if not ready:
                            return
                        for key, _ in ready:
                            data = cast(socket.socket, key.fileobj).recv(65536)
                            if not data:
                                return
                            allowed = budget.take(data)
                            if not allowed:
                                return
                            key.data.sendall(allowed)
                            if len(allowed) != len(data):
                                return
        except (OSError, TimeoutError):
            return


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, budget: TransferBudget) -> None:
        super().__init__(("127.0.0.1", 0), CountingConnectProxy)
        self.budget = budget


def _git(
    checkout: Path, *args: str, proxy: str | None = None, timeout: int = 45
) -> subprocess.CompletedProcess[bytes]:
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    command = ["git"]
    if proxy:
        command.extend(["-c", f"http.proxy={proxy}", "-c", "http.followRedirects=false"])
    command.extend(["-C", str(checkout), *args])
    return subprocess.run(command, capture_output=True, check=False, env=env, timeout=timeout)


def _ordered_seeds(path: Path) -> list[dict]:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED_SEEDS_SHA256:
        raise ValueError("frozen seed artifact hash mismatch")
    groups: dict[str, list[dict]] = {language: [] for language in LANGUAGES}
    for line in source.splitlines():
        row = json.loads(line)
        language = row["student_state_seed"]["filetype"]
        if language not in groups:
            raise ValueError("unsupported seed language")
        metadata = row["authoring_metadata"]
        if (
            metadata["candidate_split"] != "unassigned_authoring_only"
            or metadata["source_license"] != "MIT"
        ):
            raise ValueError("seed split or license changed")
        groups[language].append(row)
    if any(len(rows) != 25 for rows in groups.values()):
        raise ValueError("expected exactly 25 frozen seeds per language")
    return [groups[language][index] for index in range(25) for language in LANGUAGES]


def run(args: argparse.Namespace) -> dict:
    seeds = _ordered_seeds(args.seeds)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    budget = TransferBudget(args.transfer_cap_bytes)
    proxy_server = ProxyServer(budget)
    proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    proxy_thread.start()
    proxy = f"http://127.0.0.1:{proxy_server.server_address[1]}"
    rows: list[dict] = []
    results: list[dict] = []
    deadline = time.monotonic() + args.max_total_seconds

    def record(result: dict) -> None:
        results.append(result)
        (args.report_dir / "progress.json").write_text(
            json.dumps(
                {
                    "results": results,
                    "proxied_transfer_bytes": budget.used,
                    "transfer_cap_bytes": budget.limit,
                    "candidate_count": len(rows),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    try:
        for seed in seeds[: args.max_repos]:
            if budget.exhausted.is_set() or time.monotonic() >= deadline:
                break
            metadata = seed["authoring_metadata"]
            repo = metadata["source_repo"]
            revision = metadata["source_revision"]
            result = {
                "repo": repo,
                "language": seed["student_state_seed"]["filetype"],
                "status": "missing",
            }
            if metadata["source_path"].endswith((".d.ts", ".pb.go")):
                result["status"] = "excluded_generated_path"
                record(result)
                continue
            with tempfile.TemporaryDirectory(prefix="git-", dir=args.artifact_dir) as temp:
                checkout = Path(temp)
                init = _git(checkout, "init", "--bare", timeout=10)
                if init.returncode:
                    result["status"] = "local_init_failed"
                    record(result)
                    continue
                try:
                    fetched = _git(
                        checkout,
                        "fetch",
                        "--no-tags",
                        f"--depth={args.max_commits + 1}",
                        "--filter=blob:none",
                        f"https://github.com/{repo}.git",
                        revision,
                        proxy=proxy,
                        timeout=args.per_repo_seconds,
                    )
                except subprocess.TimeoutExpired:
                    result["status"] = "fetch_timeout"
                    record(result)
                    continue
                if fetched.returncode:
                    result["status"] = (
                        "transfer_cap" if budget.exhausted.is_set() else "fetch_failed"
                    )
                    record(result)
                    continue
                chain = _git(
                    checkout,
                    "rev-list",
                    "--first-parent",
                    f"--max-count={args.max_commits + 1}",
                    revision,
                )
                if chain.returncode:
                    result["status"] = "missing_pinned_revision"
                    record(result)
                    continue
                commits = chain.stdout.decode().splitlines()
                result["first_parent_commits"] = len(commits)
                if len(commits) < 3:
                    result["status"] = "insufficient_history"
                    record(result)
                    continue
                try:
                    pinned = PinnedChronologySource(
                        checkout=checkout,
                        repo_id=repo,
                        public_url=f"https://github.com/{repo}",
                        base_rev=commits[-1],
                        tip_rev=revision,
                        file_path=metadata["source_path"],
                        license_spdx=metadata["source_license"],
                        license_path=metadata["license_path"],
                        license_sha256=metadata["license_sha256"],
                        aliases=tuple(metadata["source_aliases"]),
                    )
                    candidates = mine_cross_commit_candidates(
                        pinned,
                        limits=MineLimits(
                            max_commits=args.max_commits,
                            max_blob_bytes=128_000,
                            max_total_read_bytes=2_000_000,
                            max_seconds=args.per_repo_seconds,
                            max_candidates=1,
                        ),
                    )
                except (ValueError, RuntimeError, TimeoutError) as error:
                    result["status"] = type(error).__name__
                    record(result)
                    continue
                result["status"] = (
                    "structural_candidate" if candidates else "no_structural_candidate"
                )
                result["candidate_count"] = len(candidates)
                if candidates:
                    candidate = candidates[0]
                    result["candidate_id"] = candidate["id"]
                    result["file_spdx_notice"] = candidate["provenance"]["file_spdx_notice"]
                    rows.append(candidate)
                    (args.artifact_dir / "unreviewed_candidates.jsonl").write_text(
                        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
                    )
                record(result)
    finally:
        proxy_server.shutdown()
        proxy_server.server_close()
        proxy_thread.join(timeout=5)
    candidates_path = args.artifact_dir / "unreviewed_candidates.jsonl"
    candidates_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    status_counts = Counter(result["status"] for result in results)
    report = {
        "source_seed_sha256": EXPECTED_SEEDS_SHA256,
        "source_seed_count": len(seeds),
        "selected_seed_count": min(args.max_repos, len(seeds)),
        "attempted_repo_count": len(results),
        "unattempted_selected_count": min(args.max_repos, len(seeds)) - len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "structural_candidate_count": len(rows),
        "accepted_training_label_count": 0,
        "file_spdx_notice_candidate_count": sum(
            bool(row["provenance"]["file_spdx_notice"]) for row in rows
        ),
        "transfer_cap_bytes": budget.limit,
        "proxied_transfer_bytes": budget.used,
        "transfer_cap_reached": budget.exhausted.is_set(),
        "max_commits_per_repo": args.max_commits,
        "candidate_artifact": str(candidates_path),
        "candidate_artifact_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        "results": results,
    }
    (args.report_dir / "pilot_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("artifacts/research/one_line_r1/public_source_authoring_100.jsonl"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/research/one_line_r1/chronology_pilot")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/research/one_line_r1/chronology_pilot")
    )
    parser.add_argument("--max-repos", type=int, default=40)
    parser.add_argument("--max-commits", type=int, default=32)
    parser.add_argument("--transfer-cap-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--per-repo-seconds", type=int, default=30)
    parser.add_argument("--max-total-seconds", type=int, default=900)
    options = parser.parse_args()
    if not 1 <= options.max_repos <= 100 or not 3 <= options.max_commits <= 32:
        parser.error("repository or commit cap outside pilot bounds")
    if not 1 <= options.transfer_cap_bytes <= 100 * 1024 * 1024:
        parser.error("transfer cap outside pilot bounds")
    print(json.dumps(run(options), sort_keys=True))
