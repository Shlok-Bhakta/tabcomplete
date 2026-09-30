"""Prepare public Git reconstruction candidates for independent local review.

Consumes only the five already downloaded, hash-pinned CommitPackFT files.
No network, provider calls, training, or inferred human keep decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, cast

from tinycomplete.one_line.commit_sequences import VERSION, reconstruct_candidates
from tinycomplete.one_line.contract import EditAction, EditState, Filetype
from tinycomplete.one_line.data import (
    _SENSITIVE_TEXT,
    connected_groups,
    near_duplicate_key,
    validate_splits,
)
from tinycomplete.one_line.train import encode_training_row
from tinycomplete.one_line.visible_rules import VERSION as BASELINE_VERSION
from tinycomplete.one_line.visible_rules import predict_visible_identifier_copy

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/prototype/product_r2"
OUT = Path("/mnt/ssd/tabcomplete-product-r2/commitpackft/prepared")
TOKENIZER = Path(
    "/home/crabcake/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B"
    "/snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
LANGUAGES = {"python": ".py", "typescript": ".ts", "rust": ".rs", "go": ".go"}
LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"}
SEED = "public-commit-sequences-r1"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_path(path: str, suffix: str) -> bool:
    p = PurePosixPath(path)
    return (
        bool(path)
        and p.suffix == suffix
        and not p.is_absolute()
        and ".." not in p.parts
        and not {"vendor", "third_party", "generated", "node_modules"}.intersection(p.parts)
        and not any(char in path for char in ("\\", "\x00", "\n", "\r"))
    )


def load_candidates(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    candidates = []
    seen: set[str] = set()
    for language, suffix in LANGUAGES.items():
        entry = next(
            row for row in manifest["files"] if row["path"] == f"data/{language}/data.jsonl"
        )
        with Path(entry["destination"]).open() as handle:
            for raw in handle:
                original = json.loads(raw)
                counts[f"input_{language}"] += 1
                if original.get("license") not in LICENSES:
                    counts["license_excluded"] += 1
                    continue
                path = original.get("old_file")
                if (
                    not isinstance(path, str)
                    or path != original.get("new_file")
                    or not safe_path(path, suffix)
                ):
                    counts["path_or_rename_excluded"] += 1
                    continue
                aliases = sorted(set(str(original.get("repos", "")).split(",")))
                aliases = [value.strip() for value in aliases]
                if not 1 <= len(aliases) <= 10 or any(
                    not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", alias) for alias in aliases
                ):
                    counts["repository_identity_excluded"] += 1
                    continue
                commit = original.get("commit")
                if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit):
                    counts["commit_identity_excluded"] += 1
                    continue
                before, after = original.get("old_contents"), original.get("new_contents")
                if (
                    not isinstance(before, str)
                    or not isinstance(after, str)
                    or max(len(before.encode()), len(after.encode())) > 16_384
                    or "\x00" in before + after
                    or _SENSITIVE_TEXT.search(before + "\n" + after)
                ):
                    counts["size_or_sensitive_source_excluded"] += 1
                    continue
                try:
                    rows = reconstruct_candidates(
                        before, after, file_id=path, filetype=cast(Filetype, language)
                    )
                except (UnicodeError, ValueError):
                    counts["unsupported_line_encoding"] += 1
                    continue
                for row in rows:
                    identity = sha(
                        json.dumps([aliases, commit, path, row["state"]], sort_keys=True).encode()
                    )
                    if identity in seen:
                        counts["duplicate_state"] += 1
                        continue
                    seen.add(identity)
                    row.update(
                        {
                            "id": "commit-sequence/" + identity[:24],
                            "source_repo": aliases[0],
                            "source_aliases": aliases[1:],
                            "source_revision": commit,
                            "source_license": original["license"],
                            "session_or_commit": commit,
                            "mechanism": "unreviewed_relatedness",
                            "generator_family": "public-commit/" + aliases[0],
                            "template_id": f"public-commit/{commit}/{path}",
                        }
                    )
                    row["provenance"].update(
                        {
                            "dataset": "bigcode/commitpackft",
                            "dataset_revision": manifest["dataset_revision"],
                            "source_record_sha256": sha(raw.encode()),
                            "public_url": f"https://github.com/{aliases[0]}/commit/{commit}",
                            "repository_license_status": (
                                "publisher metadata; file scope unreviewed"
                            ),
                            "commit_message_used_in_prompt": False,
                            "history_is_reconstructed_not_observed": True,
                        }
                    )
                    candidates.append(row)
    counts["reconstructed_candidates"] = len(candidates)
    return candidates, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    manifest_path = REPORT / "commit_source_download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    module = ROOT / "src/tinycomplete/one_line/commit_sequences.py"
    identity: dict[str, Any] = {
        "schema": "commit-sequence-review-plan-v1",
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "reconstruction_version": VERSION,
        "module_sha256": sha(module.read_bytes()),
        "source_manifest_sha256": sha(manifest_path.read_bytes()),
        "tokenizer_sha256": sha((TOKENIZER / "tokenizer.json").read_bytes()),
        "seed": SEED,
        "max_file_bytes": 16_384,
        "max_candidate_rows": 4_096,
        "max_rows_per_repository": 8,
        "max_raw_pool_bytes": 256 * 1024**2,
        "max_new_output_bytes": 512 * 1024**2,
        "input_token_ceiling": 1_024,
        "action_eos_token_ceiling": 64,
        "baseline_version": BASELINE_VERSION,
        "license_allowlist": sorted(LICENSES),
        "candidate_accepted_training": False,
        "split_policy": (
            "connected repo aliases, commits and near duplicates; deterministic 70/15/15 group hash"
        ),
    }
    plan_path = REPORT / "commit_sequence_review_plan.json"
    if args.freeze:
        if plan_path.exists() or OUT.exists():
            raise FileExistsError("immutable plan or outputs already exist")
        plan_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    if not plan_path.exists() or json.loads(plan_path.read_text()) != identity:
        raise ValueError("matching frozen candidate plan required")
    if not args.execute:
        print(json.dumps({"status": "frozen", "plan_sha256": sha(plan_path.read_bytes())}))
        return
    if OUT.exists():
        raise FileExistsError("existing candidate outputs must not be overwritten")
    if shutil.disk_usage(OUT.parent).free < 8 * 1024**3:
        raise RuntimeError("storage reserve exhausted")
    for entry in manifest["files"]:
        p = Path(entry["destination"])
        if p.stat().st_size != entry["size"] or sha(p.read_bytes()) != entry["sha256"]:
            raise ValueError("source artifact identity mismatch")
    rows, counts = load_candidates(manifest)
    if sum(len(json.dumps(row).encode()) for row in rows) > identity["max_raw_pool_bytes"]:
        raise RuntimeError("candidate pool storage budget exceeded")
    groups = connected_groups(rows)
    for group in groups:
        group_id = sha(json.dumps(sorted(rows[index]["id"] for index in group)).encode())
        bucket = int(sha((SEED + group_id).encode())[:8], 16) % 100
        split = "train" if bucket < 70 else "development" if bucket < 85 else "test_new_repo"
        for index in group:
            rows[index]["split"] = split
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER, local_files_only=True, trust_remote_code=False
    )
    ranked = sorted(rows, key=lambda row: sha((SEED + row["id"]).encode()))
    chosen: list[dict[str, Any]] = []
    repo_counts: Counter[str] = Counter()
    near_seen: set[str] = set()
    input_counts: Counter[str] = Counter()
    for row in ranked:
        if len(chosen) >= identity["max_candidate_rows"]:
            break
        if any(repo_counts[repo] >= 8 for repo in [row["source_repo"], *row["source_aliases"]]):
            counts["repository_sampling_cap"] += 1
            continue
        near = near_duplicate_key(row)
        if near in near_seen:
            counts["normalized_duplicate_excluded"] += 1
            continue
        state = EditState.from_mapping(row["state"])
        action = EditAction(**row["action"])
        try:
            encoded = encode_training_row(tokenizer, state, action)
        except ValueError:
            counts["input_or_action_token_ceiling"] += 1
            continue
        if "Recent edits, oldest to newest (JSON old and new text):\nRelevant" in encoded.prompt:
            counts["history_omitted_from_prompt"] += 1
            continue
        control = predict_visible_identifier_copy(state)
        row["diagnostic_baseline"] = {
            "version": BASELINE_VERSION,
            "action": {"kind": control.kind, "text": control.text},
            "exact": control == action,
            "label_oracle": False,
        }
        row["prompt"] = encoded.prompt
        row["token_counts"] = {
            "input": encoded.prompt_tokens,
            "response_eos": encoded.response_tokens,
        }
        input_counts[f"{row['split']}_input"] += encoded.prompt_tokens
        input_counts[f"{row['split']}_response_eos"] += encoded.response_tokens
        chosen.append(row)
        near_seen.add(near)
        repo_counts.update([row["source_repo"], *row["source_aliases"]])
    split_audit = validate_splits(chosen)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen
    ).encode()
    if len(payload) > identity["max_new_output_bytes"]:
        raise RuntimeError("output storage budget exceeded")
    OUT.mkdir(mode=0o700)
    output = OUT / "candidates.jsonl"
    output.write_bytes(payload)
    summary = {
        "schema": "commit-sequence-review-result-v1",
        "plan_sha256": sha(plan_path.read_bytes()),
        "private_candidate_path": str(output),
        "candidate_sha256": sha(payload),
        "candidate_bytes": len(payload),
        "counts": dict(counts),
        "selected_rows": len(chosen),
        "source_repositories": len({row["source_repo"] for row in chosen}),
        "languages": dict(Counter(row["state"]["filetype"] for row in chosen)),
        "actions": dict(Counter(row["action"]["kind"] for row in chosen)),
        "baseline_exact": sum(row["diagnostic_baseline"]["exact"] for row in chosen),
        "token_counts": dict(input_counts),
        "split_audit": split_audit,
        "accepted_training": 0,
        "no_edit_labels": 0,
        "human_chronology_claimed": False,
        "file_license_scope_reviewed": False,
    }
    (REPORT / "commit_sequence_review_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
