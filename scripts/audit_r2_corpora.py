"""Reconcile frozen corpus samples and report policy differences, without selecting data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tinycomplete.code_cpt.model_data_r2 import MARKERS, PATHS, line_source_rejection
from tinycomplete.observability.spans import sanitize_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show-sample", action="store_true")
    parser.add_argument("--line-suite", type=Path)
    args = parser.parse_args()
    audits = []
    samples = []
    line_suite = args.line_suite or args.corpora / "causal_line_v1.jsonl"
    line_cases = [json.loads(row) for row in line_suite.read_text().splitlines()]
    overlap = {}
    for arm in ("R2_STANDARD", "R2_FILTERED"):
        selected = [
            json.loads(row)
            for path in (args.corpora / arm / "manifests").glob("*.jsonl")
            for row in path.read_text().splitlines()
        ]
        hashes = {row["content_sha256"] for row in selected}
        aliases = {alias for row in selected for alias in row["repository_aliases"]}
        content_overlap = [row["id"] for row in line_cases if row["source_sha256"] in hashes]
        repo_overlap = [row["id"] for row in line_cases if set(row["repository_aliases"]) & aliases]
        assert not content_overlap and not repo_overlap, "held-out diagnostic leaked into training"
        overlap[arm] = {
            "content_overlap": content_overlap,
            "repository_alias_overlap": repo_overlap,
        }
    for path in sorted((args.corpora / "pool").glob("*.jsonl")):
        language = path.stem
        pool = [json.loads(line) for line in path.read_text().splitlines()]
        by_hash = {row["content_sha256"]: row for row in pool}
        audit = json.loads((args.corpora / "audit" / (language + ".json")).read_text())
        audit["retained_pool_files"] = len(pool)
        audit["retained_pool_tokens"] = sum(len(row["tokens"]) for row in pool)
        audit["raw_candidate_tokens"] = None
        audit["raw_candidate_tokens_note"] = "Rejected source was not tokenized or retained"
        for arm in ("R2_STANDARD", "R2_FILTERED"):
            selected = [
                json.loads(line)
                for line in (args.corpora / arm / "manifests" / (language + ".jsonl"))
                .read_text()
                .splitlines()
            ]
            counts = Counter(row["content_sha256"] for row in selected)
            stats = audit["arms"][arm]
            stats["exact_duplicate_files"] = sum(n - 1 for n in counts.values())
            stats["exact_duplicate_file_fraction"] = stats["exact_duplicate_files"] / len(selected)
            stats["path_marker_files"] = sum(bool(PATHS.search(row["path"])) for row in selected)
            stats["content_marker_files"] = sum(
                bool(MARKERS.search(by_hash[row["content_sha256"]]["content"])) for row in selected
            )
            stats["retrospective_broader_generated_marker_counts"] = dict(
                Counter(
                    reason
                    for row in selected
                    if (reason := line_source_rejection(by_hash[row["content_sha256"]]))
                )
            )
            stats["retrospective_marker_note"] = (
                "Coverage audit using corrected diagnostic markers; no frozen selection changed. "
                "These broader markers were not part of the preregistered training treatment."
            )
            sample = json.loads(
                (args.corpora / "audit" / f"{language}-{arm}-sample.json").read_text()
            )
            for index, row in enumerate(sample):
                original = by_hash[row["content_sha256"]]
                assert (
                    hashlib.sha256(original["content"].encode()).hexdigest()
                    == row["content_sha256"]
                )
                assert original["repository"] == row["repository"]
                assert row["licenses"], "sample license provenance absent"
                record = {
                    "arm": arm,
                    "language": language,
                    "sample_index": index,
                    "content_sha256": row["content_sha256"],
                    "repository": row["repository"],
                    "path": row["path"],
                    "licenses": row["licenses"],
                    "parser": row["parse_status"],
                    "tokens": row["selected_tokens"],
                    "test_path": bool(
                        re.search(r"(?:^|[/_.-])tests?(?:[/_.-]|$)", row["path"], re.I)
                    ),
                    "content_bytes_verified": True,
                }
                samples.append(record)
                if args.show_sample and index == 0:
                    print(
                        json.dumps(
                            {**record, "preview": sanitize_text(original["content"], limit=480)}
                        )
                    )
        audits.append(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "observed_at": datetime.now(UTC).isoformat(),
                "audit_only_no_selection_changes": True,
                "line_training_overlap": overlap,
                "line_suite_sha256": hashlib.sha256(line_suite.read_bytes()).hexdigest(),
                "sample_files_verified": len(samples),
                "samples": samples,
                "languages": audits,
                "syntax_caveat": "Tree-sitter acceptance is syntax, not compilation or quality",
                "overlap_caveat": "Known identities excluded; original pretraining is unknown",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
