"""Rebuild only the held-out diagnostic from frozen source, without changing training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tinycomplete.code_cpt.model_data_r2 import line_candidate, line_source_rejection
from tinycomplete.code_cpt.prepare import CORE_LANGUAGES
from tinycomplete.eval.code_benchmark import _parse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/research/model_data_r2/frozen-corpora"
REPORT = ROOT / "reports/research/model_data_r2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-revision", type=int, required=True)
    args = parser.parse_args()
    original = DATA / "causal_line_v1.jsonl"
    target = DATA / f"causal_line_v1-r{args.fixture_revision}.jsonl"
    if target.exists():
        raise ValueError("corrected fixture already frozen; verify rather than overwrite")
    rows, audit = [], {}
    for language in CORE_LANGUAGES:
        eligible, rejected = {}, Counter()
        pool = DATA / "pool" / (language + ".jsonl")
        with pool.open() as handle:
            for line in handle:
                record = json.loads(line)
                if not 30 <= record["bucket"] < 40:
                    continue
                reason = line_source_rejection(record)
                case = None if reason else line_candidate(record)
                if case is not None:
                    eligible[case["id"]] = case
                else:
                    rejected[reason or "no_valid_target"] += 1
        selected = sorted(eligible.values(), key=lambda row: row["selection_key"])[:20]
        if len(selected) != 20:
            raise ValueError("insufficient held-out pool for " + language)
        rows.extend(selected)
        audit[language] = {"eligible": len(eligible), "rejected": dict(rejected), "selected": 20}
    assert len(rows) == len({row["id"] for row in rows}) == 180
    for row in rows:
        restored = row["source_before"] + row["reference"] + row["source_after"]
        assert hashlib.sha256(restored.encode()).hexdigest() == row["source_sha256"]
        assert _parse(restored, row["language"]).status == "pass"
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    old_ids = {json.loads(line)["id"] for line in original.read_text().splitlines()}
    new_ids = {row["id"] for row in rows}
    record = {
        "fixture_revision": args.fixture_revision,
        "fixture_filename": target.name,
        "original_sha256": digest(original),
        "corrected_sha256": digest(target),
        "plan_sha256": digest(REPORT / "preregistered_plan.json"),
        "removed": sorted(old_ids - new_ids),
        "added": sorted(new_ids - old_ids),
        "audit": audit,
        "training_corpora_changed": False,
        "previous_predictions": "Preserved as superseded exploratory diagnostic results",
        "known_outcomes": "Qwen2.5 Q4 original line results; neither pilot's line outcomes",
    }
    (REPORT / f"data_audit/line-fixture-correction-r{args.fixture_revision}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    (REPORT / "data_audit/line-fixture-correction.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"cases": 180, "replaced": len(old_ids - new_ids), "sha256": digest(target)}))


if __name__ == "__main__":
    main()
