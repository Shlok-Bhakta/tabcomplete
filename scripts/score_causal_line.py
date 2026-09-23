"""Score frozen line predictions without regenerating, repairing, or reading gold to stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_causal_line import score_line

from tinycomplete.eval.code_generation import file_sha256, prediction_metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(prediction_metadata_path(args.predictions).read_text())
    assert metadata["protocol"] == "causal_line_v1"
    assert metadata["suite_sha256"] == file_sha256(args.suite)
    cases = [json.loads(line) for line in args.suite.read_text().splitlines()]
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    mapped = {row["case_id"]: row for row in predictions}
    assert len(mapped) == len(predictions) == len(cases) == 180
    assert set(mapped) == {case["id"] for case in cases}
    native = metadata["provider"] == "llama.cpp-native"
    records = []
    for case in cases:
        prediction = mapped[case["id"]]
        record = score_line(
            case,
            prediction["completion"],
            native_newline_omitted=(native and prediction.get("finish_reason") == "word"),
        )
        record.update(
            latency_seconds=prediction["latency_seconds"],
            returned_line_latency_seconds=prediction["latency_seconds"],
            output_tokens=prediction["generated_tokens"],
            finish_reason=prediction["finish_reason"],
            fallback_cap=prediction["hit_token_cap"],
            stopping="native newline or EOS" if native else "incremental token newline or EOS",
        )
        records.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "total": len(records),
                "exact": sum(row["exact"] for row in records),
                "syntax_pass": sum(row["syntax"] == "pass" for row in records),
                "predictions_sha256": file_sha256(args.predictions),
                "suite_sha256": metadata["suite_sha256"],
                "metadata": metadata,
                "scoring_rule": "Registered LF/CRLF removal only; code whitespace preserved",
                "scoring_code_sha256": file_sha256(Path(__file__)),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
