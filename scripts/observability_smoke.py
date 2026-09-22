#!/usr/bin/env python3
"""Small synthetic evaluation and one optional existing local-model request."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tinycomplete.eval.code_benchmark import BenchmarkCase, CheckSpec, evaluate_prediction
from tinycomplete.eval.code_generation import (
    DetailedGeneration,
    OpenAICompatibleGenerationProvider,
    generate_predictions,
)
from tinycomplete.observability.artifacts import flush_artifact_transfers
from tinycomplete.observability.bootstrap import current_runtime
from tinycomplete.observability.context import RunContext, current_run_context
from tinycomplete.observability.spans import operation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-url")
    parser.add_argument("--model-id")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "known-pass": "print('ok')\n",
        "known-syntax-fail": "def broken(:\n",
        "known-test-fail": "print('wrong')\n",
    }
    contexts = {}

    class Provider:
        def generate_detailed(self, prompt, max_new_tokens):
            context = current_run_context()
            contexts[context.case_id] = context
            return DetailedGeneration(outputs[context.case_id], None, "fixture_complete")

    cases = [
        BenchmarkCase(
            id=key,
            language="python",
            path="solution.py",
            prefix="",
            expected="print('ok')\n",
            check=CheckSpec(
                compile=[sys.executable, "-m", "py_compile", "solution.py"],
                test=[sys.executable, "solution.py"],
                expected_stdout="ok\n",
            ),
        )
        for key in outputs
    ]
    suite_hash = hashlib.sha256(
        json.dumps([c.model_dump() for c in cases], sort_keys=True).encode()
    ).hexdigest()
    metadata = {
        "schema_version": 1,
        "suite_sha256": suite_hash,
        "protocol": "observability-smoke-v1",
        "provider": "fixture",
        "model_source": "deterministic-fixture",
        "decoding": {"do_sample": False},
        "case_count": len(cases),
    }
    predictions = generate_predictions(
        cases,
        Provider(),
        args.output / "predictions.jsonl",
        run_metadata=metadata,
        workers=2,
        prompt_builder=lambda case: "# Synthetic observability fixture\n# Print ok exactly once.\n",
    )
    results = []
    for case, prediction in zip(cases, predictions, strict=True):
        with contexts[case.id].activate():
            results.append(
                evaluate_prediction(
                    case,
                    prediction,
                    work_root=args.output / "work",
                    execution_backend="trusted-host",
                ).model_dump(mode="json")
            )
    observed_identity = next(iter(contexts.values()))
    identity = RunContext(
        observed_identity.campaign_id,
        observed_identity.run_id,
        observed_identity.run_attempt_id,
        workload=observed_identity.workload,
    )
    with identity.for_case("timeout-fixture").activate():
        try:
            with operation(
                "model.generate",
                attributes={
                    "tabcomplete.retry.number": 0,
                    "gen_ai.request.model": "timeout-fixture",
                },
            ):
                raise TimeoutError("Synthetic provider timeout for observability verification")
        except TimeoutError:
            pass
    with identity.activate():
        with operation(
            "run.heartbeat",
            attributes={"tabcomplete.run.state": "active", "tabcomplete.cases.completed": 3},
        ):
            pass
    real = None
    if args.model_url:
        if not args.model_id:
            parser.error("--model-id is required with --model-url")
        provider = OpenAICompatibleGenerationProvider(args.model_url, args.model_id)
        context = RunContext(
            identity.campaign_id, identity.run_id, identity.run_attempt_id
        ).for_case("real-local-model")
        with context.activate():
            result = provider.generate_detailed("# Python\ndef add(a, b):\n    return", 8)
            real = {
                "request_id": context.request_id,
                "text": result.text,
                "tokens": result.tokens,
                "finish_reason": result.finish_reason,
            }
    summary = {
        "schema_version": 1,
        "run_id": identity.run_id,
        "suite_sha256": suite_hash,
        "cases": results,
        "expected": {"cases": 3, "functional_pass": 1, "functional_fail": 2},
        "real_model": real,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (
        identity.activate(),
        operation(
            "run.summary",
            attributes={
                "tabcomplete.run.state": "completed",
                "tabcomplete.cases.completed": 3,
                "tabcomplete.cases.planned": 3,
                "tabcomplete.cases.failed": 2,
                "tabcomplete.suite.sha256": suite_hash,
                "tabcomplete.protocol": "observability-smoke-v1",
                "tabcomplete.terminal_event_id": identity.run_id + ":summary",
            },
        ),
    ):
        pass
    flush_artifact_transfers()
    current_runtime().force_flush()
    print(
        json.dumps(
            {
                "run_id": identity.run_id,
                "case_ids": list(outputs),
                "request_ids": [c.request_id for c in contexts.values()],
                "real_request_id": real["request_id"] if real else None,
                "result_file": str(args.output / "results.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
