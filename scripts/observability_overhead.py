"""Measure deterministic boundary overhead and a small existing local-model sample."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from tinycomplete.eval.code_generation import OpenAICompatibleGenerationProvider
from tinycomplete.observability.bootstrap import initialize_observability
from tinycomplete.observability.config import ObservabilityConfig
from tinycomplete.observability.spans import operation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    results = {}
    for enabled in (False, True):
        runtime = initialize_observability(
            ObservabilityConfig(enabled=enabled, mode="live" if enabled else "disabled")
        )
        with runtime.activate():
            start = time.perf_counter()
            for _ in range(1000):
                with operation(
                    "overhead.fixture", attributes={"tabcomplete.task": "deterministic"}
                ):
                    sum(range(100))
            boundary = (time.perf_counter() - start) / 1000 * 1e6
            provider = OpenAICompatibleGenerationProvider(args.model_url, args.model_id)
            durations = []
            for _ in range(3):
                start = time.perf_counter()
                provider.generate_detailed("# Python\ndef add(a, b):\n    return", 8)
                durations.append((time.perf_counter() - start) * 1000)
        runtime.shutdown()
        results["enabled" if enabled else "disabled"] = {
            "boundary_mean_us": boundary,
            "local_model_request_ms": durations,
            "local_model_median_ms": statistics.median(durations),
        }
    results["interpretation"] = (
        "Small sequential CPU sample; cache/warmup and host load confound inference delta. "
        "Not a decode-throughput benchmark. Content copying disabled in this timing probe."
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
