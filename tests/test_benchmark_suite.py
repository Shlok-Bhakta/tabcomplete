import hashlib
import json
from collections import Counter
from pathlib import Path

from tinycomplete.eval.code_benchmark import Prediction, evaluate_prediction, load_suite

SUITE = Path("data/benchmarks/code_completion_v1.jsonl")
MANIFEST = Path("data/benchmarks/code_completion_v1.manifest.json")


def test_suite_matches_frozen_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["case_count"] == 200
    assert manifest["gold_parse_passes"] == 200
    assert manifest["gold_compile_passes"] == 200
    assert manifest["gold_test_passes"] == 200


def test_suite_has_the_required_shape():
    cases = load_suite(SUITE)

    assert len(cases) == 200
    assert len({case.id for case in cases}) == 200
    assert Counter(case.language for case in cases) == {
        "python": 20 + 3,
        "typescript": 20 + 3,
        "javascript": 20 + 2,
        "java": 20 + 2,
        "cpp": 20 + 2,
        "rust": 20 + 2,
        "go": 20 + 2,
        "c": 20 + 2,
        "csharp": 20 + 2,
    }
    assert sum(case.repository_context for case in cases) == 20
    assert sum(not case.repository_context for case in cases) == 180
    assert all(case.check.compile for case in cases)
    assert all(case.check.test for case in cases)


def test_every_gold_assembly_is_parseable_without_execution():
    cases = load_suite(SUITE)
    for index, case in enumerate(cases):
        result = evaluate_prediction(
            case,
            Prediction(case_id=case.id, completion=case.expected),
            work_root=Path("/tmp") / "tinycomplete-benchmark-test" / str(index),
            execution_backend="none",
        )
        assert result.parse.status == "pass", case.id


def test_suite_is_deterministic_and_has_varied_categories():
    first = [case.model_dump(mode="json") for case in load_suite(SUITE)]
    second = [case.model_dump(mode="json") for case in load_suite(SUITE)]
    assert first == second
    assert len({case["category"] for case in first}) >= 5
    assert all(case["expected"] for case in first)
