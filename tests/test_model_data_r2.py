import hashlib
from pathlib import Path

import yaml


def test_registered_r2_budget_and_shared_treatment_settings():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs/research/model_data_r2.yaml").read_text())
    train = config["training"]
    assert train["successful_updates"] * train["tokens_per_update"] == 5_013_504
    assert train["parent"] == "q35-p12"
    assert 2 * 5_013_504 + (6 + 16) * 32768 < config["limits"]["additional_training_input_tokens"]
    assert config["evaluation"]["causal"]["max_new_tokens"] == 96
    assert config["limits"]["simultaneous_gpu_allocations"] == 1
    assert config["selection"]["deployed_model_replacement"] == "forbidden"
    assert config["long_context"]["gpu_wall_seconds"] == 3600


def test_multiline_comment_is_not_a_line_target():
    from tinycomplete.code_cpt.model_data_r2 import line_candidate

    content = "/*\nthis is a comment without a leading star\n*/\nint meaning = 123456;\n"
    record = {
        "content": content,
        "parse_status": "pass",
        "language": "c",
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "repository": "synthetic/control",
        "repository_aliases": ["synthetic/control"],
        "path": "control.c",
    }
    case = line_candidate(record)
    assert case["line_number"] == 4
    assert case["source_before"] + case["reference"] + case["source_after"] == content
    assert case["prompt"] == case["source_before"]


def test_near_duplicates_and_repository_cap():
    from collections import Counter

    from tinycomplete.code_cpt.model_data_r2 import NearDuplicates, filtered_reason

    near = NearDuplicates()
    assert not near.check_and_add("def addition(a, b):\n    return a + b\n")
    assert near.check_and_add("def addition(a, b):\n  return a + b\n")
    record = {
        "content": "int x = 1;",
        "path": "code.c",
        "content_sha256": "x",
        "tokens": [1] * 20,
        "parse_status": "pass",
        "repository": "public/repo",
    }
    assert filtered_reason(record, NearDuplicates(), set(), Counter(), 1000) == (
        "repository_token_cap"
    )


def test_explicit_campaign_survives_prediction_resume(tmp_path):
    import json

    from tinycomplete.observability.context import ensure_persistent_run_identity

    path = tmp_path / "run.json"
    metadata = {"campaign_id": "synthetic-r2", "suite_sha256": "fixed"}
    first = ensure_persistent_run_identity(path, metadata)
    path.write_text(json.dumps(first.metadata))
    second = ensure_persistent_run_identity(path, metadata)
    assert first.context.campaign_id == second.context.campaign_id == "synthetic-r2"
    assert first.context.run_id == second.context.run_id
    assert first.context.run_attempt_id != second.context.run_attempt_id
