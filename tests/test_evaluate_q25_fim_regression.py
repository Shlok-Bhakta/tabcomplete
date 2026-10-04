from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "scripts/evaluate_q25_fim_regression.py"


def _load_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("q25_fim_regression_wrapper_test", WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wrapper_validates_policy_and_keeps_metadata_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _load_wrapper()
    output = tmp_path / "regression"
    output.mkdir()
    metadata_path = output / "predictions.jsonl.metadata.json"
    plan_sha = "a" * 64
    baseline_metadata = {"plan_sha256": plan_sha, "suite_sha256": "b" * 64}
    metadata_path.write_text(json.dumps(baseline_metadata), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_q25_fim_regression.py",
            "--kind",
            "causal",
            "--attention-backend",
            wrapper.ATTENTION_BACKEND,
            "--model",
            str(tmp_path / "model"),
            "--suite",
            str(tmp_path / "suite.jsonl"),
            "--output",
            str(output),
            "--alias",
            "test-model",
            "--plan-sha",
            plan_sha,
        ],
    )
    monkeypatch.setattr(wrapper, "install_q25_fim_attention", lambda _backend: None)

    forwarded_argv: list[str] = []

    def run_target(script_path: str, *, run_name: str) -> None:
        assert Path(script_path).name == "evaluate_q25_code_cpt.py"
        assert run_name == "__main__"
        forwarded_argv.extend(sys.argv)
        assert "--attention-backend" not in sys.argv
        # The adapter removes its fields while the legacy evaluator verifies resume identity.
        assert json.loads(metadata_path.read_text(encoding="utf-8")) == baseline_metadata
        metadata_path.write_text(json.dumps(baseline_metadata), encoding="utf-8")

    monkeypatch.setattr(wrapper.runpy, "run_path", run_target)
    wrapper.main()

    assert "--model" in forwarded_argv
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["plan_sha256"] == plan_sha
    assert metadata["attention_backend"] == wrapper.ATTENTION_BACKEND
    assert metadata["key_value_head_expansion"] == "explicit-repeat"
    assert metadata["attention_heads"] == {"query": 14, "key_value": 2, "head_dim": 64}


def test_wrapper_rejects_an_unfrozen_attention_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _load_wrapper()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_q25_fim_regression.py",
            "--kind",
            "line",
            "--attention-backend",
            "sdpa",
        ],
    )
    with pytest.raises(SystemExit):
        wrapper.main()
