from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_one_line_gpu_pilot.py"
SPEC = importlib.util.spec_from_file_location("evaluate_one_line_gpu_pilot_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_development_input_requires_exact_replay_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    row = {
        "id": "dev-1",
        "split": "development",
        "state": {
            "file_id": "core/example.ts",
            "filetype": "typescript",
            "source": "const n = 1;\n",
            "target_row": 0,
            "cursor_col": 0,
        },
        "action": {"kind": "replace_line", "text": "const n = 2;"},
        "after_source": "const n = 2;\n",
    }
    path.write_text(json.dumps(row) + "\n")
    digest = evaluator.sha256_file(path)
    assert evaluator.load_rows(path, digest) == [row]
    with pytest.raises(ValueError, match="SHA-256"):
        evaluator.load_rows(path, "0" * 64)
    row["after_source"] = "const n = 3;\n"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="does not replay"):
        evaluator.load_rows(path, evaluator.sha256_file(path))


@pytest.mark.parametrize('reason,truncated,wire,valid', [
    ('eos', False, 'R\té = 2', True),
    ('limit', False, 'R\té = 2', False),
    ('eos', True, 'R\té = 2', False),
    ('eos', False, 'R\té = 2\n', False),
    (None, False, 'N', False),
])
def test_native_actions_require_observed_eos_and_unchanged_unicode_wire(
    reason, truncated, wire, valid
) -> None:
    from types import SimpleNamespace

    row = {'id': 'unicode', 'state': {
        'file_id': 'example.py', 'filetype': 'python', 'source': 'é = 1\n',
        'target_row': 0, 'cursor_col': 0,
    }, 'action': {'kind': 'replace_line', 'text': 'é = 2'}, 'after_source': 'é = 2\n'}
    context = SimpleNamespace(text='fixed prompt', input_tokens=4)
    generated = SimpleNamespace(tokens=5, finish_reason=reason, text=wire)
    result = evaluator.native_observation(row, context, generated, {
        'truncated': truncated, 'total_seconds': 0.1,
    })
    assert result['wire'] == wire
    assert result['valid_action'] is valid
    assert result['exact_after'] is valid
    assert result['elapsed_ms'] == 100.0


def test_native_evaluation_rejects_external_endpoint_before_inference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='loopback'):
        evaluator.evaluate_native(tmp_path / 'absent.gguf', [], server_url='https://example.com',
                                  server_pid=1, model_weight_sha256='0' * 64,
                                  source_weight_sha256='1' * 64, data_sha256='2' * 64)
