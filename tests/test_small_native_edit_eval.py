import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_small_next_edit_native import score, summarize  # noqa: E402


def row(action: str) -> dict:
    return {
        "id": "fixture",
        "action": action,
        "current": "ab",
        "history_before": "a",
        "region_start": 1,
        "region_end": 2,
        "after": "a" if action == "delete" else "ab",
    }


def response(text: str, stop_type: str = "eos") -> dict:
    return {"text": text, "tokens": 2, "stop_type": stop_type,
            "timings": {}, "latency_seconds": 0.1}


def test_native_score_separates_no_edit_deletion_and_reversal() -> None:
    silence = score(row("no_edit"), response("N\n"))
    assert silence["exact_after_state"]
    assert not silence["false_positive"]
    deleted = score(row("delete"), response("R\n"))
    assert deleted["exact_after_state"]
    assert deleted["reverses_latest_edit"]
    unnecessary = score(row("no_edit"), response("R\n"))
    assert unnecessary["false_positive"]
    assert unnecessary["unnecessary_change"]
    assert unnecessary["reverses_latest_edit"]
    incomplete = score(row("delete"), response("R\n", "limit"))
    assert incomplete["truncated"]
    assert incomplete["parse_status"] != "ok"
    assert not incomplete["exact_after_state"]
    summary = summarize([silence, deleted, unnecessary, incomplete])
    assert summary["edit_required_exact_after_state"] == 1
    assert summary["no_edit_false_positive"] == 1
    assert summary["reversals_of_latest_edit"] == 2
    assert summary["truncations"] == 1
