from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION
from tinycomplete.one_line.contract import EditAction, EditState, RecentEdit, apply_action
from tinycomplete.one_line.fixed_state_role_receipts import (
    export_fixed_state_role_execution_proof,
    verify_fixed_state_role_execution_proof,
)
from tinycomplete.one_line.fixed_state_roles import (
    HISTORY_ORDER,
    SOURCE_TYPE,
    build_fixed_state_prompt,
)
from tinycomplete.one_line.teacher import MODEL_ID, TeacherResponse, TeacherUsageLedger

_ROOT = Path(__file__).resolve().parents[1]
_RUNNER_PATH = _ROOT / "scripts/run_fixed_state_role_pilot.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "fixed_state_role_receipts_test_runner", _RUNNER_PATH
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = runner
_RUNNER_SPEC.loader.exec_module(runner)


class ByteTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode("utf-8")) + ([0] if add_special_tokens else [])


def _private_write(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _state() -> EditState:
    return EditState(
        file_id="public/receipt_fixture.py",
        filetype="python",
        source="def value(x):\n    return x\n",
        target_row=1,
        cursor_col=len(b"    return x"),
        history=(RecentEdit(1, "    ", "    return x"),),
        relevant=(),
    )


def _release(plan: dict[str, Any]) -> bytes:
    ids = [case["request_ids"][role] for case in plan["cases"] for role in runner.CALL_ROLES]
    result = {
        "schema": runner.RELEASE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "baseline_ledger_sha256": plan["ledger"]["baseline_sha256"],
        "baseline_request_ids_sha256": plan["ledger"]["baseline_request_ids_sha256"],
        "planned_request_ids": ids,
        "max_calls": len(ids),
        "max_wall_seconds": runner.MAX_PHASE_WALL_SECONDS,
        "reserve_input_tokens": runner.RESERVE_INPUT_TOKENS,
        "reserve_output_tokens": runner.RESERVE_OUTPUT_TOKENS,
        "global_caps": {
            "calls": runner.MAX_CALLS,
            "input_tokens": runner.MAX_INPUT_TOKENS,
            "output_tokens": runner.MAX_OUTPUT_TOKENS,
        },
        "released_by": "test-only-local-fixture",
        "released_at_utc": "2026-09-30T00:00:00Z",
    }
    return runner._canonical(result) + b"\n"


def _candidate(tokenizer: ByteTokenizer) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _state()
    prompt = build_fixed_state_prompt(state, tokenizer)
    input_row = {
        "candidate_id": "public-source/receipt-case-001",
        "seed_id": "public-source/seed-001",
        "state": asdict(state),
        "prompt": prompt.user_prompt,
        "context_sha256": prompt.context_sha256,
    }
    candidate = {
        **input_row,
        "id": input_row["candidate_id"],
        "source_type": SOURCE_TYPE,
        "history_order": HISTORY_ORDER,
        "human_chronology_observed": False,
        "context_policy": CONTEXT_POLICY_VERSION,
        "history_sha256": prompt.history_sha256,
        "action": asdict(EditAction("keep")),
        "after_source": apply_action(state, EditAction("keep")),
        "split": "train",
        "source_repo": "public/example",
        "source_revision": "a" * 40,
        "source_path": "src/example.py",
        "source_license": "MIT",
        "source_license_sha256": "b" * 64,
        "source_group_id": "public/example@parent",
        "session_or_commit": "parent-commit-001",
        "task_family_id": "python-fixed-prefix",
        "template_id": "fixed-prefix-001",
    }
    return input_row, candidate


def _complete_fake_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    resume_after_durable_settlement: bool = False,
    reviewer_retains: bool = True,
):
    tokenizer = ByteTokenizer()
    input_row, candidate = _candidate(tokenizer)
    packet_path = tmp_path / "inputs.jsonl"
    _private_write(packet_path, (json.dumps(input_row, ensure_ascii=False) + "\n").encode())
    run_dir = tmp_path / "local-fake-run"
    ledger_path = tmp_path / "private-global-usage.jsonl"
    plan = runner.freeze_plan(
        packet_path=packet_path,
        run_dir=run_dir,
        ledger_path=ledger_path,
        tokenizer=tokenizer,
        runtime={"fixture": "local-fake-only"},
    )
    release_path = tmp_path / "root-release.json"
    release_bytes = _release(plan)
    _private_write(release_path, release_bytes)
    monkeypatch.setattr(runner, "run_scope", lambda *_args: nullcontext(None))
    monkeypatch.setattr(runner, "_enable_offline_observability", lambda *_args: None)
    monkeypatch.setattr(runner, "operation", lambda *_args, **_kwargs: nullcontext(None))

    provider_requests: list[str] = []
    interrupt_once = [resume_after_durable_settlement]

    class FakeClient:
        def __init__(self, ledger: TeacherUsageLedger) -> None:
            self.ledger = ledger

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def run_role(self, **kwargs: Any) -> TeacherResponse:
            request_id = kwargs["request_id"]
            provider_requests.append(request_id)
            call_number = len(provider_requests)
            self.ledger.reserve(
                request_id,
                input_tokens=kwargs["reserve_input_tokens"],
                max_output_tokens=kwargs["reserve_output_tokens"],
            )
            is_reviewer = kwargs["purpose"] == "automated_score"
            text = (
                json.dumps(
                    {
                        "retain": reviewer_retains,
                        "ambiguous": False,
                        "reason": "Local test fixture.",
                    },
                    separators=(",", ":"),
                )
                if is_reviewer
                else "N"
            )
            response = TeacherResponse(
                content=text,
                session_id=f"fake-session-{call_number}",
                response_id=f"fake-response-{call_number}",
                model_id=MODEL_ID,
                input_tokens=10,
                output_tokens=len(text.encode("utf-8")),
                reasoning_tokens=1,
                total_tokens_reported=None,
                cached_read_tokens=0,
                cached_write_tokens=0,
                finish_reason="stop",
                cost_usd_reported=None,
            )
            kwargs["persist_completed_response"](response)
            self.ledger.settle(
                request_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens + response.reasoning_tokens,
            )
            if interrupt_once[0]:
                interrupt_once[0] = False
                raise RuntimeError("local fixture process exit after settlement")
            return response

    execute_kwargs = {
        "run_dir": run_dir,
        "release_path": release_path,
        "release_sha256": runner._sha(release_bytes),
        "tokenizer": tokenizer,
        "runtime": {"fixture": "local-fake-only"},
        "ledger_path_override": ledger_path,
        "client_factory": FakeClient,
    }
    if resume_after_durable_settlement:
        with pytest.raises(RuntimeError, match="process exit after settlement"):
            runner.execute_plan(**execute_kwargs)
    summary = runner.execute_plan(
        **execute_kwargs,
    )
    assert summary["phase_status"] == "complete"
    assert len(provider_requests) == 3
    output_root = tmp_path / "training-package"
    output_root.mkdir(mode=0o700)
    proof_ref = export_fixed_state_role_execution_proof(
        candidate,
        run_dir=run_dir,
        release_path=release_path,
        release_sha256=runner._sha(release_bytes),
        ledger_path=ledger_path,
        package_root=output_root,
        tokenizer=tokenizer,
    )
    return tokenizer, candidate, proof_ref, output_root, run_dir


def _verify(
    tokenizer: ByteTokenizer,
    candidate: dict[str, Any],
    proof_ref: dict[str, Any],
    root: Path,
):
    return verify_fixed_state_role_execution_proof(
        candidate, proof_ref, package_root=root, tokenizer=tokenizer
    )


def test_local_fake_runner_exports_relocatable_minimal_role_receipt_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer, candidate, proof_ref, package_root, _run_dir = _complete_fake_run(
        tmp_path, monkeypatch
    )
    decision = _verify(tokenizer, candidate, proof_ref, package_root)
    assert decision.accepted
    assert decision.reason == "fixed_state_role_execution_proof_verified"
    assert decision.evidence["functional_status"] == "not_evaluated_by_role_runner"
    assert decision.evidence["provider_signature"] is False
    assert len(decision.evidence["request_ids"]) == 3

    relative_root = Path(proof_ref["root"])
    proof_dir = package_root / relative_root
    manifest = json.loads((proof_dir / "proof_manifest.json").read_text())
    assert set(manifest["files"]) == {
        "root_release.json",
        "plan_projection.json",
        "events.jsonl",
        "role_evidence.json",
        "receipts/author.json",
        "receipts/solver.json",
        "receipts/reviewer.json",
    }
    projection = json.loads((proof_dir / "plan_projection.json").read_text())
    assert "path" not in projection["ledger_summary"]
    assert "baseline_rows" not in projection["ledger_summary"]
    assert "baseline_request_ids" not in projection["ledger_summary"]
    assert not (proof_dir / "private-global-usage.jsonl").exists()

    relocated = tmp_path / "relocated-package"
    shutil.copytree(package_root, relocated)
    relocated.chmod(0o700)
    assert _verify(tokenizer, candidate, proof_ref, relocated).accepted


def test_export_reconciles_roles_recovered_after_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer, candidate, proof_ref, package_root, run_dir = _complete_fake_run(
        tmp_path, monkeypatch, resume_after_durable_settlement=True
    )
    events = (run_dir / "events.jsonl").read_text().splitlines()
    first_event = json.loads(events[0])
    assert first_event["recovered_from_persisted_receipt"] is True
    assert _verify(tokenizer, candidate, proof_ref, package_root).accepted


def test_export_refuses_review_rejection_even_when_three_calls_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = ByteTokenizer()
    input_row, candidate = _candidate(tokenizer)
    packet_path = tmp_path / "inputs.jsonl"
    _private_write(packet_path, (json.dumps(input_row, ensure_ascii=False) + "\n").encode())
    run_dir = tmp_path / "rejected-review-run"
    ledger_path = tmp_path / "private-global-usage.jsonl"
    plan = runner.freeze_plan(
        packet_path=packet_path,
        run_dir=run_dir,
        ledger_path=ledger_path,
        tokenizer=tokenizer,
        runtime={"fixture": "local-fake-only"},
    )
    release_path = tmp_path / "root-release.json"
    release_bytes = _release(plan)
    _private_write(release_path, release_bytes)
    monkeypatch.setattr(runner, "run_scope", lambda *_args: nullcontext(None))
    monkeypatch.setattr(runner, "_enable_offline_observability", lambda *_args: None)
    monkeypatch.setattr(runner, "operation", lambda *_args, **_kwargs: nullcontext(None))

    class RejectingReviewerClient:
        def __init__(self, ledger: TeacherUsageLedger) -> None:
            self.ledger = ledger
            self.calls = 0

        def __enter__(self) -> RejectingReviewerClient:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def run_role(self, **kwargs: Any) -> TeacherResponse:
            self.calls += 1
            request_id = kwargs["request_id"]
            self.ledger.reserve(
                request_id,
                input_tokens=kwargs["reserve_input_tokens"],
                max_output_tokens=kwargs["reserve_output_tokens"],
            )
            is_reviewer = kwargs["purpose"] == "automated_score"
            text = (
                '{"retain":false,"ambiguous":true,"reason":"Insufficient evidence."}'
                if is_reviewer
                else "N"
            )
            response = TeacherResponse(
                content=text,
                session_id=f"reject-session-{self.calls}",
                response_id=f"reject-response-{self.calls}",
                model_id=MODEL_ID,
                input_tokens=10,
                output_tokens=len(text.encode()),
                reasoning_tokens=1,
                total_tokens_reported=None,
                cached_read_tokens=0,
                cached_write_tokens=0,
                finish_reason="stop",
                cost_usd_reported=None,
            )
            kwargs["persist_completed_response"](response)
            self.ledger.settle(
                request_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens + response.reasoning_tokens,
            )
            return response

    summary = runner.execute_plan(
        run_dir=run_dir,
        release_path=release_path,
        release_sha256=runner._sha(release_bytes),
        tokenizer=tokenizer,
        runtime={"fixture": "local-fake-only"},
        ledger_path_override=ledger_path,
        client_factory=RejectingReviewerClient,
    )
    assert summary["cases"][0]["role_evidence_accepted"] is False
    output_root = tmp_path / "training-package"
    output_root.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="independent role evidence was rejected"):
        export_fixed_state_role_execution_proof(
            candidate,
            run_dir=run_dir,
            release_path=release_path,
            release_sha256=runner._sha(release_bytes),
            ledger_path=ledger_path,
            package_root=output_root,
            tokenizer=tokenizer,
        )


def test_candidate_state_or_after_source_mismatch_fails_portable_receipt_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer, candidate, proof_ref, package_root, _run_dir = _complete_fake_run(
        tmp_path, monkeypatch
    )
    altered = dict(candidate)
    altered["after_source"] = altered["after_source"] + "# changed\n"
    assert not _verify(tokenizer, altered, proof_ref, package_root).accepted
    changed_state = dict(candidate)
    changed_state["state"] = dict(candidate["state"])
    changed_state["state"]["source"] += "# changed\n"
    assert not _verify(tokenizer, changed_state, proof_ref, package_root).accepted


def test_event_chain_reordering_and_missing_receipt_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer, candidate, proof_ref, package_root, _run_dir = _complete_fake_run(
        tmp_path, monkeypatch
    )
    proof_dir = package_root / proof_ref["root"]
    event_path = proof_dir / "events.jsonl"
    event_path.write_bytes(b"".join(reversed(event_path.read_bytes().splitlines(keepends=True))))
    event_path.chmod(0o600)
    assert not _verify(tokenizer, candidate, proof_ref, package_root).accepted

    # Restore from the local run through a fresh proof for the missing-file case.
    next_tmp = tmp_path / "second"
    next_tmp.mkdir(mode=0o700)
    tokenizer, candidate, proof_ref, package_root, _run_dir = _complete_fake_run(
        next_tmp, monkeypatch
    )
    (package_root / proof_ref["root"] / "receipts" / "solver.json").unlink()
    assert not _verify(tokenizer, candidate, proof_ref, package_root).accepted


def test_usage_projection_tampering_is_rejected_even_with_updated_manifest_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer, candidate, proof_ref, package_root, _run_dir = _complete_fake_run(
        tmp_path, monkeypatch
    )
    proof_dir = package_root / proof_ref["root"]
    manifest_path = proof_dir / "proof_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["calls"][0]["settled_usage"]["actual_input_tokens"] += 1
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    _private_write(manifest_path, manifest_bytes)
    proof_ref = {
        **proof_ref,
        "manifest_sha256": runner._sha(manifest_bytes),
        "manifest_bytes": len(manifest_bytes),
    }
    assert not _verify(tokenizer, candidate, proof_ref, package_root).accepted


def test_fake_receipts_are_local_test_fixtures_not_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer, candidate, proof_ref, package_root, _run_dir = _complete_fake_run(
        tmp_path, monkeypatch
    )
    result = _verify(tokenizer, candidate, proof_ref, package_root)
    assert result.accepted
    assert result.evidence["provider_signature"] is False
