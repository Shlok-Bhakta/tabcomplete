#!/usr/bin/env python3
"""Run every frozen two-seed action through the existing offline code sandbox."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

from tinycomplete.eval.code_benchmark import (
    BenchmarkCase,
    CheckSpec,
    Prediction,
    evaluate_prediction,
)
from tinycomplete.one_line.contract import EditAction, EditState, apply_action

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "scripts/run_public_mechanism_pilot.py"
PREFLIGHT_SCHEMA = "two-seed-oracle-preflight-v1"
_WORKING_PACKET = Path("/mnt/ssd/tabcomplete-product-r2/next-pilot-frozen-v7")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pilot() -> Any:
    spec = importlib.util.spec_from_file_location("public_mechanism_pilot", PILOT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("existing pilot runner could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _pinned_action_case(
    *,
    case_id: str,
    state_raw: dict[str, Any],
    action: dict[str, Any],
    fixture: dict[str, Any],
    action_key: str,
    expected_functional: str,
    work_root: Path,
) -> dict[str, Any]:
    state = EditState.from_mapping(state_raw)
    typed_action = EditAction(kind=action["kind"], text=action.get("text"))
    complete_source = apply_action(state, typed_action)
    source_sha = _sha_bytes(complete_source.encode("utf-8"))
    action_sha = _sha_bytes(_canonical(action))
    case_language = state.filetype
    source_filename = "solution.py" if case_language == "python" else "tooltip-view.ts"
    oracle_filename = "oracle.py" if case_language == "python" else "oracle.js"
    check = CheckSpec(
        test=fixture["runtime_command"],
        expected_stdout=fixture["expected_stdout"],
        files={oracle_filename: fixture["test_source"]},
        timeout_seconds=30,
        container_image=fixture["runtime_image"],
    )
    benchmark_case = BenchmarkCase(
        id=f"{case_id}/{action_key}",
        language=case_language,
        path=source_filename,
        prefix=complete_source,
        suffix="",
        expected="synthetic oracle fixture",
        check=check,
        category="synthetic-two-seed-oracle-control",
        repository_context=False,
    )
    result = evaluate_prediction(
        benchmark_case,
        Prediction(case_id=benchmark_case.id, completion=""),
        work_root=work_root,
        execution_backend="container",
    )
    parse_status = result.parse.status
    test_status = result.test.status
    functional_status = (
        "pass"
        if parse_status == "pass" and test_status == "pass"
        else "fail"
        if parse_status == "pass" and test_status == "fail"
        else "unavailable_or_invalid"
    )
    return {
        "case_action_id": action_key,
        "case_id": case_id,
        "state_sha256": _sha_bytes(_canonical(state_raw)),
        "action_sha256": action_sha,
        "complete_source_sha256": source_sha,
        "functional_expected": expected_functional,
        "functional_status": functional_status,
        "parse_status": parse_status,
        "test_status": test_status,
        "execution_backend": "container",
        "container_image": fixture["runtime_image"],
        "working_tree_sha256": result.working_tree_sha256,
    }


def run_preflight(packet_dir: Path) -> dict[str, Any]:
    packet = packet_dir.resolve(strict=True)
    if packet.is_symlink() or not stat.S_ISDIR(packet.stat().st_mode):
        raise ValueError("frozen packet directory is invalid")
    if packet.stat().st_uid != os.getuid() or stat.S_IMODE(packet.stat().st_mode) != 0o700:
        raise ValueError("frozen packet directory must be owner-only")
    plan_path = packet / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pilot = _load_pilot()
    ledger = pilot.TeacherUsageLedger(pilot.LEDGER)
    ledger_rows, _totals = pilot._ledger_snapshot(ledger)
    _plan, input_by_case = pilot._validate_frozen_two_seed_packet(
        packet,
        runner_sha256=_sha_file(PILOT_PATH),
        current_request_ids=sorted(ledger_rows),
        current_ledger_sha256=_sha_file(ledger.path),
        runtime=pilot._runtime_fingerprint(),
        require_oracle_preflight=False,
    )
    oracle_path = packet / "oracle_fixtures.jsonl"
    if _sha_file(oracle_path) != plan["oracle_bundle"]["sha256"]:
        raise ValueError("oracle fixture identity differs from frozen plan")
    oracle_by_case = {row["case_id"]: row for row in _read_jsonl(oracle_path)}
    if set(oracle_by_case) != set(input_by_case):
        raise ValueError("oracle fixture cases do not match the frozen inputs")

    output_path = packet / "oracle_preflight_result.json"
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("oracle preflight output already exists")
    work_base = packet / "oracle-preflight-work"
    work_base.mkdir(mode=0o700)
    records: list[dict[str, Any]] = []
    for case_id in plan["cases"]:
        state_raw = input_by_case[case_id]["state"]
        fixture = oracle_by_case[case_id]
        frozen_case = plan["cases"][case_id]
        action_specs = [("gold", frozen_case["gold_action"], "pass")]
        action_specs.extend(
            (f"wrong:{control['name']}", control["action"], "fail")
            for control in frozen_case["wrong_controls"]
        )
        for action_key, action, expected_functional in action_specs:
            safe_key = action_key.replace(":", "-")
            work_root = work_base / case_id / safe_key
            work_root.mkdir(mode=0o700, parents=True)
            record = _pinned_action_case(
                case_id=case_id,
                state_raw=state_raw,
                action=action,
                fixture=fixture,
                action_key=f"{case_id}:{action_key}",
                expected_functional=expected_functional,
                work_root=work_root,
            )
            records.append(record)

    all_expected = all(
        record["functional_status"] == record["functional_expected"]
        and record["parse_status"] == "pass"
        and record["test_status"]
        == ("pass" if record["functional_expected"] == "pass" else "fail")
        for record in records
    )
    doc: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "complete" if all_expected else "failed",
        "plan_sha256": plan["plan_sha256"],
        "input_sha256": plan["input_bundle"]["sha256"],
        "oracle_sha256": plan["oracle_bundle"]["sha256"],
        "preflight_script_sha256": _sha_file(Path(__file__)),
        "execution_backend": "existing pinned code_benchmark sandbox",
        "provider_calls": 0,
        "training_started": False,
        "recorded_at_unix_ns": time.time_ns(),
        "case_count": len(records),
        "cases": records,
    }
    doc["result_sha256"] = _sha_bytes(_canonical(doc))
    payload = json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(packet, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=_WORKING_PACKET)
    args = parser.parse_args()
    result = run_preflight(args.packet)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": result["case_count"],
                "plan_sha256": result["plan_sha256"],
                "result_sha256": result["result_sha256"],
                "provider_calls": result["provider_calls"],
            },
            sort_keys=True,
        )
    )
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
