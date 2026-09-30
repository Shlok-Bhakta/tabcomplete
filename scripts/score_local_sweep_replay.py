#!/usr/bin/env python3
"""Score frozen local Sweep outputs; same-prompt repeats are consistency checks only."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_sweep_comparison as base  # noqa: E402

SCHEMA = "sweep-local-replay-scoring-plan-v1"
RUN = Path(
    "/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1/local_replay/20260930T090103Z-00540772e0"
)
PLAN = ROOT / "reports/prototype/sweep_comparison_r1/local_replay_scoring_plan_v2.json"
BASE_PLAN = ROOT / "reports/prototype/sweep_comparison_r1/scoring_plan_v7.json"
OUT = Path("/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1/local_replay_scoring_v2")
LOCAL_PLAN = ROOT / "reports/prototype/sweep_comparison_r1/local_replay_plan.json"
LOCAL_UPSTREAM_PLAN = ROOT / "reports/prototype/sweep_comparison_r1/plan-v2.json"
ARCHIVE = ROOT / "reports/prototype/sweep_comparison_r1/local_replay_scoring_inputs"
PRODUCER_COMMIT = "a681d5c4ba88388742b4217a838d840ed5c257a5"
PRECISIONS = ("q8_0", "q4_k_m")
KINDS = ("changed_state", "immediate_same_prompt_repeat")
CONTEXT_TOKENS = 3072
OUTPUT_TOKENS = 512


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def plan_id(value: dict[str, Any]) -> str:
    return base.digest_bytes(
        canonical({k: v for k, v in value.items() if k != "scoring_plan_sha256"})
    )


def local_plan_id(value: dict[str, Any]) -> str:
    return base.digest_bytes(canonical({k: v for k, v in value.items() if k != "plan_sha256"}))


def file(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path.name}")
    return value


def function_hash(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f"expected one frozen function {name}")
    segment = ast.get_source_segment(source, nodes[0])
    if segment is None:
        raise ValueError(f"could not identify frozen function {name}")
    return base.digest_bytes(segment.encode())


def freeze(run_dir: Path, plan_path: Path, output: Path) -> dict[str, Any]:
    lp, bp = read(LOCAL_PLAN), read(BASE_PLAN)
    if (
        lp.get("plan_sha256") != local_plan_id(lp)
        or lp.get("schema") != "sweep-local-replay-plan-v1"
    ):
        raise ValueError("local replay plan identity mismatch")
    upstream = read(LOCAL_UPSTREAM_PLAN)
    if upstream.get("plan_sha256") != lp["inputs"].get("upstream_plan_sha256") or sha(
        LOCAL_UPSTREAM_PLAN
    ) != lp["inputs"].get("upstream_plan_file_sha256"):
        raise ValueError("local replay upstream plan identity mismatch")
    runner_archive = ARCHIVE / "measure_sweep_local.py.a681d5c.py.txt"
    tests_archive = ARCHIVE / "test_measure_sweep_local.py.a681d5c.py.txt"
    prompt_archive = ARCHIVE / "run_sweep_comparison.py.a681d5c.py.txt"
    producer_hashes = {
        runner_archive: "d05914df807cac5fc21b014e24aef310a08244396217ebb178115df4993c456f",
        tests_archive: "d3232eb6c5f25eca24e8d0b1fa79af736fe586ee5d0a70bb68f334e85ea68c1c",
        prompt_archive: "2a896dc411928ad0392444232999f7eeb24da3ba7f6dbb680750dab8ed4b85e0",
    }
    if any(sha(path) != expected for path, expected in producer_hashes.items()):
        raise ValueError("archived producer source does not match its recorded Git commit")
    if lp["inputs"].get("prompt_builder_source_sha256") != producer_hashes[prompt_archive]:
        raise ValueError("replay prompt source differs from the archived producer")
    functions = {
        name: function_hash(prompt_archive, name)
        for name in ("build_sweep_prompt", "map_full_file")
    }
    current_functions = {
        name: function_hash(ROOT / "scripts/run_sweep_comparison.py", name) for name in functions
    }
    if functions != current_functions:
        raise ValueError("current scoring prompt/mapper functions differ from the producer")
    if (
        bp.get("schema") != base.SCHEMA
        or bp.get("revision") != 3
        or plan_id(bp) != bp.get("scoring_plan_sha256")
    ):
        raise ValueError("base scoring plan identity mismatch")
    if bp.get("implementation_sha256") != base._fingerprints():
        raise ValueError("base scorer source changed before local scoring freeze")
    for item in bp["inputs"].values():
        path = base.resolve_recorded_path(item["path"])
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("a base scoring fixture changed before local scoring freeze")
    campaign_path = base.resolve_recorded_path(bp["campaign"]["path"])
    run = run_dir.resolve()
    meta, summary = read(run / "metadata.json"), read(run / "summary.json")
    if (
        meta.get("plan_sha256") != lp["plan_sha256"]
        or summary.get("plan_sha256") != lp["plan_sha256"]
        or meta.get("quality_evidence") is not False
        or meta.get("no_gold_labels_loaded") is not True
        or meta.get("prompt_bundle_sha256") != sha(base.NEXT_EDIT_INPUTS_DEFAULT)
        or summary.get("completed_requests") != 192
        or summary.get("expected_requests") != 192
    ):
        raise ValueError("local replay is incomplete or differs from its measurement plan")
    prompt_rows = base._read_jsonl(base.NEXT_EDIT_INPUTS_DEFAULT)
    if [row.get("case_id") for row in prompt_rows] != lp["inputs"].get("expected_case_ids"):
        raise ValueError("frozen prompt bundle order differs from the measured replay")
    paths = {
        "local_plan": LOCAL_PLAN,
        "local_upstream_plan": LOCAL_UPSTREAM_PLAN,
        "base_plan": BASE_PLAN,
        "superseded_local_score_plan_v1": ROOT
        / "reports/prototype/sweep_comparison_r1/local_replay_scoring_plan_v1.json",
        "campaign_plan": campaign_path,
        "prompt_inputs": base.NEXT_EDIT_INPUTS_DEFAULT,
        "diagnostic": base.DIAGNOSTIC_PLAN_DEFAULT,
        "source_suite": base.NEXT_EDIT_SUITE_DEFAULT,
        "metadata": run / "metadata.json",
        "summary": run / "summary.json",
        "telemetry": run / "observability.jsonl",
        "telemetry_run": run / "observability-run.json",
    }
    inputs = {key: file(path) for key, path in paths.items()}
    predictions = {p: file(run / f"predictions-{p}.jsonl") for p in PRECISIONS}
    measurements = {p: file(run / f"server-measurement-{p}.json") for p in PRECISIONS}
    for p in PRECISIONS:
        m = read(run / f"server-measurement-{p}.json")
        if (
            m.get("model_sha256") != bp["models"][f"{p}_sha256"]
            or m.get("quality_evidence") is not False
        ):
            raise ValueError(f"runtime measurement identity mismatch: {p}")
    sources = {
        name: file(path)
        for name, path in {
            "adapter": Path(__file__),
            "tests": ROOT / "tests/test_score_local_sweep_replay.py",
            "base_scorer": ROOT / "scripts/score_sweep_comparison.py",
            "base_tests": ROOT / "tests/test_score_sweep_comparison.py",
            "archived_producer": runner_archive,
            "archived_producer_tests": tests_archive,
            "archived_prompt_source": prompt_archive,
            "evaluator": ROOT / "src/tinycomplete/eval/code_benchmark.py",
            "next_edit_contract": ROOT / "src/tinycomplete/eval/next_edit_benchmark.py",
        }.items()
    }
    value = {
        "schema": SCHEMA,
        "revision": 2,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "run_dir": str(run),
        "revision_reason": (
            "Bind the actual archived baseline producer source after the current local runner "
            "changed; preserve v1 and do not relabel its source identity."
        ),
        "producer": {
            "git_commit": PRODUCER_COMMIT,
            "runner_sha256": producer_hashes[runner_archive],
            "tests_sha256": producer_hashes[tests_archive],
            "prompt_source_sha256": producer_hashes[prompt_archive],
            "prompt_and_mapper_function_sha256": functions,
        },
        "local_plan_sha256": lp["plan_sha256"],
        "campaign_plan_sha256": bp["campaign"]["embedded_plan_sha256"],
        "base_scoring_plan_sha256": bp["scoring_plan_sha256"],
        "inputs": inputs,
        "predictions": predictions,
        "measurements": measurements,
        "sources": sources,
        "models": bp["models"],
        "output": str(output.resolve()),
        "contract": {
            "cases": 24,
            "repetitions": 2,
            "primary": KINDS[0],
            "repeat": KINDS[1],
            "repeat_quality_samples": 0,
            "response_repair": "none",
            "request_id_join": (
                "actual offline request.start ID by case/precision/kind and unique timestamp "
                "within 1s; never row order"
            ),
            "telemetry_content_hashes": False,
            "missing_or_ambiguous_id": "unknown, never fabricated",
            "joined_id_bound_to": ["validated prompt_sha256", "verified raw output_sha256"],
            "functional_backend": "base scoring plan pinned network-none images",
            "quality_scope": "fixed benchmark evidence, not human acceptance/general quality",
        },
    }
    value["scoring_plan_sha256"] = plan_id(value)
    if plan_path.exists():
        raise FileExistsError("preserve existing plan; choose a new revision")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    return value


def load_plan(path: Path) -> dict[str, Any]:
    p = read(path)
    if (
        p.get("schema") != SCHEMA
        or p.get("revision") != 2
        or plan_id(p) != p.get("scoring_plan_sha256")
    ):
        raise ValueError("local scoring plan is invalid")
    for group in ("inputs", "predictions", "measurements", "sources"):
        for name, item in p[group].items():
            target = Path(item["path"])
            if not target.is_file() or sha(target) != item["sha256"]:
                raise ValueError(f"frozen {group} hash changed: {name}")
    return p


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("local prediction file exceeds frozen storage cap")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _ns(text: str) -> int:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("request timestamp lacks timezone")
    return int(dt.timestamp() * 1e9)


def join_ids(
    rows: list[dict[str, Any]], telemetry: Path
) -> tuple[dict[tuple[str, str, int, str], str], dict[str, int]]:
    req: dict[str, tuple[tuple[str, str, str], int]] = {}
    gen: dict[str, tuple[str, str, str]] = {}
    for span in _jsonl(telemetry):
        name, a = span.get("name"), span.get("attributes", {})
        if name not in {"request.start", "model.generate"} or not isinstance(a, dict):
            continue
        rid, case, q, kind = (
            a.get("tabcomplete.request_id"),
            a.get("tabcomplete.case_id"),
            a.get("tabcomplete.quantization"),
            a.get("tabcomplete.cache.condition"),
        )
        if (
            not isinstance(rid, str)
            or not isinstance(case, str)
            or not isinstance(q, str)
            or not isinstance(kind, str)
            or not isinstance(span.get("start_time_unix_nano"), int)
        ):
            continue
        target = req if name == "request.start" else gen
        if rid in target:
            raise ValueError("duplicate telemetry request ID")
        span_key = (case, q, kind)
        if name == "request.start":
            req[rid] = (span_key, span["start_time_unix_nano"])
        else:
            gen[rid] = span_key
    if set(req) != set(gen) or any(req[k][0] != gen[k] for k in req):
        raise ValueError("telemetry request and generation spans do not pair")
    buckets: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
    for rid, (key, stamp) in req.items():
        buckets[key].append((rid, int(stamp)))
    joined: dict[tuple[str, str, int, str], str] = {}
    used: set[str] = set()
    missing = ambiguous = unmatched = 0
    for row in rows:
        p, case, rep, kind = (
            row.get("precision"),
            row.get("case_id"),
            row.get("repetition"),
            row.get("request_kind"),
        )
        if (
            not isinstance(p, str)
            or not isinstance(case, str)
            or not isinstance(rep, int)
            or isinstance(rep, bool)
            or not isinstance(kind, str)
            or kind not in KINDS
            or q is None
        ):
            raise ValueError("malformed local request identity")
        q = {"q4_k_m": "Q4_K_M", "q8_0": "Q8_0"}.get(p)
        if q is None:
            raise ValueError("unknown local precision")
        if not isinstance(row.get("request_started_at_utc"), str):
            missing += 1
            continue
        anchor = _ns(row["request_started_at_utc"])
        options = sorted(
            (abs(stamp - anchor), rid)
            for rid, stamp in buckets[(case, q, kind)]
            if rid not in used and abs(stamp - anchor) <= 1_000_000_000
        )
        if not options:
            unmatched += 1
            continue
        if len(options) > 1 and options[0][0] == options[1][0]:
            ambiguous += 1
            continue
        used.add(options[0][1])
        row_key = (p, case, rep, kind)
        joined[row_key] = options[0][1]
    return joined, {
        "rows": len(rows),
        "matched": len(joined),
        "missing_timestamp": missing,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
        "telemetry_requests": len(req),
        "unmatched_telemetry": len(req) - len(used),
    }


def adapt(
    row: dict[str, Any],
    prompt: dict[str, Any],
    precision: str,
    model_sha: str,
    local_plan_sha: str,
    request_id: str | None,
) -> dict[str, Any]:
    if (
        row.get("case_id") != prompt["case_id"]
        or row.get("precision") != precision
        or row.get("model_sha256") != model_sha
        or row.get("plan_sha256") != local_plan_sha
        or row.get("prompt_sha256") != prompt["prompt_sha256"]
    ):
        raise ValueError("local result identity differs from frozen prompt/model")
    text = row.get("raw_output")
    if not isinstance(text, str) or base.digest_bytes(text.encode()) != row.get("output_sha256"):
        raise ValueError("raw response bytes/hash mismatch")
    tokens = row.get("input_tokens")
    if (
        not isinstance(tokens, int)
        or isinstance(tokens, bool)
        or tokens + OUTPUT_TOKENS > CONTEXT_TOKENS
    ):
        raise ValueError("local input exceeds frozen total context")
    terminal, cap, mapping = (
        row.get("actual_terminal_observed"),
        row.get("hit_output_cap"),
        row.get("output_file_mapping"),
    )
    if not isinstance(terminal, bool) or not isinstance(cap, bool) or not isinstance(mapping, dict):
        raise ValueError("local row lacks terminal/cap/region mapping evidence")
    return {
        "case_id": prompt["case_id"],
        "repetition": row["repetition"],
        "raw_output": text,
        "output_sha256": row["output_sha256"],
        "context_eligible": True,
        "explicit_terminal": terminal,
        "finish_reason": row.get("finish_reason"),
        "hit_token_cap": cap,
        "editable_range_mapping": mapping,
        "completed_response_ms": row.get("completed_response_ms"),
        "input_tokens": tokens,
        "output_tokens": row.get("output_token_count"),
        "observability_ids": (
            {"case_id": prompt["case_id"], "request_id": request_id} if request_id else None
        ),
    }


def score(args: argparse.Namespace, p: dict[str, Any]) -> Path:
    plan = base.load_frozen_plan(base.resolve_recorded_path(p["inputs"]["base_plan"]["path"]))
    campaign = base._load_plan(base.resolve_recorded_path(plan["campaign"]["path"]))
    cases = {c.id: c for c in base.load_next_edit_suite(base.NEXT_EDIT_SUITE_DEFAULT)}
    diagnostic = {x["case_id"]: x for x in read(base.DIAGNOSTIC_PLAN_DEFAULT)["fixtures"]}
    prompts = base._read_jsonl(base.NEXT_EDIT_INPUTS_DEFAULT)
    if [x["case_id"] for x in prompts] != campaign["comparison"]["next_edit"]["fixture_ids"]:
        raise ValueError("controller prompt order changed")
    raw, all_rows = {}, []
    for q in PRECISIONS:
        rows = _jsonl(Path(p["predictions"][q]["path"]))
        by = {(r["case_id"], r["repetition"], r["request_kind"]): r for r in rows}
        expect = {(x["case_id"], n, k) for x in prompts for n in range(2) for k in KINDS}
        if len(rows) != 96 or set(by) != expect:
            raise ValueError(f"{q} replay coverage mismatch")
        raw[q] = by
        all_rows.extend(rows)
    ids, id_summary = join_ids(all_rows, Path(p["inputs"]["telemetry"]["path"]))
    out = args.output.resolve()
    ssd = Path("/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1").resolve()
    if ssd not in out.parents or out.exists():
        raise ValueError("choose a new private SSD output directory")
    out.mkdir(parents=True, mode=0o700)
    os.environ.update(
        {
            "TABCOMPLETE_OBSERVABILITY_ENABLED": "1",
            "TABCOMPLETE_OBSERVABILITY_MODE": "offline",
            "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE": str(out / "observability.jsonl"),
            "TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT": "0",
        }
    )
    scored, summaries = {}, {}
    with base.run_scope(out / "observability-run.json", "sweep-local-replay-score") as ctx:
        run = ctx or base.RunContext.new(campaign_id="sweep-local-replay-score")
        for q in PRECISIONS:
            model_sha = plan["models"][f"{q}_sha256"]
            primary = []
            repeat_checks = []
            for i, prompt in enumerate(prompts):
                case = prompt["case_id"]
                fixture = diagnostic[case]
                src = cases.get(fixture.get("source_case_id"))
                adapted = (
                    base._adapt_source_case(
                        src, prompt, fixture, plan["sandbox"]["image_ids_by_tag"]
                    )
                    if src
                    else None
                )
                for rep in range(2):
                    for kind in KINDS:
                        r = raw[q][(case, rep, kind)]
                        request_id = ids.get((q, case, rep, kind))
                        ar = adapt(r, prompt, q, model_sha, p["local_plan_sha256"], request_id)
                        if kind == KINDS[0]:
                            result = base._score_next_edit_result(
                                row=ar,
                                prompt_row=prompt,
                                fixture=fixture,
                                source_case=adapted,
                                precision=q,
                                repetition=rep,
                                case_index=i,
                                temp_root=out / "sandbox",
                                run_context=run,
                            )
                            result.update(
                                prompt_sha256=prompt["prompt_sha256"],
                                request_id_join="matched" if request_id else "unknown",
                            )
                            primary.append(result)
                            _append(out / q / "changed_state.jsonl", result)
                        else:
                            good = base._row_terminal_valid(ar)
                            mapping = (
                                base.map_full_file(
                                    prompt["current_content"],
                                    r["raw_output"],
                                    prompt["editable_start_byte"],
                                    prompt["editable_end_byte"],
                                )
                                if good
                                else {
                                    "mapping": "incomplete_or_unterminated_output_not_scored",
                                    "out_of_range": None,
                                }
                            )
                            if mapping != ar["editable_range_mapping"]:
                                raise ValueError("repeat mapping mismatch")
                            repeat_checks.append(
                                {
                                    "case_id": case,
                                    "repetition": rep,
                                    "request_id": request_id,
                                    "prompt_sha256": prompt["prompt_sha256"],
                                    "response_sha256": r["output_sha256"],
                                    "predicted_action": mapping.get("action") if good else None,
                                    "terminal_valid": good,
                                }
                            )
            primary_by = {(r["case_id"], r["repetition"]): r for r in primary}
            for r in repeat_checks:
                main = primary_by[(r["case_id"], r["repetition"])]
                r["response_identical"] = r["response_sha256"] == main["response_sha256"]
                r["action_identical"] = r["predicted_action"] == main["predicted_action"]
                _append(out / q / "same_prompt_repeats.jsonl", r)
            summaries[q] = base._summarize_next_edit(primary)
            summaries[q]["same_prompt_repeat_consistency"] = {
                "denominator": len(repeat_checks),
                "identical_responses": sum(r["response_identical"] for r in repeat_checks),
                "identical_actions": sum(r["action_identical"] for r in repeat_checks),
                "quality_samples": 0,
            }
            scored[q] = primary
            _write(out / q / "summary.json", summaries[q])
    paired = {}
    for metric, intent in (
        ("terminal_valid", None),
        ("exact_reference_match", "clear_functional_intent"),
        ("clear_intent_functional_success", "clear_functional_intent"),
        ("no_edit_agreement", "no_edit_plausible"),
        ("exact_reference_match", "ambiguous_requirement"),
        ("synthetic_deletion_reference_match", "explicit_synthetic_target"),
    ):
        paired[f"{intent or 'all'}/{metric}"] = base._pairwise_bootstrap(
            base._case_metric(scored["q8_0"], metric, intent),
            base._case_metric(scored["q4_k_m"], metric, intent),
        )
    _write(
        out / "summary.json",
        {
            "schema": "sweep-local-replay-score-v1",
            "state": "complete",
            "quality_evidence": True,
            "interpretation": "fixed benchmark evidence, not human acceptance/general quality",
            "scoring_plan_sha256": p["scoring_plan_sha256"],
            "request_id_recovery": id_summary,
            "changed_state": summaries,
            "paired_q4_minus_q8": paired,
            "same_prompt_repeat_is_separate": True,
            "raw_prediction_text_copied": False,
            "automatic_personalization": False,
        },
    )
    return out


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-plan", action="store_true")
    group.add_argument("--score", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.freeze_plan:
        value = freeze(args.run_dir, args.plan, args.output)
        print(
            json.dumps(
                {
                    "state": "plan_frozen",
                    "identity": value["scoring_plan_sha256"],
                    "raw_plan_sha256": sha(args.plan),
                    "prediction_text_read": False,
                },
                sort_keys=True,
            )
        )
    else:
        result = score(args, load_plan(args.plan))
        print(json.dumps({"state": "complete", "output": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
