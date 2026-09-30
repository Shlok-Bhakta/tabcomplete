#!/usr/bin/env python3
"""Freeze two public-source synthetic next-edit cases before teacher calls."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinycomplete.one_line.author_protocol_v3 import build_author_prompt_v3
from tinycomplete.one_line.contract import EditState, RecentEdit
from tinycomplete.one_line.pilot_roles import build_blind_solver_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/mnt/ssd/tabcomplete-product-r2/next-pilot-frozen-v7")
SOURCE_PACKET = Path(
    "/mnt/ssd/tabcomplete-product-r2/commitpackft/next-pilot-review-v1/source_only_inputs.jsonl"
)
SOURCE_AUDIT = Path("/mnt/ssd/tabcomplete-product-r2/commitpackft/source-verification-v1")
REVIEW_INDEX = Path(
    "/mnt/ssd/tabcomplete-product-r2/commitpackft/authoring-queue-v4/review_index.jsonl"
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _write_private(path: Path, value: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite frozen artifact")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _load_pilot() -> Any:
    spec = importlib.util.spec_from_file_location(
        "public_mechanism_pilot", ROOT / "scripts/run_public_mechanism_pilot.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("existing pilot runner could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case_specs() -> dict[str, dict[str, Any]]:
    return {
        "synthetic-a": {
            "source_case": "seed-01",
            "repo": "10se1ucgo/cassiopeia",
            "path": "examples/match.py",
            "parent": "14d51aa701dcc8d1d3f026af947c935abb0eabe3",
            "child": "8556fc0b6fb024ab6cc68364270462681209108a",
            "parent_source_sha256": (
                "6bcdc4567f658e6bf9c02f759f769ce82d589e71e527fc4742755fb88707705b"
            ),
            "root_license_sha256": (
                "3a59959e531297ff6d391dee036e42f0b1a27048edac87019c3ab9515498e536"
            ),
            "history": {
                "row": 4,
                "old_text": "def print_summoner():",
                "new_text": "def print_summoner(name: str, id: int):",
            },
            "target_row": 5,
            "filetype": "python",
            "goal": (
                "Pass both current function parameters to the Summoner constructor. A fake "
                "constructor must receive the supplied synthetic name and ID. Preserve the "
                "existing match output behavior."
            ),
            "objective": {
                "kind": "argument_forwarding",
                "description": (
                    "The function uses its visible name and id parameters when constructing "
                    "Summoner."
                ),
                "checks": [
                    "Call with synthetic values and verify that both reach the fake constructor.",
                    "Preserve the existing match ID output.",
                ],
            },
            "visible_intent_cue": (),
            "gold_action": {
                "kind": "replace_line",
                "text": "    me = Summoner(name=name, id=id)",
            },
            "wrong_controls": [
                {
                    "name": "no_edit",
                    "action": {"kind": "keep"},
                },
                {
                    "name": "unchanged_redacted_literals",
                    "action": {
                        "kind": "replace_line",
                        "text": '    me = Summoner(name="<redacted-name>", id=0)',
                    },
                },
                {
                    "name": "partial_name_only",
                    "action": {
                        "kind": "replace_line",
                        "text": "    me = Summoner(name=name, id=0)",
                    },
                },
                {
                    "name": "partial_id_only",
                    "action": {
                        "kind": "replace_line",
                        "text": '    me = Summoner(name="<redacted-name>", id=id)',
                    },
                },
            ],
            "oracle_source": """import runpy
import sys
import types

captured = []

class Match:
    id = "synthetic-match"
    participants = ()

class FakeSummoner:
    def __init__(self, *args, **kwargs):
        captured.append((args, kwargs))
        self.matches = [Match()]

package = types.ModuleType("cassiopeia")
package.__path__ = []
core = types.ModuleType("cassiopeia.core")
core.Summoner = FakeSummoner
sys.modules["cassiopeia"] = package
sys.modules["cassiopeia.core"] = core
module = runpy.run_path("solution.py", run_name="synthetic_case")
module["print_summoner"]("synthetic-user", 731)
assert len(captured) == 1
args, kwargs = captured[0]
assert (args == ("synthetic-user", 731) and kwargs == {}) or (
    args == () and kwargs == {"name": "synthetic-user", "id": 731}
)
""",
            "runtime_image": (
                "docker.io/library/python:3.12-slim@sha256:"
                "2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9"
            ),
            "runtime_command": ["python3", "oracle.py"],
            "expected_stdout": "synthetic-match\n",
        },
        "synthetic-b": {
            "source_case": "seed-05",
            "repo": "OmniSharp/omnisharp-atom",
            "path": "lib/omnisharp-atom/views/tooltip-view.ts",
            "parent": "58ffed41f525c979651ec329b30c726eec120481",
            "child": "649361b1d4d963e301ae561dacf9906efd0af76a",
            "parent_source_sha256": (
                "85dcf3906393521abadfe7ef06b710e8a7ded2cc33664891036a25ec75bd64d5"
            ),
            "root_license_sha256": (
                "7a684aca08742d0944c7fb0e63c08a2b6528422e12765b1454e27e321449eadf"
            ),
            "history": {
                "row": 32,
                "old_text": "        var offset = 10;",
                "new_text": "        var offset = 12;",
            },
            "target_row": 48,
            "filetype": "typescript",
            "goal": (
                "In the existing Y-axis overflow branch, use the shared offset as a lower bound "
                "for the computed top. Preserve the computed above-anchor position whenever it "
                "already meets that lower bound."
            ),
            "objective": {
                "kind": "tooltip_inset_clamp",
                "description": (
                    "The overflow-above fallback does not place the tooltip above the shared "
                    "inset and keeps a fitting above-anchor position."
                ),
                "checks": [
                    "With offset 12, anchor top 5 and tooltip height 20 yields top 12.",
                    "With offset 12, anchor top 40 and tooltip height 20 yields top 20.",
                ],
            },
            "visible_intent_cue": (
                "Synthetic task context: Keep the tooltip within the viewport inset and "
                "preserve its above-anchor placement whenever that placement fits.",
            ),
            "gold_action": {
                "kind": "replace_line",
                "text": (
                    "            top = Math.max(offset, this.rect.top - this[0].offsetHeight)"
                ),
            },
            "wrong_controls": [
                {
                    "name": "no_edit",
                    "action": {"kind": "keep"},
                },
                {
                    "name": "unchanged_unclamped_computation",
                    "action": {
                        "kind": "replace_line",
                        "text": "            top = this.rect.top - this[0].offsetHeight",
                    },
                },
                {
                    "name": "always_use_inset",
                    "action": {"kind": "replace_line", "text": "            top = offset"},
                },
            ],
            "oracle_source": """const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const ts = require("/usr/local/lib/node_modules/typescript");
const text = fs.readFileSync("tooltip-view.ts", "utf8");
const result = ts.transpileModule(text, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2019 },
});
assert.equal(
  (result.diagnostics || []).filter((item) => item.category === ts.DiagnosticCategory.Error).length,
  0,
);
class View {
  constructor() {
    this[0] = { offsetWidth: 30, offsetHeight: 20 };
    this.styles = {};
  }
  css(value) { Object.assign(this.styles, value); }
}
const spacePen = { View };
function jquery() {
  return { append() {}, width() { return 100; }, height() { return 100; } };
}
const mockModule = { exports: {} };
const document = { body: {} };
vm.runInNewContext(result.outputText, {
  module: mockModule,
  exports: mockModule.exports,
  document,
  require(name) {
    if (name === "atom-space-pen-views") return spacePen;
    if (name === "jquery") return jquery;
    throw new Error("unexpected import");
  },
});
const TooltipView = mockModule.exports;
const near = new TooltipView({ left: 5, right: 10, top: 5, bottom: 95 });
assert.equal(near.styles.top, 12);
const fits = new TooltipView({ left: 5, right: 10, top: 40, bottom: 95 });
assert.equal(fits.styles.top, 20);
""",
            "runtime_image": (
                "localhost/tabcomplete-typescript-bench:5.9.2@sha256:"
                "3cc808896e2be1342f6fc82b12be5f6371b9b13b6e70ef92489ebaa34eff5b73"
            ),
            "runtime_command": ["node", "oracle.js"],
            "expected_stdout": "",
        },
    }


def main() -> None:
    OUT.mkdir(mode=0o700, parents=False)
    os.chmod(OUT, 0o700)
    (OUT / "prompts").mkdir(mode=0o700)
    (OUT / "oracle").mkdir(mode=0o700)
    pilot = _load_pilot()
    source_rows = {
        row["case_id"]: row
        for row in (json.loads(line) for line in SOURCE_PACKET.read_text().splitlines())
    }
    tokenizer = pilot._install_tokenizer()
    specs = _case_specs()
    input_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    prompt_manifest: dict[str, Any] = {}
    case_manifest: dict[str, Any] = {}

    for case_id, spec in specs.items():
        parent_source = source_rows[spec["source_case"]]["source_text"]
        parent_sha = _sha(parent_source.encode())
        expected_sha = (
            "636c77f1500ffabcb22ae50c3db4f7678888647c76194cc1879c48207f773378"
            if case_id == "synthetic-a"
            else spec["parent_source_sha256"]
        )
        if parent_sha != expected_sha:
            raise ValueError("selected source snapshot hash mismatch")
        history = spec["history"]
        author_source = parent_source
        transform: dict[str, Any] | None = None
        if case_id == "synthetic-a":
            synthetic_lines = parent_source.splitlines(keepends=True)
            snapshot_line = synthetic_lines[history["row"]]
            snapshot_text = snapshot_line.removesuffix("\n").removesuffix("\r")
            ending = snapshot_line[len(snapshot_text) :]
            if snapshot_text != history["new_text"]:
                raise ValueError("seed A source line differs from its pinned snapshot")
            synthetic_lines[history["row"]] = history["old_text"] + ending
            author_source = "".join(synthetic_lines)
            transform = {
                "kind": "synthetic_single_line_prestate_from_public_source",
                "row": history["row"],
                "public_snapshot_line_sha256": _sha(history["new_text"].encode()),
                "synthetic_author_line_sha256": _sha(history["old_text"].encode()),
                "purpose": "make the declared synthetic history replay to pinned source bytes",
            }
        spec["author_source_transform"] = transform
        spec["source_parent_snapshot_sha256"] = parent_sha
        parent_lines = author_source.splitlines(keepends=True)
        parent_line = parent_lines[history["row"]]
        parent_line_text = parent_line.removesuffix("\n").removesuffix("\r")
        ending = parent_line[len(parent_line_text) :]
        if parent_line_text != history["old_text"]:
            raise ValueError("synthetic history does not match source at its declared row")
        parent_lines[history["row"]] = history["new_text"] + ending
        source = "".join(parent_lines)
        source_sha = _sha(source.encode())
        spec["state_source_sha256"] = source_sha
        lines = source.splitlines()
        if lines[history["row"]] != history["new_text"]:
            raise ValueError("synthetic history new text differs from current state")
        state = EditState(
            file_id=f"{spec['repo']}/{spec['path']}",
            filetype=spec["filetype"],
            source=source,
            target_row=spec["target_row"],
            cursor_col=0,
            history=(RecentEdit(**history),),
            relevant=tuple(spec["visible_intent_cue"]),
        )
        physical = source.splitlines(keepends=True)
        reversed_lines = list(physical)
        reversed_lines[history["row"]] = history["old_text"] + ending
        replayed_lines = list(reversed_lines)
        replayed_lines[history["row"]] = history["new_text"] + ending
        if "".join(replayed_lines) != source:
            raise ValueError("synthetic edit history does not replay byte-for-byte")

        source_row = {
            "id": case_id,
            "student_state_seed": {
                "file_id": state.file_id,
                "filetype": state.filetype,
                "source": author_source,
            },
            "authoring_metadata": {
                "source_sha256": _sha(author_source.encode()),
                "source_repo": spec["repo"],
                "source_aliases": [spec["repo"]],
                "source_revision": spec["parent"],
                "source_path": spec["path"],
                "license_sha256": spec["root_license_sha256"],
                "source_license": "MIT",
                "source_provenance_verified": True,
                "authoring_focus": (
                    "argument propagation"
                    if case_id == "synthetic-a"
                    else "literal or boundary correction"
                ),
            },
        }
        author = build_author_prompt_v3(source_row)
        author_constraints = (
            "This is a deliberately synthetic source-grounded task, not a reconstruction of "
            "actual editor chronology. Use exactly this prior edit: row "
            f"{history['row']} changes from {json.dumps(history['old_text'])} to "
            f"{json.dumps(history['new_text'])}. Set target_row to {spec['target_row']}. "
            f"Synthetic goal: {spec['goal']} Return one candidate matching the JSON schema. "
            "Do not claim observed user intent or whole-program correctness."
        )
        author += "\nTask-specific synthetic constraints (author only):\n" + author_constraints
        solver_prompt = build_blind_solver_prompt({"state": asdict(state)}, tokenizer)
        prompt_texts = {
            "author": author,
            "solver": solver_prompt,
        }
        prompt_manifest[case_id] = {
            role: {
                "sha256": _sha(text.encode()),
                "bytes": len(text.encode()),
                "q25_token_estimate": len(tokenizer.encode(text, add_special_tokens=True)),
            }
            for role, text in prompt_texts.items()
        }
        for role, text in prompt_texts.items():
            _write_private(OUT / "prompts" / f"{case_id}-{role}.txt", text.encode())
        input_rows.append(
            {
                "case_id": case_id,
                "source_class": "public_redacted_source",
                "source_repo": spec["repo"],
                "source_path": spec["path"],
                "source_parent_revision": spec["parent"],
                "source_parent_sha256": spec["parent_source_sha256"],
                "source_parent_snapshot_sha256": parent_sha,
                "author_source_sha256": _sha(author_source.encode()),
                "author_source_text": author_source,
                "author_source_origin": (
                    "synthetic_variant_of_pinned_public_source"
                    if transform is not None
                    else "pinned_public_parent_source"
                ),
                "author_source_transform": transform,
                "prompt_source_sha256": source_sha,
                "source_text": source,
                "state": asdict(state),
                "visible_intent_cue": list(spec["visible_intent_cue"]),
                "history_origin": "synthetic_constructed_for_task; no human chronology observed",
                "answer_oracle_or_test_included": False,
                "synthetic_visible_intent_included": bool(spec["visible_intent_cue"]),
            }
        )
        oracle_rows.append(
            {
                "case_id": case_id,
                "state_sha256": _sha(_canonical(asdict(state))),
                "source_sha256": source_sha,
            "goal": spec["goal"],
                "objective": spec["objective"],
                "gold_action": spec["gold_action"],
                "wrong_controls": spec["wrong_controls"],
                "test_source": spec["oracle_source"],
                "runtime_image": spec["runtime_image"],
                "runtime_command": spec["runtime_command"],
                "expected_stdout": spec["expected_stdout"],
                "source_identity": {
                    "repository": spec["repo"],
                    "path": spec["path"],
                    "parent_commit": spec["parent"],
                    "child_commit": spec["child"],
                    "parent_source_sha256": spec["parent_source_sha256"],
                    "state_source_sha256": source_sha,
                    "license": "MIT",
                    "parent_and_child_root_license_sha256": spec["root_license_sha256"],
                    "file_specific_license_notice": "none observed",
                },
                "synthetic_history": history,
                "history_byte_exact_reconstruction": True,
            }
        )
        case_manifest[case_id] = {
            key: spec[key]
            for key in (
                "source_case",
                "repo",
                "path",
                "parent",
                "child",
                "parent_source_sha256",
                "state_source_sha256",
                "root_license_sha256",
                "history",
                "target_row",
                "filetype",
                "goal",
                "objective",
                "gold_action",
                "wrong_controls",
                "runtime_image",
                "visible_intent_cue",
                "author_source_transform",
                "source_parent_snapshot_sha256",
            )
        }
        case_manifest[case_id]["gold_action_sha256"] = _sha(
            _canonical(spec["gold_action"])
        )
        for control in case_manifest[case_id]["wrong_controls"]:
            control["action_sha256"] = _sha(_canonical(control["action"]))
        _write_private(
            OUT / "oracle" / f"{case_id}-oracle.{'py' if state.filetype == 'python' else 'js'}",
            spec["oracle_source"].encode(),
        )

    input_bytes = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in input_rows
    ).encode()
    oracle_bytes = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in oracle_rows
    ).encode()
    _write_private(OUT / "source_only_inputs.jsonl", input_bytes)
    _write_private(OUT / "oracle_fixtures.jsonl", oracle_bytes)

    ledger = pilot.TeacherUsageLedger(pilot.LEDGER)
    ledger_rows, totals = pilot._ledger_snapshot(ledger)
    original = json.loads((pilot.PRIOR_RUN_DIR / "preflight_plan.json").read_text())
    baseline = set(original["ledger"]["baseline_request_ids"])
    used = len(set(ledger_rows) - baseline)
    remaining = pilot.CAMPAIGN_HARD_MAX_CALLS - used
    request_ids = {
        f"{case_id}:{role}": (f"two-seed-source-grounded-v1-{role}-" + _sha(case_id.encode())[:20])
        for case_id in case_manifest
        for role in ("author", "solver", "reviewer")
    }
    if remaining < 6 or set(request_ids.values()) & set(ledger_rows):
        raise ValueError("campaign budget or request identity guard failed")

    source_manifest = SOURCE_AUDIT / "manifest.json"
    source_results = SOURCE_AUDIT / "candidate_results.jsonl"
    plan: dict[str, Any] = {
        "schema": "public-source-two-seed-synthetic-author-solver-review-v3",
        "status": "frozen_before_provider_calls",
        "created_at_unix_ns": time.time_ns(),
        "objective": (
            "Qualify whether q25-coder next-edit interactions align with two "
            "public-source-grounded "
            "synthetic tasks. No claim of original human edit chronology, broad quality, training "
            "acceptance, or model promotion."
        ),
        "source_audit": {
            "source_only_packet_sha256": _sha(SOURCE_PACKET.read_bytes()),
            "source_verification_manifest_sha256": _sha(source_manifest.read_bytes()),
            "source_verification_results_sha256": _sha(source_results.read_bytes()),
            "review_index_sha256": _sha(REVIEW_INDEX.read_bytes()),
            "license_scope_audit_v2_path": (
                "/mnt/ssd/tabcomplete-product-r2/next-pilot-license-scope-audit-v1/audit-v2.json"
            ),
            "license_scope_audit_v2_sha256": (
                "c6ba8efb36aee1b9800d16fd02043c30f56396c164884d4b0e2a3d39c0aa3f92"
            ),
            "candidate_ids": [
                "commit-sequence/61ae298749efaa9208148e50",
                "commit-sequence/665b79fac0533bc50f3859ba",
            ],
            "history_observed": False,
            "repo_license": "MIT verified at parent and child; no file-level SPDX notice observed",
            "seed01_redaction": (
                "Only redacted source sent: public account name -> <redacted-name>, "
                "numeric ID -> 0. "
                "Unredacted source bytes never enter prompts."
            ),
        },
        "cases": case_manifest,
        "input_bundle": {
            "path": str(OUT / "source_only_inputs.jsonl"),
            "sha256": _sha(input_bytes),
            "rows": len(input_rows),
            "contains_answer_oracle_or_test": False,
            "contains_synthetic_visible_intent_cue": True,
        },
        "oracle_bundle": {
            "path": str(OUT / "oracle_fixtures.jsonl"),
            "sha256": _sha(oracle_bytes),
            "rows": len(oracle_rows),
            "execution_backend": "existing pinned code_benchmark sandbox only",
            "literal_action_match_required": False,
            "wrong_controls": "semantic controls selected before provider outputs",
            "preflight_required_before_provider_calls": True,
        },
        "prompts": prompt_manifest,
        "protocol": {
            "action_wire": "single-line-edit-v1",
            "max_action_tokens_including_eos": 64,
            "solver_requires_provider_finish_stop": True,
            "solver_requires_complete_FINAL_ACTION_block": True,
            "author_builder": (
                "existing build_author_prompt_v3 plus fixed synthetic task constraints"
            ),
            "solver_builder": "existing build_blind_solver_prompt; source/history only",
            "reviewer_builder": (
                "materialized after actual author and solver outputs using existing "
                "build_reviewer_prompt; no fixed reviewer template frozen"
            ),
            "invalid_or_incomplete_responses_rejected": True,
        },
        "provider": {
            "model_id": pilot.MODEL_ID,
            "authorization_basis": pilot.AUTHORIZATION_BASIS,
            "source_class": "public",
            "route": "existing configured OpenCode Go route; no fallback",
            "tool_calls": False,
            "retries": False,
            "sequential_one_session_per_role": True,
            "role_order": ["author", "solver", "reviewer"],
            "max_calls": 6,
            "stop_after_one_new_ambiguous_failure": True,
            "max_input_tokens_per_role": 30000,
            "max_output_plus_reasoning_tokens_per_role": 8192,
            "maximum_reserved_input_tokens": 180000,
            "maximum_reserved_output_plus_reasoning_tokens": 49152,
            "request_ids": request_ids,
            "execution_gate": "parent reviews frozen plan before the first call",
        },
        "execution_dir": str(OUT / "execution-v7"),
        "budget_observation": {
            "observed_at_unix_ns": time.time_ns(),
            "ledger_path": str(pilot.LEDGER),
            "ledger_sha256": _sha(pilot.LEDGER.read_bytes()),
            "current_request_count": len(ledger_rows),
            "current_request_ids": sorted(ledger_rows),
            "current_totals": totals,
            "campaign_baseline_request_count": len(baseline),
            "campaign_baseline_request_ids_sha256": original["ledger"][
                "baseline_request_ids_sha256"
            ],
            "campaign_baseline_plan_sha256": _sha(
                (pilot.PRIOR_RUN_DIR / "preflight_plan.json").read_bytes()
            ),
            "campaign_calls_used": used,
            "campaign_call_cap": pilot.CAMPAIGN_HARD_MAX_CALLS,
            "calls_remaining_before_plan": remaining,
            "calls_planned": 6,
            "calls_remaining_after_plan": remaining - 6,
            "current_unsettled_reservations": sum(
                "input" not in row for row in ledger_rows.values()
            ),
        },
        "runtime": pilot._runtime_fingerprint(),
        "code_hashes": {
            "pilot": pilot.sha_file(ROOT / "scripts/run_public_mechanism_pilot.py"),
            "review_recovery": pilot.sha_file(
                ROOT / "scripts/run_public_mechanism_review_recovery.py"
            ),
            "author_protocol": pilot.sha_file(
                ROOT / "src/tinycomplete/one_line/author_protocol_v3.py"
            ),
            "roles": pilot.sha_file(ROOT / "src/tinycomplete/one_line/pilot_roles.py"),
            "context": pilot.sha_file(ROOT / "src/tinycomplete/one_line/context.py"),
            "contract": pilot.sha_file(ROOT / "src/tinycomplete/one_line/contract.py"),
            "teacher": pilot.sha_file(ROOT / "src/tinycomplete/one_line/teacher.py"),
            "packet_builder": pilot.sha_file(Path(__file__)),
            "oracle_preflight": pilot.sha_file(
                ROOT / "scripts/verify_two_seed_oracle_preflight.py"
            ),
            "license_scope_audit_v2": (
                "c6ba8efb36aee1b9800d16fd02043c30f56396c164884d4b0e2a3d39c0aa3f92"
            ),
        },
        "training": {
            "started": False,
            "accepted_candidates": 0,
            "personalization_enabled": False,
            "model_weights_changed": False,
        },
            "limitations": [
            "Two synthetic tasks cannot establish broad next-edit quality.",
            "Git commit history is version provenance, not observed editor chronology.",
            "Seed01 uses redacted public-account identifiers and synthetic oracle inputs.",
            "Seed01 author prestate is a declared synthetic one-line variant; it is not observed ",
            "chronology.",
            "Repository MIT licenses are verified; file-specific SPDX notices were absent.",
            "Reviewer verdict is advisory; pinned behavior is authoritative.",
            "Seed05 includes a synthetic visible-intent cue not present in the source history.",
        ],
    }
    plan["plan_sha256"] = _sha(_canonical(plan))
    plan_bytes = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    _write_private(OUT / "plan.json", plan_bytes)
    artifacts = {
        "plan.json": _sha(plan_bytes),
        "source_only_inputs.jsonl": _sha(input_bytes),
        "oracle_fixtures.jsonl": _sha(oracle_bytes),
        **{f"prompts/{path.name}": _sha(path.read_bytes()) for path in (OUT / "prompts").iterdir()},
        **{f"oracle/{path.name}": _sha(path.read_bytes()) for path in (OUT / "oracle").iterdir()},
    }
    manifest = {
        "schema": "two-seed-synthetic-pilot-artifact-manifest-v1",
        "plan_file_sha256": _sha(plan_bytes),
        "plan_canonical_sha256": plan["plan_sha256"],
        "files": artifacts,
    }
    _write_private(
        OUT / "artifact_manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "private_root": str(OUT),
                "plan_file_sha256": _sha(plan_bytes),
                "plan_canonical_sha256": plan["plan_sha256"],
                "input_sha256": _sha(input_bytes),
                "oracle_sha256": _sha(oracle_bytes),
                "manifest_sha256": _sha((OUT / "artifact_manifest.json").read_bytes()),
                "campaign_calls_used": used,
                "campaign_calls_remaining": remaining,
                "planned_calls": 6,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
