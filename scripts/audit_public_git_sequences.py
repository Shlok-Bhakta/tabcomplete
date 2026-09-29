"""Audit unreviewed public Git sequences without accepting training labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from tinycomplete.one_line.context import serialize_state_bounded
from tinycomplete.one_line.contract import EditAction, EditState, apply_action, encode_action


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()  # noqa: S324


def _pinned_raw(repo: str, rev: str, path: str, *, maximum_bytes: int) -> bytes:
    url = f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "tabcomplete-public-git-audit"})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise ValueError("pinned public file exceeds audit byte cap")
    return raw


def audit_row(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    from tree_sitter_language_pack import get_parser

    state = EditState.from_mapping(row["state"])
    action = EditAction(**row["action"])
    provenance = row["provenance"]
    child = _pinned_raw(
        row["source_repo"], provenance["child_commit"], provenance["path"], maximum_bytes=262144
    )
    license_raw = _pinned_raw(
        row["source_repo"],
        provenance["child_commit"],
        provenance["license_path"],
        maximum_bytes=262144,
    )
    child_blob_matches = _git_blob_oid(child) == provenance["child_git_blob"]
    license_matches = _sha(license_raw) == provenance["license_sha256"]
    applied = apply_action(state, action)
    action_matches = applied == row["after_source"]
    context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
    wire = encode_action(action)
    output_tokens_including_eos = len(tokenizer.encode(wire, add_special_tokens=False)) + 1
    parser = get_parser(state.filetype)
    pre_parse = not parser.parse(state.source.encode()).root_node.has_error
    post_parse = not parser.parse(applied.encode()).root_node.has_error
    target_absent_from_prompt = wire not in context.text
    return {
        "candidate_id": row["id"],
        "repository": row["source_repo"],
        "source_revision": row["source_revision"],
        "filetype": state.filetype,
        "source_file_sha256": _sha(child),
        "source_git_blob_matches": child_blob_matches,
        "license_sha256_matches": license_matches,
        "action_replays_to_recorded_after": action_matches,
        "input_tokens": context.input_tokens,
        "included_history_count": context.included_history,
        "output_tokens_including_eos": output_tokens_including_eos,
        "output_within_64_token_cap": output_tokens_including_eos <= 64,
        "pre_state_parses": pre_parse,
        "post_action_state_parses": post_parse,
        "wire_action_absent_from_prompt": target_absent_from_prompt,
        "editor_order_observed": provenance["editor_edit_order_observed"],
        "intermediate_state_observed": provenance["intermediate_buffer_observed"],
        "blind_inferability_reviewed": False,
        "independent_objective_tested": False,
        "accepted_training_label": False,
    }


def run(plan_path: Path, candidates_path: Path, output_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    plan = json.loads(plan_path.read_text())
    tokenizer_path = Path(plan["existing_model_files"]["tokenizer.json"]["path"]).parent
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    rows = [json.loads(line) for line in candidates_path.read_text().splitlines() if line]
    audits = [audit_row(row, tokenizer) for row in rows]
    result = {
        "candidate_artifact": str(candidates_path),
        "candidate_artifact_sha256": _sha(candidates_path.read_bytes()),
        "tokenizer_revision": tokenizer_path.name,
        "audits": audits,
        "mechanically_verified_count": sum(
            all(
                audit[key]
                for key in (
                    "source_git_blob_matches",
                    "license_sha256_matches",
                    "action_replays_to_recorded_after",
                    "output_within_64_token_cap",
                    "pre_state_parses",
                    "post_action_state_parses",
                    "wire_action_absent_from_prompt",
                )
            )
            for audit in audits
        ),
        "accepted_training_label_count": 0,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-plan", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.campaign_plan, args.candidates, args.output)["mechanically_verified_count"])
