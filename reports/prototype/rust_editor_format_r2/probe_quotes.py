"""Paired, bounded public-fixture prompt check on the resident owned service."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
from measure_r2_local import NativeProvider
from tree_sitter_language_pack import get_parser

from tinycomplete.observability.runs import run_scope
from tinycomplete.one_line.contract import EditState, apply_action, decode_action

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:19196"
HINT = (
    "R/I text is raw source, not a JSON string. Decode JSON context strings before copying code. "
    "Do not add JSON quote escapes. Preserve real backslashes inside source strings."
)


def main() -> None:
    plan = json.loads((ROOT / "plan_v2.json").read_text())
    fixtures = []
    for path, expected in [
        (ROOT / "quote_states_v1.jsonl", plan["fixtures"]["quote_states_v1.jsonl"]),
        (
            ROOT.parent / "rust_editor_r1/quality_states_v1.jsonl",
            plan["fixtures"]["related_development_24"],
        ),
    ]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        fixtures.extend(json.loads(line) for line in path.read_text().splitlines())
    output = ROOT / "quote_probe_results.jsonl"
    assert not output.exists(), "New suite/plan required to rerun; preserve scientific output."
    deadline = time.monotonic() + plan["budget_seconds"]
    client = httpx.Client(timeout=30)
    identity = client.get(URL + "/health").json()
    assert identity["model_sha256"] == plan["model_sha256"]
    (ROOT / "quote_probe_runtime.json").write_text(json.dumps(identity, indent=2) + "\n")
    provider = NativeProvider(URL, "q25-format-probe")
    provider.repository_identity = "public-synthetic-format-probe-v1"
    provider.timeout_seconds = 30
    provider.cache = True
    parser = get_parser("rust")
    with run_scope(ROOT / "quote_probe_run.json", "rust-editor-format-probe") as run:
        with output.open("w") as handle:
            for row in fixtures:
                if time.monotonic() > deadline:
                    raise TimeoutError("format probe deadline reached")
                state = EditState.from_mapping(row["state"])
                prepared = client.post(
                    URL + "/v1/editor/context",
                    json={
                        "state": row["state"],
                        "repository_identity": provider.repository_identity,
                        "buffers": [],
                    },
                )
                prepared.raise_for_status()
                control = prepared.json()["prompt"]
                for layout in plan["layouts"]:
                    prompt = (
                        control
                        if layout == "cursor-last-v1"
                        else control.replace("File: ", HINT + "\nFile: ", 1)
                    )
                    tokenized = client.post(
                        URL + "/tokenize", json={"content": prompt, "add_special": True}
                    )
                    tokenized.raise_for_status()
                    length = len(tokenized.json()["tokens"])
                    if length > plan["input_budget"]:
                        raise ValueError(
                            "Hint needs a different bounded context; no silent truncation"
                        )
                    with run.for_case(row["id"] + "/" + layout).activate():
                        result = provider.generate_detailed(prompt, 64)
                    decoded = decode_action(
                        result.text,
                        terminated=result.finish_reason == "eos",
                        generated_tokens=result.tokens,
                    )
                    gold = row.get("gold_action", row.get("expected_action"))
                    predicted = (
                        None
                        if decoded.action is None
                        else {"kind": decoded.action.kind, "text": decoded.action.text}
                    )
                    syntax_ok = None
                    if decoded.action is not None and state.filetype == "rust":
                        syntax_ok = not parser.parse(
                            apply_action(state, decoded.action).encode()
                        ).root_node.has_error
                    record = {
                        "id": row["id"],
                        "layout": layout,
                        "filetype": state.filetype,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "input_tokens": length,
                        "model_sha256": identity["model_sha256"],
                        "runtime_config_hash": identity["runtime_config_hash"],
                        "raw_wire": result.text,
                        "actual_eos": result.finish_reason == "eos",
                        "codec_valid": decoded.status == "ok",
                        "predicted": predicted,
                        "exact": predicted == gold,
                        "rust_tree_sitter_parse_ok": syntax_ok,
                        "timings": provider.last["server_timings"],
                        "completed_ms": provider.last["total_seconds"] * 1000,
                        "public_synthetic": True,
                        "quality_scope": "development regression; not measured human quality",
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "id": row["id"],
                                "layout": layout,
                                "exact": record["exact"],
                                "syntax_ok": syntax_ok,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
