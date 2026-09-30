import importlib.util
import json
from pathlib import Path

import pytest

from tinycomplete.one_line.contract import EditAction, EditState, apply_action, physical_lines

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/select_commit_sequence_review_queue.py"
SPEC = importlib.util.spec_from_file_location("commit_sequence_review_queue", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def _row(
    *,
    identifier: str,
    repo: str,
    split: str,
    language: str = "python",
    action_kind: str = "replace_line",
    priority_context: str = "typed",
) -> dict[str, object]:
    tail = identifier.rsplit("-", maxsplit=1)[-1]
    variant = int(tail) + 1 if tail.isdigit() else 31
    marker = ("!" if split == "train" else "?") * variant
    extension = {"python": "py", "typescript": "ts", "go": "go", "rust": "rs"}[language]
    if language == "python":
        source = (
            f'def parse(value: int) -> int:\n    key = "{marker}"\n'
            "    # preserve the contract\n    return value\n"
        )
        target = 3
        history = [{"row": 0, "old_text": "old", "new_text": "new"}]
    elif language == "typescript":
        source = (
            f'function parse(value: number): number {{\n  const key = "{marker}";\n'
            "  // preserve the contract\n  return value;\n}\n"
        )
        target = 3
        history = [{"row": 0, "old_text": "old", "new_text": "new"}]
    elif language == "go":
        source = (
            f'func parse(value int) int {{\n    key := "{marker}"\n'
            "    // preserve the contract\n    return value\n}\n"
        )
        target = 3
        history = [{"row": 0, "old_text": "old", "new_text": "new"}]
    else:
        source = (
            f'fn parse(value: i32) -> i32 {{\n    let key = "{marker}";\n'
            "    // preserve the contract\n    value\n}\n"
        )
        target = 3
        history = [{"row": 0, "old_text": "old", "new_text": "new"}]
    text = "    return value + 1" if language == "python" else "  return value + 1;"
    if action_kind == "delete_line":
        action: dict[str, object] = {"kind": action_kind, "text": None}
    else:
        action = {"kind": action_kind, "text": text}
    result: dict[str, object] = {
        "id": identifier,
        "split": split,
        "state": {
            "file_id": f"src/{identifier}.{extension}",
            "filetype": language,
            "source": source,
            "target_row": target,
            "cursor_col": 0,
            "history": history,
        },
        "action": action,
        "after_source": source,
        "prompt": "source-only prompt",
        "source_repo": repo,
        "source_aliases": [],
        "source_revision": "a" * 40,
        "source_license": "mit",
        "session_or_commit": identifier,
        "generator_family": f"family/{repo}",
        "template_id": f"template/{identifier}",
        "provenance": {
            "public_url": f"https://github.com/{repo}/commit/{'a' * 40}",
            "source_record_sha256": "b" * 64,
            "history_origin_sha256": "c" * 64,
            "committed_child_sha256": "d" * 64,
        },
        "_priority_context": priority_context,
    }
    result["after_source"] = apply_action(
        EditState.from_mapping(result["state"]), EditAction(**action)
    )
    return result


def test_reserved_rows_are_skipped_before_json_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "candidates.jsonl"
    train = _row(identifier="train-1", repo="owner/train", split="train")
    reserved_malformed = b'{"split": "test_new_repo", deliberately-not-json\n'
    path.write_bytes(
        json.dumps(train, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        + reserved_malformed
    )
    original_loads = queue.json.loads

    def guarded_loads(value: object, *args: object, **kwargs: object) -> object:
        if isinstance(value, bytes) and b'test_new_repo' in value:
            raise AssertionError("reserved record reached the JSON decoder")
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(queue.json, "loads", guarded_loads)
    rows, counts = queue._read_eligible_rows(path)
    assert [row["id"] for row in rows] == ["train-1"]
    assert counts["reserved_rows_skipped_before_json_decode"] == 1


def test_queue_is_group_unique_stratified_and_exports_metadata_only():
    rows = []
    for split, per_stratum in (("train", 9), ("development", 4)):
        for language in queue.LANGUAGE_ORDER:
            for action in queue.ACTION_ORDER:
                for index in range(per_stratum):
                    rows.append(
                        _row(
                            identifier=f"{split}-{language}-{action}-{index}",
                            repo=f"owners/{split}-{language}-{action}-{index}",
                            split=split,
                            language=language,
                            action_kind=action,
                        )
                    )
    # A second candidate from one upstream repo is the same source group.
    rows.append(
        _row(
            identifier="train-python-duplicate",
            repo="owners/train-python-replace_line-0",
            split="train",
            language="python",
        )
    )
    selected, summary = queue.select_review_queue(rows)
    assert len(selected) == 128
    assert len({row["source_group_id"] for row in selected}) == 128
    assert summary["split_counts"] == {"development": 32, "train": 96}
    language_counts = list(summary["language_counts"].values())
    assert len(language_counts) == 4
    assert sum(language_counts) == 128
    assert max(language_counts) - min(language_counts) <= 3
    assert summary["all_training_accepted_false"] is True
    assert all("source" not in row and "after_source" not in row for row in selected)
    assert all("action_text" not in row for row in selected)
    assert all(
        row["source_file_license_status"] == "unverified_at_exact_revision" for row in selected
    )


def test_connected_source_group_cannot_cross_train_and_development():
    rows = [
        _row(identifier="train", repo="owners/shared", split="train"),
        _row(identifier="dev", repo="owners/shared", split="development"),
    ]
    with pytest.raises(ValueError, match="crosses or enters a reserved split"):
        queue.select_review_queue(rows)


def test_insert_before_eof_uses_valid_physical_line_boundary():
    row = _row(
        identifier="eof-insert",
        repo="owners/eof-insert",
        split="train",
        action_kind="insert_before",
    )
    state = row["state"]
    state["target_row"] = len(physical_lines(state["source"].encode("utf-8")))
    row["after_source"] = apply_action(
        EditState.from_mapping(state), EditAction(**row["action"])
    )
    selected, _ = queue.select_review_queue([row])
    assert len(selected) == 1
    assert selected[0]["action_kind"] == "insert_before"
    assert selected[0]["target_row"] == 4


def test_canonical_eof_boundaries_preserve_crlf_and_empty_file_cases():
    cases = (
        (
            "rust",
            "fn f(value: i32) -> i32 {\r\n    value\r\n}",
            "// eof",
        ),
        ("python", "", "# first line"),
    )
    for index, (language, source, inserted) in enumerate(cases):
        row = _row(
            identifier=f"eof-case-{index}",
            repo=f"owners/eof-case-{index}",
            split="train",
            language=language,
            action_kind="insert_before",
        )
        state = row["state"]
        state["source"] = source
        state["target_row"] = len(physical_lines(source.encode("utf-8")))
        state["history"] = []
        row["action"] = {"kind": "insert_before", "text": inserted}
        row["after_source"] = apply_action(
            EditState.from_mapping(state), EditAction(**row["action"])
        )
        selected, _ = queue.select_review_queue([row])
        assert len(selected) == 1
        assert selected[0]["target_row"] == len(physical_lines(source.encode("utf-8")))


def test_frozen_plan_must_precede_selection_and_binds_pool_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_plan = tmp_path / "source-plan.json"
    source_manifest = tmp_path / "source.json"
    parent_plan = tmp_path / "parent-plan.json"
    parent_result = tmp_path / "parent-result.json"
    pool = tmp_path / "candidates.jsonl"
    output = tmp_path / "queue"
    plan_path = tmp_path / "frozen-plan.json"
    row = _row(identifier="train-1", repo="owners/train", split="train")
    pool.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    pool_hash, pool_bytes = queue.sha256_file(pool)
    source_manifest.write_text(
        json.dumps(
            {
                "dataset_revision": "fc56fe33c030c6daa414c2b112c932b8eed085e6",
                "files": [{"path": "data/python/data.jsonl", "size": 1, "sha256": "b" * 64}],
            }
        )
    )
    source_plan.write_text(
        json.dumps(
            {
                "dataset": "bigcode/commitpackft",
                "revision": "fc56fe33c030c6daa414c2b112c932b8eed085e6",
            }
        )
    )
    parent_plan.write_text(json.dumps({"seed": "old"}))
    parent_result.write_text(
        json.dumps(
            {
                "candidate_sha256": pool_hash,
                "candidate_bytes": pool_bytes,
                "split_audit": {"rows": 1, "groups": 1, "splits": {"train": 1}},
            }
        )
    )
    monkeypatch.setattr(queue, "SOURCE_PLAN", source_plan)
    monkeypatch.setattr(queue, "SOURCE_MANIFEST", source_manifest)
    monkeypatch.setattr(queue, "SEQUENCE_PLAN", parent_plan)
    monkeypatch.setattr(queue, "SEQUENCE_RESULT", parent_result)
    monkeypatch.setattr(queue, "POOL", pool)
    monkeypatch.setattr(queue, "OUTPUT_DIR", output)
    monkeypatch.setattr(queue, "PLAN_PATH", plan_path)
    with pytest.raises(ValueError, match="freeze the review queue plan"):
        queue._execute()
    frozen = queue._write_plan()
    assert frozen["candidate_pool_sha256"] == pool_hash
    assert frozen["reserved_splits_skipped_before_json_decode"] == [
        "test_new_repo",
        "test_new_mechanism",
    ]
    result = queue._execute()
    assert result["accepted_training"] == 0
    assert result["quality_evidence"] is False
    assert json.loads((output / "manifest.json").read_text())["summary"]["selected_rows"] == 1
    pool.write_text(pool.read_text() + "\n")
    with pytest.raises(ValueError, match="pinned candidate artifact"):
        queue._load_frozen_plan()
