from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tinycomplete.code_cpt import q25_fim
from tinycomplete.one_line.train import EncodedExample, TrainingCursor


def _sha(value: str | bytes) -> str:
    encoded = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _row(identifier: int, split: str, *, new: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": identifier,
        "split": split,
        "language": "python",
        "mode": "whole_logical_line",
        "prompt_format": "psm",
        "variant": 0,
        "source_content_sha256": _sha(f"source-{split}-{identifier}"),
        "source_path": f"repo/{identifier}.py",
        "repository_identity_sha256": _sha(f"repository-{split}-{identifier}"),
        "repository_alias_sha256": [],
        "licenses": ["MIT"],
        "dataset_id": "public-scale-fixture",
        "dataset_revision": "frozen-fixture",
        "region_start": 0,
        "region_end": 4,
        "prompt_sha256": _sha(f"prompt-{split}-{identifier}"),
        "target_sha256": _sha(f"target-{split}-{identifier}"),
        "input_ids": [1, 10, 2, 11, 3, 12, 4],
        "prompt_tokens": 5,
        "target_tokens": 2,
        "total_tokens": 7,
    }
    if new:
        row["repository_group_sha256"] = _sha(f"group-{split}-{identifier}")
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> bytes:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(payload)
    return payload


def _artifact(path: Path, rows: list[dict[str, Any]]) -> tuple[bytes, dict[str, Any]]:
    payload = _write_rows(path, rows)
    return payload, {
        "file": path.name,
        "sha256": _sha(payload),
        "bytes": len(payload),
        "row_count": len(rows),
        "input_tokens": 7 * len(rows),
        "target_tokens": 2 * len(rows),
    }


def _make_scale_corpus(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = {
        "repeat_train": tmp_path / "repeat_train.jsonl",
        "scaled_train": tmp_path / "scaled_train.jsonl",
        "development_new": tmp_path / "development_new.jsonl",
        "development_previous": tmp_path / "development_previous.jsonl",
        "metadata": tmp_path / "corpus_metadata.json",
        "plan": tmp_path / "plan.json",
    }
    original = [_row(index, "train") for index in range(4096)]
    added = [_row(4096 + index, "train", new=True) for index in range(4096)]
    new_dev = [_row(8192 + index, "development", new=True) for index in range(512)]
    old_dev = [_row(4096 + index, "development") for index in range(240)]
    _, repeat_record = _artifact(paths["repeat_train"], original)
    _, scaled_record = _artifact(paths["scaled_train"], original + added)
    _, new_dev_record = _artifact(paths["development_new"], new_dev)
    _, old_dev_record = _artifact(paths["development_previous"], old_dev)

    metadata = {
        "schema": q25_fim.SCALE_CORPUS_SCHEMA,
        "raw_source_content_emitted": False,
        "sealed_test_accessed": False,
        "preparation_plan_sha256": _sha("scale-preparation"),
        "parent_cpt_plan_sha256": _sha("parent-cpt"),
        "tokenizer_id": q25_fim.MODEL_ID,
        "tokenizer_revision": q25_fim.MODEL_REVISION,
        "tokenizer_sha256": q25_fim.TOKENIZER_SHA256,
        "eos_token_id": 4,
        "fim_marker_ids": {"fim_prefix": 1, "fim_suffix": 2, "fim_middle": 3},
        "files": {
            f"{name}.jsonl": {"sha256": record["sha256"], "bytes": record["bytes"]}
            for name, record in (
                ("repeat_train", repeat_record),
                ("scaled_train", scaled_record),
                ("development_new", new_dev_record),
                ("development_previous", old_dev_record),
            )
        },
        "splits": {
            "repeat_train": repeat_record,
            "scaled_train": scaled_record,
            "development_new": new_dev_record,
            "development_previous": old_dev_record,
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    records = {
        "repeat_train": repeat_record,
        "scaled_train": scaled_record,
        "development_new": new_dev_record,
        "development_previous": old_dev_record,
    }
    data = {
        "corpus_metadata_sha256": _sha(paths["metadata"].read_bytes()),
        **records,
        "previous_training_sha256": repeat_record["sha256"],
        "previous_development_sha256": old_dev_record["sha256"],
    }
    plan: dict[str, Any] = {
        "schema": q25_fim.SCALE_PLAN_SCHEMA,
        "scale_variant": "repeat",
        "preparation_plan_sha256": metadata["preparation_plan_sha256"],
        "parent_cpt_plan_sha256": metadata["parent_cpt_plan_sha256"],
        "data": data,
        "experiment": {
            "variants": {
                "repeat": {"distinct_states": 4096, "epochs": 2, "example_exposures": 8192},
                "scaled": {"distinct_states": 8192, "epochs": 1, "example_exposures": 8192},
            }
        },
    }
    return paths, plan


def _examples(count: int) -> tuple[EncodedExample, ...]:
    return tuple(
        EncodedExample(
            input_ids=(1, 2, 3, 4),
            labels=(-100, -100, 3, 4),
            prompt_tokens=2,
            response_tokens=2,
            total_tokens=4 + index % 3,
            prompt="",
            response="",
        )
        for index in range(count)
    )


def test_scale_schedule_preserves_old_first_stage_order_and_matches_exposures() -> None:
    original = _examples(4096)
    added = _examples(4096)
    old_batches = q25_fim.training_batches(original, effective_batch=16, seed=314159)
    replay_batches = q25_fim.training_batches(original, effective_batch=16, seed=314160)

    repeat = q25_fim.scale_training_batches(
        original, variant="repeat", effective_batch=16, seed=314159
    )
    scaled = q25_fim.scale_training_batches(
        original + added, variant="scaled", effective_batch=16, seed=314159
    )

    assert repeat[:256] == old_batches
    assert repeat[256:] == replay_batches
    assert scaled[:256] == old_batches
    assert scaled[256:] == tuple(tuple(index + 4096 for index in batch) for batch in replay_batches)
    assert len(repeat) == len(scaled) == 512
    assert sum(map(len, repeat)) == sum(map(len, scaled)) == 8192
    assert q25_fim.batch_order_sha256(repeat[:256]) == q25_fim.batch_order_sha256(scaled[:256])
    with pytest.raises(ValueError, match="rows differ"):
        q25_fim.scale_training_batches(
            original[:32], variant="repeat", effective_batch=16, seed=314159
        )


def test_scale_corpus_binds_four_files_and_checks_both_heldout_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, plan = _make_scale_corpus(tmp_path)
    monkeypatch.setattr(
        q25_fim,
        "SCALE_PREVIOUS_TRAIN_SHA256",
        plan["data"]["previous_training_sha256"],
    )
    monkeypatch.setattr(
        q25_fim,
        "SCALE_PREVIOUS_DEVELOPMENT_SHA256",
        plan["data"]["previous_development_sha256"],
    )
    plan_path = paths["plan"]
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")

    repeat, new_dev, old_dev, tokenizer_rows, identity = q25_fim.load_scale_corpus(
        repeat_train_path=paths["repeat_train"],
        scaled_train_path=paths["scaled_train"],
        development_path=paths["development_new"],
        historical_development_path=paths["development_previous"],
        metadata_path=paths["metadata"],
        plan=plan,
        variant="repeat",
    )

    assert len(repeat) == 4096
    assert len(new_dev) == 512
    assert len(old_dev) == 240
    assert len(tokenizer_rows) == 8192 + 512 + 240
    assert identity["splits"]["repeat_train"]["rows"] == 4096
    assert identity["splits"]["scaled_train"]["rows"] == 8192


def test_scale_corpus_rejects_new_train_overlap_with_historical_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, plan = _make_scale_corpus(tmp_path)
    monkeypatch.setattr(
        q25_fim,
        "SCALE_PREVIOUS_TRAIN_SHA256",
        plan["data"]["previous_training_sha256"],
    )
    monkeypatch.setattr(
        q25_fim,
        "SCALE_PREVIOUS_DEVELOPMENT_SHA256",
        plan["data"]["previous_development_sha256"],
    )
    new_rows = paths["scaled_train"].read_text(encoding="utf-8").splitlines()
    old_dev = [json.loads(line) for line in paths["development_previous"].read_text().splitlines()]
    conflicting = json.loads(new_rows[4096])
    old_dev[0]["repository_alias_sha256"] = [conflicting["repository_group_sha256"]]
    old_dev_payload = _write_rows(paths["development_previous"], old_dev)
    old_record = plan["data"]["development_previous"]
    old_record.update(sha256=_sha(old_dev_payload), bytes=len(old_dev_payload))
    plan["data"]["previous_development_sha256"] = _sha(old_dev_payload)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata["files"]["development_previous.jsonl"].update(
        sha256=_sha(old_dev_payload), bytes=len(old_dev_payload)
    )
    metadata["splits"]["development_previous"].update(
        sha256=_sha(old_dev_payload), bytes=len(old_dev_payload)
    )
    paths["metadata"].write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    plan["data"]["corpus_metadata_sha256"] = _sha(paths["metadata"].read_bytes())
    plan["data"]["previous_development_sha256"] = _sha(old_dev_payload)
    monkeypatch.setattr(q25_fim, "SCALE_PREVIOUS_DEVELOPMENT_SHA256", _sha(old_dev_payload))

    with pytest.raises(ValueError, match="source groups overlap"):
        q25_fim.load_scale_corpus(
            repeat_train_path=paths["repeat_train"],
            scaled_train_path=paths["scaled_train"],
            development_path=paths["development_new"],
            historical_development_path=paths["development_previous"],
            metadata_path=paths["metadata"],
            plan=plan,
            variant="repeat",
        )


def test_scaled_rows_reject_duplicate_new_state_identity(tmp_path: Path) -> None:
    old = [_row(index, "train") for index in range(4096)]
    new = [_row(4096 + index, "train", new=True) for index in range(4096)]
    new[1]["source_content_sha256"] = new[0]["source_content_sha256"]
    new[1]["mode"] = new[0]["mode"]
    new[1]["region_start"] = new[0]["region_start"]
    new[1]["region_end"] = new[0]["region_end"]
    new[1]["target_sha256"] = new[0]["target_sha256"]
    path = tmp_path / "scaled_train.jsonl"
    _write_rows(path, old + new)

    with pytest.raises(ValueError, match="duplicate completion state"):
        q25_fim._validate_distinct_scale_states(path)


@pytest.mark.parametrize(
    ("variant", "distinct_examples", "input_multiplier"),
    [("repeat", 4096, 2), ("scaled", 8192, 1)],
)
def test_scale_preflight_reports_matched_updates_and_exposures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    distinct_examples: int,
    input_multiplier: int,
) -> None:
    paths, plan = _make_scale_corpus(tmp_path)
    plan["scale_variant"] = variant
    monkeypatch.setattr(
        q25_fim,
        "SCALE_PREVIOUS_TRAIN_SHA256",
        plan["data"]["previous_training_sha256"],
    )
    monkeypatch.setattr(
        q25_fim,
        "SCALE_PREVIOUS_DEVELOPMENT_SHA256",
        plan["data"]["previous_development_sha256"],
    )
    plan.update(
        gpu_execution_authorized=True,
        initializers={q25_fim.TRAIN_ARM: {"kind": "untouched_pretrained"}},
        configuration={
            "training": {
                "sequence_length": 1024,
                "effective_batch": 16,
                "microbatch_examples": 1,
                "checkpoint_every_updates": 64,
                "seed": 314159,
                "attention": "sdpa",
                "gradient_checkpointing": True,
                "master_weights": "fp32",
                "compute": "fp16",
                "optimizer": "AdamW8bit",
                "objective": "example_mean_response_only_FIM_target_and_EOS",
                "learning_rate": 1e-5,
                "warmup_fraction": 0.03,
                "cosine_floor_fraction": 0.1,
                "gradient_clip": 1.0,
                "weight_decay": 0.01,
                "initial_loss_scale": 128,
                "max_input_tokens": 8_000_000,
            },
            "budget": {
                "session_seconds": 10_800,
                "maximum_campaign_input_tokens": 10_000_000,
                "minimum_finalization_reserve_seconds": 1800,
            },
        },
    )
    paths["plan"].write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        q25_fim,
        "verify_initializer",
        lambda *_args, **_kwargs: {"files": {"model.safetensors": {"bytes": 1}}},
    )
    monkeypatch.setattr(q25_fim, "validate_local_tokenizer_corpus", lambda *_args: None)

    result = q25_fim.run_training(
        model_path=tmp_path / "unused-model",
        train_path=paths[f"{variant}_train"],
        repeat_train_path=paths["repeat_train"],
        scaled_train_path=paths["scaled_train"],
        development_path=paths["development_new"],
        historical_development_path=paths["development_previous"],
        metadata_path=paths["metadata"],
        plan_path=paths["plan"],
        arm=q25_fim.TRAIN_ARM,
        scale_variant=variant,
        output=tmp_path / "out",
        session_seconds=10_800,
        execute=False,
    )

    assert result["scale_variant"] == variant
    assert result["data"]["splits"]["repeat_train"]["rows"] == 4096
    assert result["data"]["splits"]["scaled_train"]["rows"] == 8192
    assert result["training_examples"] == distinct_examples
    assert result["training_example_exposures"] == 8192
    assert result["updates"] == 512
    assert result["training_input_tokens"] == distinct_examples * 7 * input_multiplier
    assert result["supervised_target_tokens_including_eos"] == (
        distinct_examples * 2 * input_multiplier
    )


def test_checkpoint_history_records_only_verified_committed_marker(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume-step-000256.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    fingerprint = _sha("run")
    cursor = TrainingCursor(
        next_example_index=4096,
        completed_updates=256,
        attempted_updates=256,
        training_input_tokens=1234,
        supervised_target_tokens=567,
        epoch=1,
    )
    checkpoint_sha = _sha(checkpoint.read_bytes())
    marker = {
        "sha256": checkpoint_sha,
        "fingerprint": fingerprint,
    }
    checkpoint.with_suffix(checkpoint.suffix + ".complete.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    (tmp_path / "latest.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "path": checkpoint.name,
                "sha256": checkpoint_sha,
                "cursor": cursor.__dict__,
            }
        ),
        encoding="utf-8",
    )

    q25_fim._record_scale_checkpoint(
        tmp_path, checkpoint, fingerprint=fingerprint, cursor=cursor
    )
    with (tmp_path / "checkpoint_history.jsonl").open("ab") as stream:
        stream.write(b'{"incomplete":')
    q25_fim._record_scale_checkpoint(
        tmp_path, checkpoint, fingerprint=fingerprint, cursor=cursor
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "checkpoint_history.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["cursor"] == cursor.__dict__
    assert records[0]["checkpoint_sha256"] == checkpoint_sha

    marker["sha256"] = _sha("different")
    checkpoint.with_suffix(checkpoint.suffix + ".complete.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="committed matching marker"):
        q25_fim._record_scale_checkpoint(
            tmp_path, checkpoint, fingerprint=fingerprint, cursor=cursor
        )


def test_original_fim_v1_validator_still_rejects_two_epochs() -> None:
    settings = {
        "epochs": 2,
        "sequence_length": 1024,
        "effective_batch": 16,
        "microbatch_examples": 1,
        "checkpoint_every_updates": 64,
        "seed": 314159,
        "attention": "sdpa",
        "gradient_checkpointing": True,
        "master_weights": "fp32",
        "compute": "fp16",
        "optimizer": "AdamW8bit",
        "objective": "example_mean_response_only_FIM_target_and_EOS",
        "learning_rate": 1e-5,
        "warmup_fraction": 0.03,
        "cosine_floor_fraction": 0.1,
        "gradient_clip": 1.0,
        "weight_decay": 0.01,
        "initial_loss_scale": 128,
        "max_input_tokens": 4_194_304,
    }
    with pytest.raises(ValueError, match="epochs"):
        q25_fim.validate_training_configuration(settings)


def test_scale_checkpoint_reloads_in_fresh_process_and_retains_pruned_identity(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from tinycomplete.one_line.train import CosineUpdateSchedule

    torch.manual_seed(91)
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = CosineUpdateSchedule(optimizer, peak_lr=1e-5, total_updates=512)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(256):
        scheduler.step()
    fingerprint = _sha("process-reload-scale")
    cursor = TrainingCursor(
        next_example_index=4096,
        completed_updates=256,
        attempted_updates=256,
        training_input_tokens=1234,
        supervised_target_tokens=567,
        epoch=1,
    )
    save_options = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": None,
        "fingerprint": fingerprint,
        "output_cap_bytes": 2**28,
        "total_cap_bytes": 2**28,
        "input_artifact_bytes": 0,
        "minimum_free_bytes": 0,
        "source_weight_bytes": 16,
    }
    checkpoint = q25_fim.save_fim_training_cursor(tmp_path, cursor=cursor, **save_options)
    q25_fim._record_scale_checkpoint(
        tmp_path, checkpoint, fingerprint=fingerprint, cursor=cursor
    )
    child_code = """
import json, sys, torch
from pathlib import Path
from tinycomplete.code_cpt.q25 import load_training_cursor
from tinycomplete.one_line.train import CosineUpdateSchedule
model = torch.nn.Linear(2, 2, bias=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = CosineUpdateSchedule(optimizer, peak_lr=1e-5, total_updates=512)
cursor = load_training_cursor(Path(sys.argv[1]), model=model, optimizer=optimizer,
    scheduler=scheduler, scaler=None, fingerprint=sys.argv[2])
print(json.dumps({"cursor": cursor.__dict__, "weights": model.weight.tolist(),
    "optimizer_entries": len(optimizer.state), "schedule": scheduler.state_dict()}))
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path / "latest.json"), fingerprint],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    reloaded = json.loads(child.stdout)
    assert reloaded["cursor"] == cursor.__dict__
    assert reloaded["weights"] == model.weight.tolist()
    assert reloaded["optimizer_entries"] == 1
    assert reloaded["schedule"] == scheduler.state_dict()
    next_cursor = TrainingCursor(
        next_example_index=8192,
        completed_updates=512,
        attempted_updates=512,
        training_input_tokens=2468,
        supervised_target_tokens=1134,
        epoch=2,
    )
    next_checkpoint = q25_fim.save_fim_training_cursor(
        tmp_path, cursor=next_cursor, **save_options
    )
    q25_fim._record_scale_checkpoint(
        tmp_path, next_checkpoint, fingerprint=fingerprint, cursor=next_cursor
    )
    assert not checkpoint.exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "checkpoint_history.jsonl").read_text().splitlines()
    ]
    assert [row["cursor"]["attempted_updates"] for row in rows] == [256, 512]
    assert rows[0]["checkpoint_sha256"] != rows[1]["checkpoint_sha256"]
