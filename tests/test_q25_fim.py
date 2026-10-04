from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tinycomplete.code_cpt.q25_fim import (
    CPT_ARM,
    Q25_CONFIG_SEMANTICS,
    TRAIN_ARM,
    _input_artifact_bytes,
    load_fim_corpus,
    load_fim_examples,
    run_training,
    save_fim_training_cursor,
    training_batches,
    validate_fim_tokenizer,
    validate_training_configuration,
    verify_initializer,
)
from tinycomplete.one_line.train import EncodedExample, TrainingCursor, train_encoded

MARKERS = {"fim_prefix": 1, "fim_suffix": 2, "fim_middle": 3}
EOS = 4


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _row(
    identifier: int,
    split: str,
    *,
    repository_hash: str | None = None,
    source_hash: str | None = None,
    input_ids: list[int] | None = None,
    target_tokens: int = 2,
) -> dict:
    ids = [1, 10, 2, 11, 3, 12, EOS] if input_ids is None else input_ids
    prompt_tokens = len(ids) - target_tokens
    return {
        "id": identifier,
        "split": split,
        "language": "python",
        "mode": "whole_logical_line",
        "prompt_format": "psm",
        "variant": 0,
        "source_content_sha256": source_hash or _sha(f"{identifier}-source"),
        "source_path": f"repo/{identifier}.py",
        "repository_identity_sha256": repository_hash or _sha(f"{identifier}-repo"),
        "repository_alias_sha256": [],
        "licenses": ["MIT"],
        "dataset_id": "public-code-fixture",
        "dataset_revision": "frozen-revision",
        "region_start": 0,
        "region_end": 4,
        "prompt_sha256": _sha(f"{identifier}-prompt"),
        "target_sha256": _sha(f"{identifier}-target"),
        "input_ids": ids,
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "total_tokens": len(ids),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def test_load_fim_examples_masks_prompt_and_supervises_target_plus_eos(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_rows(path, [_row(0, "train")])

    examples = load_fim_examples(
        path,
        split="train",
        eos_token_id=EOS,
        fim_marker_ids=MARKERS,
        vocab_size=32,
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.input_ids == (1, 10, 2, 11, 3, 12, EOS)
    assert example.labels == (-100, -100, -100, -100, -100, 12, EOS)
    assert example.prompt_tokens == 5
    assert example.response_tokens == 2
    assert example.total_tokens == 7


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"input_ids": [1, 10, 2, 11, 3, 12, 5]}, "EOS"),
        ({"input_ids": [1, 10, 11, 3, 12, 4]}, "PSM"),
        ({"prompt_format": "spm"}, "PSM"),
        ({"id": 1}, "IDs"),
    ],
)
def test_load_fim_examples_rejects_invalid_boundaries(
    tmp_path: Path, updates: dict, expected: str
) -> None:
    row = _row(0, "train")
    row.update(updates)
    if "input_ids" in updates:
        row["total_tokens"] = len(row["input_ids"])
        row["prompt_tokens"] = len(row["input_ids"]) - row["target_tokens"]
    path = tmp_path / "train.jsonl"
    _write_rows(path, [row])

    with pytest.raises(ValueError, match=expected):
        load_fim_examples(
            path,
            split="train",
            eos_token_id=EOS,
            fim_marker_ids=MARKERS,
            vocab_size=32,
        )


def test_fim_train_batch_order_uses_every_example_once_and_is_repeatable() -> None:
    example = EncodedExample(
        input_ids=(1, 10, 2, 11, 3, 12, EOS),
        labels=(-100, -100, -100, -100, -100, 12, EOS),
        prompt_tokens=5,
        response_tokens=2,
        total_tokens=7,
        prompt="",
        response="",
    )
    examples = (example,) * 4096
    first = training_batches(examples, effective_batch=16, seed=314159)
    second = training_batches(examples, effective_batch=16, seed=314159)

    assert first == second
    assert len(first) == 256
    assert all(len(batch) == 16 for batch in first)
    assert sorted(index for batch in first for index in batch) == list(range(4096))


def test_load_fim_corpus_binds_file_hash_tokenizer_and_disjoint_source_groups(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    development_path = tmp_path / "development.jsonl"
    _write_rows(train_path, [_row(0, "train")])
    _write_rows(
        development_path,
        [_row(4096, "development", repository_hash=_sha("held-out-repo"))],
    )
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    development_sha = hashlib.sha256(development_path.read_bytes()).hexdigest()
    metadata = {
        "schema": "q25-fim-prepared-corpus-v1",
        "raw_source_content_emitted": False,
        "sealed_test_accessed": False,
        "preparation_plan_sha256": _sha("prep-plan"),
        "parent_cpt_plan_sha256": _sha("parent-cpt-plan"),
        "tokenizer_id": "Qwen/Qwen2.5-Coder-0.5B",
        "tokenizer_revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
        "tokenizer_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "eos_token_id": EOS,
        "fim_marker_ids": MARKERS,
        "files": {
            "train.jsonl": {"sha256": train_sha, "bytes": train_path.stat().st_size},
            "development.jsonl": {
                "sha256": development_sha,
                "bytes": development_path.stat().st_size,
            },
        },
        "splits": {
            "train": {
                "file": "train.jsonl",
                "sha256": train_sha,
                "row_count": 1,
                "input_tokens": 7,
                "target_tokens": 2,
            },
            "development": {
                "file": "development.jsonl",
                "sha256": development_sha,
                "row_count": 1,
                "input_tokens": 7,
                "target_tokens": 2,
            },
        },
    }
    metadata_path = tmp_path / "corpus_metadata.json"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True))
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    prep_sha = metadata["preparation_plan_sha256"]
    plan: dict[str, Any] = {
        "schema": "q25-fim-training-plan-v1",
        "preparation_plan_sha256": prep_sha,
        "parent_cpt_plan_sha256": metadata["parent_cpt_plan_sha256"],
        "data": {
            "corpus_metadata_sha256": metadata_sha,
            "train": {
                "sha256": train_sha,
                "bytes": train_path.stat().st_size,
                "row_count": 1,
                "input_tokens": 7,
                "target_tokens": 2,
            },
            "development": {
                "sha256": development_sha,
                "bytes": development_path.stat().st_size,
                "row_count": 1,
                "input_tokens": 7,
                "target_tokens": 2,
            },
        },
    }

    train, development, identity = load_fim_corpus(
        train_path=train_path,
        development_path=development_path,
        metadata_path=metadata_path,
        plan=plan,
    )

    assert len(train) == len(development) == 1
    assert identity["splits"]["train"]["input_tokens"] == 7
    plan["data"]["train"].pop("bytes")
    with pytest.raises(ValueError, match="incomplete train data identity"):
        load_fim_corpus(
            train_path=train_path,
            development_path=development_path,
            metadata_path=metadata_path,
            plan=plan,
        )
    plan["data"]["train"]["bytes"] = train_path.stat().st_size
    metadata["raw_source_content_emitted"] = True
    metadata_path.write_text(json.dumps(metadata, sort_keys=True))
    with pytest.raises(ValueError, match="must not emit raw source"):
        load_fim_corpus(
            train_path=train_path,
            development_path=development_path,
            metadata_path=metadata_path,
            plan=plan,
        )


def test_load_fim_corpus_rejects_source_group_overlap(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    development_path = tmp_path / "development.jsonl"
    shared = _sha("same-repository")
    _write_rows(train_path, [_row(0, "train", repository_hash=shared)])
    _write_rows(
        development_path,
        [_row(4096, "development", repository_hash=shared)],
    )
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    dev_sha = hashlib.sha256(development_path.read_bytes()).hexdigest()
    metadata = {
        "schema": "q25-fim-prepared-corpus-v1",
        "raw_source_content_emitted": False,
        "sealed_test_accessed": False,
        "preparation_plan_sha256": _sha("prep-plan"),
        "parent_cpt_plan_sha256": _sha("parent-cpt-plan"),
        "tokenizer_id": "Qwen/Qwen2.5-Coder-0.5B",
        "tokenizer_revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
        "tokenizer_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "eos_token_id": EOS,
        "fim_marker_ids": MARKERS,
        "files": {
            "train.jsonl": {"sha256": train_sha, "bytes": train_path.stat().st_size},
            "development.jsonl": {
                "sha256": dev_sha,
                "bytes": development_path.stat().st_size,
            },
        },
        "splits": {
            "train": {
                "file": "train.jsonl",
                "sha256": train_sha,
                "row_count": 1,
                "input_tokens": 7,
                "target_tokens": 2,
            },
            "development": {
                "file": "development.jsonl",
                "sha256": dev_sha,
                "row_count": 1,
                "input_tokens": 7,
                "target_tokens": 2,
            },
        },
    }
    metadata_path = tmp_path / "corpus_metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    plan = {
        "preparation_plan_sha256": metadata["preparation_plan_sha256"],
        "parent_cpt_plan_sha256": metadata["parent_cpt_plan_sha256"],
    }

    with pytest.raises(ValueError, match="source groups overlap"):
        load_fim_corpus(
            train_path=train_path,
            development_path=development_path,
            metadata_path=metadata_path,
            plan=plan,
        )


class _FakeTokenizer:
    eos_token_id = EOS
    all_special_tokens = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]
    _ids = {"<|fim_prefix|>": 1, "<|fim_suffix|>": 2, "<|fim_middle|>": 3}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [self._ids[text]]

    def convert_tokens_to_ids(self, text: str) -> int:
        return self._ids[text]


class _RoundTripTokenizer(_FakeTokenizer):
    def decode(self, token_ids, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool):
        del skip_special_tokens, clean_up_tokenization_spaces
        return ",".join(str(token_id) for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if text in self._ids:
            return [self._ids[text]]
        return [int(value) for value in text.split(",")]


def test_validate_fim_tokenizer_preserves_special_marker_ids() -> None:
    validate_fim_tokenizer(
        _FakeTokenizer(),
        {"eos_token_id": EOS, "fim_marker_ids": MARKERS},
    )


def test_validate_fim_tokenizer_roundtrips_all_frozen_example_ids() -> None:
    example = EncodedExample(
        input_ids=(1, 10, 2, 11, 3, 12, EOS),
        labels=(-100, -100, -100, -100, -100, 12, EOS),
        prompt_tokens=5,
        response_tokens=2,
        total_tokens=7,
        prompt="",
        response="",
    )
    validate_fim_tokenizer(
        _RoundTripTokenizer(),
        {"eos_token_id": EOS, "fim_marker_ids": MARKERS},
        (example,),
    )

    class ChangedTokenization(_RoundTripTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            del add_special_tokens
            if text in self._ids:
                return [self._ids[text]]
            return [1]

    with pytest.raises(ValueError, match="changes prepared FIM token IDs"):
        validate_fim_tokenizer(
            ChangedTokenization(),
            {"eos_token_id": EOS, "fim_marker_ids": MARKERS},
            (example,),
        )


def test_storage_input_estimate_counts_resume_marker_and_mounted_files(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    files: dict[str, Path] = {}
    for name in ("train.jsonl", "development.jsonl", "metadata.json", "plan.json"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    previous = tmp_path / "previous"
    previous.mkdir()
    checkpoint = previous / "resume-step-000064.pt"
    checkpoint.write_bytes(b"checkpoint")
    marker = checkpoint.with_suffix(checkpoint.suffix + ".complete.json")
    marker.write_bytes(b"complete-marker")

    counted = _input_artifact_bytes(
        model_path=model,
        train_path=files["train.jsonl"],
        development_path=files["development.jsonl"],
        metadata_path=files["metadata.json"],
        plan_path=files["plan.json"],
        resume_path=checkpoint,
        output=tmp_path / "current",
        mounted_artifact_bytes=23,
    )
    expected = (
        sum(path.stat().st_size for path in model.rglob("*"))
        + sum(path.stat().st_size for path in files.values())
        + checkpoint.stat().st_size
        + marker.stat().st_size
        + 23
    )
    assert counted == expected

    marker.unlink()
    with pytest.raises(ValueError, match="completion marker"):
        _input_artifact_bytes(
            model_path=model,
            train_path=files["train.jsonl"],
            development_path=files["development.jsonl"],
            metadata_path=files["metadata.json"],
            plan_path=files["plan.json"],
            resume_path=checkpoint,
            output=tmp_path / "current",
        )


def test_training_configuration_rejects_schedule_or_objective_drift() -> None:
    configuration = {
        "epochs": 1,
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

    assert validate_training_configuration(configuration) == configuration
    drifted = {**configuration, "learning_rate": 2e-5}
    with pytest.raises(ValueError, match="learning_rate"):
        validate_training_configuration(drifted)


def test_cpu_preparation_plan_cannot_authorize_execution(tmp_path: Path) -> None:
    configuration = {
        "epochs": 1,
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
    plan_path = tmp_path / "cpu-preparation-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "q25-fim-cpu-preparation-plan-v1",
                "gpu_execution_authorized": False,
                "configuration": {"training": configuration, "budget": {}},
            }
        )
    )

    with pytest.raises(PermissionError, match="does not authorize GPU execution"):
        run_training(
            model_path=tmp_path / "model",
            train_path=tmp_path / "train.jsonl",
            development_path=tmp_path / "development.jsonl",
            metadata_path=tmp_path / "corpus_metadata.json",
            plan_path=plan_path,
            arm=TRAIN_ARM,
            output=tmp_path / "output",
            session_seconds=3600,
            execute=True,
        )


def test_initializer_identity_requires_frozen_base_files(tmp_path: Path, monkeypatch) -> None:
    import tinycomplete.code_cpt.q25_fim as q25_fim

    model_path = tmp_path / "untouched"
    model_path.mkdir()
    content = {
        "config.json": json.dumps(Q25_CONFIG_SEMANTICS, sort_keys=True).encode(),
        "model.safetensors": b"synthetic-weight-artifact",
        "tokenizer.json": b"synthetic-tokenizer-artifact",
    }
    for name, value in content.items():
        (model_path / name).write_bytes(value)
    digests = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
    monkeypatch.setattr(q25_fim, "CONFIG_SHA256", digests["config.json"])
    monkeypatch.setattr(q25_fim, "WEIGHT_SHA256", digests["model.safetensors"])
    monkeypatch.setattr(q25_fim, "TOKENIZER_SHA256", digests["tokenizer.json"])
    entry = {
        "kind": "untouched_pretrained",
        "files": {
            name: {"sha256": digest, "bytes": (model_path / name).stat().st_size}
            for name, digest in digests.items()
        },
    }

    identity = verify_initializer(model_path, arm=TRAIN_ARM, entry=entry)

    assert identity["kind"] == "untouched_pretrained"
    assert identity["files"]["model.safetensors"]["sha256"] == digests["model.safetensors"]
    (model_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen identity"):
        verify_initializer(model_path, arm=TRAIN_ARM, entry=entry)


def test_cpt_initializer_requires_complete_manifest_and_all_files(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "cpt-export"
    model_path.mkdir()
    content = {
        "config.json": json.dumps(Q25_CONFIG_SEMANTICS, sort_keys=True).encode(),
        "model.safetensors": b"synthetic-cpt-weight-artifact",
        "tokenizer.json": b"synthetic-tokenizer-artifact",
    }
    for name, value in content.items():
        (model_path / name).write_bytes(value)
    files = {
        name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in content.items()
    }
    cursor = {
        "next_example_index": 7688,
        "attempted_updates": 481,
        "completed_updates": 481,
        "skipped_updates": 0,
        "training_input_tokens": 7_872_512,
        "supervised_target_tokens": 7_872_512,
        "epoch": 1,
    }
    cpt_fingerprint = _sha("completed-cpt")
    artifact_manifest = {
        "schema": "q25-cpt-inference-f16-v1",
        "fingerprint": cpt_fingerprint,
        "base_model": "Qwen/Qwen2.5-Coder-0.5B",
        "base_revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
        "training_cursor": cursor,
        "files": files,
    }
    manifest_path = model_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(artifact_manifest, sort_keys=True))
    entry = {
        "kind": "completed_cpt_export",
        "artifact_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "fingerprint": cpt_fingerprint,
        "training_cursor": cursor,
        "files": files,
        "expected_complete_updates": 481,
        "expected_training_input_tokens": 7_872_512,
    }

    identity = verify_initializer(model_path, arm=CPT_ARM, entry=entry)

    assert identity["kind"] == "completed_cpt_export"
    assert identity["training_cursor"] == cursor
    incomplete = {**artifact_manifest, "training_cursor": {**cursor, "completed_updates": 255}}
    manifest_path.write_text(json.dumps(incomplete, sort_keys=True))
    entry["artifact_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    entry["training_cursor"] = incomplete["training_cursor"]
    with pytest.raises(ValueError, match="skipped or incomplete"):
        verify_initializer(model_path, arm=CPT_ARM, entry=entry)


class _NoOpScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def get_scale(self) -> float:
        return 1.0

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        assert state == {}


def _tiny_examples() -> tuple[EncodedExample, ...]:
    examples = []
    for offset in range(8):
        ids = (1, 10 + offset, 2, 11, 3, 12 + offset, EOS)
        examples.append(
            EncodedExample(
                input_ids=ids,
                labels=(-100, -100, -100, -100, -100, 12 + offset, EOS),
                prompt_tokens=5,
                response_tokens=2,
                total_tokens=7,
                prompt="",
                response="",
            )
        )
    return tuple(examples)


def _torch_or_skip():
    return pytest.importorskip("torch")


def _tiny_model(torch):
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.embedding = torch.nn.Embedding(32, 12)
            self.lm_head = torch.nn.Linear(12, 32, bias=False)

        def forward(
            self,
            *,
            input_ids,
            attention_mask=None,
            use_cache=False,
            logits_to_keep=0,
        ):
            del attention_mask, use_cache
            hidden = self.embedding(input_ids)
            if isinstance(logits_to_keep, torch.Tensor):
                hidden = hidden.index_select(1, logits_to_keep)
            return SimpleNamespace(logits=self.lm_head(hidden))

    return Model()


def _train_state(torch, state, *, total_updates: int):
    from tinycomplete.one_line.train import CosineUpdateSchedule

    model = _tiny_model(torch)
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = CosineUpdateSchedule(
        optimizer, peak_lr=0.01, total_updates=total_updates, warmup_fraction=0.25
    )
    return model, optimizer, scheduler, _NoOpScaler()


def test_cpu_response_only_fim_training_resumes_exactly(tmp_path: Path) -> None:
    torch = _torch_or_skip()
    from tinycomplete.code_cpt.q25 import load_training_cursor

    torch.manual_seed(147)
    seed_model = _tiny_model(torch)
    initial_state = {key: value.detach().clone() for key, value in seed_model.state_dict().items()}
    examples = _tiny_examples()
    batches = training_batches(examples, effective_batch=2, seed=11)
    assert len(batches) == 4
    external_campaign_tokens = 4_194_305
    campaign_cap = 32_000_000

    full_model, full_optimizer, full_scheduler, full_scaler = _train_state(
        torch, initial_state, total_updates=len(batches)
    )
    train_encoded(
        full_model,
        examples,
        batches,
        optimizer=full_optimizer,
        scheduler=full_scheduler,
        scaler=full_scaler,
        device=torch.device("cpu"),
        pad_token_id=EOS,
        microbatch_examples=1,
        max_input_tokens=campaign_cap,
        external_campaign_tokens=external_campaign_tokens,
    )
    assert external_campaign_tokens > 4_194_304
    assert external_campaign_tokens + sum(
        example.total_tokens for example in examples
    ) < campaign_cap
    limited_model, limited_optimizer, limited_scheduler, limited_scaler = _train_state(
        torch, initial_state, total_updates=len(batches)
    )
    with pytest.raises(ValueError, match="campaign input-token budget"):
        train_encoded(
            limited_model,
            examples,
            batches,
            optimizer=limited_optimizer,
            scheduler=limited_scheduler,
            scaler=limited_scaler,
            device=torch.device("cpu"),
            pad_token_id=EOS,
            microbatch_examples=1,
            max_input_tokens=external_campaign_tokens + 55,
            external_campaign_tokens=external_campaign_tokens,
        )

    interrupted, optimizer, scheduler, scaler = _train_state(
        torch, initial_state, total_updates=len(batches)
    )
    output = tmp_path / "resume"
    fingerprint = "fim-cpu-test-fingerprint"

    class StopAfterTwoUpdates(Exception):
        pass

    def stop_after_two(record) -> None:
        if record["attempted_update"] != 2:
            return
        consumed = [index for batch in batches[:2] for index in batch]
        cursor = TrainingCursor(
            next_example_index=len(consumed),
            completed_updates=2,
            attempted_updates=2,
            training_input_tokens=sum(examples[index].total_tokens for index in consumed),
            supervised_target_tokens=sum(examples[index].response_tokens for index in consumed),
            epoch=0,
        )
        save_fim_training_cursor(
            output,
            model=interrupted,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            fingerprint=fingerprint,
            cursor=cursor,
            output_cap_bytes=512 * 1024**2,
            total_cap_bytes=1024 * 1024**2,
            input_artifact_bytes=0,
            minimum_free_bytes=0,
            source_weight_bytes=1024,
        )
        raise StopAfterTwoUpdates

    with pytest.raises(StopAfterTwoUpdates):
        train_encoded(
            interrupted,
            examples,
            batches,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=torch.device("cpu"),
            pad_token_id=EOS,
            microbatch_examples=1,
            max_input_tokens=campaign_cap,
            external_campaign_tokens=external_campaign_tokens,
            on_update=stop_after_two,
        )

    latest = json.loads((output / "latest.json").read_text())
    assert latest["path"] == "resume-step-000002.pt"
    assert latest["schema"] == "q25-fim-latest-checkpoint-v1"
    resumed, resumed_optimizer, resumed_scheduler, resumed_scaler = _train_state(
        torch, initial_state, total_updates=len(batches)
    )
    cursor = load_training_cursor(
        output / latest["path"],
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        fingerprint=fingerprint,
    )
    assert cursor.attempted_updates == 2
    train_encoded(
        resumed,
        examples,
        batches,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        device=torch.device("cpu"),
        pad_token_id=EOS,
        microbatch_examples=1,
        max_input_tokens=campaign_cap,
        external_campaign_tokens=external_campaign_tokens,
        cursor=cursor,
    )

    for expected, actual in zip(full_model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(expected, actual)
    assert len(list(output.glob("resume-step-*.pt"))) == 1


def test_resume_fingerprint_refuses_a_different_fim_arm(tmp_path: Path) -> None:
    torch = _torch_or_skip()
    from tinycomplete.code_cpt.q25 import load_training_cursor, save_training_cursor

    model = _tiny_model(torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    from tinycomplete.one_line.train import CosineUpdateSchedule

    scheduler = CosineUpdateSchedule(optimizer, peak_lr=1e-3, total_updates=1)
    scaler = _NoOpScaler()
    save_training_cursor(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        fingerprint="untouched-arm-fingerprint",
        cursor=TrainingCursor(),
        output_cap_bytes=512 * 1024**2,
        total_cap_bytes=1024 * 1024**2,
        input_artifact_bytes=0,
        minimum_free_bytes=0,
        source_weight_bytes=1024,
    )

    with pytest.raises(ValueError, match="identity"):
        load_training_cursor(
            tmp_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            fingerprint="completed-cpt-arm-fingerprint",
        )
