from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tinycomplete.code_cpt.q25 import (
    encode_raw_code_blocks,
    evaluate_source_nll,
    load_training_cursor,
    training_batches,
    validate_campaign_token_budget,
)
from tinycomplete.one_line.train import (
    CosineUpdateSchedule,
    TrainingCursor,
    train_encoded,
)


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


def _torch_or_skip():
    return pytest.importorskip("torch")


def test_raw_code_blocks_supervise_every_next_source_token() -> None:
    blocks = np.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    examples = encode_raw_code_blocks(blocks, sequence_length=4, vocab_size=8)

    assert len(examples) == 2
    for example, source in zip(examples, blocks, strict=True):
        expected = tuple(int(token) for token in source)
        assert example.input_ids == expected
        assert example.labels == expected
        assert example.prompt_tokens == 0
        assert example.response_tokens == 3
        assert example.total_tokens == 4
    assert sum(example.response_tokens for example in examples) == 6


def test_eight_million_token_plan_has_partial_last_batch_and_four_million_replay_headroom() -> None:
    source = tuple(index % 32 for index in range(1024))
    example = encode_raw_code_blocks(
        np.asarray([source], dtype=np.int32), sequence_length=1024, vocab_size=32
    )[0]
    examples = (example,) * 7_812
    batches = training_batches(examples, effective_batch=16, seed=314_159)

    assert len(batches) == 489
    sizes = [len(batch) for batch in batches]
    assert sizes.count(16) == 488
    assert sizes.count(4) == 1
    assert sorted(index for batch in batches for index in batch) == list(range(7_812))
    logical_tokens = sum(examples[index].total_tokens for batch in batches for index in batch)
    assert logical_tokens == 7_999_488
    assert validate_campaign_token_budget(
        logical_training_tokens=logical_tokens,
        external_campaign_tokens=4_000_000,
        maximum_additional_tokens=12_000_000,
    ) == 11_999_488
    with pytest.raises(ValueError, match="campaign cap"):
        validate_campaign_token_budget(
            logical_training_tokens=logical_tokens,
            external_campaign_tokens=4_000_513,
            maximum_additional_tokens=12_000_000,
        )


class _TinyCausal:
    def __new__(cls, torch):
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


def _fixture(torch):
    raw = np.asarray(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
            [6, 7, 8, 9],
            [7, 8, 9, 10],
            [8, 9, 10, 11],
        ],
        dtype=np.int64,
    )
    examples = encode_raw_code_blocks(raw, sequence_length=4, vocab_size=32)
    batches = training_batches(examples, effective_batch=2, seed=27)
    return raw, examples, batches


def _new_train_state(torch, initial_state, *, total_updates: int):
    model = _TinyCausal(torch)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = CosineUpdateSchedule(
        optimizer, peak_lr=0.01, total_updates=total_updates, warmup_fraction=0.25
    )
    scaler = _NoOpScaler()
    return model, optimizer, scheduler, scaler


def test_cpu_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    torch = _torch_or_skip()
    from tinycomplete.code_cpt.q25 import save_training_cursor
    from tinycomplete.one_line.train import save_resume_checkpoint

    torch.manual_seed(314)
    initial = _TinyCausal(torch)
    initial_state = {
        key: value.detach().clone() for key, value in initial.state_dict().items()
    }
    _, examples, batches = _fixture(torch)

    initial_dir = tmp_path / "initial"
    initial_model, initial_optimizer, initial_scheduler, initial_scaler = _new_train_state(
        torch, initial_state, total_updates=len(batches)
    )
    initial_cursor = TrainingCursor()
    save_training_cursor(
        initial_dir,
        model=initial_model,
        optimizer=initial_optimizer,
        scheduler=initial_scheduler,
        scaler=initial_scaler,
        fingerprint="initial-fingerprint",
        cursor=initial_cursor,
        output_cap_bytes=512 * 1024**2,
        total_cap_bytes=1024 * 1024**2,
        input_artifact_bytes=0,
        minimum_free_bytes=0,
        source_weight_bytes=1024,
    )
    untouched_model, untouched_optimizer, untouched_scheduler, untouched_scaler = (
        _new_train_state(torch, initial_state, total_updates=len(batches))
    )
    reloaded_initial = load_training_cursor(
        initial_dir,
        model=untouched_model,
        optimizer=untouched_optimizer,
        scheduler=untouched_scheduler,
        scaler=untouched_scaler,
        fingerprint="initial-fingerprint",
    )
    assert reloaded_initial == initial_cursor
    for expected, actual in zip(
        initial_model.parameters(), untouched_model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)

    orphan_dir = tmp_path / "orphan-complete-marker"
    orphan_dir.mkdir()
    orphan = orphan_dir / "resume-step-000000.pt"
    save_resume_checkpoint(
        orphan,
        model=initial_model,
        optimizer=initial_optimizer,
        scheduler=initial_scheduler,
        scaler=initial_scaler,
        fingerprint="orphan-fingerprint",
        **initial_cursor.__dict__,
    )
    save_training_cursor(
        orphan_dir,
        model=initial_model,
        optimizer=initial_optimizer,
        scheduler=initial_scheduler,
        scaler=initial_scaler,
        fingerprint="orphan-fingerprint",
        cursor=initial_cursor,
        output_cap_bytes=512 * 1024**2,
        total_cap_bytes=1024 * 1024**2,
        input_artifact_bytes=0,
        minimum_free_bytes=0,
        source_weight_bytes=1024,
    )
    assert json.loads((orphan_dir / "latest.json").read_text())["path"] == orphan.name

    full_model, full_optimizer, full_scheduler, full_scaler = _new_train_state(
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
        pad_token_id=0,
        microbatch_examples=1,
        max_input_tokens=128,
    )

    interrupted_model, optimizer, scheduler, scaler = _new_train_state(
        torch, initial_state, total_updates=len(batches)
    )
    checkpoint_dir = tmp_path / "training"
    fingerprint = "test-fingerprint"

    class _StopAfterCheckpoint(Exception):
        pass

    def stop_at_update(record) -> None:
        if int(record["attempted_update"]) != 2:
            return
        consumed = [index for batch in batches[:2] for index in batch]
        cursor = TrainingCursor(
            next_example_index=len(consumed),
            completed_updates=2,
            attempted_updates=2,
            training_input_tokens=sum(examples[index].total_tokens for index in consumed),
            supervised_target_tokens=sum(
                examples[index].response_tokens for index in consumed
            ),
            epoch=0,
        )
        save_training_cursor(
            checkpoint_dir,
            model=interrupted_model,
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
        raise _StopAfterCheckpoint

    with pytest.raises(_StopAfterCheckpoint):
        train_encoded(
            interrupted_model,
            examples,
            batches,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=torch.device("cpu"),
            pad_token_id=0,
            microbatch_examples=1,
            max_input_tokens=128,
            on_update=stop_at_update,
        )

    latest = json.loads((checkpoint_dir / "latest.json").read_text())
    assert latest["path"] == "resume-step-000002.pt"
    checkpoint = checkpoint_dir / latest["path"]
    resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler = _new_train_state(
        torch, initial_state, total_updates=len(batches)
    )
    cursor = load_training_cursor(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        fingerprint=fingerprint,
    )
    assert cursor.attempted_updates == 2
    assert cursor.next_example_index == 4
    result = train_encoded(
        resumed_model,
        examples,
        batches,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        device=torch.device("cpu"),
        pad_token_id=0,
        microbatch_examples=1,
        max_input_tokens=128,
        cursor=cursor,
        on_checkpoint=lambda value: save_training_cursor(
            checkpoint_dir,
            model=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            scaler=resumed_scaler,
            fingerprint=fingerprint,
            cursor=value,
            output_cap_bytes=512 * 1024**2,
            total_cap_bytes=1024 * 1024**2,
            input_artifact_bytes=0,
            minimum_free_bytes=0,
            source_weight_bytes=1024,
        ),
    )

    assert result.status == "complete"
    assert result.cursor.attempted_updates == 4
    for (name, expected), (_, actual) in zip(
        full_model.state_dict().items(), resumed_model.state_dict().items(), strict=True
    ):
        assert name
        assert torch.equal(expected, actual)
    committed = list(checkpoint_dir.glob("resume-step-*.pt"))
    assert len(committed) == 1
    assert committed[0].name == "resume-step-000004.pt"
    final_pointer = json.loads((checkpoint_dir / "latest.json").read_text())
    assert final_pointer["path"] == committed[0].name


def test_source_nll_scores_all_shifted_tokens_on_same_blocks() -> None:
    torch = _torch_or_skip()
    model = _TinyCausal(torch)
    raw, _, _ = _fixture(torch)
    result = evaluate_source_nll(
        model, raw[:2], device=torch.device("cpu"), max_blocks=2
    )

    assert result["metric"] == "causal_next_token_nll"
    assert result["blocks"] == 2
    assert result["scored_tokens"] == 2 * (raw.shape[1] - 1)
    assert np.isfinite(result["nll"])
