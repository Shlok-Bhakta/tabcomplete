from __future__ import annotations

import math

import pytest
import torch

from tinycomplete.code_cpt.data import (
    BlockPacker,
    FilterReason,
    LanguageMix,
    RepoSplit,
    SourceFilter,
    allocate_blocks,
    repository_identity,
    repository_path,
)
from tinycomplete.code_cpt.eval import causal_nll_from_logits
from tinycomplete.code_cpt.train import (
    PackedBlocksDataset,
    TrainingCounters,
    distributed_block_indices,
    milestones_crossed,
)


def test_repo_split_is_stable_and_keeps_whole_repo_together() -> None:
    split = RepoSplit(validation_buckets=range(10), bucket_count=1000)

    first = split.assignment("github.com/example/project")
    assert first == split.assignment("github.com/example/project")
    assert split.bucket("github.com/example/project") == 815
    assert split.assignment("github.com/example/project", path="a.py") == first
    assert split.assignment("github.com/example/project", path="b.py") == first


def test_repo_split_requires_repository_identity() -> None:
    split = RepoSplit(validation_buckets=range(10))

    with pytest.raises(ValueError, match="repository identity"):
        split.assignment("")


@pytest.mark.parametrize(
    ("content", "path", "reason"),
    [
        ("", "empty.py", FilterReason.EMPTY),
        ("x = 1\n", "tiny.py", FilterReason.TOO_SMALL),
        ("\x00binary" + "x" * 200, "blob.py", FilterReason.BINARY),
        ("a" * 2_000, "bundle.min.js", FilterReason.MINIFIED),
        (("generated_line = 1\n" * 500), "vendor/generated.py", FilterReason.VENDOR),
        (("\n".join(f"value_{i} = {i}" for i in range(500))), "ok.py", None),
    ],
)
def test_source_filter(content: str, path: str, reason: FilterReason | None) -> None:
    result = SourceFilter(min_chars=32, max_chars=100_000).check(content, path)
    assert result.reason == reason
    assert result.accepted is (reason is None)


def test_block_packer_inserts_eos_and_emits_exact_blocks() -> None:
    packer = BlockPacker(block_size=5, eos_token_id=99)

    assert packer.add_document([1, 2, 3]) == []
    blocks = packer.add_document([4, 5, 6, 7])

    assert blocks == [[1, 2, 3, 99, 4]]
    assert packer.pending_tokens == [5, 6, 7]
    assert packer.source_tokens == 7
    assert packer.boundary_tokens == 1
    assert packer.emitted_tokens == 5
    assert math.isclose(packer.packing_efficiency, 1.0)


def test_language_mix_uses_token_targets_and_reports_actual_percentages() -> None:
    mix = LanguageMix({"python": 0.6, "rust": 0.4}, total_tokens=1_000)

    assert mix.target_tokens == {"python": 600, "rust": 400}
    mix.record("python", 550)
    mix.record("rust", 450)

    assert mix.complete
    assert mix.actual_percentages() == {"python": 55.0, "rust": 45.0}


def test_language_mix_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        LanguageMix({"python": 0.2, "rust": 0.2}, total_tokens=100)


def test_allocate_blocks_preserves_total_and_weighting() -> None:
    allocation = allocate_blocks({"python": 0.6, "rust": 0.4}, 11)

    assert allocation == {"python": 7, "rust": 4}
    assert sum(allocation.values()) == 11


def test_stack_row_uses_one_canonical_repository_identity() -> None:
    row = {
        "max_stars_repo_name": "owner/stars",
        "max_stars_repo_path": "src/main.py",
        "max_forks_repo_name": "owner/forks",
        "max_forks_repo_path": "copy.py",
    }

    assert repository_identity(row) == "owner/stars"
    assert repository_path(row) == "src/main.py"


def test_stack_row_rejects_missing_repository_identity() -> None:
    with pytest.raises(ValueError, match="repository identity"):
        repository_identity({"content": "print('hello')"})


def test_causal_nll_shifts_targets_and_ignores_padding() -> None:
    logits = torch.full((1, 4, 3), -20.0)
    input_ids = torch.tensor([[2, 1, 0, 2]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    logits[0, 0, 1] = 20.0
    logits[0, 1, 0] = 20.0
    logits[0, 2, 1] = 20.0  # deliberately wrong, but target position is padded

    total_nll, token_count = causal_nll_from_logits(logits, input_ids, attention_mask)

    assert token_count == 2
    assert total_nll.item() < 1e-6


def test_distributed_block_indices_are_disjoint_and_equal_length() -> None:
    rank_zero = distributed_block_indices(11, rank=0, world_size=2)
    rank_one = distributed_block_indices(11, rank=1, world_size=2)

    assert rank_zero == [0, 2, 4, 6, 8]
    assert rank_one == [1, 3, 5, 7, 9]
    assert set(rank_zero).isdisjoint(rank_one)


def test_training_counters_count_real_tokens_and_microsteps() -> None:
    counters = TrainingCounters(world_size=2)
    counters.record_microstep(local_nonpadding_tokens=4_096, data_wait_seconds=0.25)
    counters.record_microstep(local_nonpadding_tokens=4_096, data_wait_seconds=0.10)
    counters.record_optimizer_step()

    assert counters.training_tokens == 16_384
    assert counters.microsteps == 2
    assert counters.optimizer_steps == 1
    assert counters.data_wait_seconds == pytest.approx(0.35)


def test_milestones_crossed_returns_each_new_threshold_once() -> None:
    milestones = [250_000, 500_000, 1_000_000]

    assert milestones_crossed(240_000, 520_000, milestones) == [250_000, 500_000]
    assert milestones_crossed(520_000, 800_000, milestones) == []


def test_packed_dataset_reads_exact_memmap_blocks(tmp_path) -> None:
    import numpy as np

    path = tmp_path / "blocks.npy"
    np.save(path, np.arange(24, dtype=np.uint32).reshape(4, 6))
    dataset = PackedBlocksDataset(path, start_block=1, block_count=2)

    assert len(dataset) == 2
    assert dataset[0]["input_ids"].tolist() == [6, 7, 8, 9, 10, 11]
    assert torch.equal(dataset[0]["input_ids"], dataset[0]["labels"])
