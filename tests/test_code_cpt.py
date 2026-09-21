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
from tinycomplete.code_cpt.prepare import (
    corpus_fingerprint,
    hash_packed_blocks,
    research_split_for_bucket,
)
from tinycomplete.code_cpt.runtime import ProductionRuntime
from tinycomplete.code_cpt.train import (
    PackedBlocksDataset,
    RunConfig,
    TrainingCounters,
    bounded_optimizer_steps,
    distributed_block_indices,
    extract_mtp_from_snapshot,
    is_broad_deterioration,
    language_mix_for_prefix,
    learning_rate_factor,
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


def test_language_mix_for_consumed_prefix_counts_tokens(tmp_path) -> None:
    import numpy as np

    language_path = tmp_path / "train_languages.npy"
    np.save(language_path, np.array([0, 1, 0, 2, 2], dtype=np.uint8))

    result = language_mix_for_prefix(language_path, ["python", "rust", "go"], 4, 2048)

    assert result["token_counts"] == {"python": 4096, "rust": 2048, "go": 2048}
    assert result["percentages"] == {"python": 50.0, "rust": 25.0, "go": 25.0}


def test_extract_mtp_sidecar_keeps_only_native_mtp_tensors(tmp_path) -> None:
    import json

    from safetensors import safe_open
    from safetensors.torch import save_file

    snapshot = tmp_path / "snapshot"
    destination = tmp_path / "checkpoint"
    snapshot.mkdir()
    save_file(
        {"model.weight": torch.ones(2), "mtp.fc.weight": torch.arange(3)},
        snapshot / "model-00001-of-00001.safetensors",
    )
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.weight": "model-00001-of-00001.safetensors",
                    "mtp.fc.weight": "model-00001-of-00001.safetensors",
                }
            }
        )
    )

    manifest = extract_mtp_from_snapshot(snapshot, destination)

    assert manifest["tensor_count"] == 1
    assert manifest["parameter_count"] == 3
    with safe_open(destination / "mtp-original.safetensors", framework="pt") as handle:
        assert list(handle.keys()) == ["mtp.fc.weight"]


def test_broad_deterioration_requires_clear_multi_language_regression() -> None:
    names = ["python", "typescript", "javascript", "java", "cpp", "rust", "go", "c", "csharp"]
    baseline = {name: {"nll": 1.0} for name in names}
    baseline.update(overall_code={"nll": 1.0}, general={"nll": 1.0})
    current = {name: {"nll": 1.02} for name in names}
    current.update(overall_code={"nll": 1.02}, general={"nll": 1.0})

    assert is_broad_deterioration(current, baseline)
    current["overall_code"]["nll"] = 0.99
    assert not is_broad_deterioration(current, baseline)


def test_optimizer_steps_drop_incomplete_corpus_tail() -> None:
    assert (
        bounded_optimizer_steps(
            remaining_tokens=12_000_000,
            tokens_per_update=32_768,
            available_blocks=5_859,
            blocks_per_update=16,
        )
        == 366
    )


def test_research_split_reserves_old_validation_and_fresh_dev_test() -> None:
    assert research_split_for_bucket(0) == "excluded_stage1_validation"
    assert research_split_for_bucket(9) == "excluded_stage1_validation"
    assert research_split_for_bucket(10) == "development"
    assert research_split_for_bucket(19) == "development"
    assert research_split_for_bucket(20) == "test"
    assert research_split_for_bucket(29) == "test"
    assert research_split_for_bucket(30) == "train"


def test_packed_block_hashes_and_corpus_fingerprint_are_content_bound(tmp_path) -> None:
    import numpy as np

    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.arange(12, dtype=np.uint32).reshape(3, 4))
    np.save(second, np.arange(8, dtype=np.uint32).reshape(2, 4))

    hashes = hash_packed_blocks(first)
    assert len(hashes) == 3
    assert len(set(hashes)) == 3
    before = corpus_fingerprint([first, second])
    values = np.load(second)
    values[0, 0] = 99
    np.save(second, values)
    assert corpus_fingerprint([first, second]) != before


def test_production_runtime_matches_round2_winner() -> None:
    runtime = ProductionRuntime()
    assert runtime.gradient_accumulation == 8
    assert runtime.backward_prefetch == "BACKWARD_PRE"
    assert runtime.forward_prefetch is True
    assert runtime.compile_mode == "default"
    assert runtime.compile_dynamic is False
    assert runtime.initial_loss_scale == 256.0


def test_declared_constant_and_cosine_learning_rate_schedules() -> None:
    assert learning_rate_factor(0, "constant", 5, 153, 0.1) == pytest.approx(0.2)
    assert learning_rate_factor(4, "constant", 5, 153, 0.1) == pytest.approx(1.0)
    assert learning_rate_factor(152, "constant", 5, 153, 0.1) == pytest.approx(1.0)
    assert learning_rate_factor(4, "cosine", 5, 153, 0.1) == pytest.approx(1.0)
    assert learning_rate_factor(5, "cosine", 5, 153, 0.1) == pytest.approx(1.0)
    assert learning_rate_factor(152, "cosine", 5, 153, 0.1) == pytest.approx(0.1)
    assert learning_rate_factor(153, "cosine", 5, 153, 0.1) == pytest.approx(0.1)


def test_weight_initialization_and_exact_resume_are_mutually_exclusive(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RunConfig(
            corpus_dir=tmp_path,
            output_dir=tmp_path,
            learning_rate=3e-6,
            init_from=tmp_path / "weights",
            resume_from=tmp_path / "state",
        )
