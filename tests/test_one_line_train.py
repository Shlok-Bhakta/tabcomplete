from __future__ import annotations

import random
import time
from dataclasses import asdict

import pytest

from tinycomplete.one_line.contract import EditAction, EditState
from tinycomplete.one_line.train import (
    EFFECTIVE_BATCH,
    IGNORE_INDEX,
    CosineUpdateSchedule,
    EncodedExample,
    TrainingCursor,
    batch_order_sha256,
    bucketed_batches,
    collate_examples,
    encode_training_row,
    enforce_training_budget,
    example_weighted_causal_loss,
    load_resume_checkpoint,
    save_resume_checkpoint,
    selected_position_causal_loss,
    token_counts,
    train_encoded,
)


class CharacterTokenizer:
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return ([3] if add_special_tokens else []) + [ord(character) + 4 for character in text]


def _state(source: str = "x = 1\n") -> EditState:
    return EditState("public/example.py", "python", source, 0, 0)


def test_response_and_eos_are_the_only_supervised_positions() -> None:
    tokenizer = CharacterTokenizer()
    keep = encode_training_row(tokenizer, _state(), EditAction("keep"))
    replacement = encode_training_row(tokenizer, _state(), EditAction("replace_line", "λ"))
    assert keep.response == "N"
    assert keep.labels[: keep.prompt_tokens] == (IGNORE_INDEX,) * keep.prompt_tokens
    assert keep.labels[keep.prompt_tokens :] == keep.input_ids[keep.prompt_tokens :]
    assert keep.labels[-1] == tokenizer.eos_token_id
    assert replacement.response_tokens > keep.response_tokens
    counts = token_counts((keep, replacement))
    assert counts["nonpadding_training_input_tokens"] == (
        keep.total_tokens + replacement.total_tokens
    )
    assert counts["supervised_response_and_eos_tokens"] == (
        keep.response_tokens + replacement.response_tokens
    )


def test_padding_is_masked_by_position_even_when_pad_equals_eos() -> None:
    pytest.importorskip("torch")
    tokenizer = CharacterTokenizer()
    examples = [
        encode_training_row(tokenizer, _state(), EditAction("keep")),
        encode_training_row(tokenizer, _state(), EditAction("replace_line", "longer")),
    ]
    batch = collate_examples(examples, pad_token_id=tokenizer.eos_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"][0, -1].item() == IGNORE_INDEX
    assert batch["attention_mask"][0, -1].item() == 0
    assert batch["labels"][1, -1].item() == tokenizer.eos_token_id


def test_long_action_is_excluded_instead_of_truncated() -> None:
    with pytest.raises(ValueError, match="out of scope"):
        encode_training_row(CharacterTokenizer(), _state(), EditAction("replace_line", "x" * 64))


def test_training_budget_and_batch_defaults() -> None:
    assert EFFECTIVE_BATCH == 32
    example = encode_training_row(CharacterTokenizer(), _state(), EditAction("keep"))
    assert enforce_training_budget([example], passes=2) == 2 * example.total_tokens
    with pytest.raises(ValueError, match="two full"):
        enforce_training_budget([example], passes=3)
    with pytest.raises(ValueError, match="budget"):
        enforce_training_budget([example], passes=1, used_tokens=100_000_000)


def test_example_weighted_loss_matches_independent_example_means() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [[0.0, 0.0], [5.0, -5.0], [-5.0, 5.0], [0.0, 0.0]],
            [[0.0, 0.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 0, 1, -100], [-100, 1, 1, 1]])
    expected = []
    for row in range(2):
        active = labels[row, 1:] != -100
        expected.append(
            torch.nn.functional.cross_entropy(logits[row, :-1][active], labels[row, 1:][active])
        )
    result = example_weighted_causal_loss(logits, labels)
    assert torch.allclose(result, torch.stack(expected).mean())
    result.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_accumulation_equivalent_to_nonaccumulated_example_weighted_update() -> None:
    torch = pytest.importorskip("torch")
    xs = torch.tensor([[float(index), 1.0] for index in range(4)])
    ys = torch.tensor([float(index % 2) for index in range(4)])
    first = torch.nn.Linear(2, 1, bias=False)
    second = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        first.weight.fill_(0.25)
        second.weight.copy_(first.weight)
    for x, y in zip(xs, ys, strict=True):
        (((first(x).squeeze() - y) ** 2) / len(xs)).backward()
    ((second(xs).squeeze() - ys) ** 2).mean().backward()
    assert torch.allclose(first.weight.grad, second.weight.grad, rtol=1e-6, atol=1e-6)


def test_selected_logits_match_full_loss_gradients_and_update_for_mixed_lengths() -> None:
    torch = pytest.importorskip("torch")

    class TinyCausal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)

        def forward(self, *, input_ids, attention_mask, use_cache=False, logits_to_keep=0):
            del attention_mask, use_cache
            hidden = self.embed(input_ids)
            if isinstance(logits_to_keep, torch.Tensor):
                hidden = hidden.index_select(1, logits_to_keep)
            return type("Output", (), {"logits": self.head(hidden)})()

    torch.manual_seed(17)
    full = TinyCausal()
    selected = TinyCausal()
    selected.load_state_dict(full.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 2], [3, 4, 5, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, 3, 4, -100, -100], [-100, -100, -100, -100, 7, 8]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    full_loss = example_weighted_causal_loss(
        full(input_ids=input_ids, attention_mask=mask).logits, labels
    )
    selected_loss = selected_position_causal_loss(
        selected, input_ids=input_ids, labels=labels, attention_mask=mask
    )
    assert torch.allclose(full_loss, selected_loss, rtol=1e-6, atol=1e-6)
    full_loss.backward()
    selected_loss.backward()
    for (full_name, full_param), (selected_name, selected_param) in zip(
        full.named_parameters(), selected.named_parameters(), strict=True
    ):
        assert full_name == selected_name
        assert torch.allclose(full_param.grad, selected_param.grad, rtol=1e-6, atol=1e-6)
    full_step = torch.optim.SGD(full.parameters(), lr=0.01)
    selected_step = torch.optim.SGD(selected.parameters(), lr=0.01)
    full_step.step()
    selected_step.step()
    for full_param, selected_param in zip(full.parameters(), selected.parameters(), strict=True):
        assert torch.allclose(full_param, selected_param, rtol=1e-6, atol=1e-6)


def test_installed_qwen2_selected_logits_agree_with_full_logits() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(17)
    full = Qwen2ForCausalLM(config)
    selected = Qwen2ForCausalLM(config)
    selected.load_state_dict(full.state_dict())
    full.eval()
    selected.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 2], [3, 4, 5, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, 3, 4, -100, -100], [-100, -100, -100, -100, 7, 8]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    full_loss = example_weighted_causal_loss(
        full(input_ids=ids, attention_mask=mask, use_cache=False).logits, labels
    )
    selected_loss = selected_position_causal_loss(
        selected, input_ids=ids, labels=labels, attention_mask=mask
    )
    assert torch.allclose(full_loss, selected_loss, rtol=1e-5, atol=1e-6)
    full_loss.backward()
    selected_loss.backward()
    for full_parameter, selected_parameter in zip(
        full.parameters(), selected.parameters(), strict=True
    ):
        assert torch.allclose(full_parameter.grad, selected_parameter.grad, rtol=1e-4, atol=1e-5)


def test_resume_checkpoint_restores_update_state_and_rng(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    optimizer.zero_grad()
    model(torch.ones(2)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "resume.pt"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state().clone()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_resume_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        fingerprint="model-data-config-identity",
        next_example_index=32,
        completed_updates=1,
        training_input_tokens=2048,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    random.random()
    np.random.random()
    torch.rand(1)
    loaded = load_resume_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_fingerprint="model-data-config-identity",
    )
    assert loaded == {
        "next_example_index": 32,
        "completed_updates": 1,
        "attempted_updates": 0,
        "skipped_updates": 0,
        "training_input_tokens": 2048,
        "supervised_target_tokens": 0,
        "epoch": 0,
    }
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())
    assert random.getstate() == saved_python
    assert np.array_equal(np.random.get_state()[1], saved_numpy[1])
    assert torch.equal(torch.get_rng_state(), saved_torch)
    with pytest.raises(ValueError, match="identity"):
        load_resume_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_fingerprint="different",
        )


def _tiny_examples() -> list[EncodedExample]:
    return [
        EncodedExample(
            input_ids=(1, 2, 3 + index, 4 + index),
            labels=(-100, -100, 3 + index, 4 + index),
            prompt_tokens=2,
            response_tokens=2,
            total_tokens=4,
            prompt="toy",
            response="toy",
        )
        for index in range(4)
    ]


def test_bucket_order_and_cosine_schedule_are_fixed() -> None:
    torch = pytest.importorskip("torch")
    examples = _tiny_examples()
    batches = bucketed_batches(examples, epochs=2, effective_batch=2, seed=7)
    assert len(batches) == 4
    assert sorted(index for batch in batches[:2] for index in batch) == list(range(4))
    assert sorted(index for batch in batches[2:] for index in batch) == list(range(4))
    assert batches == bucketed_batches(examples, epochs=2, effective_batch=2, seed=7)
    assert batch_order_sha256(batches) != batch_order_sha256(
        bucketed_batches(examples, epochs=2, effective_batch=2, seed=8)
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    schedule = CosineUpdateSchedule(optimizer, peak_lr=0.01, total_updates=100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 / 3)
    for _ in range(100):
        schedule.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    with pytest.raises(ValueError, match="horizon"):
        schedule.step()


def test_exact_resume_matches_continuous_cpu_updates(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    class TinyCausal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16)

        def forward(self, *, input_ids, attention_mask, use_cache=False, logits_to_keep=0):
            del attention_mask, use_cache
            hidden = self.embed(input_ids)
            if isinstance(logits_to_keep, torch.Tensor):
                hidden = hidden.index_select(1, logits_to_keep)
            return type("Output", (), {"logits": self.head(hidden)})()

    examples = _tiny_examples()
    batches = ((0, 1), (2, 3))
    torch.manual_seed(19)
    continuous = TinyCausal()
    interrupted = TinyCausal()
    interrupted.load_state_dict(continuous.state_dict())

    def components(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        schedule = CosineUpdateSchedule(optimizer, peak_lr=0.01, total_updates=2)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        return optimizer, schedule, scaler

    continuous_opt, continuous_schedule, continuous_scaler = components(continuous)
    continuous_result = train_encoded(
        continuous,
        examples,
        batches,
        optimizer=continuous_opt,
        scheduler=continuous_schedule,
        scaler=continuous_scaler,
        device=torch.device("cpu"),
        pad_token_id=2,
        microbatch_examples=1,
        finalization_reserve_seconds=0,
    )
    assert continuous_result.status == "complete"
    interrupted_opt, interrupted_schedule, interrupted_scaler = components(interrupted)
    checkpoint = tmp_path / "update-one.pt"

    class StopAfterFirstUpdate(Exception):
        pass

    def stop(cursor: TrainingCursor) -> None:
        save_resume_checkpoint(
            checkpoint,
            model=interrupted,
            optimizer=interrupted_opt,
            scheduler=interrupted_schedule,
            scaler=interrupted_scaler,
            fingerprint="fixed-order-and-environment",
            **asdict(cursor),
        )
        raise StopAfterFirstUpdate

    with pytest.raises(StopAfterFirstUpdate):
        train_encoded(
            interrupted,
            examples,
            batches,
            optimizer=interrupted_opt,
            scheduler=interrupted_schedule,
            scaler=interrupted_scaler,
            device=torch.device("cpu"),
            pad_token_id=2,
            microbatch_examples=1,
            checkpoint_every_updates=1,
            on_checkpoint=stop,
            finalization_reserve_seconds=0,
        )
    restarted = TinyCausal()
    restarted_opt, restarted_schedule, restarted_scaler = components(restarted)
    restored = TrainingCursor(
        **load_resume_checkpoint(
            checkpoint,
            model=restarted,
            optimizer=restarted_opt,
            scheduler=restarted_schedule,
            scaler=restarted_scaler,
            expected_fingerprint="fixed-order-and-environment",
        )
    )
    assert restored.next_example_index == 2 and restored.attempted_updates == 1
    resumed_result = train_encoded(
        restarted,
        examples,
        batches,
        optimizer=restarted_opt,
        scheduler=restarted_schedule,
        scaler=restarted_scaler,
        device=torch.device("cpu"),
        pad_token_id=2,
        microbatch_examples=1,
        cursor=restored,
        finalization_reserve_seconds=0,
    )
    assert resumed_result.status == "complete"
    assert resumed_result.cursor == continuous_result.cursor
    for expected, actual in zip(continuous.parameters(), restarted.parameters(), strict=True):
        assert torch.allclose(expected, actual, atol=1e-7, rtol=1e-7)
    assert restarted_opt.state_dict()["state"].keys() == continuous_opt.state_dict()["state"].keys()


def test_deadline_stops_at_update_boundary_and_rejects_bad_resume_cursor() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Embedding(16, 8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    schedule = CosineUpdateSchedule(optimizer, peak_lr=0.01, total_updates=1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    snapshots = []
    result = train_encoded(
        model,
        _tiny_examples(),
        ((0, 1),),
        optimizer=optimizer,
        scheduler=schedule,
        scaler=scaler,
        device=torch.device("cpu"),
        pad_token_id=2,
        deadline_monotonic=time.monotonic() - 1,
        finalization_reserve_seconds=0,
        on_checkpoint=snapshots.append,
    )
    assert result.status == "deadline_stop"
    assert snapshots == [TrainingCursor()]
    with pytest.raises(ValueError, match="boundary"):
        train_encoded(
            model,
            _tiny_examples(),
            ((0, 1),),
            optimizer=optimizer,
            scheduler=schedule,
            scaler=scaler,
            device=torch.device("cpu"),
            pad_token_id=2,
            cursor=TrainingCursor(next_example_index=1),
        )
