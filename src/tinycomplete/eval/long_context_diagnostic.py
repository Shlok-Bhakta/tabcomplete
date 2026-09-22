"""Matched long-context dependency diagnostic and target-only NLL helpers."""

from __future__ import annotations

import hashlib
import random
import string
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from tinycomplete.eval.code_benchmark import BenchmarkCase, CheckSpec

DiagnosticFamily = Literal["constant", "enum", "signature", "field", "config"]
DiagnosticCondition = Literal[
    "short_control", "long_near", "long_far", "absent", "counterfactual"
]
CONDITIONS: tuple[DiagnosticCondition, ...] = (
    "short_control",
    "long_near",
    "long_far",
    "absent",
    "counterfactual",
)


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


class LongContextDiagnosticCase(BaseModel):
    id: str
    schema_version: int = 2
    family: DiagnosticFamily
    condition: DiagnosticCondition
    seed: int
    requested_context_tokens: int
    prompt_tokens: int
    prompt: str
    target: str
    distractor: str
    test_code: str
    dependency_token_position: int | None
    dependency_distance_tokens: int | None
    prompt_sha256: str


def _token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _identifier(rng: random.Random, prefix: str) -> str:
    suffix = "".join(rng.choice(string.ascii_uppercase) for _ in range(7))
    return f"{prefix}_{suffix}"


def _fixture(
    family: DiagnosticFamily, seed: int, *, counterfactual: bool
) -> tuple[str, str, str, str, str]:
    rng = random.Random(seed)
    first_value = rng.randrange(10_000, 90_000)
    second_value = first_value + rng.randrange(101, 997)
    chosen_value = second_value if counterfactual else first_value
    other_value = first_value if counterfactual else second_value
    if family == "constant":
        name = _identifier(rng, "BACKGROUND_RETRY_MS")
        dependency = f'"""Shared retry contract."""\n\n{name} = {chosen_value}\n'
        prefix = (
            f"from repository.contract import {name}\n\n"
            "def retry_delay_ms() -> int:\n    return "
        )
        test = (
            "from solution import retry_delay_ms\n"
            f"assert retry_delay_ms() == {chosen_value}\n"
        )
        return dependency, prefix, str(chosen_value), str(other_value), test
    if family == "enum":
        first = _identifier(rng, "WIRE")
        second = _identifier(rng, "WIRE")
        chosen = second if counterfactual else first
        other = first if counterfactual else second
        dependency = (
            "from enum import Enum\n\n"
            "class WireFormat(str, Enum):\n"
            f'    {first} = "{first.lower()}"\n'
            f'    {second} = "{second.lower()}"\n\n'
            f"ACTIVE_WIRE_FORMAT = WireFormat.{chosen}\n"
        )
        prefix = (
            "from repository.contract import WireFormat\n\n"
            "def wire_format() -> WireFormat:\n    return "
        )
        test = (
            "from solution import wire_format\n"
            f'assert wire_format().name == "{chosen}"\n'
        )
        return dependency, prefix, f"WireFormat.{chosen}", f"WireFormat.{other}", test
    if family == "signature":
        first = _identifier(rng, "payload")
        second = _identifier(rng, "payload")
        chosen = second if counterfactual else first
        other = first if counterfactual else second
        dependency = (
            "def encode_payload(payload: bytes, *, "
            f"{chosen}: bool, checksum: bool) -> bytes:\n"
            f"    return bytes([{chosen}, checksum]) + payload\n"
        )
        prefix = (
            "from repository.contract import encode_payload\n\n"
            "def encode_event(payload: bytes) -> bytes:\n    return "
        )
        test = (
            "from solution import encode_event\n"
            "assert encode_event(b\"x\") == bytes([1, 1]) + b\"x\"\n"
        )
        return (
            dependency,
            prefix,
            f"encode_payload(payload, {chosen}=True, checksum=True)",
            f"encode_payload(payload, {other}=True, checksum=True)",
            test,
        )
    if family == "field":
        first = _identifier(rng, "connect_timeout")
        second = _identifier(rng, "request_timeout")
        chosen = second if counterfactual else first
        other = first if counterfactual else second
        dependency = (
            "from dataclasses import dataclass\n\n"
            "@dataclass(frozen=True)\n"
            "class UploadPolicy:\n"
            f"    {first}: int\n"
            f"    {second}: int\n\n"
            f"UPLOAD_POLICY = UploadPolicy({first}={first_value}, {second}={second_value})\n"
            f'PRIMARY_TIMEOUT_FIELD = "{chosen}"\n'
        )
        prefix = (
            "from repository.contract import UPLOAD_POLICY\n\n"
            "def primary_timeout() -> int:\n    return "
        )
        expected_value = second_value if chosen == second else first_value
        test = (
            "from solution import primary_timeout\n"
            f"assert primary_timeout() == {expected_value}\n"
        )
        return (
            dependency,
            prefix,
            f"UPLOAD_POLICY.{chosen}",
            f"UPLOAD_POLICY.{other}",
            test,
        )
    first = _identifier(rng, "service").lower()
    second = _identifier(rng, "service").lower()
    chosen = second if counterfactual else first
    other = first if counterfactual else second
    dependency = (
        "SERVICE_LIMITS = {\n"
        f'    "{first}": {{"steady": 3, "burst": {first_value}}},\n'
        f'    "{second}": {{"steady": 5, "burst": {second_value}}},\n'
        "}\n"
        f'ACTIVE_SERVICE = "{chosen}"\n'
    )
    prefix = (
        "from repository.contract import SERVICE_LIMITS\n\n"
        "def active_burst_limit() -> int:\n    return "
    )
    expected_value = second_value if chosen == second else first_value
    test = (
        "from solution import active_burst_limit\n"
        f"assert active_burst_limit() == {expected_value}\n"
    )
    return (
        dependency,
        prefix,
        f'SERVICE_LIMITS["{chosen}"]["burst"]',
        f'SERVICE_LIMITS["{other}"]["burst"]',
        test,
    )


def _filler(index: int, seed: int) -> str:
    rng = random.Random(seed * 1_000_003 + index)
    values = [rng.randrange(100_000, 900_000) for _ in range(8)]
    return (
        f'"""Independent telemetry component {index:05d}."""\n\n'
        f"VALUES_{index:05d} = {values!r}\n\n"
        f"def normalize_component_{index:05d}(value: int) -> int:\n"
        f"    return (value * {rng.randrange(17, 97)} + {rng.randrange(101, 997)}) % "
        f"{rng.randrange(1009, 5003)}\n"
    )


def _prompt_parts(
    family: DiagnosticFamily,
    condition: DiagnosticCondition,
    filler_count: int,
    seed: int,
) -> tuple[str, str, str, str, str | None]:
    counterfactual = condition == "counterfactual"
    dependency, prefix, target, distractor, test_code = _fixture(
        family, seed, counterfactual=counterfactual
    )
    filler_directory = (
        "z_components" if condition in {"long_far", "counterfactual"} else "a_components"
    )
    files = {
        f"repository/{filler_directory}/component_{index:05d}.py": _filler(index, seed)
        for index in range(filler_count)
    }
    dependency_path = None
    if condition != "absent":
        dependency_path = "repository/contract.py"
        files[dependency_path] = dependency
    pieces = [
        f'<file path="{path}">\n{content}\n</file>\n'
        for path, content in sorted(files.items())
    ]
    pieces.append(f'<target path="solution.py" language="python">\n{prefix}')
    return "".join(pieces), target, distractor, test_code, dependency_path


def _calibrated_prompt(
    tokenizer: Tokenizer,
    family: DiagnosticFamily,
    condition: DiagnosticCondition,
    target_tokens: int,
    seed: int,
    tolerance_fraction: float,
) -> tuple[str, str, str, str, str | None]:
    if condition == "short_control":
        return _prompt_parts(family, condition, 0, seed)
    base = _prompt_parts(family, condition, 0, seed)
    probe = _prompt_parts(family, condition, 4, seed)
    base_tokens = _token_count(tokenizer, base[0])
    per_filler = max(1.0, (_token_count(tokenizer, probe[0]) - base_tokens) / 4)
    estimate = max(0, round((target_tokens - base_tokens) / per_filler))
    best = None
    for count in range(max(0, estimate - 8), estimate + 9):
        candidate = _prompt_parts(family, condition, count, seed)
        tokens = _token_count(tokenizer, candidate[0])
        if best is None or abs(tokens - target_tokens) < abs(best[0] - target_tokens):
            best = (tokens, candidate)
    assert best is not None
    tolerance = max(64, round(target_tokens * tolerance_fraction))
    if abs(best[0] - target_tokens) > tolerance:
        raise RuntimeError(
            f"could not calibrate {family}/{condition}: target={target_tokens}, actual={best[0]}"
        )
    return best[1]


def build_diagnostic_family(
    *,
    tokenizer: Tokenizer,
    family: DiagnosticFamily,
    target_tokens: int,
    seed: int,
    tolerance_fraction: float = 0.04,
) -> list[LongContextDiagnosticCase]:
    """Build five matched dependency conditions for one family and context size."""
    cases = []
    for condition in CONDITIONS:
        prompt, target, distractor, test_code, dependency_path = _calibrated_prompt(
            tokenizer,
            family,
            condition,
            target_tokens,
            seed,
            tolerance_fraction,
        )
        prompt_tokens = _token_count(tokenizer, prompt)
        position = None
        distance = None
        if dependency_path is not None:
            marker = f'<file path="{dependency_path}">\n'
            position = _token_count(tokenizer, prompt[: prompt.index(marker)])
            distance = prompt_tokens - position
        cases.append(
            LongContextDiagnosticCase(
                id=f"{family}/{target_tokens}/{condition}/seed-{seed}",
                family=family,
                condition=condition,
                seed=seed,
                requested_context_tokens=target_tokens,
                prompt_tokens=prompt_tokens,
                prompt=prompt,
                target=target,
                distractor=distractor,
                test_code=test_code,
                dependency_token_position=position,
                dependency_distance_tokens=distance,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            )
        )
    return cases


def diagnostic_case_to_benchmark(case: LongContextDiagnosticCase) -> BenchmarkCase:
    """Recover an executable Python fixture from the serialized diagnostic prompt."""
    import re

    file_pattern = re.compile(r'<file path="([^"]+)">\n(.*?)\n</file>\n', re.DOTALL)
    context_files = {path: content for path, content in file_pattern.findall(case.prompt)}
    target_marker = '<target path="solution.py" language="python">\n'
    if target_marker not in case.prompt:
        raise ValueError("diagnostic prompt has no target marker")
    prefix = case.prompt.split(target_marker, 1)[1]
    return BenchmarkCase(
        id=case.id,
        language="python",
        path="solution.py",
        prefix=prefix,
        expected=case.target,
        context_files=context_files,
        check=CheckSpec(
            compile=["python3", "-m", "py_compile", "solution.py"],
            test=["python3", "tests.py"],
            files={"tests.py": case.test_code},
            container_image="docker.io/library/python:3.12-slim",
        ),
        category=f"long_context_v2_{case.family}_{case.condition}",
        repository_context=True,
    )


def target_logit_positions(prompt_length: int, target_length: int, device: Any):
    """Positions whose logits predict the target, excluding every prompt target."""
    import torch

    if prompt_length < 1 or target_length < 1:
        raise ValueError("prompt and target must each contain at least one token")
    return torch.arange(
        prompt_length - 1,
        prompt_length + target_length - 1,
        device=device,
        dtype=torch.long,
    )


def selected_target_nll(logits, target_ids) -> tuple[Any, int]:
    """Score target IDs against logits computed only for target prediction positions."""
    import torch.nn.functional as F

    if logits.ndim != 3 or target_ids.ndim != 2:
        raise ValueError("expected logits [batch,target,vocab] and target_ids [batch,target]")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("selected logits and target token dimensions differ")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction="sum"
    )
    return losses, target_ids.numel()


def encode_prompt_and_target(tokenizer, prompt: str, target: str) -> tuple[list[int], list[int]]:
    """Tokenize prompt and target separately so prompt tokens are never scored."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not prompt_ids or not target_ids:
        raise ValueError("prompt and target must each tokenize to at least one token")
    return prompt_ids, target_ids


def score_target_continuation(model, tokenizer, prompt: str, target: str, device) -> dict[str, Any]:
    """Score only target logits, avoiding a context-length by vocabulary allocation."""
    import torch

    prompt_ids, target_ids = encode_prompt_and_target(tokenizer, prompt, target)
    combined = torch.tensor([prompt_ids + target_ids], device=device, dtype=torch.long)
    target_tensor = torch.tensor([target_ids], device=device, dtype=torch.long)
    positions = target_logit_positions(len(prompt_ids), len(target_ids), device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        output = model(input_ids=combined, use_cache=False, logits_to_keep=positions)
    nll_sum, token_count = selected_target_nll(output.logits.float(), target_tensor)
    return {
        "prompt_tokens": len(prompt_ids),
        "target_tokens": token_count,
        "target_nll_sum": float(nll_sum.item()),
        "target_nll_mean": float(nll_sum.item()) / token_count,
        "selected_logit_positions": [int(positions[0].item()), int(positions[-1].item())],
    }


def _stream_prefix_cache(
    model: Any,
    token_ids: list[int],
    device: Any,
    *,
    chunk_tokens: int,
) -> Any:
    """Prefill a causal cache without a quadratic full-prompt attention allocation."""
    import torch

    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    cache = None
    for start in range(0, len(token_ids), chunk_tokens):
        chunk = torch.tensor(
            [token_ids[start : start + chunk_tokens]], device=device, dtype=torch.long
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(
                input_ids=chunk,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        cache = output.past_key_values
    return cache


def score_target_continuation_streamed(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
    device: Any,
    *,
    chunk_tokens: int = 2048,
) -> dict[str, Any]:
    """Score target tokens after chunked prompt prefill, excluding every prompt position."""
    import torch

    prompt_ids, target_ids = encode_prompt_and_target(tokenizer, prompt, target)
    cache = _stream_prefix_cache(
        model,
        prompt_ids[:-1],
        device,
        chunk_tokens=chunk_tokens,
    )
    scoring_ids = [prompt_ids[-1], *target_ids[:-1]]
    inputs = torch.tensor([scoring_ids], device=device, dtype=torch.long)
    target_tensor = torch.tensor([target_ids], device=device, dtype=torch.long)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        output = model(
            input_ids=inputs,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=len(target_ids),
        )
    nll_sum, token_count = selected_target_nll(output.logits.float(), target_tensor)
    return {
        "prompt_tokens": len(prompt_ids),
        "target_tokens": token_count,
        "target_nll_sum": float(nll_sum.item()),
        "target_nll_mean": float(nll_sum.item()) / token_count,
        "selected_logit_positions": [
            len(prompt_ids) - 1,
            len(prompt_ids) + len(target_ids) - 2,
        ],
        "streaming_chunk_tokens": chunk_tokens,
    }


def greedy_generate_streamed(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    *,
    max_new_tokens: int,
    chunk_tokens: int = 2048,
) -> dict[str, Any]:
    """Greedy generation after chunked prompt prefill."""
    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("prompt must tokenize to at least one token")
    cache = _stream_prefix_cache(
        model,
        prompt_ids[:-1],
        device,
        chunk_tokens=chunk_tokens,
    )
    current = torch.tensor([[prompt_ids[-1]]], device=device, dtype=torch.long)
    generated: list[int] = []
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    finish_reason = "length"
    for _ in range(max_new_tokens):
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(
                input_ids=current,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        cache = output.past_key_values
        token = int(output.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(token)
        if token in eos_ids:
            finish_reason = "eos_or_stop"
            break
        current = torch.tensor([[token]], device=device, dtype=torch.long)
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "tokens": len(generated),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "streaming_chunk_tokens": chunk_tokens,
    }


def verify_streamed_scoring(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
    device: Any,
    *,
    chunk_tokens: int = 2048,
) -> dict[str, float]:
    """Compare cache-streamed target NLL with full-logit scoring on a short input."""
    full = score_target_continuation(model, tokenizer, prompt, target, device)
    streamed = score_target_continuation_streamed(
        model,
        tokenizer,
        prompt,
        target,
        device,
        chunk_tokens=chunk_tokens,
    )
    return {
        "full_selected_nll_sum": float(full["target_nll_sum"]),
        "streamed_nll_sum": float(streamed["target_nll_sum"]),
        "absolute_difference": abs(
            float(full["target_nll_sum"]) - float(streamed["target_nll_sum"])
        ),
    }


def verify_selected_scoring(model, tokenizer, prompt: str, target: str, device) -> dict[str, float]:
    """Compare selected-logit scoring with full logits on a bounded short input."""
    import torch

    prompt_ids, target_ids = encode_prompt_and_target(tokenizer, prompt, target)
    combined = torch.tensor([prompt_ids + target_ids], device=device, dtype=torch.long)
    target_tensor = torch.tensor([target_ids], device=device, dtype=torch.long)
    positions = target_logit_positions(len(prompt_ids), len(target_ids), device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        selected = model(input_ids=combined, use_cache=False, logits_to_keep=positions).logits
        full = model(input_ids=combined, use_cache=False, logits_to_keep=0).logits[:, positions, :]
    selected_sum, _ = selected_target_nll(selected.float(), target_tensor)
    full_sum, _ = selected_target_nll(full.float(), target_tensor)
    return {
        "selected_nll_sum": float(selected_sum.item()),
        "full_nll_sum": float(full_sum.item()),
        "absolute_difference": abs(float(selected_sum.item() - full_sum.item())),
    }
