"""Loss calculations shared by Stage-1 baseline and checkpoint evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def causal_nll_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Return summed next-token NLL and the exact number of scored tokens."""
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [batch, seq, vocab] and input_ids [batch, seq]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids batch/sequence dimensions differ")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = input_ids[:, 1:].contiguous()
    if attention_mask is None:
        target_mask = torch.ones_like(shifted_targets, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape differs from input_ids")
        target_mask = attention_mask[:, 1:].to(dtype=torch.bool)
    token_count = int(target_mask.sum().item())
    if token_count == 0:
        return logits.new_zeros(()), 0
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_targets.view(-1),
        reduction="none",
    ).view_as(shifted_targets)
    return losses.masked_select(target_mask).sum(), token_count


def attribute_token_losses(
    shifted_losses: torch.Tensor, spans: list[dict]
) -> dict[str, dict[str, float | int]]:
    """Assign next-token losses to repository spans, excluding boundaries."""
    if shifted_losses.ndim != 1:
        raise ValueError("shifted_losses must have one value per scored target position")
    result: dict[str, dict[str, float | int]] = {}
    for span in spans:
        repository = str(span["repository"])
        if repository == "__boundary__":
            continue
        start = max(1, int(span["start"]))
        end = min(int(span["end"]), shifted_losses.numel() + 1)
        if start >= end:
            continue
        values = shifted_losses[start - 1 : end - 1]
        row = result.setdefault(repository, {"nll_sum": 0.0, "tokens": 0})
        row["nll_sum"] = float(row["nll_sum"]) + float(values.sum().item())
        row["tokens"] = int(row["tokens"]) + values.numel()
    return result


def paired_repository_bootstrap(
    first: list[dict],
    second: list[dict],
    *,
    samples: int = 2_000,
    seed: int = 271828,
) -> dict:
    """Estimate paired NLL differences by resampling shared repository identities."""
    if samples < 1:
        raise ValueError("samples must be positive")

    def grouped(rows: list[dict]) -> dict[str, dict[str, dict[str, float | int]]]:
        output: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
        for row in rows:
            repository = str(row["repository"])
            language = str(row["language"])
            current = output[repository].setdefault(language, {"nll_sum": 0.0, "tokens": 0})
            current["nll_sum"] = float(current["nll_sum"]) + float(row["nll_sum"])
            current["tokens"] = int(current["tokens"]) + int(row["tokens"])
        return output

    left = grouped(first)
    right = grouped(second)
    repositories = sorted(set(left) & set(right))
    if not repositories:
        raise ValueError("paired comparison has no shared repositories")

    def metrics(sampled: list[str]) -> tuple[float, float]:
        language_left: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        language_right: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for repository in sampled:
            shared_languages = set(left[repository]) & set(right[repository])
            for language in shared_languages:
                left_row = left[repository][language]
                right_row = right[repository][language]
                if int(left_row["tokens"]) != int(right_row["tokens"]):
                    raise ValueError("paired repository token counts differ")
                language_left[language][0] += float(left_row["nll_sum"])
                language_left[language][1] += int(left_row["tokens"])
                language_right[language][0] += float(right_row["nll_sum"])
                language_right[language][1] += int(right_row["tokens"])
        languages = sorted(set(language_left) & set(language_right))
        left_sum = sum(language_left[name][0] for name in languages)
        right_sum = sum(language_right[name][0] for name in languages)
        tokens = sum(language_left[name][1] for name in languages)
        token_weighted = right_sum / tokens - left_sum / tokens
        balanced = float(
            np.mean(
                [
                    language_right[name][0] / language_right[name][1]
                    - language_left[name][0] / language_left[name][1]
                    for name in languages
                ]
            )
        )
        return token_weighted, balanced

    token_point, balanced_point = metrics(repositories)
    generator = np.random.default_rng(seed)
    token_samples = []
    balanced_samples = []
    for _ in range(samples):
        indices = generator.integers(0, len(repositories), len(repositories))
        sampled = [repositories[index] for index in indices]
        token_value, balanced_value = metrics(sampled)
        token_samples.append(token_value)
        balanced_samples.append(balanced_value)
    return {
        "repository_count": len(repositories),
        "samples": samples,
        "seed": seed,
        "difference_definition": "second minus first",
        "token_weighted_difference": token_point,
        "token_weighted_95ci": [
            float(np.quantile(token_samples, 0.025)),
            float(np.quantile(token_samples, 0.975)),
        ],
        "balanced_language_difference": balanced_point,
        "balanced_language_95ci": [
            float(np.quantile(balanced_samples, 0.025)),
            float(np.quantile(balanced_samples, 0.975)),
        ],
    }


def select_development_candidate(
    *,
    aggregate_nll: dict[str, float],
    parent_by_candidate: dict[str, str],
    repository_rows: dict[str, list[dict]],
    samples: int = 2_000,
    seed: int = 271828,
) -> dict:
    """Select only a candidate whose balanced repository improvement is resolved."""
    comparisons = {}
    eligible = []
    for candidate, parent in parent_by_candidate.items():
        comparison = paired_repository_bootstrap(
            repository_rows[parent],
            repository_rows[candidate],
            samples=samples,
            seed=seed,
        )
        comparisons[f"{candidate}-minus-{parent}"] = comparison
        if (
            aggregate_nll[candidate] < aggregate_nll[parent]
            and comparison["balanced_language_95ci"][1] < 0
        ):
            eligible.append(candidate)

    best = min(eligible, key=lambda name: aggregate_nll[name]) if eligible else None
    selected = best
    if best is not None:
        for challenger in eligible:
            if challenger == best:
                continue
            comparison = paired_repository_bootstrap(
                repository_rows[challenger],
                repository_rows[best],
                samples=samples,
                seed=seed,
            )
            comparisons[f"{best}-minus-{challenger}"] = comparison
            if comparison["balanced_language_95ci"][1] >= 0:
                selected = None
    return {
        "development_metric": "balanced-language NLL with paired repository bootstrap",
        "aggregate_nll": aggregate_nll,
        "comparisons": comparisons,
        "eligible_improvements": eligible,
        "selected_candidate": selected,
        "result": "unique" if selected else ("tie" if eligible else "no-improvement"),
        "untouched_test_opened": selected is not None,
    }
