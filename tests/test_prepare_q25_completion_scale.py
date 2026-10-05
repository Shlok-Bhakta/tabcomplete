from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_q25_completion_scale import (
    PreparationError,
    _components,
    _load_plan,
    _select_matching_new_train,
    _select_new_development,
    _state_key,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def doc(content: str, language: str, aliases: tuple[str, ...]):
    return SimpleNamespace(
        content_sha256=digest(content),
        language=language,
        repository_identity_sha256=aliases[0],
        repository_alias_sha256=aliases,
    )


def state(
    content: str,
    language: str,
    *,
    total_tokens: int,
    target_tokens: int,
    mode: str = "whole_logical_line",
    variant: int = 0,
    region_start: int = 10,
    region_end: int = 20,
):
    return {
        "source_content_sha256": digest(content),
        "language": language,
        "mode": mode,
        "variant": variant,
        "region_start": region_start,
        "region_end": region_end,
        "target_sha256": digest(f"target:{content}:{variant}"),
        "total_tokens": total_tokens,
        "target_tokens": target_tokens,
        "_order": digest(f"order:{content}:{variant}").encode(),
    }


def test_frozen_plan_hash_and_revision_are_checked():
    plan = _load_plan(__import__("prepare_q25_completion_scale").PLAN_PATH)
    assert plan["plan_revision"] == 2


def test_repository_alias_components_are_transitive():
    documents = [
        doc("one", "python", (digest("repo-a"), digest("repo-b"))),
        doc("two", "rust", (digest("repo-b"), digest("repo-c"))),
        doc("three", "go", (digest("repo-c"), digest("repo-d"))),
        doc("four", "typescript", (digest("repo-independent"),)),
    ]
    groups, content_to_group = _components(documents)
    assert len(groups) == 2
    connected = content_to_group[digest("one")]
    assert connected == content_to_group[digest("two")] == content_to_group[digest("three")]
    assert connected != content_to_group[digest("four")]
    assert len(next(group for group in groups if group["group_id"] == connected)["aliases"]) == 4


def test_new_development_reserves_whole_groups_and_excludes_prior_aliases():
    a, b, c, isolated = (digest(name) for name in ("a", "b", "c", "isolated"))
    documents = [
        doc("p1", "python", (a, b)),
        doc("r1", "rust", (b, c)),
        doc("p2", "python", (isolated,)),
    ]
    groups, content_to_group = _components(documents)
    candidates = [
        {
            **state("p1", "python", total_tokens=100, target_tokens=8),
            "source_content_sha256": digest("p1"),
        },
        {
            **state("r1", "rust", total_tokens=120, target_tokens=10),
            "source_content_sha256": digest("r1"),
        },
        {
            **state("p2", "python", total_tokens=110, target_tokens=8),
            "source_content_sha256": digest("p2"),
        },
    ]
    selected, heldout, audit = _select_new_development(
        candidates,
        groups,
        content_to_group,
        previous_aliases={digest("old-train")},
        weights={"python": 0.5, "rust": 0.25, "typescript": 0.0, "go": 0.25},
        requested=3,
        seed=271828,
    )
    assert len(selected) == 2  # The absent Go state is reported, not synthesized.
    assert audit["actual_states"] == 2
    assert audit["shortages_by_language"]["go"] == 1
    selected_aliases = {alias for group in heldout for alias in group["aliases"]}
    assert selected_aliases.isdisjoint({digest("old-train")})
    assert len(selected_aliases) >= 1
    selected_group_ids = {group["group_id"] for group in heldout}
    assert all(
        content_to_group[row["source_content_sha256"]] in selected_group_ids for row in selected
    )


def test_new_development_rejects_whole_transitive_group_touching_prior_data():
    a, b, c, fresh = (digest(name) for name in ("a", "b", "c", "fresh"))
    documents = [
        doc("python-a", "python", (a, b)),
        doc("rust-b", "rust", (b, c)),
        doc("go-fresh", "go", (fresh,)),
    ]
    groups, content_to_group = _components(documents)
    candidates = [
        {
            **state("python-a", "python", total_tokens=100, target_tokens=8),
            "source_content_sha256": digest("python-a"),
        },
        {
            **state("rust-b", "rust", total_tokens=100, target_tokens=8),
            "source_content_sha256": digest("rust-b"),
        },
        {
            **state("go-fresh", "go", total_tokens=100, target_tokens=8),
            "source_content_sha256": digest("go-fresh"),
        },
    ]
    selected, heldout, audit = _select_new_development(
        candidates,
        groups,
        content_to_group,
        previous_aliases={c},
        weights={"python": 0.0, "rust": 0.0, "typescript": 0.0, "go": 1.0},
        requested=1,
        seed=271828,
    )
    assert len(selected) == 1
    assert selected[0]["language"] == "go"
    assert len(heldout) == 1
    assert heldout[0]["aliases"] == [fresh]
    assert audit["excluded_group_count_by_reason"]["overlaps_previous_training_or_development"] == 1


def test_training_matches_exact_bins_then_nearest_same_language_without_replacement():
    old_exact = state(
        "old-exact", "python", total_tokens=100, target_tokens=5, region_start=1, region_end=2
    )
    old_other = state(
        "old-other", "python", total_tokens=300, target_tokens=20, region_start=3, region_end=4
    )
    candidate_exact = state(
        "candidate-exact",
        "python",
        total_tokens=310,
        target_tokens=18,
        region_start=5,
        region_end=6,
    )
    candidate_near = state(
        "candidate-near", "python", total_tokens=200, target_tokens=7, region_start=7, region_end=8
    )
    duplicate_previous = {**old_exact, "_order": b"duplicate"}
    selected, audit = _select_matching_new_train(
        [duplicate_previous, candidate_near, candidate_exact],
        [old_exact, old_other],
        count=2,
        seed=314160,
    )
    assert len(selected) == 2
    assert {_state_key(row) for row in selected}.isdisjoint(
        {_state_key(old_exact), _state_key(old_other)}
    )
    assert {row["source_content_sha256"] for row in selected} == {
        digest("candidate-near"),
        digest("candidate-exact"),
    }
    assert audit["shortages_by_language_and_bins"] == {}
    assert audit["candidate_rejections"]["exact_previous_state"] == 1


def test_training_reports_real_shortage_instead_of_repeating_states():
    old_rows = [
        state(
            f"old-{index}",
            "go",
            total_tokens=120,
            target_tokens=8,
            region_start=index,
            region_end=index + 1,
        )
        for index in range(2)
    ]
    candidate = state("only-candidate", "go", total_tokens=120, target_tokens=8)
    selected, audit = _select_matching_new_train([candidate], old_rows, count=2, seed=314160)
    assert len(selected) == 1
    assert audit["actual_states"] == 1
    assert sum(audit["shortages_by_language_and_bins"]["go"].values()) == 1


def test_new_states_respect_two_variants_per_document_across_the_old_boundary():
    old = state(
        "shared-document",
        "python",
        total_tokens=120,
        target_tokens=8,
        variant=0,
        region_start=0,
        region_end=1,
    )
    candidates = [
        state(
            "shared-document",
            "python",
            total_tokens=120,
            target_tokens=8,
            variant=1,
            region_start=2,
            region_end=3,
        ),
        state(
            "shared-document",
            "python",
            total_tokens=120,
            target_tokens=8,
            variant=2,
            region_start=4,
            region_end=5,
        ),
    ]
    selected, audit = _select_matching_new_train(candidates, [old], count=2, seed=314160)
    assert len(selected) == 1
    assert audit["candidate_rejections"]["combined_document_variant_cap"] == 1


def test_invalid_state_region_is_rejected():
    with pytest.raises(PreparationError, match="serialized_row_region_invalid"):
        _state_key(
            {
                "source_content_sha256": "a",
                "mode": "whole",
                "region_start": True,
                "region_end": 4,
                "target_sha256": "b",
            }
        )
