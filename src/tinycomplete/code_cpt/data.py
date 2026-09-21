"""Deterministic data rules for Stage-1 code continued pretraining."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import PurePosixPath


class RepoAssignment(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


REPOSITORY_PREFIXES = ("max_stars", "max_forks", "max_issues")


def repository_identity(row: dict) -> str:
    """Return one canonical repository identity and never fall back to file identity."""
    for prefix in REPOSITORY_PREFIXES:
        value = row.get(f"{prefix}_repo_name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("repository identity is required for every Stack record")


def repository_path(row: dict) -> str:
    """Return the path paired with the same canonical repository as its identity."""
    for prefix in REPOSITORY_PREFIXES:
        name = row.get(f"{prefix}_repo_name")
        if isinstance(name, str) and name.strip():
            value = row.get(f"{prefix}_repo_path")
            return value.strip() if isinstance(value, str) else ""
    raise ValueError("repository identity is required for every Stack record")


def allocate_blocks(weights: dict[str, float], block_count: int) -> dict[str, int]:
    """Allocate indivisible blocks with the largest-remainder method."""
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if not weights or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("language weights must sum to 1")
    raw = {language: block_count * weight for language, weight in weights.items()}
    allocation = {language: int(value) for language, value in raw.items()}
    remainder = block_count - sum(allocation.values())
    order = sorted(
        raw,
        key=lambda language: (raw[language] - allocation[language], language),
        reverse=True,
    )
    for language in order[:remainder]:
        allocation[language] += 1
    return allocation


@dataclass(frozen=True)
class RepoSplit:
    """Assign complete repositories with a stable SHA-256 bucket."""

    validation_buckets: range = range(10)
    bucket_count: int = 1000

    def __post_init__(self) -> None:
        if self.bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        if any(bucket < 0 or bucket >= self.bucket_count for bucket in self.validation_buckets):
            raise ValueError("validation bucket is outside bucket_count")

    def bucket(self, repo_name: str) -> int:
        identity = repo_name.strip()
        if not identity:
            raise ValueError("repository identity is required")
        digest = sha256(identity.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.bucket_count

    def assignment(self, repo_name: str, path: str | None = None) -> RepoAssignment:
        del path  # A path must never affect a repository-level assignment.
        if self.bucket(repo_name) in self.validation_buckets:
            return RepoAssignment.VALIDATION
        return RepoAssignment.TRAIN


class FilterReason(StrEnum):
    EMPTY = "empty"
    TOO_SMALL = "too_small"
    TOO_LARGE = "too_large"
    BINARY = "binary"
    VENDOR = "vendor_or_generated"
    MINIFIED = "minified"
    LONG_LINE = "long_line"
    LOW_ALPHANUMERIC = "low_alphanumeric"
    REPETITIVE = "repetitive"


@dataclass(frozen=True)
class FilterResult:
    accepted: bool
    reason: FilterReason | None = None


@dataclass(frozen=True)
class SourceFilter:
    min_chars: int = 64
    max_chars: int = 1_000_000
    max_line_chars: int = 20_000
    max_average_line_chars: int = 500
    min_alphanumeric_fraction: float = 0.10
    max_duplicate_line_fraction: float = 0.90

    def check(self, content: str, path: str = "") -> FilterResult:
        if not content:
            return FilterResult(False, FilterReason.EMPTY)
        if len(content) < self.min_chars:
            return FilterResult(False, FilterReason.TOO_SMALL)
        if len(content) > self.max_chars:
            return FilterResult(False, FilterReason.TOO_LARGE)
        if "\x00" in content or self._control_fraction(content) > 0.01:
            return FilterResult(False, FilterReason.BINARY)

        lowered_parts = {part.lower() for part in PurePosixPath(path).parts}
        blocked_parts = {
            "vendor",
            "vendors",
            "generated",
            "dist",
            "build",
            "node_modules",
            "third_party",
            "third-party",
        }
        if lowered_parts & blocked_parts:
            return FilterResult(False, FilterReason.VENDOR)
        if path.lower().endswith((".min.js", ".min.css")):
            return FilterResult(False, FilterReason.MINIFIED)

        lines = content.splitlines() or [content]
        max_line = max(map(len, lines))
        average_line = sum(map(len, lines)) / len(lines)
        if max_line > self.max_line_chars:
            return FilterResult(False, FilterReason.LONG_LINE)
        if average_line > self.max_average_line_chars:
            reason = (
                FilterReason.MINIFIED
                if path.lower().endswith((".js", ".css"))
                else FilterReason.LONG_LINE
            )
            return FilterResult(False, reason)

        if sum(char.isalnum() for char in content) / len(content) < self.min_alphanumeric_fraction:
            return FilterResult(False, FilterReason.LOW_ALPHANUMERIC)
        nonempty = [line.strip() for line in lines if line.strip()]
        if len(nonempty) >= 20:
            most_common = Counter(nonempty).most_common(1)[0][1]
            if most_common / len(nonempty) > self.max_duplicate_line_fraction:
                return FilterResult(False, FilterReason.REPETITIVE)
        return FilterResult(True)

    @staticmethod
    def _control_fraction(content: str) -> float:
        controls = sum(ord(char) < 32 and char not in "\n\r\t\f" for char in content)
        return controls / len(content)


@dataclass
class BlockPacker:
    block_size: int
    eos_token_id: int
    pending_tokens: list[int] = field(default_factory=list)
    source_tokens: int = 0
    boundary_tokens: int = 0
    emitted_tokens: int = 0
    discarded_tokens: int = 0
    documents: int = 0

    def __post_init__(self) -> None:
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")

    def add_document(self, token_ids: Iterable[int]) -> list[list[int]]:
        document = list(token_ids)
        if not document:
            return []
        if self.documents:
            self.pending_tokens.append(self.eos_token_id)
            self.boundary_tokens += 1
        self.documents += 1
        self.source_tokens += len(document)
        self.pending_tokens.extend(document)
        blocks = []
        while len(self.pending_tokens) >= self.block_size:
            blocks.append(self.pending_tokens[: self.block_size])
            del self.pending_tokens[: self.block_size]
            self.emitted_tokens += self.block_size
        return blocks

    def discard_remainder(self) -> int:
        discarded = len(self.pending_tokens)
        self.discarded_tokens += discarded
        self.pending_tokens.clear()
        return discarded

    @property
    def packing_efficiency(self) -> float:
        denominator = self.emitted_tokens + self.discarded_tokens
        return self.emitted_tokens / denominator if denominator else 0.0


@dataclass
class BlockProvenanceTracker:
    """Mirror BlockPacker while retaining repository spans for emitted blocks."""

    block_size: int
    pending: list[list[str | int]] = field(default_factory=list)
    documents: int = 0

    def _append(self, repository: str, tokens: int) -> None:
        if tokens <= 0:
            return
        if self.pending and self.pending[-1][0] == repository:
            self.pending[-1][1] = int(self.pending[-1][1]) + tokens
        else:
            self.pending.append([repository, tokens])

    @property
    def pending_tokens(self) -> int:
        return sum(int(segment[1]) for segment in self.pending)

    def add_document(self, repository: str, token_count: int) -> list[list[dict[str, str | int]]]:
        if not repository:
            raise ValueError("repository is required for block provenance")
        if token_count <= 0:
            return []
        if self.documents:
            self._append("__boundary__", 1)
        self.documents += 1
        self._append(repository, token_count)
        blocks = []
        while self.pending_tokens >= self.block_size:
            remaining = self.block_size
            position = 0
            spans = []
            while remaining:
                repository_name = str(self.pending[0][0])
                available = int(self.pending[0][1])
                consumed = min(remaining, available)
                spans.append(
                    {
                        "repository": repository_name,
                        "start": position,
                        "end": position + consumed,
                    }
                )
                position += consumed
                remaining -= consumed
                if consumed == available:
                    self.pending.pop(0)
                else:
                    self.pending[0][1] = available - consumed
            blocks.append(spans)
        return blocks


@dataclass
class LanguageMix:
    weights: dict[str, float]
    total_tokens: int
    consumed_tokens: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if not self.weights or any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("language weights must be positive")
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("language weights must sum to 1")

    @property
    def target_tokens(self) -> dict[str, int]:
        raw = {language: self.total_tokens * weight for language, weight in self.weights.items()}
        targets = {language: int(value) for language, value in raw.items()}
        remainder = self.total_tokens - sum(targets.values())
        order = sorted(
            raw,
            key=lambda language: (raw[language] - targets[language], language),
            reverse=True,
        )
        for language in order[:remainder]:
            targets[language] += 1
        return targets

    def record(self, language: str, tokens: int) -> None:
        if language not in self.weights:
            raise KeyError(language)
        if tokens < 0:
            raise ValueError("tokens must be nonnegative")
        self.consumed_tokens[language] += tokens

    @property
    def complete(self) -> bool:
        return sum(self.consumed_tokens.values()) >= self.total_tokens

    def actual_percentages(self) -> dict[str, float]:
        total = sum(self.consumed_tokens.values())
        if not total:
            return {language: 0.0 for language in self.weights}
        return {
            language: 100.0 * self.consumed_tokens[language] / total
            for language in self.weights
        }
