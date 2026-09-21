"""Deterministic long-context code dependency benchmark generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, model_validator

from .code_benchmark import BenchmarkCase, BenchmarkResult, CheckSpec, summarize_results
from .code_generation import build_causal_prompt

DependencyKind = Literal["constant", "enum", "signature", "field", "config"]


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


class LongContextSuiteSpec(BaseModel):
    schema_version: Literal[1] = 1
    generator_version: Literal["repository-dependency-v1"] = "repository-dependency-v1"
    model_id: str
    tokenizer_revision: str
    target_tokens: list[int] = Field(min_length=1)
    needle_depths: list[float] = Field(min_length=1)
    dependency_kinds: list[DependencyKind] = Field(min_length=1)
    seed: int = 24052025
    tolerance_fraction: float = Field(default=0.04, gt=0, le=0.1)
    expected_suite_sha256: str | None = None

    @model_validator(mode="after")
    def validate_axes(self) -> LongContextSuiteSpec:
        if len(set(self.target_tokens)) != len(self.target_tokens):
            raise ValueError("target token counts must be unique")
        if len(set(self.dependency_kinds)) != len(self.dependency_kinds):
            raise ValueError("dependency kinds must be unique")
        if any(tokens < 512 for tokens in self.target_tokens):
            raise ValueError("long-context targets must contain at least 512 tokens")
        if any(not 0.05 <= depth <= 0.95 for depth in self.needle_depths):
            raise ValueError("needle depths must be between 0.05 and 0.95")
        return self


@dataclass(frozen=True)
class MaterializedLongContextCase:
    case: BenchmarkCase
    target_tokens: int
    prompt_tokens: int
    requested_needle_depth: float
    actual_needle_depth: float
    dependency_path: str
    prompt_sha256: str


def load_long_context_spec(path: Path) -> LongContextSuiteSpec:
    return LongContextSuiteSpec.model_validate_json(path.read_text(encoding="utf-8"))


def _filler_module(index: int, seed: int) -> str:
    shard_count = 7 + (index + seed) % 23
    multiplier = 13 + ((index * 7 + seed) % 71)
    offset = 17 + ((index * 11 + seed) % 83)
    batch_size = 8 + ((index * 5 + seed) % 57)
    return (
        '"""Telemetry pipeline component generated from the repository service catalog."""\n'
        "\n"
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "@dataclass(frozen=True)\n"
        f"class Component{index:04d}:\n"
        f'    service: str = "telemetry-worker-{index:04d}"\n'
        f"    batch_size: int = {batch_size}\n"
        f"    shard_count: int = {shard_count}\n"
        "\n"
        "\n"
        f"def normalize_events_{index:04d}(events: list[str]) -> tuple[str, ...]:\n"
        '    """Normalize nonempty event labels before batching."""\n'
        "    return tuple(event.strip().lower() for event in events if event.strip())\n"
        "\n"
        "\n"
        f"def shard_for_job_{index:04d}(job_id: int) -> int:\n"
        '    """Choose the stable service shard for one job."""\n'
        f"    return (job_id * {multiplier} + {offset}) % {shard_count}\n"
    )


def _dependency_fixture(
    kind: DependencyKind, module_name: str
) -> tuple[str, str, str, str, str]:
    """Return dependency source, target prefix/suffix, gold, and executable test."""
    import_path = f"project.modules.{module_name}"
    if kind == "constant":
        dependency = (
            '"""Retry timing shared by the queue workers."""\n\n'
            "RETRY_DELAYS_SECONDS: dict[str, int] = {\n"
            '    "interactive": 11,\n'
            '    "background": 43,\n'
            '    "bulk_import": 97,\n'
            "}\n"
        )
        prefix = (
            f"from {import_path} import RETRY_DELAYS_SECONDS\n\n\n"
            "def retry_delay_for_background_jobs() -> int:\n"
            '    """Return the repository-defined delay for background jobs."""\n'
            "    return "
        )
        expected = 'RETRY_DELAYS_SECONDS["background"]'
        test = (
            "from solution import retry_delay_for_background_jobs\n\n"
            "assert retry_delay_for_background_jobs() == 43\n"
        )
    elif kind == "enum":
        dependency = (
            '"""Wire protocol choices for event replication."""\n\n'
            "from enum import Enum\n\n\n"
            "class WireFormat(str, Enum):\n"
            '    JSON_LINES = "jsonl"\n'
            '    MESSAGE_PACK = "msgpack"\n'
            '    CBOR_SEQUENCE = "cbor-seq"\n\n\n'
            "REPLICATION_FORMAT = WireFormat.CBOR_SEQUENCE\n"
        )
        prefix = (
            f"from {import_path} import WireFormat\n\n\n"
            "def replication_wire_format() -> WireFormat:\n"
            '    """Return the format selected by the replication protocol."""\n'
            "    return "
        )
        expected = "WireFormat.CBOR_SEQUENCE"
        test = (
            f"from {import_path} import WireFormat\n"
            "from solution import replication_wire_format\n\n"
            "assert replication_wire_format() is WireFormat.CBOR_SEQUENCE\n"
            'assert replication_wire_format().value == "cbor-seq"\n'
        )
    elif kind == "signature":
        dependency = (
            '"""Payload codec used by the durable event writer."""\n\n'
            "from enum import Enum\n\n\n"
            "class Compression(str, Enum):\n"
            '    NONE = "none"\n'
            '    GZIP = "gzip"\n'
            '    ZSTD = "zstd"\n\n\n'
            "def encode_payload(\n"
            "    payload: bytes, *, compression: Compression, checksum: bool\n"
            ") -> bytes:\n"
            "    marker = compression.value.encode()\n"
            '    integrity = b"checked" if checksum else b"unchecked"\n'
            '    return marker + b":" + integrity + b":" + payload\n'
        )
        prefix = (
            f"from {import_path} import Compression, encode_payload\n\n\n"
            "def encode_replication_event(payload: bytes) -> bytes:\n"
            '    """Encode a checked event with the repository standard compression."""\n'
            "    return "
        )
        expected = "encode_payload(payload, compression=Compression.ZSTD, checksum=True)"
        test = (
            "from solution import encode_replication_event\n\n"
            'assert encode_replication_event(b"evt") == b"zstd:checked:evt"\n'
        )
    elif kind == "field":
        dependency = (
            '"""Endpoint policies shared by upload clients."""\n\n'
            "from dataclasses import dataclass\n\n\n"
            "@dataclass(frozen=True)\n"
            "class EndpointPolicy:\n"
            "    request_timeout_ms: int\n"
            "    max_attempts: int\n"
            "    allow_partial: bool\n\n\n"
            "UPLOAD_POLICY = EndpointPolicy(\n"
            "    request_timeout_ms=1750, max_attempts=6, allow_partial=False\n"
            ")\n"
        )
        prefix = (
            f"from {import_path} import UPLOAD_POLICY\n\n\n"
            "def upload_timeout_ms() -> int:\n"
            '    """Return the timeout field from the shared upload policy."""\n'
            "    return "
        )
        expected = "UPLOAD_POLICY.request_timeout_ms"
        test = "from solution import upload_timeout_ms\n\nassert upload_timeout_ms() == 1750\n"
    else:
        dependency = (
            '"""Service-specific concurrency limits."""\n\n'
            "SERVICE_LIMITS: dict[str, dict[str, int]] = {\n"
            '    "indexer": {"steady": 9, "burst": 15},\n'
            '    "thumbnailer": {"steady": 7, "burst": 23},\n'
            '    "archive_writer": {"steady": 3, "burst": 5},\n'
            "}\n"
        )
        prefix = (
            f"from {import_path} import SERVICE_LIMITS\n\n\n"
            "def thumbnail_burst_limit() -> int:\n"
            '    """Read the thumbnail service burst limit from repository config."""\n'
            "    return "
        )
        expected = 'SERVICE_LIMITS["thumbnailer"]["burst"]'
        test = (
            "from solution import thumbnail_burst_limit\n\n"
            "assert thumbnail_burst_limit() == 23\n"
        )
    return dependency, prefix, "\n", expected, test


def _build_case(
    *,
    case_id: str,
    kind: DependencyKind,
    filler_count: int,
    needle_depth: float,
    seed: int,
    dependency_index: int | None = None,
) -> tuple[BenchmarkCase, str]:
    if dependency_index is None:
        dependency_index = min(filler_count, max(0, round(needle_depth * filler_count)))
    module_name = f"module_{dependency_index:04d}_contract"
    dependency_path = f"project/modules/{module_name}.py"
    dependency, prefix, suffix, expected, test = _dependency_fixture(kind, module_name)
    context_files = {
        "project/__init__.py": '"""Synthetic telemetry service repository."""\n',
        "project/modules/__init__.py": '"""Telemetry pipeline modules."""\n',
        dependency_path: dependency,
    }
    for index in range(filler_count):
        suffix_name = "component"
        if index == dependency_index:
            suffix_name = "component_after_contract"
        path = f"project/modules/module_{index:04d}_{suffix_name}.py"
        context_files[path] = _filler_module(index, seed)
    case = BenchmarkCase(
        id=case_id,
        language="python",
        path="solution.py",
        prefix=prefix,
        suffix=suffix,
        expected=expected,
        context_files=context_files,
        check=CheckSpec(
            compile=["python3", "-m", "py_compile", "solution.py"],
            test=["python3", "tests.py"],
            files={"tests.py": test},
            timeout_seconds=10,
            container_image="python:3.12-slim",
        ),
        category=f"long_context_{kind}",
        repository_context=True,
    )
    return case, dependency_path


def _token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def materialize_long_context_case(
    *,
    tokenizer: Tokenizer,
    target_tokens: int,
    needle_depth: float,
    kind: DependencyKind,
    seed: int,
    tolerance_fraction: float = 0.04,
) -> MaterializedLongContextCase:
    """Build one case within a narrow token band using a deterministic local search."""
    if target_tokens < 512:
        raise ValueError("target_tokens must be at least 512")
    if not 0.05 <= needle_depth <= 0.95:
        raise ValueError("needle_depth must be between 0.05 and 0.95")
    case_id = f"python/{target_tokens:05d}/{kind}/depth-{needle_depth:.2f}"

    probe_case, _ = _build_case(
        case_id=case_id, kind=kind, filler_count=4, needle_depth=needle_depth, seed=seed
    )
    base_case, _ = _build_case(
        case_id=case_id, kind=kind, filler_count=0, needle_depth=needle_depth, seed=seed
    )
    probe_tokens = _token_count(tokenizer, build_causal_prompt(probe_case))
    base_tokens = _token_count(tokenizer, build_causal_prompt(base_case))
    tokens_per_filler = max(1.0, (probe_tokens - base_tokens) / 4)
    estimate = max(0, round((target_tokens - base_tokens) / tokens_per_filler))

    best: tuple[int, BenchmarkCase, str, str] | None = None
    radius = 5
    for filler_count in range(max(0, estimate - radius), estimate + radius + 1):
        case, dependency_path = _build_case(
            case_id=case_id,
            kind=kind,
            filler_count=filler_count,
            needle_depth=needle_depth,
            seed=seed,
        )
        prompt = build_causal_prompt(case)
        count = _token_count(tokenizer, prompt)
        size_candidate = (count, case, dependency_path, prompt)
        if best is None or abs(count - target_tokens) < abs(best[0] - target_tokens):
            best = size_candidate
    assert best is not None
    prompt_tokens, case, dependency_path, prompt = best
    filler_count = sum("class Component" in source for source in case.context_files.values())
    positions: dict[int, tuple[float, int, BenchmarkCase, str, str]] = {}

    def position_at(dependency_index: int) -> tuple[float, int, BenchmarkCase, str, str]:
        if dependency_index in positions:
            return positions[dependency_index]
        candidate_case, candidate_path = _build_case(
            case_id=case_id,
            kind=kind,
            filler_count=filler_count,
            needle_depth=needle_depth,
            seed=seed,
            dependency_index=dependency_index,
        )
        candidate_prompt = build_causal_prompt(candidate_case)
        candidate_tokens = _token_count(tokenizer, candidate_prompt)
        marker = f'<file path="{candidate_path}">\n'
        marker_tokens = _token_count(tokenizer, candidate_prompt[: candidate_prompt.index(marker)])
        actual_depth = marker_tokens / candidate_tokens
        positions[dependency_index] = (
            abs(actual_depth - needle_depth),
            candidate_tokens,
            candidate_case,
            candidate_path,
            candidate_prompt,
        )
        return positions[dependency_index]

    lower = 0
    upper = filler_count
    while lower <= upper:
        midpoint = (lower + upper) // 2
        position = position_at(midpoint)
        candidate_prompt = position[4]
        marker = f'<file path="{position[3]}">\n'
        marker_tokens = _token_count(
            tokenizer, candidate_prompt[: candidate_prompt.index(marker)]
        )
        actual_depth = marker_tokens / position[1]
        if actual_depth < needle_depth:
            lower = midpoint + 1
        else:
            upper = midpoint - 1
    for index in range(max(0, upper - 1), min(filler_count, lower + 1) + 1):
        position_at(index)
    positioned = min(positions.values(), key=lambda candidate: candidate[0])
    _, prompt_tokens, case, dependency_path, prompt = positioned
    tolerance = max(64, round(target_tokens * tolerance_fraction))
    if abs(prompt_tokens - target_tokens) > tolerance:
        raise RuntimeError(
            f"could not calibrate {case_id}: target={target_tokens}, actual={prompt_tokens}"
        )
    dependency_marker = f'<file path="{dependency_path}">\n'
    marker_offset = prompt.index(dependency_marker)
    depth_tokens = _token_count(tokenizer, prompt[:marker_offset])
    return MaterializedLongContextCase(
        case=case,
        target_tokens=target_tokens,
        prompt_tokens=prompt_tokens,
        requested_needle_depth=needle_depth,
        actual_needle_depth=depth_tokens / prompt_tokens,
        dependency_path=dependency_path,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )


def materialize_long_context_suite(
    spec: LongContextSuiteSpec, tokenizer: Tokenizer
) -> list[MaterializedLongContextCase]:
    cases = []
    for target_index, target_tokens in enumerate(spec.target_tokens):
        for kind_index, kind in enumerate(spec.dependency_kinds):
            depth = spec.needle_depths[(target_index + kind_index) % len(spec.needle_depths)]
            cases.append(
                materialize_long_context_case(
                    tokenizer=tokenizer,
                    target_tokens=target_tokens,
                    needle_depth=depth,
                    kind=kind,
                    seed=spec.seed + target_index * 101 + kind_index,
                    tolerance_fraction=spec.tolerance_fraction,
                )
            )
    return cases


def suite_metadata(
    spec: LongContextSuiteSpec, cases: list[MaterializedLongContextCase]
) -> dict:
    records = [
        {
            "id": item.case.id,
            "kind": item.case.category.removeprefix("long_context_"),
            "target_tokens": item.target_tokens,
            "prompt_tokens": item.prompt_tokens,
            "requested_needle_depth": item.requested_needle_depth,
            "actual_needle_depth": item.actual_needle_depth,
            "dependency_path": item.dependency_path,
            "prompt_sha256": item.prompt_sha256,
        }
        for item in cases
    ]
    suite_digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "generator_version": spec.generator_version,
        "model_id": spec.model_id,
        "tokenizer_revision": spec.tokenizer_revision,
        "case_count": len(cases),
        "suite_sha256": suite_digest,
        "cases": records,
    }


def summarize_long_context_results(metadata: dict, results: list[BenchmarkResult]) -> dict:
    """Add context-length and needle-depth slices to the generic executable summary."""
    result_by_id = {result.case_id: result for result in results}
    if len(result_by_id) != len(results):
        raise ValueError("duplicate result ids")
    record_by_id = {record["id"]: record for record in metadata["cases"]}
    missing = sorted(set(record_by_id) - set(result_by_id))
    unexpected = sorted(set(result_by_id) - set(record_by_id))
    if missing or unexpected:
        raise ValueError(
            f"results do not match suite metadata: missing={missing[:1]}, "
            f"unexpected={unexpected[:1]}"
        )

    def rows_for(field: str, value: object) -> list[BenchmarkResult]:
        return [
            result_by_id[case_id]
            for case_id, record in record_by_id.items()
            if record[field] == value
        ]

    target_values = sorted({record["target_tokens"] for record in metadata["cases"]})
    depth_values = sorted({record["requested_needle_depth"] for record in metadata["cases"]})
    kind_values = sorted({record["kind"] for record in metadata["cases"]})
    return {
        "suite_sha256": metadata["suite_sha256"],
        "overall": summarize_results(results),
        "by_target_tokens": {
            str(value): summarize_results(rows_for("target_tokens", value))
            for value in target_values
        },
        "by_requested_needle_depth": {
            f"{value:.2f}": summarize_results(rows_for("requested_needle_depth", value))
            for value in depth_values
        },
        "by_dependency_kind": {
            str(value): summarize_results(rows_for("kind", value)) for value in kind_values
        },
    }
