"""Strict FIM completion diagnostics through the existing model and run helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeGuard

from evaluate_causal_line import score_line

from tinycomplete.data.fim import format_psm
from tinycomplete.eval.code_benchmark import _parse
from tinycomplete.eval.code_generation import (
    DetailedGeneration,
    TransformersGenerationProvider,
    build_prediction_run_metadata,
    file_sha256,
    generate_predictions,
)
from tinycomplete.eval.q25_fim_attention import (
    ATTENTION_BACKEND,
    install_q25_fim_attention,
    registered_sdpa_forward,
)
from tinycomplete.observability.bootstrap import current_runtime
from tinycomplete.observability.context import RunContext
from tinycomplete.observability.runs import run_scope
from tinycomplete.observability.spans import operation

LINE_SUITE_SHA = "2eb55e7db35957007572cb15db2d27cd597b25322e85ae776ff4e723ddec2ead"
SOURCE_SYNTAX_PROTOCOL = "q25-fim-source-syntax-v1"
FIM_PREPARATION_PLAN = (
    Path(__file__).resolve().parents[1]
    / "reports/research/q25_code_cpt_r2/fim_preparation_plan.json"
)
SCALE_PLAN_SCHEMA = "q25-completion-scale-training-plan-v1"
SCALE_PREPARATION_PLAN = (
    Path(__file__).resolve().parents[1]
    / "reports/research/q25_completion_scale_r1/preparation_plan-r2.json"
)
SCALE_DEVELOPMENT_SPLITS = {
    "development_new": (512, 8192),
    "development_previous": (240, 4096),
}


def validated_development_input(
    plan: dict[str, Any], path: Path, *, selected_split: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Bind each versioned dev group before loading a model or starting CUDA."""
    schema = plan.get("schema")
    if schema == "q25-fim-training-plan-v1":
        if selected_split is not None:
            raise ValueError("scale development split cannot be used with the original FIM plan")
        split = "development"
    elif schema == SCALE_PLAN_SCHEMA:
        split = selected_split or path.stem
        if split not in SCALE_DEVELOPMENT_SPLITS or path.name != f"{split}.jsonl":
            raise ValueError("scale evaluation requires an exact named development split")
    else:
        raise ValueError("a frozen FIM training/evaluation plan is required")
    data = plan.get("data")
    record = data.get(split) if isinstance(data, dict) else None
    if not isinstance(record, dict) or file_sha256(path) != record.get("sha256"):
        raise ValueError("FIM evaluation input identity differs")
    rows = _read_jsonl(path, "FIM development input")
    if len(rows) != record.get("row_count"):
        raise ValueError("FIM development state count differs")
    if schema == SCALE_PLAN_SCHEMA:
        expected_count, offset = SCALE_DEVELOPMENT_SPLITS[split]
        if (
            len(rows) != expected_count
            or path.stat().st_size != record.get("bytes")
            or any(
                type(row.get("id")) is not int
                or row["id"] != offset + index
                or row.get("split") != "development"
                for index, row in enumerate(rows)
            )
        ):
            raise ValueError("scale development size, count, or ordered IDs differ")
    return rows, record, split


def paired_development(
    first: list[dict[str, Any]], second: list[dict[str, Any]], *, metric: str
) -> dict[str, Any]:
    """Case-weighted paired difference, resampling whole repository groups."""
    import numpy as np

    if metric not in {"exact", "exact_and_terminated"}:
        raise ValueError("unsupported registered development metric")
    left = {row["case_id"]: row for row in first}
    right = {row["case_id"]: row for row in second}
    if not left or len(left) != len(first) or len(right) != len(second) or set(left) != set(right):
        raise ValueError("paired development identities are empty, duplicate or different")
    groups: dict[str, list[int]] = {}
    wins, losses, shared = [], [], []
    for case_id in sorted(left):
        a, b = left[case_id], right[case_id]
        if (
            not isinstance(a.get("repository"), str)
            or not a["repository"]
            or any(a.get(key) != b.get(key) for key in ("repository", "context_sha256"))
            or not a.get("context_sha256")
            or type(a.get(metric)) is not bool
            or type(b.get(metric)) is not bool
        ):
            raise ValueError("paired development evidence or boolean outcome differs")
        delta = int(b[metric]) - int(a[metric])
        groups.setdefault(a["repository"], []).append(delta)
        if delta > 0:
            wins.append(case_id)
        elif delta < 0:
            losses.append(case_id)
        elif a[metric]:
            shared.append(case_id)
    ordered = [groups[key] for key in sorted(groups)]
    sums = np.array([sum(group) for group in ordered])
    counts = np.array([len(group) for group in ordered])
    rng = np.random.default_rng(271828)
    samples = []
    for _ in range(2000):
        selected = rng.integers(0, len(ordered), len(ordered))
        samples.append(float(sums[selected].sum() / counts[selected].sum()))
    return {
        "metric": metric,
        "cases": len(left),
        "repository_groups": len(groups),
        "first_successes": len(shared) + len(losses),
        "second_successes": len(shared) + len(wins),
        "wins": wins,
        "losses": losses,
        "shared_successes": shared,
        "difference_second_minus_first": float(sums.sum() / counts.sum()),
        "paired_repository_bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "bootstrap_samples": 2000,
        "bootstrap_seed": 271828,
        "bootstrap_unit": "repository identity",
        "interpretation": (
            "Synthetic completion. A small or inconclusive difference is not equivalence."
        ),
    }


def development_case(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    ids = row["input_ids"]
    offset = row["prompt_tokens"]
    if ids[-1] != tokenizer.eos_token_id:
        raise ValueError("prepared FIM target lacks its declared EOS")
    prompt = tokenizer.decode(
        ids[:offset], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    target = tokenizer.decode(
        ids[offset:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"] or (
        hashlib.sha256(target.encode()).hexdigest() != row["target_sha256"]
    ):
        raise ValueError("prepared token IDs do not reconstruct the exact FIM strings")
    if tokenizer.encode(prompt, add_special_tokens=False) != ids[:offset]:
        raise ValueError("served tokenizer changes prepared FIM input token IDs")
    return {
        **row,
        "id": f"fim-development-{row['id']}",
        "prompt": prompt,
        "target": target,
        "repository": row["repository_identity_sha256"],
    }


def decoded_completion(
    tokenizer: Any, token_ids: list[int], *, ceiling: int, newline_stop: bool
) -> tuple[str, str, dict[str, Any]]:
    """Remove only an observed terminal EOS; retain malformed control text."""
    eos = tokenizer.eos_token_id
    ended_by_eos = bool(token_ids) and token_ids[-1] == eos
    content_ids = token_ids[:-1] if ended_by_eos else token_ids
    vocabulary = tokenizer.get_vocab()
    known_ids = set(vocabulary.values())
    unknown = [value for value in content_ids if value not in known_ids]
    # Qwen's FIM control markers are added tokens with special=False. The HF
    # special-token list alone does not classify them as invalid response text.
    control_ids = (
        set(tokenizer.all_special_ids)
        | set(getattr(tokenizer, "added_tokens_decoder", {}))
        | {
            vocabulary[token]
            for token in ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
            if token in vocabulary
        }
    )
    unexpected = [value for value in content_ids if value in control_ids]
    text = tokenizer.decode(
        content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if unknown or unexpected:
        reason = "invalid_control_or_vocabulary"
    elif ended_by_eos:
        reason = "eos"
    elif newline_stop and "\n" in text:
        reason = "newline"
    elif len(token_ids) >= ceiling:
        reason = "length"
    else:
        reason = "unverified_stop"
    return (
        text,
        reason,
        {
            "output_token_ids": token_ids,
            "ended_by_eos": ended_by_eos,
            "unexpected_special_token_ids": unexpected,
            "unexpected_control_token_ids": unexpected,
            "unknown_token_ids": unknown,
            "reached_token_ceiling": len(token_ids) >= ceiling,
            "truncated": reason == "length",
        },
    )


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise ValueError(f"{description} is unreadable") from None
    if not isinstance(value, dict):
        raise ValueError(f"{description} is invalid")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError(f"{description} contains a blank row")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{description} contains an invalid row")
                rows.append(row)
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise ValueError(f"{description} is unreadable") from None
    return rows


def _valid_sha256(value: Any) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parser_version() -> str | None:
    try:
        return importlib.metadata.version("tree-sitter-language-pack")
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_status(source: str, language: str, role: str) -> str:
    with operation(
        "eval.parse",
        attributes={
            "tabcomplete.task": SOURCE_SYNTAX_PROTOCOL,
            "tabcomplete.language": language,
            "tabcomplete.parse.source_role": role,
        },
    ) as span:
        result = _parse(source, language)
        span.set_attribute("tabcomplete.check.status", result.status)
    return result.status


def _source_syntax_rows(
    development_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    development_documents: Iterable[Any],
    *,
    parse_status: Callable[[str, str, str], str] = _parse_status,
    run: RunContext | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse original and generated full sources without persisting their text."""
    documents: dict[str, Any] = {}
    for document in development_documents:
        content_hash = getattr(document, "content_sha256", None)
        if not _valid_sha256(content_hash) or content_hash in documents:
            raise ValueError("pinned development source identity is invalid or duplicated")
        documents[content_hash] = document

    predictions: dict[str, dict[str, Any]] = {}
    for prediction in prediction_rows:
        case_id = prediction.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in predictions:
            raise ValueError("FIM result case identities are invalid or duplicated")
        predictions[case_id] = prediction

    expected_ids = {
        f"fim-development-{row.get('id')}"
        for row in development_rows
        if isinstance(row.get("id"), int) and not isinstance(row.get("id"), bool)
    }
    if len(expected_ids) != len(development_rows) or set(predictions) != expected_ids:
        raise ValueError("FIM result case identities differ from development inputs")

    result_rows: list[dict[str, Any]] = []
    original_counts: Counter[str] = Counter()
    generated_counts: Counter[str] = Counter()
    regression_cases = 0
    regression_denominator = 0
    baseline_pass_cases = 0
    generated_unavailable_after_baseline_pass = 0

    for row in development_rows:
        identifier = row.get("id")
        case_id = f"fim-development-{identifier}"
        content_hash = row.get("source_content_sha256")
        repository_hash = row.get("repository_identity_sha256")
        language = row.get("language")
        source_path = row.get("source_path")
        region_start = row.get("region_start")
        region_end = row.get("region_end")
        target_hash = row.get("target_sha256")
        prompt_hash = row.get("prompt_sha256")
        if (
            not _valid_sha256(content_hash)
            or not _valid_sha256(repository_hash)
            or not _valid_sha256(target_hash)
            or not _valid_sha256(prompt_hash)
            or row.get("split") != "development"
            or row.get("prompt_format") != "psm"
            or row.get("mode")
            not in {"whole_logical_line", "remaining_logical_line_after_utf8_cursor"}
            or not isinstance(language, str)
            or not isinstance(source_path, str)
        ):
            raise ValueError("FIM source, repository, language, or context identity differs")
        document = documents.get(content_hash)
        prediction = predictions[case_id]
        if (
            document is None
            or getattr(document, "language", None) != language
            or getattr(document, "path", None) != source_path
            or getattr(document, "repository_identity_sha256", None) != repository_hash
            or prediction.get("repository") != repository_hash
            or prediction.get("context_sha256") != prompt_hash
            or prediction.get("language") != language
        ):
            raise ValueError("FIM source, repository, language, or context identity differs")
        aliases = row.get("repository_alias_sha256")
        document_aliases = getattr(document, "repository_alias_sha256", None)
        if (
            not isinstance(aliases, list)
            or any(not _valid_sha256(alias) for alias in aliases)
            or sorted(aliases) != sorted(document_aliases or ())
        ):
            raise ValueError("FIM source repository aliases differ from the pinned pool")
        if (
            not isinstance(region_start, int)
            or isinstance(region_start, bool)
            or not isinstance(region_end, int)
            or isinstance(region_end, bool)
            or region_start < 0
            or region_end <= region_start
        ):
            raise ValueError("FIM source byte region is invalid")
        completion = prediction.get("raw_response")
        if not isinstance(completion, str):
            raise ValueError("FIM result is missing the verbatim generated text")
        completion_bytes = completion.encode("utf-8")
        completion_hash = hashlib.sha256(completion_bytes).hexdigest()
        recorded_completion_hash = prediction.get("completion_sha256")
        if (
            not _valid_sha256(recorded_completion_hash)
            or recorded_completion_hash != completion_hash
        ):
            raise ValueError("FIM generated-text hash differs from saved token evidence")

        source_bytes = document.content.encode("utf-8")
        if (
            hashlib.sha256(source_bytes).hexdigest() != content_hash
            or getattr(document, "content_sha256", None) != content_hash
            or region_end > len(source_bytes)
        ):
            raise ValueError("pinned source hash or FIM byte region differs")
        try:
            prefix = source_bytes[:region_start].decode("utf-8")
            target = source_bytes[region_start:region_end].decode("utf-8")
            suffix = source_bytes[region_end:].decode("utf-8")
            completed_source = (
                source_bytes[:region_start] + completion_bytes + source_bytes[region_end:]
            ).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("FIM source region is not aligned to UTF-8 boundaries") from None
        if hashlib.sha256(target.encode("utf-8")).hexdigest() != target_hash:
            raise ValueError("FIM target hash differs from the pinned source region")
        if prefix + target + suffix != document.content:
            raise ValueError("FIM byte region does not reconstruct the pinned source")

        parser_grammar = (
            "tsx"
            if language == "typescript" and Path(source_path).suffix.lower() == ".tsx"
            else language
        )
        context = run.for_case(case_id) if run is not None else None
        activation = context.activate() if context is not None else nullcontext()
        with activation:
            original_status = parse_status(document.content, parser_grammar, "original")
            generated_status = parse_status(completed_source, parser_grammar, "generated")
        if original_status not in {"pass", "fail", "unavailable"} or generated_status not in {
            "pass",
            "fail",
            "unavailable",
        }:
            raise ValueError("source parser returned an unsupported status")
        original_counts[original_status] += 1
        generated_counts[generated_status] += 1
        if original_status == "pass":
            baseline_pass_cases += 1
            if generated_status in {"pass", "fail"}:
                regression_denominator += 1
                regression_cases += generated_status == "fail"
            else:
                generated_unavailable_after_baseline_pass += 1

        result_rows.append(
            {
                "case_id": case_id,
                "language": language,
                "parser_grammar": parser_grammar,
                "mode": row.get("mode"),
                "variant": row.get("variant"),
                "source_content_sha256": content_hash,
                "repository_identity_sha256": repository_hash,
                "context_sha256": prompt_hash,
                "target_sha256": target_hash,
                "prediction_sha256": completion_hash,
                "original_source_parse_status": original_status,
                "generated_source_parse_status": generated_status,
                "parser_regression": (
                    generated_status == "fail" if original_status == "pass" else None
                ),
            }
        )

    summary = {
        "schema": "q25-fim-source-syntax-diagnostic-v1",
        "protocol": SOURCE_SYNTAX_PROTOCOL,
        "cases": len(result_rows),
        "original_source_parse_status_counts": dict(sorted(original_counts.items())),
        "generated_source_parse_status_counts": dict(sorted(generated_counts.items())),
        "original_parse_pass_cases": baseline_pass_cases,
        "parser_regression_cases": regression_cases,
        "parser_regression_denominator": regression_denominator,
        "generated_parse_unavailable_after_original_pass": (
            generated_unavailable_after_baseline_pass
        ),
        "parser_regression_rate": (
            regression_cases / regression_denominator if regression_denominator else None
        ),
        "regression_definition": (
            "Generated full source fails Tree-sitter after the original full source passes; "
            "denominator requires both parses to return pass/fail."
        ),
        "interpretation": (
            "Descriptive whole-source parser outcome only. Original-source status is the control. "
            "This is not compilation, functional behavior, observed edit intent, or a display gate."
        ),
    }
    return result_rows, summary


def run_source_syntax_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Validate frozen inputs and score a saved GPU development result on CPU."""
    if args.mode != "development" or args.model is not None or args.alias is not None:
        raise ValueError("source-syntax mode requires development inputs and no model alias")
    if args.predictions is None or args.parent_cpt_plan is None:
        raise ValueError("source-syntax mode requires saved FIM results and the parent CPT plan")
    plan = _read_json(args.plan, "FIM plan")
    scale_profile = plan.get("schema") == SCALE_PLAN_SCHEMA
    if (
        plan.get("schema") not in ("q25-fim-training-plan-v1", SCALE_PLAN_SCHEMA)
        or plan.get("gpu_execution_authorized") is not True
    ):
        raise ValueError("source-syntax mode requires the frozen full FIM plan")
    evaluation = plan.get("evaluation")
    source_registration = evaluation.get("source_syntax") if isinstance(evaluation, dict) else None
    if (
        not isinstance(source_registration, dict)
        or source_registration.get("protocol") != SOURCE_SYNTAX_PROTOCOL
    ):
        raise ValueError("full FIM plan does not register this source-syntax diagnostic")

    parent_sha = file_sha256(args.parent_cpt_plan)
    if plan.get("parent_cpt_plan_sha256") != parent_sha:
        raise ValueError("parent CPT plan hash differs from the frozen FIM plan")
    _read_json(args.parent_cpt_plan, "parent CPT plan")
    preparation_path = SCALE_PREPARATION_PLAN if scale_profile else FIM_PREPARATION_PLAN
    preparation_sha = file_sha256(preparation_path)
    if plan.get("preparation_plan_sha256") != preparation_sha:
        raise ValueError("FIM preparation plan hash differs from the frozen full plan")
    preparation = _read_json(preparation_path, "FIM preparation plan")
    if preparation.get("parent_cpt_plan_sha256") != parent_sha:
        raise ValueError("parent CPT identity differs from the FIM preparation plan")

    data_plan = plan.get("data")
    if not isinstance(data_plan, dict):
        raise ValueError("full FIM plan has no development data identity")
    development_rows, development_plan, development_split = validated_development_input(
        plan, args.input, selected_split=getattr(args, "development_split", None)
    )
    if scale_profile and development_split not in source_registration.get(
        "development_splits", []
    ):
        raise ValueError("scale plan does not register source syntax for this development split")
    development_sha = development_plan.get("sha256")
    expected_rows = development_plan.get("row_count")
    expected_bytes = development_plan.get("bytes")
    if (
        not _valid_sha256(development_sha)
        or not isinstance(expected_rows, int)
        or isinstance(expected_rows, bool)
        or expected_rows < 1
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 1
        or args.input.stat().st_size != expected_bytes
        or file_sha256(args.input) != development_sha
    ):
        raise ValueError("development input file differs from the frozen FIM plan")

    metadata_path = args.input.parent / "corpus_metadata.json"
    if file_sha256(metadata_path) != data_plan.get("corpus_metadata_sha256"):
        raise ValueError("FIM corpus metadata hash differs from the frozen plan")
    metadata = _read_json(metadata_path, "FIM corpus metadata")
    metadata_splits = metadata.get("splits")
    metadata_files = metadata.get("files")
    metadata_split = (
        metadata_splits.get(development_split) if isinstance(metadata_splits, dict) else None
    )
    metadata_file = (
        metadata_files.get(args.input.name) if isinstance(metadata_files, dict) else None
    )
    if (
        metadata.get("preparation_plan_sha256") != preparation_sha
        or metadata.get("parent_cpt_plan_sha256") != parent_sha
        or not isinstance(metadata_split, dict)
        or not isinstance(metadata_file, dict)
        or metadata_split.get("file") != args.input.name
        or metadata_split.get("sha256") != development_sha
        or metadata_split.get("row_count") != expected_rows
        or metadata_file.get("sha256") != development_sha
        or metadata_file.get("bytes") != expected_bytes
        or (scale_profile and metadata.get("schema") != "q25-completion-scale-corpus-v1")
    ):
        raise ValueError("FIM corpus metadata does not match the frozen development inputs")

    from prepare_q25_fim import _load_pinned_inputs

    _, preparation_inputs, train_documents, development_documents = _load_pinned_inputs(
        FIM_PREPARATION_PLAN
    )
    if scale_profile:
        # New reserved groups were selected from the already pinned public
        # training source pool. Evaluation still uses only the frozen dev rows.
        development_documents = [*train_documents, *development_documents]
    pool_hashes = preparation_inputs.get("pool_hashes")
    if not isinstance(pool_hashes, dict):
        raise ValueError("pinned original source-pool identities are missing")
    source_pool_identities = {
        language: {
            "sha256": pool_hashes[language]["sha256"],
            "sidecar_sha256": pool_hashes[language]["sidecar_sha256"],
        }
        for language in sorted(pool_hashes)
    }

    prediction_rows = _read_jsonl(args.predictions, "FIM GPU result file")
    result_path = args.predictions.parent / "summary.json"
    result_summary = _read_json(result_path, "FIM GPU summary")
    if (
        result_summary.get("plan_sha256") != file_sha256(args.plan)
        or result_summary.get("cases") != expected_rows
        or not _valid_sha256(result_summary.get("model_sha256"))
    ):
        raise ValueError("FIM GPU results do not match the frozen plan or development count")

    args.output.mkdir(parents=True, exist_ok=True)
    with run_scope(args.output / "observability-run.json", "q25-fim-source-syntax") as run:
        result_rows, summary = _source_syntax_rows(
            development_rows,
            prediction_rows,
            development_documents,
            run=run if isinstance(run, RunContext) else None,
        )
    summary.update(
        plan_sha256=file_sha256(args.plan),
        parent_cpt_plan_sha256=parent_sha,
        preparation_plan_sha256=preparation_sha,
        development_sha256=development_sha,
        development_row_count=expected_rows,
        **(
            {
                "development_split": development_split,
                "source_pool_lookup": "pinned public train and original development documents",
            }
            if scale_profile
            else {}
        ),
        gpu_results_sha256=file_sha256(args.predictions),
        model_sha256=result_summary["model_sha256"],
        parser="tree-sitter-language-pack",
        parser_version=_parser_version(),
        raw_source_pool_file_hashes=source_pool_identities,
    )
    (args.output / "source-syntax-results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    (args.output / "source-syntax-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


class StrictFimProvider(TransformersGenerationProvider):
    def __init__(
        self,
        model_path: str,
        *,
        newline_stop: bool,
        evidence_path: Path,
        attention_backend: str,
    ) -> None:
        install_q25_fim_attention(attention_backend)
        super().__init__(model_path, device="cuda:0")
        self.newline_stop = newline_stop
        self.evidence: dict[str, dict[str, Any]] = {}
        self.evidence_path = evidence_path
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        if evidence_path.exists():
            for line in evidence_path.read_text().splitlines():
                row = json.loads(line)
                self.evidence[row["context_sha256"]] = row

    def _generate(
        self, prompt: str, max_new_tokens: int, *, stop_first_line: bool
    ) -> DetailedGeneration:
        inputs = {
            name: value.to(self.device)
            for name, value in self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).items()
        }
        prefix_length = int(inputs["input_ids"].shape[1])
        extra: dict[str, Any] = {}
        if self.newline_stop:
            from transformers import StoppingCriteria, StoppingCriteriaList

            tokenizer = self.tokenizer

            class FirstNewline(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    return "\n" in tokenizer.decode(
                        input_ids[0, prefix_length:],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )

            extra["stopping_criteria"] = StoppingCriteriaList([FirstNewline()])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                **extra,
            )
        ids = generated[0, prefix_length:].tolist()
        text, reason, evidence = decoded_completion(
            self.tokenizer, ids, ceiling=max_new_tokens, newline_stop=self.newline_stop
        )
        key = hashlib.sha256(prompt.encode()).hexdigest()
        evidence.update(
            context_sha256=key,
            completion_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
        self.evidence[key] = evidence
        with self.evidence_path.open("a") as handle:
            handle.write(json.dumps(evidence, sort_keys=True) + "\n")
        return DetailedGeneration(
            text, len(ids), reason, input_tokens=prefix_length, usage_source="tokenizer"
        )


def _attention_smoke(
    *,
    model_path: Path,
    development_path: Path,
    line_path: Path,
    output_path: Path,
    plan_sha256: str,
) -> dict[str, Any]:
    """Require the selected CUDA kernel on a synthetic tensor sized to real prompts."""
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("FIM attention smoke requires CUDA")
    if not model_path.is_dir() or not (model_path / "tokenizer.json").is_file():
        raise ValueError("FIM attention smoke requires the attached local tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        local_files_only=True,
    )
    development_rows = [
        json.loads(line) for line in development_path.read_text().splitlines() if line.strip()
    ]
    line_rows = [json.loads(line) for line in line_path.read_text().splitlines() if line.strip()]
    if not development_rows or not line_rows:
        raise ValueError("FIM attention smoke fixtures must both be nonempty")
    development_lengths = [int(row["prompt_tokens"]) for row in development_rows]
    line_lengths = [
        len(
            tokenizer.encode(
                format_psm(row["source_before"], row["source_after"]),
                add_special_tokens=False,
            )
        )
        for row in line_rows
    ]
    max_prompt_tokens = max(*development_lengths, *line_lengths)
    if max_prompt_tokens < 1:
        raise ValueError("FIM attention smoke found an empty prompt")

    install_q25_fim_attention(ATTENTION_BACKEND)
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    query = torch.zeros((1, 14, max_prompt_tokens, 64), dtype=torch.float16, device=device)
    key = torch.zeros((1, 2, max_prompt_tokens, 64), dtype=torch.float16, device=device)
    value = torch.zeros_like(key)
    output, weights = registered_sdpa_forward()(
        SimpleNamespace(num_key_value_groups=7, is_causal=True),
        query,
        key,
        value,
        None,
        dropout=0.0,
        scaling=64**-0.5,
        is_causal=True,
    )
    torch.cuda.synchronize(device)
    expected_shape = (1, max_prompt_tokens, 14, 64)
    if (
        tuple(output.shape) != expected_shape
        or weights is not None
        or not bool(torch.isfinite(output).all().item())
    ):
        raise RuntimeError("FIM efficient SDPA smoke returned an unexpected result")
    report = {
        "schema": "q25-fim-attention-smoke-v1",
        "attention_backend": ATTENTION_BACKEND,
        "key_value_head_expansion": "explicit-repeat",
        "plan_sha256": plan_sha256,
        "success": True,
        "query_tokens": max_prompt_tokens,
        "query_heads": 14,
        "key_value_heads": 2,
        "head_dim": 64,
        "dtype": "float16",
        "cuda_device": torch.cuda.get_device_name(device),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "dense_fp32_attention_score_bytes": max_prompt_tokens * max_prompt_tokens * 14 * 4,
        "development_cases": len(development_rows),
        "line_cases": len(line_rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--alias")
    parser.add_argument("--mode", choices=("development", "line"), required=True)
    parser.add_argument("--attention-smoke", action="store_true")
    parser.add_argument("--line-input", type=Path)
    parser.add_argument("--source-syntax", action="store_true")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--parent-cpt-plan", type=Path)
    parser.add_argument("--development-split", choices=tuple(SCALE_DEVELOPMENT_SPLITS))
    args = parser.parse_args()
    if args.source_syntax:
        summary = run_source_syntax_diagnostic(args)
        print(json.dumps(summary, sort_keys=True))
        current_runtime().shutdown()
        return
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") not in ("q25-fim-training-plan-v1", SCALE_PLAN_SCHEMA):
        raise ValueError("a frozen FIM training/evaluation plan is required")
    evaluation = plan.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("attention_backend") != ATTENTION_BACKEND:
        raise ValueError("frozen FIM plan does not bind the supported attention backend")
    development_split = None
    if args.mode == "development":
        rows, _, development_split = validated_development_input(
            plan, args.input, selected_split=args.development_split
        )
    else:
        if args.development_split is not None:
            parser.error("development split is only valid for development evaluation")
        if file_sha256(args.input) != LINE_SUITE_SHA:
            raise ValueError("FIM evaluation input identity differs")
        rows = _read_jsonl(args.input, "FIM line input")
    if args.attention_smoke:
        if args.mode != "development" or args.line_input is None or args.model is None:
            parser.error("attention smoke requires development input, line input, and local model")
        line_identity = evaluation.get("fixtures", {}).get("line", {}).get("sha256")
        if file_sha256(args.line_input) != line_identity:
            raise ValueError("FIM attention smoke line input identity differs")
        report = _attention_smoke(
            model_path=args.model,
            development_path=args.input,
            line_path=args.line_input,
            output_path=args.output / "attention-smoke.json",
            plan_sha256=file_sha256(args.plan),
        )
        print(json.dumps(report, sort_keys=True))
        current_runtime().shutdown()
        return
    if args.model is None or args.alias is None:
        parser.error("GPU FIM evaluation requires --model and --alias")
    if not args.model.is_dir() or not (args.model / "model.safetensors").is_file():
        raise ValueError("FIM evaluation requires existing local weights; downloads are forbidden")
    if args.mode == "line" and len(rows) != 180:
        raise ValueError("the unchanged line suite must contain 180 cases")
    provider = StrictFimProvider(
        str(args.model),
        newline_stop=args.mode == "line",
        evidence_path=args.output / "token_evidence.jsonl",
        attention_backend=ATTENTION_BACKEND,
    )
    if (
        type(provider.model).__name__ != "Qwen2ForCausalLM"
        or sum(parameter.numel() for parameter in provider.model.parameters()) != 494032768
    ):
        raise ValueError("FIM evaluation loaded a different model architecture or size")
    if args.mode == "development":
        rows = [development_case(row, provider.tokenizer) for row in rows]
    prepared = []
    for row in rows:
        item = dict(row)
        item["prompt"] = (
            format_psm(row["source_before"], row["source_after"])
            if args.mode == "line"
            else row["prompt"]
        )
        prepared.append(SimpleNamespace(**item))
    metadata = build_prediction_run_metadata(
        suite_path=args.input,
        case_count=len(rows),
        provider="transformers-strict-fim",
        model_source=args.alias,
        model_revision=file_sha256(args.model / "model.safetensors"),
        max_new_tokens=96,
        workers=1,
        protocol=f"q25-fim-{args.mode}-v1",
    )
    metadata.update(
        plan_sha256=file_sha256(args.plan),
        tokenizer_sha256=file_sha256(args.model / "tokenizer.json"),
        attention_backend=ATTENTION_BACKEND,
        key_value_head_expansion="explicit-repeat",
        attention_heads={"query": 14, "key_value": 2, "head_dim": 64},
        precision="fp16",
        runtime_revision=str(provider.torch.__version__),
        model_inventory={
            "class": type(provider.model).__name__,
            "parameters": sum(p.numel() for p in provider.model.parameters()),
            "device": provider.torch.cuda.get_device_name(0),
        },
        stopping="observed EOS only" if args.mode == "development" else "registered newline or EOS",
        control_tokens="retained and invalidated; only terminal EOS removed",
        task="synthetic FIM completion, not observed next-edit intent",
        **(
            {"development_split": development_split}
            if plan.get("schema") == SCALE_PLAN_SCHEMA and development_split is not None
            else {}
        ),
    )
    predictions = generate_predictions(
        prepared,
        provider,
        args.output / "predictions.jsonl",
        run_metadata=metadata,
        max_new_tokens=96,
        workers=1,
        prompt_builder=lambda case: case.prompt,
    )
    records = []
    for row, case, prediction in zip(rows, prepared, predictions, strict=True):
        record = (
            score_line(row, prediction.completion)
            if args.mode == "line"
            else {
                "case_id": row["id"],
                "repository": row["repository"],
                "language": row["language"],
                "exact": prediction.completion == row["target"],
                "raw_response": prediction.completion,
                "syntax": "not_measured",
            }
        )
        context_sha = hashlib.sha256(case.prompt.encode()).hexdigest()
        evidence = provider.evidence[context_sha]
        if (
            evidence["completion_sha256"]
            != hashlib.sha256(prediction.completion.encode()).hexdigest()
        ):
            raise ValueError("persisted token evidence and prediction text differ")
        terminated = evidence["ended_by_eos"] or prediction.finish_reason == "newline"
        valid_tokens = (
            not evidence["unexpected_special_token_ids"] and not evidence["unknown_token_ids"]
        )
        record.update(
            terminated=terminated,
            valid_output_tokens=valid_tokens,
            exact_and_terminated=bool(record["exact"]) and terminated and valid_tokens,
            finish_reason=prediction.finish_reason,
            input_tokens=len(provider.tokenizer.encode(case.prompt, add_special_tokens=False)),
            output_tokens=prediction.generated_tokens,
            latency_seconds=prediction.latency_seconds,
            context_sha256=context_sha,
        )
        record.update(evidence)
        records.append(record)
    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    summary = {
        "cases": len(records),
        "exact": sum(row["exact"] for row in records),
        "exact_and_terminated": sum(row["exact_and_terminated"] for row in records),
        "terminated": sum(row["terminated"] for row in records),
        "valid_output_tokens": sum(row["valid_output_tokens"] for row in records),
        "empty_outputs": sum(not row["raw_response"] for row in records),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in records)),
        "syntax_pass": sum(row["syntax"] == "pass" for row in records),
        "syntax_denominator": len(records) if args.mode == "line" else 0,
        "quality_caveat": (
            "Synthetic source completion; no general next-edit or human acceptance claim."
        ),
        "plan_sha256": metadata["plan_sha256"],
        "model_sha256": metadata["model_revision"],
        **(
            {"development_split": development_split}
            if plan.get("schema") == SCALE_PLAN_SCHEMA and development_split is not None
            else {}
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    current_runtime().shutdown()


if __name__ == "__main__":
    main()
