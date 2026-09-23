"""Model adapters and causal prompt serialization for code benchmarks."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import httpx

from tinycomplete.observability.artifacts import ArtifactStore
from tinycomplete.observability.bootstrap import current_runtime
from tinycomplete.observability.context import (
    PersistentRunIdentity,
    RunContext,
    context_callable,
    ensure_persistent_run_identity,
)
from tinycomplete.observability.hooks import (
    model_metrics,
    observed,
    observed_generation,
    usage_metrics,
)
from tinycomplete.observability.spans import operation

from .code_benchmark import BenchmarkCase, Prediction

PREDICTION_PROTOCOL = "causal-context-v1"


def returned_first_line(raw: str) -> str:
    """Remove one LF/CRLF terminator only; preserve meaningful whitespace."""
    line, newline, _ = raw.partition("\n")
    return line[:-1] if newline and line.endswith("\r") else line


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def build_prediction_run_metadata(
    *,
    suite_path: Path,
    case_count: int,
    provider: str,
    model_source: str,
    model_revision: str,
    max_new_tokens: int,
    workers: int,
    protocol: str = PREDICTION_PROTOCOL,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": protocol,
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "case_count": case_count,
        "provider": provider,
        "model_source": model_source,
        "model_revision": model_revision,
        "max_new_tokens": max_new_tokens,
        "workers": workers,
        "decoding": {"do_sample": False, "temperature": 0},
    }


def model_weight_fingerprint(model_path: Path) -> str:
    """Hash local weight bytes so a reused path cannot silently change models."""
    digest = hashlib.sha256()
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"no safetensors weights found in {model_path}")
    for path in weights:
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 2**20):
                digest.update(chunk)
    return digest.hexdigest()


def build_causal_prompt(case: BenchmarkCase) -> str:
    if not case.context_files:
        return case.prefix
    parts = []
    for path, content in sorted(case.context_files.items()):
        parts.append(f'<file path="{path}">\n{content}\n</file>\n')
    parts.append(f'<target path="{case.path}" language="{case.language}">\n{case.prefix}')
    return "".join(parts)


class GenerationProvider(Protocol):
    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int | None]: ...


@dataclass(frozen=True)
class DetailedGeneration:
    text: str
    tokens: int | None
    finish_reason: str | None
    input_tokens: int | None = None
    cache_tokens: int | None = None
    usage_source: str | None = None
    first_output_ms: float | None = None
    load_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    output_tokens_known: bool = True


class TransformersGenerationProvider:
    @observed("model.load")
    def __init__(
        self, model_path: str, *, device: str = "cpu", tokenizer_path: str | None = None
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested generation device is unavailable: {device}")
        self.model_name = model_path
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model_path, trust_remote_code=False
        )
        self.tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=False,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model: Any = loaded_model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int | None]:
        result = self.generate_detailed(prompt, max_new_tokens)
        return result.text, result.tokens

    @observed_generation
    def generate_detailed(self, prompt: str, max_new_tokens: int) -> DetailedGeneration:
        return self._generate(prompt, max_new_tokens, stop_first_line=False)

    @observed_generation
    def generate_line_detailed(self, prompt: str, max_new_tokens: int) -> DetailedGeneration:
        """Opt-in causal-line diagnostic; ordinary generation is unchanged."""
        return self._generate(prompt, max_new_tokens, stop_first_line=True)

    def _generate(
        self, prompt: str, max_new_tokens: int, *, stop_first_line: bool
    ) -> DetailedGeneration:
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in self.tokenizer(prompt, return_tensors="pt").items()
        }
        extra: dict[str, Any] = {}
        arrival: dict[str, float] = {}
        started = time.perf_counter()
        if stop_first_line:
            from transformers import StoppingCriteria, StoppingCriteriaList

            tokenizer = self.tokenizer
            start_length = inputs["input_ids"].shape[1]

            class FirstNewline(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    generated = input_ids[0, start_length:]
                    decoded = tokenizer.decode(generated, skip_special_tokens=True)
                    elapsed = time.perf_counter() - started
                    if generated.shape[0] in (1, 8, 16, 32):
                        arrival.setdefault(str(generated.shape[0]), elapsed)
                    return "\n" in decoded

            extra["stopping_criteria"] = StoppingCriteriaList([FirstNewline()])
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **extra,
            )
        tokens = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        token_count = int(tokens.numel())
        finish_reason = "length" if token_count >= max_new_tokens else "eos_or_stop"
        if stop_first_line and "\n" in text:
            finish_reason = "newline"
        self.last_line_token_arrivals = arrival if stop_first_line else {}
        return DetailedGeneration(
            str(text),
            token_count,
            finish_reason,
            input_tokens=int(inputs["input_ids"].numel()),
            usage_source="tokenizer",
            first_output_ms=arrival.get("1", 0) * 1000 if "1" in arrival else None,
        )


class OpenAICompatibleGenerationProvider:
    def __init__(self, server_url: str, model: str, *, timeout_seconds: float = 300) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.url = server_url.rstrip("/") + "/v1/completions"
        self.model = model
        self.model_name = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int | None]:
        result = self.generate_detailed(prompt, max_new_tokens)
        return result.text, result.tokens

    @observed_generation
    def generate_detailed(self, prompt: str, max_new_tokens: int) -> DetailedGeneration:
        response = httpx.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "stream": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        text = str(choice["text"])
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        completion_tokens = usage.get("completion_tokens")
        input_tokens = usage.get("prompt_tokens")
        cache_tokens = usage.get("cached_tokens")
        return DetailedGeneration(
            text,
            int(completion_tokens) if completion_tokens is not None else 0,
            choice.get("finish_reason"),
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            cache_tokens=int(cache_tokens) if cache_tokens is not None else None,
            usage_source="provider" if usage else None,
            output_tokens_known=completion_tokens is not None,
        )


def generate_predictions(
    cases: Iterable[Any],
    provider: GenerationProvider,
    output_path: Path,
    *,
    max_new_tokens: int = 128,
    run_metadata: dict[str, Any],
    workers: int = 1,
    prompt_builder: Callable[[Any], str] = build_causal_prompt,
) -> list[Prediction]:
    if workers < 1:
        raise ValueError("workers must be positive")
    metadata_path = prediction_metadata_path(output_path)
    runtime = current_runtime()
    identity = (
        ensure_persistent_run_identity(metadata_path, run_metadata)
        if runtime.config.enabled
        else PersistentRunIdentity(
            dict(run_metadata), RunContext("disabled", "disabled", "disabled")
        )
    )
    identity = replace(
        identity,
        context=replace(
            identity.context,
            workload={
                "tabcomplete.suite.sha256": str(run_metadata.get("suite_sha256", "unknown")),
                "tabcomplete.protocol": str(run_metadata.get("protocol", "unknown")),
                "tabcomplete.decoding": json.dumps(
                    run_metadata.get("decoding", {}), sort_keys=True
                ),
                "gen_ai.request.model": str(run_metadata.get("model_source", "unknown")),
                "tabcomplete.model.revision": str(run_metadata.get("model_revision", "unknown")),
            },
        ),
    )
    effective_metadata = identity.metadata
    if output_path.exists() and not metadata_path.exists():
        raise ValueError(f"prediction metadata missing for existing output: {output_path}")
    if metadata_path.exists():
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scientific_saved = {
            key: value for key, value in saved_metadata.items() if key != "observability"
        }
        if scientific_saved != run_metadata:
            raise ValueError("prediction run metadata does not match existing output")
        if runtime.config.enabled and "observability" not in saved_metadata:
            metadata_path.write_text(
                json.dumps(effective_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    else:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(effective_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    case_list = list(cases)
    completed: dict[str, Prediction] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line:
                prediction = Prediction.model_validate_json(line)
                if prediction.case_id in completed:
                    raise ValueError(f"duplicate prediction: {prediction.case_id}")
                completed[prediction.case_id] = prediction
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = [case for case in case_list if case.id not in completed]
    runtime = current_runtime()
    artifact_store = ArtifactStore(
        runtime.config.artifact_root,
        max_payload_bytes=runtime.config.artifact_max_payload_bytes,
        enabled=runtime.config.capture_content,
    )

    def predict(case: BenchmarkCase) -> Prediction:
        case_context = identity.context.for_case(case.id)
        with (
            case_context.activate(),
            operation(
                "eval.case",
                attributes={
                    "tabcomplete.task": effective_metadata.get("protocol", "unknown"),
                    "tabcomplete.language": case.language,
                },
            ) as case_span,
        ):
            prompt = prompt_builder(case)
            input_artifact = artifact_store.capture_text(
                "model-input",
                prompt,
                authorized=runtime.config.capture_content,
                source_ref=f"benchmark-case:{case.id}",
            )
            attributes: dict[str, Any] = {
                "gen_ai.operation.name": "text_completion",
                "gen_ai.provider.name": effective_metadata.get("provider", "unknown"),
                "gen_ai.request.model": effective_metadata.get("model_source", "unknown"),
                "gen_ai.request.max_tokens": max_new_tokens,
                "tabcomplete.model.revision": effective_metadata.get("model_revision", "unknown"),
                "tabcomplete.protocol": effective_metadata.get("protocol", "unknown"),
                **input_artifact.attributes("input"),
            }
            with operation("request.start", attributes=attributes):
                pass
            with operation("model.generate", attributes=attributes) as model_span:
                with model_metrics({"backend": str(effective_metadata.get("provider", "unknown"))}):
                    started = time.perf_counter()
                    detailed = getattr(provider, "generate_detailed", None)
                    if callable(detailed):
                        generation = detailed(prompt, max_new_tokens)
                        text = generation.text
                        tokens = generation.tokens
                        finish_reason = generation.finish_reason
                    else:
                        text, tokens = provider.generate(prompt, max_new_tokens)
                        finish_reason = None
                        generation = DetailedGeneration(text, tokens, finish_reason)
                    latency_seconds = time.perf_counter() - started
                model_span.set_attribute("tabcomplete.timing.total_ms", latency_seconds * 1000)
                usage_metrics(
                    generation, {"backend": str(effective_metadata.get("provider", "unknown"))}
                )
                model_span.set_attribute("tabcomplete.timing.kind", "end_to_end")
                if generation.input_tokens is not None:
                    model_span.set_attribute("gen_ai.usage.input_tokens", generation.input_tokens)
                if tokens is not None and generation.output_tokens_known:
                    model_span.set_attribute("gen_ai.usage.output_tokens", tokens)
                if generation.cache_tokens is not None:
                    model_span.set_attribute(
                        "gen_ai.usage.cache_read.input_tokens", generation.cache_tokens
                    )
                if generation.usage_source is not None:
                    model_span.set_attribute("tabcomplete.usage.source", generation.usage_source)
                if generation.first_output_ms is not None:
                    model_span.set_attribute(
                        "tabcomplete.timing.first_output_ms", generation.first_output_ms
                    )
                for timing_name, timing_value in (
                    ("load_ms", generation.load_ms),
                    ("prefill_ms", generation.prefill_ms),
                    ("decode_ms", generation.decode_ms),
                ):
                    if timing_value is not None:
                        model_span.set_attribute(f"tabcomplete.timing.{timing_name}", timing_value)
                if finish_reason is not None:
                    model_span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
                output_artifact = artifact_store.capture_text(
                    "model-output",
                    text,
                    authorized=runtime.config.capture_content,
                    source_ref=f"{output_path}#case={case.id}",
                )
                for key, value in output_artifact.attributes("output").items():
                    model_span.set_attribute(key, value)
                model_span.set_attribute(
                    "tabcomplete.output.truncated",
                    finish_reason in {"length", "max_tokens"}
                    or (tokens is not None and tokens >= max_new_tokens),
                )
            hit_token_cap = finish_reason in {"length", "max_tokens"} or (
                tokens is not None and tokens >= max_new_tokens
            )
            case_span.set_attribute("tabcomplete.outcome", "completed")
            case_span.set_attribute("tabcomplete.output.truncated", hit_token_cap)
            return Prediction(
                case_id=case.id,
                completion=text,
                generated_tokens=tokens,
                latency_seconds=latency_seconds,
                finish_reason=finish_reason,
                hit_token_cap=hit_token_cap,
            )

    with (
        identity.context.activate(),
        operation(
            "run.start",
            attributes={
                "tabcomplete.run.state": "started",
                "tabcomplete.cases.planned": len(case_list),
                "tabcomplete.cases.previously_completed": len(completed),
            },
        ),
    ):
        pass
    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(runtime.config.heartbeat_seconds):
            with (
                runtime.activate(),
                identity.context.activate(),
                operation(
                    "run.heartbeat",
                    attributes={
                        "tabcomplete.run.state": "active",
                        "tabcomplete.cases.planned": len(case_list),
                        "tabcomplete.cases.completed": len(completed),
                    },
                ),
            ):
                pass

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name="tabcomplete-observability-heartbeat",
        daemon=True,
    )
    if runtime.config.enabled:
        heartbeat_thread.start()
    failed = False
    try:
        with identity.context.activate(), output_path.open("a", encoding="utf-8") as handle:

            def save(prediction: Prediction) -> None:
                handle.write(json.dumps(prediction.model_dump(mode="json"), sort_keys=True) + "\n")
                handle.flush()
                completed[prediction.case_id] = prediction

            if workers == 1:
                for case in pending:
                    save(predict(case))
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(context_callable(lambda row=case: predict(row)))
                        for case in pending
                    ]
                    for future in as_completed(futures):
                        save(future.result())
    except BaseException:
        failed = True
        raise
    finally:
        heartbeat_stop.set()
        if runtime.config.enabled:
            heartbeat_thread.join(timeout=max(runtime.config.heartbeat_seconds * 2, 0.1))
        with (
            identity.context.activate(),
            operation(
                "run.summary",
                attributes={
                    "tabcomplete.run.state": "failed" if failed else "completed",
                    "tabcomplete.cases.planned": len(case_list),
                    "tabcomplete.cases.completed": len(completed),
                },
            ),
        ):
            pass
    return [completed[case.id] for case in case_list]
