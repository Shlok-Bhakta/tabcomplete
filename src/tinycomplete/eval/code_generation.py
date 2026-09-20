"""Model adapters and causal prompt serialization for code benchmarks."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

import httpx

from .code_benchmark import BenchmarkCase, Prediction

PREDICTION_PROTOCOL = "causal-context-v1"


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
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
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
    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int]: ...


class TransformersGenerationProvider:
    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=False,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return str(text), int(tokens.numel())


class OpenAICompatibleGenerationProvider:
    def __init__(self, server_url: str, model: str) -> None:
        self.url = server_url.rstrip("/") + "/v1/completions"
        self.model = model

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
        response = httpx.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        body = response.json()
        text = str(body["choices"][0]["text"])
        tokens = int(body.get("usage", {}).get("completion_tokens", 0))
        return text, tokens


def generate_predictions(
    cases: Iterable[BenchmarkCase],
    provider: GenerationProvider,
    output_path: Path,
    *,
    max_new_tokens: int = 128,
    run_metadata: dict[str, Any],
    workers: int = 1,
) -> list[Prediction]:
    if workers < 1:
        raise ValueError("workers must be positive")
    metadata_path = prediction_metadata_path(output_path)
    if output_path.exists() and not metadata_path.exists():
        raise ValueError(f"prediction metadata missing for existing output: {output_path}")
    if metadata_path.exists():
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved_metadata != run_metadata:
            raise ValueError("prediction run metadata does not match existing output")
    else:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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

    def predict(case: BenchmarkCase) -> Prediction:
        started = time.perf_counter()
        text, tokens = provider.generate(build_causal_prompt(case), max_new_tokens)
        return Prediction(
            case_id=case.id,
            completion=text,
            generated_tokens=tokens,
            latency_seconds=time.perf_counter() - started,
        )

    with output_path.open("a", encoding="utf-8") as handle:

        def save(prediction: Prediction) -> None:
            handle.write(json.dumps(prediction.model_dump(mode="json"), sort_keys=True) + "\n")
            handle.flush()
            completed[prediction.case_id] = prediction

        if workers == 1:
            for case in pending:
                save(predict(case))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(predict, case) for case in pending]
                for future in as_completed(futures):
                    save(future.result())
    return [completed[case.id] for case in case_list]
