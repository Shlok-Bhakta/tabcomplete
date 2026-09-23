"""Record exact local source and conversion bytes without copying checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from tinycomplete.eval.code_generation import file_sha256

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/research/model_data_r2"
REPORT = ROOT / "reports/research/model_data_r2/models"


def main():
    cache = Path.home() / ".cache/huggingface/hub"
    models = {
        "q35-p12": ROOT.parent
        / "tabcomplete/outputs/kaggle/code_cpt_train_v3/code_cpt_run/main/final",
        "q35-d12": ROOT.parent
        / "tabcomplete-cpt-recipe-r1/artifacts/code_cpt/research_r1/campaign_full"
        / "code_cpt_campaign_r1/arms/D12/final",
        "q35-base": cache
        / "models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "q25-coder": cache
        / "models--Qwen--Qwen2.5-Coder-0.5B/snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
        "granite-h350": cache
        / "models--ibm-granite--granite-4.0-h-350m-base/snapshots"
        / "dc555b6939c863bb96034d1eae7601da36bd42e4",
    }
    runtime = ROOT.parent / "tabcomplete/outputs/tools/llama.cpp"
    conversions = {"q35-p12": "p12-text", "q25-coder": "q25", "granite-h350": "granite"}
    baseline = json.loads((ARTIFACTS / "baseline/model_data_r2_baseline/progress.json").read_text())
    records = {}
    for alias, directory in models.items():
        config = json.loads((directory / "config.json").read_text())
        tokenizer = json.loads((directory / "tokenizer_config.json").read_text())
        record = {
            "source_files": [
                {
                    "file": name,
                    "sha256": file_sha256(directory / name),
                    "bytes": (directory / name).stat().st_size,
                }
                for name in ("config.json", "tokenizer.json", "model.safetensors")
                if (directory / name).exists()
            ],
            "configuration": config,
            "tokenizer_class": tokenizer.get("tokenizer_class"),
            "actual_loaded_parameters": baseline.get("models", {}).get(alias, {}).get("parameters"),
            "actual_loaded_model_class": baseline.get("models", {})
            .get(alias, {})
            .get("model_class"),
            "parameter_count_source": "R2 GPU loaded text model; null means not yet observed",
            "local_weight_bytes_available": (directory / "model.safetensors").exists(),
        }
        if alias in conversions:
            prefix = conversions[alias]
            record["conversion"] = {
                "runtime_revision": "f072b103714dfa1eee531f80b24512faf38e3dd2",
                "converter_sha256": file_sha256(runtime / "convert_hf_to_gguf.py"),
                "server_sha256": file_sha256(runtime / "build/bin/llama-server"),
                "quantizer_sha256": file_sha256(runtime / "build/bin/llama-quantize"),
                "quantization": "Q4_K_M",
                "source_precision": "f16",
                "extra_options": ["--no-mtp"] if alias == "q35-p12" else [],
                "outputs": [
                    {
                        "file": prefix + suffix,
                        "sha256": file_sha256(ARTIFACTS / (prefix + suffix)),
                        "bytes": (ARTIFACTS / (prefix + suffix)).stat().st_size,
                    }
                    for suffix in ("-f16.gguf", "-Q4_K_M.gguf")
                ],
                "source_checkpoint_modified": False,
                "quantized_comparison_caveat": "Own HF source; runtime and precision both differ",
            }
        records[alias] = record
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "verified-local-models.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"models": len(records), "own_conversions": len(conversions)}))


if __name__ == "__main__":
    main()
