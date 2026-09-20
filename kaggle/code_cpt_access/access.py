"""CPU-only Kaggle gate for Hugging Face identity and Stack Dedup access."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


DATASET_ID = "bigcode/the-stack-dedup"
DATASET_REVISION = "17cad72c886a2858e08d4c349a00d6466f54df63"
DATA_DIRS = {
    "python": "data/python",
    "typescript": "data/typescript",
    "javascript": "data/javascript",
    "java": "data/java",
    "cpp": "data/cpp",
    "rust": "data/rust",
    "go": "data/go",
    "c": "data/c",
    "csharp": "data/c-sharp",
    "shell": "data/shell",
}
OUTPUT = Path("/kaggle/working/stack_access.json")


def install_dependencies() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "datasets>=4.0", "huggingface-hub>=0.34"],
        check=True,
    )


def get_token() -> tuple[str, str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token, "environment"
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token, "kaggle_secret"
    except Exception:
        pass
    for path in Path("/kaggle/input").glob("**/hf_token.txt"):
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token, "private_kaggle_dataset"
    raise RuntimeError("HF_TOKEN is unavailable in all configured credential sources")


def main() -> None:
    install_dependencies()
    from datasets import load_dataset
    from huggingface_hub import HfApi

    report: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "hf_authentication": "FAIL",
        "stack_dedup_access": "FAIL",
        "data_dirs": {},
    }
    try:
        token, credential_source = get_token()
        identity = HfApi(token=token).whoami()
        username = identity.get("name") or identity.get("fullname") or "authenticated-user"
        report.update(
            hf_authentication="PASS",
            authenticated_username=username,
            credential_source=credential_source,
        )
        print("HF authentication: PASS")
        print("Authenticated username:", username)

        for language, data_dir in DATA_DIRS.items():
            stream = load_dataset(
                DATASET_ID,
                data_dir=data_dir,
                split="train",
                streaming=True,
                token=token,
                revision=DATASET_REVISION,
            )
            row = next(iter(stream))
            assert isinstance(row.get("content"), str) and row["content"]
            report["data_dirs"][language] = data_dir
            print(f"{language}: PASS")
        report["stack_dedup_access"] = "PASS"
        print("The Stack Dedup access: PASS")
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        print("Access probe failed with", type(exc).__name__)
        raise
    finally:
        OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
