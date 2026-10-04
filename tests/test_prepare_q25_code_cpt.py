from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_q25_code_cpt import (
    DATASET_REVISION,
    LANGUAGE_ORDER,
    MODEL_ID,
    MODEL_REVISION,
    RepoSplit,
    _sha256_text,
    prepare_corpus,
)


class CharacterTokenizer:
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [3 + (ord(character) % 250) for character in text]


def _repo_in_bucket(prefix: str, buckets: range) -> str:
    splitter = RepoSplit(bucket_count=1000)
    for index in range(100_000):
        name = f"{prefix}-{index}"
        if splitter.bucket(name) in buckets:
            return name
    raise AssertionError("could not generate a repository in the desired bucket")


def _code(tag: str, *, lines: int = 100) -> str:
    return "\n".join(
        f"def {tag}_{index}(): result_{index} = {index} + 7; return result_{index}"
        for index in range(lines)
    )


def _sort_key(seed: int, split: str, content: str, repository: str) -> bytes:
    value = f"{seed}\0{split}\0{_sha256_text(content)}\0{repository}".encode()
    return hashlib.sha256(value).digest()


def _make_row(language: str, repository: str, content: str, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "language": language,
        "repository": repository,
        "repository_aliases": [],
        "bucket": RepoSplit(bucket_count=1000).bucket(repository),
        "path": f"src/{language}_module.{language}",
        "content": content,
        "content_sha256": _sha256_text(content),
        "licenses": ["MIT"],
    }
    row.update(extra)
    return row


def _write_frozen_pool(pool: Path, rows_by_language: dict[str, list[dict[str, object]]]):
    pool.mkdir(parents=True)
    source_manifest: dict[str, dict[str, object]] = {}
    for language in LANGUAGE_ORDER:
        data_path = pool / f"{language}.jsonl"
        payload = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows_by_language[language]
        )
        data_path.write_text(payload, encoding="utf-8")
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        sidecar_path = pool / f"{language}.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "source_revision": DATASET_REVISION,
                    "frozen_before_training": True,
                    "sha256": digest,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        source_manifest[language] = {
            "bytes": data_path.stat().st_size,
            "sha256": digest,
            "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        }
    return source_manifest


def test_preparation_applies_reservations_and_writes_only_train_and_dev(tmp_path: Path) -> None:
    seed = 314159
    rows: dict[str, list[dict[str, object]]] = {language: [] for language in LANGUAGE_ORDER}
    regular_train_repositories: dict[str, str] = {}
    development_repositories: dict[str, str] = {}

    for language in LANGUAGE_ORDER:
        train_repo = _repo_in_bucket(f"train-{language}", range(30, 970))
        dev_repo = _repo_in_bucket(f"dev-{language}", range(970, 1000))
        regular_train_repositories[language] = train_repo
        development_repositories[language] = dev_repo
        rows[language].append(_make_row(language, train_repo, _code(f"train_{language}")))
        rows[language].append(_make_row(language, dev_repo, _code(f"dev_{language}")))

    # A short accepted document feeds the next file's blocks and must still be in
    # provenance, even though it contributes zero complete blocks itself.
    short_content = "def tiny(): return " + " + ".join(str(i) for i in range(20))
    long_content = _code("go_train_long")
    go_short_repo = None
    for index in range(100_000):
        candidate = f"go-short-{index}"
        if RepoSplit(bucket_count=1000).bucket(candidate) not in range(30, 970):
            continue
        if _sort_key(seed, "train", short_content, candidate) < _sort_key(
            seed, "train", long_content, regular_train_repositories["go"]
        ):
            go_short_repo = candidate
            break
    assert go_short_repo is not None
    rows["go"] = [
        row for row in rows["go"] if row["repository"] != regular_train_repositories["go"]
    ]
    rows["go"].extend(
        [
            _make_row("go", go_short_repo, short_content),
            _make_row("go", regular_train_repositories["go"], long_content),
        ]
    )

    benchmark_alias = "fixtures/shared-benchmark-repository"
    benchmark_repo = _repo_in_bucket("benchmark-alias-row", range(30, 970))
    rows["typescript"].append(
        _make_row(
            "typescript",
            benchmark_repo,
            _code("benchmark_alias_row"),
            repository_aliases=[benchmark_alias],
        )
    )
    historical_repo = _repo_in_bucket("historical-reservation", range(30, 970))
    rows["python"].append(_make_row("python", historical_repo, _code("historical")))

    duplicate_content = _code("duplicate_across_splits", lines=30)
    duplicate_train_repo = _repo_in_bucket("duplicate-train", range(30, 970))
    duplicate_dev_repo = None
    normal_python_dev_content = _code("dev_python")
    for index in range(100_000):
        candidate = f"duplicate-dev-{index}"
        if RepoSplit(bucket_count=1000).bucket(candidate) not in range(970, 1000):
            continue
        if _sort_key(seed, "development", duplicate_content, candidate) < _sort_key(
            seed,
            "development",
            normal_python_dev_content,
            development_repositories["python"],
        ):
            duplicate_dev_repo = candidate
            break
    assert duplicate_dev_repo is not None
    alias_primary_repo = None
    alias_development_repo = _repo_in_bucket("alias-development", range(970, 1000))
    alias_content = _code("train-alias-assigned-to-dev", lines=30)
    for index in range(100_000):
        candidate = f"alias-primary-{index}"
        if RepoSplit(bucket_count=1000).bucket(candidate) not in range(30, 970):
            continue
        if _sort_key(seed, "development", alias_content, candidate) < _sort_key(
            seed,
            "development",
            normal_python_dev_content,
            development_repositories["python"],
        ):
            alias_primary_repo = candidate
            break
    assert alias_primary_repo is not None
    rows["python"].extend(
        [
            _make_row("python", duplicate_train_repo, duplicate_content),
            _make_row("python", duplicate_dev_repo, duplicate_content),
            _make_row(
                "python",
                alias_primary_repo,
                alias_content,
                repository_aliases=[alias_development_repo],
            ),
        ]
    )

    invalid_license_repo = _repo_in_bucket("invalid-license", range(30, 970))
    rows["rust"].append(
        _make_row("rust", invalid_license_repo, _code("invalid_license"), licenses=["Other"])
    )
    original_reserved_repo = _repo_in_bucket("original-reserved", range(0, 30))
    rows["rust"].append(_make_row("rust", original_reserved_repo, _code("original_reserved")))

    pool = tmp_path / "frozen-pool"
    source_manifest = _write_frozen_pool(pool, rows)
    exclusion_path = tmp_path / "excluded_repositories.json"
    exclusion_path.write_text(json.dumps([benchmark_alias]) + "\n", encoding="utf-8")
    exclusion_hash = hashlib.sha256(exclusion_path.read_bytes()).hexdigest()
    configuration = {
        "data": {
            "languages": {"python": 0.4, "rust": 0.25, "typescript": 0.2, "go": 0.15},
            "sequence_length": 1024,
            "new_q25_development_buckets": [970, 1000],
            "original_reserved_buckets": [0, 30],
            "train_buckets": [30, 970],
            "test_access": False,
            "train_tokens": 8192,
            "development_tokens": 16384,
            "allowed_licenses": [
                "MIT",
                "Apache-2.0",
                "BSD-2-Clause",
                "BSD-3-Clause",
                "ISC",
                "0BSD",
                "Unlicense",
                "CC0-1.0",
            ],
            "excluded_repositories_path": exclusion_path.name,
            "excluded_repositories_sha256": exclusion_hash,
        },
        "training": {"seed": seed},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "tokenizer_sha256": "test-tokenizer"},
        "budget": {
            "new_artifact_bytes_cap": 1_000_000,
            "minimum_free_bytes": 0,
            "maximum_additional_training_input_tokens": 100_000,
            "maximum_discarded_replay_input_tokens": 50_000,
        },
    }
    output = tmp_path / "prepared-corpus"
    metadata = prepare_corpus(
        source_pool=pool,
        output_dir=output,
        plan={"source_pool": source_manifest},
        configuration=configuration,
        tokenizer=CharacterTokenizer(),
        reserved_repositories={historical_repo},
        benchmark_reserved_repositories={benchmark_alias},
        benchmark_exclusion_path=exclusion_path,
        benchmark_exclusion_sha256=exclusion_hash,
    )

    assert (
        metadata["plan_sha256"]
        == "1aaff2ee4f4e0adeeafb87e616f99e657f7ff0be4fa438f40ecbb1150aeed1f9"
    )
    assert metadata["split"]["benchmark_overlap_source_rows_by_language"] == {"typescript": 1}
    assert metadata["split"]["benchmark_overlap_selected_rows_after_exclusion"] == 0
    assert (
        metadata["actual_training_input_tokens"]
        <= metadata["maximum_additional_training_input_tokens"]
    )
    assert metadata["maximum_discarded_replay_input_tokens"] == 50_000

    train = np.load(output / "train_blocks.npy", mmap_mode="r")
    development = np.load(output / "development_blocks.npy", mmap_mode="r")
    assert train.ndim == development.ndim == 2
    assert train.shape[1] == development.shape[1] == 1024
    assert not any("test" in path.name for path in output.iterdir())
    assert {path.name for path in output.iterdir()} == {
        "train_blocks.npy",
        "train_languages.npy",
        "development_blocks.npy",
        "train_manifest.jsonl",
        "development_manifest.jsonl",
        "corpus_metadata.json",
    }

    train_manifest = [
        json.loads(line) for line in (output / "train_manifest.jsonl").read_text().splitlines()
    ]
    development_manifest = [
        json.loads(line)
        for line in (output / "development_manifest.jsonl").read_text().splitlines()
    ]
    all_manifest = train_manifest + development_manifest
    all_repo_hashes = {
        value
        for record in all_manifest
        for value in [record["repository_identity_sha256"], *record["repository_alias_sha256"]]
    }
    assert _sha256_text(benchmark_alias) not in all_repo_hashes
    assert _sha256_text(historical_repo) not in all_repo_hashes
    assert _sha256_text(original_reserved_repo) not in all_repo_hashes
    assert _sha256_text(duplicate_train_repo) not in all_repo_hashes
    assert _sha256_text(duplicate_dev_repo) in all_repo_hashes
    assert _sha256_text(alias_primary_repo) not in {
        record["repository_identity_sha256"] for record in train_manifest
    }
    assert _sha256_text(alias_primary_repo) in {
        record["repository_identity_sha256"] for record in development_manifest
    }
    assert _sha256_text(invalid_license_repo) not in all_repo_hashes
    assert any(record["blocks_added"] == 0 for record in train_manifest)
    assert all("content" not in record for record in all_manifest)
