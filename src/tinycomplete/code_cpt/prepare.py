"""Stream, filter, tokenize, and freeze the Stage-1 training and MICRO corpora."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from tinycomplete.code_cpt.data import (
    BlockPacker,
    BlockProvenanceTracker,
    RepoAssignment,
    RepoSplit,
    SourceFilter,
    allocate_blocks,
    repository_identity,
    repository_path,
)

DATASET_ID = "bigcode/the-stack-dedup"
DATASET_REVISION = "17cad72c886a2858e08d4c349a00d6466f54df63"
MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
LANGUAGE_SPECS = {
    "python": ("data/python", 0.20),
    "typescript": ("data/typescript", 0.15),
    "javascript": ("data/javascript", 0.10),
    "java": ("data/java", 0.10),
    "cpp": ("data/cpp", 0.10),
    "rust": ("data/rust", 0.10),
    "go": ("data/go", 0.08),
    "c": ("data/c", 0.06),
    "csharp": ("data/c-sharp", 0.06),
    "shell": ("data/shell", 0.05),
}
CORE_LANGUAGES = tuple(language for language in LANGUAGE_SPECS if language != "shell")


def research_split_for_bucket(bucket: int) -> str:
    """Map stable repository buckets to the research-r1 data partitions."""
    if not 0 <= bucket < 1000:
        raise ValueError("repository bucket must be in [0, 1000)")
    if bucket < 10:
        return "excluded_stage1_validation"
    if bucket < 20:
        return "development"
    if bucket < 30:
        return "test"
    return "train"


def hash_packed_blocks(path: Path) -> set[str]:
    """Return exact hashes for packed token blocks without loading the array."""
    blocks = np.load(path, mmap_mode="r")
    if blocks.ndim != 2:
        raise ValueError("packed block array must have shape [blocks, sequence]")
    return {sha256(np.asarray(block).tobytes()).hexdigest() for block in blocks}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_fingerprint(paths: list[Path]) -> str:
    """Bind a corpus fingerprint to file names, sizes, and exact bytes."""
    digest = sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


@dataclass
class StreamStats:
    raw_files: int = 0
    raw_bytes: int = 0
    accepted_files: int = 0
    accepted_bytes: int = 0
    tokenized_tokens: int = 0
    tokenizer_seconds: float = 0.0
    wall_seconds: float = 0.0
    train_blocks: int = 0
    validation_blocks: int = 0
    training_packing_efficiency: float = 0.0
    validation_packing_efficiency: float = 0.0

    def rates(self) -> dict[str, float]:
        return {
            "raw_network_mb_s": self.raw_bytes / 1_000_000 / self.wall_seconds,
            "accepted_source_mb_s": self.accepted_bytes / 1_000_000 / self.wall_seconds,
            "source_files_s": self.raw_files / self.wall_seconds,
            "tokenizer_tokens_s": self.tokenized_tokens / self.tokenizer_seconds,
            "packer_tokens_s": self.tokenized_tokens / self.wall_seconds,
        }


def resolve_hf_token() -> tuple[str, str]:
    """Resolve a token without displaying or storing it in an artifact."""
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
    private_files = list(
        Path("/kaggle/input").glob("**/tabcomplete-hf-read-token*/hf_token.txt")
    ) + list(Path("/kaggle/input").glob("**/hf_token.txt"))
    for path in private_files:
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token, "private_kaggle_dataset"
    raise RuntimeError("HF_TOKEN is unavailable in all configured credential sources")


def verify_access(token: str) -> dict[str, object]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    identity = HfApi(token=token).whoami()
    username = identity.get("name") or identity.get("fullname") or "authenticated-user"
    available = {}
    for language, (data_dir, _) in LANGUAGE_SPECS.items():
        stream = load_dataset(
            DATASET_ID,
            data_dir=data_dir,
            split="train",
            streaming=True,
            token=token,
            revision=DATASET_REVISION,
        )
        row = next(iter(stream))
        if not isinstance(row.get("content"), str) or not row["content"]:
            raise RuntimeError(f"{data_dir} did not yield a nonempty source record")
        available[language] = data_dir
    return {"username": username, "available_target_language_dirs": available}


def _open_array(path: Path, shape: tuple[int, ...], dtype=np.uint32):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _write_blocks(array, start: int, blocks: list[list[int]], limit: int) -> int:
    count = min(len(blocks), limit - start)
    if count:
        array[start : start + count] = np.asarray(blocks[:count], dtype=np.uint32)
    return count


def _metadata_reject(row: dict) -> str | None:
    size = row.get("size")
    if isinstance(size, int) and (size < 64 or size > 1_000_000):
        return "metadata_size"
    average = row.get("avg_line_length")
    if isinstance(average, (int, float)) and average > 500:
        return "metadata_average_line"
    maximum = row.get("max_line_length")
    if isinstance(maximum, int) and maximum > 20_000:
        return "metadata_max_line"
    fraction = row.get("alphanum_fraction")
    if isinstance(fraction, (int, float)) and fraction < 0.10:
        return "metadata_alphanumeric"
    return None


def _prepare_language(
    *,
    language: str,
    data_dir: str,
    train_blocks: int,
    validation_blocks: int,
    block_size: int,
    tokenizer,
    token: str,
    output_dir: Path,
    seed: int,
    shuffle_buffer: int,
) -> tuple[Path, Path | None, StreamStats, Counter[str], list[dict], dict[str, dict]]:
    from datasets import load_dataset

    stream = load_dataset(
        DATASET_ID,
        data_dir=data_dir,
        split="train",
        streaming=True,
        token=token,
        revision=DATASET_REVISION,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)
    train_path = output_dir / "language_blocks" / f"{language}.npy"
    validation_path = output_dir / "micro" / f"{language}.npy" if validation_blocks else None
    train_array = _open_array(train_path, (train_blocks, block_size))
    validation_array = (
        _open_array(validation_path, (validation_blocks, block_size))
        if validation_path is not None
        else None
    )
    split = RepoSplit(validation_buckets=range(10), bucket_count=1000)
    source_filter = SourceFilter()
    train_packer = BlockPacker(block_size, tokenizer.eos_token_id)
    validation_packer = BlockPacker(block_size, tokenizer.eos_token_id)
    train_written = 0
    validation_written = 0
    rejected: Counter[str] = Counter()
    validation_records: list[dict] = []
    validation_repositories: dict[str, dict] = {}
    stats = StreamStats()
    wall_start = time.perf_counter()
    for row in stream:
        stats.raw_files += 1
        content = row.get("content")
        if not isinstance(content, str):
            rejected["missing_content"] += 1
            continue
        stats.raw_bytes += len(content.encode("utf-8", errors="ignore"))
        metadata_reason = _metadata_reject(row)
        if metadata_reason:
            rejected[metadata_reason] += 1
            continue
        try:
            repo_name = repository_identity(row)
            path = repository_path(row)
        except ValueError:
            rejected["missing_repository"] += 1
            continue
        checked = source_filter.check(content, path)
        if not checked.accepted:
            rejected[str(checked.reason)] += 1
            continue
        assignment = split.assignment(repo_name)
        if assignment == RepoAssignment.TRAIN and train_written >= train_blocks:
            continue
        if assignment == RepoAssignment.VALIDATION and validation_written >= validation_blocks:
            continue
        stats.accepted_files += 1
        stats.accepted_bytes += len(content.encode("utf-8", errors="ignore"))
        tokenize_start = time.perf_counter()
        ids = tokenizer.encode(content, add_special_tokens=False)
        stats.tokenizer_seconds += time.perf_counter() - tokenize_start
        stats.tokenized_tokens += len(ids)
        if not ids:
            rejected["empty_tokens"] += 1
            continue
        if assignment == RepoAssignment.TRAIN:
            blocks = train_packer.add_document(ids)
            train_written += _write_blocks(train_array, train_written, blocks, train_blocks)
        else:
            blocks = validation_packer.add_document(ids)
            validation_written += _write_blocks(
                validation_array, validation_written, blocks, validation_blocks
            )
            content_hash = sha256(content.encode("utf-8")).hexdigest()
            validation_records.append(
                {
                    "language": language,
                    "repository": repo_name,
                    "repo_bucket": split.bucket(repo_name),
                    "path": path,
                    "hexsha": row.get("hexsha"),
                    "content_sha256": content_hash,
                    "tokens": len(ids),
                }
            )
            validation_repositories.setdefault(
                repo_name,
                {
                    "identity_sha256": sha256(repo_name.encode("utf-8")).hexdigest(),
                    "repo_bucket": split.bucket(repo_name),
                },
            )
        if train_written >= train_blocks and validation_written >= validation_blocks:
            break
    stats.wall_seconds = time.perf_counter() - wall_start
    stats.train_blocks = train_written
    stats.validation_blocks = validation_written
    train_available = train_packer.source_tokens + train_packer.boundary_tokens
    validation_available = validation_packer.source_tokens + validation_packer.boundary_tokens
    stats.training_packing_efficiency = train_packer.emitted_tokens / train_available
    stats.validation_packing_efficiency = (
        validation_packer.emitted_tokens / validation_available if validation_available else 0.0
    )
    train_array.flush()
    if validation_array is not None:
        validation_array.flush()
    if train_written != train_blocks or validation_written != validation_blocks:
        raise RuntimeError(
            f"{language} stream exhausted early: train {train_written}/{train_blocks}, "
            f"validation {validation_written}/{validation_blocks}"
        )
    return (
        train_path,
        validation_path,
        stats,
        rejected,
        validation_records,
        validation_repositories,
    )


def _write_fresh_blocks(
    array,
    start: int,
    blocks: list[list[int]],
    limit: int,
    *,
    forbidden_hashes: set[str],
    emitted_hashes: set[str],
) -> tuple[int, int, list[int]]:
    """Write nonduplicate blocks and return counts plus accepted input indices."""
    written = 0
    rejected = 0
    accepted_indices = []
    for input_index, block in enumerate(blocks):
        if start + written >= limit:
            break
        packed = np.asarray(block, dtype=np.uint32)
        digest = sha256(packed.tobytes()).hexdigest()
        if digest in forbidden_hashes or digest in emitted_hashes:
            rejected += 1
            continue
        array[start + written] = packed
        emitted_hashes.add(digest)
        accepted_indices.append(input_index)
        written += 1
    return written, rejected, accepted_indices


def _prepare_research_language(
    *,
    language: str,
    data_dir: str,
    train_blocks: int,
    development_blocks: int,
    test_blocks: int,
    block_size: int,
    tokenizer,
    token: str,
    output_dir: Path,
    seed: int,
    shuffle_buffer: int,
    excluded_repositories: set[str],
    excluded_content_hashes: set[str],
    old_block_hashes: set[str],
    seen_content_hashes: set[str],
    emitted_block_hashes: set[str],
) -> tuple[dict[str, Path], StreamStats, Counter[str], dict[str, list[dict]]]:
    """Build one language of the frozen fresh train/dev/test corpus."""
    from datasets import load_dataset

    stream = load_dataset(
        DATASET_ID,
        data_dir=data_dir,
        split="train",
        streaming=True,
        token=token,
        revision=DATASET_REVISION,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)
    limits = {
        "train": train_blocks,
        "development": development_blocks,
        "test": test_blocks,
    }
    paths = {
        "train": output_dir / "language_blocks" / f"{language}.npy",
        "development": output_dir / "micro" / f"{language}.npy",
        "test": output_dir / "test" / f"{language}.npy",
    }
    arrays = {
        split_name: _open_array(path, (limit, block_size))
        for split_name, (path, limit) in {
            name: (paths[name], limits[name]) for name in limits
        }.items()
        if limit
    }
    packers = {
        name: BlockPacker(block_size, tokenizer.eos_token_id)
        for name, limit in limits.items()
        if limit
    }
    provenance_trackers = {
        name: BlockProvenanceTracker(block_size) for name, limit in limits.items() if limit
    }
    block_provenance: dict[str, list[dict]] = {name: [] for name in limits}
    written = Counter()
    rejected: Counter[str] = Counter()
    manifests: dict[str, list[dict]] = {name: [] for name in limits}
    stats = StreamStats()
    splitter = RepoSplit(validation_buckets=range(10), bucket_count=1000)
    source_filter = SourceFilter()
    wall_start = time.perf_counter()
    for row in stream:
        if all(written[name] >= limit for name, limit in limits.items()):
            break
        stats.raw_files += 1
        content = row.get("content")
        if not isinstance(content, str):
            rejected["missing_content"] += 1
            continue
        stats.raw_bytes += len(content.encode("utf-8", errors="ignore"))
        metadata_reason = _metadata_reject(row)
        if metadata_reason:
            rejected[metadata_reason] += 1
            continue
        try:
            repo_name = repository_identity(row)
            path = repository_path(row)
        except ValueError:
            rejected["missing_repository"] += 1
            continue
        if repo_name in excluded_repositories:
            rejected["excluded_repository"] += 1
            continue
        split_name = research_split_for_bucket(splitter.bucket(repo_name))
        if split_name.startswith("excluded_"):
            rejected[split_name] += 1
            continue
        if split_name not in limits or written[split_name] >= limits[split_name]:
            continue
        checked = source_filter.check(content, path)
        if not checked.accepted:
            rejected[str(checked.reason)] += 1
            continue
        content_bytes = content.encode("utf-8")
        content_hash = sha256(content_bytes).hexdigest()
        if content_hash in excluded_content_hashes:
            rejected["excluded_content"] += 1
            continue
        if content_hash in seen_content_hashes:
            rejected["duplicate_content"] += 1
            continue
        seen_content_hashes.add(content_hash)
        stats.accepted_files += 1
        stats.accepted_bytes += len(content_bytes)
        tokenize_start = time.perf_counter()
        ids = tokenizer.encode(content, add_special_tokens=False)
        stats.tokenizer_seconds += time.perf_counter() - tokenize_start
        stats.tokenized_tokens += len(ids)
        if not ids:
            rejected["empty_tokens"] += 1
            continue
        blocks = packers[split_name].add_document(ids)
        repository_hash = sha256(repo_name.encode("utf-8")).hexdigest()
        provenance_blocks = provenance_trackers[split_name].add_document(
            repository_hash, len(ids)
        )
        if len(provenance_blocks) != len(blocks):
            raise RuntimeError("token blocks and provenance blocks diverged")
        forbidden = old_block_hashes
        output_start = written[split_name]
        count, duplicate_blocks, accepted_indices = _write_fresh_blocks(
            arrays[split_name],
            output_start,
            blocks,
            limits[split_name],
            forbidden_hashes=forbidden,
            emitted_hashes=emitted_block_hashes,
        )
        written[split_name] += count
        rejected["duplicate_packed_block"] += duplicate_blocks
        for output_offset, input_index in enumerate(accepted_indices):
            block_provenance[split_name].append(
                {
                    "block_index": output_start + output_offset,
                    "language": language,
                    "spans": provenance_blocks[input_index],
                }
            )
        manifests[split_name].append(
            {
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "language": language,
                "repository": repo_name,
                "repository_identity_sha256": sha256(repo_name.encode("utf-8")).hexdigest(),
                "repo_bucket": splitter.bucket(repo_name),
                "path": path,
                "hexsha": row.get("hexsha"),
                "content_sha256": content_hash,
                "source_tokens": len(ids),
                "split": split_name,
            }
        )
    stats.wall_seconds = time.perf_counter() - wall_start
    stats.train_blocks = written["train"]
    stats.validation_blocks = written["development"]
    train_packer = packers.get("train")
    if train_packer is not None:
        available = train_packer.source_tokens + train_packer.boundary_tokens
        stats.training_packing_efficiency = train_packer.emitted_tokens / available
    dev_packer = packers.get("development")
    if dev_packer is not None:
        available = dev_packer.source_tokens + dev_packer.boundary_tokens
        stats.validation_packing_efficiency = dev_packer.emitted_tokens / available
    for array in arrays.values():
        array.flush()
    for split_name in ("development", "test"):
        if not block_provenance[split_name]:
            continue
        directory = output_dir / ("micro" if split_name == "development" else "test")
        provenance_path = directory / f"{language}_provenance.jsonl"
        with provenance_path.open("w", encoding="utf-8") as handle:
            for record in block_provenance[split_name]:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        paths[f"{split_name}_provenance"] = provenance_path
    for name, limit in limits.items():
        if written[name] != limit:
            raise RuntimeError(
                f"{language} stream exhausted early for {name}: {written[name]}/{limit} blocks"
            )
    return paths, stats, rejected, manifests


def _combine_training_blocks(
    *,
    paths: dict[str, Path],
    allocations: dict[str, int],
    block_size: int,
    output_dir: Path,
    seed: int,
) -> tuple[Path, Path, list[str]]:
    languages = list(allocations)
    schedule = [language for language in languages for _ in range(allocations[language])]
    random.Random(seed).shuffle(schedule)
    output_path = output_dir / "train_blocks.npy"
    language_path = output_dir / "train_languages.npy"
    output = _open_array(output_path, (len(schedule), block_size))
    language_ids = _open_array(language_path, (len(schedule),), dtype=np.uint8)
    sources = {language: np.load(paths[language], mmap_mode="r") for language in languages}
    positions: Counter[str] = Counter()
    language_index = {language: index for index, language in enumerate(languages)}
    for target_index, language in enumerate(schedule):
        output[target_index] = sources[language][positions[language]]
        language_ids[target_index] = language_index[language]
        positions[language] += 1
    output.flush()
    language_ids.flush()
    return output_path, language_path, languages


def _prepare_general_micro(
    *, tokenizer, token: str, block_size: int, blocks: int, output_dir: Path
) -> Path:
    from datasets import load_dataset

    stream = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="validation",
        streaming=True,
        token=token,
    )
    path = output_dir / "micro" / "general.npy"
    array = _open_array(path, (blocks, block_size))
    packer = BlockPacker(block_size, tokenizer.eos_token_id)
    written = 0
    for row in stream:
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        written += _write_blocks(
            array,
            written,
            packer.add_document(tokenizer.encode(text, add_special_tokens=False)),
            blocks,
        )
        if written >= blocks:
            break
    array.flush()
    if written != blocks:
        raise RuntimeError(f"WikiText stream yielded only {written}/{blocks} general blocks")
    return path


def _load_known_exclusions(
    old_corpus_dir: Path, exclusion_root: Path | None
) -> tuple[set[str], set[str]]:
    repositories: set[str] = set()
    content_hashes: set[str] = set()
    repository_path = old_corpus_dir / "validation_repositories.json"
    if repository_path.exists():
        repositories.update(json.loads(repository_path.read_text(encoding="utf-8")))
    records_path = old_corpus_dir / "validation_records.jsonl"
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("repository"):
                repositories.add(record["repository"])
            if record.get("content_sha256"):
                content_hashes.add(record["content_sha256"])
    if exclusion_root is not None and exclusion_root.exists():
        for path in sorted(exclusion_root.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stack = [record]
                while stack:
                    value = stack.pop()
                    if isinstance(value, dict):
                        stack.extend(value.values())
                    elif isinstance(value, list):
                        stack.extend(value)
                    elif isinstance(value, str) and value:
                        content_hashes.add(sha256(value.encode("utf-8")).hexdigest())
                if isinstance(record, dict):
                    prefix = record.get("prefix")
                    expected = record.get("expected")
                    suffix = record.get("suffix")
                    if all(isinstance(value, str) for value in (prefix, expected, suffix)):
                        content_hashes.add(
                            sha256(f"{prefix}{expected}{suffix}".encode()).hexdigest()
                        )
                    current = record.get("current")
                    if isinstance(current, str):
                        content_hashes.add(sha256(current.encode("utf-8")).hexdigest())
    return repositories, content_hashes


def prepare_research_corpus(
    output_dir: Path,
    *,
    old_corpus_dir: Path,
    exclusion_root: Path | None = None,
    train_tokens: int = 5_013_504,
    development_tokens_per_language: int = 114_688,
    test_tokens_per_language: int = 114_688,
    general_tokens: int = 16_384,
    block_size: int = 2048,
    shuffle_buffer: int = 10_000,
    seed: int = 424_242,
) -> dict:
    """Freeze the fresh, matched research-r1 train/dev/test corpus."""
    from transformers import AutoTokenizer

    if train_tokens % block_size:
        raise ValueError("train_tokens must be an exact multiple of block_size")
    output_dir.mkdir(parents=True, exist_ok=True)
    token, credential_source = resolve_hf_token()
    access = verify_access(token)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, token=token
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token for source boundaries")
    old_train_path = old_corpus_dir / "train_blocks.npy"
    if not old_train_path.exists():
        raise FileNotFoundError(f"old Stage-1 train blocks missing: {old_train_path}")
    old_block_hashes = hash_packed_blocks(old_train_path)
    excluded_repositories, excluded_content_hashes = _load_known_exclusions(
        old_corpus_dir, exclusion_root
    )
    total_blocks = train_tokens // block_size
    weights = {language: weight for language, (_, weight) in LANGUAGE_SPECS.items()}
    allocations = allocate_blocks(weights, total_blocks)
    development_blocks = development_tokens_per_language // block_size
    test_blocks = test_tokens_per_language // block_size
    if development_blocks < 1 or test_blocks < 1:
        raise ValueError("development and test slices need at least one block per language")
    train_paths: dict[str, Path] = {}
    feeder = {}
    manifests: dict[str, list[dict]] = {"train": [], "development": [], "test": []}
    provenance_paths: list[Path] = []
    seen_content_hashes: set[str] = set()
    emitted_block_hashes: set[str] = set()
    for index, (language, (data_dir, _)) in enumerate(LANGUAGE_SPECS.items()):
        core = language in CORE_LANGUAGES
        paths, stats, rejected, language_manifests = _prepare_research_language(
            language=language,
            data_dir=data_dir,
            train_blocks=allocations[language],
            development_blocks=development_blocks if core else 0,
            test_blocks=test_blocks if core else 0,
            block_size=block_size,
            tokenizer=tokenizer,
            token=token,
            output_dir=output_dir,
            seed=seed + index,
            shuffle_buffer=shuffle_buffer,
            excluded_repositories=excluded_repositories,
            excluded_content_hashes=excluded_content_hashes,
            old_block_hashes=old_block_hashes,
            seen_content_hashes=seen_content_hashes,
            emitted_block_hashes=emitted_block_hashes,
        )
        train_paths[language] = paths["train"]
        provenance_paths.extend(
            path for name, path in paths.items() if name.endswith("_provenance")
        )
        feeder[language] = {**asdict(stats), **stats.rates(), "rejected": dict(rejected)}
        for split_name in manifests:
            manifests[split_name].extend(language_manifests[split_name])
    train_path, language_path, language_order = _combine_training_blocks(
        paths=train_paths,
        allocations=allocations,
        block_size=block_size,
        output_dir=output_dir,
        seed=seed,
    )
    for path in train_paths.values():
        path.unlink()
    shutil.rmtree(output_dir / "language_blocks")
    general_path = _prepare_general_micro(
        tokenizer=tokenizer,
        token=token,
        block_size=block_size,
        blocks=max(1, general_tokens // block_size),
        output_dir=output_dir,
    )
    manifest_paths = {}
    for split_name, records in manifests.items():
        path = output_dir / f"{split_name}_manifest.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        manifest_paths[split_name] = path
    corpus_files = [train_path, language_path, general_path, *manifest_paths.values()]
    corpus_files.extend(sorted((output_dir / "micro").glob("*.npy")))
    corpus_files.extend(sorted((output_dir / "test").glob("*.npy")))
    package_versions = {}
    for name in ("datasets", "huggingface_hub", "numpy", "tokenizers", "transformers"):
        module = __import__(name)
        package_versions[name] = getattr(module, "__version__", "unknown")
    metadata = {
        "schema_version": 2,
        "campaign": "code_cpt_research_r1",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "authenticated_username": access["username"],
        "credential_source": credential_source,
        "seed": seed,
        "packing_seed": seed,
        "split": {
            "hash": "sha256(repository_name)[:8] big-endian modulo 1000",
            "stage1_validation_excluded_buckets": list(range(10)),
            "development_buckets": list(range(10, 20)),
            "test_buckets": list(range(20, 30)),
            "training_buckets": list(range(30, 1000)),
        },
        "freshness": {
            "known_stage1_validation_repositories_excluded": len(excluded_repositories),
            "known_content_hashes_excluded": len(excluded_content_hashes),
            "old_packed_block_hashes_excluded": len(old_block_hashes),
            "new_content_hash_deduplication": True,
            "new_packed_block_deduplication": True,
            "fully_disjoint_claim": False,
            "overlap_uncertainty": (
                "Stage-1 did not preserve training repository/file manifests. Exact old packed "
                "blocks are excluded, but differently packed partial-document overlap cannot "
                "be ruled out."
            ),
            "old_train_blocks_sha256": _file_sha256(old_train_path),
        },
        "block_size": block_size,
        "eos_token_id": tokenizer.eos_token_id,
        "shuffle_buffer_records": shuffle_buffer,
        "requested_train_tokens": train_tokens,
        "actual_train_tokens": total_blocks * block_size,
        "training_blocks": total_blocks,
        "language_block_allocations": allocations,
        "language_order": language_order,
        "development_blocks_per_core_language": development_blocks,
        "development_input_tokens": development_blocks * block_size * len(CORE_LANGUAGES),
        "development_scored_target_tokens": development_blocks
        * (block_size - 1)
        * len(CORE_LANGUAGES),
        "test_blocks_per_core_language": test_blocks,
        "test_input_tokens": test_blocks * block_size * len(CORE_LANGUAGES),
        "test_scored_target_tokens": test_blocks * (block_size - 1) * len(CORE_LANGUAGES),
        "general_tokens": max(1, general_tokens // block_size) * block_size,
        "train_path": str(train_path),
        "train_language_path": str(language_path),
        "manifest_paths": {name: str(path) for name, path in manifest_paths.items()},
        "repository_block_provenance": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in sorted(provenance_paths)
        ],
        "feeder": feeder,
        "corpus_fingerprint": corpus_fingerprint(corpus_files),
        "python": platform.python_version(),
        "packages": package_versions,
    }
    (output_dir / "corpus_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def prepare_corpus(
    output_dir: Path,
    *,
    train_tokens: int = 12_000_000,
    validation_tokens_per_language: int = 16_384,
    general_tokens: int = 16_384,
    block_size: int = 2048,
    shuffle_buffer: int = 10_000,
    seed: int = 314159,
) -> dict:
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    token, credential_source = resolve_hf_token()
    access = verify_access(token)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, token=token
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token for source boundaries")
    total_blocks = train_tokens // block_size
    weights = {language: weight for language, (_, weight) in LANGUAGE_SPECS.items()}
    allocations = allocate_blocks(weights, total_blocks)
    validation_blocks = max(1, validation_tokens_per_language // block_size)
    train_paths = {}
    feeder = {}
    all_validation_records = []
    all_validation_repositories = {}
    for index, (language, (data_dir, _)) in enumerate(LANGUAGE_SPECS.items()):
        result = _prepare_language(
            language=language,
            data_dir=data_dir,
            train_blocks=allocations[language],
            validation_blocks=validation_blocks if language in CORE_LANGUAGES else 0,
            block_size=block_size,
            tokenizer=tokenizer,
            token=token,
            output_dir=output_dir,
            seed=seed + index,
            shuffle_buffer=shuffle_buffer,
        )
        path, _, stats, rejected, records, repositories = result
        train_paths[language] = path
        feeder[language] = {
            **asdict(stats),
            **stats.rates(),
            "rejected": dict(rejected),
        }
        all_validation_records.extend(records)
        all_validation_repositories.update(repositories)
    train_path, language_path, language_order = _combine_training_blocks(
        paths=train_paths,
        allocations=allocations,
        block_size=block_size,
        output_dir=output_dir,
        seed=seed,
    )
    general_path = _prepare_general_micro(
        tokenizer=tokenizer,
        token=token,
        block_size=block_size,
        blocks=max(1, general_tokens // block_size),
        output_dir=output_dir,
    )
    for path in train_paths.values():
        path.unlink()
    shutil.rmtree(output_dir / "language_blocks")
    with (output_dir / "validation_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_validation_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output_dir / "validation_repositories.json").write_text(
        json.dumps(all_validation_repositories, indent=2, sort_keys=True) + "\n"
    )
    package_versions = {}
    for name in ("datasets", "huggingface_hub", "numpy", "tokenizers", "transformers"):
        module = __import__(name)
        package_versions[name] = getattr(module, "__version__", "unknown")
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "authenticated_username": access["username"],
        "credential_source": credential_source,
        "available_target_language_dirs": access["available_target_language_dirs"],
        "seed": seed,
        "split": {
            "hash": "sha256(repository_name)[:8] big-endian modulo 1000",
            "validation_buckets": list(range(10)),
        },
        "block_size": block_size,
        "eos_token_id": tokenizer.eos_token_id,
        "shuffle_buffer_records": shuffle_buffer,
        "requested_train_tokens": train_tokens,
        "actual_train_tokens": total_blocks * block_size,
        "training_blocks": total_blocks,
        "language_block_allocations": allocations,
        "actual_language_token_percentages": {
            language: 100.0 * blocks / total_blocks for language, blocks in allocations.items()
        },
        "language_order": language_order,
        "validation_tokens_per_core_language": validation_blocks * block_size,
        "general_tokens": max(1, general_tokens // block_size) * block_size,
        "train_path": str(train_path),
        "train_language_path": str(language_path),
        "general_path": str(general_path),
        "feeder": feeder,
        "python": platform.python_version(),
        "packages": package_versions,
    }
    (output_dir / "corpus_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=12_000_000)
    parser.add_argument("--validation-tokens", type=int, default=16_384)
    parser.add_argument("--general-tokens", type=int, default=16_384)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--research-r1", action="store_true")
    parser.add_argument("--old-corpus-dir", type=Path)
    parser.add_argument("--exclusion-root", type=Path)
    args = parser.parse_args()
    if args.research_r1:
        if args.old_corpus_dir is None:
            parser.error("--research-r1 requires --old-corpus-dir")
        metadata = prepare_research_corpus(
            args.output_dir,
            old_corpus_dir=args.old_corpus_dir,
            exclusion_root=args.exclusion_root,
            train_tokens=args.train_tokens,
            development_tokens_per_language=args.validation_tokens,
            test_tokens_per_language=args.validation_tokens,
            general_tokens=args.general_tokens,
            block_size=args.block_size,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
        )
    else:
        metadata = prepare_corpus(
            args.output_dir,
            train_tokens=args.train_tokens,
            validation_tokens_per_language=args.validation_tokens,
            general_tokens=args.general_tokens,
            block_size=args.block_size,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
        )
    print(json.dumps({"status": "PASS", "corpus": metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
