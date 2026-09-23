"""Frozen public-source pool, matched packed corpora, and causal-line controls for R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tinycomplete.eval.code_benchmark import _parse

from .data import (
    BlockPacker,
    RepoSplit,
    SourceFilter,
    allocate_blocks,
    repository_identity,
    repository_path,
)
from .prepare import (
    CORE_LANGUAGES,
    DATASET_ID,
    DATASET_REVISION,
    LANGUAGE_SPECS,
    MODEL_ID,
    MODEL_REVISION,
    _combine_training_blocks,
    _file_sha256,
    _load_known_exclusions,
    _metadata_reject,
    corpus_fingerprint,
    hash_packed_blocks,
)

SEED = 928173
SIZE = 2048
TOKENS = 5_013_504
MARKERS = re.compile(
    r"generated\s+(?:code|file)|do\s+not\s+edit|automatically\s+generated|sourceMappingURL", re.I
)
PATHS = re.compile(
    r"(?:^|/)(?:vendor|third_party|node_modules|dist|generated|bindings_generated)(?:/|$)|\.min\.",
    re.I,
)


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def shingles(text):
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    n = min(5, len(lines))
    return {
        hashlib.blake2b("\n".join(lines[i : i + n]).encode(), digest_size=8).digest()
        for i in range(max(1, len(lines) - n + 1))
    }


class NearDuplicates:
    """32 fixed minhash components, 8 LSH bands, bounded exact-Jaccard checking."""

    def __init__(self):
        self.bands = defaultdict(list)
        self.documents = []
        rng = np.random.default_rng(SEED)
        self.a = rng.integers(1, 2**63, size=32, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, size=32, dtype=np.uint64)

    def check_and_add(self, text):
        grams = shingles(text)
        hashes = np.array([int.from_bytes(g, "little") for g in grams], dtype=np.uint64)
        signature = ((hashes[:, None] * self.a[None, :]) + self.b[None, :]).min(axis=0)
        keys = [(i, signature[i * 4 : (i + 1) * 4].tobytes()) for i in range(8)]
        candidates = sorted({j for key in keys for j in self.bands[key]})[:128]
        for j in candidates:
            old = self.documents[j]
            if len(old & grams) / len(old | grams) >= 0.85:
                return True
        index = len(self.documents)
        self.documents.append(grams)
        for key in keys:
            self.bands[key].append(index)
        return False


def filtered_reason(record, near, seen, totals, target):
    text = record["content"]
    if PATHS.search(record["path"]):
        return "additional_path"
    if MARKERS.search(text):
        return "generated_content_marker"
    if record["content_sha256"] in seen:
        return "exact_duplicate"
    seen.add(record["content_sha256"])
    if near.check_and_add(text):
        return "near_duplicate"
    if record["parse_status"] == "fail":
        return "syntax_invalid"
    amount = len(record["tokens"]) + 1
    if totals[record["repository"]] + amount > int(target * 0.02):
        return "repository_token_cap"
    totals[record["repository"]] += amount
    return None


def line_candidate(record):
    """One deterministic target per held-out file, before observing any model output."""
    if record["parse_status"] != "pass":
        return None
    text = record["content"]
    # A line inside a multi-line comment need not start with a comment marker.
    # Mask parser-identified comments before applying the preregistered rule.
    from tree_sitter_language_pack import get_parser

    raw = text.encode()
    visible = bytearray(raw)
    stack = [get_parser(record["language"]).parse(raw).root_node]
    while stack:
        node = stack.pop()
        if "comment" in node.type:
            for position in range(node.start_byte, node.end_byte):
                if visible[position] not in (10, 13):
                    visible[position] = 32
        else:
            stack.extend(node.children)
    code_lines = visible.decode().splitlines()
    lines = text.splitlines(keepends=True)
    candidates = []
    offset = 0
    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        stripped = body.lstrip()
        if (
            len(stripped) >= 12
            and sum(c.isalnum() for c in stripped) >= 6
            and sum(c.isalnum() for c in code_lines[index]) >= 6
            and not stripped.startswith(("#", "//", "/*", "*", "<!--"))
        ):
            key = hashlib.sha256(f"{SEED}:{record['content_sha256']}:{index}".encode()).hexdigest()
            candidates.append((key, index, offset, body))
        offset += len(line)
    if not candidates:
        return None
    _, line_number, start, body = min(candidates)
    indent = len(body) - len(body.lstrip())
    split = indent + max(1, (len(body) - indent) // 2)
    cursor = start + split
    reference = body[split:]
    source_before = text[:cursor]
    source_after = text[start + len(body) :]
    assert source_before + reference + source_after == text
    return {
        "id": record["language"] + "/" + record["content_sha256"][:20],
        "language": record["language"],
        "repository": record["repository"],
        "repository_aliases": record["repository_aliases"],
        "path": record["path"],
        "source_sha256": record["content_sha256"],
        "line_number": line_number + 1,
        "prompt": source_before[-8192:],
        "reference": reference,
        "source_before": source_before,
        "source_after": source_after,
        "gold_parse": "pass",
        "protocol": "causal_line_v1",
        "selection_key": hashlib.sha256(
            f"{SEED}:{record['repository']}:{record['path']}:{record['content_sha256']}".encode()
        ).hexdigest(),
    }


def exclusions(old, previous, benchmarks):
    repos, hashes = _load_known_exclusions(old, benchmarks)
    # Metadata identities only. Never load/evaluate sealed test token arrays.
    for name in ("train_manifest.jsonl", "development_manifest.jsonl", "test_manifest.jsonl"):
        for line in (previous / name).read_text().splitlines():
            row = json.loads(line)
            hashes.add(row["content_sha256"])
            repos.add(row["repository"])
    blocks = hash_packed_blocks(old / "train_blocks.npy") | hash_packed_blocks(
        previous / "train_blocks.npy"
    )
    return repos, hashes, blocks


def prepare_language(
    language, target, tokenizer, token, root, excluded_repos, excluded_hashes, old_blocks
):
    from datasets import load_dataset

    pool_path = root / "pool" / f"{language}.jsonl"
    metadata_path = root / "pool" / f"{language}.json"
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved["sha256"] != _file_sha256(pool_path):
            raise ValueError("pool checksum mismatch")
        if saved["source_revision"] != DATASET_REVISION:
            raise ValueError("pool source revision mismatch")
        records = [json.loads(line) for line in pool_path.read_text().splitlines()]
        for record in records:
            if (
                set(record["repository_aliases"]) & excluded_repos
                or record["content_sha256"] in excluded_hashes
                or hashlib.sha256(record["content"].encode()).hexdigest()
                != record["content_sha256"]
            ):
                raise ValueError("cached pool identity/exclusion mismatch")
        counts = Counter(saved["raw_rejections"])
        raw_count = saved["raw_files"]
    else:
        stream = load_dataset(
            DATASET_ID,
            data_dir=LANGUAGE_SPECS[language][0],
            split="train",
            streaming=True,
            token=token,
            revision=DATASET_REVISION,
        ).shuffle(seed=SEED + list(LANGUAGE_SPECS).index(language), buffer_size=10000)
        counts = Counter()
        records = []
        raw_count = 0
        eligible = 0
        line_count = 0
        near = NearDuplicates()
        seen = set()
        repo_tokens = Counter()
        splitter = RepoSplit()
        for row in stream:
            raw_count += 1
            text = row.get("content")
            if not isinstance(text, str):
                counts["missing_content"] += 1
                continue
            reason = _metadata_reject(row)
            if reason:
                counts[reason] += 1
                continue
            try:
                repo = repository_identity(row)
                path = repository_path(row)
            except ValueError:
                counts["missing_repository"] += 1
                continue
            aliases = sorted(
                {
                    row.get(prefix + "_repo_name")
                    for prefix in ("max_stars", "max_forks", "max_issues")
                    if row.get(prefix + "_repo_name")
                }
            )
            if any(alias in excluded_repos for alias in aliases):
                counts["known_repository"] += 1
                continue
            bucket = splitter.bucket(repo)
            if bucket < 30:
                counts["reserved_repository_bucket"] += 1
                continue
            checked = SourceFilter().check(text, path)
            if not checked.accepted:
                counts[str(checked.reason)] += 1
                continue
            hashed = hashlib.sha256(text.encode()).hexdigest()
            if hashed in excluded_hashes:
                counts["known_consumed_content"] += 1
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            parsed = _parse(text, language).status
            record = {
                "language": language,
                "repository": repo,
                "repository_aliases": aliases,
                "path": path,
                "content": text,
                "content_sha256": hashed,
                "hexsha": row.get("hexsha"),
                "licenses": row.get("max_stars_repo_licenses")
                or row.get("max_forks_repo_licenses")
                or row.get("max_issues_repo_licenses"),
                "bucket": bucket,
                "tokens": ids,
                "parse_status": parsed,
            }
            records.append(record)
            if bucket < 40:
                if language in CORE_LANGUAGES and line_candidate(record):
                    line_count += 1
            else:
                reason = filtered_reason(record, near, seen, repo_tokens, target)
                if reason is None:
                    eligible += len(ids) + 1
            # Eligibility depends on fixed data rules, never model outcomes.
            if eligible >= target + SIZE * 4 and (
                line_count >= 40 or language not in CORE_LANGUAGES
            ):
                break
            if raw_count % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "language": language,
                            "scanned": raw_count,
                            "filtered_tokens": eligible,
                            "line_candidates": line_count,
                        }
                    ),
                    flush=True,
                )
            if raw_count >= 200000:
                raise RuntimeError("bounded pool insufficient; expand before training")
        jsonl(pool_path, records)
        save(
            metadata_path,
            {
                "sha256": _file_sha256(pool_path),
                "raw_files": raw_count,
                "raw_rejections": dict(counts),
                "source_revision": DATASET_REVISION,
                "frozen_before_training": True,
            },
        )
    audit = {
        "language": language,
        "candidate_files": raw_count,
        "common_rejections": dict(counts),
        "pool_sha256": _file_sha256(pool_path),
        "arms": {},
    }
    line_rows = (
        sorted(
            {
                x["id"]: x for r in records if 30 <= r["bucket"] < 40 and (x := line_candidate(r))
            }.values(),
            key=lambda r: r["selection_key"],
        )[:20]
        if language in CORE_LANGUAGES
        else []
    )
    for arm in ("R2_STANDARD", "R2_FILTERED"):
        packer = BlockPacker(SIZE, tokenizer.eos_token_id)
        near = NearDuplicates()
        seen = set()
        repo_tokens = Counter()
        reject = Counter()
        accepted = []
        blocks = []
        emitted = set()
        credited = Counter()
        needed = target // SIZE
        selected_tokens = 0
        for record in records:
            if record["bucket"] < 40:
                continue
            if len(blocks) >= needed:
                break
            if arm == "R2_FILTERED":
                reason = filtered_reason(record, near, seen, repo_tokens, target)
                if reason:
                    reject[reason] += 1
                    continue
            available = min(
                len(record["tokens"]) + int(packer.documents > 0), target - selected_tokens
            )
            selected_tokens += available
            credited[record["repository"]] += available
            accepted.append({k: v for k, v in record.items() if k not in ("content", "tokens")})
            accepted[-1]["source_tokens"] = len(record["tokens"])
            accepted[-1]["selected_tokens"] = available
            for block in packer.add_document(record["tokens"]):
                hashed = hashlib.sha256(np.asarray(block, dtype=np.uint32).tobytes()).hexdigest()
                if hashed in old_blocks or hashed in emitted:
                    # Rather than silently disturbing repository accounting, reject this pool.
                    raise RuntimeError(
                        "forbidden packed-block overlap; rebuild with next frozen pool revision"
                    )
                blocks.append(block)
                emitted.add(hashed)
                if len(blocks) == needed:
                    break
        if len(blocks) != needed:
            raise RuntimeError("insufficient frozen pool for " + arm + "/" + language)
        directory = root / arm / "language_blocks"
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / (language + ".npy"), np.asarray(blocks, dtype=np.uint32))
        jsonl(root / arm / "manifests" / (language + ".jsonl"), accepted)
        test_files = sum(
            bool(re.search(r"(?:^|[/_.-])tests?(?:[/_.-]|$)", r["path"], re.I)) for r in accepted
        )
        sizes = [r["source_tokens"] for r in accepted]
        concentration = max(credited.values()) / target
        if arm == "R2_FILTERED" and concentration > 0.02000001:
            raise AssertionError("repository cap violated")
        audit["arms"][arm] = {
            "files": len(accepted),
            "packed_tokens": target,
            "source_tokens": sum(sizes),
            "rejections": dict(reject),
            "repository_count": len(credited),
            "largest_repository_fraction": concentration,
            "parser": dict(Counter(r["parse_status"] for r in accepted)),
            "test_file_share": test_files / len(accepted),
            "source_token_quantiles": np.quantile(sizes, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "generated_marker_files": sum(
                bool(MARKERS.search(r["content"]))
                for r in records
                if r["content_sha256"] in {a["content_sha256"] for a in accepted}
            ),
        }
        sample = sorted(
            accepted,
            key=lambda r: hashlib.sha256((str(SEED) + r["content_sha256"]).encode()).hexdigest(),
        )[:10]
        save(root / "audit" / f"{language}-{arm}-sample.json", sample)
    save(root / "audit" / (language + ".json"), audit)
    jsonl(root / "line_candidates" / (language + ".jsonl"), line_rows)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-corpus", type=Path, required=True)
    parser.add_argument("--previous-corpus", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--language", choices=list(LANGUAGE_SPECS))
    args = parser.parse_args()
    if shutil.disk_usage(args.output.parent).free < 20 * 2**30:
        raise RuntimeError("20 GiB free-space reserve required")
    from transformers import AutoTokenizer

    token = args.token_file.read_text().strip()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    repo_root = Path(__file__).resolve().parents[3]
    repos, hashes, old_blocks = exclusions(
        args.old_corpus, args.previous_corpus, repo_root / "data/benchmarks"
    )
    allocations = allocate_blocks({k: v[1] for k, v in LANGUAGE_SPECS.items()}, TOKENS // SIZE)
    for language in [args.language] if args.language else LANGUAGE_SPECS:
        prepare_language(
            language,
            allocations[language] * SIZE,
            tokenizer,
            token,
            args.output,
            repos,
            hashes,
            old_blocks,
        )
    if args.language:
        return
    for arm in ("R2_STANDARD", "R2_FILTERED"):
        destination = args.output / arm
        paths = {lang: destination / "language_blocks" / (lang + ".npy") for lang in LANGUAGE_SPECS}
        train, labels, order = _combine_training_blocks(
            paths=paths, allocations=allocations, block_size=SIZE, output_dir=destination, seed=SEED
        )
        shutil.copytree(args.previous_corpus / "micro", destination / "micro", dirs_exist_ok=True)
        manifest_files = list((destination / "manifests").glob("*.jsonl"))
        save(
            destination / "corpus_metadata.json",
            {
                "campaign": "model_data_r2",
                "policy": arm,
                "actual_train_tokens": TOKENS,
                "training_blocks": TOKENS // SIZE,
                "block_size": SIZE,
                "language_order": order,
                "language_block_allocations": allocations,
                "corpus_fingerprint": corpus_fingerprint([train, labels, *manifest_files]),
                "source_revision": DATASET_REVISION,
                "tokenizer_revision": MODEL_REVISION,
                "packing_seed": SEED,
                "packing": "EOS-separated files; not repository-coherent",
                "prior_content_hashes_excluded": len(hashes),
                "prior_repositories_excluded": len(repos),
                "prior_block_hashes_excluded": len(old_blocks),
                "sealed_test_evaluated": False,
                "overlap_uncertainty": (
                    "incomplete Stage-1 provenance and unknown original pretraining"
                ),
            },
        )
    rows = [
        json.loads(line)
        for language in CORE_LANGUAGES
        for line in (args.output / "line_candidates" / (language + ".jsonl"))
        .read_text()
        .splitlines()
    ]
    if len(rows) != 180:
        raise RuntimeError("line diagnostic needs exactly 180 controls")
    jsonl(args.output / "causal_line_v1.jsonl", rows)
    save(
        args.output / "completion.json",
        {
            "status": "complete",
            "source": DATASET_ID,
            "revision": DATASET_REVISION,
            "line_cases": len(rows),
            "line_sha256": _file_sha256(args.output / "causal_line_v1.jsonl"),
            "corpora": {
                arm: json.loads((args.output / arm / "corpus_metadata.json").read_text())
                for arm in ("R2_STANDARD", "R2_FILTERED")
            },
        },
    )


if __name__ == "__main__":
    main()
