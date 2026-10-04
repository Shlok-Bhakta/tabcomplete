from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_q25_fim import (  # noqa: E402
    Document,
    PreparationError,
    _chosen_line,
    _make_state,
    _mode_for_variant,
    _RejectedState,
    prepare_examples,
    write_corpus,
)


class CharacterTokenizer:
    eos_token_id = 103
    _markers = {
        "<|fim_prefix|>": 100,
        "<|fim_suffix|>": 101,
        "<|fim_middle|>": 102,
    }

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert not add_special_tokens
        assert return_offsets_mapping
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        position = 0
        while position < len(text):
            marker = next(
                (value for value in self._markers if text.startswith(value, position)), None
            )
            if marker is not None:
                ids.append(self._markers[marker])
                offsets.append((position, position + len(marker)))
                position += len(marker)
            else:
                ids.append(1000 + ord(text[position]))
                offsets.append((position, position + 1))
                position += 1
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return self(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)[
            "input_ids"
        ]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        marker_by_id = {token_id: marker for marker, token_id in self._markers.items()}
        return "".join(
            marker_by_id[token_id] if token_id in marker_by_id else chr(token_id - 1000)
            for token_id in token_ids
        )


class NfcTokenizer(CharacterTokenizer):
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        return super().__call__(
            unicodedata.normalize("NFC", text),
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        normalized = unicodedata.normalize("NFC", text)
        return super().encode(normalized, add_special_tokens=add_special_tokens)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        return unicodedata.normalize(
            "NFC",
            super().decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            ),
        )


def _document(split: str, language: str, number: int) -> Document:
    content = (
        "".join(f"π_{number}_{line} = {line} + 1\n" for line in range(24))
        + f"return_{number} = 7\n"
    )
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    repository = f"{split}/{language}/{number}"
    repository_hash = hashlib.sha256(repository.encode()).hexdigest()
    return Document(
        split=split,
        language=language,
        content=content,
        content_sha256=content_hash,
        repository_identity_sha256=repository_hash,
        repository_alias_sha256=(repository_hash,),
        path=f"src/module_{number}.py",
        licenses=("MIT",),
        dataset_id="bigcode/the-stack-dedup",
        dataset_revision="frozen-test-revision",
    )


def _document_with_content(split: str, language: str, number: int, content: str) -> Document:
    original = _document(split, language, number)
    return Document(
        split=original.split,
        language=original.language,
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        repository_identity_sha256=original.repository_identity_sha256,
        repository_alias_sha256=original.repository_alias_sha256,
        path=original.path,
        licenses=original.licenses,
        dataset_id=original.dataset_id,
        dataset_revision=original.dataset_revision,
    )


def _nfc_context_document() -> Document:
    prefix = "e\u0301_marker = 1\n"
    for number in range(10_000):
        content = (
            prefix + "x = 2\n" + "".join(f"value_{number}_{line} = {line}\n" for line in range(8))
        )
        document = _document_with_content("train", "python", number, content)
        spans = [
            _chosen_line(content, document.content_sha256, variant, 314159) for variant in (0, 1)
        ]
        if all(span is not None and span[0] >= len(prefix) for span in spans):
            return document
    raise AssertionError("could not construct deterministic NFC prompt fixture")


def _plan() -> dict[str, object]:
    return {
        "data": {
            "language_weights": {"python": 0.4, "rust": 0.25, "typescript": 0.2, "go": 0.15},
            "seed": 314159,
            "prefix_tokens_max": 12,
            "suffix_tokens_max": 8,
            "target_tokens_including_eos_max": 96,
            "max_total_tokens": 1024,
            "train_states_max": 16,
            "development_states_max": 16,
        },
        "budget": {"minimum_free_bytes": 1, "new_artifact_bytes_cap": 1_000_000},
        "sources": {"revision": "frozen-test-revision"},
    }


def _marker_positions(ids: list[int]) -> list[int]:
    positions = [ids.index(marker) for marker in (100, 101, 102)]
    assert positions == sorted(positions)
    assert [ids.count(marker) for marker in (100, 101, 102)] == [1, 1, 1]
    return positions


def test_preparation_is_deterministic_utf8_safe_and_response_complete(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer()
    train_docs = [
        _document("train", language, index)
        for language in ("python", "rust", "typescript", "go")
        for index in range(2)
    ]
    development_docs = [
        _document("development", language, index)
        for language in ("python", "rust", "typescript", "go")
        for index in range(1)
    ]
    train_rows, development_rows, prepared = prepare_examples(
        train_docs, development_docs, tokenizer, _plan()
    )
    second_train, second_development, second_prepared = prepare_examples(
        train_docs, development_docs, tokenizer, _plan()
    )

    assert train_rows == second_train
    assert development_rows == second_development
    assert prepared == second_prepared
    assert train_rows and development_rows
    assert len({row["id"] for row in train_rows + development_rows}) == len(
        train_rows + development_rows
    )

    documents = {doc.content_sha256: doc for doc in train_docs + development_docs}
    for row in train_rows + development_rows:
        document = documents[row["source_content_sha256"]]
        source_bytes = document.content.encode("utf-8")
        prefix_bytes = source_bytes[: row["region_start"]]
        target_bytes = source_bytes[row["region_start"] : row["region_end"]]
        suffix_bytes = source_bytes[row["region_end"] :]
        assert prefix_bytes + target_bytes + suffix_bytes == source_bytes
        assert hashlib.sha256(target_bytes).hexdigest() == row["target_sha256"]
        prefix_bytes.decode("utf-8")
        target_bytes.decode("utf-8")
        suffix_bytes.decode("utf-8")

        ids = row["input_ids"]
        markers = _marker_positions(ids[: row["prompt_tokens"]])
        assert markers[1] - markers[0] - 1 <= 12
        assert markers[2] - markers[1] - 1 <= 8
        assert row["target_tokens"] <= 96
        assert row["total_tokens"] <= 1024
        assert ids[-1] == tokenizer.eos_token_id
        assert "content" not in row and "target" not in row

    assert any(row["mode"] == "whole_logical_line" for row in train_rows + development_rows)
    assert any(
        row["mode"] == "remaining_logical_line_after_utf8_cursor"
        for row in train_rows + development_rows
    )

    inputs = {
        "parent_cpt_plan_sha256": "cpt-plan-sha",
        "pool_path": "/frozen/pool",
        "pool_hashes": {},
        "train_manifest": {"path": "/frozen/train.jsonl", "sha256": "train-sha"},
        "development_manifest": {"path": "/frozen/dev.jsonl", "sha256": "dev-sha"},
        "benchmark_exclusion_sha256": "exclusion-sha",
        "benchmark_exclusion_count": 172,
        "allowed_licenses": ["MIT"],
    }
    metadata = write_corpus(
        tmp_path / "corpus",
        train_rows,
        development_rows,
        _plan(),
        inputs,
        prepared["audit"],
        "tokenizer-sha",
        "tokenizer-revision",
        prepared["marker_ids"],
        prepared["eos_token_id"],
    )
    corpus = tmp_path / "corpus"
    on_disk_metadata = json.loads((corpus / "corpus_metadata.json").read_text())
    assert metadata["artifact_bytes"] == sum(path.stat().st_size for path in corpus.iterdir())
    assert (
        on_disk_metadata["preparation_plan_sha256"]
        == "460770a6bab77eeecd7b31eb2c8da278c5c29cea98cc872d8fbd36b22a6dc6de"
    )
    for filename in ("train.jsonl", "development.jsonl"):
        record = on_disk_metadata["files"][filename]
        assert hashlib.sha256((corpus / filename).read_bytes()).hexdigest() == record["sha256"]
        assert record["bytes"] == (corpus / filename).stat().st_size


def test_nfc_normalized_prompts_are_rejected_and_valid_states_replenish() -> None:
    tokenizer = NfcTokenizer()
    normalized_prompt_doc = _nfc_context_document()
    valid_train_doc = _document("train", "python", 777)
    valid_development_doc = _document("development", "python", 888)
    plan = _plan()
    plan["data"]["train_states_max"] = 1
    plan["data"]["development_states_max"] = 1
    plan["data"]["prefix_tokens_max"] = 256

    train_rows, development_rows, prepared = prepare_examples(
        [normalized_prompt_doc, valid_train_doc], [valid_development_doc], tokenizer, plan
    )

    assert len(train_rows) == 1
    assert train_rows[0]["source_content_sha256"] == valid_train_doc.content_sha256
    assert all(
        row["source_content_sha256"] != normalized_prompt_doc.content_sha256
        for row in train_rows + development_rows
    )
    assert prepared["audit"]["train_rejected_states"]["token_roundtrip_mismatch"] == 2
    assert development_rows


def test_nfc_normalized_targets_are_rejected() -> None:
    tokenizer = NfcTokenizer()
    document = _document_with_content("train", "python", 900, "e\u0301_value = 5\n")
    whole_line_variant = next(
        variant
        for variant in (0, 1)
        if _mode_for_variant(document.content_sha256, variant) == "whole_logical_line"
    )

    with pytest.raises(_RejectedState, match="token_roundtrip_mismatch"):
        _make_state(
            document,
            whole_line_variant,
            tokenizer,
            seed=314159,
            prefix_limit=12,
            suffix_limit=8,
            target_limit=96,
            total_limit=1024,
            eos_token_id=tokenizer.eos_token_id,
            marker_ids=(100, 101, 102),
        )


def test_writer_rejects_rows_without_target_eos(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer()
    docs = [_document("train", "python", 0), _document("development", "python", 1)]
    train_rows, development_rows, prepared = prepare_examples(
        docs[:1], docs[1:], tokenizer, _plan()
    )
    assert train_rows and development_rows
    train_rows[0]["input_ids"][-1] = 999
    inputs = {
        "parent_cpt_plan_sha256": "cpt-plan-sha",
        "pool_path": "/frozen/pool",
        "pool_hashes": {},
        "train_manifest": {},
        "development_manifest": {},
        "benchmark_exclusion_sha256": "exclusion-sha",
        "benchmark_exclusion_count": 172,
        "allowed_licenses": ["MIT"],
    }
    with pytest.raises(PreparationError, match="serialized_row_token_counts_invalid"):
        write_corpus(
            tmp_path / "invalid",
            train_rows,
            development_rows,
            _plan(),
            inputs,
            prepared["audit"],
            "tokenizer-sha",
            "tokenizer-revision",
            prepared["marker_ids"],
            prepared["eos_token_id"],
        )
