#!/usr/bin/env python3
"""Append one explicitly supplied, identity-bound FIM GGUF to an ELF engine."""

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

MAGIC = b"TABCOMPLETEGGUF1"
MAX_ENGINE_BYTES = 64 * 1024 * 1024
MAX_MODEL_BYTES = 1024 * 1024 * 1024
MAX_PROFILE_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
CONTRACT_VERSION = "q25-fim-tokenizer-contract-v1"
COMPLETION_MODE = "remaining_logical_line_after_utf8_cursor"
WIRE_VERSION = "q25-fim-line-completion-v1"
CONTEXT_LAYOUT = "q25-fim-psm-bounded-v2"
TOKEN_IDS = {
    "eos": (151643, "<|endoftext|>"),
    "fim_prefix": (151659, "<|fim_prefix|>"),
    "fim_suffix": (151661, "<|fim_suffix|>"),
    "fim_middle": (151660, "<|fim_middle|>"),
}


class BuildError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BuildError(message)


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def exact_keys(value, keys, name):
    require(isinstance(value, dict) and set(value) == set(keys), f"invalid {name} fields")


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate FIM profile field")
        result[key] = value
    return result


def read_profile(path):
    require(path.is_file(), "FIM profile is not a regular file")
    require(path.stat().st_size <= MAX_PROFILE_BYTES, "FIM profile exceeds size limit")
    try:
        profile_bytes = path.read_bytes()
        require(len(profile_bytes) <= MAX_PROFILE_BYTES, "FIM profile exceeds size limit")
        wrapper = json.loads(profile_bytes, object_pairs_hook=no_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError("FIM profile is unreadable JSON") from error

    exact_keys(wrapper, ("schema", "model_sha256", "serving_profile"), "FIM profile")
    require(
        wrapper["schema"] == "tabcomplete-q25-fim-embedded-profile-v1",
        "unsupported FIM profile schema",
    )
    require(is_sha256(wrapper["model_sha256"]), "invalid FIM model digest")

    profile = wrapper["serving_profile"]
    exact_keys(profile, ("artifact_manifest_sha256", "tokenizer"), "serving profile")
    require(is_sha256(profile["artifact_manifest_sha256"]), "invalid artifact manifest digest")

    tokenizer = profile["tokenizer"]
    exact_keys(
        tokenizer,
        (
            "tokenizer_id",
            "tokenizer_revision",
            "tokenizer_sha256",
            "tokenizer_contract_sha256",
            "tokenizer_vocab_size",
            "tokenizer_vocab_ids_sha256",
            "tokenizer_vocab_ids",
            "eos_id",
            "fim_prefix_id",
            "fim_suffix_id",
            "fim_middle_id",
            "completion_mode",
            "special_tokens",
        ),
        "tokenizer profile",
    )
    for key in ("tokenizer_id", "tokenizer_revision"):
        require(
            isinstance(tokenizer[key], str)
            and bool(tokenizer[key])
            and not any(character in tokenizer[key] for character in "\n\r\t"),
            "invalid tokenizer identity",
        )
    for key in ("tokenizer_sha256", "tokenizer_contract_sha256", "tokenizer_vocab_ids_sha256"):
        require(is_sha256(tokenizer[key]), f"invalid {key}")

    ids = tokenizer["tokenizer_vocab_ids"]
    require(isinstance(ids, list) and 0 < len(ids) <= 200_000, "invalid tokenizer vocabulary IDs")
    require(
        all(type(token_id) is int and token_id >= 0 for token_id in ids)
        and all(left < right for left, right in zip(ids, ids[1:], strict=False)),
        "tokenizer vocabulary IDs must be sorted and unique",
    )
    require(
        type(tokenizer["tokenizer_vocab_size"]) is int
        and tokenizer["tokenizer_vocab_size"] == len(ids),
        "tokenizer vocabulary size mismatch",
    )
    ids_bytes = "".join(f"{token_id}\n" for token_id in ids).encode("ascii")
    require(
        hashlib.sha256(ids_bytes).hexdigest() == tokenizer["tokenizer_vocab_ids_sha256"].lower(),
        "tokenizer vocabulary digest mismatch",
    )

    for name, (expected_id, _) in TOKEN_IDS.items():
        key = "eos_id" if name == "eos" else f"{name}_id"
        require(
            type(tokenizer[key]) is int and tokenizer[key] == expected_id, "FIM token ID mismatch"
        )
        require(expected_id in ids, "FIM token ID missing from tokenizer vocabulary")
    require(tokenizer["completion_mode"] == COMPLETION_MODE, "unsupported FIM completion mode")

    specials = tokenizer["special_tokens"]
    require(
        isinstance(specials, list) and 0 < len(specials) <= 4096,
        "invalid tokenizer control-token inventory",
    )
    seen_ids = set()
    seen_spellings = set()
    normalized_specials = []
    for token in specials:
        exact_keys(token, ("id", "spelling"), "special token")
        token_id = token["id"]
        spelling = token["spelling"]
        require(type(token_id) is int and token_id >= 0, "invalid special-token ID")
        require(
            isinstance(spelling, str)
            and bool(spelling)
            and spelling.isascii()
            and not any(character in spelling for character in "\n\r\t"),
            "invalid special-token spelling",
        )
        require(token_id in ids, "special-token ID missing from tokenizer vocabulary")
        require(
            token_id not in seen_ids and spelling not in seen_spellings, "duplicate special token"
        )
        seen_ids.add(token_id)
        seen_spellings.add(spelling)
        normalized_specials.append((token_id, spelling))
    require(
        all(
            left[0] < right[0]
            for left, right in zip(normalized_specials, normalized_specials[1:], strict=False)
        ),
        "tokenizer control-token inventory must be sorted by ID",
    )
    for _name, (expected_id, expected_spelling) in TOKEN_IDS.items():
        require(
            (expected_id, expected_spelling) in normalized_specials,
            "required FIM/EOS token missing",
        )

    canonical = (
        f"{CONTRACT_VERSION}\n"
        f"{tokenizer['tokenizer_id']}\n"
        f"{tokenizer['tokenizer_revision']}\n"
        f"{tokenizer['tokenizer_sha256'].lower()}\n"
        f"{tokenizer['eos_id']}\n"
        f"{tokenizer['fim_prefix_id']}\n"
        f"{tokenizer['fim_suffix_id']}\n"
        f"{tokenizer['fim_middle_id']}\n"
        f"{tokenizer['completion_mode']}\n"
        f"{tokenizer['tokenizer_vocab_size']}\n"
        f"{tokenizer['tokenizer_vocab_ids_sha256'].lower()}\n"
    )
    for token_id, spelling in sorted(normalized_specials):
        canonical += f"{token_id}\t{spelling}\n"
    contract_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    require(
        contract_digest == tokenizer["tokenizer_contract_sha256"].lower(),
        "tokenizer contract digest mismatch",
    )
    return wrapper


def append_model(engine_path, model_path, profile_path, output_path):
    require(not output_path.exists(), "output already exists")
    require(engine_path.is_file(), "engine is not a regular file")
    require(model_path.is_file(), "model is not a regular file")
    require(4096 <= engine_path.stat().st_size <= MAX_ENGINE_BYTES, "engine size outside bounds")
    require(4 <= model_path.stat().st_size <= MAX_MODEL_BYTES, "model size outside bounds")
    with engine_path.open("rb") as engine:
        require(engine.read(4) == b"\x7fELF", "engine must be a finalized ELF executable")

    wrapper = read_profile(profile_path)
    metadata = {
        "version": 1,
        "sha256": wrapper["model_sha256"],
        "alias": "q25-fim",
        "protocol": WIRE_VERSION,
        "output_tokens": 96,
        "context_size": 2304,
        "input_tokens": 1024,
        "batch_size": 256,
        "microbatch_size": 64,
        "threads": 4,
        "cache_type": "f16",
        "context_layout": CONTEXT_LAYOUT,
        "fim_profile": wrapper["serving_profile"],
    }
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    created_temporary = False
    try:
        with (
            engine_path.open("rb") as engine,
            model_path.open("rb") as model,
            temporary.open("xb") as output,
        ):
            created_temporary = True
            shutil.copyfileobj(engine, output, 65536)
            offset = (output.tell() + 4095) // 4096 * 4096
            require(4096 <= offset <= MAX_ENGINE_BYTES, "embedded executable offset outside bounds")
            output.write(b"\0" * (offset - output.tell()))
            digest = hashlib.sha256()
            header = model.read(4)
            require(header == b"GGUF", "model is not GGUF")
            model.seek(0)
            length = 0
            while chunk := model.read(65536):
                output.write(chunk)
                digest.update(chunk)
                length += len(chunk)
            require(4 <= length <= MAX_MODEL_BYTES, "model size outside bounds")
            require(digest.hexdigest() == wrapper["model_sha256"], "model identity mismatch")
            metadata.update(offset=offset, length=length)
            payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
            require(len(payload) <= MAX_METADATA_BYTES, "embedded metadata exceeds size limit")
            output.write(payload)
            output.write(struct.pack("<I", len(payload)))
            output.write(MAGIC)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o755)
        temporary.rename(output_path)
    finally:
        if created_temporary:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        append_model(args.engine, args.model, args.profile, args.output)
    except BuildError as error:
        parser.error(str(error))
    except OSError:
        parser.error("could not read inputs or write embedded executable")


if __name__ == "__main__":
    main()
