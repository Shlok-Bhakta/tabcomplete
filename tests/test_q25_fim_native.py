from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_q25_fim_native import (
    CONTEXT_LAYOUT,
    CONTEXT_POLICY_VERSION,
    EOS_TOKEN_ID,
    FIM_ACTION_POLICY,
    FIM_TOKEN_IDS,
    FIM_TOKEN_SPELLINGS,
    MODE,
    PROTOCOL,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    TOKENIZER_SHA,
    attest_process,
    context_digest,
    expected_canonical_action,
    expected_context,
    prepare_request,
    score_terminal,
    source_state,
    tokenizer_contract_sha256,
    tokenizer_vocab_ids,
    tokenizer_vocab_ids_sha256,
    validate_native_tokenizer_identity,
    validate_result_row,
    validate_terminal,
)
from measure_r2_local import NativeProvider


class TinyTokenizer:
    eos_token_id = EOS_TOKEN_ID
    all_special_ids = [EOS_TOKEN_ID]
    added_tokens_decoder = {
        151659: object(),
        151660: object(),
        151661: object(),
        151645: object(),
    }

    def __init__(self):
        self.vocab = {
            "λ": 42,
            "\n": 43,
            "x": 44,
            "<|endoftext|>": EOS_TOKEN_ID,
            "<|fim_prefix|>": 151659,
            "<|fim_middle|>": 151660,
            "<|fim_suffix|>": 151661,
            "<|im_end|>": 151645,
        }

    def get_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {
            "input_ids": self.encode(text, add_special_tokens=add_special_tokens),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids, **kwargs):
        inverse = {value: key for key, value in self.vocab.items()}
        return "".join(inverse[token_id] for token_id in token_ids)


class OffsetTokenizer:
    """Small character tokenizer for deterministic context-crop tests."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        value = {
            "input_ids": self.encode(text, add_special_tokens=add_special_tokens),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }
        return value


def _health_and_binding():
    vocab_ids = tokenizer_vocab_ids(TinyTokenizer())
    vocab_ids_sha = tokenizer_vocab_ids_sha256(vocab_ids)
    inventory = [
        {"id": token_id, "spelling": spelling}
        for token_id, spelling in FIM_TOKEN_SPELLINGS.items()
    ] + [{"id": 151645, "spelling": "<|im_end|>"}]
    inventory.sort(key=lambda item: item["id"])
    profile_tokenizer = {
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA,
        "tokenizer_contract_sha256": "",
        "tokenizer_vocab_size": len(vocab_ids),
        "tokenizer_vocab_ids_sha256": vocab_ids_sha,
        "eos_id": EOS_TOKEN_ID,
        "fim_prefix_id": FIM_TOKEN_IDS["fim_prefix"],
        "fim_middle_id": FIM_TOKEN_IDS["fim_middle"],
        "fim_suffix_id": FIM_TOKEN_IDS["fim_suffix"],
        "completion_mode": MODE,
        "special_tokens": inventory,
    }
    profile_tokenizer["tokenizer_contract_sha256"] = tokenizer_contract_sha256(
        profile_tokenizer
    )
    contract_sha = profile_tokenizer["tokenizer_contract_sha256"]
    health = {
        "model_sha256": "a" * 64,
        "context_layout": CONTEXT_LAYOUT,
        "tokenizer_sha256": TOKENIZER_SHA,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_vocab_size": len(vocab_ids),
        "tokenizer_vocab_ids_sha256": vocab_ids_sha,
        "tokenizer_contract_sha256": contract_sha,
        "fim_profile": {
            "artifact_manifest_sha256": "d" * 64,
            "tokenizer": profile_tokenizer,
        },
    }
    binding = {
        "request_id": "request-current",
        "context_hash": "e" * 64,
        "completion_mode": MODE,
    }
    return health, binding


def _terminal(health, binding, *, body_ids=None, terminal_id=EOS_TOKEN_ID, stop_type="eos"):
    profile = health["fim_profile"]
    return {
        **binding,
        "model_protocol": PROTOCOL,
        "model_sha256": health["model_sha256"],
        "context_layout": health["context_layout"],
        "tokenizer_sha256": TOKENIZER_SHA,
        "tokenizer_contract_sha256": health["tokenizer_contract_sha256"],
        "fim_token_ids": FIM_TOKEN_IDS,
        "artifact_manifest_sha256": profile["artifact_manifest_sha256"],
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_vocab_size": profile["tokenizer"]["tokenizer_vocab_size"],
        "tokenizer_vocab_ids_sha256": profile["tokenizer"]["tokenizer_vocab_ids_sha256"],
        "stop": True,
        "stop_type": stop_type,
        "terminal_token_id": terminal_id,
        "sampled_token_ids": [42, 43] if body_ids is None else body_ids,
        "tokens_predicted": 2 if body_ids is None else len(body_ids),
        "canonical_action": {"kind": "replace_line", "text": "λ"},
        "action_validation": {"policy": FIM_ACTION_POLICY, "status": "not_applicable"},
        "timings": {
            "cache_n": 0,
            "prompt_n": 1,
            "predicted_n": 2 if body_ids is None else len(body_ids),
            "prompt_ms": 1.0,
            "predicted_ms": 2.0,
            "total_ms": 3.0,
        },
    }


@pytest.mark.parametrize("ending", ["\n", "\r\n", ""])
def test_original_unicode_cursor_and_line_ending_are_preserved(ending):
    source = "# context\nvalue = λ + 🌿" + ending
    start = len(b"# context\nvalue = ")
    target = "λ + 🌿" + ending
    row = {
        "region_start": start,
        "region_end": len(source.encode()),
        "source_content_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_path": "example.py",
        "language": "python",
    }
    state = source_state(row, source, target)
    assert state["target_row"] == 1
    assert state["cursor_col"] == len(b"value = ")
    assert state["source"] == source


@pytest.mark.parametrize("failure", ["split_utf8", "wrong_source", "wrong_target", "out_of_range"])
def test_reconstruction_refuses_corrupt_ranges_and_sources(failure):
    source = "λ\n"
    row = {
        "region_start": 0,
        "region_end": len(source.encode()),
        "source_content_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_path": "example.py",
        "language": "python",
    }
    target = source
    if failure == "split_utf8":
        row["region_start"] = 1
    elif failure == "wrong_source":
        row["source_content_sha256"] = "0" * 64
    elif failure == "wrong_target":
        target = "x\n"
    else:
        row["region_end"] = 99
    with pytest.raises(ValueError):
        source_state(row, source, target)


@pytest.mark.parametrize(
    "field",
    [
        "request_id",
        "context_hash",
        "completion_mode",
        "model_sha256",
        "model_protocol",
        "tokenizer_contract_sha256",
        "stop",
    ],
)
def test_wrong_terminal_identity_is_not_quality_evidence(field):
    health, binding = _health_and_binding()
    terminal = _terminal(health, binding)
    assert validate_terminal(terminal, health, binding)
    terminal[field] = False if field == "stop" else "stale"
    assert not validate_terminal(terminal, health, binding)


@pytest.mark.parametrize(
    "field",
    [
        "artifact_manifest_sha256",
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_vocab_size",
        "tokenizer_vocab_ids_sha256",
    ],
)
def test_terminal_must_bind_complete_frozen_tokenizer_profile(field):
    health, binding = _health_and_binding()
    terminal = _terminal(health, binding)
    terminal[field] = "stale"
    assert not validate_terminal(terminal, health, binding)


def test_shared_native_provider_sends_fim_binding_and_retains_terminal(monkeypatch):
    import measure_r2_local

    calls = []
    binding = {"request_id": "request-current", "context_hash": "c" * 64, "completion_mode": MODE}
    terminal = {
        "stop": True,
        "stop_type": "eos",
        "terminal_token_id": 151643,
        "sampled_token_ids": [42],
        "timings": {"predicted_n": 1},
        **binding,
    }

    class Response:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield "data: " + json.dumps({"content": "λ", "tokens": [42], "stop": False})
            yield "data: " + json.dumps(terminal)

    @contextmanager
    def stream(*args, **kwargs):
        calls.append(kwargs["json"])
        yield Response()

    monkeypatch.setattr(measure_r2_local.httpx, "stream", stream)
    provider = NativeProvider("http://127.0.0.1:19104", "q25-fim")
    provider.editor_request_binding = binding
    result = provider.generate_detailed("<|fim_prefix|>value=<|fim_suffix|><|fim_middle|>", 96)
    assert result.text == "λ"
    assert provider.last["terminal_event"] == terminal
    assert all(calls[0][key] == value for key, value in binding.items())
    provider.editor_request_binding = None
    provider.generate_detailed("legacy causal prompt", 32)
    assert not any(key in calls[1] for key in binding)
    provider.editor_request_binding = {"request_id": "missing context identity"}
    with pytest.raises(ValueError, match="binding"):
        provider.generate_detailed("invalid request", 96)
    assert len(calls) == 2


@pytest.mark.parametrize("ending", ["\n", "\r\n", ""])
def test_native_action_retains_literal_escapes_spaces_and_unicode(ending):
    state = {"source": "λ = previous" + ending, "target_row": 0, "cursor_col": len("λ = ".encode())}
    raw = 'print("\\n")  ' + ending
    assert expected_canonical_action(state, raw) == {
        "kind": "replace_line",
        "text": 'λ = print("\\n")  ',
    }
    assert expected_canonical_action(state, raw + "\n") is None


def test_unicode_line_separator_is_a_source_character_not_a_buffer_line():
    state = {"source": 'value = "a\u2028b"\n', "target_row": 0, "cursor_col": 8}
    assert expected_canonical_action(state, '"c\u2028d"\n') == {
        "kind": "replace_line",
        "text": 'value = "c\u2028d"',
    }


def test_expected_context_uses_psm_cursor_to_line_end_and_exact_line_endings():
    case = {
        "state": {"source": "α = old\r\nnext()\r\n", "target_row": 0, "cursor_col": 5}
    }
    expected = expected_context(case, OffsetTokenizer())
    assert expected["prompt"] == "<|fim_prefix|>α = <|fim_suffix|>next()\r\n<|fim_middle|>"
    assert expected["line_ending"] == "CRLF"
    assert expected["model_hole_range"] == {
        "start_byte": len("α = ".encode()),
        "end_byte": len("α = old\r\n".encode()),
        "end_exclusive": True,
    }
    assert expected["apply_range"]["start_byte"] == 0
    assert expected["apply_range"]["end_byte"] == len("α = old".encode())


def test_expected_context_matches_frozen_640_256_training_crop_and_digest():
    from evaluate_q25_fim_native import context_digest

    tokenizer = OffsetTokenizer()
    source = "p" * 700 + "old\n" + "s" * 300 + "\n"
    case = {
        "prompt": (
            "<|fim_prefix|>" + "p" * 640 + "<|fim_suffix|>" + "s" * 256 + "<|fim_middle|>"
        ),
        "state": {"source": source, "target_row": 0, "cursor_col": 700},
    }
    expected = expected_context(case, tokenizer)
    assert expected["prefix_range"] == {
        "start_byte": 60,
        "end_byte": 700,
        "end_exclusive": True,
    }
    assert expected["prefix_token_count"] == 640
    assert expected["suffix_range"] == {
        "start_byte": len(("p" * 700 + "old\n").encode()),
        "end_byte": len(("p" * 700 + "old\n" + "s" * 256).encode()),
        "end_exclusive": True,
    }
    assert expected["suffix_token_count"] == 256
    assert expected["prompt"] == case["prompt"]

    request_id = "request-123"
    contract = "b" * 64
    canonical = (
        "q25-fim-context-v2\n"
        + request_id
        + "\n"
        + hashlib.sha256(source.encode()).hexdigest()
        + "\n0\n700\n"
        + hashlib.sha256(case["prompt"].encode()).hexdigest()
        + "\n60\n700\n640\n704\n960\n256\n"
        + CONTEXT_POLICY_VERSION
        + "\n"
        + CONTEXT_LAYOUT
        + "\n"
        + contract
        + "\n"
    )
    assert context_digest(request_id, case, contract, tokenizer) == hashlib.sha256(
        canonical.encode()
    ).hexdigest()


def test_tokenizer_check_precedes_one_shot_context_preparation():
    from evaluate_q25_fim_native import context_digest

    health = {"model_sha256": "a" * 64, "tokenizer_contract_sha256": "b" * 64}
    case = {
        "prompt": "<|fim_prefix|>x = <|fim_suffix|><|fim_middle|>",
        "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7],
        "repository": "c" * 64,
        "state": {"source": "x = old\n", "target_row": 0, "cursor_col": 4},
    }
    calls = []

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            pass

        def json(self):
            return self.value

    class Client:
        def post(self, url, json):
            calls.append(url.rsplit("/", 1)[-1])
            if url.endswith("/tokenize"):
                return Response({"tokens": case["prompt_token_ids"]})
            context = expected_context(case, OffsetTokenizer())
            request_id = json["request_id"]
            return Response(
                {
                    **context,
                    "prompt": case["prompt"],
                    "request_id": request_id,
                    "completion_mode": MODE,
                    "model_protocol": PROTOCOL,
                    "context_policy_version": CONTEXT_POLICY_VERSION,
                    "context_layout": CONTEXT_LAYOUT,
                    "tokenizer_sha256": TOKENIZER_SHA,
                    "tokenizer_contract_sha256": health["tokenizer_contract_sha256"],
                    "prompt_tokens": len(case["prompt_token_ids"]),
                    "context_hash": context_digest(
                        request_id, case, "b" * 64, OffsetTokenizer()
                    ),
                    "model_identity": {
                        "model_sha256": health["model_sha256"],
                        "model_protocol": PROTOCOL,
                        "tokenizer_sha256": TOKENIZER_SHA,
                        "tokenizer_contract_sha256": health["tokenizer_contract_sha256"],
                    },
                }
            )

    prepared = prepare_request(
        Client(), "http://127.0.0.1:19104", case, "request-1", health, OffsetTokenizer()
    )
    assert calls == ["tokenize", "context"]
    assert prepared["prompt"] == case["prompt"]


def test_process_attestation_requires_pid_to_own_listening_socket(tmp_path):
    proc = tmp_path / "proc"
    process = proc / "123"
    (process / "fd").mkdir(parents=True)
    binary = tmp_path / "native-service"
    binary.write_bytes(b"known native executable")
    (process / "exe").symlink_to(binary)
    (process / "net").mkdir()
    (process / "net" / "tcp").write_text(
        "header\n0: 0100007F:4AA0 00000000:0000 0A 0:0 00:00000000 00000000 0 0 12345\n"
    )
    (process / "net" / "tcp6").write_text("header\n")
    (process / "fd" / "4").symlink_to("socket:[12345]")
    fields = ["S"] + ["0"] * 18 + ["77"]
    (process / "stat").write_text("123 (native service) " + " ".join(fields))
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    identity = attest_process(123, binary_sha, 19104, proc_root=proc)
    assert identity["process_start_ticks"] == 77
    assert identity["listening_socket_inodes"] == ["12345"]
    with pytest.raises(ValueError, match="listening port"):
        attest_process(123, binary_sha, 19105, proc_root=proc)


def test_native_tokenizer_endpoint_is_bound_to_complete_vocab_identity():
    tokenizer = TinyTokenizer()
    ids = tokenizer_vocab_ids(tokenizer)
    health, _ = _health_and_binding()
    profile = health["fim_profile"]["tokenizer"]
    profile["tokenizer_vocab_size"] = len(ids)
    profile["tokenizer_vocab_ids_sha256"] = tokenizer_vocab_ids_sha256(ids)
    conversion = {
        "source_export_manifest_sha256": "d" * 64,
        "tokenizer": {
            "model_id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "sha256": TOKENIZER_SHA,
            "eos_token_id": EOS_TOKEN_ID,
            "fim_marker_ids": {
                "fim_prefix": FIM_TOKEN_IDS["fim_prefix"],
                "fim_middle": FIM_TOKEN_IDS["fim_middle"],
                "fim_suffix": FIM_TOKEN_IDS["fim_suffix"],
            },
        },
    }
    payload = {
        "model_sha256": health["model_sha256"],
        "artifact_manifest_sha256": "d" * 64,
        "tokenizer": profile,
        "tokenizer_vocab_ids": ids,
    }
    expected_specials = deepcopy(profile["special_tokens"])
    assert validate_native_tokenizer_identity(
        conversion, health, payload, tokenizer, expected_specials
    )["tokenizer_vocab_ids_sha256"] == profile["tokenizer_vocab_ids_sha256"]
    payload["tokenizer_vocab_ids"] = ids[:-1] + [ids[-1] + 1]
    with pytest.raises(ValueError, match="vocabulary IDs"):
        validate_native_tokenizer_identity(
            conversion, health, payload, tokenizer, expected_specials
        )

    payload["tokenizer_vocab_ids"] = ids
    with pytest.raises(ValueError, match="special-token inventory"):
        validate_native_tokenizer_identity(
            conversion, health, payload, tokenizer, expected_specials[:-1]
        )


def _valid_result_row():
    tokenizer = TinyTokenizer()
    health, _ = _health_and_binding()
    source = "x\n"
    case = {
        "case_id": "fim-development-4096",
        "repository": "f" * 64,
        "context_sha256": hashlib.sha256(b"prompt").hexdigest(),
        "prompt": "<|fim_prefix|><|fim_suffix|><|fim_middle|>",
        "prompt_token_ids": [44],
        "target": "λ\n",
        "state": {"source": source, "target_row": 0, "cursor_col": 0},
    }
    request_id = "request-current"
    binding = {
        "request_id": request_id,
        "context_hash": context_digest(
            request_id, case, health["tokenizer_contract_sha256"], OffsetTokenizer()
        ),
        "completion_mode": MODE,
    }
    terminal = _terminal(health, binding)
    process = {
        "pid": 123,
        "binary_sha256": "9" * 64,
        "process_start_ticks": 77,
        "listening_port": 19104,
        "listening_socket_inodes": ["12345"],
    }
    scored = score_terminal(case, terminal, "λ\n", tokenizer)
    row = {
        "case_id": case["case_id"],
        "repository": case["repository"],
        "context_sha256": case["context_sha256"],
        "raw_response": "λ\n",
        "completion_sha256": hashlib.sha256("λ\n".encode()).hexdigest(),
        "exact": True,
        "exact_and_terminated": True,
        "ended_by_eos": True,
        "terminated": True,
        "finish_reason": "eos",
        "native_stop_type": "eos",
        "quality_error_code": None,
        "input_tokens": 1,
        "output_tokens_including_terminal": 3,
        "latency_seconds": 0.1,
        "first_token_seconds": 0.05,
        "server_timings": terminal["timings"],
        "canonical_action": scored["canonical_action"],
        "post_request_process_memory": {"VmRSS_bytes": 4096},
        "request_id": request_id,
        "campaign_id": "campaign-abc",
        "run_id": "run-abc",
        "run_attempt_id": "attempt-abc",
        "case_attempt_id": "case-attempt-abc",
        "process_identity": process,
        "terminal_event": terminal,
        **scored["evidence"],
    }
    return row, case, tokenizer, health, process


def test_resume_row_redecodes_token_ids_and_rejects_nonfinite_metrics():
    row, case, tokenizer, health, process = _valid_result_row()
    validate_result_row(row, case, tokenizer, health, process)
    stale = deepcopy(row)
    stale["terminal_event"]["sampled_token_ids"] = [44, 43]
    with pytest.raises(ValueError):
        validate_result_row(stale, case, tokenizer, health, process)
    nonfinite = deepcopy(row)
    nonfinite["latency_seconds"] = float("nan")
    with pytest.raises(ValueError, match="latency"):
        validate_result_row(nonfinite, case, tokenizer, health, process)


def test_bound_non_eos_terminal_is_a_quality_outcome_not_a_pipeline_error():
    tokenizer = TinyTokenizer()
    health, binding = _health_and_binding()
    case = {"target": "λ\n", "state": {"source": "x\n", "target_row": 0, "cursor_col": 0}}
    terminal = _terminal(
        health,
        binding,
        body_ids=[42],
        terminal_id=151645,
        stop_type="control",
    )
    terminal["canonical_action"] = None
    terminal["action_validation"] = {
        "policy": FIM_ACTION_POLICY,
        "status": "invalid",
        "code": "fim_terminal_not_eos",
    }
    scored = score_terminal(case, terminal, "λ", tokenizer)
    assert not scored["terminated"]
    assert scored["quality_error_code"] == "fim_terminal_not_eos"


def test_bound_out_of_vocabulary_token_is_recorded_without_unpinned_decode():
    tokenizer = TinyTokenizer()
    health, binding = _health_and_binding()
    case = {"target": "λ\n", "state": {"source": "x\n", "target_row": 0, "cursor_col": 0}}
    terminal = _terminal(health, binding, body_ids=[999_999])
    terminal["canonical_action"] = None
    terminal["action_validation"] = {
        "policy": FIM_ACTION_POLICY,
        "status": "invalid",
        "code": "fim_token_id_outside_vocabulary",
    }

    scored = score_terminal(case, terminal, "opaque-unmapped-token\n", tokenizer)
    assert scored["decoded"] is None
    assert scored["quality_error_code"] == "fim_token_id_outside_vocabulary"
    assert scored["terminated"] is False
    assert scored["canonical_action"] is None
    assert scored["evidence"]["unknown_token_ids"] == [999_999]
