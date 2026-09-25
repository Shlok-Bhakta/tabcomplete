from tinycomplete.eval.compact_next_edit import decode, encode
from tinycomplete.eval.next_edit_protocol import NextEditAction


def test_roundtrip_exact_replacement_bytes() -> None:
    for value in ("", " ", "\n", " x\n\n", "λ = '🦀'\n", "```python\nx\n```"):
        action = NextEditAction(action="replace", text=value)
        parsed = decode(encode(action), finish_reason="eos", generated_tokens=4)
        assert parsed.status == "ok"
        assert parsed.action == action


def test_no_edit_and_bad_headers() -> None:
    assert decode("N\n", finish_reason="eos").action == NextEditAction(action="no_edit")
    for raw in ("N", "N\nx", "R", "r\nx", "x\n", "", " R\nx"):
        assert decode(raw, finish_reason="eos").status == "malformed"


def test_cutoff_is_never_termination() -> None:
    for reason in ("length", "max_tokens", None):
        assert decode("R\nhello", finish_reason=reason, generated_tokens=96).status == "incomplete"
