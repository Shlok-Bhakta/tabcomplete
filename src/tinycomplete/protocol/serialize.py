"""Compact textual serialization of editor state for model prompts.

Design goals: deterministic, unambiguous, escapable, round-trippable,
reasonably compact. Never feed verbose JSON per keystroke; this line-based
tag format keeps structural overhead to a few bytes per event.

Grammar (line-based; tags occupy whole lines except where noted):

    <F path>              open file section (path = remainder of line)
    <S>                   snapshot: full source-of-truth content follows
    content...
    </S>
    <C offset>            cursor byte offset in this file
    <D start end severity>
    message...
    </D>                  one diagnostic entry
    </F>                  end of file section
    <X source path>       retrieved context block (source e.g. tree-sitter)
    content...
    </X>
    <A completion-id status>   completion status: accepted|partial|rejected
    <P path offset>       PREDICT request — always the last line

Escaping: inside <S>/<D>/<X> blocks, any content line starting with '<' or
'\\' is prefixed with one extra '\\'. The parser strips exactly one leading
'\\' from lines that start with '\\'. All other lines pass through unchanged.
Paths may not contain newlines (rejected loudly instead of misparsed).
"""

from __future__ import annotations

from .events import RetrievedContext
from .replay import EditorState

__all__ = [
    "serialize_state",
    "parse_serialized",
    "count_chars",
    "qwen_token_count",
    "PREDICT_TAG",
]

PREDICT_TAG = "<P"


def _escape(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        if line.startswith("<") or line.startswith("\\"):
            out.append("\\" + line)
        else:
            out.append(line)
    return out


def _unescape(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        if line.startswith("\\"):
            out.append(line[1:])
        else:
            out.append(line)
    return out


def _check_path(path: str) -> str:
    if "\n" in path or "\r" in path:
        raise ValueError(f"path contains newline: {path!r}")
    if not path:
        raise ValueError("empty path")
    return path


def serialize_state(state: EditorState, predict_path: str = "", predict_offset: int = 0) -> str:
    """Serialize editor state. Ends with a <P> line (empty path = no request)."""
    lines: list[str] = []
    for path in state.open_files:
        _check_path(path)
        lines.append(f"<F {path}>")
        lines.append("<S>")
        lines.extend(_escape(state.files.get(path, "").split("\n")))
        lines.append("</S>")
        if path in state.cursors:
            lines.append(f"<C {state.cursors[path]}>")
        for entry in state.diagnostics.get(path, ()):
            lines.append(f"<D {entry.start} {entry.end} {entry.severity}>")
            lines.extend(_escape(entry.message.split("\n")))
            lines.append("</D>")
        lines.append("</F>")
    for ctx in state.retrieved:
        _check_path(ctx.path or "ctx")
        lines.append(f"<X {ctx.source} {ctx.path}>")
        lines.extend(_escape(ctx.content.split("\n")))
        lines.append("</X>")
    for completion_id in sorted(state.completions):
        lines.append(f"<A {completion_id} {state.completions[completion_id]}>")
    if state.last_predict is not None:
        lines.append(f"<P {state.last_predict.path} {state.last_predict.offset}>")
    elif predict_path:
        lines.append(f"<P {predict_path} {predict_offset}>")
    else:
        lines.append("<P>")
    return "\n".join(lines) + "\n"


def parse_serialized(text: str) -> EditorState:
    """Inverse of serialize_state (completions + predict restored)."""
    state = EditorState()
    raw = text.split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    i = 0
    current: str | None = None
    n = len(raw)

    from .events import DiagnosticEntry, Predict

    def take_block(end_tag: str) -> list[str]:
        nonlocal i
        buf = []
        while i < n and raw[i] != end_tag:
            buf.append(raw[i])
            i += 1
        if i >= n:
            raise ValueError(f"unterminated block, missing {end_tag}")
        i += 1  # consume end tag
        return _unescape(buf)

    def payload(line: str, prefix: str) -> str:
        assert line.startswith(prefix) and line.endswith(">"), f"malformed tag: {line!r}"
        return line[len(prefix) : -1]

    while i < n:
        line = raw[i]
        i += 1
        if line.startswith("<F "):
            current = _check_path(payload(line, "<F "))
            state.files[current] = ""
            if current not in state.open_files:
                state.open_files.append(current)
        elif line == "</F>":
            current = None
        elif line == "<S>":
            if current is None:
                raise ValueError("<S> outside <F>")
            state.files[current] = "\n".join(take_block("</S>"))
        elif line.startswith("<C "):
            if current is None:
                raise ValueError("<C> outside <F>")
            state.cursors[current] = int(payload(line, "<C "))
        elif line.startswith("<D "):
            if current is None:
                raise ValueError("<D> outside <F>")
            s, e, sev = payload(line, "<D ").split(" ", 2)
            msg = "\n".join(take_block("</D>"))
            entries = state.diagnostics.get(current, ())
            state.diagnostics[current] = entries + (
                DiagnosticEntry(start=int(s), end=int(e), message=msg, severity=sev),
            )
        elif line.startswith("<X "):
            source, cpath = payload(line, "<X ").split(" ", 1)
            content = "\n".join(take_block("</X>"))
            state.retrieved.append(
                RetrievedContext(seq=0, path=cpath, content=content, source=source)
            )
        elif line.startswith("<A "):
            cid, status = payload(line, "<A ").split(" ", 1)
            if status not in ("accepted", "partial", "rejected"):
                raise ValueError(f"bad completion status: {status!r}")
            state.completions[cid] = status
        elif line.startswith(PREDICT_TAG):
            rest = payload(line, PREDICT_TAG).strip()
            if rest:
                ppath, off = rest.rsplit(" ", 1)
                state.last_predict = Predict(seq=0, path=ppath, offset=int(off))
            if i != n:
                raise ValueError("<P> must be the last line")
        elif line == "":
            continue
        else:
            raise ValueError(f"unexpected line: {line!r}")
    return state


def count_chars(serialized: str) -> int:
    return len(serialized)


def qwen_token_count(serialized: str, model_id: str = "Qwen/Qwen3.5-0.8B-Base") -> int:
    """Count Qwen tokens via the hub tokenizer (config/tokenizer only, no weights).

    Raises RuntimeError when the tokenizer cannot be downloaded (e.g. offline);
    callers should fall back to character counts and note the absence.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers not installed") from exc
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError(f"cannot load tokenizer {model_id}: {exc}") from exc
    return len(tok.encode(serialized))
