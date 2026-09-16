"""Serialization round-trip + compactness tests."""

from tinycomplete.protocol.events import (
    Cursor,
    DiagnosticEntry,
    Diagnostics,
    OpenFile,
    Predict,
    RetrievedContext,
    Snapshot,
)
from tinycomplete.protocol.replay import EditorState, replay
from tinycomplete.protocol.serialize import (
    count_chars,
    parse_serialized,
    serialize_state,
)


def _demo_state() -> EditorState:
    events = [
        OpenFile(seq=0, path="src/parser.py"),
        Snapshot(seq=1, path="src/parser.py", content="def parse(tok):\n    return tok\n"),
        Cursor(seq=2, path="src/parser.py", offset=4),
        Diagnostics(
            seq=3,
            path="src/parser.py",
            entries=(DiagnosticEntry(start=0, end=3, message="E0308 odd", severity="warning"),),
        ),
        RetrievedContext(
            seq=4, path="src/lex.py", content="def lex(s):\n    pass\n", source="tree-sitter"
        ),
        Predict(seq=5, path="src/parser.py", offset=20),
    ]
    return replay(events)


def test_round_trip():
    state = _demo_state()
    text = serialize_state(state)
    assert text.rstrip("\n").split("\n")[-1].startswith("<P ")
    back = parse_serialized(text)
    assert back.files == state.files
    assert back.open_files == state.open_files
    assert back.cursors == state.cursors
    assert back.diagnostics == state.diagnostics
    assert [ (c.path, c.content, c.source) for c in back.retrieved ] == [
        (c.path, c.content, c.source) for c in state.retrieved
    ]
    assert back.last_predict is not None
    assert back.last_predict.path == "src/parser.py"
    assert back.last_predict.offset == 20


def test_escaping_round_trip():
    tricky = "<F fake>\n\\<not a tag>\nplain"
    state = EditorState(files={"a.py": tricky}, open_files=["a.py"])
    assert parse_serialized(serialize_state(state)).files["a.py"] == tricky


def test_compactness_vs_json():
    import json

    from tinycomplete.protocol.events import event_to_dict

    state = _demo_state()
    text = serialize_state(state)
    verbose = json.dumps(
        [event_to_dict(e) for e in [OpenFile(seq=0, path="x")] for _ in [0]], default=str
    )
    assert count_chars(text) < 2000  # small session stays small
    assert "<P src/parser.py 20>" in text
    assert verbose  # json path exists; format itself is the compact one
