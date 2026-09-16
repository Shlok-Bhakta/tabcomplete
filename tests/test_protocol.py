"""Editor event protocol: replay semantics, validation, UTF-8 behavior."""

import pytest

from tinycomplete.protocol.events import (
    AcceptCompletion,
    CloseFile,
    Cursor,
    Delete,
    DiagnosticEntry,
    Diagnostics,
    Insert,
    OpenFile,
    PartialAcceptCompletion,
    Predict,
    RejectCompletion,
    Replace,
    RetrievedContext,
    Snapshot,
    byte_len,
    event_from_dict,
    event_to_dict,
)
from tinycomplete.protocol.replay import replay


def _seq(events):
    """Assign 0-based sequence numbers in order (helper for test readability)."""
    out = []
    for i, e in enumerate(events):
        out.append(e.model_copy(update={"seq": i}))
    return out


def test_open_snapshot_insert():
    events = _seq(
        [
            OpenFile(seq=0, path="a.py"),
            Snapshot(seq=0, path="a.py", content="def f():\n    pass\n"),
            Insert(seq=0, path="a.py", offset=byte_len("def f():\n    "), text="x = 1\n    "),
        ]
    )
    state = replay(events)
    assert state.files["a.py"] == "def f():\n    x = 1\n    pass\n"
    assert state.open_files == ["a.py"]


def test_insert_delete():
    events = _seq(
        [
            OpenFile(seq=0, path="a.py"),
            Snapshot(seq=0, path="a.py", content="hello world"),
            Insert(seq=0, path="a.py", offset=5, text=" brave"),
            Delete(seq=0, path="a.py", start=0, end=5),
        ]
    )
    state = replay(events)
    assert state.files["a.py"] == " brave world"


def test_replace_encodes_bounds_and_text():
    ev = Replace(seq=1, path="a.py", start=6, end=11, text="there")
    assert (ev.start, ev.end, ev.text) == (6, 11, "there")
    events = _seq(
        [
            OpenFile(seq=0, path="a.py"),
            Snapshot(seq=0, path="a.py", content="hello world"),
            ev,
        ]
    )
    assert replay(events).files["a.py"] == "hello there"


def test_unicode_byte_offsets():
    content = "héllo 🌍 world"
    # 'é' is 2 bytes, emoji is 4 bytes: byte offsets differ from char offsets.
    assert byte_len("héllo") == 6
    prefix = "héllo 🌍 ".encode()
    events = _seq(
        [
            OpenFile(seq=0, path="u.py"),
            Snapshot(seq=0, path="u.py", content=content),
            Insert(seq=0, path="u.py", offset=len(prefix), text="brave "),
        ]
    )
    assert replay(events).files["u.py"] == "héllo 🌍 brave world"
    # Splitting a multi-byte sequence must be rejected.
    bad = _seq(
        [
            OpenFile(seq=0, path="u.py"),
            Snapshot(seq=0, path="u.py", content=content),
            Insert(seq=0, path="u.py", offset=2, text="X"),  # inside 'é'
        ]
    )
    with pytest.raises(ValueError):
        replay(bad)


def test_several_files():
    events = _seq(
        [
            OpenFile(seq=0, path="a.py"),
            OpenFile(seq=0, path="b.py"),
            Snapshot(seq=0, path="a.py", content="aaa"),
            Snapshot(seq=0, path="b.py", content="bbb"),
            Insert(seq=0, path="a.py", offset=3, text="A"),
            Insert(seq=0, path="b.py", offset=0, text="B"),
            CloseFile(seq=0, path="b.py"),
        ]
    )
    state = replay(events)
    assert state.files == {"a.py": "aaaA", "b.py": "Bbbb"}
    assert state.open_files == ["a.py"]


def test_cursor_movement():
    events = _seq(
        [
            OpenFile(seq=0, path="a.py"),
            Snapshot(seq=0, path="a.py", content="abcdef"),
            Cursor(seq=0, path="a.py", offset=2),
            Cursor(seq=0, path="a.py", offset=6),
        ]
    )
    assert replay(events).cursors["a.py"] == 6


def test_snapshot_rebase():
    events = _seq(
        [
            OpenFile(seq=0, path="a.py"),
            Snapshot(seq=0, path="a.py", content="v1"),
            Insert(seq=0, path="a.py", offset=2, text="-dirty"),
            Snapshot(seq=0, path="a.py", content="v2-clean"),
        ]
    )
    assert replay(events).files["a.py"] == "v2-clean"


def test_deterministic_replay_twice():
    events = _seq(
        [
            OpenFile(seq=0, path="a.py"),
            Snapshot(seq=0, path="a.py", content="x = 1"),
            Insert(seq=0, path="a.py", offset=5, text="\ny = 2"),
            Cursor(seq=0, path="a.py", offset=3),
            Diagnostics(
                seq=0,
                path="a.py",
                entries=(DiagnosticEntry(start=0, end=1, message="E1"),),
            ),
            RetrievedContext(seq=0, path="a.py", content="ctx", source="tree-sitter"),
            AcceptCompletion(seq=0, completion_id="c1", text="y = 2"),
            PartialAcceptCompletion(seq=0, completion_id="c2", text="par"),
            RejectCompletion(seq=0, completion_id="c3", reason="bad"),
            Predict(seq=0, path="a.py", offset=5),
        ]
    )
    first = replay(events)
    second = replay(events)
    assert first.files == second.files
    assert first.cursors == second.cursors
    assert first.completions == second.completions
    assert first.last_predict == second.last_predict
    assert first.files["a.py"] == "x = 1\ny = 2"
    assert first.completions == {"c1": "accepted", "c2": "partial", "c3": "rejected"}


def test_invalid_sequence_numbers_rejected():
    events = [
        OpenFile(seq=0, path="a.py"),
        Snapshot(seq=5, path="a.py", content="x"),
    ]
    with pytest.raises(ValueError, match="invalid sequence number"):
        replay(events)
    duplicate = [
        OpenFile(seq=0, path="a.py"),
        Snapshot(seq=0, path="a.py", content="x"),
    ]
    with pytest.raises(ValueError, match="invalid sequence number"):
        replay(duplicate)


def test_edits_outside_bounds_rejected():
    base = [OpenFile(seq=0, path="a.py"), Snapshot(seq=1, path="a.py", content="abc")]
    with pytest.raises(ValueError):
        replay([*base, Insert(seq=2, path="a.py", offset=99, text="x")])
    with pytest.raises(ValueError):
        replay([*base, Delete(seq=2, path="a.py", start=0, end=99)])
    with pytest.raises(ValueError):
        replay([*base, Replace(seq=2, path="a.py", start=2, end=1, text="x")])
    with pytest.raises(ValueError):
        replay([*base, Cursor(seq=2, path="a.py", offset=99)])
    with pytest.raises(ValueError, match="file not open"):
        replay([Insert(seq=0, path="ghost.py", offset=0, text="x")])


def test_insert_does_not_mutate_historical_event():
    ins = Insert(seq=2, path="a.py", offset=1, text="X")
    before = event_to_dict(ins)
    events = [
        OpenFile(seq=0, path="a.py"),
        Snapshot(seq=1, path="a.py", content="abc"),
        ins,
    ]
    replay(events)
    assert event_to_dict(ins) == before
    assert event_from_dict(before) == ins


def test_event_dict_round_trip_all_kinds():
    events = _seq(
        [
            OpenFile(seq=0, path="a.py"),
            CloseFile(seq=0, path="a.py"),
            Snapshot(seq=0, path="a.py", content="x"),
            Insert(seq=0, path="a.py", offset=0, text="x"),
            Delete(seq=0, path="a.py", start=0, end=1),
            Replace(seq=0, path="a.py", start=0, end=1, text="y"),
            Cursor(seq=0, path="a.py", offset=0),
            Diagnostics(seq=0, path="a.py", entries=()),
            RetrievedContext(seq=0, content="c"),
            AcceptCompletion(seq=0, completion_id="a"),
            PartialAcceptCompletion(seq=0, completion_id="b"),
            RejectCompletion(seq=0, completion_id="c"),
            Predict(seq=0, path="a.py", offset=0),
        ]
    )
    for e in events:
        assert event_from_dict(event_to_dict(e)) == e
    with pytest.raises(ValueError, match="unknown event kind"):
        event_from_dict({"kind": "NOPE", "seq": 0})
