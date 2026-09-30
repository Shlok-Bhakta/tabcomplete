from tinycomplete.one_line.commit_sequences import reconstruct_candidates
from tinycomplete.one_line.contract import EditAction, EditState, apply_action
from tinycomplete.one_line.data import replay_replacement_history


def test_reconstructed_history_is_prior_and_replays_exact_unicode_crlf():
    before = "α = old()\r\nβ = old()\r\nreturn β\r\n"
    after = "α = new()\r\nβ = new()\r\nreturn β\r\n"
    rows = reconstruct_candidates(before, after, file_id="sample.py", filetype="python")
    assert len(rows) == 1
    row = rows[0]
    state = EditState.from_mapping(row["state"])
    assert len(state.history) == 1
    assert (
        replay_replacement_history(
            before, state.history, file_id=state.file_id, filetype=state.filetype
        )
        == state.source
    )
    assert apply_action(state, EditAction(**row["action"])) == after
    assert row["validation"]["accepted_training"] is False


def test_actual_line_insertion_and_deletion_preserve_other_bytes():
    before = "a = old()\nx = 1\ndebug(x)\nreturn a\n"
    for after, kind in (
        ("a = new()\nx = 1\ncheck(x)\ndebug(x)\nreturn a\n", "insert_before"),
        ("a = new()\nx = 1\nreturn a\n", "delete_line"),
    ):
        rows = reconstruct_candidates(before, after, file_id="sample.py", filetype="python")
        assert len(rows) == 1
        assert rows[0]["action"]["kind"] == kind
        assert rows[0]["after_source"] == after


def test_no_idle_keep_and_no_target_answer_in_history():
    before = "a = old()\nb = old()\n"
    assert reconstruct_candidates(before, before, file_id="a.py", filetype="python") == []
    rows = reconstruct_candidates(
        before, "a = new()\nb = new()\n", file_id="a.py", filetype="python"
    )
    assert rows[0]["state"]["history"] == (
        {"row": 0, "old_text": "a = old()", "new_text": "a = new()"},
    )


def test_mixed_terminator_history_is_not_silently_normalized():
    rows = reconstruct_candidates(
        "a = old()\r\nb = old()\r\n",
        "a = new()\nb = new()\r\n",
        file_id="a.py",
        filetype="python",
    )
    assert rows == []
