import pytest

from tinycomplete.one_line.contract import EditAction, EditState, RecentEdit
from tinycomplete.one_line.visible_rules import predict_visible_identifier_copy


@pytest.mark.parametrize(
    "language,old,new,target,expected",
    [
        ("python", "value = old_name", "value = new_name", "return old_name", "return new_name"),
        (
            "typescript",
            "const oldName = 1;",
            "const newName = 1;",
            "use(oldName);",
            "use(newName);",
        ),
        ("go", "out := writer", "w := writer", "defer out.Flush()", "defer w.Flush()"),
        (
            "rust",
            "let old_name = item;",
            "let new_name = item;",
            "use_item(old_name);",
            "use_item(new_name);",
        ),
        ("python", "π = 1", "τ = 1", "print(π)", "print(τ)"),
    ],
)
def test_propagates_one_visible_identifier(language, old, new, target, expected):
    state = EditState(
        "test", language, new + "\n" + target + "\n", 1, 0, (RecentEdit(0, old, new),)
    )
    assert predict_visible_identifier_copy(state) == EditAction("replace_line", expected)


@pytest.mark.parametrize(
    "old,new,target",
    [
        ("value = old_name", "value = new_name", "print('old_name')"),
        ("value = old_name", "value = new_name", "# old_name"),
        ("value = old_name", "value = new_name", "old_name + old_name"),
        ("value = old_name", "value = new_name", "old_name_longer"),
        ("value = old_name + 1", "value = new_name + 2", "return old_name"),
        ("value = old_name + other", "value = new_name + another", "return old_name"),
    ],
)
def test_ambiguous_or_nonidentifier_copy_keeps(old, new, target):
    state = EditState("test", "python", new + "\n" + target, 1, 0, (RecentEdit(0, old, new),))
    assert predict_visible_identifier_copy(state) == EditAction("keep")


def test_does_not_reverse_latest_intent_or_trust_unreplayed_history():
    history = (RecentEdit(0, "old = 1", "new = 1"),)
    assert predict_visible_identifier_copy(
        EditState("test", "python", "new = 1", 0, 0, history)
    ) == EditAction("keep")
    assert predict_visible_identifier_copy(
        EditState("test", "python", "other = 1\nold", 1, 0, history)
    ) == EditAction("keep")
