"""A declared identifier-copy control, not a model or acceptance oracle.

The historical whole-line-copy baseline remains unchanged. New comparisons
must name this control explicitly and freeze its version before scoring.
"""

from __future__ import annotations

from tinycomplete.one_line.contract import EditAction, EditState, physical_lines

VERSION = "visible-identifier-copy-v1"
IDENTIFIER_TYPES = frozenset(
    {
        "identifier",
        "field_identifier",
        "property_identifier",
        "type_identifier",
        "shorthand_property_identifier",
        "shorthand_property_identifier_pattern",
    }
)


def _identifiers(text: bytes, language: str) -> list[tuple[int, int, bytes]]:
    from tree_sitter_language_pack import get_parser

    tree = get_parser(language).parse(text)
    found: list[tuple[int, int, bytes]] = []
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        # A string interpolation is intentionally excluded too. This control
        # must not rewrite comments or text merely because spelling matches.
        if "comment" in node.type or "string" in node.type or node.type == "char_literal":
            continue
        if node.type in IDENTIFIER_TYPES:
            found.append((node.start_byte, node.end_byte, text[node.start_byte : node.end_byte]))
        else:
            pending.extend(node.children)
    return sorted(found)


def predict_visible_identifier_copy(state: EditState) -> EditAction:
    """Propagate one exact identifier substitution from the latest prior edit.

    This deliberately simple control can be wrong about scope or intent. Its
    outcomes must be measured alongside keep and the model, never relabeled as
    ground truth. Only bytes and history supplied to the student are inspected.
    """
    if not state.history:
        return EditAction("keep")
    latest = state.history[-1]
    if latest.row == state.target_row:
        return EditAction("keep")
    lines = physical_lines(state.source.encode("utf-8"))
    if latest.row >= len(lines) or state.target_row >= len(lines):
        return EditAction("keep")
    if lines[latest.row].content.decode("utf-8") != latest.new_text:
        return EditAction("keep")
    old, new = latest.old_text.encode("utf-8"), latest.new_text.encode("utf-8")
    old_tokens, new_tokens = _identifiers(old, state.filetype), _identifiers(new, state.filetype)
    if len(old_tokens) != len(new_tokens):
        return EditAction("keep")
    changed = [(a, b) for a, b in zip(old_tokens, new_tokens, strict=True) if a[2] != b[2]]
    if len(changed) != 1:
        return EditAction("keep")
    before, after = changed[0]
    if old[: before[0]] + after[2] + old[before[1] :] != new:
        return EditAction("keep")
    target = lines[state.target_row].content
    occurrences = [token for token in _identifiers(target, state.filetype) if token[2] == before[2]]
    if len(occurrences) != 1:
        return EditAction("keep")
    token = occurrences[0]
    replacement = target[: token[0]] + after[2] + target[token[1] :]
    return EditAction("replace_line", replacement.decode("utf-8"))
