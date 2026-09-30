"""Exact, explicitly reconstructed edit sequences from public content pairs.

These are review candidates. Git ordering, related intent, and useful no-edit
behavior are not established by a final diff, so this module certifies only
byte reconstruction. It never labels an unchanged/idle state as keep.
"""

from __future__ import annotations

import difflib
from dataclasses import asdict
from typing import Any

from .contract import EditAction, EditState, Filetype, RecentEdit, apply_action, physical_lines
from .data import replay_replacement_history, sha256_bytes

VERSION = "public-content-sequence-candidates-v1"


def reconstruct_candidates(
    before: str,
    after: str,
    *,
    file_id: str,
    filetype: Filetype,
    max_history: int = 4,
    max_changes: int = 24,
) -> list[dict[str, Any]]:
    """Replay earlier replacements and one target in deterministic file order.

    Insert/delete histories are excluded because they change row coordinates.
    Equal-size replacement blocks may contain up to four physical lines; every
    emitted target still changes exactly one line. All terminators must match
    their canonical action; unsupported EOF changes are rejected, not repaired.
    """
    if not 1 <= max_history <= 5 or max_changes < 2:
        raise ValueError("invalid reconstruction bounds")
    old = physical_lines(before.encode())
    new = physical_lines(after.encode())
    changes = [
        opcode
        for opcode in difflib.SequenceMatcher(
            None, [line.raw for line in old], [line.raw for line in new], autojunk=False
        ).get_opcodes()
        if opcode[0] != "equal"
    ]
    if not 1 <= len(changes) <= max_changes:
        return []
    atomic: list[tuple[int, EditAction, bytes]] = []
    for tag, a0, a1, b0, b1 in changes:
        a, b = a1 - a0, b1 - b0
        if tag == "replace" and a == b and 1 <= a <= 4:
            for offset in range(a):
                if old[a0 + offset].raw != new[b0 + offset].raw:
                    atomic.append(
                        (
                            a0 + offset,
                            EditAction("replace_line", new[b0 + offset].content.decode()),
                            new[b0 + offset].raw,
                        )
                    )
        elif tag == "insert" and b == 1:
            atomic.append((a0, EditAction("insert_before", new[b0].content.decode()), new[b0].raw))
        elif tag == "delete" and a == 1:
            atomic.append((a0, EditAction("delete_line"), b""))
    previous: list[tuple[int, EditAction]] = []
    result = []
    for target, action, target_raw in atomic:
        selected = [(row, prior) for row, prior in previous if row < target][-max_history:]
        if selected:
            history = tuple(
                RecentEdit(row, old[row].content.decode(), str(prior.text))
                for row, prior in selected
            )
            current = replay_replacement_history(
                before, history, file_id=file_id, filetype=filetype
            )
            state = EditState(file_id, filetype, current, target, 0, history)
            rebuilt = apply_action(state, action)
            current_lines = list(physical_lines(current.encode()))
            expected = (
                b"".join(line.raw for line in current_lines[:target])
                + target_raw
                + b"".join(
                    line.raw for line in current_lines[target + (action.kind != "insert_before") :]
                )
            )
            if rebuilt.encode() == expected:
                result.append(
                    {
                        "state": asdict(state),
                        "action": asdict(action),
                        "after_source": rebuilt,
                        "source_type": "git_reconstructed_order",
                        "provenance": {
                            "human_edit_order_observed": False,
                            "history_order": "earlier replacement rows, ascending file order",
                            "history_origin_sha256": sha256_bytes(before.encode()),
                            "committed_child_sha256": sha256_bytes(after.encode()),
                        },
                        "validation": {
                            "history_replays_to_state": True,
                            "apply_reconstructs_after": True,
                            "equals_committed_child": rebuilt == after,
                            "inferability_reviewed": False,
                            "accepted_training": False,
                        },
                    }
                )
        if action.kind == "replace_line":
            # Reject terminator changes from history as well as from targets.
            probe = EditState(file_id, filetype, before, target, 0)
            actual = physical_lines(apply_action(probe, action).encode())[target].raw
            if actual == target_raw:
                previous.append((target, action))
    return result
