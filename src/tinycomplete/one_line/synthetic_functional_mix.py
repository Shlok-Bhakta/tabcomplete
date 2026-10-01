"""Author-owned functional N/I/D examples for the public-prefix pilot route.

The code examples are deliberately synthetic. Their small private oracle
fixtures are executed by the benchmark sandbox, never included in model input.
One algorithm family and its variants remain together in a single split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION, serialize_state
from tinycomplete.one_line.contract import (
    EditAction,
    EditState,
    RecentEdit,
    apply_action,
    encode_action,
    physical_lines,
)
from tinycomplete.one_line.data import near_duplicate_key
from tinycomplete.one_line.public_prefix_pilot import canonical_sha256, sha256_bytes

SYNTHETIC_FUNCTIONAL_SOURCE_TYPE = "author_owned_synthetic_functional"
SYNTHETIC_FUNCTIONAL_LICENSE = "author_owned_synthetic_not_published"
SYNTHETIC_HISTORY_ORDER = "synthetic_history_conditioned_functional_v1"
SYNTHETIC_RECEIPT_SCHEMA = "one-line-synthetic-functional-receipt-v1"
SYNTHETIC_PILOT_SCHEMA = "one-line-public-synthetic-functional-mix-v1"
SYNTHETIC_FUNCTIONAL_FAMILY_COUNT = 16
SYNTHETIC_VARIANTS_PER_ACTION = 4
SYNTHETIC_TRAIN_FAMILY_FLOOR = 12
SYNTHETIC_DEVELOPMENT_FAMILY_FLOOR = 4
SYNTHETIC_GENERATOR_REVISION = sha256_bytes(Path(__file__).read_bytes())
ORDER_POLICIES = ("preserve", "reverse", "ascending", "descending")


@dataclass(frozen=True)
class AlgorithmFamily:
    key: str
    helpers: str
    normalize: str
    project: str
    data: tuple[Any, Any, Any, Any]
    expected: tuple[Any, Any, Any, Any]
    empty_results: tuple[Any, Any, Any, Any]
    wrong_empty_results: tuple[Any, Any, Any, Any]


FAMILIES: tuple[AlgorithmFamily, ...] = (
    AlgorithmFamily(
        "stable_unique_text",
        """def _keep_first(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
""",
        "[str(value).strip().casefold() for value in raw]",
        "normalized",
        (
            ("A", "b", "a", "C"),
            ("red", "blue", "red", "green"),
            ("Kiwi", "lime", "kiwi", "plum"),
            ("east", "west", "north", "west"),
        ),
        (
            ("a", "b", "c"),
            ("red", "blue", "green"),
            ("kiwi", "lime", "plum"),
            ("east", "west", "north"),
        ),
        (("<empty>",), ("<missing>",), ("<unset>",), ("<default>",)),
        (("<wrong-empty>",), ("<wrong-missing>",), ("<wrong-unset>",), ("<wrong-default>",)),
    ),
    AlgorithmFamily(
        "bounded_numeric_values",
        "LOW = 0\nHIGH = 10\n",
        "list(raw)",
        "normalized",
        ((-5, 2, 13), (3, 9, 12), (-8, -2, 4), (-2, 1, 13)),
        ((0, 2, 10), (3, 9, 10), (0, 0, 4), (0, 1, 10)),
        ((0,), (-1,), (99,), (5,)),
        ((1,), (2,), (3,), (4,)),
    ),
    AlgorithmFamily(
        "ordered_group_summary",
        """def _summarize_pairs(pairs):
    grouped = {}
    for key, value in pairs:
        grouped.setdefault(key, []).append(value)
    return [f"{key}={','.join(grouped[key])}" for key in grouped]
""",
        "[(record['key'], str(record['value'])) for record in raw]",
        "normalized",
        (
            ({"key": "b", "value": 2}, {"key": "a", "value": 1}, {"key": "a", "value": 3}),
            ({"key": "x", "value": 4}, {"key": "y", "value": 5}, {"key": "x", "value": 6}),
            (
                {"key": "late", "value": 8},
                {"key": "early", "value": 2},
                {"key": "late", "value": 9},
            ),
            (
                {"key": "north", "value": 1},
                {"key": "south", "value": 7},
                {"key": "west", "value": 3},
            ),
        ),
        (
            ("b=2", "a=1,3"),
            ("x=4,6", "y=5"),
            ("late=8,9", "early=2"),
            ("north=1", "south=7", "west=3"),
        ),
        (("no-groups",), ("empty-group",), ("default-group",), ("unassigned",)),
        (("wrong-group",), ("wrong-empty",), ("wrong-default",), ("wrong-unassigned",)),
    ),
    AlgorithmFamily(
        "left_rotation",
        "SHIFT = 1\n",
        "list(raw)",
        "normalized",
        ((1, 2, 3), (10, 20, 30, 40), (7, 5, 9), (8, 3, 6, 1, 4)),
        ((2, 3, 1), (20, 30, 40, 10), (5, 9, 7), (3, 6, 1, 4, 8)),
        ((0,), (-1,), (99,), (5,)),
        ((1,), (2,), (3,), (4,)),
    ),
    AlgorithmFamily(
        "fixed_width_chunking",
        "CHUNK_WIDTH = 2\n",
        "list(raw)",
        "normalized",
        ((1, 2, 3, 4, 5), (9, 8, 7, 6), (2, 5, 8, 11, 14), (4, 1, 3, 9, 2, 8)),
        (
            ((1, 2), (3, 4), (5,)),
            ((9, 8), (7, 6)),
            ((2, 5), (8, 11), (14,)),
            ((4, 1), (3, 9), (2, 8)),
        ),
        (((0,),), ((-1,),), ((99,),), ((5,),)),
        (((1,),), ((2,),), ((3,),), ((4,),)),
    ),
    AlgorithmFamily(
        "explicit_config_merge",
        """def _merge_config(base, overrides):
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return [f"{key}={merged[key]}" for key in merged]
""",
        "raw",
        "normalized",
        (
            ({"mode": "safe", "limit": 4}, {"mode": "fast", "limit": None, "trace": True}),
            ({"path": "/tmp/a", "retry": 2}, {"retry": 0, "cache": False}),
            ({"format": "json", "strict": True}, {"strict": False, "format": None}),
            ({"worker": 1, "enabled": True}, {"worker": 3, "enabled": False}),
        ),
        (
            ("mode=fast", "limit=4", "trace=True"),
            ("path=/tmp/a", "retry=0", "cache=False"),
            ("format=json", "strict=False"),
            ("worker=3", "enabled=False"),
        ),
        (("default-config",), ("empty-config",), ("inherited-config",), ("safe-config",)),
        (("wrong-config",), ("wrong-empty",), ("wrong-inherited",), ("wrong-safe",)),
    ),
    AlgorithmFamily(
        "html_text_escaping",
        """def _escape(value):
    return (value.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
""",
        "[str(value) for value in raw]",
        "normalized",
        (
            ('<tag>&"', "plain"),
            ("fish & chips", "<b>bold</b>"),
            ('"quote"', "a < b"),
            ("x > y", "&done;"),
        ),
        (
            ("&lt;tag&gt;&amp;&quot;", "plain"),
            ("fish &amp; chips", "&lt;b&gt;bold&lt;/b&gt;"),
            ("&quot;quote&quot;", "a &lt; b"),
            ("x &gt; y", "&amp;done;"),
        ),
        (("&lt;empty&gt;",), ("&lt;missing&gt;",), ("&lt;unset&gt;",), ("&lt;default&gt;",)),
        (("wrong-empty",), ("wrong-missing",), ("wrong-unset",), ("wrong-default",)),
    ),
    AlgorithmFamily(
        "adjacent_numeric_differences",
        "",
        "list(raw)",
        "normalized",
        ((2, 5, 3, 9), (10, 7, 8, 2), (-4, 1, 6), (3, 3, 8, 1)),
        ((3, -2, 6), (-3, 1, -6), (5, 5), (0, 5, -7)),
        ((0,), (-1,), (99,), (5,)),
        ((1,), (2,), (3,), (4,)),
    ),
    AlgorithmFamily(
        "bounded_page_selection",
        "PAGE_START = 1\nPAGE_SIZE = 2\n",
        "list(raw)",
        "normalized",
        (
            ("a", "b", "c", "d"),
            (10, 20, 30, 40, 50),
            ("north", "east", "south"),
            (4, 8, 15, 16, 23),
        ),
        (("b", "c"), (20, 30), ("east", "south"), (8, 15)),
        (("<empty-page>",), ("<no-page>",), ("<default-page>",), ("<missing-page>",)),
        (("wrong-page",), ("wrong-empty",), ("wrong-default",), ("wrong-missing",)),
    ),
    AlgorithmFamily(
        "interval_coalescing",
        """def _coalesce(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
""",
        "[tuple(interval) for interval in raw]",
        "normalized",
        (
            ((5, 8), (1, 3), (3, 6), (10, 12)),
            ((0, 2), (7, 9), (1, 4), (12, 15)),
            ((4, 5), (2, 4), (8, 10), (9, 11)),
            ((20, 22), (1, 4), (3, 7), (10, 13)),
        ),
        (
            ((1, 8), (10, 12)),
            ((0, 4), (7, 9), (12, 15)),
            ((2, 5), (8, 11)),
            ((1, 7), (10, 13), (20, 22)),
        ),
        (((0, 0),), ((-1, -1),), ((99, 99),), ((5, 5),)),
        (((1, 1),), ((2, 2),), ((3, 3),), ((4, 4),)),
    ),
    AlgorithmFamily(
        "decimal_token_parsing",
        "",
        "list(raw)",
        "normalized",
        (("1_000", "-24", "7"), ("8", "-3_200", "14"), ("42", "+6", "-11"), ("5_001", "90", "-2")),
        ((1000, -24, 7), (8, -3200, 14), (42, 6, -11), (5001, 90, -2)),
        ((0,), (-1,), (99,), (5,)),
        ((1,), (2,), (3,), (4,)),
    ),
    AlgorithmFamily(
        "run_length_summary",
        """def _count_runs(values):
    result = []
    for value in values:
        if result and result[-1][0] == value:
            result[-1] = (value, result[-1][1] + 1)
        else:
            result.append((value, 1))
    return [f"{value}:{count}" for value, count in result]
""",
        "list(raw)",
        "normalized",
        (
            ("a", "a", "b", "b", "b", "c"),
            ("up", "down", "down", "up"),
            (1, 1, 1, 2, 2),
            ("x", "y", "z", "z", "x"),
        ),
        (
            ("a:2", "b:3", "c:1"),
            ("up:1", "down:2", "up:1"),
            ("1:3", "2:2"),
            ("x:1", "y:1", "z:2", "x:1"),
        ),
        (("no-runs",), ("empty-runs",), ("default-runs",), ("missing-runs",)),
        (("wrong-runs",), ("wrong-empty",), ("wrong-default",), ("wrong-missing",)),
    ),
    AlgorithmFamily(
        "sequence_interleaving",
        "",
        "(list(raw['left']), list(raw['right']))",
        "normalized",
        (
            {"left": (1, 3, 5), "right": (2, 4)},
            {"left": (10, 30), "right": (20, 40, 60)},
            {"left": (7, 8, 9, 10), "right": (1, 2, 3)},
            {"left": (5, 15, 25), "right": (10, 20, 30, 40)},
        ),
        (
            (1, 2, 3, 4, 5),
            (10, 20, 30, 40, 60),
            (7, 1, 8, 2, 9, 3, 10),
            (5, 10, 15, 20, 25, 30, 40),
        ),
        ((0,), (-1,), (99,), (5,)),
        ((1,), (2,), (3,), (4,)),
    ),
    AlgorithmFamily(
        "record_key_deduplication",
        """def _keep_first(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
""",
        "[str(record['id']).casefold() for record in raw]",
        "normalized",
        (
            ({"id": "A"}, {"id": "B"}, {"id": "a"}),
            ({"id": "x"}, {"id": "y"}, {"id": "x"}),
            ({"id": "first"}, {"id": "second"}, {"id": "FIRST"}),
            ({"id": "p"}, {"id": "q"}, {"id": "r"}, {"id": "P"}),
        ),
        (("a", "b"), ("x", "y"), ("first", "second"), ("p", "q", "r")),
        (("no-records",), ("empty-records",), ("default-records",), ("missing-records",)),
        (("wrong-records",), ("wrong-empty",), ("wrong-default",), ("wrong-missing",)),
    ),
    AlgorithmFamily(
        "csv_row_parsing",
        """import csv
def _csv_fields(value):
    return next(csv.reader([value]))
""",
        "[str(value) for value in raw]",
        "normalized",
        (
            ("north,5,ok", "south,8,hold"),
            ('"New York",3,ready', '"Los Angeles",2,wait'),
            ("a,1,x", '"b,c",2,y'),
            ("one,10,yes", "two,20,no"),
        ),
        (
            (("north", "5", "ok"), ("south", "8", "hold")),
            (("New York", "3", "ready"), ("Los Angeles", "2", "wait")),
            (("a", "1", "x"), ("b,c", "2", "y")),
            (("one", "10", "yes"), ("two", "20", "no")),
        ),
        ((("empty",),), (("missing",),), (("no-rows",),), (("default-row",),)),
        ((("wrong-empty",),), (("wrong-missing",),), (("wrong-rows",),), (("wrong-default",),)),
    ),
    AlgorithmFamily(
        "semantic_version_ordering",
        "",
        "list(raw)",
        "normalized",
        (
            ("1.10", "1.2", "2.0"),
            ("3.1.4", "3.1.2", "3.0.9"),
            ("0.12", "0.3", "0.20"),
            ("2.0.1", "1.9.9", "2.0.0"),
        ),
        (
            ((1, 2), (1, 10), (2, 0)),
            ((3, 0, 9), (3, 1, 2), (3, 1, 4)),
            ((0, 3), (0, 12), (0, 20)),
            ((1, 9, 9), (2, 0, 0), (2, 0, 1)),
        ),
        (((0, 0),), ((-1, -1),), ((99, 99),), ((5, 5),)),
        (((1, 1),), ((2, 2),), ((3, 3),), ((4, 4),)),
    ),
)

if (
    len(FAMILIES) != SYNTHETIC_FUNCTIONAL_FAMILY_COUNT
    or len({family.key for family in FAMILIES}) != SYNTHETIC_FUNCTIONAL_FAMILY_COUNT
):
    raise RuntimeError("synthetic algorithm-family inventory is malformed")

REQUIRED_OPERATIONS = {
    "stable_unique_text": "_keep_first(result)",
    "bounded_numeric_values": "[min(max(value, LOW), HIGH) for value in result]",
    "ordered_group_summary": "_summarize_pairs(result)",
    "left_rotation": "result[SHIFT % max(1, len(result)):] + result[:SHIFT % max(1, len(result))]",
    "fixed_width_chunking": (
        "[result[start:start + CHUNK_WIDTH] for start in range(0, len(result), CHUNK_WIDTH)]"
    ),
    "explicit_config_merge": "_merge_config(result[0], result[1])",
    "html_text_escaping": "[_escape(value) for value in result]",
    "adjacent_numeric_differences": "[right - left for left, right in zip(result, result[1:])]",
    "bounded_page_selection": "result[PAGE_START:PAGE_START + PAGE_SIZE]",
    "interval_coalescing": "_coalesce(result)",
    "decimal_token_parsing": "[int(value.replace('_', '')) for value in result]",
    "run_length_summary": "_count_runs(result)",
    "sequence_interleaving": (
        "[item for pair in zip(result[0], result[1]) for item in pair] + "
        "result[0][len(result[1]):] + result[1][len(result[0]):]"
    ),
    "record_key_deduplication": "list(dict.fromkeys(result))",
    "csv_row_parsing": "[_csv_fields(value) for value in result]",
    "semantic_version_ordering": (
        "sorted(tuple(int(part) for part in value.split('.')) for value in result)"
    ),
}

FUNCTION_CONTRACTS = {
    "stable_unique_text": "Normalize text and retain only each first occurrence.",
    "bounded_numeric_values": "Clamp numeric values to the inclusive LOW and HIGH bounds.",
    "ordered_group_summary": "Group records by key in first-key order and summarize their values.",
    "left_rotation": "Rotate the input sequence left by SHIFT positions.",
    "fixed_width_chunking": "Partition the input into chunks of CHUNK_WIDTH, retaining the tail.",
    "explicit_config_merge": "Merge overrides into base configuration, ignoring None values.",
    "html_text_escaping": "Escape ampersands, angle brackets and quotes as HTML text.",
    "adjacent_numeric_differences": "Compute the difference between each adjacent pair.",
    "bounded_page_selection": "Select PAGE_SIZE values starting at PAGE_START.",
    "interval_coalescing": "Merge overlapping or touching intervals.",
    "decimal_token_parsing": "Parse signed decimal tokens, allowing underscore separators.",
    "run_length_summary": "Summarize consecutive equal values with their run lengths.",
    "sequence_interleaving": "Interleave two sequences and retain the longer sequence's tail.",
    "record_key_deduplication": "Normalize record identifiers and retain each first occurrence.",
    "csv_row_parsing": "Parse each CSV row with proper quoted-field handling.",
    "semantic_version_ordering": "Parse numeric version components and sort versions numerically.",
}

OBSOLETE_EFFECTS = {
    "stable_unique_text": "[value + '!' for value in result]",
    "bounded_numeric_values": "[value + 1 for value in result]",
    "ordered_group_summary": "[value.upper() for value in result]",
    "left_rotation": "result + [result[0]]",
    "fixed_width_chunking": "result + [[]]",
    "explicit_config_merge": "result + ['stale=0']",
    "html_text_escaping": "[value.replace('&', '') for value in result]",
    "adjacent_numeric_differences": "[-value for value in result]",
    "bounded_page_selection": "result + ['<stale-page>']",
    "interval_coalescing": "[(end, start) for start, end in result]",
    "decimal_token_parsing": "[value * 10 for value in result]",
    "run_length_summary": "result[:1]",
    "sequence_interleaving": "result + [999]",
    "record_key_deduplication": "result + ['stale-id']",
    "csv_row_parsing": "[row[1:] for row in result]",
    "semantic_version_ordering": "result + [(999,)]",
}

if set(REQUIRED_OPERATIONS) != {family.key for family in FAMILIES} or set(OBSOLETE_EFFECTS) != {
    family.key for family in FAMILIES
}:
    raise RuntimeError("synthetic operation inventory differs from its family registry")


@dataclass(frozen=True)
class SyntheticCandidate:
    row: dict[str, Any]
    wrong_action: EditAction
    test_source: str
    fixture_sha256: str


def _order(values: Any, policy: str) -> Any:
    if policy == "preserve":
        return list(values)
    if policy == "reverse":
        return list(reversed(values))
    if policy == "ascending":
        return sorted(values)
    if policy == "descending":
        return sorted(values, reverse=True)
    raise ValueError("synthetic output-order policy is invalid")


def _family_by_key(key: str) -> AlgorithmFamily:
    try:
        return next(family for family in FAMILIES if family.key == key)
    except StopIteration:
        raise ValueError("synthetic family key is unknown") from None


def _guard_line() -> str:
    return "    if not DATA: return EMPTY_RESULT"


def _source_lines(
    family: AlgorithmFamily,
    *,
    data: Any,
    order_policy: str,
    empty_result: Any,
    include_required_operation: bool = True,
    include_obsolete_effect: bool = False,
) -> list[str]:
    if order_policy not in ORDER_POLICIES:
        raise ValueError("synthetic order policy is invalid")
    lines = [
        f"ORDER_POLICY = {order_policy!r}",
        f"DATA = {data!r}",
        f"EMPTY_RESULT = {empty_result!r}",
    ]
    if family.helpers:
        lines.extend(family.helpers.rstrip("\n").splitlines())
    lines.extend(("def solve():",))
    lines.append('    """' + FUNCTION_CONTRACTS[family.key] + ' Then apply ORDER_POLICY."""')
    stages = [
        "    raw = DATA",
        f"    normalized = {family.normalize}",
        f"    projected = {family.project}",
        "    result = projected",
    ]
    lines.extend(stages)
    if include_required_operation:
        lines.append(f"    result = {REQUIRED_OPERATIONS[family.key]}")
    lines.extend(
        (
            '    if ORDER_POLICY == "preserve":',
            "        final = result",
            '    elif ORDER_POLICY == "reverse":',
            "        final = result[::-1]",
            '    elif ORDER_POLICY == "ascending":',
            "        final = sorted(result)",
            "    else:",
            "        final = sorted(result, reverse=True)",
        )
    )
    if include_obsolete_effect:
        lines.insert(
            lines.index(f"    result = {REQUIRED_OPERATIONS[family.key]}") + 1,
            f"    result = {OBSOLETE_EFFECTS[family.key]}",
        )
    lines.extend(("    return final", ""))
    return lines


def _render_source(lines: list[str]) -> str:
    return "\n".join(lines)


def _line_number(lines: list[str], value: str) -> int:
    try:
        return lines.index(value)
    except ValueError:
        raise ValueError("synthetic state target line is absent") from None


def _history_identity(lines: list[str], current: str, old: str) -> tuple[RecentEdit, str]:
    row = _line_number(lines, current)
    before = list(lines)
    before[row] = old
    before_source = _render_source(before)
    return RecentEdit(row=row, old_text=old, new_text=current), before_source


def _candidate(
    family: AlgorithmFamily,
    *,
    action_family: str,
    variant: int,
) -> SyntheticCandidate:
    if not 0 <= variant < SYNTHETIC_VARIANTS_PER_ACTION:
        raise ValueError("synthetic family variant is outside the frozen set")
    order_policy = (
        ("preserve", "reverse", "preserve", "reverse")[variant]
        if family.key == "left_rotation"
        else ORDER_POLICIES[variant]
    )
    if action_family == "N":
        data = family.data[variant]
        expected_core = family.expected[variant]
        lines = _source_lines(
            family,
            data=data,
            order_policy=order_policy,
            empty_result=family.empty_results[variant],
        )
        current = f"    result = {REQUIRED_OPERATIONS[family.key]}"
        recent, before_source = _history_identity(lines, current, "    result = projected")
        target = _line_number(lines, current)
        action = EditAction("keep")
        wrong = EditAction("replace_line", "    result = projected")
        expected = _order(expected_core, order_policy)
        test_source = _oracle_test(expected)
    elif action_family == "I":
        data = family.data[variant]
        empty_result = family.empty_results[variant]
        lines = _source_lines(
            family,
            data=data,
            order_policy=order_policy,
            empty_result=empty_result,
            include_required_operation=False,
        )
        old_data = family.data[(variant + 1) % SYNTHETIC_VARIANTS_PER_ACTION]
        current = f"DATA = {data!r}"
        recent, before_source = _history_identity(lines, current, f"DATA = {old_data!r}")
        target = _line_number(lines, '    if ORDER_POLICY == "preserve":')
        operation = "    result = " + REQUIRED_OPERATIONS[family.key]
        action = EditAction("insert_before", operation)
        wrong = EditAction("insert_before", "    result = projected")
        expected = _order(family.expected[variant], order_policy)
        test_source = _oracle_test(expected)
    elif action_family == "D":
        data = family.data[variant]
        expected_core = family.expected[variant]
        lines = _source_lines(
            family,
            data=data,
            order_policy=order_policy,
            empty_result=family.empty_results[variant],
            include_obsolete_effect=True,
        )
        stale = f"    result = {OBSOLETE_EFFECTS[family.key]}"
        target = _line_number(lines, stale)
        # The latest edit establishes the intended transformation. The deleted
        # effect predates it; retaining the user's new transformation is required.
        current = f"    result = {REQUIRED_OPERATIONS[family.key]}"
        recent, before_source = _history_identity(lines, current, "    result = projected")
        action = EditAction("delete_line")
        wrong = EditAction("keep")
        expected = _order(expected_core, order_policy)
        test_source = _oracle_test(expected)
    else:
        raise ValueError("synthetic action family is invalid")

    source = _render_source(lines)
    state = EditState(
        file_id="synthetic/example.py",
        filetype="python",
        source=source,
        target_row=target,
        cursor_col=0,
        history=(recent,),
        relevant=(),
    )
    after_source = apply_action(state, action)
    prompt = serialize_state(state)
    action_mapping = {"kind": action.kind, "text": action.text}
    row: dict[str, Any] = {
        "id": f"synthetic/{family.key}/{action_family}/{variant}",
        "candidate_id": f"synthetic/{family.key}/{action_family}/{variant}",
        "seed_id": f"synthetic/{family.key}/{action_family}/{variant}",
        "schema": SYNTHETIC_PILOT_SCHEMA,
        "split": "train" if family.key in _TRAIN_FAMILY_KEYS else "development",
        "source_type": SYNTHETIC_FUNCTIONAL_SOURCE_TYPE,
        "source_license": SYNTHETIC_FUNCTIONAL_LICENSE,
        "third_party_source": False,
        "human_chronology_observed": False,
        "history_order": SYNTHETIC_HISTORY_ORDER,
        "context_policy": CONTEXT_POLICY_VERSION,
        "state": {
            "file_id": state.file_id,
            "filetype": state.filetype,
            "source": state.source,
            "target_row": state.target_row,
            "cursor_col": state.cursor_col,
            "history": [
                {"row": item.row, "old_text": item.old_text, "new_text": item.new_text}
                for item in state.history
            ],
            "relevant": [],
        },
        "prompt": prompt,
        "context_sha256": sha256_bytes(prompt.encode("utf-8")),
        "history_sha256": canonical_sha256(
            [{"row": recent.row, "old_text": recent.old_text, "new_text": recent.new_text}]
        ),
        "history_before_sha256": sha256_bytes(before_source.encode("utf-8")),
        "action": action_mapping,
        "action_wire": encode_action(action),
        "after_source": after_source,
        "after_source_sha256": sha256_bytes(after_source.encode("utf-8")),
        "source_repo": "author-owned-synthetic",
        "source_revision": SYNTHETIC_GENERATOR_REVISION,
        "source_path": f"generated/{family.key}/{action_family}/{variant}.py",
        "source_sha256": sha256_bytes(source.encode("utf-8")),
        "source_group_id": f"author-owned-synthetic/{family.key}",
        "session_or_commit": "generator:" + SYNTHETIC_GENERATOR_REVISION,
        "task_family_id": "algorithm-family/" + family.key,
        "template_id": f"synthetic-functional-v1/{family.key}/{action_family}/{variant}",
        "near_duplicate_sha256": "",
        "authoring_metadata": {
            "source_repo": "author-owned-synthetic",
            "source_revision": SYNTHETIC_GENERATOR_REVISION,
            "source_path": f"generated/{family.key}/{action_family}/{variant}.py",
            "source_sha256": sha256_bytes(source.encode("utf-8")),
            "source_license": SYNTHETIC_FUNCTIONAL_LICENSE,
            "third_party_source": False,
            "generator_revision_sha256": SYNTHETIC_GENERATOR_REVISION,
            "algorithm_family_id": "algorithm-family/" + family.key,
            "synthetic_chronology": True,
        },
        "algorithm_family_id": "algorithm-family/" + family.key,
        "variant_id": f"{action_family}-{variant}",
        "objective_verified": False,
        "quality_evidence": False,
        "accepted_training": False,
    }
    row["near_duplicate_sha256"] = near_duplicate_key(row)
    validate_synthetic_candidate(row)
    return SyntheticCandidate(
        row=row,
        wrong_action=wrong,
        test_source=test_source,
        fixture_sha256=sha256_bytes(test_source.encode("utf-8")),
    )


def _oracle_test(expected: Any) -> str:
    encoded = json.dumps(expected, ensure_ascii=False, sort_keys=True)
    return (
        "import json\nfrom solution import solve\n\n"
        + "assert json.loads(json.dumps(solve(), sort_keys=True)) == json.loads("
        + repr(encoded)
        + ")\n"
    )


# Keep the HTML/CSV normalized-context family together. This assignment is
# fixed before authoritative OCI checks or any model outputs.
_DEVELOPMENT_FAMILY_KEYS = frozenset(
    {
        "sequence_interleaving",
        "record_key_deduplication",
        "semantic_version_ordering",
        "bounded_page_selection",
    }
)
_TRAIN_FAMILY_KEYS = frozenset(
    family.key for family in FAMILIES if family.key not in _DEVELOPMENT_FAMILY_KEYS
)


def build_synthetic_candidates() -> list[SyntheticCandidate]:
    """Freeze the 12-train/4-development family assignment and N/I/D variants."""
    candidates = [
        _candidate(family, action_family=action, variant=variant)
        for family in FAMILIES
        for action in ("N", "I", "D")
        for variant in range(SYNTHETIC_VARIANTS_PER_ACTION)
    ]
    return candidates


def validate_synthetic_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Check the fixed synthetic edit/history/action bindings without execution."""
    try:
        state = EditState.from_mapping(row["state"])
        action_raw = row["action"]
        action = EditAction(**action_raw)
        source = state.source.encode("utf-8")
        history = state.history
        prompt = row["prompt"]
        after_source = row["after_source"]
    except (KeyError, TypeError, ValueError):
        raise ValueError("synthetic functional row identity is invalid") from None
    if (
        row.get("schema") != SYNTHETIC_PILOT_SCHEMA
        or row.get("source_type") != SYNTHETIC_FUNCTIONAL_SOURCE_TYPE
        or row.get("source_license") != SYNTHETIC_FUNCTIONAL_LICENSE
        or row.get("third_party_source") is not False
        or row.get("human_chronology_observed") is not False
        or row.get("history_order") != SYNTHETIC_HISTORY_ORDER
        or row.get("context_policy") != CONTEXT_POLICY_VERSION
        or len(history) != 1
        or prompt != serialize_state(state)
        or row.get("context_sha256") != sha256_bytes(prompt.encode("utf-8"))
        or row.get("history_sha256")
        != canonical_sha256(
            [
                {
                    "row": history[0].row,
                    "old_text": history[0].old_text,
                    "new_text": history[0].new_text,
                }
            ]
        )
        or action.kind not in {"keep", "insert_before", "delete_line"}
        or row.get("action_wire") != encode_action(action)
        or apply_action(state, action) != after_source
        or row.get("after_source_sha256") != sha256_bytes(after_source.encode("utf-8"))
        or row.get("source_sha256") != sha256_bytes(source)
        or row.get("source_revision")
        != row.get("authoring_metadata", {}).get("generator_revision_sha256")
        or row.get("near_duplicate_sha256") != near_duplicate_key(row)
    ):
        raise ValueError("synthetic functional row has inconsistent context or action binding")
    if row.get("split") not in {"train", "development"}:
        raise ValueError("synthetic functional row has an invalid split")
    family_id = row.get("algorithm_family_id")
    if (
        not isinstance(family_id, str)
        or not family_id.startswith("algorithm-family/")
        or row.get("task_family_id") != family_id
        or row.get("source_group_id")
        != "author-owned-synthetic/" + family_id.removeprefix("algorithm-family/")
        or row.get("authoring_metadata", {}).get("algorithm_family_id") != family_id
    ):
        raise ValueError("synthetic functional row family grouping is inconsistent")
    state_lines = physical_lines(source)
    edit = history[0]
    if (
        edit.row >= len(state_lines)
        or state_lines[edit.row].content.decode("utf-8") != edit.new_text
    ):
        raise ValueError("synthetic functional history does not match its current source")
    before = list(state_lines)
    before[edit.row] = type(before[edit.row])(
        edit.old_text.encode("utf-8"), before[edit.row].terminator
    )
    before_source = b"".join(line.raw for line in before).decode("utf-8")
    if row.get("history_before_sha256") != sha256_bytes(before_source.encode("utf-8")):
        raise ValueError("synthetic functional history-before hash mismatch")
    return {
        "state_sha256": canonical_sha256(row["state"]),
        "action_sha256": canonical_sha256(row["action"]),
        "history_before_sha256": row["history_before_sha256"],
        "action_family": action.kind,
        "algorithm_family_id": family_id,
    }


def validate_synthetic_receipt(row: dict[str, Any], *, evaluator_sha256: str) -> None:
    receipt = row.get("synthetic_objective_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("synthetic functional row lacks its oracle receipt")
    if (
        receipt.get("schema") != SYNTHETIC_RECEIPT_SCHEMA
        or receipt.get("execution_backend") != "container"
        or receipt.get("network_access") != "none"
        or receipt.get("gold_parse") != "pass"
        or receipt.get("gold_compile") != "pass"
        or receipt.get("gold_test") != "pass"
        or receipt.get("wrong_parse") != "pass"
        or receipt.get("wrong_compile") != "pass"
        or receipt.get("wrong_test") != "fail"
        or receipt.get("gold_action_sha256") != canonical_sha256(row["action"])
        or receipt.get("state_sha256") != canonical_sha256(row["state"])
        or receipt.get("fixture_sha256") is None
        or receipt.get("evaluator_sha256") != evaluator_sha256
        or receipt.get("image_identity") is None
    ):
        raise ValueError("synthetic functional row oracle receipt is invalid")


def validate_synthetic_family_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require all variants of one algorithm family to remain in one split."""
    family_splits: dict[str, str] = {}
    family_actions: dict[str, dict[str, set[str]]] = {}
    states: set[str] = set()
    inputs: set[str] = set()
    ids: set[str] = set()
    for row in rows:
        validate_synthetic_candidate(row)
        identifier = row.get("candidate_id")
        family_id = row["algorithm_family_id"]
        split = row["split"]
        action = row["action"]
        action_name = {"keep": "N", "insert_before": "I", "delete_line": "D"}[action["kind"]]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("synthetic functional candidate IDs are duplicated")
        ids.add(identifier)
        prior = family_splits.setdefault(family_id, split)
        if prior != split:
            raise ValueError("synthetic algorithm family crosses train/development")
        family_actions.setdefault(family_id, {name: set() for name in ("N", "I", "D")})
        state_key = canonical_sha256(row["state"])
        normalized_state = {
            key: row["state"][key]
            for key in ("filetype", "source", "target_row", "cursor_col", "history", "relevant")
        }
        input_key = canonical_sha256(normalized_state)
        if state_key in states or input_key in inputs:
            raise ValueError("synthetic functional model input is duplicated")
        states.add(state_key)
        inputs.add(input_key)
        variant = row.get("variant_id")
        if not isinstance(variant, str) or variant in family_actions[family_id][action_name]:
            raise ValueError("synthetic family/action variants are missing or duplicated")
        family_actions[family_id][action_name].add(variant)
    expected_families = {
        "train": _TRAIN_FAMILY_KEYS,
        "development": _DEVELOPMENT_FAMILY_KEYS,
    }
    for family_id, split in family_splits.items():
        family_key = family_id.removeprefix("algorithm-family/")
        if family_key not in expected_families[split]:
            raise ValueError("synthetic family was assigned to a non-frozen split")
        for action in ("N", "I", "D"):
            if len(family_actions[family_id][action]) != SYNTHETIC_VARIANTS_PER_ACTION:
                raise ValueError("synthetic algorithm family lacks its fixed action variants")
    if family_splits != {
        **{"algorithm-family/" + key: "train" for key in _TRAIN_FAMILY_KEYS},
        **{"algorithm-family/" + key: "development" for key in _DEVELOPMENT_FAMILY_KEYS},
    }:
        raise ValueError("synthetic algorithm-family split inventory is incomplete")
    return {
        "family_count": len(family_splits),
        "families_by_split": {
            "train": len(_TRAIN_FAMILY_KEYS),
            "development": len(_DEVELOPMENT_FAMILY_KEYS),
        },
        "rows": len(rows),
        "action_variants_per_family": SYNTHETIC_VARIANTS_PER_ACTION,
    }
