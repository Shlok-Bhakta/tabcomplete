"""FIM + next-edit data: determinism, hole coverage, tree-sitter alignment."""

import pytest

from tinycomplete.data.fim import (
    HOLE_TYPES,
    format_psm,
    format_spm,
    generate_examples,
    make_example,
    make_hole,
)
from tinycomplete.data.schema import FIM_MIDDLE
from tinycomplete.data.static_edits import generate_next_edits, make_next_edit

FIXTURE = '''"""Fixture module for FIM hole tests."""
import os
import sys
from pathlib import Path

CONSTANT = 42


def add(a, b):
    total = a + b
    return total


def greet(name):
    if name:
        return "hi " + name
    return "hi"
'''


def test_all_hole_types_cover_source():
    for hole_type in HOLE_TYPES:
        hole = make_hole(FIXTURE, hole_type, seed=7)
        assert hole.prefix + hole.middle + hole.suffix == FIXTURE
        assert hole.start < hole.end
        FIXTURE.encode("utf-8")[: hole.start].decode("utf-8")  # boundary ok
        FIXTURE.encode("utf-8")[: hole.end].decode("utf-8")


def test_determinism_byte_identical():
    for hole_type in HOLE_TYPES:
        first = make_hole(FIXTURE, hole_type, seed=1234)
        second = make_hole(FIXTURE, hole_type, seed=1234)
        assert first == second
        assert first.middle.encode("utf-8") == second.middle.encode("utf-8")
    assert make_next_edit(FIXTURE, 99) == make_next_edit(FIXTURE, 99)
    assert generate_examples(FIXTURE, 8, 5) == generate_examples(FIXTURE, 8, 5)


def test_psm_spm_format():
    hole = make_hole(FIXTURE, "identifier", seed=3)
    psm = format_psm(hole.prefix, hole.suffix)
    spm = format_spm(hole.prefix, hole.suffix)
    assert psm == f"<|fim_prefix|>{hole.prefix}<|fim_suffix|>{hole.suffix}{FIM_MIDDLE}"
    assert spm == f"<|fim_suffix|>{hole.suffix}<|fim_prefix|>{hole.prefix}{FIM_MIDDLE}"
    # live prefix sits immediately before generation in SPM
    assert spm.endswith(f"{hole.prefix}{FIM_MIDDLE}")
    ex = make_example(FIXTURE, "identifier", seed=3, mode="spm")
    assert ex.target == hole.middle
    assert ex.input_text == spm


def test_tree_sitter_alignment():
    from tree_sitter_language_pack import get_parser

    parser = get_parser("python")
    tree = parser.parse(FIXTURE.encode("utf-8"))

    found = {}

    def walk(node):
        yield node
        for child in node.children:
            yield from walk(child)

    nodes = list(walk(tree.root_node))
    for hole_type in ("identifier", "expression", "statement", "function_body", "import", "block"):
        hole = make_hole(FIXTURE, hole_type, seed=11)
        if hole.hole_type != hole_type:
            continue  # fell back to span; grammar had no candidate
        found[hole_type] = any(
            n.start_byte == hole.start and n.end_byte == hole.end for n in nodes
        )
    # A rich fixture must align for the core syntactic types.
    assert found.get("identifier"), "identifier hole not on a syntax node"
    assert found.get("import"), "import hole not on a syntax node"
    assert found.get("function_body"), "function body hole not on a block node"


def test_function_body_and_block_are_multiline():
    hole = make_hole(FIXTURE, "function_body", seed=11)
    assert "\n" in hole.middle
    assert "return" in hole.middle


def test_next_edit_region_and_noop():
    seen_noop = False
    for seed in range(30):
        ex = make_next_edit(FIXTURE, seed)
        body = ex.input_text.split("<file", 1)[1].split(">", 1)[1].rsplit("</file>", 1)[0]
        assert body.startswith("\n") and body.endswith("\n")
        body = body[1:-1]
        marked = body.split("[[EDIT]]", 1)[1].split("[[/EDIT]]", 1)[0]
        current = body.replace("[[EDIT]]", "").replace("[[/EDIT]]", "")
        assert current.encode("utf-8")[ex.region_start : ex.region_end].decode("utf-8") == marked
        assert ex.input_text.splitlines()[-1].startswith("<P ")
        assert len(ex.recent_edits) >= 1
        if ex.action == "noop":
            assert ex.target == ""
            seen_noop = True
        else:
            assert ex.target != ""
    assert seen_noop, "expected at least one NO_EDIT target in 30 seeds"


def test_generate_counts_and_modes():
    examples = generate_examples(FIXTURE, 16, seed=1)
    assert len(examples) == 16
    assert {e.mode for e in examples} == {"psm", "spm"}
    assert all(e.provenance.value == "static_fim" for e in examples)
    edits = generate_next_edits(FIXTURE, 4, seed=2)
    assert len(edits) == 4
    assert all(e.mode == "next_edit" for e in edits)


def test_empty_source_rejected():
    with pytest.raises(ValueError):
        make_hole("", "span", seed=0)
    with pytest.raises(ValueError):
        make_next_edit("", seed=0)
    with pytest.raises(ValueError):
        make_hole(FIXTURE, "bogus", seed=0)
