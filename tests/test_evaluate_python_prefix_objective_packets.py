from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from tinycomplete.one_line.contract import EditState, RecentEdit

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_python_prefix_objective_packets.py"
SPEC = importlib.util.spec_from_file_location("objective_packet_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packets)


def test_synthetic_history_reverses_and_replays_exact_prestate() -> None:
    state = EditState(
        "sample.py",
        "python",
        "def value():\n    return",
        1,
        len("    return"),
        (RecentEdit(1, "    ", "    return"),),
    )

    packets._verify_history_replay(state)


def test_synthetic_history_rejects_mismatched_observed_delta() -> None:
    state = EditState(
        "sample.py",
        "python",
        "def value():\n    return",
        1,
        len("    return"),
        (RecentEdit(1, "    ", "    return None"),),
    )

    with pytest.raises(packets.ObjectivePacketError, match="history_reverse_replay_mismatch"):
        packets._verify_history_replay(state)


def test_function_slice_is_exactly_the_target_enclosing_function() -> None:
    source = "def first():\n    return 1\n\ndef second():\n    return 2\n"

    extracted, identity = packets._extract_function(source, 4)

    assert extracted == "def second():\n    return 2\n"
    assert identity["function_name"] == "second"
    assert identity["source_sha256"] == hashlib.sha256(extracted.encode()).hexdigest()


def test_source_artifact_loader_rejects_symlink_and_hash_mismatch(tmp_path, monkeypatch) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    external = tmp_path / "outside"
    external.write_bytes(b"public source")
    (package_root / "linked.src").symlink_to(external)
    monkeypatch.setattr(packets, "PACKAGE_ROOT", package_root)

    with pytest.raises(packets.ObjectivePacketError, match="source_artifact_not_regular"):
        packets._load_bound_artifact(
            {
                "path": "linked.src",
                "bytes": external.stat().st_size,
                "sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
            }
        )

    (package_root / "source.src").write_bytes(b"changed")
    with pytest.raises(packets.ObjectivePacketError, match="source_artifact_identity_mismatch"):
        packets._load_bound_artifact({"path": "source.src", "bytes": 4, "sha256": "0" * 64})
