import copy
import hashlib
import re

from tinycomplete.one_line.author_protocol_v3 import (
    AUTHOR_SCHEMA_V3,
    OBJECTIVE_KIND_PATTERN,
    SYSTEM_INSTRUCTION,
    build_author_prompt_v3,
)
from tinycomplete.one_line.pilot_roles import AUTHOR_SCHEMA, build_author_prompt_v2


def test_new_schema_matches_existing_parser_and_does_not_mutate_history():
    old = copy.deepcopy(AUTHOR_SCHEMA)
    kind = AUTHOR_SCHEMA_V3["properties"]["objective"]["properties"]["kind"]
    assert kind["pattern"] == OBJECTIVE_KIND_PATTERN
    for value in ("fix", "local_rename", "a" + "_" * 39):
        assert re.fullmatch(OBJECTIVE_KIND_PATTERN, value)
    for value in ("", "Fix", "local rename", "local-rename", "a" * 41, "fix\n"):
        assert re.fullmatch(OBJECTIVE_KIND_PATTERN, value) is None
    assert AUTHOR_SCHEMA_V3["properties"]["action"]["properties"]["text"]["type"] == [
        "string",
        "null",
    ]
    assert AUTHOR_SCHEMA == old
    assert "No spaces, hyphens or uppercase" in SYSTEM_INSTRUCTION


def test_new_prompt_preserves_zero_based_source_framing_and_states_evidence_limits():
    source = "def count(items):\n    return len(items)\n"
    row = {
        "student_state_seed": {"file_id": "count.py", "filetype": "python", "source": source},
        "authoring_metadata": {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_license": "MIT",
            "source_provenance_verified": True,
            "authoring_focus": "local API change",
        },
    }
    assert build_author_prompt_v3(row).endswith(build_author_prompt_v2(row))
    assert "1,024 tokenizer tokens" in build_author_prompt_v3(row)
    assert "later whole-sequence validation" in build_author_prompt_v3(row)
