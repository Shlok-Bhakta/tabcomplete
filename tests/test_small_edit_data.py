import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_small_edit_data import ACTIONS, LANGUAGES, example, splice  # noqa: E402


def test_synthetic_history_and_target_reconstruct_exactly() -> None:
    for language in LANGUAGES:
        for action in ACTIONS:
            row = example(language, action, 0, 0)
            assert (
                splice(
                    row["history_before"],
                    row["history_start"],
                    row["history_end"],
                    row["history_replacement"],
                )
                == row["current"]
            )
            if action == "no_edit":
                assert row["current"] == row["after"]
                assert row["response"] == "N\n"
            else:
                assert (
                    splice(
                        row["current"], row["region_start"], row["region_end"], row["target_text"]
                    )
                    == row["after"]
                )
                assert row["response"] == "R\n" + row["target_text"]
            assert "<synthetic-recent-edit" in row["prompt"]
