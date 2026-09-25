"""Generate grouped, replay-checked synthetic next-edit trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

LANGUAGES = ("python", "javascript", "go", "rust")
ACTIONS = ("replace", "insert", "delete", "no_edit")
EXTENSIONS = {"python": "py", "javascript": "js", "go": "go", "rust": "rs"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def replace_once(text: str, old: str, new: str) -> tuple[str, int, int]:
    start = text.index(old)
    assert text.count(old) == 1
    raw_start = len(text[:start].encode())
    raw_end = raw_start + len(old.encode())
    return text[:start] + new + text[start + len(old) :], raw_start, raw_end


def splice(text: str, start: int, end: int, replacement: str) -> str:
    raw = text.encode()
    return (raw[:start] + replacement.encode() + raw[end:]).decode()


def lines(language: str, name: str, number: int) -> dict[str, str]:
    if language == "python":
        return {
            "header": f"def {name}(item):\n",
            "assign_value": f"    value = item + {number}\n",
            "assign_total": f"    total = item + {number}\n",
            "return_value": "    return value\n",
            "return_total": "    return total\n",
            "debug_a": "    print('debug a', item)\n",
            "debug_b": "    print('debug b', item)\n",
            "close": "",
            "check_a": f"assert {name}(1) == {number + 1}\n",
            "check_b": f"assert {name}(2) == {number + 2}\n",
        }
    if language == "javascript":
        return {
            "header": f"function {name}(item) {{\n",
            "assign_value": f"  let value = item + {number};\n",
            "assign_total": f"  let total = item + {number};\n",
            "return_value": "  return value;\n",
            "return_total": "  return total;\n",
            "debug_a": "  console.log('debug a', item);\n",
            "debug_b": "  console.log('debug b', item);\n",
            "close": "}\n",
            "check_a": f"if ({name}(1) !== {number + 1}) throw Error('a');\n",
            "check_b": f"if ({name}(2) !== {number + 2}) throw Error('b');\n",
        }
    if language == "go":
        return {
            "header": f"package main\nfunc {name}(item int) int {{\n",
            "assign_value": f"  value := item + {number}\n",
            "assign_total": f"  total := item + {number}\n",
            "return_value": "  return value\n",
            "return_total": "  return total\n",
            "debug_a": '  println("debug a", item)\n',
            "debug_b": '  println("debug b", item)\n',
            "close": "}\n",
            "check_a": f"// checked {name}(1) == {number + 1}\n",
            "check_b": f"// checked {name}(2) == {number + 2}\n",
        }
    return {
        "header": f"fn {name}(item: i32) -> i32 {{\n",
        "assign_value": f"  let value = item + {number};\n",
        "assign_total": f"  let total = item + {number};\n",
        "return_value": "  value\n",
        "return_total": "  total\n",
        "debug_a": '  println!("debug a {}", item);\n',
        "debug_b": '  println!("debug b {}", item);\n',
        "close": "}\n",
        "check_a": f"// checked {name}(1) == {number + 1}\n",
        "check_b": f"// checked {name}(2) == {number + 2}\n",
    }


def example(language: str, action: str, family: int, variant: int) -> dict:
    number = 11 + family * 64 + variant
    name = f"compute_{family}_{variant}"
    parts = lines(language, name, number)
    header = parts["header"]
    tail = parts["close"]
    if action == "replace":
        history_before = header + parts["assign_value"] + parts["return_value"] + tail
        old_assignment = "value :=" if language == "go" else "value ="
        new_assignment = "total :=" if language == "go" else "total ="
        current, previous_start, previous_end = replace_once(
            history_before, old_assignment, new_assignment
        )
        marker = "return value" if language != "rust" else "  value\n"
        target = "return total" if language != "rust" else "  total\n"
        after, region_start, region_end = replace_once(current, marker, target)
        replacement = target
    elif action == "delete":
        history_before = (
            header
            + parts["assign_value"]
            + parts["debug_a"]
            + parts["debug_b"]
            + parts["return_value"]
            + tail
        )
        current, previous_start, previous_end = replace_once(history_before, parts["debug_a"], "")
        after, region_start, region_end = replace_once(current, parts["debug_b"], "")
        replacement = ""
    elif action == "insert":
        history_before = header + parts["assign_value"] + parts["return_value"] + tail
        previous_start = len(history_before.encode())
        previous_end = previous_start
        current = history_before + parts["check_a"]
        region_start = len(current.encode())
        region_end = region_start
        replacement = parts["check_b"]
        after = current + replacement
    else:
        history_before = header + parts["assign_total"] + parts["return_value"] + tail
        marker = "return value" if language != "rust" else "  value\n"
        target = "return total" if language != "rust" else "  total\n"
        current, previous_start, previous_end = replace_once(history_before, marker, target)
        after = current
        region_start = len(current[: current.index(target)].encode())
        region_end = region_start + len(target.encode())
        replacement = ""

    previous_text = (
        current.encode()[previous_start:].decode()
        if action == "insert"
        else ("" if action == "delete" else (new_assignment if action == "replace" else target))
    )
    assert splice(history_before, previous_start, previous_end, previous_text) == current
    assert (
        splice(current, region_start, region_end, replacement) == after
        if action != "no_edit"
        else current == after
    )
    assert current != history_before
    if action != "no_edit":
        assert after != current
    path = f"synthetic/{language}/family_{family}/{name}.{EXTENSIONS[language]}"
    raw = current.encode()
    before = raw[:region_start].decode()
    region = raw[region_start:region_end].decode()
    suffix = raw[region_end:].decode()
    history = (
        f"<synthetic-recent-edit start={previous_start} end={previous_end}>\n"
        f"{previous_text}\n</synthetic-recent-edit>\n"
    )
    prompt = (
        f"<repo {path}>\n<filetype {language}>\n"
        + history
        + f"<file {path}>\n{before}[[EDIT]]{region}[[/EDIT]]{suffix}\n</file>\n"
        + f"<P {path} {region_start}>\n"
        + "Return one compact next-edit action: N\\n for no edit or R\\n "
        + "followed by exact replacement text. End with EOS.\n"
    )
    response = "N\n" if action == "no_edit" else "R\n" + replacement
    split = "train" if family < 4 else ("development" if family == 4 else "heldout")
    return {
        "id": f"{language}-{action}-{family}-{variant}",
        "split": split,
        "group": f"synthetic/{language}/{action}/family-{family}",
        "source": "deterministic synthetic trajectory",
        "language": language,
        "path": path,
        "action": action,
        "history_before": history_before,
        "history_start": previous_start,
        "history_end": previous_end,
        "history_replacement": previous_text,
        "current": current,
        "current_sha256": sha(current),
        "region_start": region_start,
        "region_end": region_end,
        "after": after,
        "after_sha256": sha(after),
        "target_text": replacement,
        "prompt": prompt,
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "development", "heldout"):
        (args.output / (split + ".jsonl")).unlink(missing_ok=True)
    counts: Counter[str] = Counter()
    seen = set()
    for language in LANGUAGES:
        for action in ACTIONS:
            for family in range(6):
                for variant in range(64):
                    row = example(language, action, family, variant)
                    if row["current_sha256"] in seen:
                        raise ValueError("duplicate pre-edit state")
                    seen.add(row["current_sha256"])
                    counts[row["split"]] += 1
                    with (args.output / (row["split"] + ".jsonl")).open("a") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "generator": "build_small_edit_data.py",
        "groups": 96,
        "counts": dict(counts),
        "sha256": {
            split: hashlib.sha256((args.output / (split + ".jsonl")).read_bytes()).hexdigest()
            for split in counts
        },
        "heldout_is_separate": True,
        "source": "synthetic; no personal or employer repository content",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
