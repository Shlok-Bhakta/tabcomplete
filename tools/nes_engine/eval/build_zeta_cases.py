"""Convert Zed's public Zeta next-edit dataset into replay cases.

Each Zeta example has recent edit events (unified diffs), an excerpt with an
editable region and cursor marker, and the expected rewritten excerpt. Earlier
file states are rebuilt by reverse-applying the events to the excerpt, so the
engine derives the same edit history itself.

Usage:
    python3 -I build_zeta_cases.py --out zeta_cases.jsonl eval.jsonl train.jsonl
"""

import argparse
import json
import re
import sys

EVENT_RE = re.compile(r'User edited (?:file: )?"?([^"\n]+?)"?:?\n+```diff\n(.*?)```', re.S)
FENCE_RE = re.compile(r"```([^\n]*)\n(.*)```", re.S)


def strip_markers(text):
    return text.replace("<|editable_region_start|>\n", "").replace("<|editable_region_end|>\n", "").replace(
        "<|editable_region_start|>", "").replace("<|editable_region_end|>", "")


def parse_fence(block):
    m = FENCE_RE.search(block)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2)


def hunk_texts(diff):
    """(old, new) text per hunk, newline-joined."""
    out, old, new, active = [], [], [], False
    for line in diff.split("\n"):
        if line.startswith("@@"):
            if active:
                out.append(("\n".join(old), "\n".join(new)))
            old, new, active = [], [], True
            continue
        if not active:
            continue
        if line.startswith("-"):
            old.append(line[1:])
        elif line.startswith("+"):
            new.append(line[1:])
        elif line.startswith(" ") or line == "":
            old.append(line[1:] if line else "")
            new.append(line[1:] if line else "")
        elif line.startswith("\\"):
            continue
    if active:
        out.append(("\n".join(old), "\n".join(new)))
    return out


def unapply(content, diff):
    """Reverse one event on the excerpt, or None if it does not match."""
    for old, new in reversed(hunk_texts(diff)):
        new_s, old_s = new.rstrip("\n"), old.rstrip("\n")
        if new_s and new_s in content:
            content = content.replace(new_s, old_s, 1)
        else:
            return None
    return content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    cases, skipped = [], 0
    for path in args.inputs:
        split_name = path.rsplit("/", 1)[-1].split(".")[0]
        for i, line in enumerate(open(path)):
            row = json.loads(line)
            name, excerpt = parse_fence(row["input"])
            _, expected = parse_fence(row["output"])
            if excerpt is None or expected is None or "<|user_cursor_is_here|>" not in excerpt:
                skipped += 1
                continue
            cursor_offset = strip_markers(excerpt).index("<|user_cursor_is_here|>")
            content = strip_markers(excerpt).replace("<|user_cursor_is_here|>", "")
            target = strip_markers(expected).replace("<|user_cursor_is_here|>", "")
            if content == target:
                skipped += 1
                continue
            events = EVENT_RE.findall(row.get("events", ""))
            states = [content]
            for _file, diff in reversed(events):
                prev = unapply(states[0], diff)
                if prev is None or prev == states[0]:
                    break
                states.insert(0, prev)
            before = content[:cursor_offset]
            cursor = {"line": before.count("\n"), "col": len(before.rsplit("\n", 1)[-1].encode("utf-8"))}
            file_path = f"/zeta/{split_name}-{i:04d}/{name or 'file.txt'}"
            base_ts = 1_000_000
            history = [
                {"ts_ms": base_ts + 5000 * k, "content": s, "cursor": None} for k, s in enumerate(states[:-1])
            ]
            cases.append({
                "id": f"zeta-{split_name}-{i:04d}",
                "path": file_path,
                "history": history,
                "ts_ms": base_ts + 5000 * len(states),
                "content": content,
                "cursor": cursor,
                "target": target,
                "events_used": len(states) - 1,
                "events_total": len(events),
            })
    with open(args.out, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cases)} cases, skipped {skipped}; events rebuilt "
          f"{sum(c['events_used'] for c in cases)}/{sum(c['events_total'] for c in cases)}", file=sys.stderr)


if __name__ == "__main__":
    main()
