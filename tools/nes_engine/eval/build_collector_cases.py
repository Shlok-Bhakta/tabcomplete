"""Build next-edit evaluation cases from the trajectory collector (READ-ONLY).

Each file's buffer is rebuilt from its anchors (buffer_open/buffer_write
blobs) and ordered edit_delta events, verifying every delta's deleted text
against the running buffer (same rules as analysis/replay.ts). Typing is split
into bursts at pauses. A case is the state at the end of one burst (with the
earlier bursts as edit history) and the target is the state after the next
burst: "what did the user do next".

Usage:
    python3 -I build_collector_cases.py --out cases.jsonl [--db PATH]
"""

import argparse
import gzip
import json
import sqlite3
import sys

PAUSE_MS = 1200
MAX_TARGET_CHANGED_LINES = 12
EXCLUDE = ("/tmp/", "smoke", "disposable", "tabcomplete", "trajectory-")


def blob_text(db, sha):
    row = db.execute("SELECT content, compression FROM blobs WHERE sha256 = ?", (sha,)).fetchone()
    if row is None:
        return None
    data, compression = row
    if compression == "gzip":
        data = gzip.decompress(data)
    return data.decode("utf-8")


def split(text):
    trailing = text.endswith("\n")
    body = text[:-1] if trailing else text
    return ([] if body == "" else body.split("\n")), trailing


def join(lines, trailing):
    return "\n".join(lines) + ("\n" if trailing else "")


def changed_lines(a, b):
    import difflib

    sm = difflib.SequenceMatcher(None, a.split("\n"), b.split("\n"), autojunk=False)
    n = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            n += max(i2 - i1, j2 - j1)
    return n


def file_events(db, path):
    rows = db.execute(
        """SELECT timestamp_ms, sequence_number, session_id, event_type, payload_json FROM events
           WHERE event_type IN ('buffer_open','buffer_write','edit_delta')
             AND json_extract(payload_json, '$.path') = ?
           ORDER BY timestamp_ms ASC, sequence_number ASC""",
        (path,),
    ).fetchall()
    return [(ts, seq, sess, et, json.loads(p)) for ts, seq, sess, et, p in rows]


def timeline(db, path):
    """Yield (ts, content, cursor) after every verified delta."""
    lines, trailing, valid = None, True, False
    out = []
    stats = {"deltas": 0, "mismatch": 0, "resync": 0}
    for ts, _seq, _sess, et, p in file_events(db, path):
        if et in ("buffer_open", "buffer_write"):
            content = blob_text(db, p.get("content_hash") or "")
            if content is None:
                continue
            new_lines, new_trailing = split(content)
            if lines is None or not valid or lines != new_lines:
                if lines is not None:
                    stats["resync"] += 1
                lines, trailing, valid = new_lines, new_trailing, True
                out.append((ts, join(lines, trailing), None, "anchor"))
            continue
        if lines is None or not valid:
            continue
        stats["deltas"] += 1
        actual = "\n".join(lines[p["start_row"] : p["old_end_row"]])
        if actual != p["deleted_text"]:
            stats["mismatch"] += 1
            valid = False
            continue
        n = p["new_end_row"] - p["start_row"]
        ins = [""] * n if p["inserted_text"] == "" else p["inserted_text"].split("\n")
        if len(ins) != n:
            stats["mismatch"] += 1
            valid = False
            continue
        lines[p["start_row"] : p["old_end_row"]] = ins
        out.append((ts, join(lines, trailing), p.get("cursor_after"), "delta"))
    return out, stats


def reverse_timeline(db, path, final_text):
    """Rebuild a file's history backwards from its current content when the
    anchor blobs were never uploaded. Each delta's inserted text must match
    the buffer before it is undone; anchor SHA-256s confirm the result."""
    import hashlib

    events = file_events(db, path)
    deltas = [e for e in events if e[3] == "edit_delta"]
    lines, trailing = split(final_text)
    states = []  # (ts, content, cursor) after each delta, newest first
    for ts, _seq, _sess, _et, p in reversed(deltas):
        n = p["new_end_row"] - p["start_row"]
        ins = [""] * n if p["inserted_text"] == "" else p["inserted_text"].split("\n")
        if len(ins) != n or lines[p["start_row"] : p["new_end_row"]] != ins:
            break
        states.append((ts, join(lines, trailing), p.get("cursor_after")))
        m = p["old_end_row"] - p["start_row"]
        dele = [""] * m if p["deleted_text"] == "" else p["deleted_text"].split("\n")
        lines[p["start_row"] : p["new_end_row"]] = dele
    states.reverse()
    by_ts = {}
    for ts, content, _ in states:
        by_ts[ts] = content
    checked = matched = 0
    known = sorted(by_ts)
    for ts, _seq, _sess, et, p in events:
        if et not in ("buffer_open", "buffer_write") or not known:
            continue
        prior = [t for t in known if t <= ts]
        if not prior:
            continue
        checked += 1
        content = by_ts[prior[-1]]
        if hashlib.sha256(content.encode("utf-8")).hexdigest() == p.get("content_hash"):
            matched += 1
    out = [(ts, content, cursor, "delta") for ts, content, cursor in states]
    return out, {"deltas": len(deltas), "recovered": len(states), "anchors_checked": checked, "anchors_matched": matched}


def bursts(tl):
    """Group consecutive deltas separated by less than PAUSE_MS."""
    groups, current = [], []
    for item in tl:
        if item[3] == "anchor":
            if current:
                groups.append(current)
            groups.append([item])
            current = []
            continue
        if current and item[0] - current[-1][0] >= PAUSE_MS:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/mnt/ssd/collector-data/collector.sqlite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mirror", default="", help="local root holding current copies of edited files")
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    paths = [
        r[0]
        for r in db.execute(
            "SELECT DISTINCT json_extract(payload_json,'$.path') FROM events WHERE event_type='edit_delta'"
        )
        if r[0] and not any(x in r[0] for x in EXCLUDE)
    ]
    cases = []
    for path in paths:
        tl, stats = timeline(db, path)
        if stats["deltas"] == 0 and args.mirror:
            local = args.mirror + path
            try:
                final_text = open(local, encoding="utf-8").read()
            except OSError:
                final_text = None
            if final_text is not None:
                tl, stats = reverse_timeline(db, path, final_text)
                # Keep only histories that the anchors confirm.
                if stats["anchors_checked"] == 0 or stats["anchors_matched"] * 2 < stats["anchors_checked"]:
                    print(f"{path}: reverse rebuild not confirmed {stats}", file=sys.stderr)
                    tl = []
                elif tl:
                    tl.insert(0, (tl[0][0] - 1, tl[0][1], None, "anchor"))
        groups = bursts(tl)
        for k in range(len(groups) - 1):
            cur, nxt = groups[k], groups[k + 1]
            if cur[-1][3] == "anchor" or nxt[0][3] == "anchor":
                continue
            state, target = cur[-1][1], nxt[-1][1]
            if state == target:
                continue
            n = changed_lines(state, target)
            if n == 0 or n > MAX_TARGET_CHANGED_LINES:
                continue
            if state.split() == target.split():
                continue  # whitespace-only
            cursor = cur[-1][2] or {"row": 0, "col": 0}
            # Where the user ended up shortly after: the next few burst ends
            # within a minute, stopping at the next anchor.
            future = []
            for g in groups[k + 1 : k + 6]:
                if g[0][3] == "anchor" or g[-1][0] - cur[-1][0] > 60_000:
                    break
                future.append(g[-1][1])
            # History: the last anchor before this burst, then each burst end.
            history = []
            start = max(i for i in range(k + 1) if groups[i][0][3] == "anchor") if any(
                groups[i][0][3] == "anchor" for i in range(k + 1)
            ) else 0
            for g in groups[start:k]:
                last = g[-1]
                history.append({"ts_ms": last[0], "content": last[1], "cursor": last[2]})
            # The last keystrokes of this burst, for keystroke-level replay.
            pre_states = [
                {"ts_ms": item[0], "content": item[1], "cursor": item[2]}
                for item in cur[-4:-1]
                if item[3] == "delta"
            ]
            cases.append(
                {
                    "id": f"{path.rsplit('/', 1)[-1]}@{cur[-1][0]}",
                    "pre_states": pre_states,
                    "path": path,
                    "history": history[-40:],
                    "ts_ms": cur[-1][0],
                    "content": state,
                    "cursor": {"line": cursor["row"], "col": cursor["col"]},
                    "target": target,
                    "future": future,
                    "target_changed_lines": n,
                }
            )
        print(f"{path}: {stats} cases={sum(1 for c in cases if c['path'] == path)}", file=sys.stderr)
    with open(args.out, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
