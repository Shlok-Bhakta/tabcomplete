"""Bounded, single-slot native replay on the identified inference host."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import socket
import subprocess
import threading
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tinycomplete.observability.context import current_run_context


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proc_memory(pid: int) -> dict[str, int]:
    result = {}
    for source in (f"/proc/{pid}/smaps_rollup", f"/proc/{pid}/status"):
        try:
            for line in Path(source).read_text().splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key in (
                    "Rss",
                    "Pss",
                    "Pss_Anon",
                    "Pss_File",
                    "Anonymous",
                    "Swap",
                    "VmSwap",
                    "VmHWM",
                ):
                    parts = value.strip().split()
                    result[key + ("_status" if source.endswith("status") else "")] = (
                        int(parts[0]) * 1024
                    )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def counters() -> dict[str, int]:
    wanted = {"pswpin", "pswpout", "pgmajfault"}
    return {
        parts[0]: int(parts[1])
        for line in Path("/proc/vmstat").read_text().splitlines()
        if (parts := line.split()) and parts[0] in wanted
    }


def pressure() -> dict[str, str]:
    return {name: Path(f"/proc/pressure/{name}").read_text().strip() for name in ("memory", "io")}


def source_states() -> list[dict]:
    """Twelve ordered editor states, including a file switch and return."""
    rows = []
    for target, count in ((512, 29), (1024, 58), (2048, 116)):
        base = "".join(f"def f_{i}(value):\n    return value + {i}\n" for i in range(count))
        a = "# file: src/example.py\n" + base
        b = "# file: tests/other.py\n" + base
        edited_a = a.replace("return value + 1", "return value + 2", 1)
        tail = "def compute(value):\n    return "
        if target == 512:
            states = [
                ("fresh_open", "src/example.py", a + tail),
                ("append_chars", "src/example.py", a + tail + "value + "),
                ("replace_near_cursor", "src/example.py", a + tail + "value * "),
                ("reject_then_type_different", "src/example.py", a + tail + "value - "),
            ]
        elif target == 1024:
            states = [
                ("edit_earlier_in_file", "src/example.py", edited_a + tail + "value - "),
                ("switch_to_other_file", "tests/other.py", b + tail + "value * "),
                ("return_to_first_file", "src/example.py", edited_a + tail + "value - "),
                ("append_after_return", "src/example.py", edited_a + tail + "value - 1"),
            ]
        else:
            states = [
                ("fresh_open", "src/example.py", a + tail),
                ("replace_near_cursor", "src/example.py", a + tail + "value * "),
                ("reject_then_type_different", "src/example.py", a + tail + "value - "),
                ("edit_earlier_in_file", "src/example.py", edited_a + tail + "value - "),
            ]
        for index, (operation, file_id, prompt) in enumerate(states):
            rows.append(
                {
                    "id": f"synthetic-{target}-{index}",
                    "source_target": target,
                    "operation": operation,
                    "file_id": file_id,
                    "prompt": prompt,
                    "state_sha256": sha(prompt.encode()),
                }
            )
    assert len(rows) == 12
    return rows


def generate(url: str, prompt: str, *, cache_prompt: bool, max_new_tokens: int = 24) -> dict:
    # Shared provider owns SSE parsing, timing and one bounded model span.
    from measure_r2_local import NativeProvider

    provider = NativeProvider(url, "q25-coder-selected-native")
    provider.cache = cache_prompt
    provider.timeout_seconds = 120
    result = provider.generate_detailed(prompt, max_new_tokens)
    last = provider.last
    return {
        "response": result.text,
        "response_sha256": sha(result.text.encode()),
        "output_token_ids": last["token_ids"],
        "token_arrival_seconds": last["token_arrival_seconds"],
        "completed_seconds": last["total_seconds"],
        "tokens_cached": last["tokens_cached"],
        "tokens_evaluated": last["tokens_evaluated"],
        "stop_type": result.finish_reason,
        "truncated": last["truncated"],
        "generated_tokens": result.tokens,
        "server_timings": last["server_timings"],
    }


def load_frozen_states(path: Path, expected_sha256: str) -> list[dict]:
    if file_sha(path) != expected_sha256:
        raise ValueError("editor replay state fingerprint mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("editor replay states must have unique IDs")
    for row in rows:
        if not isinstance(row.get("prompt"), str) or row.get("state_sha256") != sha(
            row["prompt"].encode()
        ):
            raise ValueError("editor replay prompt hash mismatch")
        if not all(key in row for key in ("operation", "file_id", "source_target")):
            raise ValueError("editor replay state metadata is incomplete")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19094)
    parser.add_argument("--diagnostic-saved-idle-cache", action="store_true")
    parser.add_argument("--states", type=Path, help="optional frozen trained-format editor states")
    parser.add_argument("--states-sha256")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--rust-url", help="owned resident Rust engine, never an external API")
    parser.add_argument("--ssh-host", help="configured inference host for process measurements")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--setting", help="frozen plan setting identity")
    args = parser.parse_args()
    if args.rust_url:
        measure_rust(args)
        return
    if args.model is None or args.runtime is None:
        parser.error("legacy replay requires --model and --runtime")
    if (args.states is None) != (args.states_sha256 is None):
        raise ValueError("editor replay requires both states and their fingerprint")
    if not 1 <= args.max_new_tokens <= 96:
        raise ValueError("editor replay output cap must be within 1..96")
    states = load_frozen_states(args.states, args.states_sha256) if args.states else source_states()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "measurements.jsonl").exists():
        raise FileExistsError(
            "replay output already exists; use a new suite or condition directory"
        )
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if (
        subprocess.check_output(
            ["git", "-C", str(args.runtime), "rev-parse", "HEAD"], text=True
        ).strip()
        != "f072b103714dfa1eee531f80b24512faf38e3dd2"
    ):
        raise ValueError("unregistered llama.cpp revision")
    memory_baseline = pressure()
    vm_before = counters()
    command = [
        str(args.runtime / "build/bin/llama-server"),
        "-m",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-t",
        "4",
        "-tb",
        "4",
        "-ngl",
        "0",
        "-c",
        "2304",
        "-b",
        "256",
        "-ub",
        "64",
        "-np",
        "1",
        "--cache-ram",
        "128",
        "--ctx-checkpoints",
        "32",
        "--cache-idle-slots" if args.diagnostic_saved_idle_cache else "--no-cache-idle-slots",
        "--no-warmup",
        "--perf",
    ]
    # Active context checkpoints remain at 32. Saved idle slots are optional;
    # compare them separately because lowering checkpoint count loses reuse.
    log = (args.output / "server.log").open("w")
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    sampler_stop = threading.Event()
    memory_samples = []

    def sample() -> None:
        while not sampler_stop.wait(0.05):
            memory_samples.append(proc_memory(process.pid))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{args.port}"
    records: list[dict] = []
    try:
        for _ in range(900):
            if process.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                if httpx.get(url + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise TimeoutError("model load exceeded 180 seconds")
        load_seconds = time.perf_counter() - start
        previous_ids: list[int] = []
        for state in states:
            tokens_response = httpx.post(
                url + "/tokenize",
                timeout=30,
                json={"content": state["prompt"], "add_special": True, "parse_special": True},
            )
            tokens_response.raise_for_status()
            input_ids = tokens_response.json()["tokens"]
            if len(input_ids) + args.max_new_tokens > 2304:
                raise ValueError("source exceeds frozen context budget")
            common = 0
            for left, right in zip(previous_ids, input_ids, strict=False):
                if left != right:
                    break
                common += 1
            previous_ids = input_ids
            for repetition in range(2):
                before = proc_memory(process.pid)
                run = current_run_context()
                case_scope = (
                    run.for_case(state["id"] + f"/rep-{repetition}").activate()
                    if run
                    else nullcontext()
                )
                with case_scope:
                    result = generate(
                        url, state["prompt"], cache_prompt=True, max_new_tokens=args.max_new_tokens
                    )
                after = proc_memory(process.pid)
                records.append(
                    {
                        "state_id": state["id"],
                        "operation": state["operation"],
                        "file_id": state["file_id"],
                        "state_sha256": state["state_sha256"],
                        "source_target": state["source_target"],
                        "input_tokens": len(input_ids),
                        "token_prefix_with_previous_state": common,
                        "repetition": repetition,
                        "process_cold_request": len(records) == 0,
                        "before_memory": before,
                        "after_memory": after,
                        **result,
                    }
                )
                with (args.output / "measurements.jsonl").open("a") as handle:
                    handle.write(json.dumps(records[-1]) + "\n")
        vm_after = counters()
        peak = {
            key: max((row.get(key, 0) for row in memory_samples), default=0)
            for key in {key for row in memory_samples for key in row}
        }
        summary = {
            "alias": args.alias,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cpu": subprocess.check_output(["lscpu"], text=True),
            "ram": Path("/proc/meminfo").read_text(),
            "runtime_revision": "f072b103714dfa1eee531f80b24512faf38e3dd2",
            "model_sha256": file_sha(args.model),
            "model_bytes": args.model.stat().st_size,
            "threads": 4,
            "slots": 1,
            "context_tokens": 2304,
            "cache_ram_mib": 128,
            "context_checkpoints": 32,
            "batch_tokens": 256,
            "microbatch_tokens": 64,
            "saved_idle_slots": args.diagnostic_saved_idle_cache,
            "model_load_seconds": load_seconds,
            "peak_memory": peak,
            "post_request_memory": proc_memory(process.pid),
            "pressure_before": memory_baseline,
            "pressure_after": pressure(),
            "vmstat_delta": {key: vm_after[key] - vm_before[key] for key in vm_before},
            "requests": len(records),
            "states_sha256": args.states_sha256,
            "output_token_limit": args.max_new_tokens,
            "observed_at": datetime.now(UTC).isoformat(),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        sampler_stop.set()
        thread.join(timeout=2)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def measure_rust(args) -> None:
    """Replay exact editor states through the resident context and inference worker."""
    from measure_r2_local import NativeProvider

    from tinycomplete.observability.runs import run_scope

    if not args.plan or not args.states or not args.ssh_host or not args.setting:
        raise ValueError("Rust replay needs a frozen plan, states, setting and configured SSH host")
    if args.ssh_host != "thinkpad" or args.rust_url != "http://127.0.0.1:19194":
        raise ValueError("this campaign allows only the configured ThinkPad tunnel")
    plan = json.loads(args.plan.read_text())
    root = Path(__file__).resolve().parents[1]
    for source, expected in plan["source_files"].items():
        if file_sha(root / source) != expected:
            raise ValueError("producer source fingerprint differs: " + source)
    if file_sha(args.states) != plan["states_sha256"]:
        raise ValueError("editor state fingerprint differs")
    setting = next(s for s in plan["settings"] if s["id"] == args.setting)
    if setting["alias"] != args.alias:
        raise ValueError("model setting differs")
    if args.output.exists():
        raise FileExistsError("preserve old results; use a new result directory")
    args.output.mkdir(parents=True)
    health = httpx.get(args.rust_url + "/health", timeout=20).json()
    for key in ("threads", "prompt_threads", "input_tokens", "cache_type"):
        if health[key] != setting[key]:
            raise ValueError("running setting differs: " + key)
    if health["model_sha256"] != plan["models"][args.alias]["sha256"]:
        raise ValueError("running model differs")
    binary_check = subprocess.check_output(
        ["ssh", args.ssh_host, "sha256sum " + shlex.quote(plan["binary"])], text=True
    ).split()[0]
    if binary_check != plan["binary_sha256"]:
        raise ValueError("native executable differs from plan")
    service_unit = plan.get("service_unit", "tabcomplete-engine-smoke")
    if service_unit not in ("tabcomplete-engine-smoke", "tabcomplete-engine"):
        raise ValueError("unapproved inference service unit")
    if plan.get("context_owned_by_runtime"):
        if health.get("context_layout") != setting.get("context_layout"):
            raise ValueError("resident context layout differs")
    # Sampling runs on the inference host. Controller memory is never substituted.
    remote_code = """import json,pathlib,subprocess,socket
pid=subprocess.check_output(["systemctl","--user","show",SERVICE_UNIT,
 "--property=MainPID","--value"],text=True).strip()
p=pathlib.Path("/proc")/pid
result={"host":socket.gethostname(),"pid":int(pid),
 "memory":(p/"smaps_rollup").read_text(),"status":(p/"status").read_text(),
 "pressure":{n:pathlib.Path("/proc/pressure/"+n).read_text() for n in ["memory","io"]},
 "vmstat":{a[0]:int(a[1]) for l in pathlib.Path("/proc/vmstat").read_text().splitlines()
  if (a:=l.split())[0] in ["pswpin","pswpout","pgmajfault"]},
 "frequency_khz":{str(q):q.read_text().strip()
  for q in pathlib.Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq")},
 "temperatures":{str(q):q.read_text().strip()
  for q in pathlib.Path("/sys/class/thermal").glob("thermal_zone*/temp")}}
print(json.dumps(result))
"""

    remote_code = remote_code.replace("SERVICE_UNIT", repr(service_unit))

    def snapshot():
        return json.loads(
            subprocess.check_output(
                ["ssh", args.ssh_host, "python3 -c " + shlex.quote(remote_code)], text=True
            )
        )

    before = snapshot()
    if before["host"] != plan["inference_host"]:
        raise ValueError("inference host identity differs")
    provider = NativeProvider(args.rust_url, args.alias)
    provider.timeout_seconds = 90
    provider.cache = True
    rows = [json.loads(line) for line in args.states.read_text().splitlines()]
    quality_path = args.plan.parent / plan["quality_states_file"]
    if file_sha(quality_path) != plan["quality_states_sha256"]:
        raise ValueError("development fixture fingerprint differs")
    quality_rows = [json.loads(line) for line in quality_path.read_text().splitlines()]
    deadline = time.monotonic() + plan["max_setting_seconds"]
    results = []
    continuation = plan.get("continuation")
    if continuation:
        prior_path = Path(continuation["measurements_file"])
        if file_sha(prior_path) != continuation["measurements_sha256"]:
            raise ValueError("continuation result fingerprint differs")
        prior_plan = Path(continuation["plan_file"])
        if file_sha(prior_plan) != continuation["plan_sha256"]:
            raise ValueError("continuation plan fingerprint differs")
        previous = json.loads(prior_plan.read_text())
        for key in ("binary_sha256", "models", "states_sha256", "quality_states_sha256"):
            if previous[key] != plan[key]:
                raise ValueError("continuation identity differs: " + key)
        old_setting = next(s for s in previous["settings"] if s["id"] == continuation["setting_id"])
        if {k: v for k, v in old_setting.items() if k != "id"} != {
            k: v for k, v in setting.items() if k != "id"
        }:
            raise ValueError("continuation runtime settings differ")
        if health["runtime_config_hash"] != continuation["runtime_config_hash"]:
            raise ValueError("continuation resident runtime differs")
        results = [json.loads(line) for line in prior_path.read_text().splitlines()]
        if any(r.get("model_sha256") != health["model_sha256"] for r in results):
            raise ValueError("continuation model differs")
    completed = replay_completed_keys(results)
    allowed = {
        (row["id"], repetition, condition)
        for row in rows
        for repetition in (0, 1)
        for condition in ("changed_editor_state", "identical_prompt_repeat")
    } | {(row["id"], 2, "development_check") for row in quality_rows}
    if not completed <= allowed:
        raise ValueError("continuation contains an unknown fixture request")
    with run_scope(args.output / "observability-run.json", "rust-editor-replay") as run:
        for repetition in range(3):
            states = (
                rows
                if repetition < 2
                else [
                    {
                        "id": row["id"],
                        "operation": "development_" + row["expected_action"]["kind"],
                        "source_bucket": None,
                        "expected_action": row["expected_action"],
                        "request": {
                            "state": row["state"],
                            "buffers": [],
                            "repository_identity": "synthetic:rust-editor-development",
                        },
                    }
                    for row in quality_rows
                ]
            )
            for state in states:
                conditions = (
                    ("changed_editor_state", "identical_prompt_repeat")
                    if repetition < 2
                    else ("development_check",)
                )
                if all((state["id"], repetition, c) in completed for c in conditions):
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("frozen inference deadline reached")
                request = state["request"]
                started = time.perf_counter()
                response = httpx.post(
                    args.rust_url + "/v1/editor/context", json=request, timeout=20
                )
                if response.status_code == 422 and repetition == 2:
                    result = {
                        "id": state["id"],
                        "condition": "development_check",
                        "expected_action": state["expected_action"],
                        "canonical_action": None,
                        "stop_type": None,
                        "context_rejected": True,
                    }
                    results.append(result)
                    with (args.output / "measurements.jsonl").open("a") as handle:
                        handle.write(json.dumps(result) + "\n")
                    continue
                response.raise_for_status()
                prepared = response.json()
                if (
                    setting.get("context_layout") == "cursor-last-v1"
                    and not plan.get("context_owned_by_runtime")
                ):
                    prepared["prompt"] = cursor_last_prompt(prepared["prompt"])
                    tokenized = httpx.post(
                        args.rust_url + "/tokenize",
                        timeout=20,
                        json={"content": prepared["prompt"], "add_special": True},
                    )
                    tokenized.raise_for_status()
                    prepared["prompt_tokens"] = len(tokenized.json()["tokens"])
                    if prepared["prompt_tokens"] > health["input_tokens"]:
                        raise ValueError("same-information layout exceeds input budget")
                    prepared["context_hash"] = sha(prepared["prompt"].encode())
                    prepared["context_policy_version"] = "single-line-cursor-last-context-v1"
                construction_ms = (time.perf_counter() - started) * 1000
                if sha(prepared["prompt"].encode()) != prepared["context_hash"]:
                    raise ValueError("context hash mismatch")
                provider.editor_window = prepared.get("window")
                provider.repository_identity = request["repository_identity"]
                for condition in conditions:
                    if (state["id"], repetition, condition) in completed:
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("frozen inference deadline reached")
                    provider.timeout_seconds = min(90, remaining)
                    case_id = f"{state['id']}/{repetition}/{condition}"
                    context = run.for_case(case_id) if run else nullcontext()
                    with context.activate() if run else context:
                        provider.generate_detailed(prepared["prompt"], health["output_tokens"])
                    result = {
                        "id": state["id"],
                        "operation": state["operation"],
                        "source_bucket": state["source_bucket"],
                        "repetition": repetition,
                        "condition": condition,
                        "context_construction_ms": construction_ms,
                        "actual_input_tokens": prepared["prompt_tokens"],
                        "context_policy_version": prepared["context_policy_version"],
                        "prompt_sha256": prepared["context_hash"],
                        **provider.last,
                    }
                    if repetition == 2:
                        result["expected_action"] = state["expected_action"]
                    if run:
                        result["run_id"] = run.run_id
                    results.append(result)
                    with (args.output / "measurements.jsonl").open("a") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {
        "plan_sha256": file_sha(args.plan),
        "setting": setting,
        "identity": health,
        "controller_host": socket.gethostname(),
        "transport": "owned SSH tunnel",
        "before": before,
        "after": snapshot(),
        "requests": len(results),
        "conditions": {},
        "quality_claim": False,
        "continuation": continuation,
        "memory_window": "current invocation snapshots; VmHWM includes earlier resident requests",
    }
    for condition in ("changed_editor_state", "identical_prompt_repeat"):
        values = sorted(r["total_seconds"] * 1000 for r in results if r["condition"] == condition)
        subset = [r for r in results if r["condition"] == condition]
        summary["conditions"][condition] = {
            "count": len(values),
            "median_ms": values[len(values) // 2],
            "p95_ms": values[min(len(values) - 1, (95 * len(values) + 99) // 100 - 1)],
            "actual_eos": sum(r["stop_type"] == "eos" for r in subset),
            "canonical_valid": sum(r["canonical_action"] is not None for r in subset),
            "cache_tokens": sum(r["server_timings"].get("cache_n", 0) for r in subset),
            "recomputed_tokens": sum(r["server_timings"].get("prompt_n", 0) for r in subset),
        }
    quality = [r for r in results if r["condition"] == "development_check"]

    def exact(row):
        action = row["canonical_action"]
        expected = row["expected_action"]
        return bool(
            action is not None
            and action["kind"] == expected["kind"]
            and action.get("text") == expected.get("text")
        )

    summary["development"] = {
        "count": len(quality),
        "canonical_valid": sum(r["canonical_action"] is not None for r in quality),
        "actual_eos": sum(r["stop_type"] == "eos" for r in quality),
        "exact_actions": sum(exact(r) for r in quality),
        "by_action": {
            kind: {
                "count": sum(r["expected_action"]["kind"] == kind for r in quality),
                "exact": sum(exact(r) for r in quality if r["expected_action"]["kind"] == kind),
            }
            for kind in ("keep", "replace_line", "insert_before", "delete_line")
        },
        "no_edit_false_changes": sum(
            r["expected_action"]["kind"] == "keep"
            and r["canonical_action"] is not None
            and r["canonical_action"]["kind"] != "keep"
            for r in quality
        ),
        "scope": (
            "24 preselected existing development states; synthetic cases are codec "
            "regressions, not human-edit quality"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def replay_completed_keys(rows: list[dict]) -> set[tuple[str, int, str]]:
    """Only exact recorded request identities can be skipped in a declared continuation."""
    keys: set[tuple[str, int, str]] = set()
    for row in rows:
        condition = row["condition"]
        repetition = row.get("repetition", 2 if condition == "development_check" else -1)
        if condition not in {
            "changed_editor_state",
            "identical_prompt_repeat",
            "development_check",
        }:
            raise ValueError("unknown recorded replay condition")
        if repetition not in (0, 1, 2) or (repetition == 2) != (condition == "development_check"):
            raise ValueError("invalid recorded replay repetition")
        key = (row["id"], repetition, condition)
        if key in keys:
            raise ValueError("duplicate recorded replay request")
        keys.add(key)
    return keys


def cursor_last_prompt(prompt: str) -> str:
    """One declared same-information layout experiment; never trim source/markers."""
    lines = prompt.split("\n")
    cursor = [line for line in lines if line.startswith("Cursor byte column: ")]
    if len(cursor) != 1 or lines[-1] != "Action:":
        raise ValueError("unexpected control prompt layout")
    lines.remove(cursor[0])
    lines.insert(len(lines) - 1, cursor[0])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
