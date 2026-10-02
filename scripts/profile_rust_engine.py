#!/usr/bin/env python3
"""Run one bounded, content-free CPU latency screen against an owned Rust service."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tinycomplete.observability.runs import run_scope

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STATES_SHA256 = "af4b53e4a4b4ef1e81270055549662f22755a783d337341376ca0618a18cc713"
EXPECTED_MODEL_SHA256 = "4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb"
EXPECTED_BINARY_SHA256 = "a158b1f7de3ed3ea0f3ecaebc105a7f27d2335a1bd2011fb2fdfc1ea0f490ecc"
PROFILE_DIR = Path("reports/prototype/rust_editor_format_r2/profile_v2")
CAMPAIGN_SECONDS = 1200
REQUEST_TIMEOUT_SECONDS = 90


class ProfileError(RuntimeError):
    """Sanitized, safe-to-print failure reason."""


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_sha(value: object) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_states(path: Path) -> tuple[list[dict], str]:
    actual = file_sha(path)
    if actual != EXPECTED_STATES_SHA256:
        raise ProfileError("frozen synthetic state fingerprint mismatch")
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError):
        raise ProfileError("frozen synthetic state file is unreadable") from None
    if len(rows) != 24 or len({row.get("id") for row in rows}) != 24:
        raise ProfileError("frozen screen requires 24 unique editor states")
    for row in rows:
        request = row.get("request")
        if row.get("synthetic") is not True or not isinstance(request, dict):
            raise ProfileError("screen contains a non-synthetic or invalid state")
        state = request.get("state")
        if not isinstance(state, dict) or not isinstance(state.get("source"), str):
            raise ProfileError("screen state does not meet the editor request contract")
    return rows, actual


def validate_plan(plan: dict) -> None:
    if plan.get("schema") != "rust-editor-cpu-profile-plan-v2":
        raise ProfileError("unsupported performance plan")
    if plan.get("campaign_seconds") != CAMPAIGN_SECONDS:
        raise ProfileError("campaign guard differs from frozen plan")
    if plan.get("model_sha256") != EXPECTED_MODEL_SHA256:
        raise ProfileError("frozen model identity mismatch")
    if plan.get("baseline_binary_sha256") != EXPECTED_BINARY_SHA256:
        raise ProfileError("frozen baseline executable identity mismatch")
    if plan.get("runner_sha256") != file_sha(Path(__file__)):
        raise ProfileError("profile runner fingerprint mismatch")
    provider_path = ROOT / "scripts/measure_r2_local.py"
    if plan.get("native_provider_sha256") != file_sha(provider_path):
        raise ProfileError("shared native provider fingerprint mismatch")
    for relative, expected_hash in plan.get("engine_source_sha256", {}).items():
        if file_sha(ROOT / relative) != expected_hash:
            raise ProfileError("native engine source fingerprint mismatch")
    settings = plan.get("settings")
    if not isinstance(settings, list) or len(settings) != 7:
        raise ProfileError("frozen screen must contain seven settings")
    ids = [s.get("id") for s in settings]
    if len(ids) != len(set(ids)) or ids[0] != "baseline_start" or ids[-1] != "baseline_end":
        raise ProfileError("invalid frozen setting sequence")


def setting_for(plan: dict, args: argparse.Namespace) -> dict:
    matches = [s for s in plan["settings"] if s["id"] == args.setting]
    if len(matches) != 1:
        raise ProfileError("unknown setting ID")
    setting = matches[0]
    if args.threads != setting["threads"] or args.prompt_threads != setting["prompt_threads"]:
        raise ProfileError("requested thread counts differ from the frozen screen")
    if args.microbatch not in setting["allowed_microbatch_sizes"]:
        raise ProfileError("requested microbatch differs from the frozen screen")
    return {
        **setting,
        "microbatch_size": args.microbatch,
        "binary_sha256": setting.get("binary_sha256", EXPECTED_BINARY_SHA256),
    }


def check_health(url: str, setting: dict, binary_sha: str) -> dict:
    try:
        response = httpx.get(url + "/health", timeout=10)
        response.raise_for_status()
        health = response.json()
    except Exception:
        raise ProfileError("temporary engine health check failed") from None
    expected = {
        "status": "ok",
        "alias": "q25",
        "model_sha256": EXPECTED_MODEL_SHA256,
        "model_protocol": "single-line-edit-v1",
        "context_layout": "cursor-last-v1",
        "model_embedded": True,
        "model_storage": "executable-mmap",
        "model_selection": "declarative",
        "backend": "llama.cpp CPU via Rust",
        "llama_cpp_2": "0.1.157",
        "llama_cpp_sys_2": "0.1.158",
        "active_slots": 1,
        "saved_contexts": 0,
        "context_size": 2304,
        "batch_size": 256,
        "input_tokens": 1024,
        "output_tokens": 64,
        "cache_type": "f16",
        "threads": setting["threads"],
        "prompt_threads": setting["prompt_threads"],
        "microbatch_size": setting["microbatch_size"],
    }
    if any(health.get(key) != expected_value for key, expected_value in expected.items()):
        raise ProfileError("temporary engine health does not match the frozen setting")
    if (
        not isinstance(health.get("runtime_config_hash"), str)
        or len(health["runtime_config_hash"]) != 64
    ):
        raise ProfileError("temporary engine runtime identity is missing")
    if not binary_sha or len(binary_sha) != 64:
        raise ProfileError("attested executable hash is invalid")
    return health


REMOTE_SAMPLE = r"""
import hashlib,json,pathlib,select,sys,time
pid=int(sys.argv[1]); interval=float(sys.argv[2]); p=pathlib.Path('/proc')/str(pid)
def vals(path,keys):
 out={}
 try:
  for line in pathlib.Path(path).read_text().splitlines():
   parts=line.split(None,1)
   if len(parts)==2:
    k,v=parts[0].rstrip(':'),parts[1]
    if k in keys:
     a=v.strip().split()
     if a: out[k]=int(a[0])*(1024 if len(a)>1 and a[1]=='kB' else 1)
 except (OSError,ValueError): pass
 return out
def pressure(name):
 out={}
 try:
  for line in pathlib.Path('/proc/pressure/'+name).read_text().splitlines():
   a=line.split(); row={}
   for part in a[1:]:
    k,v=part.split('=',1)
    if k in ('avg10','avg60','avg300','total'): row[k]=float(v)
   out[a[0]]=row
 except (OSError,ValueError): pass
 return out
def sample():
 sm=vals(p/'smaps_rollup',{'Rss','Pss','Pss_Anon','Pss_File','Anonymous','Swap'})
 st=vals(p/'status',{'VmHWM','VmRSS','VmSwap','Threads'})
 cg=''
 try:
  for line in (p/'cgroup').read_text().splitlines():
   if line.startswith('0::'): cg=line.split('::',1)[1].lstrip('/'); break
 except OSError: pass
 cp=pathlib.Path('/sys/fs/cgroup')/cg
 def number(path):
  try: return int(pathlib.Path(path).read_text().strip())
  except (OSError,ValueError): return None
 cgmem=vals(cp/'memory.stat',{'anon','file','kernel','sock'})
 events=vals(cp/'memory.events',{'oom','oom_kill','max','high'})
 cpu=vals(cp/'cpu.stat',{'usage_usec','user_usec','system_usec','nr_periods','nr_throttled','throttled_usec'})
 vm=vals('/proc/vmstat',{'pswpin','pswpout','pgmajfault'})
 freq=[]
 for f in pathlib.Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_cur_freq'):
  try: freq.append(int(f.read_text().strip()))
  except (OSError,ValueError): pass
 temps=[]
 for f in pathlib.Path('/sys/class/thermal').glob('thermal_zone*/temp'):
  try: temps.append(int(f.read_text().strip()))
  except (OSError,ValueError): pass
 try:
  exe=pathlib.Path('/proc/self/exe')
  executable={'basename':pathlib.Path(p/'exe').resolve().name,'size_bytes':(p/'exe').stat().st_size}
 except OSError: executable={}
 def raw(path):
  try: return pathlib.Path(path).read_text().strip()
  except OSError: return None
 host_path=pathlib.Path('/etc/hostname')
 host=host_path.read_text().strip() if host_path.exists() else 'unknown'
 return {'host':host,'pid':pid,'monotonic_s':time.monotonic(),
  'process_memory_bytes':sm,'process_status':st,
  'process_executable':executable,
  'cgroup_memory_current_bytes':number(cp/'memory.current'),
  'cgroup_memory_peak_bytes':number(cp/'memory.peak'),
  'cgroup_swap_current_bytes':number(cp/'memory.swap.current'),
  'cgroup_memory_max':raw(cp/'memory.max'),'cgroup_cpu_max':raw(cp/'cpu.max'),
  'cgroup_swap_max':raw(cp/'memory.swap.max'),
  'cgroup_memory_stat_bytes':cgmem,'cgroup_memory_events':events,'cgroup_cpu':cpu,
  'vmstat':vm,'pressure':{n:pressure(n) for n in ('cpu','memory','io')},
  'cpu_frequency_khz':freq,'thermal_millicelsius':temps}
while p.exists():
 try: print(json.dumps(sample(),separators=(',',':')),flush=True)
 except Exception: break
 ready,_,_=select.select([sys.stdin],[],[],interval)
 if ready:
  sys.stdin.readline()
  break
"""


class RemoteSampler:
    def __init__(self, host: str, pid: int, interval: float):
        if not host.replace("-", "").replace("_", "").isalnum() or pid <= 1:
            raise ProfileError("invalid remote sampler identity")
        remote = "python3 -c " + shlex.quote(REMOTE_SAMPLE) + f" {pid} {interval:.3f}"
        self.process = subprocess.Popen(
            ["ssh", host, remote],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.rows: list[dict] = []
        self.lock = threading.Lock()
        self.first_sample = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        if not self.first_sample.wait(15):
            self.stop()
            raise ProfileError("remote process and cgroup sampler did not start")

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["controller_received_monotonic_s"] = time.monotonic()
            with self.lock:
                self.rows.append(row)
            self.first_sample.set()

    def copy_rows(self) -> list[dict]:
        with self.lock:
            return list(self.rows)

    def stop(self) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write("stop\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.reader.join(timeout=2)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def aggregate(records: list[dict]) -> dict:
    result: dict = {}
    for condition in ("changed_editor_state", "identical_prompt_repeat"):
        group = [r for r in records if r["condition"] == condition]

        def numbers(rows: list[dict], key: str) -> list[float]:
            return [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]

        result[condition] = {
            "count": len(group),
            "model_total_ms_median": percentile(numbers(group, "model_total_ms"), 0.5),
            "model_total_ms_p95": percentile(numbers(group, "model_total_ms"), 0.95),
            "context_construction_ms_median": percentile(
                numbers(group, "context_construction_ms"), 0.5
            ),
            "end_to_end_ms_median": percentile(numbers(group, "end_to_end_ms"), 0.5),
            "end_to_end_ms_p95": percentile(numbers(group, "end_to_end_ms"), 0.95),
            "prompt_ms_median": percentile(numbers(group, "prompt_ms"), 0.5),
            "prompt_ms_p95": percentile(numbers(group, "prompt_ms"), 0.95),
            "decode_ms_median": percentile(numbers(group, "decode_ms"), 0.5),
            "decode_ms_p95": percentile(numbers(group, "decode_ms"), 0.95),
            "first_observed_token_ms_median": percentile(
                numbers(group, "first_observed_token_ms"), 0.5
            ),
            "actual_eos": sum(r["stop_type"] == "eos" for r in group),
            "canonical_action_valid": sum(r["canonical_action_digest"] is not None for r in group),
            "cache_tokens": sum(r["cache_tokens"] for r in group),
            "recomputed_prompt_tokens": sum(r["recomputed_prompt_tokens"] for r in group),
            "generated_tokens": sum(r["generated_tokens"] for r in group),
            "observed_decode_tokens_per_second": (
                1000.0
                * sum(r["generated_tokens"] for r in group)
                / sum(r["decode_ms"] for r in group)
                if sum(r["decode_ms"] for r in group) > 0
                else None
            ),
        }
    return result


def cgroup_summary(samples: list[dict]) -> dict:
    if not samples:
        return {"sample_count": 0}

    def pick(sample: dict, dotted: tuple[str, ...]) -> int | None:
        value: object = sample
        for key in dotted:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value if isinstance(value, int) else None

    memory = [pick(s, ("process_memory_bytes", "Rss")) for s in samples]
    pss = [pick(s, ("process_memory_bytes", "Pss")) for s in samples]
    anon = [pick(s, ("process_memory_bytes", "Pss_Anon")) for s in samples]
    file = [pick(s, ("process_memory_bytes", "Pss_File")) for s in samples]
    swap = [pick(s, ("process_memory_bytes", "Swap")) for s in samples]
    high_water = [pick(s, ("process_status", "VmHWM")) for s in samples]
    cgroup_current = [s.get("cgroup_memory_current_bytes") for s in samples]
    cpu_use = [pick(s, ("cgroup_cpu", "usage_usec")) for s in samples]
    throttled = [pick(s, ("cgroup_cpu", "throttled_usec")) for s in samples]

    def valid(values: list[int | None]) -> list[int]:
        return [v for v in values if v is not None]

    return {
        "sample_count": len(samples),
        "sampling_interval_seconds": 0.5,
        "process_rss_peak_bytes": max(valid(memory), default=None),
        "process_pss_peak_bytes": max(valid(pss), default=None),
        "process_pss_anon_peak_bytes": max(valid(anon), default=None),
        "process_pss_file_peak_bytes": max(valid(file), default=None),
        "process_vm_hwm_peak_bytes": max(valid(high_water), default=None),
        "process_swap_peak_bytes": max(valid(swap), default=None),
        "cgroup_memory_current_peak_bytes": max(valid(cgroup_current), default=None),
        "cgroup_memory_peak_bytes": max(
            [v for s in samples if isinstance((v := s.get("cgroup_memory_peak_bytes")), int)],
            default=None,
        ),
        "cgroup_swap_current_peak_bytes": max(
            [v for s in samples if isinstance((v := s.get("cgroup_swap_current_bytes")), int)],
            default=None,
        ),
        "cgroup_memory_max": samples[0].get("cgroup_memory_max"),
        "cgroup_cpu_max": samples[0].get("cgroup_cpu_max"),
        "cgroup_swap_max": samples[0].get("cgroup_swap_max"),
        "cgroup_cpu_usage_delta_usec": (
            valid(cpu_use)[-1] - valid(cpu_use)[0] if len(valid(cpu_use)) > 1 else None
        ),
        "cgroup_cpu_throttled_delta_usec": (
            valid(throttled)[-1] - valid(throttled)[0] if len(valid(throttled)) > 1 else None
        ),
        "vmstat_start": samples[0].get("vmstat", {}),
        "vmstat_end": samples[-1].get("vmstat", {}),
        "pressure_start": samples[0].get("pressure", {}),
        "pressure_end": samples[-1].get("pressure", {}),
        "memory_events_start": samples[0].get("cgroup_memory_events", {}),
        "memory_events_end": samples[-1].get("cgroup_memory_events", {}),
        "cpu_frequency_sample_khz_start": samples[0].get("cpu_frequency_khz", []),
        "cpu_frequency_sample_khz_end": samples[-1].get("cpu_frequency_khz", []),
        "thermal_sample_millicelsius_start": samples[0].get("thermal_millicelsius", []),
        "thermal_sample_millicelsius_end": samples[-1].get("thermal_millicelsius", []),
    }


def campaign_gate(plan_path: Path, plan: dict, setting: dict, output: Path):
    state_path = ROOT / plan["campaign_state_file"]
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_hash = file_sha(plan_path)
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            raise ProfileError("campaign state is unreadable") from None
        if state.get("plan_sha256") != plan_hash or state.get("failed"):
            raise ProfileError("campaign state does not match this frozen plan")
    else:
        state = {
            "schema": "rust-editor-cpu-profile-campaign-v1",
            "plan_sha256": plan_hash,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "started_epoch_seconds": time.time(),
            "deadline_epoch_seconds": time.time() + CAMPAIGN_SECONDS,
            "completed_settings": [],
            "failed": False,
        }
        atomic_json(state_path, state)
    if time.time() > state["deadline_epoch_seconds"]:
        raise ProfileError("20-minute campaign budget has expired")
    completed = state["completed_settings"]
    settings = plan["settings"]
    if len(completed) >= len(settings) or settings[len(completed)]["id"] != setting["id"]:
        raise ProfileError("setting order differs from the frozen sequential screen")
    if output.exists():
        raise ProfileError("output exists; preserve evidence and choose the registered path")
    selected = setting.get("microbatch_size")
    if setting["id"] in ("prompt_threads8_best", "decode_threads2_best"):
        selected_mb = state.get("selected_microbatch_size")
        if selected_mb is None:
            if selected not in (64, 128, 256):
                raise ProfileError("select the winning microbatch from the two prior screens")
            state["selected_microbatch_size"] = selected
            atomic_json(state_path, state)
        elif selected != selected_mb:
            raise ProfileError("selected microbatch changed during the frozen screen")
    elif setting["id"] in ("microbatch128", "microbatch256"):
        if selected != setting["microbatch_size"]:
            raise ProfileError("microbatch setting differs from its registered screen")
    return state_path, state, lock


def finish_campaign(state_path: Path, state: dict, setting_id: str, output: Path) -> None:
    state["completed_settings"].append(setting_id)
    state.setdefault("outputs", []).append(
        {"setting_id": setting_id, "file": output.name, "sha256": file_sha(output)}
    )
    state["last_completed_at_utc"] = datetime.now(UTC).isoformat()
    atomic_json(state_path, state)


def run(args: argparse.Namespace) -> dict:
    plan_path = args.plan.resolve()
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError):
        raise ProfileError("frozen performance plan is unreadable") from None
    validate_plan(plan)
    if args.target_host != "thinkpad" or args.url != "http://127.0.0.1:19197":
        raise ProfileError("screen only permits the configured ThinkPad tunnel")
    setting = setting_for(plan, args)
    if args.binary_sha256 != setting["binary_sha256"]:
        raise ProfileError("attested embedded executable differs from this setting")
    if args.states.resolve() != (ROOT / plan["states_file"]).resolve():
        raise ProfileError("screen state path differs from the frozen plan")
    states, states_hash = load_states(args.states)
    if states_hash != plan["states_sha256"]:
        raise ProfileError("screen state fingerprint differs from the plan")
    output = args.output.resolve()
    if output.parent != (ROOT / PROFILE_DIR).resolve() or output.name != f"{setting['id']}.json":
        raise ProfileError("output path must use the registered screen filename")
    state_path, campaign_state, campaign_lock = campaign_gate(plan_path, plan, setting, output)
    health = check_health(args.url, setting, args.binary_sha256)
    sampler = RemoteSampler(args.target_host, args.pid, 0.5)
    initial_samples = sampler.copy_rows()
    if not initial_samples or initial_samples[0].get("pid") != args.pid:
        sampler.stop()
        raise ProfileError("remote process identity does not match the supplied PID")
    if initial_samples[0].get("host") != plan["expected_remote_hostname"]:
        sampler.stop()
        raise ProfileError("remote host identity differs from the frozen screen")
    expected_exe = setting.get("executable_basename", "tabcomplete-qwen")
    if initial_samples[0].get("process_executable", {}).get("basename") != expected_exe:
        sampler.stop()
        raise ProfileError("remote PID is not running the attested embedded executable")
    if initial_samples[0].get("process_executable", {}).get("size_bytes") != setting.get(
        "executable_bytes", 500399074
    ):
        sampler.stop()
        raise ProfileError("remote executable size differs from the frozen plan")
    provider = None
    records = []
    started = time.monotonic()
    run_meta = output.with_name(output.stem + "_observability.json")
    try:
        from measure_r2_local import NativeProvider

        provider = NativeProvider(args.url, "q25-cursor-last-cpu-screen")
        provider.cache = True
        provider.timeout_seconds = REQUEST_TIMEOUT_SECONDS
        with run_scope(run_meta, "rust-editor-cpu-latency-profile_v2") as run_context:
            for repetition in range(2):
                for index, row in enumerate(states):
                    conditions = ["changed_editor_state"]
                    if index < 12:
                        conditions.append("identical_prompt_repeat")
                    for condition in conditions:
                        try:
                            slot = httpx.get(args.url + "/slots", timeout=5)
                            slot.raise_for_status()
                            if slot.json() != [{"id": 0, "is_processing": False}]:
                                raise ProfileError("engine slot was occupied before a request")
                            request = row["request"]
                            context_started = time.perf_counter()
                            context_response = httpx.post(
                                args.url + "/v1/editor/context",
                                json=request,
                                timeout=15,
                            )
                            context_response.raise_for_status()
                            prepared = context_response.json()
                            context_ms = (time.perf_counter() - context_started) * 1000
                        except ProfileError:
                            raise
                        except Exception:
                            raise ProfileError("editor context preparation failed") from None
                        prompt = prepared.get("prompt")
                        if not isinstance(prompt, str) or digest_bytes(
                            prompt.encode()
                        ) != prepared.get("context_hash"):
                            raise ProfileError("prepared context hash verification failed")
                        if prepared.get("context_layout") != "cursor-last-v1":
                            raise ProfileError("prepared context layout changed during screen")
                        identity = prepared.get("model_identity", {})
                        if identity.get("model_sha256") != EXPECTED_MODEL_SHA256:
                            raise ProfileError("prepared context model identity changed")
                        provider.editor_window = prepared.get("window")
                        provider.repository_identity = request["repository_identity"]
                        case_id = f"{setting['id']}/{row['id']}/{repetition}/{condition}"
                        case_scope = run_context.for_case(case_id) if run_context else None
                        try:
                            if case_scope is not None:
                                with case_scope.activate():
                                    provider.generate_detailed(prompt, health["output_tokens"])
                            else:
                                provider.generate_detailed(prompt, health["output_tokens"])
                        except Exception:
                            raise ProfileError("native generation request failed") from None
                        last = provider.last
                        timing = last.get("server_timings", {})
                        cache_tokens = int(timing.get("cache_n", 0) or 0)
                        prompt_tokens = int(timing.get("prompt_n", 0) or 0)
                        predicted = int(timing.get("predicted_n", 0) or 0)
                        prompt_ms = float(timing.get("prompt_ms", 0) or 0)
                        decode_ms = float(timing.get("predicted_ms", 0) or 0)
                        model_ms = float(timing.get("total_ms", 0) or 0)
                        client_completion_ms = float(last.get("total_seconds", 0) or 0) * 1000
                        if int(prepared.get("prompt_tokens", 0)) != cache_tokens + prompt_tokens:
                            raise ProfileError("backend token count differs from prepared prompt")
                        action = last.get("canonical_action")
                        action_hash = json_sha(action) if isinstance(action, dict) else None
                        token_ids = last.get("token_ids", [])
                        first_token_ms = (
                            float(last["token_arrival_seconds"]["1"]) * 1000
                            if "1" in last.get("token_arrival_seconds", {})
                            else None
                        )
                        records.append(
                            {
                                "state_id": row["id"],
                                "source_bucket": row.get("source_bucket"),
                                "repetition": repetition,
                                "condition": condition,
                                "context_construction_ms": round(context_ms, 3),
                                "end_to_end_ms": round(context_ms + client_completion_ms, 3),
                                "client_completion_ms": round(client_completion_ms, 3),
                                "model_total_ms": round(model_ms, 3),
                                "first_observed_token_ms": (
                                    round(first_token_ms, 3) if first_token_ms is not None else None
                                ),
                                "prompt_ms": round(prompt_ms, 3),
                                "decode_ms": round(decode_ms, 3),
                                "prepared_input_tokens": int(prepared["prompt_tokens"]),
                                "cache_tokens": cache_tokens,
                                "recomputed_prompt_tokens": prompt_tokens,
                                "generated_tokens": predicted,
                                "stop_type": last.get("stop_type"),
                                "terminal_observed": bool(last.get("terminal_observed")),
                                "canonical_action_digest": action_hash,
                                "action_validation_status": None,
                                "output_token_ids_sha256": digest_bytes(
                                    json.dumps(token_ids, separators=(",", ":")).encode()
                                ),
                                "completed_monotonic_s": round(time.monotonic(), 3),
                            }
                        )
        samples = sampler.copy_rows()
    finally:
        sampler.stop()
    if not records:
        raise ProfileError("screen completed without model requests")
    expected_changed = 48
    expected_repeat = 24
    if sum(r["condition"] == "changed_editor_state" for r in records) != expected_changed:
        raise ProfileError("changed-state request count differs from the frozen screen")
    if sum(r["condition"] == "identical_prompt_repeat" for r in records) != expected_repeat:
        raise ProfileError("same-prompt request count differs from the frozen screen")
    if not samples:
        raise ProfileError("no remote process or cgroup samples were captured")
    summary = {
        "schema": "rust-editor-cpu-profile-result-v1",
        "setting": setting,
        "host": plan["host_label"],
        "cpu": plan["cpu_label"],
        "model_sha256": EXPECTED_MODEL_SHA256,
        "binary_sha256_attested": args.binary_sha256,
        "runtime_config_hash": health["runtime_config_hash"],
        "model_protocol": health["model_protocol"],
        "context_layout": health["context_layout"],
        "states_sha256": states_hash,
        "requests": len(records),
        "conditions": aggregate(records),
        "resource_summary": cgroup_summary(samples),
        "resource_samples": samples,
        "records": records,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "quality_claim": False,
        "content_recorded": False,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "campaign_deadline_epoch_seconds": campaign_state["deadline_epoch_seconds"],
    }
    atomic_json(output, summary)
    finish_campaign(state_path, campaign_state, setting["id"], output)
    campaign_lock.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--prompt-threads", type=int, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        result = run(args)
    except ProfileError as exc:
        sys.stderr.write(f"profile runner stopped: {exc}\n")
        return 2
    except Exception as exc:
        # Suppress tracebacks and exception payloads: neither is needed to diagnose
        # a failed synthetic timing screen, and provider errors can carry content.
        sys.stderr.write(f"profile runner stopped: internal {type(exc).__name__}\n")
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "setting": result["setting"]["id"],
                "requests": result["requests"],
                "elapsed_seconds": result["elapsed_seconds"],
                "changed_median_ms": result["conditions"]["changed_editor_state"][
                    "end_to_end_ms_median"
                ],
                "changed_p95_ms": result["conditions"]["changed_editor_state"]["end_to_end_ms_p95"],
                "output": "content-free profile record written",
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
