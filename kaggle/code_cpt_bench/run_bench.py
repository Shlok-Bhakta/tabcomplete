"""Kaggle T4x2 throughput benchmark orchestrator (Stage 1 lab).

Runs every candidate from ``candidates.json`` in a FRESH subprocess via
``torch.distributed.run`` so FSDP state, ``torch.compile`` caches, Qwen kernel
backend selection, and CUDA allocator state cannot contaminate measurements.
Gathers one result JSON per candidate plus ``throughput_benchmarks.jsonl`` /
``throughput_benchmarks.json`` and renders plots automatically.

Usage on Kaggle (GPU kernel, internet on, prepared-corpus kernel attached)::

    git clone --depth 1 --branch stage1/code-cpt \
        https://github.com/Shlok-Bhakta/tabcomplete.git /kaggle/working/tabcomplete
    python /kaggle/working/tabcomplete/kaggle/code_cpt_bench/run_bench.py \
        [--candidates exp00_baseline_repro,exp01_fla_conv] [--stage A] [--max-tokens 98304]

Everything shared (hardware, CUDA env, torch/transformers, dataset, model
revision, seed 271828, sequence length, token ordering) is held fixed; only
the candidate flags vary. Never launches a long CPT run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
REPOSITORY = "https://github.com/Shlok-Bhakta/tabcomplete.git"
BRANCH = "stage1/code-cpt"
CHECKOUT = Path("/kaggle/working/tabcomplete")
OUTPUT = Path("/kaggle/working/code_cpt_bench")
BASE_DEPS = ["transformers==5.5.0", "accelerate", "bitsandbytes"]
# Optional speed deps. Compatibility with T4 (cc 7.5) / torch 2.10 / CUDA 12.8
# is DISCOVERED by the lab, not assumed: install failures are recorded and the
# matrix still runs with the torch fallback. Do not pin aggressively here; the
# worker records exact resolved versions in every result JSON.
OPTIONAL_DEPS = ["flash-linear-attention", "causal-conv1d", "liger-kernel"]
SEQ_LEN = 2048


def run(cmd: list[str], *, env: dict[str, str] | None = None,
        log: Path | None = None, timeout: int = 1500) -> tuple[int, str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        proc = subprocess.run(cmd, env=merged, text=True, capture_output=True,
                              timeout=timeout)
        out = proc.stdout + proc.stderr
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "") + "\nTIMEOUT\n"
        code = 124
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(out[-200_000:], encoding="utf-8")
    print(f"[orch] exit={code} cmd={' '.join(cmd[:8])}... log={log}")
    return code, out


def pip_install(packages: list[str], log_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for package in packages:
        code, _ = run([sys.executable, "-m", "pip", "install", "-q", package],
                      log=log_dir / f"pip_{package}.log", timeout=900)
        results[package] = {"exit_code": code, "installed": code == 0}
    return results


def find_corpus() -> Path:
    candidates = list(Path("/kaggle/input").glob("**/corpus_metadata.json"))
    if not candidates:
        # Local / CI fallback: allow an explicit env override.
        override = os.environ.get("TABCOMPLETE_CORPUS_DIR")
        if override and (Path(override) / "corpus_metadata.json").exists():
            return Path(override)
        raise FileNotFoundError("attached prepared corpus is missing")
    return candidates[0].parent


def load_matrix(path: Path) -> tuple[dict, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    shared = raw.get("shared", {})
    return shared, list(raw.get("candidates", []))


def merged(candidate: dict, shared: dict) -> dict:
    """Shared defaults with per-candidate overrides."""
    return {**shared, **candidate}


def validate_alignment(candidate: dict, shared: dict) -> tuple[int, int, str | None]:
    """Return (max_tokens, tokens_per_update, error)."""
    full = merged(candidate, shared)
    max_tokens = int(full.get("max_tokens", 98304))
    microbatch = int(full["microbatch"])
    accum = int(full["gradient_accumulation"])
    tokens_per_update = 2 * microbatch * SEQ_LEN * accum
    if max_tokens % tokens_per_update != 0:
        return max_tokens, tokens_per_update, (
            f"max_tokens {max_tokens} is not a whole number of updates "
            f"(tokens_per_update {tokens_per_update}); refusing to stop mid-update"
        )
    return max_tokens, tokens_per_update, None


def launch_candidate(*, candidate: dict, shared: dict, corpus: Path,
                     checkout: Path, out_root: Path) -> dict:
    name = candidate["name"]
    dest = out_root / "candidates" / name
    dest.mkdir(parents=True, exist_ok=True)
    max_tokens, tokens_per_update, align_error = validate_alignment(candidate, shared)
    result_path = dest / "result.json"
    record: dict = {"name": name, "candidate": merged(candidate, shared),
                    "tokens_per_update": tokens_per_update}
    if align_error:
        record.update({"status": "error", "error_type": "MisalignedBudget",
                       "error_message": align_error})
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record
    worker_command = str(candidate.get("worker_command", "bench"))
    if worker_command != "bench":
        # Diagnostic subcommands (topo-level probes) take their own flags.
        cmd = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=2", "-m", "tinycomplete.code_cpt.bench",
            worker_command, "--output", str(result_path),
        ]
        if worker_command == "fsdp2_probe":
            cmd.extend(["--steps", str(int(candidate.get("steps", 3)))])
        if worker_command == "nccl_bench":
            cmd.extend(["--iters", str(int(candidate.get("iters", 20)))])
        started = time.time()
        extra_env = dict(candidate.get("env", {}) or {})
        code, _ = run(
            cmd,
            env={"CUDA_VISIBLE_DEVICES": "0,1",
                 "PYTHONPATH": str(checkout / "src"),
                 "TOKENIZERS_PARALLELISM": "false",
                 **extra_env},
            log=dest / "process.log", timeout=2400,
        )
        record.update({"exit_code": code,
                       "elapsed_seconds_including_load": time.time() - started})
        if result_path.exists():
            try:
                record.update(json.loads(result_path.read_text(encoding="utf-8")))
            except Exception as exc:
                record.update({"status": "error", "error_type": "UnreadableResult",
                               "error_message": f"{type(exc).__name__}: {exc}"[:300]})
        else:
            record.update({"status": "error", "error_type": "MissingResult",
                           "error_message": f"worker exit={code} wrote no result JSON"})
            result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record
    get = lambda key: candidate.get(key, shared.get(key))  # noqa: E731
    full = merged(candidate, shared)
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", "-m", "tinycomplete.code_cpt.bench", "bench",
        "--corpus-dir", str(corpus), "--output", str(result_path),
        "--name", name,
        "--learning-rate", str(float(get("learning_rate"))),
        "--max-tokens", str(max_tokens),
        "--start-block", str(int(get("start_block"))),
        "--seed", str(int(get("seed"))),
        "--microbatch", str(int(full["microbatch"])),
        "--gradient-accumulation", str(int(full["gradient_accumulation"])),
        "--optimizer", str(full.get("optimizer", "adamw_8bit")),
        "--fsdp-strategy", str(full.get("fsdp_strategy", "FULL_SHARD")),
        "--attn-implementation",
        str(full.get("attn_implementation", "sdpa")),
        "--fused-ce", str(full.get("fused_ce", "none")),
        "--workers", str(int(get("workers"))),
        "--prefetch-factor", str(int(get("prefetch_factor"))),
        "--warmup-steps", str(int(get("warmup_steps"))),
        "--profile-steps", str(int(full.get("profile_steps", 0))),
        "--compile-placement", str(full.get("compile_placement", "post_fsdp")),
        "--compile-mode", str(full.get("compile_mode", "default")),
        "--fsdp-backward-prefetch",
        str(full.get("fsdp_backward_prefetch", "default")),
        "--fsdp-group-layers", str(int(full.get("fsdp_group_layers", 1))),
        "--grad-sync-every", str(int(full.get("grad_sync_every", 1))),
        "--fla-gate-fusion", str(int(full.get("fla_gate_fusion", 0))),
    ]
    for option in full.get("inductor_options", []) or []:
        cmd.extend(["--inductor-option", str(option)])
    ckpt = bool(full.get("gradient_checkpointing", True))
    cmd.append("--gradient-checkpointing" if ckpt else "--no-gradient-checkpointing")
    cmd.append("--torch-compile" if full.get("torch_compile", False)
               else "--no-torch-compile")
    cmd.append("--compile-dynamic" if full.get("compile_dynamic", True)
               else "--no-compile-dynamic")
    cmd.append("--compile-fullgraph" if full.get("compile_fullgraph", False)
               else "--no-compile-fullgraph")
    cmd.append("--sync-cleanup" if full.get("sync_cleanup", False)
               else "--no-sync-cleanup")
    cmd.append("--fsdp-forward-prefetch" if full.get(
        "fsdp_forward_prefetch", False) else "--no-fsdp-forward-prefetch")
    cmd.append("--limit-all-gathers" if full.get(
        "limit_all_gathers", True) else "--no-limit-all-gathers")
    cmd.append("--liger-rmsnorm" if full.get(
        "liger_rmsnorm", False) else "--no-liger-rmsnorm")
    cmd.append("--liger-swiglu" if full.get(
        "liger_swiglu", False) else "--no-liger-swiglu")
    cmd.append("--force-torch-fallback" if full.get(
        "force_torch_fallback", False) else "--no-force-torch-fallback")
    started = time.time()
    extra_env = dict(full.get("env", {}) or {})
    code, _ = run(
        cmd,
        env={"CUDA_VISIBLE_DEVICES": "0,1",
             "PYTHONPATH": str(checkout / "src"),
             "TOKENIZERS_PARALLELISM": "false",
             **extra_env},
        log=dest / "process.log", timeout=2400,
    )
    record.update({"exit_code": code,
                   "elapsed_seconds_including_load": time.time() - started})
    if result_path.exists():
        try:
            record.update(json.loads(result_path.read_text(encoding="utf-8")))
        except Exception as exc:
            record.update({"status": "error", "error_type": "UnreadableResult",
                           "error_message": f"{type(exc).__name__}: {exc}"[:300]})
    else:
        record.update({"status": "error", "error_type": "MissingResult",
                       "error_message": f"worker exit={code} wrote no result JSON"})
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="",
                        help="comma-separated subset to run (default: all)")
    parser.add_argument("--stage", type=str, default="A")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="override per-candidate token budget (must stay update-aligned)")
    parser.add_argument("--matrix", type=Path, default=None,
                        help="candidate matrix JSON (default: <checkout>/kaggle/code_cpt_bench/candidates.json)")
    parser.add_argument("--checkout", type=Path, default=CHECKOUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-optional-deps", action="store_true")
    args = parser.parse_args()

    out_root: Path = args.output
    out_root.mkdir(parents=True, exist_ok=True)
    checkout: Path = args.checkout
    if (checkout / ".git").exists():
        run(["git", "fetch", "origin", BRANCH], log=out_root / "git_fetch.log")
    elif not checkout.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH,
             REPOSITORY, str(checkout)], log=out_root / "git_clone.log")

    pip_results: dict = {}
    if not args.skip_optional_deps:
        pip_results["base"] = pip_install(BASE_DEPS, out_root / "logs")
        pip_results["optional"] = pip_install(OPTIONAL_DEPS, out_root / "logs")
    else:
        pip_results["note"] = "skipped by flag"

    import torch  # noqa: E402  (import after pip so Kaggle env is settled)
    import accelerate  # noqa: E402
    import transformers  # noqa: E402

    environment = {
        "python": sys.version, "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__, "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i)
                 for i in range(torch.cuda.device_count())],
        "pip": pip_results,
    }
    (out_root / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    corpus = find_corpus()

    # NOTE: Kaggle script kernels execute only the code_file as /kaggle/src/script.py,
    # so sibling files (candidates.json, plot.py) are NOT present next to __file__.
    # Resolve them from the cloned checkout instead.
    bench_dir = checkout / "kaggle" / "code_cpt_bench"
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))

    # Experiment 0 backend probe before any training.
    probe_path = out_root / "backend_probe.json"
    run([sys.executable, "-m", "tinycomplete.code_cpt.bench", "probe",
         "--output", str(probe_path)],
        env={"PYTHONPATH": str(checkout / "src")},
        log=out_root / "backend_probe.log")

    # Round-2 Step 0-B: topology + compiler probes (single-process, cheap).
    run([sys.executable, "-m", "tinycomplete.code_cpt.bench", "topo",
         "--output", str(out_root / "topology.json")],
        env={"PYTHONPATH": str(checkout / "src")},
        log=out_root / "topology.log")
    run([sys.executable, "-m", "tinycomplete.code_cpt.bench", "compiler_probe",
         "--output", str(out_root / "compiler_probe.json")],
        env={"PYTHONPATH": str(checkout / "src")},
        log=out_root / "compiler_probe.log")

    # Round-2 NCCL microbenchmark on the same 2 T4s (fresh distributed pair).
    run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", "-m", "tinycomplete.code_cpt.bench", "nccl_bench",
         "--output", str(out_root / "nccl_microbench.json")],
        env={"CUDA_VISIBLE_DEVICES": "0,1",
             "PYTHONPATH": str(checkout / "src")},
        log=out_root / "nccl_microbench.log", timeout=1200,
    )

    shared, matrix = load_matrix(
        args.matrix or (bench_dir / "candidates.json")
    )
    wanted = {n.strip() for n in args.candidates.split(",") if n.strip()}
    records: list[dict] = []
    for candidate in matrix:
        if wanted and candidate["name"] not in wanted:
            continue
        if args.max_tokens:
            candidate = {**candidate, "max_tokens": args.max_tokens}
        records.append(launch_candidate(candidate=candidate, shared=shared,
                                        corpus=corpus, checkout=checkout,
                                        out_root=out_root))
        # Append incrementally so a later crash never loses earlier data.
        with (out_root / "throughput_benchmarks.jsonl").open(
                "a", encoding="utf-8") as handle:
            handle.write(json.dumps(records[-1], sort_keys=True, default=str) + "\n")
    (out_root / "throughput_benchmarks.json").write_text(
        json.dumps(records, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")

    # Automatic visualization; plotting must never fail the lab.
    try:
        from plot import render_all
        render_all(out_root / "throughput_benchmarks.json", out_root / "plots")
    except Exception as exc:  # noqa: BLE001
        (out_root / "plots_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    # Console ranking (steady-state tok/s, stable only).
    stable = [r for r in records if r.get("status") == "pass"
              and r.get("steady_state_tokens_per_second")]
    stable.sort(key=lambda r: r["steady_state_tokens_per_second"], reverse=True)
    print("\n== throughput lab ranking (steady-state tok/s) ==")
    for row in stable:
        print(f"{row['name']:32s} {row['steady_state_tokens_per_second']:8.1f} "
              f"tok/s  speedup={row.get('speedup_vs_baseline', 0):.3f}x "
              f"vram={row.get('peak_allocated_vram_gib_max', '?')}")
    failed = [r for r in records if r.get("status") != "pass"]
    if failed:
        print("\n== failed candidates (useful data) ==")
        for row in failed:
            print(f"{row['name']:32s} {row.get('status')} "
                  f"{row.get('error_type')}: {(row.get('error_message') or '')[:120]}")
    print(f"\nWrote {len(records)} records to {out_root}")


if __name__ == "__main__":
    main()
