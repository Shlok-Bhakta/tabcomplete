#!/usr/bin/env bash
# Usage: sweep_formats.sh ENGINE MODEL CASES OUTDIR [LIMIT]
set -euo pipefail
engine=$1 model=$2 cases=$3 out=$4 limit=${5:-0}
mkdir -p "$out"
here=$(dirname "$0")
run() {
  local name=$1; shift
  python3 -I "$here/run_eval.py" --engine "$engine" --model "$model" --cases "$cases" --name "$name" \
    --limit "$limit" --no-jump --out "$out/$name.rows.jsonl" -- "$@" | tee -a "$out/summary.jsonl"
}
run server_local    --cursor-marker true  --line-numbers sweep-local --diff-format server --prefill first-line
run server_nocursor --cursor-marker false --line-numbers sweep-local --diff-format server --prefill first-line
run abs_nocursor    --cursor-marker false --line-numbers absolute    --diff-format server --prefill first-line
run card            --cursor-marker false --line-numbers none        --diff-format card   --prefill first-line
run card_noprefill  --cursor-marker false --line-numbers none        --diff-format card   --prefill none
