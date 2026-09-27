# One-line R1 Kaggle worker bundle

The builder stages the already approved q25 pretrained snapshot and an accepted
public/synthetic training shard for a **private** Kaggle dataset. It verifies
the frozen plan, local hashes, licenses, data diversity, live quota, active
jobs, budget ledger, and pushed commit. `prepare` performs no upload or GPU
allocation. It refuses the current 0/20,000 accepted-data state.

The staged kernel runs on `NvidiaTeslaT4`, uses one CUDA device, and never
downloads model weights. Setup, failures, training, and checkpoint saving count
inside one session deadline. Output includes a complete update-boundary
checkpoint, F16 inference export on completion, logs, and manifests. A later
session attaches the prior private kernel output and verifies the exact
checkpoint SHA-256 and training fingerprint before restoring it.

After the campaign branch is committed and pushed, the accepted data manifest
and locked development LR selection exist, and the controller has refreshed
quota, prepare a bundle in an approved artifact directory:

```sh
uv run python kaggle/one_line_r1/build_bundle.py prepare \
  --model /path/to/pinned/q25-snapshot \
  --train /path/to/accepted/train.jsonl \
  --data-manifest /path/to/data_manifest.json \
  --selection /path/to/locked_lr_selection.json \
  --output /path/to/approved-artifacts/one_line_r1/session-01 \
  --session-minutes 120
```

The private dataset metadata uses Kaggle's `other` license option because the
input combines Apache-2.0 model files with public source rows retaining their
own per-row licenses. The bundle does not relicense those files. The Kaggle CLI
defaults dataset creation to private when `-u` is absent. Once the campaign
controller has made a final fresh budget check, its submit steps are:

```sh
kaggle datasets create -p /path/to/approved-artifacts/one_line_r1/session-01/dataset -t
kaggle kernels push -p /path/to/approved-artifacts/one_line_r1/session-01/kernel --timeout 7200
```

After kernel completion, retrieve and verify its output:

```sh
kaggle kernels output shlokbhakta/tabcomplete-one-line-r1-main-s01 \
  -p /path/to/approved-artifacts/one_line_r1/pulled-session-01
uv run python kaggle/one_line_r1/build_bundle.py verify-output \
  /path/to/approved-artifacts/one_line_r1/pulled-session-01
```

For an exact continuation, attach that kernel as `--resume-kernel-source`,
provide the verified pulled output with `--resume-output`, and use a new kernel
ID/output directory. The builder rechecks the campaign budget and refuses a
renewed quota window or active GPU job. Preserve old output until the new
checkpoint has been retrieved and verified.

The worker follows the repository's existing Kaggle pattern: private T4
kernel metadata, exact pushed Git commit, pinned short-lived library setup,
local input hash checks, and private output retrieval. It has no teacher,
embedding, draft, or automatic fallback model. The training entry point is
`scripts/train_one_line.py`; it controls the 100-million-token ceiling and
resumable FP32/FP16 SFT state.
