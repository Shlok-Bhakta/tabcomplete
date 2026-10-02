# ThinkPad package

This package builds two separate CPU executables: `tabcomplete-qwen` and
`tabcomplete-sweep`. Each contains one pinned Q4 GGUF appended to its ELF,
followed by a bounded identity footer. `tabcomplete-engine` selects the
package's declarative `modelVariant` (`qwen` by default). Model bytes stay
outside Git and public binary caches. No training or weight download occurs.

At startup the executable checks the footer bounds, aligned payload offset,
protocol, defaults, and streaming SHA-256. It opens `/proc/self/exe`, seeks to
the GGUF header, and calls the pinned native `llama_model_load_from_file_ptr`.
llama.cpp maps that same executable inode with its normal CPU mmap loader.
There is no extraction, temporary model, memfd, or separate weight allocation.
The small public Rust wrapper is vendored with one documented FILE-pointer
method; the native llama.cpp source remains unchanged. Upstream provenance
is recorded in `tools/tabcomplete_engine/vendor/llama-cpp-2/UPSTREAM.json`.

## Private build inputs

`models.nix` pins the existing authorized GGUF files. Import each private file
once, retaining the declared filename:

```sh
nix-store --add-fixed sha256 /path/to/q25-public-functional-lr1e5-Q4_K_M.gguf
nix-store --add-fixed sha256 /path/to/sweep-next-edit-1.5b.q4_k_m.gguf
```

The GGUF payloads total 1,374,688,864 bytes. The two executable outputs add
only two copies of the finalized engine and small aligned footers. The
runtime closure has no references to the separate GGUF build inputs or the
unembedded engine output. Old Nix generations, existing source imports, and
user backups may retain additional copies. Preserve them until the user
chooses to retire them; the new-output budget does not imply the entire
store contains fewer than 2 GiB of model bytes.

Cargo dependencies are locked and vendored before the offline Nix build.
The derivation provides CMake, GCC, pkg-config, libclang, and bindgenHook.
It targets `x86-64-v3`, uses at most two native compiler jobs, and disables
GPU backends and CPU repacking. The ThinkPad i7-8650U supports AVX2.
The footer is appended after stripping and ELF fixups; never strip, patchelf,
or append other data to a completed embedded executable.

## Declarative Home Manager integration

Pin the public source with `pkgs.fetchFromGitHub` using the final tested
commit and unpacked source hash. Import that source's `nix/home-manager.nix`
only from the ThinkPad home configuration. The module's default package
uses the same source and `services.tabcomplete.modelVariant`:

```nix
services.tabcomplete = {
  enable = true;
  modelVariant = "qwen"; # Change to "sweep" for the other fixed executable.
  experimentalAutoOptIn = true;
  automaticNormalMode = true; # Also predict after a normal-mode cursor pause.
};
```

The running executable cannot switch to another model. Change the Nix
selection and use the user's normal declarative rebuild. The production
unit passes compute settings only; it has no external model path, model
registry, or selected-model state file. Explicit `--model` and its required
`--model-sha256` remain available on the unembedded engine for research.

The user service binds `127.0.0.1:19094`, permits one active inference task,
limits memory to 1500 MiB with no swap, and uses four CPU threads. Defaults
are context 2304 tokens, input 1024, batch 256, microbatch 64, and f16 KV.
Fixed output budgets are 64 tokens for Qwen and 192 for Sweep. Microbatch
size is configurable within the batch size; changing it needs measurement.
Qwen defaults to `cursor-last-v1`; Sweep uses `sweep-window-v1`.
Automatic training and content telemetry capture are disabled.

The immutable Neovim plugin spec replaces only
`lazyvim/lua/plugins/tabcomplete-trajectory.lua`. It retains the existing
collector and configurable acceptance/dismissal mappings. Automatic display
requires opt-in and every edit requires acceptance. The dedicated persisted
mode file preserves later manual, shadow, or off choices.

With `automaticNormalMode`, normal-mode cursor movement, buffer entry, and edits
use the same 250 ms debounce and single-request state machine. Visual, operator
pending, replace, terminal, and command-line modes remain excluded. Cursor movement
closes a shown proposal as `dismissed_navigation`, never an incorrect prediction
or negative preference label. Unseen requests are cancelled. An ordinary
normal-mode edit is an editor-change observation, not an inferred typing rejection.
The insert-mode prefix filter protects typed text. Normal-mode next-edit proposals
may replace text before the cursor; preview, syntax, range, and acceptance checks
still apply. This is a display-policy change, not an accuracy improvement.
Explicit `:TabCompletePredict` works in normal mode without changing the persisted
automatic mode. Completion menus, snippet ownership, focus, staleness, and output
validation still apply. Status reports `automatic_block_reason` precisely.

The Home Manager activation hook invalidates only the owned spec's bytecode
cache after `linkGeneration`. It backs up that entry and preserves every
other cache entry. Verify the deterministic collision regression with:

```sh
bash ./nix/tests/cache-invalidation.sh
```

Evaluate the integrated configuration before the user's normal rebuild.
Do not invoke `nrs` (the remote repository forbids it), a full Home Manager
activation, or the historical standalone installer during this campaign.
The old standalone scripts remain solely for earlier installation rollback
records; current installation is declarative. Preserve all unrelated dirty
Nix changes and existing generations.

## Verification

Build the package against the user's pinned package set without activation,
then inspect both executables and `share/tabcomplete/runtime-manifest.json`.
Verify each process maps its own executable, reports the expected embedded
hash/protocol, and rejects runtime model switching. Check actual synthetic
editor behavior and collector receipts before calling deployment complete.
A successful build or health response alone does not verify edit quality,
automatic display, acceptance, feedback capture, or latency.
