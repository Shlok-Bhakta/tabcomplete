# ThinkPad declarative FIM update

The ThinkPad is reachable and now serves the previously selected direct FIM
Qwen2.5-Coder model through the Rust CPU engine. The new declarative LazyVim
configuration starts in automatic experimental mode, including normal-mode
cursor prediction. Quality validation remains false and automatic training is
disabled. Alt+l accepts, Alt+p forces a request in normal or insert mode, and
`:TabCompleteMode off` stops predictions.

Already-open user Neovim processes were preserved. They must restart to load
the new plugin and model allowlist. Fresh installed LazyVim startup was verified
headlessly. No visual inspection or genuine human acceptance was claimed.

## Declarative deployment

Host `shlokthinkpad`, authorized SSH alias `thinkpad`, has an Intel i7-8650U,
four physical cores, eight logical CPUs, and 24,932,175,872 bytes RAM. Inference
is CPU-only through llama.cpp bindings in Rust, loopback port 19094.

The real configuration is `/home/shlok/nixos-config`. It already contained
staged and unstaged work. Only these existing files were changed:

- `pkgs/tabcomplete-engine/default.nix`
- `home/features/tabcomplete/default.nix`

The source pin is `67c8c51748f12d10a2fbca454bc151b1cca35e4b`, with
fetchFromGitHub hash `sha256-YaZVg5Gq29wRFyG719ltphuNZ739sJZM+U0z8s3sb04=`.
Private model inputs use `requireFile`, registered with `nix-store --add-fixed
sha256`. No new model weights were downloaded or published.

GGUF Q4_K_M SHA-256 is
`ff43d25913e982c3580614ad0528722d9b261b6575d4e06349f9b8509b368682`,
491,399,808 bytes. The profile SHA is
`39f043a548e7274cd868d12889f749f1cf969368aaa57f7a54c29d0fc40f11a3`.
The actual running embedded model segment was independently rehashed.
Executable SHA is
`4100663f5672dff594bbfebc286cf737d2ebf5474379b9c0fc39c9b241ec24de`.

The package is
`/nix/store/ys9q2mv7v716n0faskhpa4sv8k999854-tabcomplete-q25-fim-candidate-0.1.0`.
The activated user Home Manager generation is
`/nix/store/85p7q1nazz43qqsahddicq2341whnbi6-home-manager-generation`.
Baseline evaluation of the previously dirty configuration produced exactly the
previous active generation. The new generation changed only the TabComplete
plugin spec, service and wants link, and a fontconfig reference induced by the
changed Home Manager package path. Other packages stayed unchanged.

This was user Home Manager activation followed by explicit service restart.
The agent did not run `nixos-rebuild switch`. Full system evaluation passed for
`nixosConfigurations.shlokthinkpad.config.system.build.toplevel.drvPath`.
The user subsequently reported running the system switch. A fresh read-only
SSH check found `tabcomplete-engine.service` active and the selected FIM model
health check passing. No reboot is needed for TabComplete. Open Neovim processes
still need a restart. The repeatable system switch command is:

```sh
sudo nixos-rebuild switch --flake ~/nixos-config#shlokthinkpad
```

The existing `nrs` alias is `nh os switch | lolcat`. Because Home Manager is
integrated into the NixOS configuration, the full system switch is needed to
make the NixOS-owned activation at future boot use the new generation too.

An initial startup failed because the generic CLI only accepts legacy layout
names. It was immediately rolled back to the working provider. The corrected
Nix package declares a launch-only `cursor-last-v1` compatibility argument.
The embedded FIM route selects and reports its fixed trained
`q25-fim-psm-bounded-v2` layout. Prompt construction and weights did not change.
The failed attempt and successful recovery are preserved in the receipts.

## Measured laptop behavior

These are eight real requests per provider, two repetitions of four fixed
synthetic source states, with existing user applications retained. The selected
FIM replay ran after its Nix build completed. A competing-process trace proving
a controlled interval was not collected. These are a small
normal-workload slice, not a controlled desktop replacement or a quality test.
The providers differ in weights, output protocol, and retained context, so this
is not a pure runtime ablation.

| Observation | Previous next-edit Qwen | Selected FIM Qwen |
|---|---:|---:|
| Actual input tokens | 392–393 | 643 |
| Changed-state completed HTTP action median | 700.6 ms | 3,719.9 ms |
| Same-prompt repeat | 434.9 ms | 79.6 ms |
| Resident first uncached request | 2,672.1 ms | 3,709.0 ms |

Exact memory, nearest-rank p95, swap counters, pressure snapshots, and retained
memory are in `runtime_summary.json` and the two raw replay receipts. Selected
service peak RSS was 592,429,056 bytes. RSS/PSS cover the model service; editor
plugin allocation and curl helper peak were not separately measured. No swap
activity occurred in the selected window, and memory pressure remained absent.

The service uses four generation and prompt threads, context 2,304, input budget
1,024, output ceiling 96, batch 256, microbatch 64, F16 KV, one active slot, and
zero optional saved contexts. Its active sequence cache remains enabled. The
systemd memory cap remains 1500M with service swap disabled.

The replay exposes a specific bottleneck. A sliding last-640-token prefix window
moves its starting token when typing changes the total prefix token length.
Those transitions reused only one token and spent about 3.65–3.77 seconds in
prefill; generation took about 21–52 ms. Compatible changed states reused 640
tokens and completed in about 94–95 ms. The identical prompt reused 642 tokens.
This is evidence of useful cache reuse and of invalidation on changing windows.
A faster context policy needs a separately versioned quality comparison.
No prompt reorder, information removal, kernel project, or training was used to
hide the bottleneck. A thread ablation was deferred while the user prepared a
system switch, preserving the declared four-thread setup.

## Editor and collector verification

Four headless integration scripts passed:

1. Installed configuration, exact model identity, mappings, off cancellation,
   and persisted automatic mode.
2. Full installed LazyVim startup in a fresh process.
3. Actual automatic model proposals, preview extmarks without source mutation,
   Alt+l application, separate undo, divergent typing dismissal, typed match,
   and unseen navigation cancellation.
4. Actual normal-mode automatic proposal, Alt+p forced inference, Alt+l
   acceptance, undo, and navigation without negative feedback.

The insert-mode script uses the existing documented headless mode seam; buffer
deltas, inference, rendering extmarks, mappings, API transport, and database are
real. Every scripted proposal and decision is marked synthetic.

Main session `8fd95af8-42a2-4419-82cc-c7b5add06131` has four requests and three
shown proposals. Acceptance prediction
`698363e5-8564-4eaa-ac4d-c662fdc5597a`, typing dismissal
`9ad780f0-b28f-4447-85ac-e2349b0ed179`, and typed match
`19788916-3886-408d-b24c-db2050ea5f11` link to the raw request, display, outcome,
and subsequent deltas. The unseen navigation request remains an unseen lifecycle
observation, not an incorrect-model label.

The existing SQLite database is `/mnt/ssd/collector-data/collector.sqlite` on
crabcake. No migration, reset, or duplicate database was needed. Replay used the
explicit server repository identity
`repo:4db24991f518e57a4107046c346d06e1e9caa8b962bbc57ea37ad14965b3983a`.
Fifteen anchors and eight deltas reconstructed the final 276 bytes exactly,
with zero mismatches, gaps, or unanchored deltas. Ten context, response, and
canonical-action blob references were retrieved and their full hashes verified.
Reposting 47 acknowledged events in reverse order ingested zero new events,
skipped all 47 duplicates, and preserved exactly four prediction projections.
The exporter excluded these synthetic records from preference pairs.

The separate actual normal-mode session is
`14939e18-81ad-430c-9ad9-25f5f018467d`. It made exactly two requests. Its first
proposal was automatic; its second was triggered through Alt+p.

Nix parsing and formatting passed. The Nix Rust build ran 45 unit tests,
45 passed and zero failed. The existing FIM embedded-appender test passed.
These checks do not establish general model accuracy. No RL or personalization
training occurred.

## Rollback and limits

Private backups are under
`/home/shlok/.local/share/tabcomplete-fim-candidate-r1/backup`. The old generation
and approved fixed inputs have explicit GC roots. The redundant uploaded GGUF
copy was removed after verification, leaving the fixed input and embedded
executable. Existing models and user research checkpoints were preserved.

To restore the prior user deployment:

```sh
/nix/store/21wajrcb2pqqkbbsqk0j1mh9jpr2amdl-home-manager-generation/activate
systemctl --user daemon-reload
systemctl --user restart tabcomplete-engine.service
```

Restore the two backed-up declarative files after reviewing any later user
changes before a future system rebuild. Restart Neovim after either activation.
The new model is experimental. Long-prefix requests can take several seconds;
fresh short-file proposals and cache-compatible edits can be much faster.
Human accuracy and acceptance remain uncalibrated. Visual inspection, complete
helper-memory measurement, and a controlled no-user-workload replay were not
performed. Automatic personalization remains disabled.
