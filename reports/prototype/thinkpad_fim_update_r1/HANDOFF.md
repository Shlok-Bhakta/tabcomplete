# ThinkPad FIM update in progress

Authorized target `ssh thinkpad`, shlok@100.111.119.76, host shlokthinkpad.
Real declarative configuration is `/home/shlok/nixos-config`, branch main at
fc29c3618557e4ecbaf73f4ae6f0ac5330bed425 with pre-existing staged and unstaged
changes. Preserve them. No sudo is available; no system rebuild has run.

Backups and private working artifacts are at
`/home/shlok/.local/share/tabcomplete-fim-candidate-r1`.
The baseline pending Home Manager configuration was evaluated and built to
exactly the currently active generation:
`/nix/store/21wajrcb2pqqkbbsqk0j1mh9jpr2amdl-home-manager-generation`.
This provides a baseline for reviewing the scope of activation.

Only `pkgs/tabcomplete-engine/default.nix` and
`home/features/tabcomplete/default.nix` were changed remotely. The source is
pinned to a86bd47889742cced2166a91a82581db7d30b4e0 with fetchFromGitHub hash
`sha256-iyhAjZIi0KGEAQ5FP9jDahrL6F4XFMKWCE2F0btcp4w=`. The selected private
GGUF and tokenizer profile are registered with `nix-store --add-fixed sha256`
and required declaratively, without public publication or model downloads.

The candidate Home Manager generation is currently building. Its output and
stderr are in `home-generation-candidate.txt` and `home-build-candidate.stderr`
under the private artifact directory. Do not activate until the managed file
and package changes are reviewed against the exact baseline above.

The old served model SHA is
4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb,
protocol single-line-edit-v1. The intended FIM SHA is
ff43d25913e982c3580614ad0528722d9b261b6575d4e06349f9b8509b368682,
491,399,808 bytes, protocol q25-fim-line-completion-v1.
The copied embedded profile SHA is
39f043a548e7274cd868d12889f749f1cf969368aaa57f7a54c29d0fc40f11a3.

The existing baseline remains active on localhost:19094. Eight real requests
in two repetitions ran before the build, retaining existing user applications.
Their raw measurements are in runtime_baseline.json. This is an exploratory
short-context normal-workload replay, not a controlled benchmark or a pure
systems ablation when compared with the new weights and protocol.

Next: finish Nix build, review scope, coordinate with root, activate Home
Manager as the user, explicitly restart the changed service, verify actual
binary/model/arguments and installed automatic configuration. Run real-model
headless Neovim acceptance/undo/dismissal/match/staleness with all scripted
prediction outcomes marked synthetic, query and reconstruct through the
existing collector database, verify restart persistence, and measure bounded
new-model latency and memory. Preserve automatic personalization disabled.
