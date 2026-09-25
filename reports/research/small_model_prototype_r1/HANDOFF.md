# Small-model prototype R1 handoff

## What is running

Desktop `crabcake` serves the selected adapted q25-coder Q4_K_M GGUF through the user unit `tabcomplete-predictor.service` on loopback port 19093. The installed LazyVim plugin starts in **manual experimental** mode. The T480s timed out and has neither an installation nor measurements. The research worktree and ignored model file must remain at their current paths while the user unit points to them.

- Selected GGUF: `/home/crabcake/Projects/tabcomplete-small-model-prototype-r1/artifacts/research/small_model_prototype_r1/selected_runtime/small_model_prototype_r1_selected_conversion/q25-coder-adapted-Q4_K_M.gguf`
- SHA-256: `a45bc50aa7c74a03e7bdcade90315b052b31675a97631ed5c96df953160fd626`
- Source adapted weight SHA-256: `ad22285111aba002f6f7a322bdac522840554e6f0926f545006b6d77ef837927`
- Source model and tokenizer: `Qwen/Qwen2.5-Coder-0.5B@8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`
- Runtime: llama.cpp `f072b103714dfa1eee531f80b24512faf38e3dd2`

The existing LazyVim file was backed up as `/home/crabcake/.config/nvim/lua/plugins/tabcomplete-trajectory.lua.backup-20260925T100355Z`. Its active path is `/home/crabcake/.config/nvim/lua/plugins/tabcomplete-trajectory.lua`. The installed user unit is `/home/crabcake/.config/systemd/user/tabcomplete-predictor.service`. The [installation record](deployment/installation.json) has their hashes and the installed process snapshot. This is a local user service; it is not exposed on the network.

In LazyVim, use `:TabCompletePredict` to request, `:TabCompleteAccept` to apply an active proposal, `:TabCompleteReject` for an explicit rejection, `:TabCompleteMode manual|shadow|off` to change safe modes, and `:TabCompleteStatus` to inspect state. Automatic mode is gated and denied. Preview a multiline replacement before accepting. Existing completion and snippet Tab behavior remains in charge.

## Check and rollback

Run `systemctl --user status tabcomplete-predictor.service` and `curl --silent --show-error http://127.0.0.1:19093/health` for service health. Check the model with `sha256sum` against the hash above. To stop serving, run `systemctl --user disable --now tabcomplete-predictor.service`. To restore the prior editor config, copy the named backup to the active LazyVim plugin path, then restart Neovim. Preserve the service model path or update the unit if moving the ignored artifact; moving the worktree first will break the service.

For another owned machine with an authorized local runtime and a copied, verified selected GGUF, the installer is `uv run python scripts/install_small_model_lazyvim.py --model /absolute/path/to/q25-coder-adapted-Q4_K_M.gguf --runtime /absolute/path/to/llama-server --model-sha256 a45bc50aa7c74a03e7bdcade90315b052b31675a97631ed5c96df953160fd626 --model-alias q25-coder --dry-run`. Review its config ownership and hardware before omitting `--dry-run`. Do not describe this as T480s installation evidence.

## Evidence and continuation

The frozen plan and result identities are in `plan.json`, `models.json`, `selection.json`, `deployment/selection.json`, and `artifact_manifest.json`. The main report explains the screen and adaptation. Kaggle jobs: `shlokbhakta/tabcomplete-small-model-prototype-r1-screening-v2`, `shlokbhakta/tabcomplete-small-model-prototype-r1-adaptation` (version 5), `shlokbhakta/tabcomplete-r1-heldout`, and CPU-only `shlokbhakta/tabcomplete-r1-conversion`. Earlier failed attempts are retained and counted in `budget_ledger.json`. The four replay-verified selected-model collector session IDs and anchor hashes are in `feedback_verification/actual_selected_model.json`; scripted decisions are synthetic.

To refresh the read-only, ignored, versioned feedback export, run `uv run python scripts/export_personalization_feedback.py --output artifacts/research/small_model_prototype_r1/personalization_feedback_v2_final.json`. It currently has zero defensible human preference pairs; automatic personalization is disabled. Do not use synthetic scripted accepts/rejects to train. Future training requires verified pairs and sessions, a temporal holdout, generic regression examples, memory and contention checks, and a separate candidate with rollback. A sustained idle window, power check, memory pressure check, and cooldown would be required for any later scheduled trainer.

The current desktop Q4 task latency is 1.236 s median and 1.933 s p95, above the automatic gates. Real editing correctness and reversals are uncalibrated. Keep manual mode until at least 50 qualified human displayed proposals and all registered gates pass. The T480s needs independent measurement before automatic mode there.
