# ThinkPad FIM deployment handoff

The laptop now runs the selected direct FIM Q4 model through the declarative
TabComplete package. Full report: `../thinkpad_fim_update_r1.md`.

- Source pin: 67c8c51748f12d10a2fbca454bc151b1cca35e4b.
- Active user generation: /nix/store/85p7q1nazz43qqsahddicq2341whnbi6-home-manager-generation.
- Package: /nix/store/ys9q2mv7v716n0faskhpa4sv8k999854-tabcomplete-q25-fim-candidate-0.1.0.
- Actual model: ff43d25913e982c3580614ad0528722d9b261b6575d4e06349f9b8509b368682, Q4_K_M.
- Automatic experimental mode enabled, normal-mode automatic enabled, quality
  validation false, training disabled. Alt+l accepts, Alt+p forces, mode off stops.
- Service explicitly restarted. Actual executable and embedded GGUF rehashed.
- User Neovim buffers/processes were not touched. Existing sessions must restart.
- User Home Manager activation ran. No agent system rebuild ran. Full-system
  toplevel evaluation passed. The user is preparing their declarative system switch.
- Preserve the pre-existing dirty/staged work under /home/shlok/nixos-config.
  Only two TabComplete declaration files changed. Backups and verified model/profile
  input GC roots are under ~/.local/share/tabcomplete-fim-candidate-r1.

Four headless integration scripts passed, including real automatic proposals,
actual normal-mode and key-dispatched force/accept, undo, typing dismissal,
typed match, unseen navigation, and fresh full LazyVim startup. All decisions
are synthetic. Main session 8fd95af8-42a2-4419-82cc-c7b5add06131 had four requests,
three displays, one acceptance, implicit typing dismissal, and typed match.
Existing SQLite replay passed with 15anchors/8 deltas and exact 276-byte final state;
ten payload hashes verified; 47replayed events all deduplicated, projection count 4.
The actual normal-mode session is 14939e18-81ad-430c-9ad9-25f5f018467d.

Nix Rust build ran 45 unit tests, all passed. Existing embedded-appender test passed.
An initial CLI layout startup failure was rolled back and corrected explicitly.
The fixed engine receives a supported launch-only cursor-last-v1 argument while
its embedded FIM route always serves q25-fim-psm-bounded-v2.

Long-prefix changed-state median is 3719.9ms versus 700.6ms for the previous model's
shorter retained prompt. These differ in model/protocol/context. FIM repeat 80ms;
compatible changed state 94-95ms; sliding 640-token prefix changes caused 1-token
reuse and 3.65-3.77s prefill. Root should investigate a separately versioned stable
window policy, with quality controls, after the training comparison. Do not claim
all normal requests are 80ms. Service peak RSS 592429056 B; helper peaks unknown.

Raw reproduction scripts and logs remain at
/home/shlok/.local/share/tabcomplete-fim-candidate-r1 on the target, and scratch
/mnt/ssd/tabcomplete-q25-completion-scale-r1/thinkpad on crabcake. No new weights
were downloaded, trained, requantized, or published for this deployment.
