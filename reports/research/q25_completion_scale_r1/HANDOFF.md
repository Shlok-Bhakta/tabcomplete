# Continuation state

Active branch: `research/q25-completion-scale-r1`.
Active isolated worktree: `/home/crabcake/Projects/tabcomplete-q25-completion-scale-r1`.
Parent pushed commit: `dd7ef12e120da7e74ffc193910b9006f0bd71dfc`.

The user authorizes continued autonomous free Kaggle experiments and asks to
separate a data bottleneck from a compute bottleneck. Do not stop after creating
a runner. Complete CPU preparation, budget enforcement, candidate training,
reload/export checks and paired evaluations. Do not ask for permission to run
this authorized study. Keep the stable editor provider unchanged while training.

Frozen CPU plan: `preparation_plan-r2.json`, SHA256
`be920b0e4e598ae35ad5e5cca85f1a2bc220d0446e2a8f8ba170689230befa2d`.
The earlier revision is preserved. No new comparison outputs existed at freeze.

Three agents are implementing separate files in this shared worktree:

- cpt_data_audit: `scripts/prepare_q25_completion_scale.py` and preparation tests.
- cpt_trainer_audit: existing Q25 FIM trainer/worker with an explicit new plan
  profile and targeted trainer/worker tests; old frozen defaults stay strict.
- cpt_campaign_audit: `scripts/run_q25_completion_scale.py` and controller tests.

The root owns config, plans, reports, integration and execution. Coordinate exact
new schema, development ID offset8192, historical development input file,
unique worker output paths and branch names before allocating.

Fresh authenticated quota: 41.08/45 account GPU-hours at 2026-10-04 14:11:20 UTC;
no active jobs; all55 job states verified; renewal October10. Refresh again
before each allocation. Never automatically consume a renewed allocation.

This experiment's independent caps: 6 aggregate session wall-hours, 3 hours per
session, 12 conservative account GPU-hours, 10 million training input tokens
including discarded/replay work, 12 GiB new artifacts and 2 GiB disk free floor.
One GPU notebook allocation at a time. Finalization reserve at least1800seconds.
Existing artifact/checkpoint trees must not be deleted or overwritten.

Stable desktop provider and collector remain installed from the parent worktree.
No live model, plugin, database, systemd unit or Nix configuration changes are
needed to diagnose training scaling. Automatic personalization remains disabled.

At this checkpoint: no new GPU allocations, no new model training, no new model
downloads. Scientific evidence: `bottleneck_evidence.json`; authenticated quota
receipt: `quota-before-preparation.json`; actual prior SigNoz query:
`prior_monitoring_query.json`. The OpenCode question was answered: no Claude
calls recorded overnight, earlier OpenCode teacher work used Muse Spark.

## Continued October4/5

Final CPU corpus `corpus-r3` verifies4096/8192/512/240 source states.
Exposures input3553816 vs3551513, target68810 vs68720.
Encoded-input duplicate counts are disclosed in `encoded_state_audit.json`;
no encoded train/development overlap. No fixtures changed after outputs.

Controller/trainer/evaluator integration finished. The evaluator accepts both
new512 and historical240 splits and registered source-syntax diagnostics.
Fresh full Python1267pass, Ruffpass, mypy92filespass; workercompiles.
Observer restart works; fresh GPU retries are not implemented by this narrow
two-allocation controller. Complete checkpoint payload reload still requires
actual collected checkpoint verification.

Fresh quota observation before freeze at2026-10-05T02:59:30.346625+00:00
found41.08 accountGPUhours, noactivejobs, renewalOctober10. Refresh again
inexecute. No newGPUallocation at this reportcheckpoint.

ThinkPad is online. A separate declarative update is building to the already
selected FIM model, not either newtrainingcandidate. Nix sourcepin a86bd478
was pushed. Remote dirtyconfig preserved; its baseline HM build exactlymatches
activegeneration. ThinkPad status lives in
`reports/prototype/thinkpad_fim_update_r1/HANDOFF.md`.
