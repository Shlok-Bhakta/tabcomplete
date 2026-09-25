# Automatic feedback R1 handoff

The installed desktop plugin points to this worktree. Keep the checkout and the selected GGUF in place while the service is active. The model service is `tabcomplete-predictor.service` on loopback port 19093. The collector is the existing `tabcomplete-collector` container on tailnet address `100.100.163.102:8787`, backed by `/mnt/ssd/collector-data/collector.sqlite` at schema version 2.

Open LazyVim and type in an eligible file. The mode starts as `automatic experimental`; the status is available with `:TabCompleteStatus`. `<M-l>` accepts the still-current one-line proposal. Continuing to type dismisses it. `:TabCompleteReject` or `<M-BS>` explicitly rejects. `:TabCompleteMode off` stops suggestions immediately; `:TabCompleteMode automatic` resumes. Manual and shadow modes remain available. `automatic_quality_validated=false` and automatic personalization stays disabled.

The synthetic verification session is `d849500c-abda-4559-894c-36d3a94f97f9`. Run read-only replay with:

```sh
cd /home/crabcake/Projects/tabcomplete-automatic-feedback-r1/tools/trajectory_collector/analysis
bun replay.ts --file main.py --repo repo:ac02ac0c2f78c692b0f8e75ae1b007aa010f4ca2f1d992d30bc09c2dbfe164c1 --session d849500c-abda-4559-894c-36d3a94f97f9 --db /mnt/ssd/collector-data/collector.sqlite
```

Versioned evidence export is at `/mnt/ssd/collector-data/exports/automatic-feedback-r1.json`. It is local data, not a committed training set. Human provenance remains unverified unless explicitly established later. Recheck sequence gaps, anchors, and delayed outcomes before building preference pairs.

## Rollback

For an immediate pause, run `:TabCompleteMode off`; this persists across editor restarts. To restore the prior editor/service configuration, copy the backed-up files listed in `pre_install_backups.json` over their active paths, then run `systemctl --user daemon-reload` and `systemctl --user restart tabcomplete-predictor.service`. The current installer also made timestamped backups ending `20260925T201018Z`. Restart Neovim after restoring its config.

The schema v2 migration is additive, and the previous collector server can ignore the added table. Rebuild the old container from the original checkout if needed. Do not replace the live database with the pre-migration backup as a routine code rollback: that would discard feedback recorded since backup. A WAL-consistent pre-migration database backup exists at `/mnt/ssd/collector-data/backups/collector-before-automatic-20260925T191646Z.sqlite` for disaster recovery.

## Operational checks

`systemctl --user status tabcomplete-predictor.service` and `curl http://127.0.0.1:19093/health` check local inference. `docker inspect tabcomplete-collector --format '{{.State.Health.Status}}'` and `curl http://100.100.163.102:8787/healthz` check the collector. `:TabCompleteStatus` shows mode, last latency, current request state, local counters, and collector connectivity. No model download, retraining, or automatic personalization is part of startup.
