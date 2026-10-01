# ThinkPad TabComplete rollback

The installed plugin includes the tested reload-capture fix. Packaging and
owned-spec cache handling are pinned to public commit
`f96d32f7e79945aea432a64ee88775313a58a713`. The exact installed arguments,
hashes, private backup paths, and permanent Nix evaluation are recorded in
`installed_configuration.json` and `permanent_nix_evaluation.json` beside
this document.

## Restore the preceding working installation

Run on the ThinkPad after the editor audit or inference controller has ended:

```sh
bash /nix/store/4klv76s00fnsk88jq10bd978xn7818gz-source/nix/install-standalone.sh \
  --rollback /home/shlok/.local/state/tabcomplete-install-backups/20261001T062427-232339
nix-store --add-root /home/shlok/.local/state/tabcomplete/standalone-current \
  --indirect --realise /nix/store/yrw2qi9fqmdxmj1z20mvpfcj1kpl2r0n-tabcomplete-standalone-qwen
```

This restores the preceding unit and collector spec, reloads user systemd,
and restores its enabled/active state. The new immutable installer also
backs up and invalidates only the owned plugin-spec bytecode cache on rollback. The preceding standalone closure is
also retained by the backup's `previous-standalone` GC root. Open a fresh
LazyVim session to load the restored plugin. Model alias and dedicated mode
preferences remain available.

To restore the preceding permanent public code pin as well, use these guarded
commands while the focused Nix changes remain uncommitted:

```sh
cd /home/shlok/nixos-config
backup_dir=/home/shlok/.local/state/tabcomplete-rust-editor-r1/permanent-nix-backups/20261001T065025-243557
test "$(git rev-parse HEAD)" = "$(cat "$backup_dir/head")" || exit 1
sha256sum --check "$backup_dir/focused-installed.sha256" || exit 1
cp -a "$backup_dir/prior-config/pkgs/tabcomplete-engine/default.nix" pkgs/tabcomplete-engine/default.nix
git add -- pkgs/tabcomplete-engine/default.nix
```

The checks stop if HEAD or any focused file changed afterward. This restores
the previous commit `f93b388089bc6dcc92fbbb9f5900e5a5c11955df` without applying
any full Home Manager or NixOS configuration. Unrelated staged and unstaged
changes are preserved.

## Remove the original integration

The original unit/spec backup is
`/home/shlok/.local/state/tabcomplete-install-backups/20261001T055602-220940`.
Pass it to the same `--rollback` command to restore the pre-install files and
service state. The original permanent Nix backup is
`/home/shlok/.local/state/tabcomplete-rust-editor-r1/permanent-nix-backups/20261001T060317-223324`.
It contains the original ThinkPad home file, focused patches, and private Git
diff/index metadata.

To remove the three focused Nix changes while the current `f96d32…` pin remains
uncommitted:

```sh
cd /home/shlok/nixos-config
original_backup=/home/shlok/.local/state/tabcomplete-rust-editor-r1/permanent-nix-backups/20261001T060317-223324
latest_backup=/home/shlok/.local/state/tabcomplete-rust-editor-r1/permanent-nix-backups/20261001T065025-243557
test "$(git rev-parse HEAD)" = "$(cat "$latest_backup/head")" || exit 1
sha256sum --check "$latest_backup/focused-installed.sha256" || exit 1
cp -a "$original_backup/thinkpad-home.nix" hosts/thinkpad/home.nix
git restore --staged -- pkgs/tabcomplete-engine/default.nix home/features/tabcomplete/default.nix
rm -- pkgs/tabcomplete-engine/default.nix home/features/tabcomplete/default.nix
```

If the preceding permanent pin was already restored, use
`/home/shlok/.local/state/tabcomplete-rust-editor-r1/permanent-nix-backups/20261001T062537-232741/focused-installed.sha256`
for that checksum check. The remote
`AGENTS.md` prohibits agents from running `nrs` and asks the user to run it.

## Reinstall the tested reload fix

```sh
bash /nix/store/4klv76s00fnsk88jq10bd978xn7818gz-source/nix/install-standalone.sh \
  /nix/store/77vqqf5a43zf6ifm1a084v50casj431q-tabcomplete-standalone-qwen
systemctl --user status tabcomplete-engine.service --no-pager
curl --fail --silent http://127.0.0.1:19094/health
curl --fail --silent http://127.0.0.1:19094/v1/models
```

Reinstallation creates a new private backup and restarts the single service.
Expected Qwen runtime hash:
`6e58a95c8f5fba5581784636cf5e042a33a14e93b7c764f681eb2ff4c10c54ba`.
Runtime identity includes model, compute settings, and bundle paths; its hash
changes when those change.
