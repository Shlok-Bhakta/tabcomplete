#!/usr/bin/env bash
set -Eeuo pipefail

unit_relative='systemd/user/tabcomplete-engine.service'
plugin_relative='lazyvim/lua/plugins/tabcomplete-trajectory.lua'
config_root="${XDG_CONFIG_HOME:-$HOME/.config}"
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
neovim_binary=$(command -v nvim)

invalidate_owned_spec_cache() {
  bash "$script_root/invalidate-owned-spec-cache.sh" "$neovim_binary" lazyvim \
    "$config_root/$plugin_relative" "$backup_root/editor-cache-$1"
}

rollback() {
  trap - ERR
  systemctl --user stop tabcomplete-engine.service 2>/dev/null || true
  systemctl --user disable tabcomplete-engine.service 2>/dev/null || true
  for relative in "$unit_relative" "$plugin_relative"; do
    destination="$config_root/$relative"
    saved="$backup_root/$relative"
    if test -f "$saved.previously-absent"; then
      if test -L "$destination" && test "$(readlink -- "$destination")" = "$standalone_source/$relative"; then
        rm -- "$destination"
      fi
    elif test -e "$saved" || test -L "$saved"; then
      temporary="$destination.tabcomplete-rollback-$$"
      cp -a -- "$saved" "$temporary"
      mv -Tf -- "$temporary" "$destination"
    fi
  done
  invalidate_owned_spec_cache rollback
  systemctl --user daemon-reload
  if test "$(cat "$backup_root/service-enabled")" = enabled; then
    systemctl --user enable tabcomplete-engine.service
  fi
  if test "$(cat "$backup_root/service-active")" = active; then
    systemctl --user restart tabcomplete-engine.service
  fi
  printf 'Restored previous TabComplete files from %s\n' "$backup_root"
}

if test "${1:-}" = --rollback; then
  backup_root="${2:?usage: install-standalone.sh --rollback BACKUP_DIRECTORY}"
  standalone_source=$(cat "$backup_root/standalone-source")
  config_root=$(cat "$backup_root/configuration-home")
  rollback
  exit
fi

# Invoke only after the package's real inference and editor checks pass.
standalone_source=$(readlink -f -- "${1:?usage: install-standalone.sh STANDALONE_RESULT}")
case "$standalone_source" in
  /nix/store/*) ;;
  *) printf '%s\n' 'The standalone result must resolve to the immutable Nix store.' >&2; exit 1 ;;
esac

test -f "$standalone_source/$unit_relative"
test -f "$standalone_source/$plugin_relative"
test -x "$standalone_source/package/bin/tabcomplete-engine"
json_query="$standalone_source/bin/jq"
test -x "$json_query"
command -v curl >/dev/null

state_root="${XDG_STATE_HOME:-$HOME/.local/state}"
backup_root="$state_root/tabcomplete-install-backups/$(date -u +%Y%m%dT%H%M%S)-$$"
umask 077
mkdir -p -- "$backup_root"
printf '%s\n' "$standalone_source" > "$backup_root/standalone-source"
printf '%s\n' "$config_root" > "$backup_root/configuration-home"

for relative in "$unit_relative" "$plugin_relative"; do
  existing="$config_root/$relative"
  saved="$backup_root/$relative"
  mkdir -p -- "$(dirname -- "$saved")"
  if test -e "$existing" || test -L "$existing"; then
    cp -a -- "$existing" "$saved"
    if test -L "$existing"; then
      readlink -- "$existing" > "$saved.previous-target"
      cp -L -- "$existing" "$saved.previous-content"
    fi
  else
    : > "$saved.previously-absent"
  fi
done

# Keep dirty configuration evidence locally with restricted permissions.
configuration_checkout='/home/shlok/nixos-config'
if test -d "$configuration_checkout/.git"; then
  git -C "$configuration_checkout" rev-parse HEAD > "$backup_root/configuration-head"
  git -C "$configuration_checkout" status --porcelain=v1 > "$backup_root/configuration-status"
  git -C "$configuration_checkout" diff --binary > "$backup_root/configuration-unstaged.diff"
  git -C "$configuration_checkout" diff --cached --binary > "$backup_root/configuration-staged.diff"
fi
systemctl --user is-enabled tabcomplete-engine.service > "$backup_root/service-enabled" 2>/dev/null || true
systemctl --user is-active tabcomplete-engine.service > "$backup_root/service-active" 2>/dev/null || true

# Register a GC root independently of temporary build result links.
mkdir -p -- "$state_root/tabcomplete"
nix-store --add-root "$state_root/tabcomplete/standalone-current" --indirect --realise "$standalone_source" >/dev/null
trap 'rollback; exit 1' ERR

for relative in "$unit_relative" "$plugin_relative"; do
  destination="$config_root/$relative"
  mkdir -p -- "$(dirname -- "$destination")"
  temporary="$destination.tabcomplete-install-$$"
  ln -s -- "$standalone_source/$relative" "$temporary"
  mv -Tf -- "$temporary" "$destination"
done

invalidate_owned_spec_cache install
systemctl --user daemon-reload
systemctl --user enable tabcomplete-engine.service
systemctl --user restart tabcomplete-engine.service
healthy=false
for attempt in $(seq 1 30); do
  if curl --fail --silent --max-time 2 http://127.0.0.1:19094/health > "$backup_root/health.json" \
      && "$json_query" --exit-status '.status == "ok"' "$backup_root/health.json" >/dev/null; then
    healthy=true
    break
  fi
  sleep 1
done
if test "$healthy" != true; then
  printf '%s\n' 'TabComplete did not become healthy; restoring the previous files.' >&2
  rollback
  exit 1
fi
trap - ERR
printf 'Installed TabComplete; previous files are in %s\n' "$backup_root"
