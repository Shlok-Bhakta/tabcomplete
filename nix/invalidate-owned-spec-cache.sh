#!/usr/bin/env bash
set -Eeuo pipefail

neovim_binary="${1:?usage: invalidate-owned-spec-cache.sh NEOVIM APP_NAME SPEC_PATH BACKUP_DIRECTORY}"
app_name="${2:?}"
owned_spec="${3:?}"
cache_backup_directory="${4:?}"
test "$(basename -- "$owned_spec")" = tabcomplete-trajectory.lua

# Use the installed Neovim's normalization and URI encoding for its exact key.
# A Nix symlink replacement can retain both file size and reproducible mtime.
owned_cache=$(
  NVIM_APPNAME="$app_name" TABCOMPLETE_OWNED_SPEC_PATH="$owned_spec" \
    "$neovim_binary" --headless -u NONE -i NONE --noplugin \
    -c 'lua local p = vim.fs.normalize(vim.env.TABCOMPLETE_OWNED_SPEC_PATH); io.write(vim.fn.stdpath("cache") .. "/luac/" .. vim.uri_encode(p, "rfc2396") .. "c")' \
    -c 'qa!'
)

if test -f "$owned_cache" || test -L "$owned_cache"; then
  umask 077
  mkdir -p -- "$cache_backup_directory"
  cache_backup="$cache_backup_directory/tabcomplete-trajectory-$(date -u +%Y%m%dT%H%M%S)-$$.luac"
  cp -a -- "$owned_cache" "$cache_backup"
  printf '%s\n' "$owned_cache" > "$cache_backup.source-path"
  rm -- "$owned_cache"
  printf 'Invalidated owned TabComplete spec cache; backup: %s\n' "$cache_backup"
fi
