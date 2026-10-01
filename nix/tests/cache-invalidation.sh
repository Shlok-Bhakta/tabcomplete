#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
test_root=$(mktemp -d)
trap 'rm -rf -- "$test_root"' EXIT
neovim_binary="${NEOVIM_BINARY:-$(command -v nvim)}"
helper=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)/invalidate-owned-spec-cache.sh
mkdir -p "$test_root/config/lazyvim/lua/plugins" "$test_root/cache"
export XDG_CONFIG_HOME="$test_root/config"
export XDG_CACHE_HOME="$test_root/cache"
export NVIM_APPNAME=lazyvim
spec="$test_root/config/lazyvim/lua/plugins/tabcomplete-trajectory.lua"
printf 'return { { dir = "old-path" } }\n' > "$test_root/spec-old.lua"
printf 'return { { dir = "new-path" } }\n' > "$test_root/spec-new.lua"
printf 'return { unrelated = true }\n' > "$test_root/config/lazyvim/lua/plugins/unrelated.lua"
touch -d @1 "$test_root/spec-old.lua" "$test_root/spec-new.lua"
test "$(stat -c '%s %Y' "$test_root/spec-old.lua")" = "$(stat -c '%s %Y' "$test_root/spec-new.lua")"
ln -s "$test_root/spec-old.lua" "$spec"
"$neovim_binary" --headless -u NONE -i NONE --noplugin \
  -c 'lua vim.loader.enable(); local p=vim.fn.stdpath("config").."/lua/plugins/"; assert(assert(loadfile(p.."tabcomplete-trajectory.lua"))()[1].dir=="old-path"); assert(assert(loadfile(p.."unrelated.lua"))().unrelated)' -c 'qa!'
ln -sfn "$test_root/spec-new.lua" "$spec"
"$neovim_binary" --headless -u NONE -i NONE --noplugin \
  -c 'lua vim.loader.enable(); assert(assert(loadfile(vim.fn.stdpath("config").."/lua/plugins/tabcomplete-trajectory.lua"))()[1].dir=="old-path"); print("Reproduced stale bytecode with matching size and mtime")' -c 'qa!'
owned_cache=$(rg --files "$test_root/cache" | rg 'tabcomplete-trajectory\.luac$')
cp -a "$owned_cache" "$test_root/owned-before.luac"
snapshot_other_cache() {
  while IFS= read -r path; do
    if test "$path" != "$owned_cache"; then sha256sum "$path"; fi
  done < <(rg --files "$test_root/cache" | LC_ALL=C sort)
}
snapshot_other_cache > "$test_root/other-before.sha256"
bash "$helper" "$neovim_binary" lazyvim "$spec" "$test_root/backup-install"
test ! -e "$owned_cache"
snapshot_other_cache > "$test_root/other-after.sha256"
cmp "$test_root/other-before.sha256" "$test_root/other-after.sha256"
saved_cache=$(rg --files "$test_root/backup-install" | rg '\.luac$')
cmp "$test_root/owned-before.luac" "$saved_cache"
bash "$helper" "$neovim_binary" lazyvim "$spec" "$test_root/backup-install"
test "$(rg --files "$test_root/backup-install" | rg '\.luac$')" = "$saved_cache"
cmp "$test_root/owned-before.luac" "$saved_cache"
test ! -e "$owned_cache"
"$neovim_binary" --headless -u NONE -i NONE --noplugin \
  -c 'lua vim.loader.enable(); assert(assert(loadfile(vim.fn.stdpath("config").."/lua/plugins/tabcomplete-trajectory.lua"))()[1].dir=="new-path"); print("Focused install invalidation loads new spec")' -c 'qa!'
ln -sfn "$test_root/spec-old.lua" "$spec"
bash "$helper" "$neovim_binary" lazyvim "$spec" "$test_root/backup-rollback"
"$neovim_binary" --headless -u NONE -i NONE --noplugin \
  -c 'lua vim.loader.enable(); assert(assert(loadfile(vim.fn.stdpath("config").."/lua/plugins/tabcomplete-trajectory.lua"))()[1].dir=="old-path"); print("Focused rollback invalidation loads preceding spec")' -c 'qa!'
snapshot_other_cache > "$test_root/other-final.sha256"
cmp "$test_root/other-before.sha256" "$test_root/other-final.sha256"
printf 'Same-size/same-mtime install and rollback passed; repeated invalidation is idempotent; unrelated cache bytes preserved.\n'
