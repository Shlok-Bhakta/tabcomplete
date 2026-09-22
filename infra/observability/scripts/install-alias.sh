#!/usr/bin/env bash
set -euo pipefail
obs_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/tabcomplete-observability"
unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
[[ -f "${config_dir}/alias.env" ]] || { echo 'Create restricted alias.env first' >&2; exit 2; }
install -m 0644 "${obs_dir}/gateway/src/alias.ts" "${config_dir}/alias.ts"
temporary=$(mktemp)
trap 'rm -f "${temporary}"' EXIT
sed -e "s|@@CONFIG@@|${config_dir}|g" -e "s|@@BUN@@|$(command -v bun)|g" "${obs_dir}/config/tabcomplete-monitor-alias.service.in" >"${temporary}"
install -m 0644 "${temporary}" "${unit_dir}/tabcomplete-monitor-alias.service"
systemctl --user daemon-reload
systemctl --user enable --now tabcomplete-monitor-alias.service
