#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
observability_dir=$(cd -- "${script_dir}/.." && pwd)
user_name=$(id -un)
user_root=$(getent passwd "${user_name}" | cut -d: -f6)
config_root="${user_root}/.config/tabcomplete-observability"
state_root="${user_root}/.local/share/tabcomplete-observability"
unit_root="${user_root}/.config/systemd/user"
collector_image="docker.io/otel/opentelemetry-collector-contrib:0.161.0@sha256:b5cf983651c32c3ca13f936deb51742015a54d121f388cac248923ddeb8cc9fc"

install -d -m 0750 "${config_root}" "${state_root}/collector-wal" "${unit_root}"
install -m 0640 "${observability_dir}/config/crabcake-collector.yaml" "${config_root}/collector.yaml"

unit_tmp=$(mktemp)
trap 'rm -f "${unit_tmp}"' EXIT
sed \
  -e "s|@@IMAGE@@|${collector_image}|g" \
  -e "s|@@CONFIG@@|${config_root}/collector.yaml|g" \
  -e "s|@@WAL@@|${state_root}/collector-wal|g" \
  -e "s|@@ENV@@|${config_root}/collector.env|g" \
  "${observability_dir}/config/tabcomplete-otel-collector.service.in" >"${unit_tmp}"
install -m 0644 "${unit_tmp}" "${unit_root}/tabcomplete-otel-collector.service"
systemctl --user daemon-reload
systemctl --user enable --now tabcomplete-otel-collector.service
