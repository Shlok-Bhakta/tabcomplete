#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
observability_dir=$(cd -- "${script_dir}/.." && pwd)
set -a
source "${observability_dir}/.env"
set +a
python3 "${script_dir}/storage_guard.py" health \
  --write-health \
  --root "${TABCOMPLETE_OBS_DATA_ROOT}" \
  --expected-source "${TABCOMPLETE_OBS_VOLUME_SOURCE}" \
  --expected-uuid "${TABCOMPLETE_OBS_VOLUME_UUID}"
