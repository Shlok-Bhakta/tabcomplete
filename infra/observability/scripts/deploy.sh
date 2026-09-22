#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
observability_dir=$(cd -- "${script_dir}/.." && pwd)
environment_file="${observability_dir}/.env"
if [[ ! -f "${environment_file}" ]]; then
  echo "missing ${environment_file}" >&2
  exit 2
fi
set -a
source "${environment_file}"
set +a

python3 "${script_dir}/storage_guard.py" check \
  --root "${TABCOMPLETE_OBS_DATA_ROOT}" \
  --expected-source "${TABCOMPLETE_OBS_VOLUME_SOURCE}" \
  --expected-uuid "${TABCOMPLETE_OBS_VOLUME_UUID}"
"${script_dir}/render.sh"
cd "${observability_dir}/pours/deployment"
docker compose --env-file "${environment_file}" config --quiet
docker compose --env-file "${environment_file}" pull --ignore-buildable
docker compose --env-file "${environment_file}" build gateway
docker compose --env-file "${environment_file}" up -d
