#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
observability_dir=$(cd -- "${script_dir}/.." && pwd)
set -a
source "${observability_dir}/.env"
set +a
python3 "${script_dir}/storage_guard.py" check \
  --root "${TABCOMPLETE_OBS_DATA_ROOT}" \
  --expected-source "${TABCOMPLETE_OBS_VOLUME_SOURCE}" \
  --expected-uuid "${TABCOMPLETE_OBS_VOLUME_UUID}"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
destination="${TABCOMPLETE_OBS_DATA_ROOT}/backups/${timestamp}"
umask 077
mkdir -m 0700 "${destination}"
docker exec tabcomplete-observability-metastore-postgres-0 \
  pg_dump --username signoz --format custom signoz > "${destination}/signoz-metadata.dump"
cp "${observability_dir}/casting.yaml" "${observability_dir}/casting.yaml.lock" "${destination}/"
cp -R "${observability_dir}/dashboards" "${observability_dir}/query-templates" "${destination}/"
cp -R "${TABCOMPLETE_OBS_DATA_ROOT}/gateway-state" "${destination}/essential-manifests"
echo "metadata backup created at ${destination}"
