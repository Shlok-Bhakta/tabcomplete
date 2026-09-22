#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: restore.sh BACKUP_DIRECTORY tabcomplete_restore_UNIQUE_NAME" >&2
  exit 2
fi
backup_dir=$1
restore_database=$2
if [[ ! "${restore_database}" =~ ^tabcomplete_restore_[a-z0-9_]{1,32}$ ]]; then
  echo "restore database must use the isolated tabcomplete_restore_ prefix" >&2
  exit 2
fi
if [[ ! -f "${backup_dir}/signoz-metadata.dump" ]]; then
  echo "backup has no signoz-metadata.dump" >&2
  exit 2
fi
# createdb refuses an existing database. Never overwrite the active metastore.
docker exec tabcomplete-observability-metastore-postgres-0 createdb --username signoz "${restore_database}"
docker exec -i tabcomplete-observability-metastore-postgres-0 \
  pg_restore --username signoz --dbname "${restore_database}" --exit-on-error \
  <"${backup_dir}/signoz-metadata.dump"
echo "Metadata restored into isolated database ${restore_database}; active SigNoz database unchanged"
