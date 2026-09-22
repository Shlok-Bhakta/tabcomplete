#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: install-importer.sh SSH_HOST DEPLOYMENT_DIRECTORY' >&2; exit 2; }
remote_host=$1
remote_directory=$2
[[ "${remote_host}" =~ ^[a-zA-Z0-9._-]+$ && "${remote_directory}" =~ ^/[a-zA-Z0-9/._-]+$ ]] || exit 2
obs_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repository=$(cd -- "${obs_dir}/../.." && pwd)
[[ "$(uv --version)" == 'uv 0.12.3 '* ]] || { echo 'uv 0.12.3 is required for this pinned installer' >&2; exit 2; }
ssh "${remote_host}" "mkdir -p ${remote_directory}/.tools ${remote_directory}/python-runtime/tinycomplete"
scp -q "$(command -v uv)" "${remote_host}:${remote_directory}/.tools/uv"
rsync -az --exclude __pycache__ "${repository}/src/tinycomplete/observability/" "${remote_host}:${remote_directory}/python-runtime/tinycomplete/observability/"
rsync -az "${obs_dir}/config/importer-requirements.txt" "${remote_host}:${remote_directory}/config/"
ssh "${remote_host}" "test -x ${remote_directory}/.venv/bin/python || ${remote_directory}/.tools/uv venv --python 3.11 ${remote_directory}/.venv"
ssh "${remote_host}" "${remote_directory}/.tools/uv pip install --python ${remote_directory}/.venv/bin/python -r ${remote_directory}/config/importer-requirements.txt"
