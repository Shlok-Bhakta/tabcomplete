#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
observability_dir=$(cd -- "${script_dir}/.." && pwd)
foundry_version=0.2.17
install_dir="${observability_dir}/.tools/foundry-${foundry_version}"
archive="${install_dir}/foundry_linux_amd64.tar.gz"
checksums="${install_dir}/foundry_${foundry_version}_checksums.txt"
binary="${install_dir}/foundry_linux_amd64/bin/foundryctl"

mkdir -p "${install_dir}"
if [[ ! -x "${binary}" ]]; then
  curl --fail --silent --show-error --location \
    "https://github.com/SigNoz/foundry/releases/download/v${foundry_version}/foundry_linux_amd64.tar.gz" \
    --output "${archive}"
  curl --fail --silent --show-error --location \
    "https://github.com/SigNoz/foundry/releases/download/v${foundry_version}/foundry_${foundry_version}_checksums.txt" \
    --output "${checksums}"
  (cd "${install_dir}" && sha256sum --check "$(basename "${checksums}")" --ignore-missing)
  tar -xzf "${archive}" -C "${install_dir}"
fi

cd "${observability_dir}"
"${binary}" gauge -f casting.yaml
"${binary}" forge -f casting.yaml
