#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
    echo "Could not find a Python executable." >&2
    exit 1
fi

SCRIPT_PATH="${REPO_ROOT}/data/scripts/create_embeddings_with_lags.py"

CONFIGS=(
    # "${REPO_ROOT}/data/source_data/agriculture/raw/config.yaml"
    # "${REPO_ROOT}/data/source_data/climate/raw/config.yaml"
    # "${REPO_ROOT}/data/source_data/environment/raw/config.yaml"
    # "${REPO_ROOT}/data/source_data/health_afr/raw/config.yaml"
    # "${REPO_ROOT}/data/source_data/health_us/raw/config.yaml"
    # "${REPO_ROOT}/data/source_data/security/raw/config.yaml"
    "${REPO_ROOT}/data/source_data/social_good/raw/config.yaml"
    "${REPO_ROOT}/data/source_data/traffic/raw/config.yaml"
)

cd "${REPO_ROOT}"

for config_path in "${CONFIGS[@]}"; do
    echo "Running embeddings for config: ${config_path}"
    "${PYTHON_BIN}" "${SCRIPT_PATH}" --config "${config_path}"
done

echo "Finished embedding generation for all new datasets."
