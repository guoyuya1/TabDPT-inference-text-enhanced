#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage: ./run_bitcoin_synthetic_experiments.sh [--config PATH]...

Options:
  --config PATH   Run only the specified config. May be provided multiple times.
  -h, --help      Show this help message.

If no --config is provided, the script runs both default synthetic configs.
Logs are written under:
  $LOG_ROOT/<config_name>_<timestamp>/
EOF
}

mkdir -p "$LOG_ROOT"

run_experiment() {
  local config_path="$1"
  local config_name="$2"
  local config_log_dir="$3"
  local experiment_name="$4"
  local script_path="$5"
  local log_path="$config_log_dir/${config_name}_${experiment_name}.txt"

  echo "============================================================"
  echo "Running ${config_name} / ${experiment_name}"
  echo "Script: ${script_path}"
  echo "Config: ${config_path}"
  echo "Log:    ${log_path}"
  echo

  "$PYTHON_BIN" "$script_path" --config "$config_path" >"$log_path" 2>&1
  echo "Finished ${config_name} / ${experiment_name}"
}

DEFAULT_CONFIG_PATHS=(
  "$REPO_ROOT/fine_tuning/configs/bitcoin_synthetic_no_text_signal.yaml"
  "$REPO_ROOT/fine_tuning/configs/bitcoin_synthetic_text_signal.yaml"
)

CONFIG_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        usage >&2
        exit 1
      fi
      config_arg="$2"
      shift 2
      if [[ "$config_arg" != /* ]]; then
        config_arg="$REPO_ROOT/$config_arg"
      fi
      if [[ ! -f "$config_arg" ]]; then
        echo "Config not found: $config_arg" >&2
        exit 1
      fi
      CONFIG_PATHS+=("$config_arg")
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#CONFIG_PATHS[@]} -eq 0 ]]; then
  CONFIG_PATHS=("${DEFAULT_CONFIG_PATHS[@]}")
fi

EXPERIMENT_NAMES=(
  "qk"
  "v"
  "qkv"
)

SCRIPT_PATHS=(
  "$REPO_ROOT/fine_tuning/fine_tune_dpt_last_qk.py"
  "$REPO_ROOT/fine_tuning/fine_tune_dpt_last_v.py"
  "$REPO_ROOT/fine_tuning/fine_tune_dpt_last_qkv.py"
)

for config_path in "${CONFIG_PATHS[@]}"; do
  config_name="$(basename "$config_path" .yaml)"
  config_log_dir="$LOG_ROOT/${config_name}_${RUN_TIMESTAMP}"
  mkdir -p "$config_log_dir"
  for idx in "${!EXPERIMENT_NAMES[@]}"; do
    run_experiment \
      "$config_path" \
      "$config_name" \
      "$config_log_dir" \
      "${EXPERIMENT_NAMES[$idx]}" \
      "${SCRIPT_PATHS[$idx]}"
  done
done

echo "All experiments completed. Logs saved under: $LOG_ROOT"
