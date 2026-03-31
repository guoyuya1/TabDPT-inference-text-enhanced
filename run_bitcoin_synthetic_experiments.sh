#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/bitcoin_synthetic_signal_runs}"

mkdir -p "$LOG_DIR"

run_experiment() {
  local config_path="$1"
  local config_name="$2"
  local experiment_name="$3"
  local script_path="$4"
  local log_path="$LOG_DIR/${config_name}_${experiment_name}.txt"

  echo "============================================================"
  echo "Running ${config_name} / ${experiment_name}"
  echo "Script: ${script_path}"
  echo "Config: ${config_path}"
  echo "Log:    ${log_path}"
  echo

  "$PYTHON_BIN" "$script_path" --config "$config_path" >"$log_path" 2>&1
  echo "Finished ${config_name} / ${experiment_name}"
}

CONFIG_PATHS=(
  "$REPO_ROOT/fine_tuning/configs/bitcoin_synthetic_no_text_signal.yaml"
  "$REPO_ROOT/fine_tuning/configs/bitcoin_synthetic_text_signal.yaml"
)

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
  for idx in "${!EXPERIMENT_NAMES[@]}"; do
    run_experiment \
      "$config_path" \
      "$config_name" \
      "${EXPERIMENT_NAMES[$idx]}" \
      "${SCRIPT_PATHS[$idx]}"
  done
done

echo "All experiments completed. Logs saved under: $LOG_DIR"
