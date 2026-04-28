#!/usr/bin/env bash

set -uo pipefail

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

TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/baseline/baseline_${TIMESTAMP}}"
CHRONOS2_CONFIG="${REPO_ROOT}/baseline/timeseries/chronos2/chronos2.yaml"
TABULAR_CONFIG="${REPO_ROOT}/baseline/timeseries/tabdpt_tabpfn/tabular.yaml"
TABULAR_NO_TIME_CONFIG="${REPO_ROOT}/baseline/timeseries/tabdpt_tabpfn/tabular_no_time_features.yaml"

mkdir -p "${RUN_DIR}"

if [[ "${BASELINE_RUNNER_NOHUP:-0}" != "1" ]]; then
    LAUNCHER_LOG="${RUN_DIR}/launcher.log"
    echo "Launching detached baseline run with nohup."
    echo "Run directory: ${RUN_DIR}"
    echo "Launcher log: ${LAUNCHER_LOG}"
    nohup env \
        BASELINE_RUNNER_NOHUP=1 \
        TIMESTAMP="${TIMESTAMP}" \
        RUN_DIR="${RUN_DIR}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        bash "${BASH_SOURCE[0]}" "$@" > "${LAUNCHER_LOG}" 2>&1 &
    echo "Background PID: $!"
    exit 0
fi

cd "${REPO_ROOT}"

declare -a SUCCEEDED_JOBS=()
declare -a FAILED_JOBS=()

run_job() {
    local name="$1"
    shift

    local log_path="${RUN_DIR}/${name}.log"
    local exit_code=0

    echo "[$(date '+%F %T')] Starting ${name}" | tee -a "${RUN_DIR}/run.log"
    echo "Command: $*" | tee -a "${RUN_DIR}/run.log"
    echo "Log: ${log_path}" | tee -a "${RUN_DIR}/run.log"

    nohup "$@" > "${log_path}" 2>&1 &
    local job_pid=$!
    echo "PID: ${job_pid}" | tee -a "${RUN_DIR}/run.log"

    if wait "${job_pid}"; then
        echo "[$(date '+%F %T')] Finished ${name} successfully" | tee -a "${RUN_DIR}/run.log"
        SUCCEEDED_JOBS+=("${name}")
        return 0
    fi

    exit_code=$?
    echo "[$(date '+%F %T')] ${name} failed with exit code ${exit_code}. Continuing to next job." \
        | tee -a "${RUN_DIR}/run.log"
    FAILED_JOBS+=("${name}:${exit_code}")
    return 0
}

cat > "${RUN_DIR}/README.txt" <<EOF
Run directory: ${RUN_DIR}
Python: ${PYTHON_BIN}
Created at: $(date '+%F %T %Z')
Launcher log: ${RUN_DIR}/launcher.log
Run log: ${RUN_DIR}/run.log

Jobs:
1. chronos2 baseline
2. tabdpt_tabpfn with tabular.yaml
3. tabdpt_tabpfn with tabular_no_time_features.yaml
4. tabdpt_tabpfn_batch with tabular.yaml
5. tabdpt_tabpfn_batch with tabular_no_time_features.yaml
EOF

run_job \
    chronos2 \
    "${PYTHON_BIN}" baseline/timeseries/chronos2/chronos2.py \
    --config "${CHRONOS2_CONFIG}"

run_job \
    tabdpt_tabpfn_tabular \
    "${PYTHON_BIN}" baseline/timeseries/tabdpt_tabpfn/tabdpt_tabpfn.py \
    --config "${TABULAR_CONFIG}"

run_job \
    tabdpt_tabpfn_tabular_no_time_features \
    "${PYTHON_BIN}" baseline/timeseries/tabdpt_tabpfn/tabdpt_tabpfn.py \
    --config "${TABULAR_NO_TIME_CONFIG}"

run_job \
    tabdpt_tabpfn_batch_tabular \
    "${PYTHON_BIN}" baseline/timeseries/tabdpt_tabpfn/tabdpt_tabpfn_batch.py \
    --config "${TABULAR_CONFIG}"

run_job \
    tabdpt_tabpfn_batch_tabular_no_time_features \
    "${PYTHON_BIN}" baseline/timeseries/tabdpt_tabpfn/tabdpt_tabpfn_batch.py \
    --config "${TABULAR_NO_TIME_CONFIG}"

{
    echo
    echo "Completed baseline runs."
    echo "Logs written to: ${RUN_DIR}"
    echo "Successful jobs: ${#SUCCEEDED_JOBS[@]}"
    for job_name in "${SUCCEEDED_JOBS[@]}"; do
        echo "  OK   ${job_name}"
    done
    echo "Failed jobs: ${#FAILED_JOBS[@]}"
    for failed_entry in "${FAILED_JOBS[@]}"; do
        echo "  FAIL ${failed_entry}"
    done
} | tee -a "${RUN_DIR}/run.log"
