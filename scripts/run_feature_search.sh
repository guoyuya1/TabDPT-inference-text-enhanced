#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="/home/yuyan/tabdpt_mz"
PYTHON_BIN="/home/yuyan/tabdpt_mz/.venv/bin/python"
SCRIPT_PATH="/home/yuyan/tabdpt_mz/baseline/timeseries/tabdpt_tabpfn/tabdpt_feature_search_baseline.py"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="/home/yuyan/tabdpt_mz/logs/feature_search_manual_${TIMESTAMP}"
RUN_LOG="${RUN_DIR}/run.log"

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

run_job() {
    local job_name="$1"
    local config_path="$2"
    local log_path="${RUN_DIR}/${job_name}.log"
    local pid_path="${RUN_DIR}/${job_name}.pid"

    echo "[$(date '+%F %T')] START ${job_name}" | tee -a "${RUN_LOG}"
    echo "config=${config_path}" | tee -a "${RUN_LOG}"

    nohup env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${SCRIPT_PATH}" --config "${config_path}" > "${log_path}" 2>&1 &
    local job_pid=$!
    echo "${job_pid}" > "${pid_path}"
    echo "pid=${job_pid}" | tee -a "${RUN_LOG}"

    wait "${job_pid}"
    local exit_code=$?
    echo "[$(date '+%F %T')] END ${job_name} exit_code=${exit_code}" | tee -a "${RUN_LOG}"
    echo | tee -a "${RUN_LOG}"
}

# run_job "agriculture_v3_feature_search" "/home/yuyan/tabdpt_mz/feature_search/config/workshop/agriculture_v3_feature_search.yaml"
# run_job "climate_v3_feature_search" "/home/yuyan/tabdpt_mz/feature_search/config/workshop/climate_v3_feature_search.yaml"
# run_job "economy_v3_feature_search" "/home/yuyan/tabdpt_mz/feature_search/config/workshop/economy_v3_feature_search.yaml"
# run_job "energy_v3_feature_search" "/home/yuyan/tabdpt_mz/feature_search/config/workshop/energy_v3_feature_search.yaml"
# run_job "social_good_v3_feature_search" "/home/yuyan/tabdpt_mz/feature_search/config/workshop/social_good_v3_feature_search.yaml"
run_job "security_v3_feature_search" "/home/yuyan/tabdpt_mz/feature_search/config/workshop/security_v3_feature_search.yaml"
run_job "traffic_v3_feature_search" "/home/yuyan/tabdpt_mz/feature_search/config/workshop/traffic_v3_feature_search.yaml"
