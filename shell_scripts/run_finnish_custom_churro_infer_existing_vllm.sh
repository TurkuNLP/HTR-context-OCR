#!/usr/bin/env bash
#SBATCH --job-name=fin_cus_churro_infer_job
#SBATCH --account=project_2000539
#SBATCH --partition=gpusmall
#SBATCH --time=5:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy
#SBATCH -o logs/finnish_custom_infer_%j.out
#SBATCH -e logs/finnish_custom_infer_%j.err
set -euo pipefail
module purge
module use /appl/local/csc/modulefiles
module load pytorch

# Resolve the script source path for manual runs. Under sbatch, BASH_SOURCE often
# points at a copied spool script, so fall back to scheduler chdir/PWD unless the
# source directory contains the expected project test entrypoint.
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
if [[ "${SCRIPT_SOURCE}" != /* ]]; then
  SCRIPT_SOURCE="$(pwd)/${SCRIPT_SOURCE}"
fi
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${SCRIPT_SOURCE}")" && pwd)"
SCRIPT_SOURCE_ABS="${SCRIPT_SOURCE_DIR}/$(basename "${SCRIPT_SOURCE}")"

if [[ -f "${SCRIPT_SOURCE_DIR}/../python_scripts/finnish_custom_churro_infer.py" ]]; then
  SCRIPT_DIR="${SCRIPT_SOURCE_DIR}"
else
  SCRIPT_DIR="$(pwd)"
fi
cd "${SCRIPT_DIR}"

# Standalone dataset inference (tests/finnish_custom_churro_infer.py).
# This script mirrors the start/wait/run/cleanup flow used in
# run_custom_churro_infer_existing_vllm.sh, and adds pass-through for all
# finnish_custom_churro_infer.py flags.

# Inference knobs (override via env when needed).
BACKEND="${BACKEND:-vllm}"                  # vllm | transformers
DATASET_ID="${DATASET_ID:-stanford-oval/churro-dataset}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"      # now accepts any split name, plus 'all'
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-0}"
MODEL_ID="${MODEL_ID:-stanford-oval/churro-3B}"
SYSTEM_MESSAGE="${SYSTEM_MESSAGE:-Transcribe the entiretly of this historical documents to XML format.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-22000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
DEVICE="${DEVICE:-auto}"                    # auto | cpu | cuda
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"         # 1 => pass --skip-existing
OUTPUT_DIR="${OUTPUT_DIR:-responses/finnish_custom_infer_run1}"
METRICS_OUTPUT_ROOT="${METRICS_OUTPUT_ROOT:-}"  # empty => defaults to run output dir in Python script
CUSTOM_INFER_REPEAT_COUNT="${CUSTOM_INFER_REPEAT_COUNT:-1}"

# vLLM request settings (passed to python when backend=vllm).
VLLM_BASE_URL="${VLLM_BASE_URL:-}"
VLLM_MODEL="${VLLM_MODEL:-churro}"
VLLM_API_KEY="${VLLM_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-3600}"

# Local vLLM server controls (used when BACKEND=vllm and START_LOCAL_VLLM=1).
START_LOCAL_VLLM="${START_LOCAL_VLLM:-1}"   # 1 => launch local vLLM; 0 => use existing endpoint
LOCAL_VLLM_PORT="${LOCAL_VLLM_PORT:-8000}"
LOCAL_VLLM_MODEL_NAME="${LOCAL_VLLM_MODEL_NAME:-${VLLM_MODEL}}"
MODEL_REPO="${MODEL_REPO:-${MODEL_ID}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-125000}"
WAIT_SECONDS="${WAIT_SECONDS:-1200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"

usage() {
  cat <<'USAGE'
Usage: run_finnish_custom_churro_infer_existing_vllm.sh [options]

Pass-through options for tests/finnish_custom_churro_infer.py:
  --backend <backend>             transformers|vllm
  --dataset-id <id>               Dataset id to load
  --dataset-split <split>         Split name (any split) or 'all'
  --max-samples-per-split <n>     Cap processed Finnish docs per split (0 = all)
  --model-id <id>                 Model id for transformers backend
  --system-message <text>         System prompt
  --max-new-tokens <n>            Max new tokens
  --temperature <float>           Sampling temperature
  --device <device>               auto|cpu|cuda
  --max-concurrency <n>           Max concurrent requests (vllm backend)
  --vllm-base-url <url>           OpenAI-compatible vLLM base URL
  --vllm-model <name>             Served vLLM model name
  --vllm-api-key <key>            API key passed to vLLM endpoint
  --vllm-timeout-seconds <n>      HTTP timeout seconds
  --output-dir <path>             Output directory (run-indexed when repeat-count > 1)
  --metrics-output-root <path>    Metrics root directory (run-indexed when repeat-count > 1)
  --skip-existing                 Pass --skip-existing to Python script

Wrapper/runtime options:
  --repeat-count <n>              Repeat inference n times while keeping one vLLM alive
  --start-local-vllm <0|1>        1 launch local vLLM, 0 use --vllm-base-url as-is
  --local-vllm-port <port>        Local vLLM port when --start-local-vllm=1
  --local-vllm-model-name <name>  Served local vLLM model alias
  --model-repo <repo>             Model repo passed to vllm serve
  --max-model-len <n>             vLLM --max_model_len
  --gpu-memory-utilization <x>    vLLM GPU memory utilization in (0, 1]
  --wait-seconds <n>              Readiness timeout for vLLM/models endpoint
  --sleep-seconds <n>             Poll interval while waiting for readiness
  -h, --help                      Show this help text
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      if [[ $# -lt 2 ]]; then
        echo "[error] --backend requires a value" >&2
        exit 1
      fi
      BACKEND="$2"
      shift 2
      ;;
    --dataset-id)
      if [[ $# -lt 2 ]]; then
        echo "[error] --dataset-id requires a value" >&2
        exit 1
      fi
      DATASET_ID="$2"
      shift 2
      ;;
    --dataset-split)
      if [[ $# -lt 2 ]]; then
        echo "[error] --dataset-split requires a value" >&2
        exit 1
      fi
      DATASET_SPLIT="$2"
      shift 2
      ;;
    --max-samples-per-split)
      if [[ $# -lt 2 ]]; then
        echo "[error] --max-samples-per-split requires a value" >&2
        exit 1
      fi
      MAX_SAMPLES_PER_SPLIT="$2"
      shift 2
      ;;
    --model-id)
      if [[ $# -lt 2 ]]; then
        echo "[error] --model-id requires a value" >&2
        exit 1
      fi
      MODEL_ID="$2"
      shift 2
      ;;
    --system-message)
      if [[ $# -lt 2 ]]; then
        echo "[error] --system-message requires a value" >&2
        exit 1
      fi
      SYSTEM_MESSAGE="$2"
      shift 2
      ;;
    --max-new-tokens)
      if [[ $# -lt 2 ]]; then
        echo "[error] --max-new-tokens requires a value" >&2
        exit 1
      fi
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      if [[ $# -lt 2 ]]; then
        echo "[error] --temperature requires a value" >&2
        exit 1
      fi
      TEMPERATURE="$2"
      shift 2
      ;;
    --device)
      if [[ $# -lt 2 ]]; then
        echo "[error] --device requires a value" >&2
        exit 1
      fi
      DEVICE="$2"
      shift 2
      ;;
    --max-concurrency)
      if [[ $# -lt 2 ]]; then
        echo "[error] --max-concurrency requires a value" >&2
        exit 1
      fi
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    --vllm-base-url)
      if [[ $# -lt 2 ]]; then
        echo "[error] --vllm-base-url requires a value" >&2
        exit 1
      fi
      VLLM_BASE_URL="$2"
      shift 2
      ;;
    --vllm-model)
      if [[ $# -lt 2 ]]; then
        echo "[error] --vllm-model requires a value" >&2
        exit 1
      fi
      VLLM_MODEL="$2"
      shift 2
      ;;
    --vllm-api-key)
      if [[ $# -lt 2 ]]; then
        echo "[error] --vllm-api-key requires a value" >&2
        exit 1
      fi
      VLLM_API_KEY="$2"
      shift 2
      ;;
    --vllm-timeout-seconds)
      if [[ $# -lt 2 ]]; then
        echo "[error] --vllm-timeout-seconds requires a value" >&2
        exit 1
      fi
      VLLM_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "[error] --output-dir requires a value" >&2
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --metrics-output-root)
      if [[ $# -lt 2 ]]; then
        echo "[error] --metrics-output-root requires a value" >&2
        exit 1
      fi
      METRICS_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --repeat-count)
      if [[ $# -lt 2 ]]; then
        echo "[error] --repeat-count requires a value" >&2
        exit 1
      fi
      CUSTOM_INFER_REPEAT_COUNT="$2"
      shift 2
      ;;
    --start-local-vllm)
      if [[ $# -lt 2 ]]; then
        echo "[error] --start-local-vllm requires a value" >&2
        exit 1
      fi
      START_LOCAL_VLLM="$2"
      shift 2
      ;;
    --local-vllm-port)
      if [[ $# -lt 2 ]]; then
        echo "[error] --local-vllm-port requires a value" >&2
        exit 1
      fi
      LOCAL_VLLM_PORT="$2"
      shift 2
      ;;
    --local-vllm-model-name)
      if [[ $# -lt 2 ]]; then
        echo "[error] --local-vllm-model-name requires a value" >&2
        exit 1
      fi
      LOCAL_VLLM_MODEL_NAME="$2"
      shift 2
      ;;
    --model-repo)
      if [[ $# -lt 2 ]]; then
        echo "[error] --model-repo requires a value" >&2
        exit 1
      fi
      MODEL_REPO="$2"
      shift 2
      ;;
    --max-model-len)
      if [[ $# -lt 2 ]]; then
        echo "[error] --max-model-len requires a value" >&2
        exit 1
      fi
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      if [[ $# -lt 2 ]]; then
        echo "[error] --gpu-memory-utilization requires a value" >&2
        exit 1
      fi
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --wait-seconds)
      if [[ $# -lt 2 ]]; then
        echo "[error] --wait-seconds requires a value" >&2
        exit 1
      fi
      WAIT_SECONDS="$2"
      shift 2
      ;;
    --sleep-seconds)
      if [[ $# -lt 2 ]]; then
        echo "[error] --sleep-seconds requires a value" >&2
        exit 1
      fi
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "[error] OUTPUT_DIR must not be empty" >&2
  exit 1
fi

if [[ -n "${METRICS_OUTPUT_ROOT}" && -z "${METRICS_OUTPUT_ROOT// }" ]]; then
  echo "[error] --metrics-output-root must not be empty when provided" >&2
  exit 1
fi

if [[ -z "${DATASET_SPLIT// }" ]]; then
  echo "[error] --dataset-split must not be empty" >&2
  exit 1
fi

case "${BACKEND}" in
  transformers|vllm) ;;
  *)
    echo "[error] --backend must be one of: transformers, vllm (got: ${BACKEND})" >&2
    exit 1
    ;;
esac

case "${DEVICE}" in
  auto|cpu|cuda) ;;
  *)
    echo "[error] --device must be one of: auto, cpu, cuda (got: ${DEVICE})" >&2
    exit 1
    ;;
esac

if ! [[ "${CUSTOM_INFER_REPEAT_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --repeat-count must be a positive integer (got: ${CUSTOM_INFER_REPEAT_COUNT})" >&2
  exit 1
fi

if ! [[ "${MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-concurrency must be a positive integer (got: ${MAX_CONCURRENCY})" >&2
  exit 1
fi

if ! [[ "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-new-tokens must be a positive integer (got: ${MAX_NEW_TOKENS})" >&2
  exit 1
fi

if ! [[ "${MAX_SAMPLES_PER_SPLIT}" =~ ^[0-9]+$ ]]; then
  echo "[error] --max-samples-per-split must be a non-negative integer (got: ${MAX_SAMPLES_PER_SPLIT})" >&2
  exit 1
fi

if ! [[ "${VLLM_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --vllm-timeout-seconds must be a positive integer (got: ${VLLM_TIMEOUT_SECONDS})" >&2
  exit 1
fi

if ! [[ "${START_LOCAL_VLLM}" =~ ^[01]$ ]]; then
  echo "[error] --start-local-vllm must be 0 or 1 (got: ${START_LOCAL_VLLM})" >&2
  exit 1
fi

if ! [[ "${MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-model-len must be a positive integer (got: ${MAX_MODEL_LEN})" >&2
  exit 1
fi

if ! [[ "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --wait-seconds must be a positive integer (got: ${WAIT_SECONDS})" >&2
  exit 1
fi

if ! [[ "${SLEEP_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[error] --sleep-seconds must be numeric and > 0 (got: ${SLEEP_SECONDS})" >&2
  exit 1
fi

if ! [[ "${GPU_MEMORY_UTILIZATION}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
  echo "[error] --gpu-memory-utilization must be a decimal in (0, 1] (got: ${GPU_MEMORY_UTILIZATION})" >&2
  exit 1
fi

if [[ "${GPU_MEMORY_UTILIZATION}" =~ ^0(\.0+)?$ ]]; then
  echo "[error] --gpu-memory-utilization must be greater than 0 (got: ${GPU_MEMORY_UTILIZATION})" >&2
  exit 1
fi

# transformers backend does not require vLLM server startup.
if [[ "${BACKEND}" == "transformers" ]]; then
  START_LOCAL_VLLM=0
fi

if [[ "${BACKEND}" == "vllm" ]]; then
  if [[ "${START_LOCAL_VLLM}" == "1" ]]; then
    VLLM_BASE_URL="http://localhost:${LOCAL_VLLM_PORT}/v1"
    VLLM_MODEL="${LOCAL_VLLM_MODEL_NAME}"
  elif [[ -z "${VLLM_BASE_URL}" ]]; then
    echo "[error] --vllm-base-url is required when --start-local-vllm=0" >&2
    exit 1
  fi
else
  # Keep a stable default for pass-through args even when backend=transformers.
  if [[ -z "${VLLM_BASE_URL}" ]]; then
    VLLM_BASE_URL="http://localhost:${LOCAL_VLLM_PORT}/v1"
  fi
fi

VLLM_MODELS_URL="${VLLM_BASE_URL%/}/models"

to_abs_path() {
  local input_path="$1"
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
  else
    printf '%s/%s\n' "${SCRIPT_DIR}" "${input_path}"
  fi
}

set_run_prefix_and_start_index() {
  local base_path="$1"
  local prefix_var_name="$2"
  local start_index_var_name="$3"
  local run_prefix=""
  local start_index=1

  if [[ "${base_path}" =~ ^(.+_run)([0-9]+)$ ]]; then
    run_prefix="${BASH_REMATCH[1]}"
    start_index=$((10#${BASH_REMATCH[2]}))
  else
    run_prefix="${base_path}_run"
  fi

  printf -v "${prefix_var_name}" '%s' "${run_prefix}"
  printf -v "${start_index_var_name}" '%s' "${start_index}"
}

next_available_run_index() {
  local run_prefix="$1"
  local requested_start_index="$2"
  local parent_dir
  local prefix_name
  local candidate_path
  local candidate_name
  local candidate_suffix
  local candidate_index
  local max_existing_index=0

  parent_dir="$(dirname "${run_prefix}")"
  prefix_name="$(basename "${run_prefix}")"

  if [[ -d "${parent_dir}" ]]; then
    shopt -s nullglob
    for candidate_path in "${parent_dir}/${prefix_name}"*; do
      [[ -d "${candidate_path}" ]] || continue
      candidate_name="$(basename "${candidate_path}")"
      candidate_suffix="${candidate_name#${prefix_name}}"
      if [[ "${candidate_suffix}" =~ ^[0-9]+$ ]]; then
        candidate_index=$((10#${candidate_suffix}))
        if (( candidate_index > max_existing_index )); then
          max_existing_index=${candidate_index}
        fi
      fi
    done
    shopt -u nullglob
  fi

  if (( requested_start_index <= max_existing_index )); then
    echo $((max_existing_index + 1))
  else
    echo "${requested_start_index}"
  fi
}

# Runtime exports used in local-vLLM paths.
export HF_HOME=/scratch/project_2017385/churro/hfcache/
export USE_EXISTING_VLLM=1
export OPENAI_API_KEY="${VLLM_API_KEY}"
export LOCAL_VLLM_PORT
export LOCAL_VLLM_MODEL_NAME

# Ensure expected cache/output paths exist before any log redirection.
mkdir -p "${HF_HOME}" "${SCRIPT_DIR}/logs"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] Python interpreter not found: python3" >&2
  exit 1
fi

VLLM_PID=""
VLLM_LOG="<none>"
CLEANUP_DONE=0

cleanup() {
  if [[ "${CLEANUP_DONE}" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1

  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[run] Stopping vLLM (pid=${VLLM_PID})"
    kill -TERM -- "-${VLLM_PID}" 2>/dev/null || kill "${VLLM_PID}" 2>/dev/null || true
    sleep 2
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
      kill -KILL -- "-${VLLM_PID}" 2>/dev/null || kill -9 "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[run] vLLM stopped."
  fi
}
trap cleanup EXIT INT TERM

if [[ "${BACKEND}" == "vllm" && "${START_LOCAL_VLLM}" == "1" ]]; then
  if ! command -v vllm >/dev/null 2>&1; then
    echo "[error] vllm command not found in PATH." >&2
    exit 1
  fi

  RUN_ID="${SLURM_JOB_ID:-$(date +%s)}"
  VLLM_LOG="${SCRIPT_DIR}/logs/vllm_finnish_custom_infer_${RUN_ID}.log"

  echo "[run] Starting local vLLM (log: ${VLLM_LOG})"
  setsid vllm serve "${MODEL_REPO}" \
    --served-model-name "${LOCAL_VLLM_MODEL_NAME}" \
    --max_model_len="${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor_parallel_size 2 \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!

  # Fail fast when the server process dies during startup.
  sleep 1
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[error] local vLLM exited immediately. See ${VLLM_LOG}" >&2
    if [[ -f "${VLLM_LOG}" ]]; then
      tail -n 80 "${VLLM_LOG}" >&2 || true
    fi
    exit 1
  fi
fi

if [[ "${BACKEND}" == "vllm" ]]; then
  echo "[run] Waiting for vLLM/model readiness at ${VLLM_MODELS_URL}"
  python3 - "${VLLM_MODELS_URL}" "${VLLM_MODEL}" "${WAIT_SECONDS}" "${SLEEP_SECONDS}" "${VLLM_PID:-0}" "${VLLM_LOG}" <<'PY'
import json
import os
import sys
import time
import urllib.request

url = sys.argv[1]
expected_model = sys.argv[2]
timeout_s = int(sys.argv[3])
sleep_s = float(sys.argv[4])
pid = int(sys.argv[5])
log_path = sys.argv[6]

start = time.time()
while True:
    if pid > 0:
        try:
            os.kill(pid, 0)
        except OSError:
            print(
                f"[error] vLLM process {pid} exited before readiness completed. See {log_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data", [])
        model_ids = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
        if 200 <= status < 300 and (not expected_model or expected_model in model_ids):
            print(f"[run] vLLM is up (HTTP {status}). Exposed models: {model_ids}")
            break
    except Exception:
        pass

    if time.time() - start > timeout_s:
        suffix = f" See {log_path}" if log_path and log_path != "<none>" else ""
        print(
            f"[error] Timed out waiting for vLLM/model readiness after {timeout_s}s at {url}.{suffix}",
            file=sys.stderr,
        )
        sys.exit(1)
    time.sleep(sleep_s)
PY
fi

echo "[run] Ready to run finnish custom inference ${CUSTOM_INFER_REPEAT_COUNT} time(s): backend=${BACKEND} split=${DATASET_SPLIT} max_samples_per_split=${MAX_SAMPLES_PER_SPLIT} max_concurrency=${MAX_CONCURRENCY}"

extra_skip_flag=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  extra_skip_flag+=(--skip-existing)
fi

output_dir_base="${OUTPUT_DIR%/}"
if [[ -z "${output_dir_base}" ]]; then
  output_dir_base="${OUTPUT_DIR}"
fi

set_run_prefix_and_start_index "${output_dir_base}" output_dir_prefix output_start_index
output_start_index="$(next_available_run_index "${output_dir_prefix}" "${output_start_index}")"

metrics_output_dir_base=""
metrics_output_dir_prefix=""
metrics_output_start_index=1
if [[ -n "${METRICS_OUTPUT_ROOT}" ]]; then
  metrics_output_dir_base="${METRICS_OUTPUT_ROOT%/}"
  if [[ -z "${metrics_output_dir_base}" ]]; then
    metrics_output_dir_base="${METRICS_OUTPUT_ROOT}"
  fi
  set_run_prefix_and_start_index "${metrics_output_dir_base}" metrics_output_dir_prefix metrics_output_start_index
  metrics_output_start_index="$(next_available_run_index "${metrics_output_dir_prefix}" "${metrics_output_start_index}")"
fi

echo "[run] Startup paths:"
echo "[run]   cwd=$(pwd)"
echo "[run]   script_dir=${SCRIPT_DIR}"
echo "[run]   script_source=${SCRIPT_SOURCE}"
echo "[run]   script_source_abs=${SCRIPT_SOURCE_ABS}"
echo "[run]   output_dir_base=${output_dir_base} (abs=$(to_abs_path "${output_dir_base}"))"
echo "[run]   output_dir_prefix=${output_dir_prefix} next_index=${output_start_index}"
if [[ -n "${METRICS_OUTPUT_ROOT}" ]]; then
  echo "[run]   metrics_output_root_base=${metrics_output_dir_base} (abs=$(to_abs_path "${metrics_output_dir_base}"))"
  echo "[run]   metrics_output_root_prefix=${metrics_output_dir_prefix} next_index=${metrics_output_start_index}"
else
  echo "[run]   metrics_output_root_base=<default to each run output dir>"
fi
if [[ "${BACKEND}" == "vllm" ]]; then
  echo "[run]   vllm_base_url=${VLLM_BASE_URL}"
  echo "[run]   vllm_model=${VLLM_MODEL}"
  echo "[run]   start_local_vllm=${START_LOCAL_VLLM}"
  if [[ "${START_LOCAL_VLLM}" == "1" ]]; then
    echo "[run]   model_repo=${MODEL_REPO}"
    echo "[run]   local_vllm_port=${LOCAL_VLLM_PORT}"
    echo "[run]   local_vllm_model_name=${LOCAL_VLLM_MODEL_NAME}"
    echo "[run]   max_model_len=${MAX_MODEL_LEN}"
    echo "[run]   gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  fi
fi

infer_exit_code=0
for ((run_index=1; run_index<=CUSTOM_INFER_REPEAT_COUNT; run_index++)); do
  output_run_number=$((output_start_index + run_index - 1))
  run_output_dir="${output_dir_prefix}${output_run_number}"
  extra_metrics_flag=()

  mkdir -p "${run_output_dir}"
  echo "[run] Starting finnish custom inference run ${run_index}/${CUSTOM_INFER_REPEAT_COUNT}: output_dir=${run_output_dir}"
  echo "[run]   resolved output_dir abs=$(to_abs_path "${run_output_dir}")"
  if [[ -n "${METRICS_OUTPUT_ROOT}" ]]; then
    metrics_output_run_number=$((metrics_output_start_index + run_index - 1))
    run_metrics_output_root="${metrics_output_dir_prefix}${metrics_output_run_number}"
    mkdir -p "${run_metrics_output_root}"
    extra_metrics_flag=(--metrics-output-root "${run_metrics_output_root}")
    echo "[run]   metrics_output_root=${run_metrics_output_root}"
    echo "[run]   resolved metrics_output_root abs=$(to_abs_path "${run_metrics_output_root}")"
  fi

  python3 /scratch/project_2017385/dorian/HTR-context-OCR/python_scripts/finnish_custom_churro_infer.py \
    --backend "${BACKEND}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-split "${DATASET_SPLIT}" \
    --max-samples-per-split "${MAX_SAMPLES_PER_SPLIT}" \
    --model-id "${MODEL_ID}" \
    --system-message "${SYSTEM_MESSAGE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --device "${DEVICE}" \
    --vllm-base-url "${VLLM_BASE_URL}" \
    --vllm-model "${VLLM_MODEL}" \
    --vllm-api-key "${VLLM_API_KEY}" \
    --vllm-timeout-seconds "${VLLM_TIMEOUT_SECONDS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --output-dir "${run_output_dir}" \
    "${extra_metrics_flag[@]}" \
    "${extra_skip_flag[@]}" || infer_exit_code=$?

  if [[ "${infer_exit_code}" -ne 0 ]]; then
    echo "[error] finnish_custom_churro_infer.py failed during run ${run_index}/${CUSTOM_INFER_REPEAT_COUNT} with exit code ${infer_exit_code}" >&2
    break
  fi
done

cleanup

if [[ "${infer_exit_code}" -ne 0 ]]; then
  echo "[error] finnish_custom_churro_infer.py failed with exit code ${infer_exit_code}" >&2
  exit "${infer_exit_code}"
fi

echo "[run] Finnish custom inference run is over."
