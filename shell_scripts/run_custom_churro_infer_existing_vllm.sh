#!/usr/bin/env bash
#SBATCH --job-name=cus_churro_infer_job
#SBATCH --account=project_2000539
#SBATCH --partition=gpusmall
#SBATCH --time=8:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:2
#SBATCH --chdir=/scratch/project_2017385/Churro_copy
#SBATCH -o logs/custom_infer_%j.out
#SBATCH -e logs/custom_infer_%j.err
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

if [[ -f "${SCRIPT_SOURCE_DIR}/../tests/custom_churro_infer.py" ]]; then
  SCRIPT_DIR="${SCRIPT_SOURCE_DIR}"
else
  SCRIPT_DIR="$(pwd)"
fi
cd "${SCRIPT_DIR}"

# Standalone dataset inference (tests/custom_churro_infer.py) against a local vLLM
# endpoint on localhost:8000. This script mirrors the start/wait/run/cleanup flow
# used in run_finetuned_benchmark_existing_vllm.sh.

# Inference knobs (override via env when needed).
DATASET_ID="${DATASET_ID:-stanford-oval/churro-dataset}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"        # custom script supports: all|dev|test
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-0}"
SYSTEM_MESSAGE="${SYSTEM_MESSAGE:-Transcribe the entiretly of this historical documents to XML format.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-22000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"           # 1 => pass --skip-existing
OUTPUT_DIR="${OUTPUT_DIR:-responses}"
METRICS_OUTPUT_ROOT="${METRICS_OUTPUT_ROOT:-results/custom_churro_infer_run1}"
CUSTOM_INFER_REPEAT_COUNT="${CUSTOM_INFER_REPEAT_COUNT:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-125000}"

usage() {
  cat <<'USAGE'
Usage: run_custom_churro_infer_existing_vllm.sh [options]

Options:
  --metrics-output-root <path>   Override METRICS_OUTPUT_ROOT (default stays unchanged if omitted)
  --output-dir <path>            Override OUTPUT_DIR (default stays unchanged if omitted)
  --max-concurrency <n>          Override MAX_CONCURRENCY (default stays unchanged if omitted)
  --vllm-timeout-seconds <n>     Override VLLM_TIMEOUT_SECONDS (default stays unchanged if omitted)
  --max-new-tokens <n>           Override MAX_NEW_TOKENS (default stays unchanged if omitted)
  --dataset-split <split>        Override DATASET_SPLIT: all|dev|test (default stays unchanged if omitted)
  --max-model-len <n>            Override MAX_MODEL_LEN (default stays unchanged if omitted)
  --repeat-count <n>             Repeat custom_churro_infer.py n times while keeping one vLLM alive
  --gpu-memory-utilization <x>   vLLM GPU memory utilization in (0, 1], default: 0.9
  -h, --help                     Show this help text
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metrics-output-root)
      if [[ $# -lt 2 ]]; then
        echo "[error] --metrics-output-root requires a value" >&2
        exit 1
      fi
      METRICS_OUTPUT_ROOT="$2"
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
    --max-concurrency)
      if [[ $# -lt 2 ]]; then
        echo "[error] --max-concurrency requires a value" >&2
        exit 1
      fi
      MAX_CONCURRENCY="$2"
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
    --max-new-tokens)
      if [[ $# -lt 2 ]]; then
        echo "[error] --max-new-tokens requires a value" >&2
        exit 1
      fi
      MAX_NEW_TOKENS="$2"
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
    --max-model-len)
      if [[ $# -lt 2 ]]; then
        echo "[error] --max-model-len requires a value" >&2
        exit 1
      fi
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --repeat-count)
      if [[ $# -lt 2 ]]; then
        echo "[error] --repeat-count requires a value" >&2
        exit 1
      fi
      CUSTOM_INFER_REPEAT_COUNT="$2"
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

if [[ -z "${METRICS_OUTPUT_ROOT}" ]]; then
  echo "[error] METRICS_OUTPUT_ROOT must not be empty" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "[error] OUTPUT_DIR must not be empty" >&2
  exit 1
fi

if ! [[ "${CUSTOM_INFER_REPEAT_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --repeat-count must be a positive integer (got: ${CUSTOM_INFER_REPEAT_COUNT})" >&2
  exit 1
fi

if ! [[ "${MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-concurrency must be a positive integer (got: ${MAX_CONCURRENCY})" >&2
  exit 1
fi

if ! [[ "${VLLM_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --vllm-timeout-seconds must be a positive integer (got: ${VLLM_TIMEOUT_SECONDS})" >&2
  exit 1
fi

if ! [[ "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-new-tokens must be a positive integer (got: ${MAX_NEW_TOKENS})" >&2
  exit 1
fi

case "${DATASET_SPLIT}" in
  all|dev|test) ;;
  *)
    echo "[error] --dataset-split must be one of: all, dev, test (got: ${DATASET_SPLIT})" >&2
    exit 1
    ;;
esac

if ! [[ "${MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-model-len must be a positive integer (got: ${MAX_MODEL_LEN})" >&2
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

# Fixed local vLLM endpoint contract.
LOCAL_VLLM_PORT=8000
LOCAL_VLLM_MODEL_NAME="${LOCAL_VLLM_MODEL_NAME:-churro}"
VLLM_BASE_URL="http://localhost:${LOCAL_VLLM_PORT}/v1"
VLLM_MODELS_URL="${VLLM_BASE_URL}/models"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-3600}"

# Model/server startup controls for vllm serve.
MODEL_REPO="${MODEL_REPO:-stanford-oval/churro-3B}"
WAIT_SECONDS="${WAIT_SECONDS:-1200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"

# Runtime exports used in existing local-vLLM paths.
export HF_HOME=/scratch/project_2017385/churro/hfcache/
export USE_EXISTING_VLLM=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LOCAL_VLLM_PORT
export LOCAL_VLLM_MODEL_NAME

# Ensure expected cache/output paths exist before any log redirection.
mkdir -p "${HF_HOME}" "${SCRIPT_DIR}/logs"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] Python interpreter not found: python3" >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "[error] vllm command not found in PATH." >&2
  exit 1
fi

RUN_ID="${SLURM_JOB_ID:-$(date +%s)}"
VLLM_LOG="${SCRIPT_DIR}/logs/vllm_custom_infer_${RUN_ID}.log"

echo "[run] Starting vLLM (log: ${VLLM_LOG})"
setsid vllm serve "${MODEL_REPO}" \
  --served-model-name "${LOCAL_VLLM_MODEL_NAME}" \
  --max_model_len="${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor_parallel_size 2 \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
CLEANUP_DONE=0

# Fail fast when the server process dies during startup instead of waiting for
# the full readiness timeout.
sleep 1
if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  echo "[error] vLLM exited immediately. See ${VLLM_LOG}" >&2
  if [[ -f "${VLLM_LOG}" ]]; then
    tail -n 80 "${VLLM_LOG}" >&2 || true
  fi
  exit 1
fi

cleanup() {
  if [[ "${CLEANUP_DONE}" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1

  if kill -0 "${VLLM_PID}" 2>/dev/null; then
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

echo "[run] Waiting for vLLM/model readiness at ${VLLM_MODELS_URL}"
python3 - "${VLLM_MODELS_URL}" "${LOCAL_VLLM_MODEL_NAME}" "${WAIT_SECONDS}" "${SLEEP_SECONDS}" "${VLLM_PID}" "${VLLM_LOG}" <<'PY'
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
        print(
            f"[error] Timed out waiting for vLLM/model readiness after {timeout_s}s at {url}. See {log_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    time.sleep(sleep_s)
PY

echo "[run] Ready to run custom inference ${CUSTOM_INFER_REPEAT_COUNT} time(s): split=${DATASET_SPLIT} max_samples_per_split=${MAX_SAMPLES_PER_SPLIT} max_concurrency=${MAX_CONCURRENCY}"

extra_skip_flag=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  extra_skip_flag+=(--skip-existing)
fi

metrics_output_root_base="${METRICS_OUTPUT_ROOT%/}"
if [[ -z "${metrics_output_root_base}" ]]; then
  metrics_output_root_base="${METRICS_OUTPUT_ROOT}"
fi

output_dir_base="${OUTPUT_DIR%/}"
if [[ -z "${output_dir_base}" ]]; then
  output_dir_base="${OUTPUT_DIR}"
fi

set_run_prefix_and_start_index "${metrics_output_root_base}" metrics_output_root_prefix metrics_start_index
metrics_start_index="$(next_available_run_index "${metrics_output_root_prefix}" "${metrics_start_index}")"

set_run_prefix_and_start_index "${output_dir_base}" output_dir_prefix output_start_index
output_start_index="$(next_available_run_index "${output_dir_prefix}" "${output_start_index}")"

echo "[run] Startup paths:"
echo "[run]   cwd=$(pwd)"
echo "[run]   script_dir=${SCRIPT_DIR}"
echo "[run]   script_source=${SCRIPT_SOURCE}"
echo "[run]   script_source_abs=${SCRIPT_SOURCE_ABS}"
echo "[run]   gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "[run]   output_dir_base=${output_dir_base} (abs=$(to_abs_path "${output_dir_base}"))"
echo "[run]   output_dir_prefix=${output_dir_prefix} next_index=${output_start_index}"
echo "[run]   metrics_output_root_base=${metrics_output_root_base} (abs=$(to_abs_path "${metrics_output_root_base}"))"
echo "[run]   metrics_output_root_prefix=${metrics_output_root_prefix} next_index=${metrics_start_index}"

infer_exit_code=0
for ((run_index=1; run_index<=CUSTOM_INFER_REPEAT_COUNT; run_index++)); do
  run_number=$((metrics_start_index + run_index - 1))
  run_metrics_output_root="${metrics_output_root_prefix}${run_number}"
  output_run_number=$((output_start_index + run_index - 1))
  run_output_dir="${output_dir_prefix}${output_run_number}"

  mkdir -p "${run_metrics_output_root}" "${run_output_dir}"
  echo "[run] Starting custom inference run ${run_index}/${CUSTOM_INFER_REPEAT_COUNT}: output_dir=${run_output_dir} metrics_output_root=${run_metrics_output_root}"
  echo "[run]   resolved output_dir abs=$(to_abs_path "${run_output_dir}")"
  echo "[run]   resolved metrics_output_root abs=$(to_abs_path "${run_metrics_output_root}")"

  python3 /scratch/project_2017385/dorian/HTR-context-OCR/tests/custom_churro_infer.py \
    --backend vllm \
    --dataset-id "${DATASET_ID}" \
    --dataset-split "${DATASET_SPLIT}" \
    --max-samples-per-split "${MAX_SAMPLES_PER_SPLIT}" \
    --model-id "${MODEL_REPO}" \
    --system-message "${SYSTEM_MESSAGE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --vllm-base-url "${VLLM_BASE_URL}" \
    --vllm-model "${LOCAL_VLLM_MODEL_NAME}" \
    --vllm-api-key "${OPENAI_API_KEY}" \
    --vllm-timeout-seconds "${VLLM_TIMEOUT_SECONDS}" \
    --output-dir "${run_output_dir}" \
    --metrics-output-root "${run_metrics_output_root}" \
    "${extra_skip_flag[@]}" || infer_exit_code=$?

  if [[ "${infer_exit_code}" -ne 0 ]]; then
    echo "[error] custom_churro_infer.py failed during run ${run_index}/${CUSTOM_INFER_REPEAT_COUNT} with exit code ${infer_exit_code}" >&2
    break
  fi
done

cleanup

if [[ "${infer_exit_code}" -ne 0 ]]; then
  echo "[error] custom_churro_infer.py failed with exit code ${infer_exit_code}" >&2
  exit "${infer_exit_code}"
fi

echo "[run] Custom inference run is over."
