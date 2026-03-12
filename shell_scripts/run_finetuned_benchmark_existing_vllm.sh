#!/usr/bin/env bash
#SBATCH --job-name=benchmark_custom_churro-job
#SBATCH --account=project_2017385
#SBATCH --partition=gpusmall
#SBATCH --time=04:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=85G
#SBATCH --gres=gpu:a100:2
#SBATCH --chdir=/scratch/project_2017385/analyse_churro/Churro
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err
set -euo pipefail

# Ensure module imports and relative paths work even when the script is launched
# from a different current working directory (non-SLURM/manual runs).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"


# Run Churro finetuned benchmark against a local vLLM endpoint on localhost:8000.
# This script intentionally does NOT use Docker/pixi/venv orchestration.
#
# High-level flow:
# 1) Export runtime variables expected by the Churro code path.
# 2) Start `vllm serve` in the background as a dedicated process group.
# 3) Poll /v1/models until the served model is visible.
# 4) Run benchmark via the custom Python module entrypoint.
# 5) Stop the whole vLLM process group even on success/failure/interrupt.

# Benchmark tuning knobs.
# These defaults keep behavior deterministic while still allowing overrides via
# environment variables for ad-hoc runs (e.g. DATASET_SPLIT=dev ...).
ENGINE="${ENGINE:-churro}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
INPUT_SIZE="${INPUT_SIZE:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"

# This script is intentionally fixed to localhost:8000 to match the target flow:
# "single local vLLM instance for finetuned benchmark runs".
LOCAL_VLLM_PORT=8000
LOCAL_VLLM_MODEL_NAME="${LOCAL_VLLM_MODEL_NAME:-$ENGINE}"
VLLM_MODELS_URL="http://localhost:${LOCAL_VLLM_PORT}/v1/models"

# Model and server-start controls for `vllm serve`.
MODEL_REPO="${MODEL_REPO:-stanford-oval/churro-3B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-125000}"
WAIT_SECONDS="${WAIT_SECONDS:-1200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"

# Required runtime exports for this repository's local-vLLM code path:
# - HF_HOME: shared Hugging Face cache location
# - USE_EXISTING_VLLM: tells Churro "don't launch Docker vLLM; use localhost"
# - OPENAI_API_KEY=EMPTY: needed by OpenAI-compatible local endpoints
export HF_HOME=/scratch/project_2017385/churro/hfcache/
export USE_EXISTING_VLLM=1
export OPENAI_API_KEY=EMPTY
export LOCAL_VLLM_PORT
export LOCAL_VLLM_MODEL_NAME

# Ensure all expected output/cache directories exist up-front.
mkdir -p "${HF_HOME}" logs results

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] Python interpreter not found: python3" >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "[error] vllm command not found in PATH." >&2
  exit 1
fi

# Prefer SLURM_JOB_ID in cluster jobs to make logs correlate with scheduler IDs.
RUN_ID="${SLURM_JOB_ID:-$(date +%s)}"
VLLM_LOG="logs/vllm_${RUN_ID}.log"

echo "[run] Starting vLLM (log: ${VLLM_LOG})"
# Start vLLM in a new session/process-group (`setsid`) so cleanup can terminate
# all related child processes reliably using a group kill.
setsid vllm serve "${MODEL_REPO}" \
  --served-model-name "${LOCAL_VLLM_MODEL_NAME}" \
  --max_model_len="${MAX_MODEL_LEN}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
CLEANUP_DONE=0

cleanup() {
  # Guard against double execution when both explicit calls and EXIT trap happen.
  if [[ "${CLEANUP_DONE}" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1

  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[run] Stopping vLLM (pid=${VLLM_PID})"

    # First try graceful stop for the whole vLLM process group.
    # `-- -PID` targets the process group whose id is PID.
    kill -TERM -- "-${VLLM_PID}" 2>/dev/null || kill "${VLLM_PID}" 2>/dev/null || true
    sleep 2

    # If still alive, force-kill the full process group.
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
      kill -KILL -- "-${VLLM_PID}" 2>/dev/null || kill -9 "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[run] vLLM stopped."
  fi
}
trap cleanup EXIT INT TERM

echo "[run] Waiting for vLLM/model readiness at ${VLLM_MODELS_URL}"
# We use a short inline Python probe instead of curl+jq to avoid extra tooling
# dependencies and to keep JSON parsing robust.
python3 - "${VLLM_MODELS_URL}" "${LOCAL_VLLM_MODEL_NAME}" "${WAIT_SECONDS}" "${SLEEP_SECONDS}" <<'PY'
import json
import sys
import time
import urllib.request

url = sys.argv[1]
expected_model = sys.argv[2]
timeout_s = int(sys.argv[3])
sleep_s = float(sys.argv[4])

start = time.time()
while True:
    try:
        # We consider vLLM ready only when:
        # 1) /v1/models returns HTTP 2xx, and
        # 2) expected served model id is visible (unless expectation is empty).
        with urllib.request.urlopen(url, timeout=3) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data", [])
        model_ids = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]

        if 200 <= status < 300 and (not expected_model or expected_model in model_ids):
            print(f"[run] vLLM is up (HTTP {status}). Exposed models: {model_ids}")
            break
    except Exception:
        # Transient startup failures are expected while model weights load.
        pass

    if time.time() - start > timeout_s:
        print(
            f"[error] Timed out waiting for vLLM/model readiness after {timeout_s}s at {url}",
            file=sys.stderr,
        )
        sys.exit(1)
    time.sleep(sleep_s)
PY

echo "[run] Starting benchmark: system=finetuned engine=${ENGINE} split=${DATASET_SPLIT} input_size=${INPUT_SIZE} concurrency=${MAX_CONCURRENCY}"
# Capture benchmark exit code explicitly so we can always stop vLLM before exiting.
benchmark_exit_code=0
PYTHONPATH="/scratch/project_2017385/dorian/HTR-context-OCR/python_scripts${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 -m custom_python_script benchmark \
  --system finetuned \
  --engine "${ENGINE}" \
  --dataset-split "${DATASET_SPLIT}" \
  --input-size "${INPUT_SIZE}" \
  --max-concurrency "${MAX_CONCURRENCY}" || benchmark_exit_code=$?

# Explicit cleanup right after benchmark run, not only via EXIT trap.
cleanup

if [[ "${benchmark_exit_code}" -ne 0 ]]; then
  echo "[error] Benchmark failed with exit code ${benchmark_exit_code}" >&2
  exit "${benchmark_exit_code}"
fi

echo "[run] Benchmark run is over."
