#!/usr/bin/env bash
#SBATCH --job-name=layout_v2
# Billing account: same project as the working churro GPU jobs. Override: sbatch --account=...
#SBATCH --account=project_2017385
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=120G
#SBATCH --chdir=/scratch/project_2017385/dorian/layout_v2
#SBATCH -o logs/layout_v2_%j.out
#SBATCH -e logs/layout_v2_%j.err
#
# layout_v2 single-node launcher: starts `vllm serve` for the selected model, waits for
# readiness, runs runner.py (three-pass layout annotation), cleans up. Self-contained.
#
# The TWO production models (project decision #15) are selected by MODEL_KEY and resolved from
# config.py (single source of truth — this script hardcodes no model ids):
#   MODEL_KEY=thinking   -> Qwen/Qwen3-VL-30B-A3B-Thinking   (default)
#   MODEL_KEY=qwen35     -> Qwen/Qwen3.5-35B-A3B             (startup toggles applied automatically)
#
# Usage (override anything via env):
#   sbatch run_layout_v2.sh                                        # full dev, thinking
#   MODEL_KEY=qwen35 RUN_LABEL=qwen35 sbatch run_layout_v2.sh      # full dev, Qwen3.5
#   MAX_SAMPLES_PER_SPLIT=10 RUN_LABEL=smoke sbatch run_layout_v2.sh
#   ONLY_BASENAMES="$(python3 fixtures.py --list)" RUN_LABEL=fixtures sbatch run_layout_v2.sh
#   DATASET_SPLIT=test RUN_LABEL=test_confirm sbatch run_layout_v2.sh   # the one test-split run
set -euo pipefail

# ---- Environment (CSC Mahti modules; identical stack to the proven old launcher) -------
module --force purge
module use /appl/local/csc/modulefiles
module load python-pytorch
module load python-vllm
# Keep the module's HF/transformers stack ahead of user-site packages (user site carries a
# huggingface-hub too new for the module's transformers).
export PYTHONPATH="/usr/local/lib/python3.12/site-packages:/usr/local/lib64/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/scratch/project_2017385/churro/hfcache/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# ---- Resolve the project dir (sbatch spools this script elsewhere; find runner.py) ------
DEFAULT_PROJECT_DIR="/scratch/project_2017385/dorian/layout_v2"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
if [[ -n "${SOURCE_DIR}" && -f "${SOURCE_DIR}/runner.py" ]]; then
  SCRIPT_DIR="${SOURCE_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/runner.py" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -f "${PWD}/runner.py" ]]; then
  SCRIPT_DIR="${PWD}"
else
  SCRIPT_DIR="${DEFAULT_PROJECT_DIR}"
fi
cd "${SCRIPT_DIR}"
[[ -f "runner.py" ]] || { echo "[error] runner.py not found in ${SCRIPT_DIR}" >&2; exit 1; }
mkdir -p "${SCRIPT_DIR}/logs" "${HF_HOME}"

# ---- Model + policy resolution from config.py (no duplicated constants in bash) ---------
MODEL_KEY="${MODEL_KEY:-thinking}"
MODEL_REPO="$(python3 -c "import config; print(config.MODELS['${MODEL_KEY}']['repo'])")"
REASONING_PARSER="$(python3 -c "import config; print(config.MODELS['${MODEL_KEY}']['reasoning_parser'])")"
# Model-specific vLLM startup toggles (the 35B family hangs after weight load without them).
mapfile -t SERVE_EXTRA < <(python3 -c "import config; [print(f) for f in config.MODELS['${MODEL_KEY}']['serve_extra']]")
MAX_PIXELS="$(python3 -c "import config; print(config.VLLM_MM_MAX_PIXELS)")"
LIMIT_MM="$(python3 -c "import config; print(config.VLLM_LIMIT_MM_PER_PROMPT)")"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(python3 -c "import config; print(config.VLLM_MAX_MODEL_LEN)")}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-layoutv2}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# ---- Inference knobs (defaults live in config.py / runner.py; env only overrides) --------
DATASET_SPLIT="${DATASET_SPLIT:-dev}"
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-0}"
ONLY_BASENAMES="${ONLY_BASENAMES:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
RUN_LABEL="${RUN_LABEL:-${MODEL_KEY}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/layout_v2_${RUN_LABEL}_run}"
SKIP_PASS2="${SKIP_PASS2:-0}"
# Bake-off arms (plan §11.2); defaults = production configuration.
PASS1_ANCHOR_MODE="${PASS1_ANCHOR_MODE:-dual}"      # dual | x_only | text_only
PASS1_INPUT="${PASS1_INPUT:-full_bands}"            # full_bands | full_only | bands_only

LOCAL_VLLM_PORT="${LOCAL_VLLM_PORT:-8000}"
VLLM_BASE_URL="http://localhost:${LOCAL_VLLM_PORT}/v1"
WAIT_SECONDS="${WAIT_SECONDS:-3600}"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-1200}"

command -v vllm >/dev/null 2>&1 || { echo "[error] vllm not in PATH (module missing?)" >&2; exit 1; }

RUN_ID="${SLURM_JOB_ID:-$(date +%s)}"
VLLM_LOG="${SCRIPT_DIR}/logs/vllm_${RUN_ID}.log"

# ---- Start vLLM --------------------------------------------------------------------------
# The two layout_v2-specific serving requirements (plan §10), both from config.py:
#   --limit-mm-per-prompt : pass 1 sends full page + band crops in ONE request;
#   --mm-processor-kwargs : the resolution policy (max_pixels) must be explicit and visible,
#                           never an implicit processor default.
reasoning_flag=()
[[ -n "${REASONING_PARSER}" ]] && reasoning_flag+=(--reasoning-parser "${REASONING_PARSER}")

echo "[run] starting vLLM: model=${MODEL_REPO} key=${MODEL_KEY} tp=${TENSOR_PARALLEL}" \
     "max_pixels=${MAX_PIXELS} limit_mm=${LIMIT_MM} extra='${SERVE_EXTRA[*]:-}' (log: ${VLLM_LOG})"
setsid vllm serve "${MODEL_REPO}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --limit-mm-per-prompt "{\"image\":${LIMIT_MM}}" \
  --mm-processor-kwargs "{\"max_pixels\":${MAX_PIXELS}}" \
  "${reasoning_flag[@]}" \
  ${SERVE_EXTRA[@]:+"${SERVE_EXTRA[@]}"} \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
CLEANUP_DONE=0

sleep 1
if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  echo "[error] vLLM exited immediately. Tail of ${VLLM_LOG}:" >&2
  tail -n 80 "${VLLM_LOG}" >&2 || true
  exit 1
fi

cleanup() {
  # Kill the whole vLLM process group exactly once, on any exit path.
  [[ "${CLEANUP_DONE}" -eq 1 ]] && return
  CLEANUP_DONE=1
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[run] stopping vLLM (pid=${VLLM_PID})"
    kill -TERM -- "-${VLLM_PID}" 2>/dev/null || kill "${VLLM_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${VLLM_PID}" 2>/dev/null || kill -9 "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---- Wait for readiness -------------------------------------------------------------------
echo "[run] waiting for vLLM readiness at ${VLLM_BASE_URL}/models"
python3 - "${VLLM_BASE_URL}/models" "${SERVED_MODEL_NAME}" "${WAIT_SECONDS}" "3" "${VLLM_PID}" "${VLLM_LOG}" <<'PY'
import json, os, sys, time, urllib.request
url, expected, timeout_s, sleep_s, pid, log_path = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
)
start = time.time()
while True:
    try:
        os.kill(pid, 0)  # is the server process still alive?
    except OSError:
        print(f"[error] vLLM process {pid} exited before readiness. See {log_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict)]
        if 200 <= resp.status < 300 and (not expected or expected in ids):
            print(f"[run] vLLM up (HTTP {resp.status}); models={ids}")
            break
    except Exception:
        pass  # not up yet; keep polling until the timeout
    if time.time() - start > timeout_s:
        print(f"[error] timed out after {timeout_s}s waiting for {url}. See {log_path}", file=sys.stderr)
        sys.exit(1)
    time.sleep(sleep_s)
PY

# ---- Run the three-pass annotation ----------------------------------------------------------
skip_pass2_flag=()
[[ "${SKIP_PASS2}" == "1" ]] && skip_pass2_flag+=(--skip-pass2)
only_flag=()
[[ -n "${ONLY_BASENAMES}" ]] && only_flag+=(--only-basenames "${ONLY_BASENAMES}")

echo "[run] inference: split=${DATASET_SPLIT} max_samples=${MAX_SAMPLES_PER_SPLIT} label=${RUN_LABEL} out=${OUTPUT_DIR}"
python3 runner.py \
  --dataset-split "${DATASET_SPLIT}" \
  --max-samples-per-split "${MAX_SAMPLES_PER_SPLIT}" \
  --vllm-base-url "${VLLM_BASE_URL}" \
  --vllm-model "${SERVED_MODEL_NAME}" \
  --model-repo "${MODEL_REPO}" \
  --vllm-api-key "${OPENAI_API_KEY}" \
  --vllm-timeout-seconds "${VLLM_TIMEOUT_SECONDS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --run-label "${RUN_LABEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --pass1-anchor-mode "${PASS1_ANCHOR_MODE}" \
  --pass1-input "${PASS1_INPUT}" \
  "${only_flag[@]}" \
  "${skip_pass2_flag[@]}"

cleanup
echo "[run] layout_v2 inference complete."
