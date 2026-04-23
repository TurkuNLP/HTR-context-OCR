#!/usr/bin/env bash
#SBATCH --job-name=text_metrics_report
#SBATCH --account=project_2000539
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=96G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy
#SBATCH -o logs/text_metrics_report_%j.out
#SBATCH -e logs/text_metrics_report_%j.err

set -euo pipefail

# Optional environment setup for CSC/HPC. Safe to ignore on machines without `module`.
if command -v module >/dev/null 2>&1; then
  module purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

SCRIPT_DIR="/scratch/project_2017385/dorian/Churro_copy"
cd "${SCRIPT_DIR}"

RUNFILE_JSON="${RUNFILE_JSON:-}"
SCORES_PKL="${SCORES_PKL:-}"
PROJECT_ROOT_RESULTS="${PROJECT_ROOT_RESULTS:-/scratch/project_2017385/dorian/Churro_copy/results/text_metrics_results}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-35}"
TARGET_FNAME="${TARGET_FNAME:-}"
MAX_ITEMS="${MAX_ITEMS:-}"
HOUGH_THRESHOLD="${HOUGH_THRESHOLD:-10}"
HOUGH_LINE_LENGTH="${HOUGH_LINE_LENGTH:-8}"
HOUGH_LINE_GAP="${HOUGH_LINE_GAP:-8}"
HOUGH_SEED="${HOUGH_SEED:-0}"
HOUGH_START="${HOUGH_START:-2.2}"
ALIGN_ABS_MIN_LEN="${ALIGN_ABS_MIN_LEN:-6.0}"
ALIGN_MIN_IOU_THRESHOLD="${ALIGN_MIN_IOU_THRESHOLD:-}"
WITH_VISUALS="${WITH_VISUALS:-0}"

usage() {
  cat <<'USAGE'
Usage: run_text_metrics_report.sh [options]

Inputs (at least one required):
  --runfile-json <path>             Path to outputs.json.
  --scores-pkl <path>               Path to scores.pkl pickle-stream.

Outputs:
  --output-dir <dir>                Exact output directory for text_metrics_report.py.
                                    If omitted, wrapper creates timestamped run dir under --project-root-results.
  --project-root-results <dir>      Root output directory used when --output-dir is not provided.
  --root-results <dir>              Alias for --project-root-results.

Text-matrix settings:
  --window-size <n>                 Sliding window size in characters. Default: 50
  --window-stride <n>               Sliding window stride in characters. Default: 35
  --target-fname <name>             Process only one matching image/file name.
  --max-items <n>                   Process only the first N items.

Visuals:
  --with-visuals                    Enable report visualisations (before/after Hough/filter/reorder).

Hough parameters:
  --hough-threshold <n>             Probabilistic Hough vote threshold. Default: 10
  --hough-line-length <n>           Minimum accepted line length. Default: 8
  --hough-line-gap <n>              Maximum allowed gap when linking line pixels. Default: 8
  --hough-seed <n>                  Base RNG seed (index is added per item). Default: 0
  --hough-start <float>             Initial adaptive threshold start value. Default: 2.2

V2.1 IoU filter parameters:
  --align-abs-min-len <float>       Minimum line length before v2.1 IoU filtering. Default: 6.0
  --align-min-iou-threshold <float> Optional override in [0,1]. If omitted, script default is used.

Other:
  -h, --help                        Show this help text.

Environment variable overrides are supported for all options:
  RUNFILE_JSON, SCORES_PKL, OUTPUT_DIR, PROJECT_ROOT_RESULTS, WINDOW_SIZE, WINDOW_STRIDE,
  TARGET_FNAME, MAX_ITEMS, HOUGH_THRESHOLD, HOUGH_LINE_LENGTH,
  HOUGH_LINE_GAP, HOUGH_SEED, HOUGH_START, ALIGN_ABS_MIN_LEN,
  ALIGN_MIN_IOU_THRESHOLD, WITH_VISUALS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scores-pkl)
      [[ $# -ge 2 ]] || { echo "[error] --scores-pkl requires a value" >&2; exit 1; }
      SCORES_PKL="$2"
      shift 2
      ;;
    --runfile-json)
      [[ $# -ge 2 ]] || { echo "[error] --runfile-json requires a value" >&2; exit 1; }
      RUNFILE_JSON="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "[error] --output-dir requires a value" >&2; exit 1; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --project-root-results|--root-results)
      [[ $# -ge 2 ]] || { echo "[error] $1 requires a value" >&2; exit 1; }
      PROJECT_ROOT_RESULTS="$2"
      shift 2
      ;;
    --window-size)
      [[ $# -ge 2 ]] || { echo "[error] --window-size requires a value" >&2; exit 1; }
      WINDOW_SIZE="$2"
      shift 2
      ;;
    --window-stride)
      [[ $# -ge 2 ]] || { echo "[error] --window-stride requires a value" >&2; exit 1; }
      WINDOW_STRIDE="$2"
      shift 2
      ;;
    --target-fname)
      [[ $# -ge 2 ]] || { echo "[error] --target-fname requires a value" >&2; exit 1; }
      TARGET_FNAME="$2"
      shift 2
      ;;
    --max-items)
      [[ $# -ge 2 ]] || { echo "[error] --max-items requires a value" >&2; exit 1; }
      MAX_ITEMS="$2"
      shift 2
      ;;
    --hough-threshold)
      [[ $# -ge 2 ]] || { echo "[error] --hough-threshold requires a value" >&2; exit 1; }
      HOUGH_THRESHOLD="$2"
      shift 2
      ;;
    --hough-line-length)
      [[ $# -ge 2 ]] || { echo "[error] --hough-line-length requires a value" >&2; exit 1; }
      HOUGH_LINE_LENGTH="$2"
      shift 2
      ;;
    --hough-line-gap)
      [[ $# -ge 2 ]] || { echo "[error] --hough-line-gap requires a value" >&2; exit 1; }
      HOUGH_LINE_GAP="$2"
      shift 2
      ;;
    --hough-seed)
      [[ $# -ge 2 ]] || { echo "[error] --hough-seed requires a value" >&2; exit 1; }
      HOUGH_SEED="$2"
      shift 2
      ;;
    --hough-start)
      [[ $# -ge 2 ]] || { echo "[error] --hough-start requires a value" >&2; exit 1; }
      HOUGH_START="$2"
      shift 2
      ;;
    --align-abs-min-len)
      [[ $# -ge 2 ]] || { echo "[error] --align-abs-min-len requires a value" >&2; exit 1; }
      ALIGN_ABS_MIN_LEN="$2"
      shift 2
      ;;
    --align-min-iou-threshold)
      [[ $# -ge 2 ]] || { echo "[error] --align-min-iou-threshold requires a value" >&2; exit 1; }
      ALIGN_MIN_IOU_THRESHOLD="$2"
      shift 2
      ;;
    --with-visuals)
      WITH_VISUALS="1"
      shift
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

if [[ -z "${RUNFILE_JSON}" && -z "${SCORES_PKL}" ]]; then
  echo "[error] Provide at least one input: --runfile-json and/or --scores-pkl" >&2
  exit 1
fi
if [[ -n "${RUNFILE_JSON}" && ! -f "${RUNFILE_JSON}" ]]; then
  echo "[error] RUNFILE_JSON does not exist: ${RUNFILE_JSON}" >&2
  exit 1
fi
if [[ -n "${SCORES_PKL}" && ! -f "${SCORES_PKL}" ]]; then
  echo "[error] SCORES_PKL does not exist: ${SCORES_PKL}" >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR}" && -z "${PROJECT_ROOT_RESULTS}" ]]; then
  echo "[error] PROJECT_ROOT_RESULTS must not be empty when --output-dir is not set" >&2
  exit 1
fi
if ! [[ "${WINDOW_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --window-size must be a positive integer (got: ${WINDOW_SIZE})" >&2
  exit 1
fi
if ! [[ "${WINDOW_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --window-stride must be a positive integer (got: ${WINDOW_STRIDE})" >&2
  exit 1
fi
if [[ -n "${MAX_ITEMS}" ]] && ! [[ "${MAX_ITEMS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-items must be a positive integer (got: ${MAX_ITEMS})" >&2
  exit 1
fi
if ! [[ "${HOUGH_THRESHOLD}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --hough-threshold must be a positive integer (got: ${HOUGH_THRESHOLD})" >&2
  exit 1
fi
if ! [[ "${HOUGH_LINE_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --hough-line-length must be a positive integer (got: ${HOUGH_LINE_LENGTH})" >&2
  exit 1
fi
if ! [[ "${HOUGH_LINE_GAP}" =~ ^[0-9]+$ ]]; then
  echo "[error] --hough-line-gap must be a non-negative integer (got: ${HOUGH_LINE_GAP})" >&2
  exit 1
fi
if ! [[ "${HOUGH_SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "[error] --hough-seed must be an integer (got: ${HOUGH_SEED})" >&2
  exit 1
fi
if ! [[ "${HOUGH_START}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[error] --hough-start must be a positive number (got: ${HOUGH_START})" >&2
  exit 1
fi
if ! [[ "${ALIGN_ABS_MIN_LEN}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[error] --align-abs-min-len must be a positive number (got: ${ALIGN_ABS_MIN_LEN})" >&2
  exit 1
fi
if [[ -n "${ALIGN_MIN_IOU_THRESHOLD}" ]]; then
  if ! [[ "${ALIGN_MIN_IOU_THRESHOLD}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "[error] --align-min-iou-threshold must be a number in [0,1] (got: ${ALIGN_MIN_IOU_THRESHOLD})" >&2
    exit 1
  fi
  if ! awk -v v="${ALIGN_MIN_IOU_THRESHOLD}" 'BEGIN{exit !(v >= 0.0 && v <= 1.0)}'; then
    echo "[error] --align-min-iou-threshold must satisfy 0.0 <= value <= 1.0 (got: ${ALIGN_MIN_IOU_THRESHOLD})" >&2
    exit 1
  fi
fi
if [[ "${WITH_VISUALS}" != "0" && "${WITH_VISUALS}" != "1" ]]; then
  echo "[error] WITH_VISUALS must be 0 or 1 (got: ${WITH_VISUALS})" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 not found in PATH" >&2
  exit 1
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  REPORT_DIR="${OUTPUT_DIR}"
  RUN_DIR="$(dirname "${REPORT_DIR}")"
  RUN_BASE_DIR="${RUN_DIR}"
  RUN_TIMESTAMP="manual_output_dir"
else
  RUN_BASE_DIR="${PROJECT_ROOT_RESULTS}/window_${WINDOW_SIZE}_stride_${WINDOW_STRIDE}"
  RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="${RUN_BASE_DIR}/${RUN_TIMESTAMP}"
  while [[ -e "${RUN_DIR}" ]]; do
    sleep 1
    RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="${RUN_BASE_DIR}/${RUN_TIMESTAMP}"
  done
  REPORT_DIR="${RUN_DIR}/text_metrics_report"
fi

mkdir -p "${REPORT_DIR}" "${SCRIPT_DIR}/logs"
if [[ -n "${RUN_DIR}" ]]; then
  mkdir -p "${RUN_DIR}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export PYTHONUNBUFFERED=1

PY_ARGS=(
  --output-dir "${REPORT_DIR}"
  --window-size "${WINDOW_SIZE}"
  --window-stride "${WINDOW_STRIDE}"
  --hough-threshold "${HOUGH_THRESHOLD}"
  --hough-line-length "${HOUGH_LINE_LENGTH}"
  --hough-line-gap "${HOUGH_LINE_GAP}"
  --hough-seed "${HOUGH_SEED}"
  --hough-start "${HOUGH_START}"
  --align-abs-min-len "${ALIGN_ABS_MIN_LEN}"
)
if [[ -n "${SCORES_PKL}" ]]; then
  PY_ARGS+=(--scores-pkl "${SCORES_PKL}")
fi
if [[ -n "${RUNFILE_JSON}" ]]; then
  PY_ARGS+=(--runfile-json "${RUNFILE_JSON}")
fi
if [[ -n "${TARGET_FNAME}" ]]; then
  PY_ARGS+=(--target-fname "${TARGET_FNAME}")
fi
if [[ -n "${MAX_ITEMS}" ]]; then
  PY_ARGS+=(--max-items "${MAX_ITEMS}")
fi
if [[ -n "${ALIGN_MIN_IOU_THRESHOLD}" ]]; then
  PY_ARGS+=(--align-min-iou-threshold "${ALIGN_MIN_IOU_THRESHOLD}")
fi
if [[ "${WITH_VISUALS}" == "1" ]]; then
  PY_ARGS+=(--with-visuals)
fi

echo "[run] text_metrics_report"
echo "[run]   script_dir=${SCRIPT_DIR}"
if [[ -n "${SCORES_PKL}" ]]; then
  echo "[run]   scores_pkl=${SCORES_PKL}"
fi
if [[ -n "${RUNFILE_JSON}" ]]; then
  echo "[run]   runfile_json=${RUNFILE_JSON}"
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  echo "[run]   output_dir=${OUTPUT_DIR}"
else
  echo "[run]   project_root_results=${PROJECT_ROOT_RESULTS}"
  echo "[run]   run_base_dir=${RUN_BASE_DIR}"
  echo "[run]   run_timestamp=${RUN_TIMESTAMP}"
  echo "[run]   run_dir=${RUN_DIR}"
fi
echo "[run]   report_dir=${REPORT_DIR}"
echo "[run]   window_size=${WINDOW_SIZE}"
echo "[run]   window_stride=${WINDOW_STRIDE}"
echo "[run]   hough_threshold=${HOUGH_THRESHOLD}"
echo "[run]   hough_line_length=${HOUGH_LINE_LENGTH}"
echo "[run]   hough_line_gap=${HOUGH_LINE_GAP}"
echo "[run]   hough_seed=${HOUGH_SEED}"
echo "[run]   hough_start=${HOUGH_START}"
echo "[run]   align_abs_min_len=${ALIGN_ABS_MIN_LEN}"
if [[ -n "${ALIGN_MIN_IOU_THRESHOLD}" ]]; then
  echo "[run]   align_min_iou_threshold=${ALIGN_MIN_IOU_THRESHOLD}"
fi
echo "[run]   line_filter_version=v2_1_true_iou"
echo "[run]   with_visuals=${WITH_VISUALS}"
if [[ -n "${TARGET_FNAME}" ]]; then
  echo "[run]   target_fname=${TARGET_FNAME}"
fi
if [[ -n "${MAX_ITEMS}" ]]; then
  echo "[run]   max_items=${MAX_ITEMS}"
fi

echo "[run] Stage 1/1: text_metrics_report.py"
python3 text_metrics_report.py "${PY_ARGS[@]}"

echo "[run] Done. Results written under: ${REPORT_DIR}"
