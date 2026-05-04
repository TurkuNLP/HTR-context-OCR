#!/usr/bin/env bash
#SBATCH --job-name=hough_param_sweep
#SBATCH --account=project_2017385
#SBATCH --partition=medium
#SBATCH --time=16:30:00
#SBATCH --nodes=10
#SBATCH --cpus-per-task=120
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/project_2017385/dorian/HTR-context-OCR/tuner_parallel_v2
#SBATCH -o /scratch/project_2017385/dorian/HTR-context-OCR/logs/hough_param_sweep_%j.out
#SBATCH -e /scratch/project_2017385/dorian/HTR-context-OCR/logs/hough_param_sweep_%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHARED_METRICS_DIR="${PROJECT_DIR}/text_metrics_v2_1_parallel"
LEGACY_METRICS_DIR="${PROJECT_DIR}/text_metrics_v2_1"
PYTHON_SCRIPTS_DIR="${PROJECT_DIR}/python_scripts"
LOG_DIR="${PROJECT_DIR}/logs"

cd "${SCRIPT_DIR}"

RUNFILE_JSON="${RUNFILE_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PROJECT_ROOT_RESULTS="${PROJECT_ROOT_RESULTS:-${PROJECT_DIR}/results/tuner_parallel_v2_hough_sweep}"
RUN_TAG="${RUN_TAG:-}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-35}"
MATRIX_CACHE_DIR="${MATRIX_CACHE_DIR:-${SCRIPT_DIR}/_matrix_cache}"
NO_MATRIX_CACHE="${NO_MATRIX_CACHE:-0}"

# Optional read-only matrix source from text_metrics score streams.
SCORES_PKL_REF_TO_PRED="${SCORES_PKL_REF_TO_PRED:-${PROJECT_DIR}/results/compares_churro_dev/ref_to_pred/scores_reference_prediction_ws50_st35.pkl}"
SCORE_INDEX_CACHE_FILE="${SCORE_INDEX_CACHE_FILE:-}"
SCORE_INDEX_CACHE_DIR="${SCORE_INDEX_CACHE_DIR:-${SHARED_METRICS_DIR}/.score_index_cache}"
DISABLE_PKL_MATRIX_SOURCE="${DISABLE_PKL_MATRIX_SOURCE:-0}"

MAX_ITEMS="${MAX_ITEMS:-}"
LEVENSHTEIN_BACKEND="${LEVENSHTEIN_BACKEND:-c}"
WORKERS="${WORKERS:-1}"
DOC_WORKERS="${DOC_WORKERS:-1}"

HOUGH_SEED="${HOUGH_SEED:-0}"
HOUGH_START="${HOUGH_START:-2.6}"
ALIGN_ABS_MIN_LEN="${ALIGN_ABS_MIN_LEN:-8.0}"
ALIGN_MIN_IOU_THRESHOLD="${ALIGN_MIN_IOU_THRESHOLD:-0.035}"

PLOT_ONLY="${PLOT_ONLY:-0}"
SUMMARY_JSON="${SUMMARY_JSON:-}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-}"
NO_OVERWRITE_SUMMARY="${NO_OVERWRITE_SUMMARY:-0}"

TARGET_FNAMES=()

usage() {
  cat <<'USAGE'
Usage: run_hough_parameter_sweep.sh [options]

Modes:
  Default mode runs full per-document tuner + summary output.
  --plot-only mode runs only plot generation from an existing summary JSON.

Required for default mode:
  --runfile-json <path>                       Path to outputs.json list

Outputs:
  --output-dir <dir>                          Exact output directory for the sweep
  --project-root-results <dir>                Root directory for auto timestamped runs
  --root-results <dir>                        Alias for --project-root-results
  --run-tag <tag>                             Optional suffix for auto timestamped run dir name

Selection:
  --target-fname <name>                       Optional file name filter. Repeat to include many.
  --max-items <n>                             Optional cap on number of selected documents

Matrix settings:
  --window-size <n>                           Sliding window size (default: 50)
  --window-stride <n>                         Sliding window stride (default: 35)
  --matrix-cache-dir <dir>                    Reusable score-matrix cache directory
  --no-matrix-cache                           Disable disk matrix cache for this run

Read-only score-stream matrix source (optional):
  --scores-pkl-ref-to-pred <path>             Path to scores_reference_prediction_ws*_st*.pkl
  --score-index-cache-file <path>             Explicit *.index.pkl file (read-only)
  --score-index-cache-dir <dir>               Directory with cached index files (read-only)
  --disable-pkl-matrix-source                 Force-disable pkl matrix source even if configured

Levenshtein:
  --levenshtein-backend <python|c>            Along-lines backend (default: c)

Parallel settings:
  --workers <n>                               Threshold-level worker threads per document
  --doc-workers <n>                           Documents processed in parallel
                                              (threshold workers auto-forced to 40 when doc-workers > 1)

Fixed tuner settings:
  --hough-seed <n>                            Fixed seed for all evaluations (default: 0)
  --hough-start <float>                       Adaptive threshold start (default: 2.6)
  --align-abs-min-len <float>                 Min line len before IoU filtering (default: 8.0)
  --align-min-iou-threshold <float>           IoU threshold (default: 0.035)

Grid is fixed automatically in code:
  threshold: 1..40, line_length: 1..50, line_gap: 1..30

Hough line-direction angle is fixed to strict diagonal bands:
  30 < x < 60 degrees (symmetric slants)

Plot-only mode:
  --plot-only                                 Skip sweep; run only plotting stage
  --summary-json <path>                       Existing hough_parameter_sweep_summary.json
  --plot-output-dir <dir>                     Optional override for plot output directory
  --no-overwrite-summary                      Do not write plot paths back into summary JSON

Other:
  -h, --help                                  Show this help text
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --run-tag)
      [[ $# -ge 2 ]] || { echo "[error] --run-tag requires a value" >&2; exit 1; }
      RUN_TAG="$2"
      shift 2
      ;;
    --target-fname)
      [[ $# -ge 2 ]] || { echo "[error] --target-fname requires a value" >&2; exit 1; }
      TARGET_FNAMES+=("$2")
      shift 2
      ;;
    --max-items)
      [[ $# -ge 2 ]] || { echo "[error] --max-items requires a value" >&2; exit 1; }
      MAX_ITEMS="$2"
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
    --matrix-cache-dir)
      [[ $# -ge 2 ]] || { echo "[error] --matrix-cache-dir requires a value" >&2; exit 1; }
      MATRIX_CACHE_DIR="$2"
      shift 2
      ;;
    --no-matrix-cache)
      NO_MATRIX_CACHE="1"
      shift
      ;;
    --scores-pkl-ref-to-pred)
      [[ $# -ge 2 ]] || { echo "[error] --scores-pkl-ref-to-pred requires a value" >&2; exit 1; }
      SCORES_PKL_REF_TO_PRED="$2"
      shift 2
      ;;
    --score-index-cache-file)
      [[ $# -ge 2 ]] || { echo "[error] --score-index-cache-file requires a value" >&2; exit 1; }
      SCORE_INDEX_CACHE_FILE="$2"
      shift 2
      ;;
    --score-index-cache-dir)
      [[ $# -ge 2 ]] || { echo "[error] --score-index-cache-dir requires a value" >&2; exit 1; }
      SCORE_INDEX_CACHE_DIR="$2"
      shift 2
      ;;
    --disable-pkl-matrix-source)
      DISABLE_PKL_MATRIX_SOURCE="1"
      shift
      ;;
    --levenshtein-backend)
      [[ $# -ge 2 ]] || { echo "[error] --levenshtein-backend requires a value" >&2; exit 1; }
      LEVENSHTEIN_BACKEND="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "[error] --workers requires a value" >&2; exit 1; }
      WORKERS="$2"
      shift 2
      ;;
    --doc-workers)
      [[ $# -ge 2 ]] || { echo "[error] --doc-workers requires a value" >&2; exit 1; }
      DOC_WORKERS="$2"
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
    --plot-only)
      PLOT_ONLY="1"
      shift
      ;;
    --summary-json)
      [[ $# -ge 2 ]] || { echo "[error] --summary-json requires a value" >&2; exit 1; }
      SUMMARY_JSON="$2"
      shift 2
      ;;
    --plot-output-dir)
      [[ $# -ge 2 ]] || { echo "[error] --plot-output-dir requires a value" >&2; exit 1; }
      PLOT_OUTPUT_DIR="$2"
      shift 2
      ;;
    --no-overwrite-summary)
      NO_OVERWRITE_SUMMARY="1"
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
if [[ "${LEVENSHTEIN_BACKEND}" != "python" && "${LEVENSHTEIN_BACKEND}" != "c" ]]; then
  echo "[error] --levenshtein-backend must be one of: python, c (got: ${LEVENSHTEIN_BACKEND})" >&2
  exit 1
fi
if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --workers must be a positive integer (got: ${WORKERS})" >&2
  exit 1
fi
if ! [[ "${DOC_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --doc-workers must be a positive integer (got: ${DOC_WORKERS})" >&2
  exit 1
fi
if ! [[ "${HOUGH_SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "[error] --hough-seed must be an integer (got: ${HOUGH_SEED})" >&2
  exit 1
fi
for raw_float in "${HOUGH_START}" "${ALIGN_ABS_MIN_LEN}" "${ALIGN_MIN_IOU_THRESHOLD}"; do
  if ! [[ "${raw_float}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "[error] Numeric parameter has invalid value: ${raw_float}" >&2
    exit 1
  fi
done
if [[ "${PLOT_ONLY}" != "0" && "${PLOT_ONLY}" != "1" ]]; then
  echo "[error] PLOT_ONLY must be 0 or 1 (got: ${PLOT_ONLY})" >&2
  exit 1
fi
if [[ "${NO_MATRIX_CACHE}" != "0" && "${NO_MATRIX_CACHE}" != "1" ]]; then
  echo "[error] NO_MATRIX_CACHE must be 0 or 1 (got: ${NO_MATRIX_CACHE})" >&2
  exit 1
fi
if [[ "${DISABLE_PKL_MATRIX_SOURCE}" != "0" && "${DISABLE_PKL_MATRIX_SOURCE}" != "1" ]]; then
  echo "[error] DISABLE_PKL_MATRIX_SOURCE must be 0 or 1 (got: ${DISABLE_PKL_MATRIX_SOURCE})" >&2
  exit 1
fi
if [[ "${NO_MATRIX_CACHE}" == "0" && -z "${MATRIX_CACHE_DIR}" ]]; then
  echo "[error] --matrix-cache-dir must not be empty unless --no-matrix-cache is set" >&2
  exit 1
fi
if [[ "${NO_OVERWRITE_SUMMARY}" != "0" && "${NO_OVERWRITE_SUMMARY}" != "1" ]]; then
  echo "[error] NO_OVERWRITE_SUMMARY must be 0 or 1 (got: ${NO_OVERWRITE_SUMMARY})" >&2
  exit 1
fi

if [[ -n "${SCORES_PKL_REF_TO_PRED}" && ! -f "${SCORES_PKL_REF_TO_PRED}" ]]; then
  echo "[error] SCORES_PKL_REF_TO_PRED does not exist: ${SCORES_PKL_REF_TO_PRED}" >&2
  exit 1
fi
if [[ -n "${SCORE_INDEX_CACHE_FILE}" && ! -f "${SCORE_INDEX_CACHE_FILE}" ]]; then
  echo "[error] SCORE_INDEX_CACHE_FILE does not exist: ${SCORE_INDEX_CACHE_FILE}" >&2
  exit 1
fi
if [[ -n "${SCORE_INDEX_CACHE_DIR}" && ! -d "${SCORE_INDEX_CACHE_DIR}" ]]; then
  echo "[warn] SCORE_INDEX_CACHE_DIR does not exist (will fall back to in-memory index if needed): ${SCORE_INDEX_CACHE_DIR}" >&2
fi

mkdir -p "${LOG_DIR}"

if [[ "${PLOT_ONLY}" == "1" ]]; then
  if [[ -z "${SUMMARY_JSON}" ]]; then
    if [[ -z "${OUTPUT_DIR}" ]]; then
      echo "[error] --plot-only requires --summary-json or --output-dir" >&2
      exit 1
    fi
    SUMMARY_JSON="${OUTPUT_DIR%/}/hough_parameter_sweep_summary.json"
  fi
  if [[ ! -f "${SUMMARY_JSON}" ]]; then
    echo "[error] summary JSON does not exist: ${SUMMARY_JSON}" >&2
    exit 1
  fi

  PLOT_ARGS=(--summary-json "${SUMMARY_JSON}")
  if [[ -n "${PLOT_OUTPUT_DIR}" ]]; then
    PLOT_ARGS+=(--output-dir "${PLOT_OUTPUT_DIR}")
  fi
  if [[ "${NO_OVERWRITE_SUMMARY}" == "1" ]]; then
    PLOT_ARGS+=(--no-overwrite-summary)
  fi

  echo "[run] hough_parameter_sweep plot-only mode"
  echo "[run]   summary_json=${SUMMARY_JSON}"
  if [[ -n "${PLOT_OUTPUT_DIR}" ]]; then
    echo "[run]   plot_output_dir=${PLOT_OUTPUT_DIR}"
  fi
  python3 plot_hough_parameter_sweep.py "${PLOT_ARGS[@]}"
  echo "[run] Done (plot-only)."
  exit 0
fi

if [[ -z "${RUNFILE_JSON}" ]]; then
  echo "[error] --runfile-json is required in default mode" >&2
  exit 1
fi
if [[ ! -f "${RUNFILE_JSON}" ]]; then
  echo "[error] RUNFILE_JSON does not exist: ${RUNFILE_JSON}" >&2
  exit 1
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  SWEEP_DIR="${OUTPUT_DIR}"
  RUN_DIR="$(dirname "${SWEEP_DIR}")"
else
  RUN_BASE_DIR="${PROJECT_ROOT_RESULTS}/window_${WINDOW_SIZE}_stride_${WINDOW_STRIDE}"
  RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_TAG_SAFE="$(printf '%s' "${RUN_TAG}" | tr -cs 'A-Za-z0-9._-' '_')"
  if [[ -n "${RUN_TAG_SAFE}" ]]; then
    RUN_DIR="${RUN_BASE_DIR}/${RUN_TIMESTAMP}_${RUN_TAG_SAFE}"
  else
    RUN_DIR="${RUN_BASE_DIR}/${RUN_TIMESTAMP}"
  fi
  while [[ -e "${RUN_DIR}" ]]; do
    sleep 1
    RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    if [[ -n "${RUN_TAG_SAFE}" ]]; then
      RUN_DIR="${RUN_BASE_DIR}/${RUN_TIMESTAMP}_${RUN_TAG_SAFE}"
    else
      RUN_DIR="${RUN_BASE_DIR}/${RUN_TIMESTAMP}"
    fi
  done
  SWEEP_DIR="${RUN_DIR}/hough_parameter_sweep"
fi

mkdir -p "${SWEEP_DIR}"
if [[ -n "${RUN_DIR:-}" ]]; then
  mkdir -p "${RUN_DIR}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}:${SHARED_METRICS_DIR}:${LEGACY_METRICS_DIR}:${PYTHON_SCRIPTS_DIR}:${PYTHONPATH:-}"

PY_ARGS=(
  --runfile-json "${RUNFILE_JSON}"
  --output-dir "${SWEEP_DIR}"
  --window-size "${WINDOW_SIZE}"
  --window-stride "${WINDOW_STRIDE}"
  --levenshtein-backend "${LEVENSHTEIN_BACKEND}"
  --workers "${WORKERS}"
  --doc-workers "${DOC_WORKERS}"
  --hough-seed "${HOUGH_SEED}"
  --hough-start "${HOUGH_START}"
  --align-abs-min-len "${ALIGN_ABS_MIN_LEN}"
  --align-min-iou-threshold "${ALIGN_MIN_IOU_THRESHOLD}"
)

if [[ "${NO_MATRIX_CACHE}" == "1" ]]; then
  PY_ARGS+=(--no-matrix-cache)
else
  PY_ARGS+=(--matrix-cache-dir "${MATRIX_CACHE_DIR}")
fi

if [[ -n "${SCORES_PKL_REF_TO_PRED}" ]]; then
  PY_ARGS+=(--scores-pkl-ref-to-pred "${SCORES_PKL_REF_TO_PRED}")
fi
if [[ -n "${SCORE_INDEX_CACHE_FILE}" ]]; then
  PY_ARGS+=(--score-index-cache-file "${SCORE_INDEX_CACHE_FILE}")
fi
if [[ -n "${SCORE_INDEX_CACHE_DIR}" ]]; then
  PY_ARGS+=(--score-index-cache-dir "${SCORE_INDEX_CACHE_DIR}")
fi
if [[ "${DISABLE_PKL_MATRIX_SOURCE}" == "1" ]]; then
  PY_ARGS+=(--disable-pkl-matrix-source)
fi

if [[ -n "${MAX_ITEMS}" ]]; then
  PY_ARGS+=(--max-items "${MAX_ITEMS}")
fi
for target in "${TARGET_FNAMES[@]}"; do
  PY_ARGS+=(--target-fname "${target}")
done

echo "[run] hough_parameter_sweep"
echo "[run]   script_dir=${SCRIPT_DIR}"
echo "[run]   runfile_json=${RUNFILE_JSON}"
echo "[run]   output_dir=${SWEEP_DIR}"
echo "[run]   window_size=${WINDOW_SIZE}"
echo "[run]   window_stride=${WINDOW_STRIDE}"
echo "[run]   fixed_grid=threshold:1..40,line_length:1..50,line_gap:1..30"
echo "[run]   hough_angle=strict(30,60)"
echo "[run]   requested_threshold_workers=${WORKERS}"
echo "[run]   doc_parallel_rule=force40_threshold_workers_when_doc_workers_gt_1"
echo "[run]   doc_workers=${DOC_WORKERS}"
echo "[run]   hough_seed=${HOUGH_SEED}"
echo "[run]   levenshtein_backend=${LEVENSHTEIN_BACKEND}"
if [[ "${NO_MATRIX_CACHE}" == "1" ]]; then
  echo "[run]   matrix_cache=disabled"
else
  echo "[run]   matrix_cache_dir=${MATRIX_CACHE_DIR}"
fi
echo "[run]   scores_pkl_ref_to_pred=${SCORES_PKL_REF_TO_PRED:-None}"
echo "[run]   score_index_cache_file=${SCORE_INDEX_CACHE_FILE:-None}"
echo "[run]   score_index_cache_dir=${SCORE_INDEX_CACHE_DIR:-None}"
echo "[run]   disable_pkl_matrix_source=${DISABLE_PKL_MATRIX_SOURCE}"
if [[ ${#TARGET_FNAMES[@]} -gt 0 ]]; then
  echo "[run]   target_fnames=${TARGET_FNAMES[*]}"
fi
if [[ -n "${MAX_ITEMS}" ]]; then
  echo "[run]   max_items=${MAX_ITEMS}"
fi

echo "[run] Stage 1/1: run_hough_parameter_sweep.py"
python3 run_hough_parameter_sweep.py "${PY_ARGS[@]}"

echo "[run] Done. Results written under: ${SWEEP_DIR}"
echo "[run] Summary JSON: ${SWEEP_DIR}/hough_parameter_sweep_summary.json"
