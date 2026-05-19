#!/usr/bin/env bash
#SBATCH --job-name=hough_param_sweep
#SBATCH --account=project_2000539
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2
#SBATCH -o /scratch/project_2017385/dorian/Churro_copy/logs/hough_param_sweep_%j.out
#SBATCH -e /scratch/project_2017385/dorian/Churro_copy/logs/hough_param_sweep_%j.err

set -euo pipefail
if command -v module >/dev/null 2>&1; then
  module purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

# Keep shell-side Slurm logs readable before Python starts.
# Python-side logs use the same timestamp format through timestamped_logging.py.
timestamp_now() {
  date '+%Y-%m-%d %H:%M:%S'
}

# Emit one timestamped informational line to stdout.
log_info() {
  printf '[%s] %s\n' "$(timestamp_now)" "$*"
}

# Emit one timestamped diagnostic/error line to stderr.
log_error() {
  printf '[%s] %s\n' "$(timestamp_now)" "$*" >&2
}

# Central project paths used throughout the wrapper.
PROJECT_DIR="/scratch/project_2017385/dorian/Churro_copy"
SCRIPT_DIR="${PROJECT_DIR}/tuner_parallel_v2"
SHARED_METRICS_DIR="${PROJECT_DIR}/text_metrics_v2_1_parallel"
LMOD_INIT="${LMOD_INIT:-/usr/share/lmod/8.6.17/init/bash}"
MODULEFILES_DIR="${MODULEFILES_DIR:-/appl/modulefiles}"
PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_CYTHON_EXTENSIONS="${BUILD_CYTHON_EXTENSIONS:-1}"
REQUIRE_CYTHON_EXTENSIONS="${REQUIRE_CYTHON_EXTENSIONS:-1}"

cd "${SCRIPT_DIR}"

if [[ ! -f "${LMOD_INIT}" ]]; then
  log_error "[error] Lmod init script not found: ${LMOD_INIT}"
  exit 1
fi

# Initialize the HPC module system and load the Python/Cython runtime used by
# the tuner.
source "${LMOD_INIT}"
module use "${MODULEFILES_DIR}"
module load "${PYTORCH_MODULE}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log_error "[error] Python executable not found after module load: ${PYTHON_BIN}"
  exit 1
fi

# Default arguments can be overridden either by CLI flags or environment vars.
RUNFILE_JSON="${RUNFILE_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_SHARD_NAME="${OUTPUT_SHARD_NAME:-}"
PROJECT_ROOT_RESULTS="${PROJECT_ROOT_RESULTS:-${PROJECT_DIR}/results/tuner_parallel_v2_hough_sweep}"
RUN_TAG="${RUN_TAG:-}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-35}"
MATRIX_CACHE_DIR="${MATRIX_CACHE_DIR:-${SCRIPT_DIR}/_matrix_cache}"
NO_MATRIX_CACHE="${NO_MATRIX_CACHE:-0}"

# Optional read-only matrix source from text_metrics score streams.
SCORES_PKL_REF_TO_PRED="${SCORES_PKL_REF_TO_PRED:-${PROJECT_DIR}/results/compares_churro_dev/ref_to_pred/scores_reference_prediction_ws50_st35.pkl}"
SCORES_PKL_REF_TO_REF="${SCORES_PKL_REF_TO_REF:-${PROJECT_DIR}/results/compares_churro_dev/ref_to_ref/scores_reference_self_ws50_st35.pkl}"
SCORE_INDEX_CACHE_FILE="${SCORE_INDEX_CACHE_FILE:-}"
SCORE_INDEX_CACHE_FILE_REF_TO_REF="${SCORE_INDEX_CACHE_FILE_REF_TO_REF:-}"
SCORE_INDEX_CACHE_DIR="${SCORE_INDEX_CACHE_DIR:-${SHARED_METRICS_DIR}/.score_index_cache}"
DISABLE_PKL_MATRIX_SOURCE="${DISABLE_PKL_MATRIX_SOURCE:-0}"
TEXT_METRICS_V212_DIR="${TEXT_METRICS_V212_DIR:-${PROJECT_DIR}/text_metrics_v2_12_parallel}"
REF_TO_REF_CACHE_MODE="${REF_TO_REF_CACHE_MODE:-auto}"
REF_TO_REF_CACHE_DIR="${REF_TO_REF_CACHE_DIR:-${PROJECT_DIR}/results/tuner_parallel_v2_cache/ref_to_ref_combo_cache_v1}"
REF_TO_REF_CACHE_WARM_ONLY="${REF_TO_REF_CACHE_WARM_ONLY:-0}"

MAX_ITEMS="${MAX_ITEMS:-}"
SELECTION_INDEX_RANGE_START="${SELECTION_INDEX_RANGE_START:-}"
SELECTION_INDEX_RANGE_END="${SELECTION_INDEX_RANGE_END:-}"
LEVENSHTEIN_BACKEND="${LEVENSHTEIN_BACKEND:-c}"
WORKERS="${WORKERS:-1}"
DOC_WORKERS="${DOC_WORKERS:-1}"

# --hough-seed and --seed-range are compatibility-only while seed search is
# temporarily disabled.  The Python runner evaluates the fixed grid seed 1.
HOUGH_SEED="${HOUGH_SEED:-1}"
HOUGH_START="${HOUGH_START:-2.6}"
ALIGN_ABS_MIN_LEN="${ALIGN_ABS_MIN_LEN:-8.0}"
ALIGN_MIN_IOU_THRESHOLD="${ALIGN_MIN_IOU_THRESHOLD:-0.035}"

HOUGH_THRESHOLD_RANGE_START="${HOUGH_THRESHOLD_RANGE_START:-}"
HOUGH_THRESHOLD_RANGE_END="${HOUGH_THRESHOLD_RANGE_END:-}"
HOUGH_LINE_LENGTH_RANGE_START="${HOUGH_LINE_LENGTH_RANGE_START:-}"
HOUGH_LINE_LENGTH_RANGE_END="${HOUGH_LINE_LENGTH_RANGE_END:-}"
HOUGH_LINE_GAP_RANGE_START="${HOUGH_LINE_GAP_RANGE_START:-}"
HOUGH_LINE_GAP_RANGE_END="${HOUGH_LINE_GAP_RANGE_END:-}"
HOUGH_SEED_RANGE_START="${HOUGH_SEED_RANGE_START:-}"
HOUGH_SEED_RANGE_END="${HOUGH_SEED_RANGE_END:-}"

PLOT_ONLY="${PLOT_ONLY:-0}"
SUMMARY_JSON="${SUMMARY_JSON:-}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-}"
NO_OVERWRITE_SUMMARY="${NO_OVERWRITE_SUMMARY:-0}"
WITH_VISUALS="${WITH_VISUALS:-0}"
HIDE_LINE_LABELS="${HIDE_LINE_LABELS:-0}"

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
  --output-shard-name <name>                  Write under <output-dir>/shards/<name> to avoid parallel overwrite
  --project-root-results <dir>                Root directory for auto timestamped runs
  --root-results <dir>                        Alias for --project-root-results
  --run-tag <tag>                             Optional suffix for auto timestamped run dir name

Selection:
  --target-fname <name>                       Optional file name filter. Repeat to include many.
  --max-items <n>                             Optional cap on number of selected documents
  --selection-index-range <start> <end>       Zero-based inclusive range after target/max selection
  --item-index-range <start> <end>            Alias for --selection-index-range
  --document-index-range <start> <end>        Alias for --selection-index-range

Matrix settings:
  --window-size <n>                           Sliding window size (default: 50)
  --window-stride <n>                         Sliding window stride (default: 35)
  --matrix-cache-dir <dir>                    Reusable score-matrix cache directory
  --no-matrix-cache                           Disable disk matrix cache for this run

Read-only score-stream matrix source (optional):
  --scores-pkl-ref-to-pred <path>             Path to scores_reference_prediction_ws*_st*.pkl
  --scores-pkl-ref-to-ref <path>              Path to scores_reference_self_ws*_st*.pkl
  --score-index-cache-file <path>             Explicit ref-to-pred *.index.pkl file (read-only)
  --score-index-cache-file-ref-to-ref <path>  Explicit ref-to-ref *.index.pkl file (read-only)
  --score-index-cache-dir <dir>               Directory with cached index files (read-only)
  --disable-pkl-matrix-source                 Force-disable pkl matrix source even if configured

V2.12 metric source:
  --text-metrics-v212-dir <dir>               Read-only v2.12 directory for coverage/hallucination logic

Reference-self combination cache:
  --ref-to-ref-cache-mode <off|auto|read-only>
                                               Cache exact ref_to_ref coverage baselines (default: auto)
  --ref-to-ref-cache-dir <dir>                 Cache directory outside the source tree
  --ref-to-ref-cache-warm-only                 Fill ref_to_ref cache and exit before prediction-side tuning

Levenshtein:
  --levenshtein-backend <python|c>            Along-lines backend (default: c)

Parallel settings:
  --workers <n>                               Threshold-level worker threads per document
  --doc-workers <n>                           Documents processed in parallel
                                              (threshold workers auto-forced to 40 when doc-workers > 1)

Shared non-grid tuner settings:
  --hough-seed <n>                            Deprecated compatibility arg; ignored by sweep runner
  --hough-start <float>                       Adaptive threshold start (default: 2.6)
  --align-abs-min-len <float>                 Min line len before IoU filtering (default: 8.0)
  --align-min-iou-threshold <float>           IoU threshold (default: 0.035)

Hough sweep ranges (inclusive; omitted ranges keep defaults):
  --hough-threshold-range <start> <end>       Threshold range, default 1..40
  --threshold <start> <end>                   Alias for --hough-threshold-range
  --line-length-range <start> <end>           Hough line_length range, default 1..50
  --line_length <start> <end>                 Alias for --line-length-range
  --line-gap-range <start> <end>              Hough line_gap range, default 1..30
  --line_gap <start> <end>                    Alias for --line-gap-range
  --seed-range <start> <end>                  Deprecated; fixed seed 1 is used
  --seed <start> <end>                        Alias for --seed-range

Hough line-direction angle is fixed to falling diagonals:
  left-to-right 30..60 degrees, upper-left to lower-right only

Plot-only mode:
  --plot-only                                 Skip sweep; run only plotting stage
  --summary-json <path>                       Existing hough_parameter_sweep_summary.json
  --plot-output-dir <dir>                     Optional override for plot output directory
  --no-overwrite-summary                      Do not write plot paths back into summary JSON
  --with-visuals                              Generate final colour plots and full combination bundles
  --hide-line-labels                          Hide line labels in stitched best-combination plots

Other:
  BUILD_CYTHON_EXTENSIONS=0|1                 Env var. Build Cython extensions before full run (default: 1)
  REQUIRE_CYTHON_EXTENSIONS=0|1               Env var. Fail if compiled helpers are unavailable (default: 1)
  LMOD_INIT=<path>                            Env var. Lmod init script (default: /usr/share/lmod/8.6.17/init/bash)
  MODULEFILES_DIR=<path>                      Env var. Modulefile root (default: /appl/modulefiles)
  PYTORCH_MODULE=<name>                       Env var. Runtime module name (default: pytorch)
  PYTHON_BIN=<name>                           Env var. Python executable after module load (default: python3)
  -h, --help                                  Show this help text
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runfile-json)
      [[ $# -ge 2 ]] || { log_error "[error] --runfile-json requires a value"; exit 1; }
      RUNFILE_JSON="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { log_error "[error] --output-dir requires a value"; exit 1; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-shard-name)
      [[ $# -ge 2 ]] || { log_error "[error] --output-shard-name requires a value"; exit 1; }
      OUTPUT_SHARD_NAME="$2"
      shift 2
      ;;
    --project-root-results|--root-results)
      [[ $# -ge 2 ]] || { log_error "[error] $1 requires a value"; exit 1; }
      PROJECT_ROOT_RESULTS="$2"
      shift 2
      ;;
    --run-tag)
      [[ $# -ge 2 ]] || { log_error "[error] --run-tag requires a value"; exit 1; }
      RUN_TAG="$2"
      shift 2
      ;;
    --target-fname)
      [[ $# -ge 2 ]] || { log_error "[error] --target-fname requires a value"; exit 1; }
      TARGET_FNAMES+=("$2")
      shift 2
      ;;
    --max-items)
      [[ $# -ge 2 ]] || { log_error "[error] --max-items requires a value"; exit 1; }
      MAX_ITEMS="$2"
      shift 2
      ;;
    --selection-index-range|--item-index-range|--document-index-range)
      [[ $# -ge 3 ]] || { log_error "[error] $1 requires start and end values"; exit 1; }
      SELECTION_INDEX_RANGE_START="$2"
      SELECTION_INDEX_RANGE_END="$3"
      shift 3
      ;;
    --window-size)
      [[ $# -ge 2 ]] || { log_error "[error] --window-size requires a value"; exit 1; }
      WINDOW_SIZE="$2"
      shift 2
      ;;
    --window-stride)
      [[ $# -ge 2 ]] || { log_error "[error] --window-stride requires a value"; exit 1; }
      WINDOW_STRIDE="$2"
      shift 2
      ;;
    --matrix-cache-dir)
      [[ $# -ge 2 ]] || { log_error "[error] --matrix-cache-dir requires a value"; exit 1; }
      MATRIX_CACHE_DIR="$2"
      shift 2
      ;;
    --no-matrix-cache)
      NO_MATRIX_CACHE="1"
      shift
      ;;
    --scores-pkl-ref-to-pred)
      [[ $# -ge 2 ]] || { log_error "[error] --scores-pkl-ref-to-pred requires a value"; exit 1; }
      SCORES_PKL_REF_TO_PRED="$2"
      shift 2
      ;;
    --score-index-cache-file)
      [[ $# -ge 2 ]] || { log_error "[error] --score-index-cache-file requires a value"; exit 1; }
      SCORE_INDEX_CACHE_FILE="$2"
      shift 2
      ;;
    --score-index-cache-file-ref-to-ref)
      [[ $# -ge 2 ]] || { log_error "[error] --score-index-cache-file-ref-to-ref requires a value"; exit 1; }
      SCORE_INDEX_CACHE_FILE_REF_TO_REF="$2"
      shift 2
      ;;
    --score-index-cache-dir)
      [[ $# -ge 2 ]] || { log_error "[error] --score-index-cache-dir requires a value"; exit 1; }
      SCORE_INDEX_CACHE_DIR="$2"
      shift 2
      ;;
    --scores-pkl-ref-to-ref)
      [[ $# -ge 2 ]] || { log_error "[error] --scores-pkl-ref-to-ref requires a value"; exit 1; }
      SCORES_PKL_REF_TO_REF="$2"
      shift 2
      ;;
    --text-metrics-v212-dir)
      [[ $# -ge 2 ]] || { log_error "[error] --text-metrics-v212-dir requires a value"; exit 1; }
      TEXT_METRICS_V212_DIR="$2"
      shift 2
      ;;
    --ref-to-ref-cache-mode)
      [[ $# -ge 2 ]] || { log_error "[error] --ref-to-ref-cache-mode requires a value"; exit 1; }
      REF_TO_REF_CACHE_MODE="$2"
      shift 2
      ;;
    --ref-to-ref-cache-dir)
      [[ $# -ge 2 ]] || { log_error "[error] --ref-to-ref-cache-dir requires a value"; exit 1; }
      REF_TO_REF_CACHE_DIR="$2"
      shift 2
      ;;
    --ref-to-ref-cache-warm-only)
      REF_TO_REF_CACHE_WARM_ONLY="1"
      shift
      ;;
    --hough-threshold-range|--threshold)
      [[ $# -ge 3 ]] || { log_error "[error] $1 requires start and end values"; exit 1; }
      HOUGH_THRESHOLD_RANGE_START="$2"
      HOUGH_THRESHOLD_RANGE_END="$3"
      shift 3
      ;;
    --line-length-range|--line_length|--line-length)
      [[ $# -ge 3 ]] || { log_error "[error] $1 requires start and end values"; exit 1; }
      HOUGH_LINE_LENGTH_RANGE_START="$2"
      HOUGH_LINE_LENGTH_RANGE_END="$3"
      shift 3
      ;;
    --line-gap-range|--line_gap|--line-gap)
      [[ $# -ge 3 ]] || { log_error "[error] $1 requires start and end values"; exit 1; }
      HOUGH_LINE_GAP_RANGE_START="$2"
      HOUGH_LINE_GAP_RANGE_END="$3"
      shift 3
      ;;
    --seed-range|--seed)
      [[ $# -ge 3 ]] || { log_error "[error] $1 requires start and end values"; exit 1; }
      HOUGH_SEED_RANGE_START="$2"
      HOUGH_SEED_RANGE_END="$3"
      shift 3
      ;;
    --disable-pkl-matrix-source)
      DISABLE_PKL_MATRIX_SOURCE="1"
      shift
      ;;
    --levenshtein-backend)
      [[ $# -ge 2 ]] || { log_error "[error] --levenshtein-backend requires a value"; exit 1; }
      LEVENSHTEIN_BACKEND="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { log_error "[error] --workers requires a value"; exit 1; }
      WORKERS="$2"
      shift 2
      ;;
    --doc-workers)
      [[ $# -ge 2 ]] || { log_error "[error] --doc-workers requires a value"; exit 1; }
      DOC_WORKERS="$2"
      shift 2
      ;;
    --hough-seed)
      [[ $# -ge 2 ]] || { log_error "[error] --hough-seed requires a value"; exit 1; }
      HOUGH_SEED="$2"
      shift 2
      ;;
    --hough-start)
      [[ $# -ge 2 ]] || { log_error "[error] --hough-start requires a value"; exit 1; }
      HOUGH_START="$2"
      shift 2
      ;;
    --align-abs-min-len)
      [[ $# -ge 2 ]] || { log_error "[error] --align-abs-min-len requires a value"; exit 1; }
      ALIGN_ABS_MIN_LEN="$2"
      shift 2
      ;;
    --align-min-iou-threshold)
      [[ $# -ge 2 ]] || { log_error "[error] --align-min-iou-threshold requires a value"; exit 1; }
      ALIGN_MIN_IOU_THRESHOLD="$2"
      shift 2
      ;;
    --plot-only)
      PLOT_ONLY="1"
      shift
      ;;
    --summary-json)
      [[ $# -ge 2 ]] || { log_error "[error] --summary-json requires a value"; exit 1; }
      SUMMARY_JSON="$2"
      shift 2
      ;;
    --plot-output-dir)
      [[ $# -ge 2 ]] || { log_error "[error] --plot-output-dir requires a value"; exit 1; }
      PLOT_OUTPUT_DIR="$2"
      shift 2
      ;;
    --no-overwrite-summary)
      NO_OVERWRITE_SUMMARY="1"
      shift
      ;;
    --with-visuals)
      WITH_VISUALS="1"
      shift
      ;;
    --hide-line-labels)
      HIDE_LINE_LABELS="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log_error "[error] Unknown argument: $1"
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "${WINDOW_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  log_error "[error] --window-size must be a positive integer (got: ${WINDOW_SIZE})"
  exit 1
fi
if ! [[ "${WINDOW_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  log_error "[error] --window-stride must be a positive integer (got: ${WINDOW_STRIDE})"
  exit 1
fi
if [[ -n "${MAX_ITEMS}" ]] && ! [[ "${MAX_ITEMS}" =~ ^[1-9][0-9]*$ ]]; then
  log_error "[error] --max-items must be a positive integer (got: ${MAX_ITEMS})"
  exit 1
fi
if [[ "${LEVENSHTEIN_BACKEND}" != "python" && "${LEVENSHTEIN_BACKEND}" != "c" ]]; then
  log_error "[error] --levenshtein-backend must be one of: python, c (got: ${LEVENSHTEIN_BACKEND})"
  exit 1
fi
if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  log_error "[error] --workers must be a positive integer (got: ${WORKERS})"
  exit 1
fi
if ! [[ "${DOC_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  log_error "[error] --doc-workers must be a positive integer (got: ${DOC_WORKERS})"
  exit 1
fi
if ! [[ "${HOUGH_SEED}" =~ ^-?[0-9]+$ ]]; then
  log_error "[error] --hough-seed must be an integer (got: ${HOUGH_SEED})"
  exit 1
fi
for raw_float in "${HOUGH_START}" "${ALIGN_ABS_MIN_LEN}" "${ALIGN_MIN_IOU_THRESHOLD}"; do
  if ! [[ "${raw_float}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    log_error "[error] Numeric parameter has invalid value: ${raw_float}"
    exit 1
  fi
done
validate_optional_range_pair() {
  local label="$1"
  local start_value="$2"
  local end_value="$3"
  local allow_zero="$4"
  if [[ -z "${start_value}" && -z "${end_value}" ]]; then
    return 0
  fi
  if [[ -z "${start_value}" || -z "${end_value}" ]]; then
    log_error "[error] ${label} requires both start and end values"
    exit 1
  fi
  if ! [[ "${start_value}" =~ ^[0-9]+$ && "${end_value}" =~ ^[0-9]+$ ]]; then
    log_error "[error] ${label} range values must be non-negative integers (got: ${start_value} ${end_value})"
    exit 1
  fi
  if [[ "${allow_zero}" != "1" && ( "${start_value}" == "0" || "${end_value}" == "0" ) ]]; then
    log_error "[error] ${label} range values must be >= 1 (got: ${start_value} ${end_value})"
    exit 1
  fi
  if (( start_value > end_value )); then
    log_error "[error] ${label} range start must be <= end (got: ${start_value} ${end_value})"
    exit 1
  fi
}

validate_optional_range_pair "hough-threshold" "${HOUGH_THRESHOLD_RANGE_START}" "${HOUGH_THRESHOLD_RANGE_END}" 0
validate_optional_range_pair "line-length" "${HOUGH_LINE_LENGTH_RANGE_START}" "${HOUGH_LINE_LENGTH_RANGE_END}" 0
validate_optional_range_pair "line-gap" "${HOUGH_LINE_GAP_RANGE_START}" "${HOUGH_LINE_GAP_RANGE_END}" 1
validate_optional_range_pair "seed" "${HOUGH_SEED_RANGE_START}" "${HOUGH_SEED_RANGE_END}" 1
validate_optional_range_pair "selection-index" "${SELECTION_INDEX_RANGE_START}" "${SELECTION_INDEX_RANGE_END}" 1

if [[ "${PLOT_ONLY}" != "0" && "${PLOT_ONLY}" != "1" ]]; then
  log_error "[error] PLOT_ONLY must be 0 or 1 (got: ${PLOT_ONLY})"
  exit 1
fi
if [[ "${NO_MATRIX_CACHE}" != "0" && "${NO_MATRIX_CACHE}" != "1" ]]; then
  log_error "[error] NO_MATRIX_CACHE must be 0 or 1 (got: ${NO_MATRIX_CACHE})"
  exit 1
fi
if [[ "${DISABLE_PKL_MATRIX_SOURCE}" != "0" && "${DISABLE_PKL_MATRIX_SOURCE}" != "1" ]]; then
  log_error "[error] DISABLE_PKL_MATRIX_SOURCE must be 0 or 1 (got: ${DISABLE_PKL_MATRIX_SOURCE})"
  exit 1
fi
if [[ "${NO_MATRIX_CACHE}" == "0" && -z "${MATRIX_CACHE_DIR}" ]]; then
  log_error "[error] --matrix-cache-dir must not be empty unless --no-matrix-cache is set"
  exit 1
fi
if [[ "${NO_OVERWRITE_SUMMARY}" != "0" && "${NO_OVERWRITE_SUMMARY}" != "1" ]]; then
  log_error "[error] NO_OVERWRITE_SUMMARY must be 0 or 1 (got: ${NO_OVERWRITE_SUMMARY})"
  exit 1
fi
if [[ "${WITH_VISUALS}" != "0" && "${WITH_VISUALS}" != "1" ]]; then
  log_error "[error] WITH_VISUALS must be 0 or 1 (got: ${WITH_VISUALS})"
  exit 1
fi
if [[ "${HIDE_LINE_LABELS}" != "0" && "${HIDE_LINE_LABELS}" != "1" ]]; then
  log_error "[error] HIDE_LINE_LABELS must be 0 or 1 (got: ${HIDE_LINE_LABELS})"
  exit 1
fi
if [[ "${BUILD_CYTHON_EXTENSIONS}" != "0" && "${BUILD_CYTHON_EXTENSIONS}" != "1" ]]; then
  log_error "[error] BUILD_CYTHON_EXTENSIONS must be 0 or 1 (got: ${BUILD_CYTHON_EXTENSIONS})"
  exit 1
fi
if [[ "${REQUIRE_CYTHON_EXTENSIONS}" != "0" && "${REQUIRE_CYTHON_EXTENSIONS}" != "1" ]]; then
  log_error "[error] REQUIRE_CYTHON_EXTENSIONS must be 0 or 1 (got: ${REQUIRE_CYTHON_EXTENSIONS})"
  exit 1
fi

if [[ -n "${SCORES_PKL_REF_TO_PRED}" && ! -f "${SCORES_PKL_REF_TO_PRED}" ]]; then
  log_error "[error] SCORES_PKL_REF_TO_PRED does not exist: ${SCORES_PKL_REF_TO_PRED}"
  exit 1
fi
if [[ -n "${SCORES_PKL_REF_TO_REF}" && ! -f "${SCORES_PKL_REF_TO_REF}" ]]; then
  log_error "[error] SCORES_PKL_REF_TO_REF does not exist: ${SCORES_PKL_REF_TO_REF}"
  exit 1
fi
if [[ -n "${TEXT_METRICS_V212_DIR}" && ! -d "${TEXT_METRICS_V212_DIR}" ]]; then
  log_error "[error] TEXT_METRICS_V212_DIR does not exist or is not a directory: ${TEXT_METRICS_V212_DIR}"
  exit 1
fi
if [[ "${REF_TO_REF_CACHE_MODE}" != "off" && "${REF_TO_REF_CACHE_MODE}" != "auto" && "${REF_TO_REF_CACHE_MODE}" != "read-only" ]]; then
  log_error "[error] --ref-to-ref-cache-mode must be one of: off, auto, read-only (got: ${REF_TO_REF_CACHE_MODE})"
  exit 1
fi
if [[ "${REF_TO_REF_CACHE_MODE}" != "off" && -z "${REF_TO_REF_CACHE_DIR}" ]]; then
  log_error "[error] --ref-to-ref-cache-dir must not be empty unless --ref-to-ref-cache-mode off is used"
  exit 1
fi
if [[ "${REF_TO_REF_CACHE_WARM_ONLY}" == "1" && "${REF_TO_REF_CACHE_MODE}" == "off" ]]; then
  log_error "[error] --ref-to-ref-cache-warm-only requires --ref-to-ref-cache-mode auto or read-only"
  exit 1
fi
if [[ -n "${SCORE_INDEX_CACHE_FILE}" && ! -f "${SCORE_INDEX_CACHE_FILE}" ]]; then
  log_error "[error] SCORE_INDEX_CACHE_FILE does not exist: ${SCORE_INDEX_CACHE_FILE}"
  exit 1
fi
if [[ -n "${SCORE_INDEX_CACHE_FILE_REF_TO_REF}" && ! -f "${SCORE_INDEX_CACHE_FILE_REF_TO_REF}" ]]; then
  log_error "[error] SCORE_INDEX_CACHE_FILE_REF_TO_REF does not exist: ${SCORE_INDEX_CACHE_FILE_REF_TO_REF}"
  exit 1
fi
if [[ -n "${SCORE_INDEX_CACHE_DIR}" && ! -d "${SCORE_INDEX_CACHE_DIR}" ]]; then
  log_error "[warn] SCORE_INDEX_CACHE_DIR does not exist (will fall back to in-memory index if needed): ${SCORE_INDEX_CACHE_DIR}"
fi

mkdir -p "${PROJECT_DIR}/logs"

if [[ "${PLOT_ONLY}" == "1" ]]; then
  if [[ -z "${SUMMARY_JSON}" ]]; then
    if [[ -z "${OUTPUT_DIR}" ]]; then
      log_error "[error] --plot-only requires --summary-json or --output-dir"
      exit 1
    fi
    SUMMARY_JSON="${OUTPUT_DIR%/}/hough_parameter_sweep_summary.json"
  fi
  if [[ ! -f "${SUMMARY_JSON}" ]]; then
    log_error "[error] summary JSON does not exist: ${SUMMARY_JSON}"
    exit 1
  fi

  PLOT_ARGS=(--summary-json "${SUMMARY_JSON}")
  if [[ -n "${PLOT_OUTPUT_DIR}" ]]; then
    PLOT_ARGS+=(--output-dir "${PLOT_OUTPUT_DIR}")
  fi
  if [[ "${NO_OVERWRITE_SUMMARY}" == "1" ]]; then
    PLOT_ARGS+=(--no-overwrite-summary)
  fi

  log_info "[run] hough_parameter_sweep plot-only mode"
  log_info "[run]   summary_json=${SUMMARY_JSON}"
  if [[ -n "${PLOT_OUTPUT_DIR}" ]]; then
    log_info "[run]   plot_output_dir=${PLOT_OUTPUT_DIR}"
  fi
  "${PYTHON_BIN}" -m outputs.plot_hough_parameter_sweep "${PLOT_ARGS[@]}"
  log_info "[run] Done (plot-only)."
  exit 0
fi

if [[ -z "${RUNFILE_JSON}" ]]; then
  log_error "[error] --runfile-json is required in default mode"
  exit 1
fi
if [[ ! -f "${RUNFILE_JSON}" ]]; then
  log_error "[error] RUNFILE_JSON does not exist: ${RUNFILE_JSON}"
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

if [[ -n "${OUTPUT_SHARD_NAME}" ]]; then
  OUTPUT_SHARD_NAME_SAFE="$(printf '%s' "${OUTPUT_SHARD_NAME}" | tr -cs 'A-Za-z0-9._-' '_')"
  if [[ -z "${OUTPUT_SHARD_NAME_SAFE}" ]]; then
    log_error "[error] --output-shard-name must contain at least one safe character"
    exit 1
  fi
  SWEEP_ROOT_DIR="${SWEEP_DIR}"
  SWEEP_DIR="${SWEEP_ROOT_DIR%/}/shards/${OUTPUT_SHARD_NAME_SAFE}"
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
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}:${SHARED_METRICS_DIR}:${PYTHONPATH:-}"

if [[ "${BUILD_CYTHON_EXTENSIONS}" == "1" ]]; then
  log_info "[cython] build_start python_bin=${PYTHON_BIN}"
  "${PYTHON_BIN}" cython_accel/build.py build_ext --inplace
  log_info "[cython] build_done"
else
  log_info "[cython] build_skipped BUILD_CYTHON_EXTENSIONS=0"
fi

if [[ "${REQUIRE_CYTHON_EXTENSIONS}" == "1" ]]; then
  log_info "[cython] verify_start"
  "${PYTHON_BIN}" - <<'PY'
from cython_accel.optional_filtering import cython_filtering_helpers_available
from cython_accel.optional_line_grouping import cython_line_grouping_available

if not cython_line_grouping_available():
    raise SystemExit("compiled along_lines_core extension is unavailable")
if not cython_filtering_helpers_available():
    raise SystemExit("compiled filter_core extension is unavailable")
print("[cython] verify_ok along_lines=True filtering=True")
PY
  log_info "[cython] verify_done"
else
  log_info "[cython] verify_skipped REQUIRE_CYTHON_EXTENSIONS=0"
fi

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
  --text-metrics-v212-dir "${TEXT_METRICS_V212_DIR}"
  --ref-to-ref-cache-mode "${REF_TO_REF_CACHE_MODE}"
  --ref-to-ref-cache-dir "${REF_TO_REF_CACHE_DIR}"
)

if [[ "${REF_TO_REF_CACHE_WARM_ONLY}" == "1" ]]; then
  PY_ARGS+=(--ref-to-ref-cache-warm-only)
fi

if [[ "${NO_MATRIX_CACHE}" == "1" ]]; then
  PY_ARGS+=(--no-matrix-cache)
else
  PY_ARGS+=(--matrix-cache-dir "${MATRIX_CACHE_DIR}")
fi

if [[ -n "${SCORES_PKL_REF_TO_PRED}" ]]; then
  PY_ARGS+=(--scores-pkl-ref-to-pred "${SCORES_PKL_REF_TO_PRED}")
fi
if [[ -n "${SCORES_PKL_REF_TO_REF}" ]]; then
  PY_ARGS+=(--scores-pkl-ref-to-ref "${SCORES_PKL_REF_TO_REF}")
fi
if [[ -n "${SCORE_INDEX_CACHE_FILE}" ]]; then
  PY_ARGS+=(--score-index-cache-file "${SCORE_INDEX_CACHE_FILE}")
fi
if [[ -n "${SCORE_INDEX_CACHE_FILE_REF_TO_REF}" ]]; then
  PY_ARGS+=(--score-index-cache-file-ref-to-ref "${SCORE_INDEX_CACHE_FILE_REF_TO_REF}")
fi
if [[ -n "${SCORE_INDEX_CACHE_DIR}" ]]; then
  PY_ARGS+=(--score-index-cache-dir "${SCORE_INDEX_CACHE_DIR}")
fi
if [[ "${DISABLE_PKL_MATRIX_SOURCE}" == "1" ]]; then
  PY_ARGS+=(--disable-pkl-matrix-source)
fi
if [[ "${WITH_VISUALS}" == "1" ]]; then
  PY_ARGS+=(--with-visuals)
fi
if [[ "${HIDE_LINE_LABELS}" == "1" ]]; then
  PY_ARGS+=(--hide-line-labels)
fi

if [[ -n "${HOUGH_THRESHOLD_RANGE_START}" ]]; then
  PY_ARGS+=(--hough-threshold-range "${HOUGH_THRESHOLD_RANGE_START}" "${HOUGH_THRESHOLD_RANGE_END}")
fi
if [[ -n "${HOUGH_LINE_LENGTH_RANGE_START}" ]]; then
  PY_ARGS+=(--line-length-range "${HOUGH_LINE_LENGTH_RANGE_START}" "${HOUGH_LINE_LENGTH_RANGE_END}")
fi
if [[ -n "${HOUGH_LINE_GAP_RANGE_START}" ]]; then
  PY_ARGS+=(--line-gap-range "${HOUGH_LINE_GAP_RANGE_START}" "${HOUGH_LINE_GAP_RANGE_END}")
fi
# Seed range forwarding is temporarily disabled with the fixed-seed grid.
# Keeping the old forwarding line here makes restoring seed search a small diff.
# if [[ -n "${HOUGH_SEED_RANGE_START}" ]]; then
#   PY_ARGS+=(--seed-range "${HOUGH_SEED_RANGE_START}" "${HOUGH_SEED_RANGE_END}")
# fi

if [[ -n "${MAX_ITEMS}" ]]; then
  PY_ARGS+=(--max-items "${MAX_ITEMS}")
fi
if [[ -n "${SELECTION_INDEX_RANGE_START}" ]]; then
  PY_ARGS+=(--selection-index-range "${SELECTION_INDEX_RANGE_START}" "${SELECTION_INDEX_RANGE_END}")
fi
for target in "${TARGET_FNAMES[@]}"; do
  PY_ARGS+=(--target-fname "${target}")
done

log_info "[run] hough_parameter_sweep"
log_info "[run]   script_dir=${SCRIPT_DIR}"
log_info "[run]   module_init=${LMOD_INIT}"
log_info "[run]   modulefiles_dir=${MODULEFILES_DIR}"
log_info "[run]   pytorch_module=${PYTORCH_MODULE}"
log_info "[run]   python_bin=${PYTHON_BIN}"
PYTHON_VERSION_TEXT="$("${PYTHON_BIN}" -V 2>&1)"
log_info "[run]   python_version=${PYTHON_VERSION_TEXT}"
log_info "[run]   build_cython_extensions=${BUILD_CYTHON_EXTENSIONS}"
log_info "[run]   require_cython_extensions=${REQUIRE_CYTHON_EXTENSIONS}"
log_info "[run]   runfile_json=${RUNFILE_JSON}"
log_info "[run]   output_dir=${SWEEP_DIR}"
log_info "[run]   window_size=${WINDOW_SIZE}"
log_info "[run]   window_stride=${WINDOW_STRIDE}"
ACTIVE_GRID_LABEL="threshold:${HOUGH_THRESHOLD_RANGE_START:-1}..${HOUGH_THRESHOLD_RANGE_END:-40},line_length:${HOUGH_LINE_LENGTH_RANGE_START:-1}..${HOUGH_LINE_LENGTH_RANGE_END:-50},line_gap:${HOUGH_LINE_GAP_RANGE_START:-1}..${HOUGH_LINE_GAP_RANGE_END:-30},seed:1..1"
log_info "[run]   active_grid=${ACTIVE_GRID_LABEL}"
log_info "[run]   hough_angle=falling_diagonal_only(30..60)"
log_info "[run]   requested_threshold_workers=${WORKERS}"
log_info "[run]   doc_parallel_rule=global_threshold_queue_force40_threshold_workers_when_doc_workers_gt_1"
log_info "[run]   doc_workers=${DOC_WORKERS}"
log_info "[run]   fixed_hough_seed=1"
log_info "[run]   levenshtein_backend=${LEVENSHTEIN_BACKEND}"
if [[ "${NO_MATRIX_CACHE}" == "1" ]]; then
  log_info "[run]   matrix_cache=disabled"
else
  log_info "[run]   matrix_cache_dir=${MATRIX_CACHE_DIR}"
fi
log_info "[run]   scores_pkl_ref_to_pred=${SCORES_PKL_REF_TO_PRED:-None}"
log_info "[run]   scores_pkl_ref_to_ref=${SCORES_PKL_REF_TO_REF:-None}"
log_info "[run]   score_index_cache_file=${SCORE_INDEX_CACHE_FILE:-None}"
log_info "[run]   score_index_cache_file_ref_to_ref=${SCORE_INDEX_CACHE_FILE_REF_TO_REF:-None}"
log_info "[run]   score_index_cache_dir=${SCORE_INDEX_CACHE_DIR:-None}"
log_info "[run]   disable_pkl_matrix_source=${DISABLE_PKL_MATRIX_SOURCE}"
log_info "[run]   text_metrics_v212_dir=${TEXT_METRICS_V212_DIR}"
log_info "[run]   ref_to_ref_cache_mode=${REF_TO_REF_CACHE_MODE}"
log_info "[run]   ref_to_ref_cache_dir=${REF_TO_REF_CACHE_DIR}"
log_info "[run]   ref_to_ref_cache_warm_only=${REF_TO_REF_CACHE_WARM_ONLY}"
log_info "[run]   with_visuals=${WITH_VISUALS}"
log_info "[run]   hide_line_labels=${HIDE_LINE_LABELS}"
if [[ -n "${OUTPUT_SHARD_NAME}" ]]; then
  log_info "[run]   output_shard_name=${OUTPUT_SHARD_NAME_SAFE}"
  log_info "[run]   output_shard_root=${SWEEP_ROOT_DIR}"
fi
if [[ ${#TARGET_FNAMES[@]} -gt 0 ]]; then
  log_info "[run]   target_fnames=${TARGET_FNAMES[*]}"
fi
if [[ -n "${MAX_ITEMS}" ]]; then
  log_info "[run]   max_items=${MAX_ITEMS}"
fi
if [[ -n "${SELECTION_INDEX_RANGE_START}" ]]; then
  log_info "[run]   selection_index_range=${SELECTION_INDEX_RANGE_START}..${SELECTION_INDEX_RANGE_END}"
fi

log_info "[run] Stage 1/1: run_hough_parameter_sweep.py"
"${PYTHON_BIN}" run_hough_parameter_sweep.py "${PY_ARGS[@]}"

log_info "[run] Done. Results written under: ${SWEEP_DIR}"
log_info "[run] Summary JSON: ${SWEEP_DIR}/hough_parameter_sweep_summary.json"
