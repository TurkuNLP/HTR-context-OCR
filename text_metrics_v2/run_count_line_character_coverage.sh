#!/usr/bin/env bash
#SBATCH --job-name=count_line_character_coverage
#SBATCH --account=project_2000539
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=48G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2
#SBATCH -o logs/count_line_character_coverage_%j.out
#SBATCH -e logs/count_line_character_coverage_%j.err

set -euo pipefail

# Optional environment setup for CSC/HPC. Safe to ignore on machines without `module`.
if command -v module >/dev/null 2>&1; then
  module purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

SCRIPT_DIR="/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2"
cd "${SCRIPT_DIR}"

TEXT_Y="${TEXT_Y:-}"
TEXT_Y_PATH="${TEXT_Y_PATH:-}"
TEXT_X="${TEXT_X:-}"
TEXT_X_PATH="${TEXT_X_PATH:-}"
RUNFILE_JSON="${RUNFILE_JSON:-}"
TARGET_FNAME="${TARGET_FNAME:-}"
LINE_ENDPOINTS_JSON="${LINE_ENDPOINTS_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PROJECT_ROOT_RESULTS="${PROJECT_ROOT_RESULTS:-/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2/results/count_line_character_coverage_results}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
WINDOW_STRIDE="${WINDOW_STRIDE:-50}"
HOUGH_THRESHOLD="${HOUGH_THRESHOLD:-26}"
HOUGH_LINE_LENGTH="${HOUGH_LINE_LENGTH:-10}"
HOUGH_LINE_GAP="${HOUGH_LINE_GAP:-15}"
HOUGH_SEED="${HOUGH_SEED:-0}"
HOUGH_START="${HOUGH_START:-2.6}"
ALIGN_ABS_MIN_LEN="${ALIGN_ABS_MIN_LEN:-8.0}"
ALIGN_MIN_IOU_THRESHOLD="${ALIGN_MIN_IOU_THRESHOLD:-0.035}"
STRICT_LINES="${STRICT_LINES:-0}"
VISUALIZE="${VISUALIZE:-0}"
VISUAL_OUTPUT="${VISUAL_OUTPUT:-}"
VISUAL_TITLE="${VISUAL_TITLE:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
PRINT_ARRAYS="${PRINT_ARRAYS:-0}"

usage() {
  cat <<'USAGE'
Usage: run_count_line_character_coverage.sh [options]

Mode A (direct):
  --line-endpoints-json <path>      JSON with filtered line endpoints.
  Provide exactly one of:
    --text-y <text>
    --text-y-path <path>
  and exactly one of:
    --text-x <text>
    --text-x-path <path>

Mode B (from outputs.json for one example):
  --runfile-json <path>             Path to outputs.json.
  --target-fname <name>             One item to run (exact or basename match).
  --line-endpoints-json <path>      Optional; if omitted endpoints are auto-derived
                                    with Hough + v2.1_true_IoU filtering.

Optional output routing:
  --output-dir <dir>                Exact output directory for this run.
                                    If omitted, wrapper creates timestamped run dir under --project-root-results.
  --project-root-results <dir>      Root output directory when --output-dir is not set.
  --root-results <dir>              Alias for --project-root-results.

Coverage parameters:
  --window-size <n>                 Window size. Default: 100
  --window-stride <n>               Window stride. Default: 50
  --strict-lines                    Fail on malformed line endpoints.

Endpoint auto-derivation parameters (Mode B when endpoints JSON omitted):
  --hough-threshold <n>
  --hough-line-length <n>
  --hough-line-gap <n>
  --hough-seed <n>
  --hough-start <float>
  --align-abs-min-len <float>
  --align-min-iou-threshold <float>

Visual/output switches:
  --visualize                       Legacy compatibility flag (no-op in v2).
  --visual-output <path>            Legacy compatibility flag (no-op in v2).
  --visual-title <text>             Legacy compatibility flag (no-op in v2).
  --output-json <path>              Optional explicit output JSON path.
  --print-arrays                    Print y_counts and x_counts arrays.

Other:
  -h, --help                        Show this help text.

Environment variable overrides are supported for all options:
  TEXT_Y, TEXT_Y_PATH, TEXT_X, TEXT_X_PATH,
  RUNFILE_JSON, TARGET_FNAME, LINE_ENDPOINTS_JSON,
  OUTPUT_DIR, PROJECT_ROOT_RESULTS,
  WINDOW_SIZE, WINDOW_STRIDE,
  HOUGH_THRESHOLD, HOUGH_LINE_LENGTH, HOUGH_LINE_GAP, HOUGH_SEED, HOUGH_START,
  ALIGN_ABS_MIN_LEN, ALIGN_MIN_IOU_THRESHOLD,
  STRICT_LINES, VISUALIZE, VISUAL_OUTPUT, VISUAL_TITLE, OUTPUT_JSON, PRINT_ARRAYS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --text-y)
      [[ $# -ge 2 ]] || { echo "[error] --text-y requires a value" >&2; exit 1; }
      TEXT_Y="$2"
      shift 2
      ;;
    --text-y-path)
      [[ $# -ge 2 ]] || { echo "[error] --text-y-path requires a value" >&2; exit 1; }
      TEXT_Y_PATH="$2"
      shift 2
      ;;
    --text-x)
      [[ $# -ge 2 ]] || { echo "[error] --text-x requires a value" >&2; exit 1; }
      TEXT_X="$2"
      shift 2
      ;;
    --text-x-path)
      [[ $# -ge 2 ]] || { echo "[error] --text-x-path requires a value" >&2; exit 1; }
      TEXT_X_PATH="$2"
      shift 2
      ;;
    --runfile-json)
      [[ $# -ge 2 ]] || { echo "[error] --runfile-json requires a value" >&2; exit 1; }
      RUNFILE_JSON="$2"
      shift 2
      ;;
    --target-fname)
      [[ $# -ge 2 ]] || { echo "[error] --target-fname requires a value" >&2; exit 1; }
      TARGET_FNAME="$2"
      shift 2
      ;;
    --line-endpoints-json)
      [[ $# -ge 2 ]] || { echo "[error] --line-endpoints-json requires a value" >&2; exit 1; }
      LINE_ENDPOINTS_JSON="$2"
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
    --strict-lines)
      STRICT_LINES="1"
      shift
      ;;
    --visualize)
      VISUALIZE="1"
      shift
      ;;
    --visual-output)
      [[ $# -ge 2 ]] || { echo "[error] --visual-output requires a value" >&2; exit 1; }
      VISUAL_OUTPUT="$2"
      shift 2
      ;;
    --visual-title)
      [[ $# -ge 2 ]] || { echo "[error] --visual-title requires a value" >&2; exit 1; }
      VISUAL_TITLE="$2"
      shift 2
      ;;
    --output-json)
      [[ $# -ge 2 ]] || { echo "[error] --output-json requires a value" >&2; exit 1; }
      OUTPUT_JSON="$2"
      shift 2
      ;;
    --print-arrays)
      PRINT_ARRAYS="1"
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
if [[ "${STRICT_LINES}" != "0" && "${STRICT_LINES}" != "1" ]]; then
  echo "[error] STRICT_LINES must be 0 or 1 (got: ${STRICT_LINES})" >&2
  exit 1
fi
if [[ "${VISUALIZE}" != "0" && "${VISUALIZE}" != "1" ]]; then
  echo "[error] VISUALIZE must be 0 or 1 (got: ${VISUALIZE})" >&2
  exit 1
fi
if [[ "${PRINT_ARRAYS}" != "0" && "${PRINT_ARRAYS}" != "1" ]]; then
  echo "[error] PRINT_ARRAYS must be 0 or 1 (got: ${PRINT_ARRAYS})" >&2
  exit 1
fi

has_runfile_mode=0
if [[ -n "${RUNFILE_JSON}" ]]; then
  has_runfile_mode=1
fi

if [[ "${has_runfile_mode}" == "1" ]]; then
  MODE="runfile"
  if [[ ! -f "${RUNFILE_JSON}" ]]; then
    echo "[error] RUNFILE_JSON does not exist: ${RUNFILE_JSON}" >&2
    exit 1
  fi
  if [[ -z "${TARGET_FNAME}" ]]; then
    echo "[error] --target-fname is required when --runfile-json is used" >&2
    exit 1
  fi
  if [[ -n "${TEXT_Y}" || -n "${TEXT_Y_PATH}" || -n "${TEXT_X}" || -n "${TEXT_X_PATH}" ]]; then
    echo "[error] Do not combine runfile mode with direct text arguments" >&2
    exit 1
  fi
  if [[ -n "${LINE_ENDPOINTS_JSON}" && ! -f "${LINE_ENDPOINTS_JSON}" ]]; then
    echo "[error] LINE_ENDPOINTS_JSON does not exist: ${LINE_ENDPOINTS_JSON}" >&2
    exit 1
  fi
else
  MODE="direct"

  if [[ -z "${LINE_ENDPOINTS_JSON}" ]]; then
    echo "[error] In direct mode, --line-endpoints-json is required" >&2
    exit 1
  fi
  if [[ ! -f "${LINE_ENDPOINTS_JSON}" ]]; then
    echo "[error] LINE_ENDPOINTS_JSON does not exist: ${LINE_ENDPOINTS_JSON}" >&2
    exit 1
  fi

  if [[ -n "${TEXT_Y}" && -n "${TEXT_Y_PATH}" ]]; then
    echo "[error] Provide only one of --text-y or --text-y-path" >&2
    exit 1
  fi
  if [[ -z "${TEXT_Y}" && -z "${TEXT_Y_PATH}" ]]; then
    echo "[error] Provide one of --text-y or --text-y-path" >&2
    exit 1
  fi
  if [[ -n "${TEXT_X}" && -n "${TEXT_X_PATH}" ]]; then
    echo "[error] Provide only one of --text-x or --text-x-path" >&2
    exit 1
  fi
  if [[ -z "${TEXT_X}" && -z "${TEXT_X_PATH}" ]]; then
    echo "[error] Provide one of --text-x or --text-x-path" >&2
    exit 1
  fi
  if [[ -n "${TEXT_Y_PATH}" && ! -f "${TEXT_Y_PATH}" ]]; then
    echo "[error] TEXT_Y_PATH does not exist: ${TEXT_Y_PATH}" >&2
    exit 1
  fi
  if [[ -n "${TEXT_X_PATH}" && ! -f "${TEXT_X_PATH}" ]]; then
    echo "[error] TEXT_X_PATH does not exist: ${TEXT_X_PATH}" >&2
    exit 1
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 not found in PATH" >&2
  exit 1
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  RUN_DIR="${OUTPUT_DIR}"
  RUN_BASE_DIR="$(dirname "${OUTPUT_DIR}")"
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
fi

mkdir -p "${RUN_DIR}" "${SCRIPT_DIR}/logs"

# If caller did not provide explicit outputs, place them in run dir.
if [[ -z "${OUTPUT_JSON}" ]]; then
  OUTPUT_JSON="${RUN_DIR}/line_character_coverage.json"
fi
if [[ "${VISUALIZE}" == "1" && -z "${VISUAL_OUTPUT}" ]]; then
  VISUAL_OUTPUT="${RUN_DIR}/line_character_coverage.png"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export PYTHONUNBUFFERED=1

PY_ARGS=(
  --window-size "${WINDOW_SIZE}"
  --window-stride "${WINDOW_STRIDE}"
  --output-json "${OUTPUT_JSON}"
)

if [[ "${MODE}" == "runfile" ]]; then
  PY_ARGS+=(
    --runfile-json "${RUNFILE_JSON}"
    --target-fname "${TARGET_FNAME}"
    --hough-threshold "${HOUGH_THRESHOLD}"
    --hough-line-length "${HOUGH_LINE_LENGTH}"
    --hough-line-gap "${HOUGH_LINE_GAP}"
    --hough-seed "${HOUGH_SEED}"
    --hough-start "${HOUGH_START}"
    --align-abs-min-len "${ALIGN_ABS_MIN_LEN}"
    --align-min-iou-threshold "${ALIGN_MIN_IOU_THRESHOLD}"
  )
  if [[ -n "${LINE_ENDPOINTS_JSON}" ]]; then
    PY_ARGS+=(--line-endpoints-json "${LINE_ENDPOINTS_JSON}")
  fi
else
  PY_ARGS+=(--line-endpoints-json "${LINE_ENDPOINTS_JSON}")
  if [[ -n "${TEXT_Y}" ]]; then
    PY_ARGS+=(--text-y "${TEXT_Y}")
  else
    PY_ARGS+=(--text-y-path "${TEXT_Y_PATH}")
  fi
  if [[ -n "${TEXT_X}" ]]; then
    PY_ARGS+=(--text-x "${TEXT_X}")
  else
    PY_ARGS+=(--text-x-path "${TEXT_X_PATH}")
  fi
fi

if [[ "${STRICT_LINES}" == "1" ]]; then
  PY_ARGS+=(--strict-lines)
fi
if [[ "${VISUALIZE}" == "1" ]]; then
  PY_ARGS+=(--visualize --visual-output "${VISUAL_OUTPUT}")
fi
if [[ -n "${VISUAL_TITLE}" ]]; then
  PY_ARGS+=(--visual-title "${VISUAL_TITLE}")
fi
if [[ "${PRINT_ARRAYS}" == "1" ]]; then
  PY_ARGS+=(--print-arrays)
fi

echo "[run] count_line_character_coverage"
echo "[run]   script_dir=${SCRIPT_DIR}"
echo "[run]   mode=${MODE}"
echo "[run]   run_base_dir=${RUN_BASE_DIR}"
echo "[run]   run_timestamp=${RUN_TIMESTAMP}"
echo "[run]   run_dir=${RUN_DIR}"
if [[ "${MODE}" == "runfile" ]]; then
  echo "[run]   runfile_json=${RUNFILE_JSON}"
  echo "[run]   target_fname=${TARGET_FNAME}"
  if [[ -n "${LINE_ENDPOINTS_JSON}" ]]; then
    echo "[run]   line_endpoints_json=${LINE_ENDPOINTS_JSON}"
  else
    echo "[run]   line_endpoints_json=<auto-derived>"
  fi
  echo "[run]   hough_threshold=${HOUGH_THRESHOLD}"
  echo "[run]   hough_line_length=${HOUGH_LINE_LENGTH}"
  echo "[run]   hough_line_gap=${HOUGH_LINE_GAP}"
  echo "[run]   hough_seed=${HOUGH_SEED}"
  echo "[run]   hough_start=${HOUGH_START}"
  echo "[run]   align_abs_min_len=${ALIGN_ABS_MIN_LEN}"
  echo "[run]   align_min_iou_threshold=${ALIGN_MIN_IOU_THRESHOLD}"
else
  if [[ -n "${TEXT_Y_PATH}" ]]; then
    echo "[run]   text_y_path=${TEXT_Y_PATH}"
  else
    echo "[run]   text_y=<inline:${#TEXT_Y} chars>"
  fi
  if [[ -n "${TEXT_X_PATH}" ]]; then
    echo "[run]   text_x_path=${TEXT_X_PATH}"
  else
    echo "[run]   text_x=<inline:${#TEXT_X} chars>"
  fi
  echo "[run]   line_endpoints_json=${LINE_ENDPOINTS_JSON}"
fi
echo "[run]   window_size=${WINDOW_SIZE}"
echo "[run]   window_stride=${WINDOW_STRIDE}"
echo "[run]   strict_lines=${STRICT_LINES}"
echo "[run]   visualize=${VISUALIZE}"
if [[ "${VISUALIZE}" == "1" ]]; then
  echo "[run]   note=visualize flags are compatibility no-ops in text_metrics_v2/count_line_character_coverage.py"
  echo "[run]   visual_output=${VISUAL_OUTPUT}"
fi
if [[ -n "${VISUAL_TITLE}" ]]; then
  echo "[run]   visual_title=${VISUAL_TITLE}"
fi
echo "[run]   output_json=${OUTPUT_JSON}"
echo "[run]   print_arrays=${PRINT_ARRAYS}"

echo "[run] Stage 1/1: count_line_character_coverage.py"
python3 count_line_character_coverage.py "${PY_ARGS[@]}"

echo "[run] Done. Results written under: ${RUN_DIR}"
