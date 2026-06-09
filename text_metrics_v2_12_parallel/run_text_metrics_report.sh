#!/usr/bin/env bash
#SBATCH --job-name=text_metrics_report_v2_12_parallel
#SBATCH --account=project_2000539
#SBATCH --partition=medium
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel
#SBATCH -o /scratch/project_2017385/dorian/Churro_copy/logs/text_metrics_report_v2_12_parallel_%j.out
#SBATCH -e /scratch/project_2017385/dorian/Churro_copy/logs/text_metrics_report_v2_12_parallel_%j.err

set -euo pipefail

# Preserve the original CLI args so we can re-submit with matching Slurm resources
# when --workers and allocated ntasks-per-node differ.
ORIGINAL_ARGS=("$@")

if ! command -v module >/dev/null 2>&1 && [[ -f /usr/share/lmod/8.6.17/init/bash ]]; then
  # Load the site Lmod integration when this shell was not initialized with it already.
  # shellcheck disable=SC1091
  source /usr/share/lmod/8.6.17/init/bash
fi

if command -v module >/dev/null 2>&1; then
  module purge
  module use /appl/modulefiles
  module load pytorch
fi

SCRIPT_DIR="/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel"
cd "${SCRIPT_DIR}"

RUNFILE_JSON="${RUNFILE_JSON:-}"
SCORES_PKL_ROOT="${SCORES_PKL_ROOT:-}"
SCORES_PKL_REF_TO_PRED="${SCORES_PKL_REF_TO_PRED:-}"
SCORES_PKL_REF_TO_REF="${SCORES_PKL_REF_TO_REF:-}"
SCORES_PKL_REF_TO_ADJUSTED_PRED="${SCORES_PKL_REF_TO_ADJUSTED_PRED:-}"
PROJECT_ROOT_RESULTS="${PROJECT_ROOT_RESULTS:-/scratch/project_2017385/dorian/Churro_copy/results/text_metrics_results_v2_12_parallel}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-35}"
TARGET_FNAME="${TARGET_FNAME:-}"
MAX_ITEMS="${MAX_ITEMS:-}"
WORKERS="${WORKERS:-1}"
HOUGH_THRESHOLD="${HOUGH_THRESHOLD:-10}"
HOUGH_LINE_LENGTH="${HOUGH_LINE_LENGTH:-8}"
HOUGH_LINE_GAP="${HOUGH_LINE_GAP:-8}"
HOUGH_SEED="${HOUGH_SEED:-0}"
HOUGH_START="${HOUGH_START:-2.2}"
HOUGH_HANDOFF_MODE="${HOUGH_HANDOFF_MODE:-raw_hough_to_true_iou}"
ALIGN_ABS_MIN_LEN="${ALIGN_ABS_MIN_LEN:-6.0}"
ALIGN_MIN_IOU_THRESHOLD="${ALIGN_MIN_IOU_THRESHOLD:-}"
WITH_VISUALS="${WITH_VISUALS:-0}"
REPORT_DEBUG="${REPORT_DEBUG:-0}"
HOUGH_PARAMS_PER_DOCUMENT_JSON="${HOUGH_PARAMS_PER_DOCUMENT_JSON:-}"
HOUGH_PARAMS_SELECTION_MODE="${HOUGH_PARAMS_SELECTION_MODE:-only_json_docs}"
HOUGH_PARAMS_STRICT="${HOUGH_PARAMS_STRICT:-0}"

usage() {
  cat <<'USAGE'
Usage: run_text_metrics_report.sh [options]

Inputs (at least one required):
  --runfile-json <path>                       Path to outputs.json.
  --scores-pkl-root <path>                    Root containing ref_to_pred/ref_to_ref/ref_to_adjusted_pred subdirs.
  --scores-pkl-ref-to-pred <path>             Explicit ref->pred scores.pkl.
  --scores-pkl-ref-to-ref <path>              Explicit ref->ref scores.pkl.
  --scores-pkl-ref-to-adjusted-pred <path>    Explicit ref->adjusted-pred scores.pkl.

Outputs:
  --output-dir <dir>                          Exact output directory for text_metrics_report.py.
                                              If omitted, wrapper creates timestamped run dir under --project-root-results.
  --project-root-results <dir>                Root output directory used when --output-dir is not provided.
  --root-results <dir>                        Alias for --project-root-results.

Parallelism:
  --workers <n>                               Document-level worker processes. Default: 1
                                              Hard fail if greater than available CPUs.

Text-matrix settings:
  --window-size <n>                           Sliding window size in characters. Default: 50
  --window-stride <n>                         Sliding window stride in characters. Default: 35
  --target-fname <name>                       Process only one matching image/file name.
  --max-items <n>                             Process only the first N items.

Visuals:
  --with-visuals                              Enable report visualisations (before/after Hough/filter/reorder).

Hough parameters:
  --hough-threshold <n>                       Probabilistic Hough vote threshold. Default: 10
  --hough-line-length <n>                     Minimum accepted line length. Default: 8
  --hough-line-gap <n>                        Maximum allowed gap when linking line pixels. Default: 8
  --hough-seed <n>                            Base RNG seed (index is added per item). Default: 0
  --hough-start <float>                       Initial adaptive threshold start value. Default: 2.2
  --hough-handoff-mode <mode>                  merged_hough_to_true_iou | raw_hough_to_true_iou.
                                              Default: raw_hough_to_true_iou

True IoU filter parameters:
  --align-abs-min-len <float>                 Minimum line length before v2.12 true-IoU filtering. Default: 6.0
  --align-min-iou-threshold <float>           Optional override in [0,1]. If omitted, script default is used.


Per-document Hough overrides:
  --hough-params-per-document-json <path>      Optional best_params_per_document.json (tuner output).
  --hough-params-selection-mode <mode>         only_json_docs | all_selected_docs. Default: only_json_docs
  --hough-params-strict                        Fail on JSON/selection mismatches instead of fallback.
Debug:
  --debug                                     Write run-level timing telemetry JSON file (report_timings.json).

Other:
  -h, --help                                  Show this help text.
USAGE
}

parse_ntasks_per_node() {
  # SLURM_NTASKS_PER_NODE may look like "124" or "124(x1)".
  local raw="${1:-}"
  raw="${raw%%(*}"
  raw="${raw//[[:space:]]/}"
  if [[ "${raw}" =~ ^[0-9]+$ ]]; then
    echo "${raw}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scores-pkl-root)
      [[ $# -ge 2 ]] || { echo "[error] --scores-pkl-root requires a value" >&2; exit 1; }
      SCORES_PKL_ROOT="$2"
      shift 2
      ;;
    --scores-pkl-ref-to-pred)
      [[ $# -ge 2 ]] || { echo "[error] --scores-pkl-ref-to-pred requires a value" >&2; exit 1; }
      SCORES_PKL_REF_TO_PRED="$2"
      shift 2
      ;;
    --scores-pkl-ref-to-ref)
      [[ $# -ge 2 ]] || { echo "[error] --scores-pkl-ref-to-ref requires a value" >&2; exit 1; }
      SCORES_PKL_REF_TO_REF="$2"
      shift 2
      ;;
    --scores-pkl-ref-to-adjusted-pred)
      [[ $# -ge 2 ]] || { echo "[error] --scores-pkl-ref-to-adjusted-pred requires a value" >&2; exit 1; }
      SCORES_PKL_REF_TO_ADJUSTED_PRED="$2"
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
    --workers)
      [[ $# -ge 2 ]] || { echo "[error] --workers requires a value" >&2; exit 1; }
      WORKERS="$2"
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
    --hough-handoff-mode)
      [[ $# -ge 2 ]] || { echo "[error] --hough-handoff-mode requires a value" >&2; exit 1; }
      HOUGH_HANDOFF_MODE="$2"
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
    --hough-params-per-document-json)
      [[ $# -ge 2 ]] || { echo "[error] --hough-params-per-document-json requires a value" >&2; exit 1; }
      HOUGH_PARAMS_PER_DOCUMENT_JSON="$2"
      shift 2
      ;;
    --hough-params-selection-mode)
      [[ $# -ge 2 ]] || { echo "[error] --hough-params-selection-mode requires a value" >&2; exit 1; }
      HOUGH_PARAMS_SELECTION_MODE="$2"
      shift 2
      ;;
    --hough-params-strict)
      HOUGH_PARAMS_STRICT="1"
      shift
      ;;
    --with-visuals)
      WITH_VISUALS="1"
      shift
      ;;
    --debug)
      REPORT_DEBUG="1"
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

if [[ -z "${RUNFILE_JSON}" && -z "${SCORES_PKL_ROOT}" && -z "${SCORES_PKL_REF_TO_PRED}" && -z "${SCORES_PKL_REF_TO_REF}" && -z "${SCORES_PKL_REF_TO_ADJUSTED_PRED}" ]]; then
  echo "[error] Provide at least one input source: --runfile-json and/or --scores-pkl-* options" >&2
  exit 1
fi

if [[ -n "${RUNFILE_JSON}" && ! -f "${RUNFILE_JSON}" ]]; then
  echo "[error] RUNFILE_JSON does not exist: ${RUNFILE_JSON}" >&2
  exit 1
fi
if [[ -n "${SCORES_PKL_ROOT}" && ! -d "${SCORES_PKL_ROOT}" ]]; then
  echo "[error] SCORES_PKL_ROOT does not exist: ${SCORES_PKL_ROOT}" >&2
  exit 1
fi
if [[ -n "${SCORES_PKL_REF_TO_PRED}" && ! -f "${SCORES_PKL_REF_TO_PRED}" ]]; then
  echo "[error] SCORES_PKL_REF_TO_PRED does not exist: ${SCORES_PKL_REF_TO_PRED}" >&2
  exit 1
fi
if [[ -n "${SCORES_PKL_REF_TO_REF}" && ! -f "${SCORES_PKL_REF_TO_REF}" ]]; then
  echo "[error] SCORES_PKL_REF_TO_REF does not exist: ${SCORES_PKL_REF_TO_REF}" >&2
  exit 1
fi
if [[ -n "${SCORES_PKL_REF_TO_ADJUSTED_PRED}" && ! -f "${SCORES_PKL_REF_TO_ADJUSTED_PRED}" ]]; then
  echo "[error] SCORES_PKL_REF_TO_ADJUSTED_PRED does not exist: ${SCORES_PKL_REF_TO_ADJUSTED_PRED}" >&2
  exit 1
fi

if [[ "${HOUGH_HANDOFF_MODE}" != "merged_hough_to_true_iou" && "${HOUGH_HANDOFF_MODE}" != "raw_hough_to_true_iou" ]]; then
  echo "[error] --hough-handoff-mode must be one of: merged_hough_to_true_iou, raw_hough_to_true_iou (got: ${HOUGH_HANDOFF_MODE})" >&2
  exit 1
fi
if [[ -n "${HOUGH_PARAMS_PER_DOCUMENT_JSON}" && ! -f "${HOUGH_PARAMS_PER_DOCUMENT_JSON}" ]]; then
  echo "[error] HOUGH_PARAMS_PER_DOCUMENT_JSON does not exist: ${HOUGH_PARAMS_PER_DOCUMENT_JSON}" >&2
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
if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --workers must be a positive integer (got: ${WORKERS})" >&2
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
if [[ "${REPORT_DEBUG}" != "0" && "${REPORT_DEBUG}" != "1" ]]; then
  echo "[error] REPORT_DEBUG must be 0 or 1 (got: ${REPORT_DEBUG})" >&2
  exit 1
fi
if [[ "${HOUGH_PARAMS_SELECTION_MODE}" != "only_json_docs" && "${HOUGH_PARAMS_SELECTION_MODE}" != "all_selected_docs" ]]; then
  echo "[error] --hough-params-selection-mode must be one of: only_json_docs, all_selected_docs (got: ${HOUGH_PARAMS_SELECTION_MODE})" >&2
  exit 1
fi
if [[ "${HOUGH_PARAMS_STRICT}" != "0" && "${HOUGH_PARAMS_STRICT}" != "1" ]]; then
  echo "[error] HOUGH_PARAMS_STRICT must be 0 or 1 (got: ${HOUGH_PARAMS_STRICT})" >&2
  exit 1
fi
if [[ "${HOUGH_PARAMS_STRICT}" == "1" && -z "${HOUGH_PARAMS_PER_DOCUMENT_JSON}" ]]; then
  echo "[error] --hough-params-strict requires --hough-params-per-document-json" >&2
  exit 1
fi
# Keep Slurm allocation aligned with --workers.
# Slurm directives are static at submit time, so we re-submit if needed.
ALLOCATED_NTASKS_PER_NODE=""
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ALLOCATED_NTASKS_PER_NODE="$(parse_ntasks_per_node "${SLURM_NTASKS_PER_NODE:-}")"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[submit] No active Slurm job detected. Submitting with --ntasks-per-node=${WORKERS}."
  sbatch --export=ALL,TMR_RESUBMITTED=1 --ntasks-per-node="${WORKERS}" "$0" "${ORIGINAL_ARGS[@]}"
  exit 0
fi

if [[ -n "${ALLOCATED_NTASKS_PER_NODE}" && "${ALLOCATED_NTASKS_PER_NODE}" != "${WORKERS}" ]]; then
  if [[ "${TMR_RESUBMITTED:-0}" != "1" ]]; then
    echo "[submit] Current allocation ntasks-per-node=${ALLOCATED_NTASKS_PER_NODE} does not match --workers=${WORKERS}."
    echo "[submit] Re-submitting this job with --ntasks-per-node=${WORKERS}."
    sbatch --export=ALL,TMR_RESUBMITTED=1 --ntasks-per-node="${WORKERS}" "$0" "${ORIGINAL_ARGS[@]}"
    exit 0
  fi
  echo "[error] Allocation mismatch persists after re-submit: ntasks-per-node=${ALLOCATED_NTASKS_PER_NODE}, --workers=${WORKERS}." >&2
  echo "[error] Submit manually with: sbatch --ntasks-per-node=${WORKERS} $0 ..." >&2
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

mkdir -p "${REPORT_DIR}" "/scratch/project_2017385/dorian/Churro_copy/logs"
if [[ -n "${RUN_DIR}" ]]; then
  mkdir -p "${RUN_DIR}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export PYTHONUNBUFFERED=1

PY_ARGS=(
  --output-dir "${REPORT_DIR}"
  --window-size "${WINDOW_SIZE}"
  --window-stride "${WINDOW_STRIDE}"
  --workers "${WORKERS}"
  --hough-threshold "${HOUGH_THRESHOLD}"
  --hough-line-length "${HOUGH_LINE_LENGTH}"
  --hough-line-gap "${HOUGH_LINE_GAP}"
  --hough-seed "${HOUGH_SEED}"
  --hough-start "${HOUGH_START}"
  --hough-handoff-mode "${HOUGH_HANDOFF_MODE}"
  --align-abs-min-len "${ALIGN_ABS_MIN_LEN}"
)
if [[ -n "${SCORES_PKL_REF_TO_PRED}" ]]; then
  PY_ARGS+=(--scores-pkl-ref-to-pred "${SCORES_PKL_REF_TO_PRED}")
fi
if [[ -n "${SCORES_PKL_REF_TO_REF}" ]]; then
  PY_ARGS+=(--scores-pkl-ref-to-ref "${SCORES_PKL_REF_TO_REF}")
fi
if [[ -n "${SCORES_PKL_REF_TO_ADJUSTED_PRED}" ]]; then
  PY_ARGS+=(--scores-pkl-ref-to-adjusted-pred "${SCORES_PKL_REF_TO_ADJUSTED_PRED}")
fi
if [[ -n "${SCORES_PKL_ROOT}" ]]; then
  PY_ARGS+=(--scores-pkl-root "${SCORES_PKL_ROOT}")
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
if [[ -n "${HOUGH_PARAMS_PER_DOCUMENT_JSON}" ]]; then
  PY_ARGS+=(--hough-params-per-document-json "${HOUGH_PARAMS_PER_DOCUMENT_JSON}")
  PY_ARGS+=(--hough-params-selection-mode "${HOUGH_PARAMS_SELECTION_MODE}")
fi
if [[ "${HOUGH_PARAMS_STRICT}" == "1" ]]; then
  PY_ARGS+=(--hough-params-strict)
fi
if [[ "${WITH_VISUALS}" == "1" ]]; then
  PY_ARGS+=(--with-visuals)
fi
if [[ "${REPORT_DEBUG}" == "1" ]]; then
  PY_ARGS+=(--debug)
fi

echo "[run] text_metrics_report_v2_12_parallel"
echo "[run]   script_dir=${SCRIPT_DIR}"
echo "[run]   workers=${WORKERS}"
if [[ -n "${RUNFILE_JSON}" ]]; then
  echo "[run]   runfile_json=${RUNFILE_JSON}"
fi
if [[ -n "${SCORES_PKL_ROOT}" ]]; then
  echo "[run]   scores_pkl_root=${SCORES_PKL_ROOT}"
fi
if [[ -n "${SCORES_PKL_REF_TO_PRED}" ]]; then
  echo "[run]   scores_pkl_ref_to_pred=${SCORES_PKL_REF_TO_PRED}"
fi
if [[ -n "${SCORES_PKL_REF_TO_REF}" ]]; then
  echo "[run]   scores_pkl_ref_to_ref=${SCORES_PKL_REF_TO_REF}"
fi
if [[ -n "${SCORES_PKL_REF_TO_ADJUSTED_PRED}" ]]; then
  echo "[run]   scores_pkl_ref_to_adjusted_pred=${SCORES_PKL_REF_TO_ADJUSTED_PRED}"
fi
if [[ -n "${HOUGH_PARAMS_PER_DOCUMENT_JSON}" ]]; then
  echo "[run]   hough_params_per_document_json=${HOUGH_PARAMS_PER_DOCUMENT_JSON}"
  echo "[run]   hough_params_selection_mode=${HOUGH_PARAMS_SELECTION_MODE}"
  echo "[run]   hough_params_strict=${HOUGH_PARAMS_STRICT}"
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  echo "[run]   output_dir=${OUTPUT_DIR}"
else
  echo "[run]   project_root_results=${PROJECT_ROOT_RESULTS}"
  echo "[run]   run_base_dir=${RUN_BASE_DIR}"
  echo "[run]   run_timestamp=${RUN_TIMESTAMP}"
  echo "[run]   run_dir=${RUN_DIR}"
fi
echo "[run]   hough_handoff_mode=${HOUGH_HANDOFF_MODE}"
echo "[run]   debug=${REPORT_DEBUG}"

echo "[run] Stage 1/1: text_metrics_report.py"
python3 text_metrics_report.py "${PY_ARGS[@]}"

echo "[run] Done. Results written under: ${REPORT_DIR}"
