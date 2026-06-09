#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNFILE_JSON=""
OUTPUT_DIR=""
MAX_ITEMS="600"
SHARD_COUNT="20"
DOCS_PER_SHARD="30"

WINDOW_SIZE="50"
WINDOW_STRIDE="35"
NO_MATRIX_CACHE="1"
MATRIX_CACHE_DIR="${SCRIPT_DIR}/_matrix_cache"
SCORES_PKL_REF_TO_PRED="${PROJECT_DIR}/results/compares_churro_dev/ref_to_pred/scores_reference_prediction_ws50_st35.pkl"
SCORES_PKL_REF_TO_REF="${PROJECT_DIR}/results/compares_churro_dev/ref_to_ref/scores_reference_self_ws50_st35.pkl"
SCORE_INDEX_CACHE_FILE=""
SCORE_INDEX_CACHE_FILE_REF_TO_REF=""
SCORE_INDEX_CACHE_DIR="${PROJECT_DIR}/text_metrics_v2_1_parallel/.score_index_cache"
DISABLE_PKL_MATRIX_SOURCE="0"
TEXT_METRICS_V212_DIR="${PROJECT_DIR}/text_metrics_v2_12_parallel"
REF_TO_REF_CACHE_MODE="auto"
REF_TO_REF_CACHE_DIR="${PROJECT_DIR}/results/tuner_parallel_v2_2_cache/ref_to_ref_threshold_pack_cache_v2"

LEVENSHTEIN_BACKEND="c"
WORKERS="25"
DOC_WORKERS="5"
HOUGH_THRESHOLD_START="11"
HOUGH_THRESHOLD_END="35"
LINE_LENGTH_START="5"
LINE_LENGTH_END="35"
LINE_GAP_START="0"
LINE_GAP_END="15"
ALIGN_MIN_IOU_THRESHOLD="0.035"
MINIMUM_SCORE_FLOOR="20.0"
SCORE_FLOOR_METHOD="mean_plus_standard_deviation"
MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER="0.0"
MEDIAN_ABSOLUTE_DEVIATION_BACKEND="manual_numpy"
NEAR_PEAK_RATIO="0.70"
NEAR_PEAK_MARGIN=""
MINIMUM_COMPONENT_CELLS="2"
MINIMUM_COMPONENT_ROWS="1"
MINIMUM_COMPONENT_COLUMNS="1"
CONNECTED_COMPONENT_BACKEND="cython"
REGION_DILATION_RADIUS="1"
FINAL_HOUGH_INPUT_MODE="roi"
ADAPTIVE_BUDGET_MASK="strong_match"
MINIMUM_ACTIVE_CELLS="0"
MINIMUM_ACTIVE_ROWS="2"
MINIMUM_ACTIVE_COLUMNS="2"
MINIMUM_X_SPAN="2"
MINIMUM_Y_SPAN="2"
MAXIMUM_ACTIVE_FRACTION="1.0"
MINIMUM_MATRIX_ROWS="4"
MINIMUM_MATRIX_COLUMNS="4"
MIN_SURVIVING_LINE_NLS=""
MIN_SURVIVING_LINE_NLS_WAS_SET="0"
SELECTION_OBJECTIVE="strict_quality"

ACCOUNT="project_2005072"
PARTITION="medium"
TIME_LIMIT="36:00:00"
CPUS_PER_TASK="128"
MEMORY="64G"
FINAL_VISUAL_CPUS_PER_TASK="8"
FINAL_VISUAL_MEMORY="64G"
COMBINATION_BUNDLE_SCOPE="none"
COMBINATION_BUNDLE_SCOPE_WAS_SET="0"
INCLUDE_CANDIDATE_LINES="1"
WITH_VISUALS="0"
HIDE_LINE_LABELS="0"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'USAGE'
Usage: run_hough_parameter_sweep_20nodes_10docs_each.sh [options]

Required:
  --runfile-json <path>                       Path to outputs.json
  --output-dir <dir>                          Shared top-level output directory

Dynamic scheduling:
  --shard-count <n>                           Number of identical sbatch workers to submit (default: 20)
  --max-items <n>                             Total selected-document cap in the shared pool (default: 600)
  --docs-per-shard <n>                        Compatibility only; ignored by dynamic scheduling

How dynamic scheduling works:
  The launcher creates <output-dir>/document_pool with document id/name files.
  Every sbatch worker reads from that same pool.  A worker keeps up to its
  per-node document capacity active.  As soon as one document finishes inside
  that worker, the worker claims one more free document immediately.

Default grid:
  --hough-threshold-range 11 35
  --line-length-range 5 35
  --line-gap-range 0 15

Default per-node parallelism:
  The node capacity is computed from --cpus-per-task and threshold_count.
  Example: 128 cores and threshold 12..35 gives floor(128 / 24) = 5 documents.
  --doc-workers is treated as an upper cap and is lowered automatically when it
  exceeds the computed node capacity.

Common options:
  --window-size <n>                           Default: 50
  --window-stride <n>                         Default: 35
  --workers <n>                               Compatibility arg; dynamic workers use active threshold_count
  --doc-workers <n>                           Max active documents per worker before CPU-cap adjustment
  --hough-threshold-range <start> <end>       Inclusive threshold range
  --line-length-range <start> <end>           Inclusive line_length range
  --line-gap-range <start> <end>              Inclusive line_gap range
  --minimum-score-floor <float>              Minimum score used by the Median Absolute Deviation floor method
  --score-floor-method <name>                 mean_plus_standard_deviation|median_plus_scaled_median_absolute_deviation
  --median-absolute-deviation-multiplier <f>  Median Absolute Deviation multiplier when that floor method is active
  --median-absolute-deviation-backend <name>  manual_numpy|scipy
  --connected-component-backend <name>        cython|scipy|python, default: cython
  --near-peak-ratio <float>                   Keep cells near the best score in their row or column
  --final-hough-input-mode <name>             roi|roi_and_score_floor|roi_or_score_floor
  --adaptive-budget-mask <name>               final_hough_input|region_of_interest|strong_match|component_region|score_floor
  --maximum-active-fraction <float>           Optional fixed density gate; use 1.0 to leave adaptive budget in charge
  --minimum-matrix-rows <n>                   Reject matrices with fewer reference-window rows
  --minimum-matrix-columns <n>                Reject matrices with fewer prediction-window columns
  --min-surviving-line-nls <float>            Optional final-line text-quality filter in [0, 1].
                                               Final ref_to_pred lines below this line-level
                                               normalized Levenshtein value are removed after
                                               geometry filtering. Example: 0.50
  --selection-objective <name>                 strict_quality|alignment_evidence|non_hallucination_weighted.
                                               strict_quality keeps the original tuning_score winner.
                                               alignment_evidence chooses matrix-supported geometry.
                                               non_hallucination_weighted doubles the
                                               non-hallucination weight inside the harmonic score.

Matrix/cache options:
  --scores-pkl-ref-to-pred <path>             Ref-to-pred score stream
  --scores-pkl-ref-to-ref <path>              Ref-to-ref score stream
  --score-index-cache-file <path>             Explicit ref-to-pred index cache
  --score-index-cache-file-ref-to-ref <path>  Explicit ref-to-ref index cache
  --score-index-cache-dir <dir>               Score index cache directory
  --matrix-cache-dir <dir>                    Matrix npz cache directory
  --use-matrix-cache                          Enable matrix npz cache writes/reads
  --disable-pkl-matrix-source                 Disable read-only pkl matrix source
  --text-metrics-v212-dir <dir>               Optional external v2.12 metric tree for audits/equivalence checks
  --ref-to-ref-cache-mode <off|auto|read-only>
  --ref-to-ref-cache-dir <dir>

Combination bundles:
  --combination-bundle-scope <scope>          none|winner-only|all|valid-only|invalid-only
                                                Default without --with-visuals: none
                                                Default with --with-visuals: winner-only
  --no-candidate-lines                        Compatibility flag; lean pklstream bundles do not store candidate geometry
  --with-visuals                              Submit one final visualization job after all workers finish
  --hide-line-labels                          Hide raw/final line labels in the final stitched panels

Slurm options:
  --account <name>                            Default: project_2005072
  --partition <name>                          Default: medium
  --time <HH:MM:SS>                           Default: 36:00:00
  --cpus-per-task <n>                         Default: 128
  --mem <amount>                              Default: 64G
  --final-visual-cpus-per-task <n>            CPUs for the final visualisation job (default: 8)
  --final-visual-mem <amount>                 Memory for the final visualisation job (default: 64G)

Other:
  -h, --help                                  Show this help text
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runfile-json)
      RUNFILE_JSON="${2:?--runfile-json requires a value}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a value}"
      shift 2
      ;;
    --max-items)
      MAX_ITEMS="${2:?--max-items requires a value}"
      shift 2
      ;;
    --shard-count)
      SHARD_COUNT="${2:?--shard-count requires a value}"
      shift 2
      ;;
    --docs-per-shard)
      DOCS_PER_SHARD="${2:?--docs-per-shard requires a value}"
      shift 2
      ;;
    --window-size)
      WINDOW_SIZE="${2:?--window-size requires a value}"
      shift 2
      ;;
    --window-stride)
      WINDOW_STRIDE="${2:?--window-stride requires a value}"
      shift 2
      ;;
    --matrix-cache-dir)
      MATRIX_CACHE_DIR="${2:?--matrix-cache-dir requires a value}"
      shift 2
      ;;
    --use-matrix-cache)
      NO_MATRIX_CACHE="0"
      shift
      ;;
    --scores-pkl-ref-to-pred)
      SCORES_PKL_REF_TO_PRED="${2:?--scores-pkl-ref-to-pred requires a value}"
      shift 2
      ;;
    --scores-pkl-ref-to-ref)
      SCORES_PKL_REF_TO_REF="${2:?--scores-pkl-ref-to-ref requires a value}"
      shift 2
      ;;
    --score-index-cache-file)
      SCORE_INDEX_CACHE_FILE="${2:?--score-index-cache-file requires a value}"
      shift 2
      ;;
    --score-index-cache-file-ref-to-ref)
      SCORE_INDEX_CACHE_FILE_REF_TO_REF="${2:?--score-index-cache-file-ref-to-ref requires a value}"
      shift 2
      ;;
    --score-index-cache-dir)
      SCORE_INDEX_CACHE_DIR="${2:?--score-index-cache-dir requires a value}"
      shift 2
      ;;
    --disable-pkl-matrix-source)
      DISABLE_PKL_MATRIX_SOURCE="1"
      shift
      ;;
    --text-metrics-v212-dir)
      TEXT_METRICS_V212_DIR="${2:?--text-metrics-v212-dir requires a value}"
      shift 2
      ;;
    --ref-to-ref-cache-mode)
      REF_TO_REF_CACHE_MODE="${2:?--ref-to-ref-cache-mode requires a value}"
      shift 2
      ;;
    --ref-to-ref-cache-dir)
      REF_TO_REF_CACHE_DIR="${2:?--ref-to-ref-cache-dir requires a value}"
      shift 2
      ;;
    --levenshtein-backend)
      LEVENSHTEIN_BACKEND="${2:?--levenshtein-backend requires a value}"
      shift 2
      ;;
    --workers)
      WORKERS="${2:?--workers requires a value}"
      shift 2
      ;;
    --doc-workers)
      DOC_WORKERS="${2:?--doc-workers requires a value}"
      shift 2
      ;;
    --hough-threshold-range)
      HOUGH_THRESHOLD_START="${2:?--hough-threshold-range requires a start value}"
      HOUGH_THRESHOLD_END="${3:?--hough-threshold-range requires an end value}"
      shift 3
      ;;
    --line-length-range)
      LINE_LENGTH_START="${2:?--line-length-range requires a start value}"
      LINE_LENGTH_END="${3:?--line-length-range requires an end value}"
      shift 3
      ;;
    --line-gap-range)
      LINE_GAP_START="${2:?--line-gap-range requires a start value}"
      LINE_GAP_END="${3:?--line-gap-range requires an end value}"
      shift 3
      ;;
    --align-min-iou-threshold)
      ALIGN_MIN_IOU_THRESHOLD="${2:?--align-min-iou-threshold requires a value}"
      shift 2
      ;;
    --minimum-score-floor|--min-score)
      MINIMUM_SCORE_FLOOR="${2:?--minimum-score-floor requires a value}"
      shift 2
      ;;
    --score-floor-method)
      SCORE_FLOOR_METHOD="${2:?--score-floor-method requires a value}"
      shift 2
      ;;
    --median-absolute-deviation-multiplier|--mad-k)
      MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER="${2:?--median-absolute-deviation-multiplier requires a value}"
      shift 2
      ;;
    --median-absolute-deviation-backend)
      MEDIAN_ABSOLUTE_DEVIATION_BACKEND="${2:?--median-absolute-deviation-backend requires a value}"
      shift 2
      ;;
    --near-peak-ratio)
      NEAR_PEAK_RATIO="${2:?--near-peak-ratio requires a value}"
      shift 2
      ;;
    --near-peak-margin)
      NEAR_PEAK_MARGIN="${2:?--near-peak-margin requires a value}"
      shift 2
      ;;
    --minimum-component-cells|--min-component-cells)
      MINIMUM_COMPONENT_CELLS="${2:?--minimum-component-cells requires a value}"
      shift 2
      ;;
    --minimum-component-rows|--min-component-rows)
      MINIMUM_COMPONENT_ROWS="${2:?--minimum-component-rows requires a value}"
      shift 2
      ;;
    --minimum-component-columns|--min-component-cols)
      MINIMUM_COMPONENT_COLUMNS="${2:?--minimum-component-columns requires a value}"
      shift 2
      ;;
    --connected-component-backend)
      CONNECTED_COMPONENT_BACKEND="${2:?--connected-component-backend requires a value}"
      shift 2
      ;;
    --region-dilation-radius|--dilation-radius)
      REGION_DILATION_RADIUS="${2:?--region-dilation-radius requires a value}"
      shift 2
      ;;
    --final-hough-input-mode)
      FINAL_HOUGH_INPUT_MODE="${2:?--final-hough-input-mode requires a value}"
      shift 2
      ;;
    --adaptive-budget-mask)
      ADAPTIVE_BUDGET_MASK="${2:?--adaptive-budget-mask requires a value}"
      shift 2
      ;;
    --minimum-active-cells|--min-active-cells)
      MINIMUM_ACTIVE_CELLS="${2:?--minimum-active-cells requires a value}"
      shift 2
      ;;
    --minimum-active-rows|--min-active-rows)
      MINIMUM_ACTIVE_ROWS="${2:?--minimum-active-rows requires a value}"
      shift 2
      ;;
    --minimum-active-columns|--min-active-cols)
      MINIMUM_ACTIVE_COLUMNS="${2:?--minimum-active-columns requires a value}"
      shift 2
      ;;
    --minimum-x-span|--min-x-span)
      MINIMUM_X_SPAN="${2:?--minimum-x-span requires a value}"
      shift 2
      ;;
    --minimum-y-span|--min-y-span)
      MINIMUM_Y_SPAN="${2:?--minimum-y-span requires a value}"
      shift 2
      ;;
    --maximum-active-fraction|--max-active-fraction)
      MAXIMUM_ACTIVE_FRACTION="${2:?--maximum-active-fraction requires a value}"
      shift 2
      ;;
    --minimum-matrix-rows)
      MINIMUM_MATRIX_ROWS="${2:?--minimum-matrix-rows requires a value}"
      shift 2
      ;;
    --minimum-matrix-columns)
      MINIMUM_MATRIX_COLUMNS="${2:?--minimum-matrix-columns requires a value}"
      shift 2
      ;;
    --min-surviving-line-nls)
      MIN_SURVIVING_LINE_NLS_WAS_SET="1"
      if [[ $# -lt 2 || "${2:-}" == --* ]]; then
        MIN_SURVIVING_LINE_NLS=""
        shift
      else
        MIN_SURVIVING_LINE_NLS="${2}"
        shift 2
      fi
      ;;
    --selection-objective)
      SELECTION_OBJECTIVE="${2:?--selection-objective requires a value}"
      shift 2
      ;;
    --account)
      ACCOUNT="${2:?--account requires a value}"
      shift 2
      ;;
    --partition)
      PARTITION="${2:?--partition requires a value}"
      shift 2
      ;;
    --time)
      TIME_LIMIT="${2:?--time requires a value}"
      shift 2
      ;;
    --cpus-per-task)
      CPUS_PER_TASK="${2:?--cpus-per-task requires a value}"
      shift 2
      ;;
    --mem)
      MEMORY="${2:?--mem requires a value}"
      shift 2
      ;;
    --final-visual-cpus-per-task)
      FINAL_VISUAL_CPUS_PER_TASK="${2:?--final-visual-cpus-per-task requires a value}"
      shift 2
      ;;
    --final-visual-mem)
      FINAL_VISUAL_MEMORY="${2:?--final-visual-mem requires a value}"
      shift 2
      ;;
    --combination-bundle-scope)
      COMBINATION_BUNDLE_SCOPE="${2:?--combination-bundle-scope requires a value}"
      COMBINATION_BUNDLE_SCOPE_WAS_SET="1"
      shift 2
      ;;
    --no-candidate-lines)
      INCLUDE_CANDIDATE_LINES="0"
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
      echo "[error] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] ${name} must be a positive integer (got: ${value})" >&2
    exit 1
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "[error] ${name} must be a non-negative integer (got: ${value})" >&2
    exit 1
  fi
}

validate_range() {
  local name="$1"
  local start="$2"
  local end="$3"
  local min_value="$4"
  require_nonnegative_int "${name} start" "${start}"
  require_nonnegative_int "${name} end" "${end}"
  if (( start < min_value || end < min_value )); then
    echo "[error] ${name} values must be >= ${min_value}" >&2
    exit 1
  fi
  if (( end < start )); then
    echo "[error] ${name} end must be >= start" >&2
    exit 1
  fi
}

json_bool() {
  local value="$1"
  if [[ "${value}" == "1" ]]; then
    printf 'true'
  else
    printf 'false'
  fi
}

json_optional_number() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    printf 'null'
  else
    printf '%s' "${value}"
  fi
}

require_nonnegative_float() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "[error] ${name} must be a non-negative number in decimal format, for example 50.0 (got: ${value})" >&2
    exit 1
  fi
}

require_unit_float() {
  local name="$1"
  local value="$2"
  require_nonnegative_float "${name}" "${value}"
  "${PYTHON_BIN}" -c 'import sys; value=float(sys.argv[1]); sys.exit(0 if 0.0 <= value <= 1.0 else 1)' "${value}" || {
    echo "[error] ${name} must be between 0.0 and 1.0 inclusive (got: ${value})" >&2
    exit 1
  }
}

if [[ -z "${RUNFILE_JSON}" || -z "${OUTPUT_DIR}" ]]; then
  echo "[error] --runfile-json and --output-dir are required" >&2
  usage >&2
  exit 1
fi
if [[ ! -f "${RUNFILE_JSON}" ]]; then
  echo "[error] runfile JSON does not exist: ${RUNFILE_JSON}" >&2
  exit 1
fi
if [[ "${LEVENSHTEIN_BACKEND}" != "python" && "${LEVENSHTEIN_BACKEND}" != "c" ]]; then
  echo "[error] --levenshtein-backend must be one of: python, c" >&2
  exit 1
fi
if [[ "${REF_TO_REF_CACHE_MODE}" != "off" && "${REF_TO_REF_CACHE_MODE}" != "auto" && "${REF_TO_REF_CACHE_MODE}" != "read-only" ]]; then
  echo "[error] --ref-to-ref-cache-mode must be one of: off, auto, read-only" >&2
  exit 1
fi
if [[ "${SCORE_FLOOR_METHOD}" != "mean_plus_standard_deviation" && "${SCORE_FLOOR_METHOD}" != "median_plus_scaled_median_absolute_deviation" ]]; then
  echo "[error] --score-floor-method must be one of: mean_plus_standard_deviation, median_plus_scaled_median_absolute_deviation" >&2
  exit 1
fi
if [[ "${FINAL_HOUGH_INPUT_MODE}" != "roi" && "${FINAL_HOUGH_INPUT_MODE}" != "roi_and_score_floor" && "${FINAL_HOUGH_INPUT_MODE}" != "roi_or_score_floor" ]]; then
  echo "[error] --final-hough-input-mode must be one of: roi, roi_and_score_floor, roi_or_score_floor" >&2
  exit 1
fi
if [[ "${ADAPTIVE_BUDGET_MASK}" != "final_hough_input" && "${ADAPTIVE_BUDGET_MASK}" != "region_of_interest" && "${ADAPTIVE_BUDGET_MASK}" != "strong_match" && "${ADAPTIVE_BUDGET_MASK}" != "component_region" && "${ADAPTIVE_BUDGET_MASK}" != "score_floor" ]]; then
  echo "[error] --adaptive-budget-mask must be one of: final_hough_input, region_of_interest, strong_match, component_region, score_floor" >&2
  exit 1
fi
if [[ "${COMBINATION_BUNDLE_SCOPE}" != "none" && "${COMBINATION_BUNDLE_SCOPE}" != "winner-only" && "${COMBINATION_BUNDLE_SCOPE}" != "all" && "${COMBINATION_BUNDLE_SCOPE}" != "valid-only" && "${COMBINATION_BUNDLE_SCOPE}" != "invalid-only" ]]; then
  echo "[error] --combination-bundle-scope must be one of: none, winner-only, all, valid-only, invalid-only" >&2
  exit 1
fi
if [[ "${SELECTION_OBJECTIVE}" != "strict_quality" && "${SELECTION_OBJECTIVE}" != "alignment_evidence" && "${SELECTION_OBJECTIVE}" != "non_hallucination_weighted" ]]; then
  echo "[error] --selection-objective must be one of: strict_quality, alignment_evidence, non_hallucination_weighted" >&2
  exit 1
fi
if [[ "${WITH_VISUALS}" != "0" && "${WITH_VISUALS}" != "1" ]]; then
  echo "[error] WITH_VISUALS must be 0 or 1 (got: ${WITH_VISUALS})" >&2
  exit 1
fi
if [[ "${HIDE_LINE_LABELS}" != "0" && "${HIDE_LINE_LABELS}" != "1" ]]; then
  echo "[error] HIDE_LINE_LABELS must be 0 or 1 (got: ${HIDE_LINE_LABELS})" >&2
  exit 1
fi
if [[ "${WITH_VISUALS}" == "1" && "${COMBINATION_BUNDLE_SCOPE_WAS_SET}" == "0" ]]; then
  COMBINATION_BUNDLE_SCOPE="winner-only"
fi
if [[ "${WITH_VISUALS}" == "1" && "${COMBINATION_BUNDLE_SCOPE}" == "none" ]]; then
  echo "[error] --with-visuals requires visualization bundles. Omit --combination-bundle-scope for the default winner-only visual bundle, or choose winner-only/all/valid-only explicitly." >&2
  exit 1
fi
require_nonnegative_float "--minimum-score-floor" "${MINIMUM_SCORE_FLOOR}"
require_nonnegative_float "--median-absolute-deviation-multiplier" "${MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER}"
require_nonnegative_float "--near-peak-ratio" "${NEAR_PEAK_RATIO}"
require_nonnegative_float "--maximum-active-fraction" "${MAXIMUM_ACTIVE_FRACTION}"
if [[ "${MEDIAN_ABSOLUTE_DEVIATION_BACKEND}" != "manual_numpy" && "${MEDIAN_ABSOLUTE_DEVIATION_BACKEND}" != "scipy" ]]; then
  echo "[error] --median-absolute-deviation-backend must be manual_numpy or scipy" >&2
  exit 1
fi
if [[ "${CONNECTED_COMPONENT_BACKEND}" != "cython" && "${CONNECTED_COMPONENT_BACKEND}" != "scipy" && "${CONNECTED_COMPONENT_BACKEND}" != "python" ]]; then
  echo "[error] --connected-component-backend must be cython, scipy, or python" >&2
  exit 1
fi
if [[ -n "${NEAR_PEAK_MARGIN}" ]]; then
  require_nonnegative_float "--near-peak-margin" "${NEAR_PEAK_MARGIN}"
fi

if [[ -z "${MIN_SURVIVING_LINE_NLS}" ]]; then
  if [[ -t 0 ]]; then
    if [[ "${MIN_SURVIVING_LINE_NLS_WAS_SET}" == "1" ]]; then
      echo "[prompt] --min-surviving-line-nls was provided without a value." >&2
      printf '[prompt] Enter --min-surviving-line-nls as a number in [0, 1], for example 0.50. Press Enter to disable: ' >&2
      read -r MIN_SURVIVING_LINE_NLS
    fi
  elif [[ "${MIN_SURVIVING_LINE_NLS_WAS_SET}" == "1" ]]; then
    echo "[error] --min-surviving-line-nls was provided without a value and no interactive terminal is available for prompting." >&2
    echo "[error] Pass a value such as --min-surviving-line-nls 0.50, or omit the flag to disable this filter." >&2
    exit 1
  fi
fi
if [[ -n "${MIN_SURVIVING_LINE_NLS}" ]]; then
  require_unit_float "--min-surviving-line-nls" "${MIN_SURVIVING_LINE_NLS}"
fi

require_positive_int "--max-items" "${MAX_ITEMS}"
require_positive_int "--shard-count" "${SHARD_COUNT}"
require_positive_int "--docs-per-shard" "${DOCS_PER_SHARD}"
require_positive_int "--window-size" "${WINDOW_SIZE}"
require_positive_int "--window-stride" "${WINDOW_STRIDE}"
require_positive_int "--workers" "${WORKERS}"
require_positive_int "--doc-workers" "${DOC_WORKERS}"
require_positive_int "--cpus-per-task" "${CPUS_PER_TASK}"
require_positive_int "--final-visual-cpus-per-task" "${FINAL_VISUAL_CPUS_PER_TASK}"
validate_range "--hough-threshold-range" "${HOUGH_THRESHOLD_START}" "${HOUGH_THRESHOLD_END}" 1
validate_range "--line-length-range" "${LINE_LENGTH_START}" "${LINE_LENGTH_END}" 1
validate_range "--line-gap-range" "${LINE_GAP_START}" "${LINE_GAP_END}" 0

THRESHOLD_COUNT=$(( HOUGH_THRESHOLD_END - HOUGH_THRESHOLD_START + 1 ))
NODE_DOCUMENT_CAPACITY=$(( CPUS_PER_TASK / THRESHOLD_COUNT ))
if (( NODE_DOCUMENT_CAPACITY < 1 )); then
  NODE_DOCUMENT_CAPACITY=1
fi
EFFECTIVE_DOC_WORKERS="${DOC_WORKERS}"
if (( EFFECTIVE_DOC_WORKERS > NODE_DOCUMENT_CAPACITY )); then
  EFFECTIVE_DOC_WORKERS="${NODE_DOCUMENT_CAPACITY}"
fi
EFFECTIVE_THRESHOLD_WORKERS="${THRESHOLD_COUNT}"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/shards"

DOCUMENT_POOL_DIR="${OUTPUT_DIR}/document_pool"
MANIFEST_PATH="${OUTPUT_DIR}/dynamic_pool_manifest.json"
LAUNCH_COMMANDS_PATH="${OUTPUT_DIR}/launch_commands.sh"

printf '[dynamic-pool] initializing pool=%s max_items=%s\n' "${DOCUMENT_POOL_DIR}" "${MAX_ITEMS}"
PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/text_metrics_v2_1_parallel:${PROJECT_DIR}/python_scripts:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/dynamic_pool/initialize_document_pool.py" \
  --runfile-json "${RUNFILE_JSON}" \
  --pool-dir "${DOCUMENT_POOL_DIR}" \
  --max-items "${MAX_ITEMS}"

cat > "${MANIFEST_PATH}" <<EOF
{
  "schema_version": "tuner_dynamic_pool_sbatch_workers_v1",
  "runfile_json": "${RUNFILE_JSON}",
  "output_dir": "${OUTPUT_DIR}",
  "document_pool_dir": "${DOCUMENT_POOL_DIR}",
  "max_items": ${MAX_ITEMS},
  "sbatch_worker_count": ${SHARD_COUNT},
  "docs_per_shard_argument_ignored": ${DOCS_PER_SHARD},
  "hough_ranges": {
    "threshold": [${HOUGH_THRESHOLD_START}, ${HOUGH_THRESHOLD_END}],
    "line_length": [${LINE_LENGTH_START}, ${LINE_LENGTH_END}],
    "line_gap": [${LINE_GAP_START}, ${LINE_GAP_END}],
    "seed": 1
  },
  "parallelism": {
    "requested_doc_workers": ${DOC_WORKERS},
    "computed_node_document_capacity": ${NODE_DOCUMENT_CAPACITY},
    "effective_doc_workers_per_sbatch_worker": ${EFFECTIVE_DOC_WORKERS},
    "threshold_workers_per_active_document": ${EFFECTIVE_THRESHOLD_WORKERS},
    "cpus_per_task": ${CPUS_PER_TASK}
  },
  "with_visuals": $(json_bool "${WITH_VISUALS}"),
  "hide_line_labels": $(json_bool "${HIDE_LINE_LABELS}"),
  "hough_preprocessing": {
    "minimum_score_floor": ${MINIMUM_SCORE_FLOOR},
    "median_absolute_deviation_multiplier": ${MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER},
    "median_absolute_deviation_backend": "${MEDIAN_ABSOLUTE_DEVIATION_BACKEND}",
    "connected_component_backend": "${CONNECTED_COMPONENT_BACKEND}",
    "near_peak_ratio": ${NEAR_PEAK_RATIO},
    "near_peak_margin": $(json_optional_number "${NEAR_PEAK_MARGIN}"),
    "maximum_active_fraction": ${MAXIMUM_ACTIVE_FRACTION}
  },
  "min_surviving_line_nls": $(json_optional_number "${MIN_SURVIVING_LINE_NLS}"),
  "selection_objective": "${SELECTION_OBJECTIVE}",
  "combination_bundle_scope": "${COMBINATION_BUNDLE_SCOPE}",
  "combination_bundle_scope_was_set": $(json_bool "${COMBINATION_BUNDLE_SCOPE_WAS_SET}"),
  "workers": [
EOF

printf '#!/usr/bin/env bash\nset -euo pipefail\n\n' > "${LAUNCH_COMMANDS_PATH}"
submitted_job_ids=()

for (( worker_index = 0; worker_index < SHARD_COUNT; worker_index++ )); do
  worker_id="worker_$(printf '%03d' "${worker_index}")"
  worker_output_dir="${OUTPUT_DIR}/shards/dynamic_${worker_id}"
  combination_bundle_dir="${worker_output_dir}/combination_bundles"
  mkdir -p "${worker_output_dir}"

  if (( worker_index > 0 )); then
    printf ',\n' >> "${MANIFEST_PATH}"
  fi
  cat >> "${MANIFEST_PATH}" <<EOF
    {
      "worker_index": ${worker_index},
      "worker_id": "${worker_id}",
      "output_dir": "${worker_output_dir}",
      "combination_bundle_dir": "${combination_bundle_dir}"
    }
EOF

  worker_args=(
    --runfile-json "${RUNFILE_JSON}"
    --output-dir "${worker_output_dir}"
    --dynamic-document-pool-dir "${DOCUMENT_POOL_DIR}"
    --dynamic-worker-id "${worker_id}"
    --dynamic-cpus-per-task "${CPUS_PER_TASK}"
    --max-items "${MAX_ITEMS}"
    --window-size "${WINDOW_SIZE}"
    --window-stride "${WINDOW_STRIDE}"
    --levenshtein-backend "${LEVENSHTEIN_BACKEND}"
    --workers "${EFFECTIVE_THRESHOLD_WORKERS}"
    --doc-workers "${EFFECTIVE_DOC_WORKERS}"
    --hough-threshold-range "${HOUGH_THRESHOLD_START}" "${HOUGH_THRESHOLD_END}"
    --line-length-range "${LINE_LENGTH_START}" "${LINE_LENGTH_END}"
    --line-gap-range "${LINE_GAP_START}" "${LINE_GAP_END}"
    --align-min-iou-threshold "${ALIGN_MIN_IOU_THRESHOLD}"
    --minimum-score-floor "${MINIMUM_SCORE_FLOOR}"
    --score-floor-method "${SCORE_FLOOR_METHOD}"
    --median-absolute-deviation-multiplier "${MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER}"
    --median-absolute-deviation-backend "${MEDIAN_ABSOLUTE_DEVIATION_BACKEND}"
    --near-peak-ratio "${NEAR_PEAK_RATIO}"
    --minimum-component-cells "${MINIMUM_COMPONENT_CELLS}"
    --minimum-component-rows "${MINIMUM_COMPONENT_ROWS}"
    --minimum-component-columns "${MINIMUM_COMPONENT_COLUMNS}"
    --connected-component-backend "${CONNECTED_COMPONENT_BACKEND}"
    --region-dilation-radius "${REGION_DILATION_RADIUS}"
    --final-hough-input-mode "${FINAL_HOUGH_INPUT_MODE}"
    --adaptive-budget-mask "${ADAPTIVE_BUDGET_MASK}"
    --minimum-active-cells "${MINIMUM_ACTIVE_CELLS}"
    --minimum-active-rows "${MINIMUM_ACTIVE_ROWS}"
    --minimum-active-columns "${MINIMUM_ACTIVE_COLUMNS}"
    --minimum-x-span "${MINIMUM_X_SPAN}"
    --minimum-y-span "${MINIMUM_Y_SPAN}"
    --maximum-active-fraction "${MAXIMUM_ACTIVE_FRACTION}"
    --minimum-matrix-rows "${MINIMUM_MATRIX_ROWS}"
    --minimum-matrix-columns "${MINIMUM_MATRIX_COLUMNS}"
    --selection-objective "${SELECTION_OBJECTIVE}"
    --text-metrics-v212-dir "${TEXT_METRICS_V212_DIR}"
    --ref-to-ref-cache-mode "${REF_TO_REF_CACHE_MODE}"
    --ref-to-ref-cache-dir "${REF_TO_REF_CACHE_DIR}"
    --combination-bundle-dir "${combination_bundle_dir}"
    --combination-bundle-scope "${COMBINATION_BUNDLE_SCOPE}"
    --shard-index "${worker_index}"
  )
  if [[ -n "${NEAR_PEAK_MARGIN}" ]]; then
    worker_args+=(--near-peak-margin "${NEAR_PEAK_MARGIN}")
  fi
  if [[ -n "${MIN_SURVIVING_LINE_NLS}" ]]; then
    worker_args+=(--min-surviving-line-nls "${MIN_SURVIVING_LINE_NLS}")
  fi

  if [[ "${NO_MATRIX_CACHE}" == "1" ]]; then
    worker_args+=(--no-matrix-cache)
  else
    worker_args+=(--matrix-cache-dir "${MATRIX_CACHE_DIR}")
  fi
  if [[ -n "${SCORES_PKL_REF_TO_PRED}" ]]; then
    worker_args+=(--scores-pkl-ref-to-pred "${SCORES_PKL_REF_TO_PRED}")
  fi
  if [[ -n "${SCORES_PKL_REF_TO_REF}" ]]; then
    worker_args+=(--scores-pkl-ref-to-ref "${SCORES_PKL_REF_TO_REF}")
  fi
  if [[ -n "${SCORE_INDEX_CACHE_FILE}" ]]; then
    worker_args+=(--score-index-cache-file "${SCORE_INDEX_CACHE_FILE}")
  fi
  if [[ -n "${SCORE_INDEX_CACHE_FILE_REF_TO_REF}" ]]; then
    worker_args+=(--score-index-cache-file-ref-to-ref "${SCORE_INDEX_CACHE_FILE_REF_TO_REF}")
  fi
  if [[ -n "${SCORE_INDEX_CACHE_DIR}" ]]; then
    worker_args+=(--score-index-cache-dir "${SCORE_INDEX_CACHE_DIR}")
  fi
  if [[ "${DISABLE_PKL_MATRIX_SOURCE}" == "1" ]]; then
    worker_args+=(--disable-pkl-matrix-source)
  fi
  if [[ "${INCLUDE_CANDIDATE_LINES}" == "1" ]]; then
    worker_args+=(--combination-bundle-include-candidate-lines)
  fi

  sbatch_command=(
    sbatch
    --parsable
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --time="${TIME_LIMIT}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --job-name="churro_tune_${worker_id}"
    --output="${OUTPUT_DIR}/logs/${worker_id}_%j.out"
    --error="${OUTPUT_DIR}/logs/${worker_id}_%j.err"
    "${SCRIPT_DIR}/run_hough_parameter_sweep_shard.sbatch"
    "${worker_args[@]}"
  )

  printf '%q ' "${sbatch_command[@]}" >> "${LAUNCH_COMMANDS_PATH}"
  printf '\n' >> "${LAUNCH_COMMANDS_PATH}"

  echo "[submit] dynamic_worker=${worker_id} output=${worker_output_dir} pool=${DOCUMENT_POOL_DIR}"
  submitted_job_id="$("${sbatch_command[@]}")"
  submitted_job_ids+=("${submitted_job_id}")
  echo "[submit] dynamic_worker=${worker_id} job_id=${submitted_job_id}"
done

cat >> "${MANIFEST_PATH}" <<'EOF'
  ]
}
EOF

chmod +x "${LAUNCH_COMMANDS_PATH}"

if [[ "${WITH_VISUALS}" == "1" && "${#submitted_job_ids[@]}" -gt 0 ]]; then
  dependency_job_list="$(IFS=:; echo "${submitted_job_ids[*]}")"
  final_visual_args=(
    --all-languages
    --all-document-types
    --runfile-json "${RUNFILE_JSON}"
    --shards-dir "${OUTPUT_DIR}/shards"
    --output-dir "${OUTPUT_DIR}"
    --documents-per-shard "1"
    --ref-to-pred-scores-pkl "${SCORES_PKL_REF_TO_PRED}"
    --ref-to-ref-scores-pkl "${SCORES_PKL_REF_TO_REF}"
  )
  if [[ "${HIDE_LINE_LABELS}" == "1" ]]; then
    final_visual_args+=(--hide-line-labels)
  fi

  final_visual_command=(
    sbatch
    --parsable
    --dependency="afterok:${dependency_job_list}"
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --time="${TIME_LIMIT}"
    --nodes=1
    --ntasks=1
    --ntasks-per-node=1
    --cpus-per-task="${FINAL_VISUAL_CPUS_PER_TASK}"
    --mem="${FINAL_VISUAL_MEMORY}"
    --job-name="churro_tune_visuals"
    --output="${OUTPUT_DIR}/logs/final_visuals_%j.out"
    --error="${OUTPUT_DIR}/logs/final_visuals_%j.err"
    "${SCRIPT_DIR}/run_language_hough_parameter_metric_analysis.sh"
    "${final_visual_args[@]}"
  )
  printf '%q ' "${final_visual_command[@]}" >> "${LAUNCH_COMMANDS_PATH}"
  printf '\n' >> "${LAUNCH_COMMANDS_PATH}"
  final_visual_job_id="$("${final_visual_command[@]}")"
  echo "[submit] final_visuals_job_id=${final_visual_job_id} dependency=afterok:${dependency_job_list}"
fi

echo "[submit] dynamic_manifest=${MANIFEST_PATH}"
echo "[submit] document_pool=${DOCUMENT_POOL_DIR}"
echo "[submit] launch_commands=${LAUNCH_COMMANDS_PATH}"
echo "[submit] logs=${OUTPUT_DIR}/logs"
echo "[submit] effective_doc_workers_per_worker=${EFFECTIVE_DOC_WORKERS} threshold_workers=${EFFECTIVE_THRESHOLD_WORKERS}"
