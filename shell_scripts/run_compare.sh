#!/usr/bin/env bash
#SBATCH --job-name=compare_py
#SBATCH --account=project_2017385
#SBATCH --partition=medium
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -o logs/compare_v2_%j.out
#SBATCH -e logs/compare_v2_%j.err
set -euo pipefail

find_repo_root_from() {
  local search_dir="$1"
  while [[ -n "${search_dir}" && "${search_dir}" != "/" ]]; do
    if [[ -f "${search_dir}/python_scripts/compare.py" && -d "${search_dir}/text_metrics_v2_12_parallel" ]]; then
      printf '%s\n' "${search_dir}"
      return 0
    fi
    search_dir="$(dirname "${search_dir}")"
  done
  return 1
}

resolve_repo_root() {
  local script_source_dir
  script_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -n "${HTR_CONTEXT_OCR_DIR:-}" && -f "${HTR_CONTEXT_OCR_DIR}/python_scripts/compare.py" ]]; then
    printf '%s\n' "$(cd "${HTR_CONTEXT_OCR_DIR}" && pwd)"
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && find_repo_root_from "${SLURM_SUBMIT_DIR}"; then
    return 0
  fi
  if find_repo_root_from "${script_source_dir}"; then
    return 0
  fi
  if find_repo_root_from "$(pwd)"; then
    return 0
  fi
  return 1
}

REPO_ROOT="$(resolve_repo_root)" || {
  echo "[error] Could not locate the HTR-context-OCR repo root. Submit from inside the repo or set HTR_CONTEXT_OCR_DIR." >&2
  exit 2
}
cd "${REPO_ROOT}"

if type module >/dev/null 2>&1; then
  module --force purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

RUNFILE_JSON="${RUNFILE_JSON:-${REPO_ROOT}/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/results/compare/scores.pkl}"
METRIC="${METRIC:-chrf}"
COMPARISON_MODE="${COMPARISON_MODE:-ref-pred}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
WINDOW_STRIDE="${WINDOW_STRIDE:-50}"
MAX_ITEMS="${MAX_ITEMS:-}"
GRAPH_OUTPUT_DIR="${GRAPH_OUTPUT_DIR:-}"
PLOT_EXISTING_SCORE_DIR="${PLOT_EXISTING_SCORE_DIR:-}"

usage() {
  cat <<'USAGE'
Usage: run_compare.sh [options]

Options:
  --runfile-json <path>      Path to outputs.json to compare
  --output <path>            Output pickle file path (scores.pkl)
  --metric <name>            Score metric: chrf, bleu, levenshtein, or all
  --comparison-mode <mode>   Text pair: ref-pred, ref-ref, or pred-pred
  --window-size <n>          Sliding window size in characters
  --window-stride <n>        Sliding window stride in characters
  --max-items <n>            Process only the first N documents
  --num-documents <n>        Alias for --max-items
  --graph-output-dir <path>  Directory for score matrix PNG graphs
  --plot-existing-score-dir <path>
                              Plot existing metric score pickle files instead of recomputing
  -h, --help                 Show this help text

Environment variable overrides are also supported for all options:
RUNFILE_JSON, OUTPUT, METRIC, COMPARISON_MODE, WINDOW_SIZE, WINDOW_STRIDE,
MAX_ITEMS, GRAPH_OUTPUT_DIR, PLOT_EXISTING_SCORE_DIR, HTR_CONTEXT_OCR_DIR
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runfile-json)
      [[ $# -ge 2 ]] || { echo "[error] --runfile-json requires a value" >&2; exit 1; }
      RUNFILE_JSON="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "[error] --output requires a value" >&2; exit 1; }
      OUTPUT="$2"
      shift 2
      ;;
    --metric)
      [[ $# -ge 2 ]] || { echo "[error] --metric requires a value" >&2; exit 1; }
      METRIC="$2"
      shift 2
      ;;
    --comparison-mode)
      [[ $# -ge 2 ]] || { echo "[error] --comparison-mode requires a value" >&2; exit 1; }
      COMPARISON_MODE="$2"
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
    --max-items|--num-documents)
      [[ $# -ge 2 ]] || { echo "[error] $1 requires a value" >&2; exit 1; }
      MAX_ITEMS="$2"
      shift 2
      ;;
    --graph-output-dir)
      [[ $# -ge 2 ]] || { echo "[error] --graph-output-dir requires a value" >&2; exit 1; }
      GRAPH_OUTPUT_DIR="$2"
      shift 2
      ;;
    --plot-existing-score-dir)
      [[ $# -ge 2 ]] || { echo "[error] --plot-existing-score-dir requires a value" >&2; exit 1; }
      PLOT_EXISTING_SCORE_DIR="$2"
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

if [[ -n "${PLOT_EXISTING_SCORE_DIR}" ]]; then
  [[ -d "${PLOT_EXISTING_SCORE_DIR}" ]] || { echo "[error] --plot-existing-score-dir does not exist: ${PLOT_EXISTING_SCORE_DIR}" >&2; exit 1; }
else
  [[ -n "${RUNFILE_JSON}" ]] || { echo "[error] RUNFILE_JSON must not be empty" >&2; exit 1; }
  [[ -f "${RUNFILE_JSON}" ]] || { echo "[error] RUNFILE_JSON does not exist: ${RUNFILE_JSON}" >&2; exit 1; }
fi

[[ -n "${OUTPUT}" ]] || { echo "[error] OUTPUT must not be empty" >&2; exit 1; }

METRIC="${METRIC,,}"
[[ "${METRIC}" != "blue" ]] || METRIC="bleu"
case "${METRIC}" in
  chrf|bleu|levenshtein|all) ;;
  *) echo "[error] --metric must be one of: chrf, bleu, levenshtein, all (got: ${METRIC})" >&2; exit 1 ;;
esac

COMPARISON_MODE="${COMPARISON_MODE,,}"
COMPARISON_MODE="${COMPARISON_MODE//_/-}"
case "${COMPARISON_MODE}" in
  ref-pred|ref-to-pred|reference-prediction) COMPARISON_MODE="ref-pred" ;;
  ref-ref|ref-to-ref|reference-reference) COMPARISON_MODE="ref-ref" ;;
  pred-pred|pred-to-pred|prediction-prediction|pre-pred) COMPARISON_MODE="pred-pred" ;;
  *) echo "[error] --comparison-mode must be one of: ref-pred, ref-ref, pred-pred (got: ${COMPARISON_MODE})" >&2; exit 1 ;;
esac

[[ "${WINDOW_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "[error] --window-size must be a positive integer (got: ${WINDOW_SIZE})" >&2; exit 1; }
[[ "${WINDOW_STRIDE}" =~ ^[1-9][0-9]*$ ]] || { echo "[error] --window-stride must be a positive integer (got: ${WINDOW_STRIDE})" >&2; exit 1; }
if [[ -n "${MAX_ITEMS}" ]] && ! [[ "${MAX_ITEMS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-items must be a positive integer (got: ${MAX_ITEMS})" >&2
  exit 1
fi

command -v python3 >/dev/null 2>&1 || { echo "[error] python3 not found in PATH" >&2; exit 1; }

mkdir -p "${REPO_ROOT}/logs" "$(dirname "${OUTPUT}")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python_args=(
  "${REPO_ROOT}/python_scripts/compare.py"
  --output "${OUTPUT}"
  --metric "${METRIC}"
  --comparison-mode "${COMPARISON_MODE}"
  --window-size "${WINDOW_SIZE}"
  --window-stride "${WINDOW_STRIDE}"
)

if [[ -n "${PLOT_EXISTING_SCORE_DIR}" ]]; then
  python_args+=(--plot-existing-score-dir "${PLOT_EXISTING_SCORE_DIR}")
else
  python_args+=(--runfile-json "${RUNFILE_JSON}")
fi
if [[ -n "${MAX_ITEMS}" ]]; then
  python_args+=(--max-items "${MAX_ITEMS}")
fi
if [[ -n "${GRAPH_OUTPUT_DIR}" ]]; then
  python_args+=(--graph-output-dir "${GRAPH_OUTPUT_DIR}")
fi

echo "[run] Starting compare.py"
echo "[run]   repo_root=${REPO_ROOT}"
if [[ -n "${PLOT_EXISTING_SCORE_DIR}" ]]; then
  echo "[run]   plot_existing_score_dir=${PLOT_EXISTING_SCORE_DIR}"
else
  echo "[run]   runfile_json=${RUNFILE_JSON}"
fi
echo "[run]   output=${OUTPUT}"
echo "[run]   metric=${METRIC}"
echo "[run]   comparison_mode=${COMPARISON_MODE}"
echo "[run]   window_size=${WINDOW_SIZE}"
echo "[run]   window_stride=${WINDOW_STRIDE}"
[[ -z "${MAX_ITEMS}" ]] || echo "[run]   max_items=${MAX_ITEMS}"
[[ -z "${GRAPH_OUTPUT_DIR}" ]] || echo "[run]   graph_output_dir=${GRAPH_OUTPUT_DIR}"
echo "[run]   python=python3"

python3 "${python_args[@]}"

echo "[run] compare.py completed successfully."
