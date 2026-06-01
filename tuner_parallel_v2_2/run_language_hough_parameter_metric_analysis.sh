#!/usr/bin/env bash
#SBATCH --job-name=lang_hough_diag
#SBATCH --account=project_2005072
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_2
#SBATCH -o logs/language_hough_parameter_metric_analysis_%j.out
#SBATCH -e logs/language_hough_parameter_metric_analysis_%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

set -euo pipefail

# Under Slurm, BASH_SOURCE points to a copied spool file; keep scheduler chdir/PWD.
# Outside Slurm, resolve the script location so manual runs from other cwd still work.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

PYTHON_SCRIPT="${PYTHON_SCRIPT:-${SCRIPT_DIR}/tools/language_hough_parameter_metric_analysis.py}"
RUNFILE_JSON="${RUNFILE_JSON:-/scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json}"
SHARDS_DIR="${SHARDS_DIR:-/scratch/project_2000539/dorian/results/tuner_parallel_v2_2_dynamic_pool_example/shards}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_2/_language_hough_parameter_metric_visuals_script}"
DOCUMENTS_PER_SHARD="${DOCUMENTS_PER_SHARD:-50}"
MAX_DOCUMENTS="${MAX_DOCUMENTS:-}"
REF_TO_PRED_SCORES_PKL="${REF_TO_PRED_SCORES_PKL:-/scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/scores_reference_prediction_ws50_st35.pkl}"
REF_TO_REF_SCORES_PKL="${REF_TO_REF_SCORES_PKL:-/scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/scores_reference_self_ws50_st35.pkl}"
MAX_CONTINUOUS_BINS="${MAX_CONTINUOUS_BINS:-50}"
SAVED_FIGURE_DPI="${SAVED_FIGURE_DPI:-140}"
ALL_LANGUAGES=0
ALL_DOCUMENT_TYPES=0
SKIP_GRAPH_GRIDS=0
SKIP_BEST_VISUAL_PANELS=0
HIDE_LINE_LABELS="${HIDE_LINE_LABELS:-0}"
LANGUAGE_VALUES=()
DOCUMENT_TYPE_VALUES=()

usage() {
  cat <<'USAGE'
Usage: run_language_hough_parameter_metric_analysis.sh [options]

Required selection options:
  --language <value>              main_language value to analyze; repeat for multiple languages
  --all-languages                 Analyze every main_language value in outputs.json
  --document-type <value>         document_type value to analyze; repeat for multiple types
  --all-document-types            Analyze every document_type value in outputs.json

Input/output options:
  --runfile-json <path>           Path to outputs.json with main_language/document_type metadata
  --shards-dir <path>             Path to tuner output shards directory
  --output-dir <path>             Directory where CSVs and PNGs will be written
  --documents-per-shard <n>       Number of document indices per shard folder
  --max-documents <n>             Max loadable documents per language/document_type pair
  --ref-to-pred-scores-pkl <path> Fallback ref_to_pred score matrix pickle
  --ref-to-ref-scores-pkl <path>  Fallback ref_to_ref score matrix pickle

Plot options:
  --skip-graph-grids              Do not create per-document 18-graph grids
  --skip-best-visual-panels       Do not create temporary best panels or final stitched panel
  --hide-line-labels              Hide raw and surviving line labels in stitched panels
  --max-continuous-bins <n>       Max bins for continuous component-vs-score line graphs
  --saved-figure-dpi <n>          DPI for saved PNG figures

Other:
  -h, --help                      Show this help text

Environment variable overrides are also supported for scalar paths/settings:
PYTHON_SCRIPT, RUNFILE_JSON, SHARDS_DIR, OUTPUT_DIR, DOCUMENTS_PER_SHARD,
MAX_DOCUMENTS, REF_TO_PRED_SCORES_PKL, REF_TO_REF_SCORES_PKL,
MAX_CONTINUOUS_BINS, SAVED_FIGURE_DPI
USAGE
}

require_value() {
  local option_name="$1"
  local value_count="$2"
  if [[ "${value_count}" -lt 2 ]]; then
    echo "[error] ${option_name} requires a value" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --language)
      require_value "$1" "$#"
      LANGUAGE_VALUES+=("$2")
      shift 2
      ;;
    --all-languages)
      ALL_LANGUAGES=1
      shift
      ;;
    --document-type)
      require_value "$1" "$#"
      DOCUMENT_TYPE_VALUES+=("$2")
      shift 2
      ;;
    --all-document-types)
      ALL_DOCUMENT_TYPES=1
      shift
      ;;
    --runfile-json)
      require_value "$1" "$#"
      RUNFILE_JSON="$2"
      shift 2
      ;;
    --shards-dir)
      require_value "$1" "$#"
      SHARDS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$1" "$#"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --documents-per-shard)
      require_value "$1" "$#"
      DOCUMENTS_PER_SHARD="$2"
      shift 2
      ;;
    --max-documents)
      require_value "$1" "$#"
      MAX_DOCUMENTS="$2"
      shift 2
      ;;
    --ref-to-pred-scores-pkl)
      require_value "$1" "$#"
      REF_TO_PRED_SCORES_PKL="$2"
      shift 2
      ;;
    --ref-to-ref-scores-pkl)
      require_value "$1" "$#"
      REF_TO_REF_SCORES_PKL="$2"
      shift 2
      ;;
    --skip-graph-grids)
      SKIP_GRAPH_GRIDS=1
      shift
      ;;
    --skip-best-visual-panels)
      SKIP_BEST_VISUAL_PANELS=1
      shift
      ;;
    --hide-line-labels)
      HIDE_LINE_LABELS=1
      shift
      ;;
    --max-continuous-bins)
      require_value "$1" "$#"
      MAX_CONTINUOUS_BINS="$2"
      shift 2
      ;;
    --saved-figure-dpi)
      require_value "$1" "$#"
      SAVED_FIGURE_DPI="$2"
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

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "[error] Python script does not exist: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${RUNFILE_JSON}" ]]; then
  echo "[error] RUNFILE_JSON does not exist: ${RUNFILE_JSON}" >&2
  exit 1
fi

if [[ ! -d "${SHARDS_DIR}" ]]; then
  echo "[error] SHARDS_DIR does not exist: ${SHARDS_DIR}" >&2
  exit 1
fi

if ! [[ "${DOCUMENTS_PER_SHARD}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --documents-per-shard must be a positive integer (got: ${DOCUMENTS_PER_SHARD})" >&2
  exit 1
fi

if [[ -n "${MAX_DOCUMENTS}" ]] && ! [[ "${MAX_DOCUMENTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-documents must be a positive integer (got: ${MAX_DOCUMENTS})" >&2
  exit 1
fi

if ! [[ "${MAX_CONTINUOUS_BINS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-continuous-bins must be a positive integer (got: ${MAX_CONTINUOUS_BINS})" >&2
  exit 1
fi

if ! [[ "${SAVED_FIGURE_DPI}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --saved-figure-dpi must be a positive integer (got: ${SAVED_FIGURE_DPI})" >&2
  exit 1
fi

if [[ "${ALL_LANGUAGES}" -eq 0 && "${#LANGUAGE_VALUES[@]}" -eq 0 ]]; then
  echo "[error] Pass at least one --language or --all-languages" >&2
  exit 1
fi

if [[ "${ALL_LANGUAGES}" -eq 1 && "${#LANGUAGE_VALUES[@]}" -gt 0 ]]; then
  echo "[error] Use either --language or --all-languages, not both" >&2
  exit 1
fi

if [[ "${ALL_DOCUMENT_TYPES}" -eq 0 && "${#DOCUMENT_TYPE_VALUES[@]}" -eq 0 ]]; then
  echo "[error] Pass at least one --document-type or --all-document-types" >&2
  exit 1
fi

if [[ "${ALL_DOCUMENT_TYPES}" -eq 1 && "${#DOCUMENT_TYPE_VALUES[@]}" -gt 0 ]]; then
  echo "[error] Use either --document-type or --all-document-types, not both" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 not found in PATH after module load pytorch" >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_DIR}"

# The analysis script imports sibling packages such as outputs/ and matrices/.
# When Python executes a file by absolute path, it places the file directory
# (tools/) on sys.path instead of the tuner root.  Exporting SCRIPT_DIR keeps the
# launcher repeatable from sbatch and terminal without modifying metric logic.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

PYTHON_ARGS=(
  "${PYTHON_SCRIPT}"
  --runfile-json "${RUNFILE_JSON}"
  --shards-dir "${SHARDS_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --documents-per-shard "${DOCUMENTS_PER_SHARD}"
  --ref-to-pred-scores-pkl "${REF_TO_PRED_SCORES_PKL}"
  --ref-to-ref-scores-pkl "${REF_TO_REF_SCORES_PKL}"
  --max-continuous-bins "${MAX_CONTINUOUS_BINS}"
  --saved-figure-dpi "${SAVED_FIGURE_DPI}"
)

if [[ "${ALL_LANGUAGES}" -eq 1 ]]; then
  PYTHON_ARGS+=(--all-languages)
else
  for language_value in "${LANGUAGE_VALUES[@]}"; do
    PYTHON_ARGS+=(--language "${language_value}")
  done
fi

if [[ "${ALL_DOCUMENT_TYPES}" -eq 1 ]]; then
  PYTHON_ARGS+=(--all-document-types)
else
  for document_type_value in "${DOCUMENT_TYPE_VALUES[@]}"; do
    PYTHON_ARGS+=(--document-type "${document_type_value}")
  done
fi

if [[ -n "${MAX_DOCUMENTS}" ]]; then
  PYTHON_ARGS+=(--max-documents "${MAX_DOCUMENTS}")
fi

if [[ "${SKIP_GRAPH_GRIDS}" -eq 1 ]]; then
  PYTHON_ARGS+=(--skip-graph-grids)
fi

if [[ "${SKIP_BEST_VISUAL_PANELS}" -eq 1 ]]; then
  PYTHON_ARGS+=(--skip-best-visual-panels)
fi

if [[ "${HIDE_LINE_LABELS}" -eq 1 ]]; then
  PYTHON_ARGS+=(--hide-line-labels)
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"

echo "[run] Starting language_hough_parameter_metric_analysis.py"
echo "[run]   python=python3"
echo "[run]   script=${PYTHON_SCRIPT}"
echo "[run]   runfile_json=${RUNFILE_JSON}"
echo "[run]   shards_dir=${SHARDS_DIR}"
echo "[run]   output_dir=${OUTPUT_DIR}"
echo "[run]   documents_per_shard=${DOCUMENTS_PER_SHARD}"
echo "[run]   max_documents=${MAX_DOCUMENTS:-<none>}"
echo "[run]   all_languages=${ALL_LANGUAGES}"
echo "[run]   languages=${LANGUAGE_VALUES[*]:-<none>}"
echo "[run]   all_document_types=${ALL_DOCUMENT_TYPES}"
echo "[run]   document_types=${DOCUMENT_TYPE_VALUES[*]:-<none>}"
echo "[run]   skip_graph_grids=${SKIP_GRAPH_GRIDS}"
echo "[run]   skip_best_visual_panels=${SKIP_BEST_VISUAL_PANELS}"
echo "[run]   hide_line_labels=${HIDE_LINE_LABELS}"

python3 "${PYTHON_ARGS[@]}"

echo "[run] language_hough_parameter_metric_analysis.py completed successfully."
