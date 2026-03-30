#!/usr/bin/env bash
#SBATCH --job-name=eval_improvement_v2
#SBATCH --account=project_2000539
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=48G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy
#SBATCH -o logs/eval_improvement_v2_%j.out
#SBATCH -e logs/eval_improvement_v2_%j.err

set -euo pipefail

# Optional environment setup for CSC/HPC. Safe to ignore on machines without `module`.
if command -v module >/dev/null 2>&1; then
  module purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

# Under Slurm, BASH_SOURCE points to a copied spool file; keep scheduler chdir/PWD.
# Outside Slurm, resolve the script location so manual runs from other cwd still work.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

IMG_DIR="${IMG_DIR:-}"
RUNFILE_JSON="${RUNFILE_JSON:-}"
PROJECT_ROOT_RESULTS="${PROJECT_ROOT_RESULTS:-/scratch/project_2017385/dorian/Churro_copy/results/eval_pipeline_results}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
WINDOW_STRIDE="${WINDOW_STRIDE:-50}"
MAX_ITEMS="${MAX_ITEMS:-}"

usage() {
  cat <<'USAGE'
Usage: run_evaluation_improvement_v2.sh [options]

Required:
  --img-dir <dir>                Directory containing the document images.
  --runfile-json <path>          Path to outputs.json (run results).

Optional:
  --project-root-results <dir>   Root output directory for the whole pipeline.
                                 Default: /scratch/project_2017385/dorian/Churro_copy/results/eval_pipeline_results
  --window-size <n>              Sliding window size in characters. Default: 100
  --window-stride <n>            Sliding window stride in characters. Default: 50
  --max-items <n>                Process only the first N items (all stages).
  -h, --help                     Show this help text.

Environment variable overrides are also supported for all options:
  IMG_DIR, RUNFILE_JSON, PROJECT_ROOT_RESULTS, WINDOW_SIZE, WINDOW_STRIDE, MAX_ITEMS

Outputs:
  A run-specific directory is created under PROJECT_ROOT_RESULTS:
    <root>/window_<WINDOW_SIZE>_stride_<WINDOW_STRIDE>/<timestamp>/

  where <timestamp> is generated as YYYYMMDD_HHMMSS.

  Inside that directory:
    align_text_blocks_from_endpoints_no_pkl/{*_adjusted_pred.txt,*_line_segments.txt,summary.json}
    compare_aligned/scores_aligned.pkl
    aligned_graphs/{full_figures,graph_only}/

  Note:
    This v2 pipeline skips pre-alignment compare.py/scores.pkl generation.
    Alignment is computed directly from outputs.json in-memory matrices.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --img-dir)
      IMG_DIR="${2:-}"
      shift 2
      ;;
    --runfile-json)
      RUNFILE_JSON="${2:-}"
      shift 2
      ;;
    --project-root-results)
      PROJECT_ROOT_RESULTS="${2:-}"
      shift 2
      ;;
    --window-size)
      WINDOW_SIZE="${2:-}"
      shift 2
      ;;
    --window-stride)
      WINDOW_STRIDE="${2:-}"
      shift 2
      ;;
    --max-items)
      MAX_ITEMS="${2:-}"
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

if [[ -z "${IMG_DIR}" ]]; then
  echo "[error] --img-dir (or IMG_DIR) must be set" >&2
  exit 1
fi
if [[ ! -d "${IMG_DIR}" ]]; then
  echo "[error] IMG_DIR does not exist or is not a directory: ${IMG_DIR}" >&2
  exit 1
fi

if [[ -z "${RUNFILE_JSON}" ]]; then
  echo "[error] --runfile-json (or RUNFILE_JSON) must be set" >&2
  exit 1
fi
if [[ ! -f "${RUNFILE_JSON}" ]]; then
  echo "[error] RUNFILE_JSON does not exist: ${RUNFILE_JSON}" >&2
  exit 1
fi

if [[ -z "${PROJECT_ROOT_RESULTS}" ]]; then
  echo "[error] --project-root-results (or PROJECT_ROOT_RESULTS) must not be empty" >&2
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

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 not found in PATH" >&2
  exit 1
fi

RUN_BASE_DIR="${PROJECT_ROOT_RESULTS}/window_${WINDOW_SIZE}_stride_${WINDOW_STRIDE}"

# Create a unique per-run timestamped directory under the window/stride bucket.
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_BASE_DIR}/${RUN_TIMESTAMP}"
while [[ -e "${RUN_DIR}" ]]; do
  sleep 1
  RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="${RUN_BASE_DIR}/${RUN_TIMESTAMP}"
done

ALIGN_DIR="${RUN_DIR}/align_text_blocks_from_endpoints_no_pkl"

COMPARE_ALIGNED_DIR="${RUN_DIR}/compare_aligned"
SCORES_ALIGNED_PKL="${COMPARE_ALIGNED_DIR}/scores_aligned.pkl"

ALIGNED_GRAPHS_DIR="${RUN_DIR}/aligned_graphs"

mkdir -p \
  "${RUN_DIR}" \
  "${ALIGN_DIR}" \
  "${COMPARE_ALIGNED_DIR}" \
  "${ALIGNED_GRAPHS_DIR}" \
  "${SCRIPT_DIR}/logs"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export PYTHONUNBUFFERED=1

echo "[run] Evaluation-improvement pipeline"
echo "[run]   script_dir=${SCRIPT_DIR}"
echo "[run]   img_dir=${IMG_DIR}"
echo "[run]   runfile_json=${RUNFILE_JSON}"
echo "[run]   project_root_results=${PROJECT_ROOT_RESULTS}"
echo "[run]   run_base_dir=${RUN_BASE_DIR}"
echo "[run]   run_timestamp=${RUN_TIMESTAMP}"
echo "[run]   run_dir=${RUN_DIR}"
echo "[run]   window_size=${WINDOW_SIZE}"
echo "[run]   window_stride=${WINDOW_STRIDE}"
if [[ -n "${MAX_ITEMS}" ]]; then
  echo "[run]   max_items=${MAX_ITEMS}"
fi

# --------------------------
# Stage 1: align all cases (in-memory matrices + in-memory Hough endpoints)
# --------------------------
echo "[run] Stage 1/3: align_text_blocks_from_endpoints_no_pkl.py"
ALIGN_ARGS=(
  --runfile-json "${RUNFILE_JSON}"
  --output-dir "${ALIGN_DIR}"
  --window-size "${WINDOW_SIZE}"
  --window-stride "${WINDOW_STRIDE}"
)
if [[ -n "${MAX_ITEMS}" ]]; then
  ALIGN_ARGS+=(--max-items "${MAX_ITEMS}")
fi
python3 align_text_blocks_from_endpoints_no_pkl.py "${ALIGN_ARGS[@]}"

# --------------------------
# Stage 2: compare aligned texts (recompute chrF matrices using adjusted_pred.txt)
# --------------------------
echo "[run] Stage 2/3: compare_aligned_texts.py"
python3 compare_aligned_texts.py \
  --runfile-json "${RUNFILE_JSON}" \
  --aligned-dir "${ALIGN_DIR}" \
  --txt-glob "*_adjusted_pred.txt" \
  --output "${SCORES_ALIGNED_PKL}" \
  --window-size "${WINDOW_SIZE}" \
  --window-stride "${WINDOW_STRIDE}"

# --------------------------
# Stage 3: pure visualisation of aligned score matrices (heatmap only)
# --------------------------
echo "[run] Stage 3/3: visualise_scores_heatmap_only.py"
HEATMAP_ARGS=(
  --img-dir "${IMG_DIR}"
  --scores-pkl "${SCORES_ALIGNED_PKL}"
  --results-dir "${ALIGNED_GRAPHS_DIR}"
)
if [[ -n "${MAX_ITEMS}" ]]; then
  HEATMAP_ARGS+=(--max-items "${MAX_ITEMS}")
fi
python3 visualise_scores_heatmap_only.py "${HEATMAP_ARGS[@]}"

echo "[run] Done. Results written under: ${RUN_DIR}"
