#!/usr/bin/env bash
#SBATCH --job-name=lev_along_lines
#SBATCH --account=project_2000539
#SBATCH --partition=medium
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=24G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2
#SBATCH -o logs/levenshtein_along_lines_%j.out
#SBATCH -e logs/levenshtein_along_lines_%j.err
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

RUNFILE_JSON="${RUNFILE_JSON:-/scratch/project_2017385/dorian/churro_finnish_dataset/run_results/dev_split/outputs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2/results/levenshtein_along_lines_metric/20260401_dev_split_v2_1_true_iou}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-35}"
TARGET_FNAME="${TARGET_FNAME:-}"
MAX_ITEMS="${MAX_ITEMS:-}"
HOUGH_THRESHOLD="${HOUGH_THRESHOLD:-12}"
HOUGH_LINE_LENGTH="${HOUGH_LINE_LENGTH:-8}"
HOUGH_LINE_GAP="${HOUGH_LINE_GAP:-8}"
HOUGH_SEED="${HOUGH_SEED:-2}"
HOUGH_START="${HOUGH_START:-2.2}"
LEVENSHTEIN_BACKEND="${LEVENSHTEIN_BACKEND:-c}"

usage() {
  cat <<'USAGE'
Usage: run_levenshtein_along_lines_metric.sh [options]

Options:
  --runfile-json <path>      Path to outputs.json
  --output-dir <path>        Output directory for reports and summary
  --window-size <n>          Sliding window size in characters
  --window-stride <n>        Sliding window stride in characters
  --target-fname <name>      Optional exact/basename target file
  --max-items <n>            Optional maximum number of processed items
  --hough-threshold <n>      Hough vote threshold
  --hough-line-length <n>    Minimum accepted Hough line length
  --hough-line-gap <n>       Maximum Hough line gap
  --hough-seed <n>           Base random seed for Hough
  --hough-start <x>          Initial adaptive Hough threshold start
  --levenshtein-backend <b>  Backend: python or c
  -h, --help                 Show this help text

Environment variable overrides are also supported for all options:
RUNFILE_JSON, OUTPUT_DIR, WINDOW_SIZE, WINDOW_STRIDE, TARGET_FNAME, MAX_ITEMS,
HOUGH_THRESHOLD, HOUGH_LINE_LENGTH, HOUGH_LINE_GAP, HOUGH_SEED, HOUGH_START,
LEVENSHTEIN_BACKEND
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runfile-json)
      if [[ $# -lt 2 ]]; then
        echo "[error] --runfile-json requires a value" >&2
        exit 1
      fi
      RUNFILE_JSON="$2"
      shift 2
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "[error] --output-dir requires a value" >&2
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --window-size)
      if [[ $# -lt 2 ]]; then
        echo "[error] --window-size requires a value" >&2
        exit 1
      fi
      WINDOW_SIZE="$2"
      shift 2
      ;;
    --window-stride)
      if [[ $# -lt 2 ]]; then
        echo "[error] --window-stride requires a value" >&2
        exit 1
      fi
      WINDOW_STRIDE="$2"
      shift 2
      ;;
    --target-fname)
      if [[ $# -lt 2 ]]; then
        echo "[error] --target-fname requires a value" >&2
        exit 1
      fi
      TARGET_FNAME="$2"
      shift 2
      ;;
    --max-items)
      if [[ $# -lt 2 ]]; then
        echo "[error] --max-items requires a value" >&2
        exit 1
      fi
      MAX_ITEMS="$2"
      shift 2
      ;;
    --hough-threshold)
      if [[ $# -lt 2 ]]; then
        echo "[error] --hough-threshold requires a value" >&2
        exit 1
      fi
      HOUGH_THRESHOLD="$2"
      shift 2
      ;;
    --hough-line-length)
      if [[ $# -lt 2 ]]; then
        echo "[error] --hough-line-length requires a value" >&2
        exit 1
      fi
      HOUGH_LINE_LENGTH="$2"
      shift 2
      ;;
    --hough-line-gap)
      if [[ $# -lt 2 ]]; then
        echo "[error] --hough-line-gap requires a value" >&2
        exit 1
      fi
      HOUGH_LINE_GAP="$2"
      shift 2
      ;;
    --hough-seed)
      if [[ $# -lt 2 ]]; then
        echo "[error] --hough-seed requires a value" >&2
        exit 1
      fi
      HOUGH_SEED="$2"
      shift 2
      ;;
    --hough-start)
      if [[ $# -lt 2 ]]; then
        echo "[error] --hough-start requires a value" >&2
        exit 1
      fi
      HOUGH_START="$2"
      shift 2
      ;;
    --levenshtein-backend)
      if [[ $# -lt 2 ]]; then
        echo "[error] --levenshtein-backend requires a value" >&2
        exit 1
      fi
      LEVENSHTEIN_BACKEND="$2"
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

if [[ -z "${RUNFILE_JSON}" ]]; then
  echo "[error] RUNFILE_JSON must not be empty" >&2
  exit 1
fi

if [[ ! -f "${RUNFILE_JSON}" ]]; then
  echo "[error] RUNFILE_JSON does not exist: ${RUNFILE_JSON}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "[error] OUTPUT_DIR must not be empty" >&2
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
  echo "[error] --max-items must be a positive integer when provided (got: ${MAX_ITEMS})" >&2
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

if ! [[ "${HOUGH_SEED}" =~ ^[0-9]+$ ]]; then
  echo "[error] --hough-seed must be a non-negative integer (got: ${HOUGH_SEED})" >&2
  exit 1
fi

if ! [[ "${HOUGH_START}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[error] --hough-start must be a positive number (got: ${HOUGH_START})" >&2
  exit 1
fi

if [[ "${LEVENSHTEIN_BACKEND}" != "python" && "${LEVENSHTEIN_BACKEND}" != "c" ]]; then
  echo "[error] --levenshtein-backend must be one of: python, c (got: ${LEVENSHTEIN_BACKEND})" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[error] python not found in PATH" >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"

echo "[run] Starting levenshtein_along_lines_metric.py"
echo "[run]   runfile_json=${RUNFILE_JSON}"
echo "[run]   output_dir=${OUTPUT_DIR}"
echo "[run]   window_size=${WINDOW_SIZE}"
echo "[run]   window_stride=${WINDOW_STRIDE}"
echo "[run]   target_fname=${TARGET_FNAME:-<all>}"
echo "[run]   max_items=${MAX_ITEMS:-<all>}"
echo "[run]   hough_threshold=${HOUGH_THRESHOLD}"
echo "[run]   hough_line_length=${HOUGH_LINE_LENGTH}"
echo "[run]   hough_line_gap=${HOUGH_LINE_GAP}"
echo "[run]   hough_seed=${HOUGH_SEED}"
echo "[run]   hough_start=${HOUGH_START}"
echo "[run]   levenshtein_backend=${LEVENSHTEIN_BACKEND}"
echo "[run]   python=python"

CMD=(
  python3
  "${SCRIPT_DIR}/levenshtein_along_lines_metric.py"
  --runfile-json "${RUNFILE_JSON}"
  --output-dir "${OUTPUT_DIR}"
  --window-size "${WINDOW_SIZE}"
  --window-stride "${WINDOW_STRIDE}"
  --hough-threshold "${HOUGH_THRESHOLD}"
  --hough-line-length "${HOUGH_LINE_LENGTH}"
  --hough-line-gap "${HOUGH_LINE_GAP}"
  --hough-seed "${HOUGH_SEED}"
  --hough-start "${HOUGH_START}"
  --levenshtein-backend "${LEVENSHTEIN_BACKEND}"
)

if [[ -n "${TARGET_FNAME}" ]]; then
  CMD+=(--target-fname "${TARGET_FNAME}")
fi

if [[ -n "${MAX_ITEMS}" ]]; then
  CMD+=(--max-items "${MAX_ITEMS}")
fi

"${CMD[@]}"

echo "[run] levenshtein_along_lines_metric.py completed successfully."
