#!/usr/bin/env bash
#SBATCH --job-name=compare_ref_to_ref_py
#SBATCH --account=project_2000539
#SBATCH --partition=medium
#SBATCH --time=6:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=16G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy
#SBATCH -o logs/eval_improvement_v2_%j.out
#SBATCH -e logs/eval_improvement_v2_%j.err
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
OUTPUT="${OUTPUT:-/scratch/project_2017385/dorian/Churro_copy/results/compare_ref_to_ref/churro_finnish_dataset_dev_split/scores_reference_self_ws50_st35.pkl}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-35}"

usage() {
  cat <<'USAGE'
Usage: run_compare_reference_self.sh [options]

Options:
  --runfile-json <path>      Path to outputs.json to compare
  --output <path>            Output pickle file path
  --window-size <n>          Sliding window size in characters
  --window-stride <n>        Sliding window stride in characters
  -h, --help                 Show this help text

Environment variable overrides are also supported for all options:
RUNFILE_JSON, OUTPUT, WINDOW_SIZE, WINDOW_STRIDE
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
    --output)
      if [[ $# -lt 2 ]]; then
        echo "[error] --output requires a value" >&2
        exit 1
      fi
      OUTPUT="$2"
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

if [[ -z "${OUTPUT}" ]]; then
  echo "[error] OUTPUT must not be empty" >&2
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

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 not found in PATH" >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs" "$(dirname "${OUTPUT}")"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"

echo "[run] Starting compare_reference_self.py"
echo "[run]   runfile_json=${RUNFILE_JSON}"
echo "[run]   output=${OUTPUT}"
echo "[run]   window_size=${WINDOW_SIZE}"
echo "[run]   window_stride=${WINDOW_STRIDE}"
echo "[run]   python=python3"

python3 compare_reference_self.py \
  --runfile-json "${RUNFILE_JSON}" \
  --output "${OUTPUT}" \
  --window-size "${WINDOW_SIZE}" \
  --window-stride "${WINDOW_STRIDE}"

echo "[run] compare_reference_self.py completed successfully."
