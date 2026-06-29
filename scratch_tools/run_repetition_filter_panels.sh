#!/usr/bin/env bash
#SBATCH --job-name=repetition_filter_panel_viewer
#SBATCH --account=project_2017385
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy
#SBATCH --output=logs/repetition_filter_panel_viewer%j.out
#SBATCH --error=logs/repetition_filter_panel_viewer%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_repetition_filter_panels.sh [options]

  --results-dir <path>        balanced/ directory containing
                              best_combination_per_document.csv (required)
  --min-repetition <float>    Include documents where
                              repetition_on_reference > value (default 0.0)
  --output-dir <path>         Directory for panels and stitched PNGs
                              (default: <results-dir>/plots/repetition_filter_min<N>/)
  --panel-columns <n>         Columns in the stitched contact sheet (default 3)
  --dpi <n>                   Figure DPI for saved panels (default 120)
  --world-readable            chmod output files o+r, directories o+x
  -h, --help                  Show this help text

Environment variable overrides:
  RESULTS_DIR, MIN_REPETITION, OUTPUT_DIR, PANEL_COLUMNS, DPI
USAGE
}

for raw_argument in "$@"; do
  if [[ "${raw_argument}" == "-h" || "${raw_argument}" == "--help" ]]; then
    usage
    exit 0
  fi
done

if type module >/dev/null 2>&1; then
  module --force purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Resolve the script directory even when Slurm runs a copied script from /var/spool/slurmd.
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_SOURCE_DIR}"

PYTHON_SCRIPT="${PYTHON_SCRIPT:-${SCRIPT_DIR}/repetition_filter_panels.py}"
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  PYTHON_SCRIPT="/scratch/project_2017385/dorian/Churro_copy/scratch_tools/repetition_filter_panels.py"
fi
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "ERROR: Could not locate repetition_filter_panels.py." >&2
  exit 2
fi

python_arguments=("$@")

has_option() {
  local option_name="$1"
  for current_argument in "${python_arguments[@]}"; do
    if [[ "${current_argument}" == "${option_name}" || "${current_argument}" == "${option_name}="* ]]; then
      return 0
    fi
  done
  return 1
}

append_env_default() {
  local option_name="$1"
  local environment_name="$2"
  local environment_value="${!environment_name:-}"
  if [[ -z "${environment_value}" ]]; then
    return 0
  fi
  if has_option "${option_name}"; then
    return 0
  fi
  python_arguments+=("${option_name}" "${environment_value}")
}

append_env_default "--results-dir"    "RESULTS_DIR"
append_env_default "--min-repetition" "MIN_REPETITION"
append_env_default "--output-dir"     "OUTPUT_DIR"
append_env_default "--panel-columns"  "PANEL_COLUMNS"
append_env_default "--dpi"            "DPI"

if ! has_option "--results-dir"; then
  echo "ERROR: --results-dir is required (or set RESULTS_DIR env var)." >&2
  echo "       Example: sbatch run_repetition_filter_panels.sh --results-dir results/.../balanced" >&2
  exit 2
fi

mkdir -p logs

python3 "${PYTHON_SCRIPT}" "${python_arguments[@]}"
