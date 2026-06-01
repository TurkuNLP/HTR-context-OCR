#!/usr/bin/env bash
#SBATCH --job-name=churro_tuner_dyn
#SBATCH --account=project_2005072
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_1

set -euo pipefail

timestamp_now() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_info() {
  printf '[%s] %s\n' "$(timestamp_now)" "$*"
}

# Slurm copies submitted sbatch scripts into /var/spool/slurmd/job... before
# execution, so BASH_SOURCE points at the temporary spool copy on compute nodes.
# Keep the real Churro source directory absolute to avoid looking for Python
# files inside Slurm's spool directory.
SCRIPT_DIR="/scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_1"
PROJECT_DIR="/scratch/project_2017385/dorian/Churro_copy"
SHARED_METRICS_DIR="${PROJECT_DIR}/text_metrics_v2_1_parallel"
PYTHON_SCRIPTS_DIR="${PROJECT_DIR}/python_scripts"
LMOD_INIT="${LMOD_INIT:-/usr/share/lmod/8.6.17/init/bash}"
MODULEFILES_DIR="${MODULEFILES_DIR:-/appl/modulefiles}"
PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${SCRIPT_DIR}"

if [[ -f "${LMOD_INIT}" ]]; then
  # Match the normal Churro runtime environment without invoking the original
  # production shell wrapper.
  source "${LMOD_INIT}"
  module use "${MODULEFILES_DIR}"
  module load "${PYTORCH_MODULE}"
elif command -v module >/dev/null 2>&1; then
  module use "${MODULEFILES_DIR}" || true
  module load "${PYTORCH_MODULE}" || true
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ -x /appl/soft/ai/wrap/pytorch-2.9/bin/python3 ]]; then
    PYTHON_BIN="/appl/soft/ai/wrap/pytorch-2.9/bin/python3"
  else
    echo "[error] Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}:${SHARED_METRICS_DIR}:${PYTHON_SCRIPTS_DIR}:${PYTHONPATH:-}"

log_info "[dynamic-worker] started_at=$(date --iso-8601=seconds)"
log_info "[dynamic-worker] host=$(hostname)"
log_info "[dynamic-worker] script_dir=${SCRIPT_DIR}"
log_info "[dynamic-worker] python_bin=${PYTHON_BIN}"
log_info "[dynamic-worker] forwarded_args=$*"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_hough_parameter_sweep.py" "$@"

log_info "[dynamic-worker] finished_at=$(date --iso-8601=seconds)"
