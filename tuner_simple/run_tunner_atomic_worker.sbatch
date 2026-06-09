#!/usr/bin/env bash
#SBATCH --job-name=tuner_simple_worker
#SBATCH --account=project_2017385
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy
#SBATCH --output=logs/tuner_simple_worker_%j.out
#SBATCH --error=logs/tuner_simple_worker_%j.err
set -euo pipefail

if type module >/dev/null 2>&1; then
  module --force purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

TUNER_SIMPLE_DIR="${TUNER_SIMPLE_DIR:-/scratch/project_2017385/dorian/Churro_copy/tuner_simple}"
python3 "${TUNER_SIMPLE_DIR}/run_tuner_simple.py" "$@"
