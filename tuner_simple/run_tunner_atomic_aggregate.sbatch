#!/usr/bin/env bash
#SBATCH --job-name=tuner_simple_aggregate
#SBATCH --account=project_2017385
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy
#SBATCH --output=logs/tuner_simple_aggregate_%j.out
#SBATCH --error=logs/tuner_simple_aggregate_%j.err
set -euo pipefail

if type module >/dev/null 2>&1; then
  module --force purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

TUNER_SIMPLE_DIR="${TUNER_SIMPLE_DIR:-/scratch/project_2017385/dorian/Churro_copy/tuner_simple}"
python3 "${TUNER_SIMPLE_DIR}/aggregate_dynamic_outputs.py" "$@"
