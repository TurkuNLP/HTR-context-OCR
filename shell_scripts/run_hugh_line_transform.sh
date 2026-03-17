#!/usr/bin/env bash
#SBATCH --job-name=hugh_line
#SBATCH --account=project_2000539
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:55:00
#SBATCH --output=/scratch/project_2017385/dorian/Churro_copy/logs/hugh_line_%j.out
#SBATCH --error=/scratch/project_2017385/dorian/Churro_copy/logs/hugh_line_%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

set -euo pipefail
cd /scratch/project_2017385/dorian/Churro_copy
mkdir -p logs

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-25}"
export OMP_NUM_THREADS="$CPUS_PER_TASK"
export MKL_NUM_THREADS="$CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$CPUS_PER_TASK"

unset INCLUDE_FNAME MAX_ITEMS

# Optional: pass results directory as first argument.
if [ "${1:-}" != "" ]; then
  export RESULTS_DIR="$1"
fi

python3 /scratch/project_2017385/dorian/Churro_copy/hugh_line_transform_dev_schuro.py
