#!/usr/bin/env bash
#SBATCH --job-name=tuner_simple_alpha_aggregate
#SBATCH --account=project_2017385
#SBATCH --partition=medium
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --chdir=/scratch/project_2017385/dorian/HTR-context-OCR
#SBATCH --output=logs/tuner_simple_alpha_aggregate_%j.out
#SBATCH --error=logs/tuner_simple_alpha_aggregate_%j.err
set -euo pipefail

if type module >/dev/null 2>&1; then
  module --force purge
  module use /appl/local/csc/modulefiles
  module load pytorch
fi

SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNER_SIMPLE_DIR="${TUNER_SIMPLE_DIR:-${SCRIPT_SOURCE_DIR}}"
python3 "${TUNER_SIMPLE_DIR}/aggregate_dynamic_outputs.py" "$@"
