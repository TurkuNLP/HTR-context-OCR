#!/usr/bin/env bash
#SBATCH --job-name=compare_ref_to_ref_py
#SBATCH --account=project_2017385
#SBATCH --partition=medium
#SBATCH --time=6:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -o logs/compare_ref_to_ref_%j.out
#SBATCH -e logs/compare_ref_to_ref_%j.err
set -euo pipefail

find_repo_root_from() {
  local search_dir="$1"
  while [[ -n "${search_dir}" && "${search_dir}" != "/" ]]; do
    if [[ -f "${search_dir}/shell_scripts/run_compare.sh" && -f "${search_dir}/python_scripts/compare.py" ]]; then
      printf '%s\n' "${search_dir}"
      return 0
    fi
    search_dir="$(dirname "${search_dir}")"
  done
  return 1
}

resolve_repo_root() {
  local script_source_dir
  script_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -n "${HTR_CONTEXT_OCR_DIR:-}" && -f "${HTR_CONTEXT_OCR_DIR}/shell_scripts/run_compare.sh" ]]; then
    printf '%s\n' "$(cd "${HTR_CONTEXT_OCR_DIR}" && pwd)"
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && find_repo_root_from "${SLURM_SUBMIT_DIR}"; then
    return 0
  fi
  if find_repo_root_from "${script_source_dir}"; then
    return 0
  fi
  if find_repo_root_from "$(pwd)"; then
    return 0
  fi
  return 1
}

usage() {
  cat <<'USAGE'
Usage: run_compare_reference_self.sh [options]

This is a convenience wrapper around run_compare.sh that forces:
  --comparison-mode ref-ref
  --metric chrf

Options:
  --runfile-json <path>      Path to outputs.json to compare
  --output <path>            Output pickle file path
  --window-size <n>          Sliding window size in characters
  --window-stride <n>        Sliding window stride in characters
  --max-items <n>            Process only the first N documents
  --num-documents <n>        Alias for --max-items
  --graph-output-dir <path>  Directory for score matrix PNG graphs
  -h, --help                 Show this help text

Environment variable overrides are also supported:
RUNFILE_JSON, OUTPUT, WINDOW_SIZE, WINDOW_STRIDE, MAX_ITEMS, GRAPH_OUTPUT_DIR, HTR_CONTEXT_OCR_DIR
USAGE
}

for raw_argument in "$@"; do
  if [[ "${raw_argument}" == "-h" || "${raw_argument}" == "--help" ]]; then
    usage
    exit 0
  fi
done

REPO_ROOT="$(resolve_repo_root)" || {
  echo "[error] Could not locate the HTR-context-OCR repo root. Submit from inside the repo or set HTR_CONTEXT_OCR_DIR." >&2
  exit 2
}

compare_arguments=("$@")

has_option() {
  local option_name="$1"
  for current_argument in "${compare_arguments[@]}"; do
    if [[ "${current_argument}" == "${option_name}" || "${current_argument}" == "${option_name}="* ]]; then
      return 0
    fi
  done
  return 1
}

if ! has_option "--runfile-json"; then
  compare_arguments+=(--runfile-json "${RUNFILE_JSON:-${REPO_ROOT}/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json}")
fi
if ! has_option "--output"; then
  compare_arguments+=(--output "${OUTPUT:-${REPO_ROOT}/results/compare_ref_to_ref/scores_reference_self.pkl}")
fi
if ! has_option "--window-size"; then
  compare_arguments+=(--window-size "${WINDOW_SIZE:-50}")
fi
if ! has_option "--window-stride"; then
  compare_arguments+=(--window-stride "${WINDOW_STRIDE:-35}")
fi
if [[ -n "${MAX_ITEMS:-}" ]] && ! has_option "--max-items" && ! has_option "--num-documents"; then
  compare_arguments+=(--max-items "${MAX_ITEMS}")
fi
if [[ -n "${GRAPH_OUTPUT_DIR:-}" ]] && ! has_option "--graph-output-dir"; then
  compare_arguments+=(--graph-output-dir "${GRAPH_OUTPUT_DIR}")
fi

compare_arguments+=(--comparison-mode ref-ref --metric chrf)

bash "${REPO_ROOT}/shell_scripts/run_compare.sh" "${compare_arguments[@]}"
