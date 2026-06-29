#!/usr/bin/env bash
#SBATCH --job-name=tuner_simple_Finnish
#SBATCH --account=project_2017385
#SBATCH --partition=medium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --chdir=/scratch/project_2017385/dorian/Churro_copy
#SBATCH --output=logs/tuner_simple_Finnish_%j.out
#SBATCH --error=logs/tuner_simple_Finnish_%j.err
# Run this file with Bash so arrays, functions, and strict error handling behave consistently.

# Fail fast on command errors, unset variables, and failed pipeline stages so broken runs stop clearly.
module purge
module use /appl/local/csc/modulefiles
module load pytorch
set -euo pipefail

# Print the full command-line help text without starting the Python pipeline.
usage() {
  # Use a quoted heredoc so Bash prints the help text exactly as written below.
  cat <<'USAGE'
Usage: run_tunner.sh [options]

Selection options:
  --language <value>                     main_language value to process; repeat for multiple languages
  --all-languages                        Process every main_language value in outputs.json; this is the default
  --document-type <value>                document_type value to process; repeat for multiple types
  --all-document-types                   Process every document_type value in outputs.json; this is the default
  --target-fname <value>                 Exact document filename to process; repeat for multiple documents
  --max-items <n>                        Maximum number of selected documents to process

Input/output options:
  --runfile-json <path>                  Path to outputs.json with document text and metadata
  --output-dir <path>                    Directory where CSV, JSON, and optional PNG files will be written
  --scores-pkl-ref-to-pred <path>        Reference-to-prediction score matrix pickle
  --scores-pkl-ref-to-ref <path>         Reference-to-reference score matrix pickle

Matrix and preprocessing options:
  --window-size <n>                      Number of characters in each score-matrix text window
  --window-stride <n>                    Number of characters between neighboring score-matrix windows
  --minimum-matrix-rows <n>              Minimum reference-window rows required before processing a document
  --minimum-matrix-columns <n>           Minimum prediction-window columns required before processing a document
  --score-floor-alpha <value>            Alpha used when --no-alpha-sweep is set
  --alpha-sweep                         Enable per-document alpha sweep; this is the default
  --no-alpha-sweep                      Disable alpha sweep and use --score-floor-alpha exactly
  --alpha-sweep-min <value>             Inclusive minimum alpha for sweep candidates
  --alpha-sweep-max <value>             Inclusive maximum alpha for sweep candidates
  --alpha-sweep-step <value>            Alpha increment between sweep candidates
  --minimum-pre-hough-levenshtein <v>   Build one fixed Levenshtein pre-Hough mask and skip alpha sweep; 0.30 and 30.0 are equivalent
  --harmonic-mode <value>               Alpha candidate scoring formula: balanced (default), coverage-hallucination-priority, coverage-hallucination-only, or nls-priority
                                        Results are always written to <output-dir>/<harmonic-mode>/ automatically
  --hough-num-runs <n>                  Number of independent ref-to-pred Hough runs per candidate; union of all outputs feeds filtering (default 1)

Hough and filtering options:
  --hough-threshold <n>                  Minimum number of Hough votes needed for a raw line candidate
  --hough-line-length <n>                Minimum accepted probabilistic Hough line length
  --hough-line-gap <n>                   Maximum gap allowed inside a probabilistic Hough line
  --hough-seed <n>                       Integer seed passed to probabilistic Hough for reproducible lines
  --align-min-iou-threshold <value>      Minimum overlap threshold used when assigning windows to lines
  --min-surviving-line-nls <value>       Minimum line-level normalised Levenshtein similarity; <=0 disables it

Plot options:
  --plot-mode <value>                    none, stitched-language, or stitched-language-and-document-grids
  --show-line-ids                        Show raw and surviving line labels on plot overlays
  --stitched-panel-columns <n>           Number of document panels per row in stitched language plots
  --saved-figure-dpi <n>                 DPI for saved PNG figures

Other:
  -h, --help                             Show this help text

Environment variable overrides are supported for scalar paths/settings:
PYTHON_SCRIPT, RUNFILE_JSON, OUTPUT_DIR, SCORES_PKL_REF_TO_PRED,
SCORES_PKL_REF_TO_REF, MAX_ITEMS, WINDOW_SIZE, WINDOW_STRIDE,
MINIMUM_MATRIX_ROWS, MINIMUM_MATRIX_COLUMNS, SCORE_FLOOR_ALPHA,
ALPHA_SWEEP_MIN, ALPHA_SWEEP_MAX, ALPHA_SWEEP_STEP,
MINIMUM_PRE_HOUGH_LEVENSHTEIN, HARMONIC_MODE,
HOUGH_THRESHOLD, HOUGH_LINE_LENGTH, HOUGH_LINE_GAP, HOUGH_SEED, HOUGH_NUM_RUNS,
ALIGN_MIN_IOU_THRESHOLD, MIN_SURVIVING_LINE_NLS, PLOT_MODE, STITCHED_PANEL_COLUMNS, SAVED_FIGURE_DPI
USAGE
}

# Check whether the user asked for wrapper help before any defaults or Python arguments are assembled.
for raw_argument in "$@"; do
  # Match either short or long help flags because both should only print usage text.
  if [[ "${raw_argument}" == "-h" || "${raw_argument}" == "--help" ]]; then
    # Show the wrapper help text and exit successfully without launching Python.
    usage
    # End the script here because help output is the complete requested action.
    exit 0
  fi
# Close the loop that scans user-provided arguments for help flags.
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

# Resolve the real tuner_simple_alpha_sweep_pre_iou_levenshtein directory, even when Slurm runs a copied script from /var/spool/slurmd.
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow advanced users or Slurm wrappers to point directly at the checked-out tuner_simple_alpha_sweep_pre_iou_levenshtein directory.
TUNER_SIMPLE_DIR="${TUNER_SIMPLE_DIR:-}"
# Prefer the explicit directory when it was provided and contains the Python entry point.
if [[ -n "${TUNER_SIMPLE_DIR}" && -f "${TUNER_SIMPLE_DIR}/run_tuner_simple.py" ]]; then
  SCRIPT_DIR="$(cd "${TUNER_SIMPLE_DIR}" && pwd)"
# Use BASH_SOURCE when the script is being run from the real repository checkout.
elif [[ -f "${SCRIPT_SOURCE_DIR}/run_tuner_simple.py" ]]; then
  SCRIPT_DIR="${SCRIPT_SOURCE_DIR}"
# Use the current working directory when the Slurm job was submitted from tuner_simple itself.
elif [[ -f "$(pwd)/run_tuner_simple.py" ]]; then
  SCRIPT_DIR="$(pwd)"
# Fall back to this repository's known absolute location when Slurm only exposes the spool copy.
elif [[ -f "/scratch/project_2017385/dorian/Churro_copy/tuner_simple_alpha_sweep_pre_iou_levenshtein/run_tuner_simple.py" ]]; then
  SCRIPT_DIR="/scratch/project_2017385/dorian/Churro_copy/tuner_simple_alpha_sweep_pre_iou_levenshtein"
else
  echo "ERROR: Could not locate run_tuner_simple.py. Set TUNER_SIMPLE_DIR=/path/to/tuner_simple_alpha_sweep_pre_iou_levenshtein." >&2
  exit 2
fi
# Pick the Python entry point, allowing PYTHON_SCRIPT to override it for advanced debugging.
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${SCRIPT_DIR}/run_tuner_simple.py}"


build_cython_accelerators() {
  if [[ "${TUNER_SIMPLE_SKIP_CYTHON_BUILD:-0}" == "1" ]]; then
    echo "[cython] skipped because TUNER_SIMPLE_SKIP_CYTHON_BUILD=1"
    return 0
  fi
  if [[ ! -f "${SCRIPT_DIR}/cython_accel/build.py" ]]; then
    echo "[cython] skipped because ${SCRIPT_DIR}/cython_accel/build.py was not found"
    return 0
  fi
  echo "[cython] building tuner_simple accelerators in ${SCRIPT_DIR}"
  if (cd "${SCRIPT_DIR}" && python3 cython_accel/build.py build_ext --inplace >/dev/null); then
    echo "[cython] tuner_simple accelerators are ready"
  else
    echo "[cython] WARNING: accelerator build failed; Python fallback will be used" >&2
  fi
}

build_cython_accelerators
# Start the Python argument list with the user arguments exactly as they were passed to this wrapper.
python_arguments=("$@")

# Return success when an option already appears in the user-provided argument list.
has_option() {
  # Store the option name we are looking for, such as --output-dir or --hough-threshold.
  local option_name="$1"
  # Scan every current Python argument because user-supplied values must override environment defaults.
  for current_argument in "${python_arguments[@]}"; do
    # Match the exact option token and the --option=value form accepted by argparse.
    if [[ "${current_argument}" == "${option_name}" || "${current_argument}" == "${option_name}="* ]]; then
      # Report that the option is already present so no environment fallback is appended.
      return 0
    fi
  done
  # Report that the option is absent so the caller may append an environment fallback.
  return 1
}

# Append an option from an environment variable only when the user did not already pass that option.
append_env_default() {
  # Store the Python option name that should be appended, for example --runfile-json.
  local option_name="$1"
  # Store the environment variable name that may contain the default value.
  local environment_name="$2"
  # Read the environment variable indirectly while treating an unset variable as an empty string.
  local environment_value="${!environment_name:-}"
  # Skip empty environment values because there is no default to append.
  if [[ -z "${environment_value}" ]]; then
    # Return without changing arguments when the environment variable is unset or empty.
    return 0
  fi
  # Skip appending when the user already provided this option explicitly on the command line.
  if has_option "${option_name}"; then
    # Keep the explicit user argument as the authoritative value.
    return 0
  fi
  # Append the option and its value as separate array elements so paths with spaces remain safe.
  python_arguments+=("${option_name}" "${environment_value}")
}

# Accept --all-languages and --all-document-types as wrapper-level no-op flags because processing all is default.
filtered_python_arguments=()
# Inspect each argument before Python sees it so wrapper-only flags do not reach argparse.
for current_argument in "${python_arguments[@]}"; do
  # Drop --all-languages because omitting --language already means every language is selected.
  if [[ "${current_argument}" == "--all-languages" ]]; then
    # Continue without appending this wrapper-only flag to the Python argument list.
    continue
  fi
  # Drop --all-document-types because omitting --document-type already means every type is selected.
  if [[ "${current_argument}" == "--all-document-types" ]]; then
    # Continue without appending this wrapper-only flag to the Python argument list.
    continue
  fi
  # Preserve every ordinary Python option and value exactly as the user provided it.
  filtered_python_arguments+=("${current_argument}")
# Close the loop that filters wrapper-only arguments.
done
# Replace the argument list with the filtered version before appending environment defaults.
python_arguments=("${filtered_python_arguments[@]}")

# Add scalar path and setting defaults from environment variables when the command line omits them.
append_env_default "--runfile-json" "RUNFILE_JSON"
# Add the output directory from OUTPUT_DIR when --output-dir is not already present.
append_env_default "--output-dir" "OUTPUT_DIR"
# Add the reference-to-prediction score pickle from the environment when requested.
append_env_default "--scores-pkl-ref-to-pred" "SCORES_PKL_REF_TO_PRED"
# Add the reference-to-reference score pickle from the environment when requested.
append_env_default "--scores-pkl-ref-to-ref" "SCORES_PKL_REF_TO_REF"
# Add the optional document limit from the environment when requested.
append_env_default "--max-items" "MAX_ITEMS"
# Add matrix window defaults from the environment when requested.
append_env_default "--window-size" "WINDOW_SIZE"
# Add matrix stride defaults from the environment when requested.
append_env_default "--window-stride" "WINDOW_STRIDE"
# Add minimum row-count defaults from the environment when requested.
append_env_default "--minimum-matrix-rows" "MINIMUM_MATRIX_ROWS"
# Add minimum column-count defaults from the environment when requested.
append_env_default "--minimum-matrix-columns" "MINIMUM_MATRIX_COLUMNS"
# Add score-floor alpha defaults from the environment when requested.
append_env_default "--score-floor-alpha" "SCORE_FLOOR_ALPHA"
# Add alpha sweep defaults from the environment when requested.
append_env_default "--alpha-sweep-min" "ALPHA_SWEEP_MIN"
append_env_default "--alpha-sweep-max" "ALPHA_SWEEP_MAX"
append_env_default "--alpha-sweep-step" "ALPHA_SWEEP_STEP"
append_env_default "--minimum-pre-hough-levenshtein" "MINIMUM_PRE_HOUGH_LEVENSHTEIN"
# Add harmonic mode defaults from the environment when requested.
append_env_default "--harmonic-mode" "HARMONIC_MODE"
# Add Hough threshold defaults from the environment when requested.
append_env_default "--hough-threshold" "HOUGH_THRESHOLD"
# Add Hough line-length defaults from the environment when requested.
append_env_default "--hough-line-length" "HOUGH_LINE_LENGTH"
# Add Hough line-gap defaults from the environment when requested.
append_env_default "--hough-line-gap" "HOUGH_LINE_GAP"
# Add Hough seed defaults from the environment when requested.
append_env_default "--hough-seed" "HOUGH_SEED"
# Add Hough multi-run count defaults from the environment when requested.
append_env_default "--hough-num-runs" "HOUGH_NUM_RUNS"
# Add line-overlap threshold defaults from the environment when requested.
append_env_default "--align-min-iou-threshold" "ALIGN_MIN_IOU_THRESHOLD"
# Add line-level similarity filter defaults from the environment when requested.
append_env_default "--min-surviving-line-nls" "MIN_SURVIVING_LINE_NLS"
# Add plotting-mode defaults from the environment when requested.
append_env_default "--plot-mode" "PLOT_MODE"
# Add stitched panel column defaults from the environment when requested.
append_env_default "--stitched-panel-columns" "STITCHED_PANEL_COLUMNS"
# Add saved figure resolution defaults from the environment when requested.
append_env_default "--saved-figure-dpi" "SAVED_FIGURE_DPI"

# Require an output directory because the Python CLI needs a concrete place for result files.
if ! has_option "--output-dir"; then
  # Print a clear error before the usage text so the missing required option is obvious.
  echo "ERROR: --output-dir is required unless OUTPUT_DIR is set." >&2
  # Show the full usage text to make the correction easy.
  usage >&2
  # Exit with a command-line error code because required input is missing.
  exit 2
fi

# Launch the simple tuner Python entry point with the assembled argument list.
python3 "${PYTHON_SCRIPT}" "${python_arguments[@]}"
