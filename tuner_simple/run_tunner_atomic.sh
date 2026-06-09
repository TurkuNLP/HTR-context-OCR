#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_tunner_atomic.sh [launcher options] [tuner options]

Launcher options:
  --worker-count <n>                 Number of Slurm worker jobs to submit
  --account <value>                  Slurm account used for worker and aggregation jobs
  --partition <value>                Slurm partition used for worker jobs
  --time <value>                     Slurm time limit used for worker jobs
  --cpus-per-task <n>                Slurm CPU count used for each worker job
  --mem <value>                      Slurm memory request used for each worker job
  --aggregate-partition <value>      Slurm partition used for the final aggregation job
  --aggregate-time <value>           Slurm time limit used for the final aggregation job
  --aggregate-mem <value>            Slurm memory request used for the final aggregation job
  --slurm-log-dir <path>             Directory where worker and aggregation logs are written
  --resume                           Reuse the existing dynamic document pool
  --requeue-claimed                  During resume, move stale claimed documents back to available
  --retry-failed                     During resume, move failed documents back to available
  --skip-final-aggregation           Submit workers only and do not submit the final CSV/plot aggregation job
  -h, --help                         Show this help text

Important tuner options passed through to each worker:
  --runfile-json <path>
  --output-dir <path>
  --scores-pkl-ref-to-pred <path>
  --scores-pkl-ref-to-ref <path>
  --language <value>                 Repeat for multiple languages
  --all-languages
  --document-type <value>            Repeat for multiple document types
  --all-document-types
  --target-fname <value>             Repeat for multiple exact filenames
  --max-items <n>
  --window-size <n>
  --window-stride <n>
  --minimum-matrix-rows <n>
  --minimum-matrix-columns <n>
  --score-floor-alpha <value>
  --hough-threshold <n>
  --hough-line-length <n>
  --hough-line-gap <n>
  --hough-seed <n>
  --align-min-iou-threshold <value>
  --min-surviving-line-nls <value>
  --plot-mode <value>
  --stitched-panel-columns <n>
  --saved-figure-dpi <n>
  --result-bucket-size <n>
  --result-bucket-seconds <value>
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNFILE_JSON="/scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json"
worker_count="1"
account="project_2017385"
partition="medium"
time_limit="01:30:00"
cpus_per_task="4"
memory_request="24G"
aggregate_partition="medium"
aggregate_time_limit="01:30:00"
aggregate_memory_request="24G"
slurm_log_dir=""
resume_pool="false"
requeue_claimed="false"
retry_failed="false"
skip_final_aggregation="false"
runfile_json="${RUNFILE_JSON:-${DEFAULT_RUNFILE_JSON}}"
output_dir="${OUTPUT_DIR:-}"
window_size="${WINDOW_SIZE:-50}"
window_stride="${WINDOW_STRIDE:-35}"
plot_mode="${PLOT_MODE:-stitched-language}"
stitched_panel_columns="${STITCHED_PANEL_COLUMNS:-3}"
python_arguments=()
pool_selection_arguments=()


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
    echo "[cython] WARNING: accelerator build failed; workers will use the Python fallback" >&2
  fi
}
while [[ $# -gt 0 ]]; do
  current_argument="$1"
  case "${current_argument}" in
    -h|--help)
      usage
      exit 0
      ;;
    --worker-count)
      worker_count="$2"
      shift 2
      ;;
    --account)
      account="$2"
      shift 2
      ;;
    --partition)
      partition="$2"
      shift 2
      ;;
    --time)
      time_limit="$2"
      shift 2
      ;;
    --cpus-per-task)
      cpus_per_task="$2"
      shift 2
      ;;
    --mem)
      memory_request="$2"
      shift 2
      ;;
    --aggregate-partition)
      aggregate_partition="$2"
      shift 2
      ;;
    --aggregate-time)
      aggregate_time_limit="$2"
      shift 2
      ;;
    --aggregate-mem)
      aggregate_memory_request="$2"
      shift 2
      ;;
    --slurm-log-dir)
      slurm_log_dir="$2"
      shift 2
      ;;
    --resume)
      resume_pool="true"
      shift
      ;;
    --requeue-claimed)
      requeue_claimed="true"
      shift
      ;;
    --retry-failed)
      retry_failed="true"
      shift
      ;;
    --skip-final-aggregation)
      skip_final_aggregation="true"
      shift
      ;;
    --runfile-json)
      runfile_json="$2"
      python_arguments+=("$1" "$2")
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      python_arguments+=("$1" "$2")
      shift 2
      ;;
    --language|--document-type|--target-fname|--max-items)
      pool_selection_arguments+=("$1" "$2")
      python_arguments+=("$1" "$2")
      shift 2
      ;;
    --all-languages|--all-document-types)
      pool_selection_arguments+=("$1")
      python_arguments+=("$1")
      shift
      ;;
    --window-size)
      window_size="$2"
      python_arguments+=("$1" "$2")
      shift 2
      ;;
    --window-stride)
      window_stride="$2"
      python_arguments+=("$1" "$2")
      shift 2
      ;;
    --plot-mode)
      plot_mode="$2"
      python_arguments+=("$1" "$2")
      shift 2
      ;;
    --stitched-panel-columns)
      stitched_panel_columns="$2"
      python_arguments+=("$1" "$2")
      shift 2
      ;;
    --*=*)
      option_name="${current_argument%%=*}"
      option_value="${current_argument#*=}"
      python_arguments+=("${current_argument}")
      case "${option_name}" in
        --runfile-json) runfile_json="${option_value}" ;;
        --output-dir) output_dir="${option_value}" ;;
        --window-size) window_size="${option_value}" ;;
        --window-stride) window_stride="${option_value}" ;;
        --plot-mode) plot_mode="${option_value}" ;;
        --stitched-panel-columns) stitched_panel_columns="${option_value}" ;;
        --language|--document-type|--target-fname|--max-items) pool_selection_arguments+=("${current_argument}") ;;
      esac
      shift
      ;;
    *)
      python_arguments+=("${current_argument}")
      shift
      ;;
  esac
done

if [[ -z "${output_dir}" ]]; then
  echo "ERROR: --output-dir is required unless OUTPUT_DIR is set." >&2
  usage >&2
  exit 2
fi

if [[ "${worker_count}" -le 0 ]]; then
  echo "ERROR: --worker-count must be positive." >&2
  exit 2
fi

pool_dir="${output_dir}/dynamic_document_pool"
if [[ -z "${slurm_log_dir}" ]]; then
  slurm_log_dir="${output_dir}/slurm_logs"
fi
mkdir -p "${slurm_log_dir}"


build_cython_accelerators
pool_initialize_arguments=(
  --runfile-json "${runfile_json}"
  --pool-dir "${pool_dir}"
  --window-size "${window_size}"
  --window-stride "${window_stride}"
)
pool_initialize_arguments+=("${pool_selection_arguments[@]}")
if [[ "${resume_pool}" == "true" ]]; then
  pool_initialize_arguments+=(--resume)
fi
if [[ "${requeue_claimed}" == "true" ]]; then
  pool_initialize_arguments+=(--requeue-claimed)
fi
if [[ "${retry_failed}" == "true" ]]; then
  pool_initialize_arguments+=(--retry-failed)
fi

python3 "${SCRIPT_DIR}/dynamic_pool/initialize_document_pool.py" "${pool_initialize_arguments[@]}"

worker_job_ids=()
worker_script="${SCRIPT_DIR}/run_tunner_atomic_worker.sbatch"
for worker_number in $(seq 1 "${worker_count}"); do
  worker_id="worker_${worker_number}"
  submitted_job_id="$(sbatch --parsable \
    --account="${account}" \
    --partition="${partition}" \
    --time="${time_limit}" \
    --cpus-per-task="${cpus_per_task}" \
    --mem="${memory_request}" \
    --job-name="tuner_simple_${worker_id}" \
    --output="${slurm_log_dir}/${worker_id}_%j.out" \
    --error="${slurm_log_dir}/${worker_id}_%j.err" \
    "${worker_script}" \
    "${python_arguments[@]}" \
    --output-dir "${output_dir}" \
    --dynamic-document-pool-dir "${pool_dir}" \
    --dynamic-worker-id "${worker_id}" \
    --atomic-output-dir "${output_dir}")"
  clean_job_id="${submitted_job_id%%;*}"
  worker_job_ids+=("${clean_job_id}")
  echo "submitted worker ${worker_id}: ${submitted_job_id}"
done

if [[ "${skip_final_aggregation}" == "true" ]]; then
  echo "final aggregation was not submitted because --skip-final-aggregation was used"
  exit 0
fi

dependency_job_list="$(IFS=:; echo "${worker_job_ids[*]}")"
aggregate_script="${SCRIPT_DIR}/run_tunner_atomic_aggregate.sbatch"
aggregate_job_id="$(sbatch --parsable \
  --dependency="afterok:${dependency_job_list}" \
  --account="${account}" \
  --partition="${aggregate_partition}" \
  --time="${aggregate_time_limit}" \
  --cpus-per-task="${cpus_per_task}" \
  --mem="${aggregate_memory_request}" \
  --job-name="tuner_simple_aggregate" \
  --output="${slurm_log_dir}/aggregate_%j.out" \
  --error="${slurm_log_dir}/aggregate_%j.err" \
  "${aggregate_script}" \
  --output-dir "${output_dir}" \
  --plot-mode "${plot_mode}" \
  --stitched-panel-columns "${stitched_panel_columns}")"

echo "submitted final aggregation: ${aggregate_job_id}"
echo "progress CSV will be written to: ${output_dir}/progress_csv/document_completion_attempts.csv"
echo "final CSVs and stitched plots will be written to: ${output_dir}"
