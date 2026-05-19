# tuner_parallel_v2_1

`tuner_parallel_v2_1` is the dynamic-document-pool version of the Hough Line Transform parameter tuner.

It was copied from `tuner_parallel_v2/` and keeps the same scoring, Hough detection, filtering, matrix loading, cache usage, best-combination selection, combination-bundle format, and visualisation logic.  The intended behavior change is only scheduling: documents are no longer assigned to fixed static shards before the run starts.

## Only Production Entry Point

Run the pipeline through this script only:

```bash
/scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_1/run_hough_parameter_sweep_20nodes_10docs_each.sh
```

The script name is kept for continuity with the previous workflow, but its behavior is now dynamic.  `--shard-count` means “how many identical sbatch workers to submit,” not “how many fixed document ranges to create.”

`--docs-per-shard` is accepted only for compatibility and is ignored by dynamic scheduling.

## What Changed From tuner_parallel_v2

Old behavior:

```text
worker 0 receives selected documents 0..29
worker 1 receives selected documents 30..59
worker 2 receives selected documents 60..89
```

New behavior:

```text
all workers read from one shared document pool
any worker with a free local document slot claims the next available document
when one document finishes, only that freed slot claims one replacement
```

The worker does not wait for all currently active documents on the node to finish.

Example with five active document slots:

```text
active documents: A, B, C, D, E

C finishes
C's slot immediately claims F

active documents: A, B, D, E, F
```

This removes the static-shard tail problem where one slow shard can keep a node running while other nodes are idle.

## Result-Preservation Rule

The dynamic pool must not change scientific results.

These remain unchanged:

```text
.pkl score-matrix loading
score-index cache usage
matrix cache behavior
ref-to-ref cache behavior
Hough parameter grid
fixed hough_seed behavior
falling-only Hough direction
raw_hough_to_true_iou filtering
v2.12 coverage and hallucination semantics
weighted along-lines Levenshtein
harmonic tuning_score
best-combination selection
combination-bundle schema
final visualisation content
```

If the same document, matrices, and Hough parameters are evaluated, the per-document metric values must match the previous implementation.

## Dynamic Pool Layout

A run creates this scheduling-only pool:

```text
<output-dir>/document_pool/
  selected_documents.jsonl
  document_pool_manifest.json
  available/
  claimed/
  done/
  failed/
  events.jsonl
```

Each pool file contains only document identity needed for scheduling:

```json
{
  "pool_ordinal": 42,
  "runfile_index": 138,
  "fname": "example_document.jpeg"
}
```

The pool does not store metrics.

The pool does not store best combinations.

The pool does not store combination bundles.

Those outputs remain in the normal tuner files written under each worker output directory.

## Normal Output Layout

A dynamic run writes:

```text
<output-dir>/
  document_pool/
  dynamic_pool_manifest.json
  launch_commands.sh
  logs/
  shards/
    dynamic_worker_000/
      hough_parameter_sweep_summary.json
      best_params_per_document.json
      csv/
        all_documents_parameter_influence.csv
        best_config_per_document.csv
        hough_threshold_summary.csv
        hough_line_length_summary.csv
        hough_line_gap_summary.csv
        hough_seed_summary.csv
        invalid_combinations.csv
      combination_bundles/
    dynamic_worker_001/
      ...same existing tuner output structure...
  plots/
```

The `shards/dynamic_worker_*/` folders use the existing tuner output code.  They are named as dynamic workers only so multiple sbatch jobs do not overwrite each other.

## How Documents Are Claimed

Each available document starts as:

```text
document_pool/available/document_000042.json
```

A worker claims the document by atomically moving it to:

```text
document_pool/claimed/document_000042.worker_000.pid_12345.json
```

The move is atomic on the same filesystem.  Only one worker can claim the file.

After the worker successfully writes the normal tuner outputs, completed leases move to:

```text
document_pool/done/document_000042.json
```

If a worker fails before its normal tuner outputs are safely written, its leases move to:

```text
document_pool/failed/document_000042.json
```

This avoids a false `done` state for documents whose metrics were not exported.

## Per-Node Capacity

The launcher computes safe document concurrency from the Hough threshold range:

```text
threshold_count = threshold_end - threshold_start + 1
node_document_capacity = floor(cpus_per_task / threshold_count)
```

Examples for a 128-core node:

```text
threshold 12..35 -> 24 threshold values -> floor(128 / 24) = 5 documents
threshold  5..40 -> 36 threshold values -> floor(128 / 36) = 3 documents
threshold 10..35 -> 26 threshold values -> floor(128 / 26) = 4 documents
```

`--doc-workers` is an upper cap.  If it is higher than the computed node capacity, the launcher and Python worker lower it automatically.

Dynamic workers pass the active threshold count as the threshold-worker count per document.  That preserves the existing global-threshold scheduler behavior.

## Visualisation

Visualisation is not part of per-document scheduling.

When `--with-visuals` is passed, the launcher submits one final visualisation job with an `afterok` dependency on all dynamic workers.  That means visualisation starts only after all worker jobs have completed successfully.

The visualisation code reads the same existing files as before:

```text
shards/dynamic_worker_*/combination_bundles/document_*/document_metadata.json
shards/dynamic_worker_*/combination_bundles/document_*/threshold_*.jsonl
```

The dynamic-pool implementation only added path discovery for `dynamic_worker_*` folders.  It did not change bundle contents or metric interpretation.

## Important Scripts

### `run_hough_parameter_sweep_20nodes_10docs_each.sh`

The only production entry point.

Responsibilities:

```text
parse user arguments
validate Hough ranges and Slurm settings
compute per-node document capacity
create the shared document pool
submit N identical sbatch worker jobs
submit final visualisation job after all workers if --with-visuals is passed
write launch_commands.sh for reproducibility
write dynamic_pool_manifest.json for run provenance
```

This script no longer assigns document index ranges to workers.

### `run_hough_parameter_sweep_shard.sbatch`

Slurm worker wrapper.

Responsibilities:

```text
load the pytorch module
set Python runtime environment variables
set PYTHONPATH for the copied tuner package
call run_hough_parameter_sweep.py with the forwarded worker arguments
```

The filename is kept for compatibility, but the job is now a dynamic worker, not a static shard.

### `run_hough_parameter_sweep.py`

Python tuner entry point.

Static behavior remains available when `--dynamic-document-pool-dir` is not passed.

Dynamic behavior is enabled by:

```text
--dynamic-document-pool-dir <pool-dir>
--dynamic-worker-id <worker-id>
--dynamic-cpus-per-task <cpu-count>
```

Responsibilities in dynamic mode:

```text
load the same selected runfile items as the pool initializer
create a DocumentLeasePool view of the shared pool
create a lazy selected-item iterator that claims one document at a time
pass that iterator into the existing tuner core
record completed leases in memory when documents finish
mark completed leases done only after normal tuner outputs are written
mark leases failed if the worker crashes before normal output export completes
```

### `run_language_hough_parameter_metric_analysis.sh`

Final visualisation wrapper.

Responsibilities:

```text
run after all dynamic workers finish
load language/document_type metadata from outputs.json
read existing combination bundle files
write final plots and JSON summaries under plots/
```

## New Dynamic Pool Package

### `dynamic_pool/document_pool.py`

Scheduling-only file-backed pool implementation.

Important functions and classes:

```text
DocumentLease
  Dataclass for one claimed document.  Stores pool_ordinal, runfile_index,
  fname, worker_id, claimed_path, and claimed_at.

DocumentLeasePool
  Worker-side queue object.  Claims documents from available/, writes scheduling
  events, and moves leases to done/ or failed/.

initialize_document_pool()
  Creates selected_documents.jsonl and available/document_*.json from selected
  runfile items.  It fails if the pool already exists, because silently reusing
  old scheduling state could skip documents.

iter_claimed_selected_run_items_from_pool()
  Lazy iterator used by the existing scheduler.  It claims exactly one document
  each time the scheduler requests another selected run item.
```

### `dynamic_pool/initialize_document_pool.py`

Lightweight internal CLI used by the launcher.

Responsibilities:

```text
call the existing select_run_items_for_tuning() helper
create the scheduling pool
exit before any matrix, Hough, filtering, metric, or visualisation work happens
```

## Existing Core Modules Reused

### `matrices/runfile_selection.py`

Keeps the existing document selection semantics.  Dynamic pooling uses this same selector, so `--max-items` selects the same document set as before.

### `matrices/document_prep.py`

Still prepares score matrices, text blocks, whole-document NLS, and Hough contexts.  Dynamic mode feeds this module a lazy claimed-document iterator instead of a fixed list.

### `tuner/tuner_core.py`

Still orchestrates one tuner run and writes the existing JSON/CSV outputs.  The dynamic implementation added only an optional `selected_run_items_override` so already-claimed documents can reuse the same preparation and sweep code.

### `tuner/sweep_scheduler.py`

Still owns document/threshold scheduling.  The existing global-threshold queue already refills local document slots when one document finishes.  Dynamic mode added an optional `on_document_completed` callback so the worker can track completed leases without storing metrics in the pool.

### `tools/language_hough_parameter_metric_analysis.py`

Still creates graph grids, stitched panels, and language/document_type JSON summaries from combination bundles.  Dynamic mode added discovery of `shards/dynamic_*/combination_bundles` directories.

## Example Command

This submits 20 dynamic sbatch workers for the first 200 selected documents, using threshold `12..35`, line length `5..25`, line gap `0..15`, the existing pkl score streams, and final visuals:

```bash
sbatch /scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_1/run_hough_parameter_sweep_20nodes_10docs_each.sh \
  --runfile-json /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json \
  --output-dir /scratch/project_2000539/dorian/results/tuner_parallel_v2_1_docs200_dynamic_pool_th12_35_len5_25_gap0_15 \
  --max-items 200 \
  --shard-count 20 \
  --hough-threshold-range 12 35 \
  --line-length-range 5 25 \
  --line-gap-range 0 15 \
  --cpus-per-task 128 \
  --mem 64G \
  --with-visuals
```

For threshold `12..35`, there are 24 threshold values.  With `--cpus-per-task 128`, each worker can keep up to five documents active at the same time.

## Monitoring A Running Dynamic Pool

Count free documents:

```bash
find <output-dir>/document_pool/available -name 'document_*.json' | wc -l
```

Count currently claimed documents:

```bash
find <output-dir>/document_pool/claimed -name 'document_*.json' | wc -l
```

Count successfully completed documents:

```bash
find <output-dir>/document_pool/done -name 'document_*.json' | wc -l
```

Count failed documents:

```bash
find <output-dir>/document_pool/failed -name 'document_*.json' | wc -l
```

Inspect scheduling events:

```bash
tail -n 50 <output-dir>/document_pool/events.jsonl
```

## Testing Performed During Implementation

The implementation was checked with:

```text
Python compile check for tuner_parallel_v2_1/
Shell syntax check for the launcher and Slurm wrappers
CLI --help smoke check for the launcher
CLI --help smoke check for run_hough_parameter_sweep.py
Scheduling-only pool smoke test with two worker views and three fake documents
Real-runfile pool initialization smoke test with --max-items 1
Tiny dynamic end-to-end run with one document and a 1x1x1 Hough grid
```

The tiny dynamic run wrote the existing normal files:

```text
hough_parameter_sweep_summary.json
best_params_per_document.json
csv/all_documents_parameter_influence.csv
csv/best_config_per_document.csv
csv/hough_threshold_summary.csv
csv/hough_line_length_summary.csv
csv/hough_line_gap_summary.csv
csv/hough_seed_summary.csv
csv/invalid_combinations.csv
```
