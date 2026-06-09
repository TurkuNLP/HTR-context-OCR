# tuner_simple atomic worker implementation plan

## Current Decision

`tuner_simple` will stay a simple fixed-parameter pipeline. It will not become a
Hough grid tuner and it will not add Python multiprocessing inside one worker.

The new execution model is:

```text
many Slurm workers
one claimed document per worker at a time
one fixed Hough parameter set
score-floor mask passed directly to Hough
bucketed result rows appended to shared CSV logs
final aggregation for summaries and stitched plots
```

Important terminology:

```text
worker = one Slurm job submitted by the atomic launcher
```

Do not call these workers shards in new code, logs, documentation, or command
help. The old parallel tuner used shard language, but the new simple pipeline is
not doing fixed static shards. Workers dynamically claim documents as they become
free.


## Current tuner_simple Status

The current `tuner_simple` already has these core pieces:

1. Self-contained package:

   ```text
   /scratch/project_2017385/dorian/Churro_copy/tuner_simple
   ```

2. Main Python entry point:

   ```text
   tuner_simple/run_tuner_simple.py
   ```

3. Current shell runner:

   ```text
   tuner_simple/run_tunner.sh
   ```

4. Runfile loading and document selection:

   ```text
   tuner_simple/document_selection/runfile_loader.py
   tuner_simple/document_selection/document_filters.py
   ```

5. Score-matrix loading from pickle streams, with in-memory pickle indexes built
   once per process:

   ```text
   tuner_simple/matrix_operations/matrix_loader.py
   tuner_simple/matrix_operations/score_pkl_index.py
   ```

6. Levenshtein score-matrix fallback computation when a pickle matrix is missing
   or invalid:

   ```text
   tuner_simple/matrix_operations/matrix_fallback_computation.py
   ```

7. Current score floor:

   ```text
   score_floor = score_mean + score_floor_alpha * score_standard_deviation
   ```

   implemented in:

   ```text
   tuner_simple/matrix_operations/score_floor.py
   ```

8. Region of Interest is disabled in active `tuner_simple` code. The final Hough
   input is the score-floor mask directly:

   ```text
   hough_input_mask = score_matrix >= score_floor
   ```

9. One fixed probabilistic Hough parameter set per run:

   ```text
   hough_threshold
   hough_line_length
   hough_line_gap
   hough_seed
   ```

   The Hough seed is the actual integer passed by `--hough-seed`, defaulting to
   `1`. It is not `document_index + 1`.

10. Falling-diagonal Hough detection and line ownership filtering:

    ```text
    tuner_simple/probabilistic_hough/hough_detection.py
    tuner_simple/probabilistic_hough/hough_input.py
    ```

11. Line-level normalized Levenshtein filtering:

    ```text
    tuner_simple/scoring/line_text_similarity.py
    ```

12. Public metrics:

    ```text
    document_normalised_levenshtein
    weighted_along_lines_normalised_levenshtein
    correct_ref_coverage
    missing_ref_coverage
    repetition_on_reference
    hallucination
    ```

13. Current direct-run flat outputs:

    ```text
    tuner_simple/results_writing/flat_csv_tables.py
    ```

14. Current plot rendering and stitched language panels:

    ```text
    tuner_simple/plotting/document_panel_renderer.py
    tuner_simple/plotting/stitched_language_panels.py
    ```

15. Timestamped logging:

    ```text
    tuner_simple/logging_utils/timestamped_logging.py
    ```

16. Current tests:

    ```text
    tuner_simple/tests/
    ```

The missing part is dynamic multi-worker execution with safe shared progress and
resume support.


## Why Atomic Workers Are Needed

A single serial run processes selected documents in order inside one process. If
one document is much slower than the others, the whole run waits behind it.

With atomic workers, the selected documents remain in one ordered queue:

```text
document_000000
document_000001
document_000002
...
```

If there are five workers, the first five available documents are claimed by the
five workers. If two of those workers are still running long documents and three
finish quickly, the three free workers claim the next three available documents.

So with `n` workers:

```text
at launch: next n documents are claimed
while running: next k documents are claimed by the k workers that become free
```

The assignment is dynamic and completion-order driven. This gives better load
balancing while keeping each worker simple.


## Core Architecture

Use two separate mechanisms:

```text
1. Atomic JSON files for scheduling only.
2. Bucketed locked CSV appends for result rows.
```

This is the agreed compromise.

Scheduling should use small JSON files because atomic file rename is simple and
safe for ownership:

```text
available/document_000123.json
  -> claimed/document_000123.worker_004.pid_12345.json
  -> done/document_000123.json
```

Results should not be stored as one JSON file per document. Instead, each worker
collects completed document rows in a small in-memory bucket and periodically
appends those rows to shared CSV logs under a lock.

This gives:

- no thousands of per-document JSON result files;
- fewer disk writes than appending after every document;
- visible progress during the run;
- small bounded memory use;
- safe document ownership;
- clean final CSVs after aggregation;
- resume support after worker or job failure.


## Borrowed Concept from tuner_parallel_v2_2

Borrow only the dynamic document-pool idea from:

```text
tuner_parallel_v2_2/dynamic_pool/document_pool.py
tuner_parallel_v2_2/dynamic_pool/initialize_document_pool.py
```

Do not import those modules. The new implementation must live inside
`tuner_simple`.

The useful idea is:

```text
same-filesystem rename is atomic
```

A worker claims a document by renaming exactly one file from `available/` into
`claimed/`. If another worker tries to claim the same file, the file is already
gone, so that second worker moves to the next available file.


## What Must Not Be Copied

Do not copy these parts of `tuner_parallel_v2_2`:

1. Hough parameter grid.
2. Threshold-worker multiprocessing.
3. Per-node document concurrency.
4. Reference-to-reference combination cache.
5. Region of Interest preprocessing.
6. Combination bundles.
7. Selection objectives such as `tuning_score`.
8. Metrics such as `score_matrix_support` and `line_guided_fraction`.
9. Shard naming in user-facing code.

`tuner_simple` atomic mode is only dynamic scheduling plus bucketed result
writing.


## New File Structure

Add:

```text
tuner_simple/dynamic_pool/
  __init__.py
  document_pool.py
  initialize_document_pool.py
  pool_status.py
```

Add:

```text
tuner_simple/results_writing/locked_csv_bucket.py
```

Add:

```text
tuner_simple/serial_runner/dynamic_worker_runner.py
```

Add:

```text
tuner_simple/results_writing/dynamic_result_aggregation.py
```

Add:

```text
tuner_simple/aggregate_dynamic_outputs.py
```

Add:

```text
tuner_simple/plotting/atomic_panel_writer.py
```

Add shell entry points:

```text
tuner_simple/run_tunner_atomic.sh
tuner_simple/run_tunner_atomic_worker.sbatch
```

Optional later recovery helper:

```text
tuner_simple/dynamic_pool/requeue_claimed_documents.py
```

Do not add it in the first implementation unless resume testing proves it is
needed as a separate command.


## Output Directory Layout

Atomic run output should look like:

```text
<output-dir>/
  atomic_run_manifest.json
  launch_commands.sh
  logs/
  locks/
  document_pool/
    available/
    claimed/
    done/
    failed/
    events.jsonl
    selected_documents.jsonl
    document_pool_manifest.json
  progress_csv/
    document_completion_attempts.csv
  plots/
    document_panels/
      <language>/
        document_000123_<safe-fname>.png
    stitched_best_combination_<language>_documents.png
  best_combination_per_document.csv
  compact_combination_metrics.csv
  loadable_documents.csv
  loaded_documents.csv
  runfile_documents.csv
  skipped_documents.csv
  document_type_summary.csv
  run_summary.json
  aggregation_manifest.json
```

The key difference from the previous plan is:

```text
No per-document result JSON files.
```

The only per-document JSON files are scheduling files in `document_pool/`.


## Document Pool State

Each scheduling file should contain only metadata needed to claim work:

```json
{
  "pool_ordinal": 123,
  "document_index": 456,
  "fname": "some_document.jpeg"
}
```

Definitions:

- `pool_ordinal` is the document position inside the selected list for this run.
- `document_index` is the original index from `outputs.json`.
- `fname` is the basename for readable logs and validation.

Pool states:

```text
available/ = document has not been claimed yet
claimed/   = document is currently owned by one worker
done/      = document result was written and lease was finalized
failed/    = worker marked this document as failed before normal completion
```

A worker must mark a document done only after the document's result row has been
successfully written to the shared progress CSV.


## Shared Result CSV

Use one wide progress CSV:

```text
<output-dir>/progress_csv/document_completion_attempts.csv
```

This CSV is an attempt log. It is not the final clean output table.

It should contain one row for each flushed completed attempt. A resumed run may
create duplicate attempts for the same document if a worker appended the row but
crashed before marking the lease done. That is acceptable because final
aggregation deduplicates by `pool_ordinal`.

Required identity columns:

```text
pool_ordinal
document_index
fname
main_language
document_type
worker_id
attempt_id
slurm_job_id
status
completed_at
```

`status` should be one of:

```text
processed
skipped
failed
```

For normal processed documents, include all fields currently needed for:

```text
best_combination_per_document.csv
compact_combination_metrics.csv
loaded_documents.csv
loadable_documents.csv
```

For skipped documents, include:

```text
skip_stage
skip_reason
row_count
column_count
```

For plotting, include:

```text
panel_path
```

if a panel was rendered.

The progress CSV should also include enough fields to rebuild final outputs
without recomputing matrices, Hough lines, text windows, or metrics.


## Why One Attempt CSV Instead of Many Shared CSVs

Do not let workers append directly to every final CSV.

Bad design:

```text
worker appends to best_combination_per_document.csv
worker appends to loadable_documents.csv
worker appends to loaded_documents.csv
worker appends to skipped_documents.csv
```

That makes partial writes hard to reason about. A worker could append to one file
and crash before appending to another.

Preferred design:

```text
workers append only to document_completion_attempts.csv
final aggregator writes the clean final CSV files
```

This gives one concurrency point and one progress log.


## Locked CSV Appending

Multiple workers must never write the same CSV at the same time.

Use one lock file:

```text
<output-dir>/locks/document_completion_attempts.lock
```

A worker flushes a bucket like this:

```text
acquire lock
open document_completion_attempts.csv in append mode
write header if file does not exist or is empty
append all bucket rows
flush file
fsync file descriptor
close file
release lock
```

The lock must cover only the write. It must not cover document processing,
Hough, scoring, or plotting.

That means while one worker is appending rows, other workers can continue
processing their current documents. They only wait briefly if they also need to
flush at the same time.

Use a real file lock, not a home-grown boolean flag. Recommended implementation:

```python
fcntl.flock(lock_handle, fcntl.LOCK_EX)
```

This is appropriate on Linux/Slurm. If a worker process dies while holding the
lock, the operating system releases the lock when the file descriptor closes.


## Worker Result Bucket

Each worker keeps a small in-memory bucket of completed scalar rows:

```python
bucket: list[dict[str, Any]]
```

The bucket must contain only small serializable values. It must not hold:

- score matrices;
- Hough masks;
- plot payloads;
- raw line arrays beyond the scalar fields already exported;
- image objects.

After each document, release large document-local data before claiming more work.

Recommended flush triggers:

```text
--result-bucket-size <n>
--result-bucket-seconds <seconds>
```

Default recommendation:

```text
--result-bucket-size 20
--result-bucket-seconds 60
```

Flush when either condition is true:

```text
len(bucket) >= result_bucket_size
current_time - last_flush_time >= result_bucket_seconds
```

Always flush before the worker exits.

Do not flush based on vague memory pressure in the first implementation. Memory
pressure is harder to define, harder to test, and less transparent in a paper or
methods section.


## Critical Completion Order

For every completed document, the order must be:

```text
1. claim document
2. process document
3. render panel if plotting is enabled
4. add scalar result row to worker bucket
5. when bucket flushes, append bucket rows under CSV lock
6. after the append succeeds, mark those leases done
7. clear bucket
8. claim more work
```

The central rule is:

```text
Do not mark a lease done before its row is durably appended to the progress CSV.
```

This prevents losing completed documents after a crash.


## Bucket Lease Tracking

Because a bucket can hold multiple completed documents that are not yet marked
done, the worker must keep both the rows and the corresponding leases.

Use a structure like:

```python
@dataclass
class PendingCompletedDocument:
    lease: DocumentLease
    row: dict[str, Any]
```

The flush function receives:

```python
pending_completed_documents: list[PendingCompletedDocument]
```

Flush behavior:

```text
append all rows to progress CSV
for each pending document:
    mark lease done
clear pending list
```

If appending fails, do not mark any of those leases done. Let the exception fail
the worker. The documents remain in `claimed/`, which makes recovery explicit.

If appending succeeds but marking one lease done fails, fail the worker. The CSV
may contain a completed attempt for a lease still in `claimed/`. Resume logic can
handle this by deduplicating attempts and requeueing claimed documents only when
the user requests it.


## Resume Semantics

Resume support must be explicit. Do not silently reuse old output directories.

Add launcher options:

```text
--resume
--requeue-claimed
--retry-failed
```

Default behavior without `--resume`:

```text
refuse to run if document_pool/ already exists
refuse to run if progress_csv/document_completion_attempts.csv already exists
```

Behavior with `--resume` only:

```text
reuse the existing pool and progress CSV
keep done documents done
keep available documents available
refuse to continue if claimed/ is non-empty
refuse to continue if failed/ is non-empty
```

Reason: a non-empty `claimed/` directory may mean old workers are still running.
The user must explicitly say how to treat those documents.

Behavior with:

```text
--resume --requeue-claimed
```

Move all documents from `claimed/` back to `available/` before submitting new
workers. This means the user confirms that old workers are dead or should be
ignored.

Behavior with:

```text
--resume --retry-failed
```

Move all documents from `failed/` back to `available/` before submitting new
workers.

Behavior with:

```text
--resume --requeue-claimed --retry-failed
```

Resume all incomplete or failed documents while keeping `done/` untouched.


## Duplicate Attempts During Resume

A duplicate attempt can happen in this failure window:

```text
worker appends row to CSV
worker crashes before marking lease done
resume requeues claimed document
document is processed again
new row is appended
```

Therefore `document_completion_attempts.csv` must be treated as an attempt log,
not the final truth.

Final aggregation deduplicates by:

```text
pool_ordinal
```

Recommended rule:

```text
for each pool_ordinal, keep the latest processed/skipped attempt that is compatible with pool done state
```

If there are multiple successful attempts for the same `pool_ordinal`, record the
duplicate count in `aggregation_manifest.json`.

If a duplicate row has a different `document_index` or `fname` for the same
`pool_ordinal`, aggregation must fail because that means pool state and progress
CSV do not describe the same selected run.


## Final Aggregation

Final aggregation should run after all workers finish successfully.

The aggregator reads:

```text
<output-dir>/document_pool/document_pool_manifest.json
<output-dir>/document_pool/done/*.json
<output-dir>/document_pool/failed/*.json
<output-dir>/document_pool/claimed/*.json
<output-dir>/document_pool/available/*.json
<output-dir>/progress_csv/document_completion_attempts.csv
```

The aggregator writes:

```text
best_combination_per_document.csv
compact_combination_metrics.csv
loadable_documents.csv
loaded_documents.csv
runfile_documents.csv
skipped_documents.csv
document_type_summary.csv
run_summary.json
aggregation_manifest.json
```

The final CSV files should match the current direct-run output format as closely
as possible.

Aggregation must fail if:

- `available/` is not empty;
- `claimed/` is not empty;
- `failed/` is not empty, unless a future partial aggregation flag is passed;
- a document is marked done but has no successful attempt row;
- an attempt row refers to a `pool_ordinal` not in the pool manifest;
- duplicate attempts disagree on document identity.

Aggregation should write counts to `aggregation_manifest.json`:

```text
selected_document_count
done_count
available_count
claimed_count
failed_count
attempt_row_count
unique_completed_document_count
duplicate_attempt_count
processed_document_count
skipped_document_count
stitched_plot_paths
```


## Plotting in Atomic Mode

Workers should render individual document panels only.

Worker panel path:

```text
<output-dir>/plots/document_panels/<safe-language>/document_<pool_ordinal>_<safe-fname>.png
```

Workers do not stitch language images.

Final aggregation stitches panels by language using the `panel_path` values in
the progress CSV.

If plotting is disabled:

```text
--plot-mode none
```

workers should not import plotting modules, should not render panels, and the
progress CSV should leave `panel_path` empty.

If plotting is enabled and panel rendering fails, the worker should fail the
lease. A plotting failure should not be silently hidden in a scientific audit
run.

Region of Interest plotting must remain disabled.


## Worker Launcher

Add:

```text
tuner_simple/run_tunner_atomic.sh
```

This is a Bash launcher, not an sbatch script.

It should accept all current `run_tunner.sh` scientific options plus:

```text
--worker-count <n>
--account <name>
--partition <name>
--time <HH:MM:SS>
--cpus-per-task <n>
--mem <amount>
--result-bucket-size <n>
--result-bucket-seconds <seconds>
--resume
--requeue-claimed
--retry-failed
--skip-aggregation
```

Recommended defaults:

```text
worker_count = 20
account = project_2017385
partition = medium
time = 24:00:00
cpus_per_task = 4
mem = 48G
result_bucket_size = 20
result_bucket_seconds = 60
aggregation enabled by default
```

The launcher should:

1. Validate inputs.
2. Create or resume the document pool.
3. Write `atomic_run_manifest.json`.
4. Write exact submitted commands to `launch_commands.sh`.
5. Submit `worker_count` Slurm workers.
6. Submit final aggregation with `afterok` dependency unless `--skip-aggregation`
   is passed.

Worker submission should use `--parsable` so job IDs can be collected for the
aggregation dependency.


## Worker sbatch Script

Add:

```text
tuner_simple/run_tunner_atomic_worker.sbatch
```

This script should:

1. Load the same runtime environment as `run_tunner.sh`.
2. Resolve the real `tuner_simple` directory with an absolute fallback.
3. Export low thread counts:

   ```text
   OMP_NUM_THREADS=1
   OPENBLAS_NUM_THREADS=1
   MKL_NUM_THREADS=1
   NUMEXPR_NUM_THREADS=1
   PYTHONUNBUFFERED=1
   ```

4. Run:

   ```text
   python3 tuner_simple/run_tuner_simple.py <forwarded args>
   ```

The document-claim loop must live in Python, not Bash, so the worker can build
runfile data and pickle indexes once and reuse them across claimed documents.


## Python Configuration Additions

Add optional fields to `PipelineConfig`:

```python
dynamic_document_pool_dir: Path | None = None
dynamic_worker_id: str | None = None
atomic_output_dir: Path | None = None
result_bucket_size: int = 20
result_bucket_seconds: float = 60.0
```

Add CLI options:

```text
--dynamic-document-pool-dir <path>
--dynamic-worker-id <value>
--atomic-output-dir <path>
--result-bucket-size <n>
--result-bucket-seconds <seconds>
```

Validation rules:

1. `dynamic_document_pool_dir`, `dynamic_worker_id`, and `atomic_output_dir` must
   be provided together.
2. `result_bucket_size` must be positive.
3. `result_bucket_seconds` must be positive.
4. Direct serial mode must still work without any dynamic options.

`run_tuner_simple.py` should route like this:

```python
if config.dynamic_document_pool_dir is None:
    run_simple_tuner(config, log=log)
else:
    run_atomic_document_worker(config, log=log)
```


## Dynamic Worker Control Flow

The worker should:

1. Load the runfile once.
2. Apply the same selection filters once.
3. Build `document_by_index` once.
4. Build ref-to-pred and ref-to-ref pickle indexes once.
5. Open a `DocumentLeasePool` once.
6. Start an empty pending-completion bucket.
7. Claim one document.
8. Process that document with the existing `process_one_document` function.
9. Render a panel if plotting is enabled.
10. Convert the result into one progress CSV row.
11. Add the row and lease to the pending bucket.
12. Flush the bucket when size or time threshold is reached.
13. Claim the next document.
14. Flush remaining bucket rows before exiting.
15. Exit cleanly when the pool is empty.

A worker must never claim a second document while one document is actively being
processed. It may have multiple completed leases waiting in the pending bucket,
but those documents are already processed and waiting only for batched CSV
flush.


## Failure Handling

If processing a document returns a normal `skipped_row`, write a skipped attempt
row and eventually mark the lease done. That is not an infrastructure failure;
it is a handled document outcome.

If processing raises an unexpected exception after a lease is claimed:

```text
flush any previous pending completed rows first if possible
mark the active lease failed
re-raise the exception
```

If bucket flushing fails:

```text
do not mark bucket leases done
raise exception
worker job fails
leases remain claimed
```

If a worker exits normally:

```text
pending bucket must be empty
active lease must be None
```

If not, raise an error so the worker job fails visibly.


## Resume Flow

A resume run should be explicit and conservative.

Fresh run:

```text
run_tunner_atomic.sh --output-dir NEW_DIR ...
```

Resume without requeue:

```text
run_tunner_atomic.sh --output-dir EXISTING_DIR --resume ...
```

This is allowed only if:

```text
claimed/ is empty
failed/ is empty
```

Resume and requeue claimed documents:

```text
run_tunner_atomic.sh --output-dir EXISTING_DIR --resume --requeue-claimed ...
```

This moves claimed documents back to available. It should log every moved
filename.

Resume and retry failed documents:

```text
run_tunner_atomic.sh --output-dir EXISTING_DIR --resume --retry-failed ...
```

This moves failed documents back to available. It should log every moved
filename.

Resume with both:

```text
run_tunner_atomic.sh --output-dir EXISTING_DIR --resume --requeue-claimed --retry-failed ...
```

This keeps done documents untouched and reruns only incomplete or failed work.

The launcher must validate that the current command parameters match the old
`atomic_run_manifest.json`. It should refuse to resume if scientific parameters
changed, unless a future explicit override is added.

Scientific parameters that must match include:

```text
runfile_json
scores_pkl_ref_to_pred
scores_pkl_ref_to_ref
window_size
window_stride
minimum_matrix_rows
minimum_matrix_columns
score_floor_alpha
hough_threshold
hough_line_length
hough_line_gap
hough_seed
align_min_iou_threshold
min_surviving_line_nls
plot_mode
saved_figure_dpi
```


## Pool Status Helper

Add:

```text
tuner_simple/dynamic_pool/pool_status.py
```

It should print:

```text
selected_document_count
available_count
claimed_count
done_count
failed_count
attempt_row_count
unique_attempted_document_count
```

It should not modify state.

This helps during long runs and before resume.


## Implementation Phases

### Phase 1: Document pool

Implement `tuner_simple/dynamic_pool/document_pool.py`.

Required behavior:

```text
initialize pool
claim next available document atomically
mark lease done
mark lease failed
write events.jsonl
```

Tests:

```text
one worker claims one document
second worker cannot claim same document
empty pool returns None
mark done moves claimed to done
mark failed moves claimed to failed
```


### Phase 2: Pool initializer

Implement `tuner_simple/dynamic_pool/initialize_document_pool.py`.

It must use current `tuner_simple` runfile loading and filtering.

Tests:

```text
language filter respected
document type filter respected
target filename filter respected
max-items respected
existing pool rejected
```


### Phase 3: Locked CSV bucket writer

Implement `tuner_simple/results_writing/locked_csv_bucket.py`.

Required functions:

```python
append_rows_with_file_lock(...)
flush_completed_document_bucket(...)
```

Tests:

```text
header written once
rows append correctly
lock file is used
two simulated writers do not corrupt CSV
fsync is called before success is returned
```


### Phase 4: Progress row builder

Add code that converts `DocumentRunResult` into one wide progress CSV row.

The row must include identity, status, result fields, skip fields, timing fields,
worker id, attempt id, Slurm job id, and optional panel path.

Tests:

```text
processed document row has metric fields
skipped document row has skip fields
row contains pool ordinal and worker id
row contains hough parameters and score-floor values
```


### Phase 5: Worker panel writer

Implement `tuner_simple/plotting/atomic_panel_writer.py`.

It should render one panel to a deterministic path and return that path.

Tests:

```text
path contains pool ordinal
path contains safe language
plot disabled does not import plotting helper
```


### Phase 6: Dynamic worker runner

Implement `tuner_simple/serial_runner/dynamic_worker_runner.py`.

It must claim one document at a time, process it, bucket the row, flush by size
or time, and mark leases done only after successful CSV append.

Tests:

```text
worker processes all documents from small pool
worker exits cleanly when pool empty
bucket flushes at size threshold
bucket flushes at exit
lease not marked done if append fails
```


### Phase 7: CLI and entry routing

Add dynamic options to config parsing and route `run_tuner_simple.py` to either
direct serial mode or dynamic worker mode.

Tests:

```text
direct serial CLI unchanged
dynamic CLI validates required fields
invalid bucket size rejected
invalid bucket seconds rejected
```


### Phase 8: Aggregator

Implement final aggregation from `document_completion_attempts.csv` and pool
state.

Tests:

```text
deduplicates duplicate attempts by pool ordinal
fails on identity conflict
fails when done document has no attempt row
writes final direct-compatible CSVs
writes run_summary.json
stitches language plots when panel paths exist
```


### Phase 9: Launcher and worker sbatch scripts

Implement:

```text
run_tunner_atomic.sh
run_tunner_atomic_worker.sbatch
```

Tests:

```text
bash -n run_tunner_atomic.sh
bash -n run_tunner_atomic_worker.sbatch
launcher writes atomic_run_manifest.json
launcher writes launch_commands.sh
launcher submits requested worker count in test-only mode
```


### Phase 10: Resume tests

Test resume behavior without Slurm first, then with small Slurm jobs.

Tests:

```text
fresh run refuses existing pool
resume refuses claimed documents by default
resume --requeue-claimed moves claimed to available
resume --retry-failed moves failed to available
resume refuses changed scientific parameters
aggregation handles duplicate attempt rows
```


### Phase 11: End-to-end smoke

Run a tiny atomic job with two workers and several target documents.

Required checks:

```text
done count equals selected document count
available count is zero
claimed count is zero
failed count is zero
progress CSV exists
final CSVs exist
stitched language plots exist when plotting enabled
metrics match direct serial run for the same target documents
```


## Future Command Shape

After implementation, a full run should look like:

```bash
bash /scratch/project_2017385/dorian/Churro_copy/tuner_simple/run_tunner_atomic.sh \
  --runfile-json /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json \
  --output-dir /scratch/project_2017385/dorian/Churro_copy/results/tuner_simple_atomic_all_1170 \
  --scores-pkl-ref-to-pred /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/old_scores_reference_prediction_ws50_st35_levenshtein.pkl \
  --scores-pkl-ref-to-ref /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/old_scores_reference_self_ws50_st35_levenshtein.pkl \
  --all-languages \
  --all-document-types \
  --max-items 1170 \
  --window-size 50 \
  --window-stride 35 \
  --minimum-matrix-rows 10 \
  --minimum-matrix-columns 10 \
  --score-floor-alpha 1.8 \
  --hough-threshold 5 \
  --hough-line-length 5 \
  --hough-line-gap 4 \
  --hough-seed 1 \
  --align-min-iou-threshold 0.045 \
  --min-surviving-line-nls 0.65 \
  --plot-mode stitched-language \
  --stitched-panel-columns 6 \
  --saved-figure-dpi 100 \
  --worker-count 20 \
  --result-bucket-size 20 \
  --result-bucket-seconds 60 \
  --account project_2017385 \
  --partition medium \
  --time 24:00:00 \
  --cpus-per-task 4 \
  --mem 48G
```

Resume example:

```bash
bash /scratch/project_2017385/dorian/Churro_copy/tuner_simple/run_tunner_atomic.sh \
  --output-dir /scratch/project_2017385/dorian/Churro_copy/results/tuner_simple_atomic_all_1170 \
  --resume \
  --requeue-claimed \
  --retry-failed \
  --worker-count 20 \
  --account project_2017385
```

The resume command should still require or recover the original scientific
parameters from `atomic_run_manifest.json`. The implementation should choose one
clear rule and document it in `--help`.


## Definition of Done

The refactor is complete when:

1. Direct `run_tunner.sh` still works unchanged.
2. Atomic launcher uses workers, not shards, in user-facing text.
3. Atomic launcher creates or resumes a document pool safely.
4. Workers claim one document at a time.
5. Workers keep only small scalar result rows in the bucket.
6. Workers flush the bucket by size, by time, and at exit.
7. CSV appends are protected by a file lock.
8. Leases are marked done only after successful CSV append.
9. Resume mode can continue from available documents.
10. Resume mode can explicitly requeue claimed documents.
11. Resume mode can explicitly retry failed documents.
12. Aggregation deduplicates duplicate attempts safely.
13. Final CSVs match the current direct-run output format as closely as possible.
14. Final stitched plots are created from worker-rendered panels.
15. No Region of Interest calculation or plotting is reintroduced.
16. No Hough grid is introduced.
17. No Python multiprocessing or thread pool is introduced.
18. Tests cover scheduling, locked appends, bucket flushing, resume, aggregation,
    and direct-mode compatibility.
