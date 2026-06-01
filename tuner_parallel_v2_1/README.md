# tuner_parallel_v2_1

`tuner_parallel_v2_1` is the current Hough Line Transform parameter tuner for score-matrix based document alignment experiments.

The tuner takes document records from an `outputs.json` runfile, obtains two score matrices for each selected document, sweeps Hough parameters, filters raw Hough segments into final line owners, computes text and coverage metrics, and selects the best Hough parameter combination per document.

The two matrix directions are:

- `ref_to_pred`: reference text windows on the y axis, prediction text windows on the x axis. This is the direction whose final line alignment is judged against the prediction.
- `ref_to_ref`: reference text windows on both axes. This is the reference-self coverage baseline used by the v2.12 coverage and hallucination metric logic.

The scientific result of one combination is a row containing line-level weighted normalized Levenshtein similarity, v2.12 coverage ratios, hallucination/repetition, and a harmonic `tuning_score`. The best row for a document is selected by a deterministic ranking key.

## Quick Start: Production Slurm Run

Use this shell launcher for normal production runs:

```bash
bash /scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_1/run_hough_parameter_sweep_20nodes_10docs_each.sh \
  --runfile-json /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_full1170_run1/vllm/dev/outputs.json \
  --output-dir /scratch/project_2000539/dorian/results/tuner_parallel_v2_1_all_docs_dynamic_pool_20shards_th10_35_len5_25_gap0_15 \
  --max-items 1170 \
  --shard-count 20 \
  --hough-threshold-range 10 35 \
  --line-length-range 5 25 \
  --line-gap-range 0 15 \
  --scores-pkl-ref-to-pred /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/scores_reference_prediction_ws50_st35.pkl \
  --scores-pkl-ref-to-ref /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/scores_reference_self_ws50_st35.pkl \
  --score-index-cache-dir /scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_1_parallel/.score_index_cache \
  --use-matrix-cache \
  --matrix-cache-dir /scratch/project_2000539/dorian/cache/tuner_parallel_v2_1/matrix_npz_cache_ws50_st35 \
  --ref-to-ref-cache-mode auto \
  --ref-to-ref-cache-dir /scratch/project_2000539/dorian/cache/tuner_parallel_v2_1/ref_to_ref_combo_cache_ws50_st35 \
  --cpus-per-task 128 \
  --mem 64G \
  --final-visual-cpus-per-task 8 \
  --final-visual-mem 48G \
  --with-visuals
```

The script submits one Slurm job per dynamic worker. The script name still says `20nodes_10docs_each` for workflow continuity, but the implementation is dynamic. Workers no longer receive fixed document ranges. Every worker claims documents from the same shared pool and asks for a replacement document as soon as one local document slot finishes.

The launcher creates:

```text
<output-dir>/
  document_pool/
  dynamic_pool_manifest.json
  launch_commands.sh
  logs/
  shards/
  plots/                         # only after final visualisation succeeds
```

Important production default: the shell launcher disables the matrix `.npz` cache unless `--use-matrix-cache` is passed. The direct Python runner has the opposite default and uses its local `_matrix_cache` unless `--no-matrix-cache` is passed.

## Quick Start: Direct Python Run

Direct Python execution is useful for debugging, small smoke tests, cache warm-up, or running without Slurm:

```bash
cd /scratch/project_2017385/dorian/Churro_copy
PYTHONPATH=/scratch/project_2017385/dorian/Churro_copy \
  /appl/soft/ai/wrap/pytorch-2.9/bin/python3 tuner_parallel_v2_1/run_hough_parameter_sweep.py \
    --runfile-json /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_full1170_run1/vllm/dev/outputs.json \
    --output-dir /scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_1_debug_run \
    --max-items 5 \
    --hough-threshold-range 10 12 \
    --line-length-range 5 7 \
    --line-gap-range 0 2 \
    --scores-pkl-ref-to-pred /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/scores_reference_prediction_ws50_st35.pkl \
    --scores-pkl-ref-to-ref /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/scores_reference_self_ws50_st35.pkl \
    --score-index-cache-dir /scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_1_parallel/.score_index_cache \
    --ref-to-ref-cache-mode off \
    --workers 3 \
    --doc-workers 1
```

Direct Python execution does not create a shared dynamic pool unless `--dynamic-document-pool-dir` is passed. Without that flag, it runs over the selected documents inside the single process output directory.

## What The Tuner Optimizes

For each document, the tuner evaluates a grid over:

```text
hough_threshold
hough_line_length
hough_line_gap
hough_seed
```

The current scientific configuration fixes `hough_seed` at `1`. The CLI still accepts seed flags for compatibility, but seed sweep is disabled in this tuner.

For every parameter combination, the evaluator:

1. Runs falling-diagonal probabilistic Hough detection on the `ref_to_pred` matrix.
2. Converts raw Hough endpoint tuples into line dictionaries.
3. Filters and merges raw candidate lines by true-IoU ownership.
4. Builds a compact v2.12-compatible scoring payload for the `ref_to_pred` final ownership.
5. Computes weighted along-lines normalized Levenshtein similarity.
6. Computes or reads the exact `ref_to_ref` payload for the same Hough parameters.
7. Builds v2.12 coverage arrays using the `ref_to_ref` baseline and `ref_to_pred` scoring payload.
8. Computes coverage ratios, repetition on reference, hallucination, and the final harmonic `tuning_score`.
9. Writes compact scalar rows and optional visualisation records.
10. Reduces all rows for the document to one deterministic best combination.

The ranking key prefers, in order:

```text
higher tuning_score
higher weighted_along_lines_nls
higher correct_ref_coverage
lower hallucination
more line_guided_columns
fewer fallback_columns
smaller hough_threshold
smaller hough_line_length
smaller hough_line_gap
smaller hough_seed
```

The last four tie-breaks keep repeated runs stable and explainable.

## Shell Launcher Flags

`run_hough_parameter_sweep_20nodes_10docs_each.sh` is the production Slurm launcher.

Required flags:

- `--runfile-json <path>`: path to the `outputs.json` file containing document records. Each selected record must provide at least the file name and the reference/prediction text fields expected by the shared runfile parser.
- `--output-dir <dir>`: shared top-level output directory. The launcher creates this directory if needed and writes scheduling state, worker outputs, logs, and optional plots under it.

Dynamic scheduling flags:

- `--shard-count <n>`: number of identical sbatch worker jobs to submit. In dynamic mode this does not mean fixed static shards; it means the number of workers competing for documents from the shared pool.
- `--max-items <n>`: total number of selected documents to put in the shared pool after any runfile selection filters. Default: `600`.
- `--docs-per-shard <n>`: compatibility-only argument. It is validated but ignored by dynamic scheduling.

Hough grid flags:

- `--hough-threshold-range <start> <end>`: inclusive integer threshold range passed to `skimage.transform.probabilistic_hough_line`. Higher thresholds require stronger evidence in Hough space.
- `--line-length-range <start> <end>`: inclusive integer `line_length` range passed to probabilistic Hough. Larger values require longer raw candidate segments.
- `--line-gap-range <start> <end>`: inclusive integer `line_gap` range passed to probabilistic Hough. Larger values allow larger gaps between points on a candidate segment.
- `--hough-start <float>`: initial value for the adaptive Hough-image threshold. The default is `8.0`; the document-level Hough context lowers this value until the remaining transformed-score intensity exceeds `1.4 * max(rows, columns)`.
- `--align-abs-min-len <float>`: coarse minimum raw candidate length before the true-IoU filter.
- `--align-min-iou-threshold <float>`: minimum true-IoU overlap threshold used when deciding whether two candidate line coverages belong in the same merge component.

Matrix and cache flags:

- `--scores-pkl-ref-to-pred <path>`: read-only pickle stream containing precomputed `ref_to_pred` score matrices.
- `--scores-pkl-ref-to-ref <path>`: read-only pickle stream containing precomputed `ref_to_ref` score matrices.
- `--score-index-cache-file <path>`: explicit index-cache file for the `ref_to_pred` pickle stream.
- `--score-index-cache-file-ref-to-ref <path>`: explicit index-cache file for the `ref_to_ref` pickle stream.
- `--score-index-cache-dir <dir>`: directory containing or receiving score-stream index caches.
- `--matrix-cache-dir <dir>`: directory for reusable `.npz` matrix cache files.
- `--use-matrix-cache`: enables matrix `.npz` cache reads and writes in the shell launcher.
- `--disable-pkl-matrix-source`: disables read-only pickle-stream matrix loading. If the `.npz` cache also misses, matrices are computed from text.
- `--text-metrics-v212-dir <dir>`: optional external `text_metrics_v2_12_parallel` tree for audits and equivalence checks. Normal tuner scoring uses local compatibility code under `metrics/v2_12_compat/`.
- `--ref-to-ref-cache-mode <off|auto|read-only>`: controls exact reference-self document-pack cache usage.
- `--ref-to-ref-cache-dir <dir>`: directory for exact `ref_to_ref` cache packs.

Combination-bundle and visualisation flags:

- `--combination-bundle-scope <none|all|valid-only|invalid-only>`: decides which evaluated combinations are written into visualisation bundles. The shell launcher defaults to `all`.
- `--no-candidate-lines`: compatibility flag. Current lean `.pklstream` bundles do not store pre-filter candidate geometry.
- `--with-visuals`: submits one final visualisation job after all dynamic worker jobs finish successfully.
- `--hide-line-labels`: hides raw/final line labels in the final stitched panels.

Slurm flags:

- `--account <name>`: Slurm account. Default: `project_2017385`.
- `--partition <name>`: Slurm partition. Default: `medium`.
- `--time <HH:MM:SS>`: Slurm time limit. Default: `36:00:00`.
- `--cpus-per-task <n>`: CPU count per worker job. Used to compute document concurrency.
- `--mem <amount>`: memory for each worker job.
- `--final-visual-cpus-per-task <n>`: CPU count for the final visualisation job.
- `--final-visual-mem <amount>`: memory for the final visualisation job.

Other flags:

- `--workers <n>`: compatibility argument. In dynamic Slurm mode the launcher sets the Python worker count to the number of active threshold values.
- `--doc-workers <n>`: requested maximum number of active documents inside one worker job. The launcher lowers it if the threshold count would oversubscribe the node.
- `--levenshtein-backend <python|c>`: exact Levenshtein backend. `c` uses RapidFuzz and is the normal fast path.
- `-h`, `--help`: prints the launcher help text and exits.

## Direct Python Flags

`run_hough_parameter_sweep.py` exposes the same core tuner plus some developer/debug controls.

Selection flags:

- `--target-fname <name>`: select only matching document basenames. Repeat the flag to select multiple names.
- `--max-items <n>`: cap selected documents after target filtering.
- `--selection-index-range <start> <end>`: zero-based inclusive range inside the selected document list. Aliases: `--item-index-range`, `--document-index-range`.

Dynamic-pool worker flags:

- `--dynamic-document-pool-dir <dir>`: enables worker-side dynamic document claiming from an existing pool.
- `--dynamic-worker-id <id>`: stable worker id embedded in claim filenames and logs.
- `--dynamic-cpus-per-task <n>`: CPU count used to cap local document concurrency by threshold count.

Cache warm-up flag:

- `--ref-to-ref-cache-warm-only`: computes and writes exact `ref_to_ref` document-pack cache entries, then exits before prediction-side tuning. Use it with `--ref-to-ref-cache-mode auto`.

Profiling/export flags:

- `--profile-combinations`: writes `combination_profile.csv` with scalar timing/count diagnostics, including filter profile fields.
- `--no-combination-score-table`: disables the normal `combination_scores.csv.gz` scalar table. Leave this enabled for parameter-range analysis.

Python-only bundle options:

- `--combination-bundle-dir <dir>`: directory for optional `.pklstream` bundles.
- `--combination-bundle-scope <none|all|valid-only|invalid-only|winner-only>`: Python supports `winner-only`, which writes only the document-winning combination geometry.
- `--combination-bundle-include-candidate-lines`: accepted for compatibility. Current lean bundles store raw Hough lines and final surviving lines, not pre-filter candidate records.
- `--with-visuals`: for direct Python runs, visualisation is generated after the sweep in the same process. If no bundle scope was provided, the Python entry point uses `winner-only` geometry.

Compatibility flags:

- `--hough-seed <n>`: accepted but ignored; fixed seed `1` is used.
- `--seed-range <start> <end>`: accepted but ignored; fixed seed `1` is used.
- `--threshold`, `--line_length`, `--line-length`, `--line_gap`, and `--line-gap` are aliases for the explicit range flags.

## Dynamic Document Pool

The dynamic pool is scheduling-only. It stores document identity and worker lease state, not scores or metrics.

Pool layout:

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

Each selected document starts as:

```text
available/document_000042.json
```

A worker claims it with an atomic same-filesystem rename to:

```text
claimed/document_000042.worker_003.pid_12345.json
```

When the document has finished and normal tuner outputs are written, the lease moves to:

```text
done/document_000042.json
```

If a worker crashes before output export completes, active and completed-but-not-exported leases move to `failed/`. This avoids a false `done` state.

The launcher computes safe document concurrency as:

```text
threshold_count = threshold_end - threshold_start + 1
node_document_capacity = floor(cpus_per_task / threshold_count)
effective_doc_workers = min(requested_doc_workers, node_document_capacity)
threshold_workers_per_active_document = threshold_count
```

Example with 128 CPUs:

```text
threshold 10..35 -> 26 threshold values -> floor(128 / 26) = 4 active documents
threshold 12..35 -> 24 threshold values -> floor(128 / 24) = 5 active documents
```

Monitoring commands:

```bash
find <output-dir>/document_pool/available -name 'document_*.json' | wc -l
find <output-dir>/document_pool/claimed -name 'document_*.json' | wc -l
find <output-dir>/document_pool/done -name 'document_*.json' | wc -l
find <output-dir>/document_pool/failed -name 'document_*.json' | wc -l
tail -n 50 <output-dir>/document_pool/events.jsonl
```

## Matrix Loading Order

Each selected document needs a `ref_to_pred` matrix and a `ref_to_ref` matrix. The preparation code tries sources in this order for each direction:

1. `.npz` matrix cache, if matrix caching is enabled and the exact text/window cache key exists.
2. Read-only scores `.pkl` stream, if the matching pickle path is provided and the score-index cache can locate the document.
3. Fresh matrix computation from text using the shared `score_matrix_builder`.

When a matrix is loaded from a pickle stream and matrix caching is enabled, the tuner writes the matrix into the `.npz` cache for future reuse. If a pickle entry is missing or mismatched, the tuner records telemetry and falls back to computation.

Document preparation also computes once per document:

- whole-document normalized Levenshtein similarity;
- stride text blocks for line-level Levenshtein;
- one Hough context for `ref_to_pred`;
- one Hough context for `ref_to_ref`.

Dynamic-pool safety rule: document preparation must stay lazy. It pulls exactly one selected item as lookahead, then consumes one additional selected item only when the scheduler has a free local document slot. Eagerly converting the dynamic iterator into a list would claim the whole pool from one worker.

## Hough Detection

`alignment/line_alignment_pipeline_fast.py` owns Hough detection.

The Hough context is precomputed once per matrix:

1. Coerce the score matrix into a clean numeric array.
2. Normalize values into a dense-style `[0, 1)` range.
3. Transform the normalized values with `1 / (1 - normalized_score_matrix)` so high-similarity cells are emphasized.
4. Lower the adaptive threshold from `--hough-start` until the thresholded Hough image has total intensity greater than `1.4 * max(rows, columns)`.
5. Store both the thresholded Hough image and a boolean active-cell mask.

For each Hough combination, detection calls `skimage.transform.probabilistic_hough_line` with:

- the precomputed Hough image;
- the active threshold value;
- the active line length;
- the active line gap;
- deterministic `rng=np.random.default_rng(hough_seed + document_index)`;
- theta values restricted to falling visual diagonals.

The tuner defines the accepted line direction in image/matrix coordinates:

```text
x increases left-to-right
y increases top-to-bottom
accepted line moves from upper-left toward lower-right
visual angle is 30..60 degrees
```

The skimage theta band is a normal-angle band, not the plotted line angle. After skimage returns candidate endpoints, the tuner applies an explicit endpoint direction guard so upward or horizontal false positives do not enter true-IoU filtering.

## True-IoU Line Filtering

The stable public filter import path is still:

```python
from tuner_parallel_v2_1.filtering.line_filtering_v2_1_IoU_fast import filter_lines_for_alignment_by_ownership
```

That file is now a small wrapper. The implementation is split into focused modules:

```text
filtering/
  line_filtering_v2_1_IoU_fast.py      public compatibility wrapper
  filter_candidate_coverages.py        raw Hough line normalization and path coverage construction
  filter_overlap_merging.py            true-IoU overlap graph and merge components
  filter_final_assignment.py           final prediction-column ownership and final line output
  filter_geometry_helpers.py           line bounds, support sampling, ref-row expansion, weighted line fitting
  filter_cython_accelerators.py        optional compiled helper import boundary
  filter_profile_fields.py             scalar timing/count profile field helpers
```

The filter returns:

```text
final_lines
assignment["mapped_y"]
assignment["mapped_line_id"]
```

`mapped_y[x]` is the selected reference row for prediction column `x`. `mapped_line_id[x]` is the final line owner for prediction column `x`, or `-1` if no line owns that column.

Filtering stages:

1. Normalize raw Hough line dictionaries and attach `raw_line_id`.
2. Fill missing line length and support values.
3. Apply coarse candidate gates:
   - candidate length must be at least `align_abs_min_len`;
   - candidate score must not be below a small fraction of the maximum raw score;
   - candidate support must be at least the 75th percentile of matrix scores.
4. If all candidates are removed by the coarse gates, keep the best raw line as a fallback candidate.
5. Sample each candidate path across prediction columns and collect:
   - `x_to_y`;
   - `x_to_score`;
   - prediction-column coverage;
   - reference-row coverage;
   - score totals and means.
6. Expand reference-row coverage between adjacent sampled columns so the overlap check sees the full crossed band.
7. Generate only candidate pairs whose prediction bounds and reference bounds can overlap.
8. Compute true-IoU as the minimum of prediction-set IoU and reference-set IoU.
9. Build connected components from pairs whose true-IoU exceeds `align_min_iou_threshold`.
10. Merge each component by keeping the strongest local sample per prediction column.
11. Fit one representative straight line through the merged sampled path.
12. Assign each prediction column to the strongest surviving coverage crossing that column.
13. Prune coverages that own no prediction columns and recompute assignment until every remaining coverage owns at least one column.
14. Export final surviving line dictionaries with ownership counts and score summaries.

The weighted line fit in `filter_geometry_helpers.py` replaces the old direct `np.polyfit(..., deg=1, w=weights)` call with the exact degree-1 weighted least-squares formula. This is not a scientific change: the formula mirrors NumPy's weight semantics, including the fact that NumPy applies `w` to residuals before squaring, so the normal-equation weights are `w ** 2`.

## Scoring Hot Loop

`tuner/hough_eval.py` owns one-combination evaluation.

For `ref_to_pred`, it:

1. Runs Hough detection.
2. Runs true-IoU filtering.
3. Builds a compact v2.12-compatible scoring payload.
4. Computes weighted along-lines normalized Levenshtein similarity from the compact payload.

For `ref_to_ref`, it:

1. Reads an exact cached reference-self payload when possible.
2. Otherwise computes the same Hough/filter/scoring-payload path as `ref_to_pred`, but on the `ref_to_ref` matrix.
3. Builds and stores `refref_y`, the reference-self y-axis coverage baseline array.

Coverage computation then uses:

```text
refref_y baseline + ref_to_pred compact payload -> y_diff and other_x arrays
y_diff and other_x arrays -> coverage ratios
coverage ratios + weighted_along_lines_nls -> tuning_score
```

The evaluator keeps the historical exported field name `timing_build_bundle_seconds`, but the measured work is now compact scoring-payload construction, not full verbose diagnostic-bundle construction.

## v2.12 Metric Boundary

The tuner is intended to be understandable from inside `tuner_parallel_v2_1/`. The v2.12 metric code required at runtime is vendored locally:

```text
metrics/v2_12_compat/
  line_metric_bundle.py
  line_coverage_arrays.py
  ordered_sequence_helpers.py
  text_window_projection.py
```

`metrics/v2_12_metric_adapter.py` is the public boundary used by the evaluator. Normal scoring calls the local compatibility implementation. The external `text_metrics_v2_12_parallel` tree is retained only for audits and equivalence tests.

Important rule: do not casually simplify formulas in `metrics/v2_12_compat/`. Those files preserve historical metric semantics. Any formula change is a scientific metric change and must be tested against the original v2.12 implementation.

## Reference-Self Cache

The `ref_to_ref` branch is expensive because it repeats for every Hough combination even though it depends only on reference text and Hough parameters. `cache/ref_to_ref_combo_cache.py` stores exact reference-self payloads.

Cache modes:

- `off`: never read or write the persistent `ref_to_ref` cache.
- `auto`: read exact cache entries when available; compute misses; write one completed document pack after the document finishes.
- `read-only`: read existing exact cache entries only; fail if a needed payload is missing.

Current production cache shape:

```text
one document-pack cache file per document and active Hough grid
```

The document pack stores all threshold/line-length/line-gap/seed payloads for that document/grid. It stores repeated `refref_y` arrays once and references them by row index:

```text
refref_y_unique_rows
refref_y_row_index_by_combination
```

Older v2/v3 threshold-pack caches can still be read for compatibility. New production `auto` runs write document packs.

Production write flow:

1. Scheduler opens one document cache session per active document.
2. Each threshold worker opens a lightweight threshold view.
3. Cache hits return exact payloads immediately.
4. Cache misses compute exact payloads and store them in the threshold view.
5. Closing the threshold view moves computed payloads into the document session in memory.
6. After the whole document finishes, the document pack is queued to a background writer.
7. Summaries are exported only after cache writers close, so cache counters and files reflect completed work.

## Combination Bundles And Visualisation

The main scientific outputs are scalar CSV/JSON files. Combination bundles are observational geometry files used by visualisation tools.

Current bundle format:

```text
combination_bundles/
  document_000123_example/
    document_metadata.json
    ref_to_pred_score_matrix.npy
    ref_to_ref_score_matrix.npy
    threshold_010.pklstream
    threshold_011.pklstream
    ...
```

The lean `.pklstream` records contain:

- schema and record format;
- worker/shard metadata;
- document metadata;
- Hough parameters;
- scalar metrics;
- raw `ref_to_pred` Hough lines;
- final surviving `ref_to_pred` lines.

They intentionally do not store full `ref_to_ref` geometry, pre-filter candidate geometry, or column-assignment debug payloads. The final visualisation code does not use those fields, and storing them would make bundles much larger.

`tools/language_hough_parameter_metric_analysis.py` reads bundle metadata and record streams, builds compact pandas tables, and creates graph grids, stitched best-combination panels, CSV summaries, and manifest JSON files. It supports both current `.pklstream` bundles and older `.jsonl` / `.jsonl.gz` bundles.

## Main Output Files

Each worker output directory contains:

```text
hough_parameter_sweep_summary.json
best_params_per_document.json
combination_scores.csv.gz
csv/
  all_documents_parameter_influence.csv
  best_config_per_document.csv
  hough_threshold_summary.csv
  hough_line_length_summary.csv
  hough_line_gap_summary.csv
  hough_seed_summary.csv
  invalid_combinations.csv
combination_bundles/              # when bundle logging is enabled
```

Important files:

- `hough_parameter_sweep_summary.json`: run summary, timing summaries, selected ranges, parallelism, cache stats, and global best-by-parameter reductions.
- `best_params_per_document.json`: best row for each document with Hough parameters and metrics.
- `combination_scores.csv.gz`: one compact scalar row per evaluated combination. This is the fastest file to use for parameter-range analysis.
- `csv/best_config_per_document.csv`: tabular best row per document.
- `csv/all_documents_parameter_influence.csv`: document/parameter influence rows.
- `csv/hough_*_summary.csv`: aggregate summaries by each swept parameter.
- `csv/invalid_combinations.csv`: combinations rejected by v2.12 coverage category validation.
- `combination_profile.csv`: optional profiling table written when `--profile-combinations` is enabled.

Dynamic Slurm runs write these files under each `shards/dynamic_worker_*/` directory. The final visualisation job reads all worker bundle directories through `<output-dir>/shards`.

## Folder And Script Guide

Top-level scripts:

- `run_hough_parameter_sweep_20nodes_10docs_each.sh`: production Slurm launcher. Creates the dynamic pool, submits workers, writes launch provenance, and optionally submits final visualisation.
- `run_hough_parameter_sweep_shard.sbatch`: Slurm worker wrapper. Loads the runtime environment, builds/requires Cython accelerators, sets `PYTHONPATH`, and calls the Python runner.
- `run_hough_parameter_sweep.py`: Python tuner entry point. Handles direct runs and dynamic-worker runs.
- `run_language_hough_parameter_metric_analysis.sh`: final visualisation wrapper for Slurm or manual use.

Core packages:

- `runtime/`: path bootstrap helpers. `runtime_paths.py` makes tuner-local modules and shared project helpers importable in script and package execution.
- `logging_utils/`: timestamped logging helper used by runner and scheduler.
- `dynamic_pool/`: atomic file-backed document leasing. This package never stores metrics.
- `matrices/`: runfile selection, score-stream index loading, `.npz` matrix cache helpers, and per-document preparation.
- `alignment/`: Hough detection, raw segment endpoint conversion, stride-block helpers, and fast along-lines grouping helpers.
- `filtering/`: true-IoU line filtering split into focused modules.
- `metrics/`: Levenshtein scoring, harmonic objective scoring, and local v2.12-compatible coverage logic.
- `cache/`: exact persistent `ref_to_ref` cache sessions and pack readers/writers.
- `tuner/`: sweep configuration, scheduler, one-combination evaluation, reductions, aggregation, and cache warm-up.
- `outputs/`: CSV/JSON writers, compact score/profile exporters, invalid-combination exports, and optional bundle logging.
- `tools/`: visualisation and language/document-type diagnostic scripts.
- `cython_accel/`: optional compiled accelerators and build script.
- `tests/`: regression tests for filtering, Cython fallbacks, dynamic-pool laziness, local v2.12 equivalence, cache behavior, and exports.

Documentation files:

- `README.md`: this operator/developer guide.
- `CODEBASE_DEEP_ANALYSIS_20260524.md`: long-form working analysis and implementation notes. It is intentionally more experimental and historical than this README.
- `metrics/v2_12_compat/README.md`: provenance and rules for the vendored v2.12 compatibility subset.
- `cython_accel/README.md`: accelerator build and import-path notes.

Generated analysis artifacts:

- `threshold_test/`: human-inspection threshold plots from earlier experiments.
- `final_surviving_*.png` and `final_surviving_*.csv`: stitched diagnostic artifacts from earlier analysis turns.

## Module Details

`dynamic_pool/document_pool.py`:

- `initialize_document_pool()` writes `selected_documents.jsonl`, manifest metadata, and one `available/document_*.json` file per selected document.
- `DocumentLeasePool.claim_next()` atomically moves one available file into `claimed/`.
- `mark_lease_done()` and `mark_lease_failed()` finalize a lease after worker output state is known.
- `iter_claimed_selected_run_items_from_pool()` lazily claims exactly one document whenever the scheduler asks for another selected run item.

`matrices/document_prep.py`:

- Keeps document preparation streaming and RAM-safe.
- Prepares both matrix directions.
- Computes document-level NLS and stride blocks.
- Precomputes Hough contexts once per matrix direction.
- Records matrix source telemetry.

`alignment/line_alignment_pipeline_fast.py`:

- Normalizes matrices for Hough.
- Creates active Hough masks.
- Runs falling-diagonal probabilistic Hough detection.
- Applies the endpoint direction guard.
- Calls the true-IoU filter.

`filtering/line_filtering_v2_1_IoU_fast.py`:

- Stable import path for the rest of the tuner.
- Delegates implementation work to the focused filtering modules.

`filtering/filter_candidate_coverages.py`:

- Turns raw line dictionaries into normalized candidate lines.
- Samples each candidate path.
- Builds coverage objects used by overlap merging and final assignment.

`filtering/filter_overlap_merging.py`:

- Generates possible overlap pairs using bounds checks.
- Computes exact true-IoU decisions.
- Builds connected components.
- Merges each component into one coverage object.

`filtering/filter_final_assignment.py`:

- Indexes coverages by prediction column.
- Assigns each prediction column to one winning coverage.
- Removes coverages that own no columns.
- Exports final line dictionaries and assignment arrays.

`filtering/filter_geometry_helpers.py`:

- Owns line bounds, support sampling, reference-row expansion, set IoU fallback, and the exact weighted degree-1 line fit.

`filtering/filter_cython_accelerators.py`:

- Centralizes optional compiled filtering helper imports.
- Modules import this boundary as a module so tests can monkeypatch individual accelerator attributes.

`tuner/sweep_scheduler.py`:

- Schedules documents and thresholds.
- Opens document-level `ref_to_ref` cache sessions.
- Collects threshold-local bundle records.
- Starts replacement documents as local slots free.
- Aggregates calculation-only timing counters.

`tuner/hough_eval.py`:

- Evaluates one combination.
- Uses compact scoring payloads in the hot loop.
- Computes invalid-row diagnostics when v2.12 coverage categories fail.
- Builds optional visualisation records without writing files directly.

`outputs/combination_bundle_logger.py`:

- Builds lean record dictionaries in memory.
- Writes one completed document bundle in a background thread.
- Keeps disk I/O out of the per-combination hot loop.

`outputs/tuner_combination_score_exports.py`:

- Writes the compact `combination_scores.csv.gz` table used for parameter-range analysis.

`cython_accel/optional_filtering.py`:

- Wraps compiled helpers for set IoU, coverage indexing, line support/path sampling, and final assignment.
- Returns `None` or Python fallback-compatible values when compiled helpers are absent.

## Cython Accelerators

Current extensions:

```text
cython_accel/along_lines_core.pyx
cython_accel/filter_core.pyx
```

`along_lines_core.pyx` accelerates:

- grouping owned prediction columns by final line id;
- weighted mean calculation from line scores and line lengths.

`filter_core.pyx` accelerates:

- set IoU;
- coverage-index construction by prediction column;
- mean support sampling for line endpoints;
- line path sampling;
- final prediction-column ownership assignment.

Build manually from the tuner directory:

```bash
cd /scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_1
source /usr/share/lmod/8.6.17/init/bash
module use /appl/modulefiles
module load pytorch
python3 cython_accel/build.py build_ext --inplace
```

The pure-Python implementation remains the readable reference/fallback path. Compiled helpers are expected to preserve exact output semantics.

## Testing

Run the local regression tests:

```bash
cd /scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_1
PYTHONPATH=/scratch/project_2017385/dorian/Churro_copy \
  /appl/soft/ai/wrap/pytorch-2.9/bin/python3 -m pytest -q --confcutdir=. tests
```

Useful focused tests:

```bash
PYTHONPATH=/scratch/project_2017385/dorian/Churro_copy \
  /appl/soft/ai/wrap/pytorch-2.9/bin/python3 -m pytest -q --confcutdir=. \
  tests/test_filtering_cython_sampling.py \
  tests/test_filter_profile_side_channel.py
```

Compile check:

```bash
/appl/soft/ai/wrap/pytorch-2.9/bin/python3 -m py_compile \
  run_hough_parameter_sweep.py \
  filtering/*.py \
  tuner/*.py \
  metrics/*.py \
  outputs/*.py
```

Shell syntax checks:

```bash
bash -n run_hough_parameter_sweep_20nodes_10docs_each.sh
bash -n run_hough_parameter_sweep_shard.sbatch
bash -n run_language_hough_parameter_metric_analysis.sh
```

When refactoring, use a small real-document run for behavior checks rather than launching the full production set. Keep the Hough parameter range representative enough to exercise the hot loop, but keep document count small.

## Maintenance Rules

- Keep scientific behavior stable unless the change is explicitly treated as a metric change.
- Prefer local tuner modules over hidden imports from sibling tuner versions.
- Keep `metrics/v2_12_compat/` formulas stable and audit changes against the external v2.12 implementation.
- Keep dynamic-pool iteration lazy.
- Keep disk writes out of the per-combination hot loop.
- Keep optional Cython accelerators behind Python fallback boundaries.
- Do not add multiple independent implementations of the same scientific logic.
- Keep comments focused on why a non-obvious implementation choice exists.
- Keep file and function names descriptive enough that a new developer can find the owner of each pipeline stage from the directory tree.
