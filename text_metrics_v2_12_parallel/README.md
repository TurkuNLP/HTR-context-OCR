# `text_metrics_v2_12_parallel`

## What this pipeline does

`text_metrics_v2_12_parallel` computes text-alignment quality metrics by combining:

- precomputed score matrices,
- diagonal line detection with probabilistic Hough,
- ownership-based true-IoU line filtering,
- line-guided Levenshtein measurement,
- and reference-coverage / hallucination analysis.

The pipeline is designed to answer a very specific question:

> given one reference text and one prediction text, how well can we align the prediction to the reference when the score matrix contains diagonal alignment structure?

The implementation in this directory is intended to preserve the current default behavior exactly when it is called with the same:

- input documents,
- `scores.pkl` streams,
- Hough parameters,
- and report flags.

That exact-result requirement is a core design constraint of this version.

## Public entrypoint

The only supported public entrypoint is:

- [`run_text_metrics_report.sh`](./run_text_metrics_report.sh)

That shell script is the contract for normal use. It validates the run configuration, prepares the output directory, initializes the Python environment, and then calls:

- [`text_metrics_report.py`](./text_metrics_report.py)

The Python file is intentionally small and delegates the real work to:

- [`pipeline/run_text_metrics_pipeline.py`](./pipeline/run_text_metrics_pipeline.py)

## High-level data flow

At a high level, one document is processed in this order:

1. Load or derive the score matrices.
2. Run probabilistic Hough on the score matrix.
3. Pass raw Hough segments directly to the true-IoU filter by default, or explicitly request the merged-Hough compatibility handoff.
4. Build stable line bundles from the surviving line assignments.
5. Compute line-guided Levenshtein metrics.
6. Compute reference-coverage and hallucination ratios.
7. Write a human-readable report.

The main production call chain is:

1. [`run_text_metrics_report.sh`](./run_text_metrics_report.sh)
2. [`text_metrics_report.py`](./text_metrics_report.py)
3. [`pipeline/run_text_metrics_pipeline.py`](./pipeline/run_text_metrics_pipeline.py)
4. [`parallelisation/execute_document_tasks.py`](./parallelisation/execute_document_tasks.py)
5. [`pipeline/process_single_document_metrics.py`](./pipeline/process_single_document_metrics.py)
6. [`line_alignment_pipeline.py`](./line_alignment_pipeline.py)
7. [`hough_line_transform_endpoints_line_direction_30_to_60_degrees.py`](./hough_line_transform_endpoints_line_direction_30_to_60_degrees.py)
8. [`line_filtering_v2_12_IoU.py`](./line_filtering_v2_12_IoU.py)
9. [`line_metric_bundle.py`](./line_metric_bundle.py)
10. [`levenshtein_metric.py`](./levenshtein_metric.py)
11. [`line_coverage_subtract.py`](./line_coverage_subtract.py)

## Inputs

The pipeline can process items from either of these sources:

- a `runfile` JSON,
- one or more explicit `scores.pkl` streams,
- or a `scores.pkl` root directory that contains the expected compare subdirectories.

The explicit `scores.pkl` inputs are:

- `--scores-pkl-ref-to-pred`
- `--scores-pkl-ref-to-ref`
- `--scores-pkl-ref-to-adjusted-pred`

There is also optional root discovery through:

- `--scores-pkl-root`

The root-discovery path resolves the expected compare subdirectories and chooses the matching `.pkl` file using stable naming rules.

## Score-matrix loading

The active score-matrix loading logic lives in:

- [`pipeline/load_or_compute_score_matrices.py`](./pipeline/load_or_compute_score_matrices.py)
- [`score_stream_index.py`](./score_stream_index.py)
- [`pipeline/resolve_text_metrics_input_sources.py`](./pipeline/resolve_text_metrics_input_sources.py)

Important design points:

- The score-stream index cache is preserved because it makes repeated runs much faster.
- Runtime caches are intentionally kept outside the source tree.
- The pipeline still supports already-created `.pkl` files exactly as before.

## Hough detection

The active detector module is:

- [`hough_line_transform_endpoints_line_direction_30_to_60_degrees.py`](./hough_line_transform_endpoints_line_direction_30_to_60_degrees.py)

That module is responsible for:

- score-matrix normalization,
- the reciprocal-emphasis transform,
- adaptive threshold selection,
- fixed diagonal theta selection,
- and the probabilistic Hough call itself.

The pipeline uses a deterministic effective seed per document:

- `effective_seed = base_hough_seed + item_index`

This keeps the probabilistic Hough result reproducible for a fixed run configuration.

## Default and alternate Hough handoff modes

The explicit Hough handoff-mode logic lives in:

- [`hough_detection/line_handoff_modes.py`](./hough_detection/line_handoff_modes.py)

Supported modes are:

- `merged_hough_to_true_iou`
- `raw_hough_to_true_iou`

### Default mode

`raw_hough_to_true_iou` is now the default production behavior.

The path is:

1. raw Hough segments are detected,
2. the raw segments are converted into line records directly,
3. and those line records enter the true-IoU filter.

This default intentionally skips the historical greedy diagonal postprocessing
stage, so normal runs avoid the old `merge_diag()` bottleneck.

### Compatibility mode

`merged_hough_to_true_iou` remains available so old apples-to-apples runs can
still be reproduced by passing `--hough-handoff-mode merged_hough_to_true_iou`.

The path is:

1. raw Hough segments are detected,
2. the greedy diagonal postprocessing stage merges them,
3. the merged segments are converted into line records,
4. and those line records enter the exact same true-IoU filter.

## Greedy diagonal postprocessing

The compatibility Hough postprocessing logic is isolated in:

- [`hough_postprocessing/greedy_diagonal_segment_merging.py`](./hough_postprocessing/greedy_diagonal_segment_merging.py)

This module contains the current Python reference implementation of the historical `merging_diag()` behavior.

The raw-Hough default does not call this module.  The goal of keeping it isolated
is to make the compatibility behavior:

- easier to read,
- easier to profile,
- easier to benchmark,
- easier to accelerate,
- and easier to remove later if the compatibility handoff is retired.

## True-IoU filtering

The active production filter is:

- [`line_filtering_v2_12_IoU.py`](./line_filtering_v2_12_IoU.py)

Its job is to:

1. prepare candidate lines,
2. sample each line onto the score-matrix grid,
3. build coverage objects,
4. merge overlapping coverages using the true-IoU rule,
5. resolve final ownership per prediction column,
6. and return the stable final line set and assignment arrays.

The expensive detailed pairwise IoU analysis lives separately in:

- [`debug/line_filtering_v2_12_detailed_iou_analysis.py`](./debug/line_filtering_v2_12_detailed_iou_analysis.py)

That module is intentionally not part of the normal production import path.

## Bundle building and metrics

After filtering, the line output is converted into a stable intermediate bundle by:

- [`line_metric_bundle.py`](./line_metric_bundle.py)

That bundle is the shared input for:

- [`levenshtein_metric.py`](./levenshtein_metric.py)
- [`line_coverage_subtract.py`](./line_coverage_subtract.py)

The bundle keeps two related ideas separate:

- line ownership along prediction columns,
- and line-derived coverage intervals used for downstream text-coverage measurements.

## Reporting

The reporting flow is split across:

- [`pipeline/process_single_document_metrics.py`](./pipeline/process_single_document_metrics.py)
- [`parallelisation/record_parallel_progress.py`](./parallelisation/record_parallel_progress.py)
- [`parallelisation/write_parallel_report_files.py`](./parallelisation/write_parallel_report_files.py)
- [`pipeline/report_item_views.py`](./pipeline/report_item_views.py)

### Non-debug reporting

When `--debug` is not enabled, `report.json` is intentionally smaller and more human-readable. Each document item shows only the public metrics view.

The public per-document item contains:

- `fname`
- `normalised_levenshtein_similarity`
- `average_weighted_normalised_levenshtein_similarity`
- `correct_ref_coverage`
- `missing_ref_coverage`
- `repetition_on_ref`
- `hallucination`

The coverage and hallucination values are normalized ratios in `[0, 1]`, not
percentages.  Lower hallucination is better; the other coverage values should be
interpreted directly as fractions of the relevant text axis.

### Public metric names shared with the tuner

The tuner in `../tuner_parallel_v2/` imports this directory as the read-only
source of truth for line bundles, coverage arrays, and coverage/hallucination
ratios.  To keep reports comparable, the public metric names used by both
pipelines are:

- `normalised_levenshtein_similarity`
- `average_weighted_normalised_levenshtein_similarity`
- `correct_ref_coverage`
- `missing_ref_coverage`
- `repetition_on_ref`
- `hallucination`

The coverage and hallucination fields stay in `[0, 1]`.  The reports do not
write duplicate `1 - metric` fields; if a consumer needs a complement such as
`1 - hallucination`, it should compute it locally from the public value.

### Debug reporting

When `--debug` is enabled, the richer internal per-document payload is preserved so the run can be inspected in depth.

## Visuals

Optional visuals are produced by:

- [`visualisation/render_text_metrics_visualisations.py`](./visualisation/render_text_metrics_visualisations.py)

with support helpers in the same `visualisation/` directory.

Visual generation is controlled by:

- `--with-visuals`

The default run path skips visuals.

## Parallel execution

The pipeline can run sequentially or with multiple workers. The main parallelisation modules are:

- [`parallelisation/execute_document_tasks.py`](./parallelisation/execute_document_tasks.py)
- [`parallelisation/record_parallel_progress.py`](./parallelisation/record_parallel_progress.py)
- [`parallelisation/write_parallel_report_files.py`](./parallelisation/write_parallel_report_files.py)

The core design rule is that document count and worker count must not create separate logic branches. The same production logic is reused regardless of whether the run processes:

- one document or many documents,
- one worker or many workers.

## Debug timing

When `--debug` is enabled, the pipeline writes timing telemetry through:

- [`debug/per_document_stage_timing.py`](./debug/per_document_stage_timing.py)
- [`debug/run_timing_telemetry.py`](./debug/run_timing_telemetry.py)

The normal production path avoids this overhead when debug is off.

## Optional exact-result Cython backends

The optional compiled backend flow is intentionally narrow and conservative.

The relevant modules are:

- [`accelerators/build_optional_exact_result_cython_backends.py`](./accelerators/build_optional_exact_result_cython_backends.py)
- [`accelerators/load_optional_exact_result_cython_backends.py`](./accelerators/load_optional_exact_result_cython_backends.py)
- [`accelerators/greedy_diagonal_segment_merging_backend.py`](./accelerators/greedy_diagonal_segment_merging_backend.py)
- [`accelerators/true_iou_filter_backend.py`](./accelerators/true_iou_filter_backend.py)
- [`accelerators/cython_sources/`](./accelerators/cython_sources)

### How the optional backend flow works

At startup, the public shell entrypoint loads the `pytorch` module when the site module system is available. That module provides:

- Python 3.12,
- `Cython`,
- `cython`,
- and `cythonize`.

Then the Python pipeline does one best-effort build-or-reuse step:

1. compute a stable build key from the `.pyx` sources and runtime ABI,
2. reuse a matching compiled backend build from the external runtime cache if it already exists,
3. otherwise compile the optional backend modules into the external runtime cache,
4. publish an active manifest that points to the compiled extension files,
5. let worker processes lazily load those modules only if the manifest is valid.

If any part of that fails, the pipeline does **not** fail over to a different algorithm. It simply keeps using the Python reference implementation.

### Why this structure matters

This structure preserves the user-visible contract:

- only one public entrypoint is supported,
- the default pipeline logic stays the same,
- compiled artifacts stay outside the source tree,
- workers do not race to build extensions,
- and the Python implementation remains the correctness reference.

The compiled backends are allowed to improve:

- speed,
- RAM usage,
- and temporary object pressure,

but they are not allowed to change:

- thresholds,
- segment ordering,
- tie-breaking,
- rounding,
- ownership,
- or final report values.

## Runtime artifacts and cache locations

Runtime-generated artifacts live outside the source tree.

The central path helpers live in:

- [`pipeline/runtime_artifact_paths.py`](./pipeline/runtime_artifact_paths.py)

At the moment that module defines the runtime roots for:

- score-stream index caches,
- optional exact-result Cython backend builds,
- and regenerated diagrams.

Those runtime directories are intentionally ignored in version control.

## Source-tree layout

### Core production modules

- [`text_metrics_report.py`](./text_metrics_report.py): thin Python entrypoint.
- [`run_text_metrics_report.sh`](./run_text_metrics_report.sh): only supported public entrypoint.
- [`line_alignment_pipeline.py`](./line_alignment_pipeline.py): connects Hough detection to true-IoU filtering.
- [`hough_line_transform_endpoints_line_direction_30_to_60_degrees.py`](./hough_line_transform_endpoints_line_direction_30_to_60_degrees.py): active detector.
- [`line_filtering_v2_12_IoU.py`](./line_filtering_v2_12_IoU.py): active production filter.
- [`line_metric_bundle.py`](./line_metric_bundle.py): bundle construction.
- [`levenshtein_metric.py`](./levenshtein_metric.py): Levenshtein metrics.
- [`line_coverage_subtract.py`](./line_coverage_subtract.py): coverage and hallucination metrics.

### Internal structure packages

- [`pipeline/`](./pipeline): argument parsing, orchestration, matrix loading, report-item projections, and runtime-artifact paths.
- [`parallelisation/`](./parallelisation): worker execution, envelope recording, JSONL spooling, and final report writing.
- [`hough_detection/`](./hough_detection): explicit Hough handoff-mode selection.
- [`hough_postprocessing/`](./hough_postprocessing): compatibility greedy post-Hough merge behavior.
- [`alignment_utils/`](./alignment_utils): shared line-geometry helpers.
- [`shared/`](./shared): shared ordered-sequence and window-projection helpers.
- [`debug/`](./debug): optional debug and timing helpers.
- [`visualisation/`](./visualisation): optional report visualisations.
- [`accelerators/`](./accelerators): optional exact-result backend boundaries, build helpers, and lazy loaders.
- [`accelerators/cython_sources/`](./accelerators/cython_sources): Cython source files for the optional exact-result backend modules.

## Result-preservation rules

This directory is being maintained under strict result-preservation rules.

That means:

- a fixed production mode must keep returning the same results for the same inputs,
- any new internal structure must preserve the existing numeric behavior,
- optional accelerators must keep Python fallback behavior,
- and any new comparison path must be explicit and opt-in.

## Practical rule for contributors

If you are unsure where a change belongs, use this rule:

- if it changes metrics or alignment behavior, it is a pipeline-logic change and must be treated as such,
- if it only changes structure, comments, cache location, or implementation efficiency while preserving results, it belongs to the exact-result-preserving maintenance track.

That separation is intentional and important for this directory.
