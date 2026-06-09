# `text_metrics_v2_12_parallel` Detailed Cleanup, Dual-Handoff, And Cython Plan

## Scope

This document analyzes **only** the code inside:

- `/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel`

This is an **analysis and implementation plan only**.
No code changes are proposed here as already applied work. The goal is to describe exactly what would need to happen if we wanted to:

1. keep the current `merging_diag()`-based behavior as the default behavior,
2. make `merging_diag()` easy to remove in a later version,
3. add a second handoff path that passes raw Hough segments directly into the current true-IoU filter,
4. clean and restructure the directory for long-term maintainability,
5. preserve the **exact current logic and results** of the default path,
6. allow only improvements in runtime, RAM use, and code clarity,
7. identify where exact-result Cython acceleration would help most.

Before the first implementation change, the plan must create a full rollback snapshot so the directory can be restored immediately if something goes wrong.

Recommended snapshot root:

- `/scratch/project_2017385/dorian/Churro_copy/_unused_backup_20260423_151633/`

Recommended snapshot name pattern:

- `text_metrics_v2_12_parallel_snapshot_before_exact_cleanup_and_cython_<timestamp>/`

The snapshot should contain the full:

- `/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/`

source tree before any edits begin.

---

## Executive Summary

### 1. Hard conclusion about the current role of `merging_diag()`

Under the **current** `text_metrics_v2_12_parallel` pipeline architecture, removing `merging_diag()` from the default path is **not compatible** with the requirement:

- "same logic"
- "same results"
- "same outputs for the same parameters, same documents, same score matrices"

The reason is simple:

- the active line-alignment path currently passes **post-`merging_diag()` segments** into the true-IoU filter,
- not raw Hough segments.

That behavior is encoded directly in:

- [line_alignment_pipeline.py:48](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:48)
- [line_alignment_pipeline.py:56](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:56)
- [line_alignment_pipeline.py:57](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:57)
- [line_alignment_pipeline.py:69](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:69)

and the merge itself happens in:

- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:247](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:247)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:390](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:390)

So if `merging_diag()` is removed from the current default path and raw Hough lines are passed to true-IoU filtering instead, the results will change.

### 2. Exact-results-preserving cleanup is still very possible

The rest of the pipeline is already in relatively good shape compared with older versions. There is still meaningful cleanup available that **can preserve exact behavior**:

- remove source-tree runtime artifacts,
- remove stale naming and comments,
- remove dead import hacks,
- keep optional branches but isolate their code more cleanly,
- reduce non-debug spool payload size,
- improve module naming and directory organization through compatibility wrappers,
- benchmark the remaining Python-side bottlenecks without changing semantics,
- add exact-result Cython accelerators for the hottest pure-Python kernels.

### 3. Best implementation stance after the user decisions

The implementation direction is now specific:

1. `run_text_metrics_report.sh` remains the **only supported public entry point**.
2. The current merged-Hough handoff remains the **default behavior**.
3. A second handoff mode must exist under the same pipeline so the same true-IoU filter can alternatively receive raw Hough lines.
4. `merging_diag()` should be isolated into its own internal module boundary so it is easy to remove later, but it remains fully active in the default path now.
5. Existing supported optional branches remain:
   - `.pkl` input paths for `ref_to_ref` and `ref_to_(adjusted_)pred`
   - `--with-visuals`
   - `--debug`
   - per-document Hough JSON overrides from `pipeline/load_hough_params_per_document.py`
6. The only acceptable behavior changes are in the future alternate raw-Hough mode, never in the default mode during this refactor.
7. For Cython acceleration, Tier 1 and Tier 2 targets are explicitly in scope for the future implementation plan, while Tier 3 targets remain out of scope unless later profiling proves otherwise.

The rest of this plan has been updated to match that direction.

---

## Active Production Pipeline Today

The active shell-to-report path is:

1. [run_text_metrics_report.sh](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/run_text_metrics_report.sh)
2. [text_metrics_report.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/text_metrics_report.py)
3. [pipeline/run_text_metrics_pipeline.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/run_text_metrics_pipeline.py)
4. [parallelisation/execute_document_tasks.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/parallelisation/execute_document_tasks.py)
5. [pipeline/process_single_document_metrics.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/process_single_document_metrics.py)
6. [line_alignment_pipeline.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py)
7. [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py)
8. [line_filtering_v2_12_IoU.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_filtering_v2_12_IoU.py)
9. [line_metric_bundle.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_metric_bundle.py)
10. [levenshtein_metric.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/levenshtein_metric.py)
11. [line_coverage_subtract.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_coverage_subtract.py)
12. [parallelisation/write_parallel_report_files.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/parallelisation/write_parallel_report_files.py)
13. [pipeline/report_item_views.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/report_item_views.py)

Optional active branches:

- `--with-visuals`
  - [visualisation/render_text_metrics_visualisations.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/visualisation/render_text_metrics_visualisations.py)
  - [visualisation/render_alignment_matrix_views.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/visualisation/render_alignment_matrix_views.py)
  - [visualisation/render_line_coverage_views.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/visualisation/render_line_coverage_views.py)

- `--debug`
  - [debug/per_document_stage_timing.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/debug/per_document_stage_timing.py)
  - [debug/run_timing_telemetry.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/debug/run_timing_telemetry.py)

Per-document Hough JSON override mode:

- [pipeline/load_hough_params_per_document.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/load_hough_params_per_document.py)

---

## Where `merging_diag()` Sits In The Current Logic

The active flow for one score matrix is:

1. matrix preprocessing and adaptive thresholding,
2. probabilistic Hough,
3. `merging_diag()` over raw Hough segments,
4. conversion of merged segments into line dicts,
5. true-IoU filtering.

This is visible here:

- Hough raw call:
  - [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:379](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:379)
- `merging_diag()` call:
  - [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:390](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:390)
- downstream selection of `merged_lines`:
  - [line_alignment_pipeline.py:56](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:56)
  - [line_alignment_pipeline.py:57](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:57)
- conversion into filter input:
  - [line_alignment_pipeline.py:69](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:69)
- true-IoU filter entry:
  - [line_alignment_pipeline.py:74](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:74)

This means `merging_diag()` is not a diagnostic-only helper. It is on the active production path.

---

## What `merging_diag()` Depends On

Inside [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py), the helper chain is:

- `line_magnitude()`
- `point_line_distance()`
- `count_points_in_range()`
- `sample_line_pixels()`
- `longest_false_run()`
- `segment_length()`
- `segment_angle()`
- `nearest_endpoints()`
- `bridge_stats()`
- `merging_diag()`

The internal references show that all of these helpers exist only to support the merge heuristic:

- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:67](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:67)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:74](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:74)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:113](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:113)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:135](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:135)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:156](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:156)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:177](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:177)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:186](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:186)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:197](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:197)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:225](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:225)
- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:247](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:247)

### Exact list of helpers that become removable if `merging_diag()` is removed without replacement

If the code path no longer uses the merge heuristic at all, these functions become removable from the active detector module:

1. `line_magnitude()`
2. `point_line_distance()`
3. `count_points_in_range()`
4. `sample_line_pixels()`
5. `longest_false_run()`
6. `segment_length()`
7. `segment_angle()`
8. `nearest_endpoints()`
9. `bridge_stats()`
10. `merging_diag()`

In addition, the following local data preparation inside `detect_lines_dense_style()` becomes unnecessary:

- `ys, xs = np.nonzero(test2)`
- `points_glo = ...`

Those are only needed to support `merging_diag()` scoring.

---

## Why Removing `merging_diag()` Cannot Preserve Exact Results

### Current contract

Today the true-IoU filter receives:

- `merged_lines`

not:

- `raw_lines`

That is hard-coded in:

- [line_alignment_pipeline.py:56](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:56)
- [line_alignment_pipeline.py:69](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_alignment_pipeline.py:69)

### Consequence

If `merged_lines` becomes `raw_lines`, then the true-IoU filter sees a different candidate set.
Once the candidate set changes, the following may also change:

- `merged_line_count`
- `used_line_count`
- `line_guided_columns`
- `fallback_columns`
- line-wise Levenshtein bundle contents
- `average_normalized_levenshtein_along_lines`
- coverage percentages
- final `report.json` values

### Strong design conclusion

If the requirement is:

- **exact same logic and results**

then the only safe answer is:

- **do not remove `merging_diag()` from the production algorithm**

You can isolate it, rename it, document it better, profile it, or rewrite it to preserve exact output, but you cannot simply eliminate it and still claim exact equivalence.

---

## Refactor Plan For `merging_diag()` Under The Chosen Direction

The earlier draft separated an exact-results cleanup branch from a complete-removal branch.
After the user decisions, the concrete plan is now different:

- keep `merging_diag()` active in the default path,
- isolate it into a clearly named internal module,
- add a second raw-Hough handoff path under the same shell entrypoint,
- make later removal easy by deleting the isolated merged-Hough path when that future version is desired.

## A1. Keep the behavior, but isolate the heuristic

Move the exact existing merge logic into a more explicit internal module, for example:

- `hough_postprocessing/greedy_diagonal_segment_merging.py`

The current Hough detector module should keep a thin compatibility wrapper and preserve the exact current outputs.

## A2. Keep the exact same default input/output contract

The detector should still expose the same default production payload fields:

- `raw_lines`
- `merged_lines`
- `threshold_start`
- `mask`

and `line_alignment_pipeline.py` should still forward `merged_lines` in the default mode.

## A3. Add a second handoff mode without changing the default mode

The pipeline should gain a second internal/CLI-selectable handoff mode under the same shell entrypoint, conceptually:

- `merged_hough_to_true_iou` -> current default behavior
- `raw_hough_to_true_iou` -> alternate comparison behavior

The important constraint is:

- the default remains the current merged-Hough handoff,
- the alternate raw-Hough mode uses the exact same downstream true-IoU filter implementation,
- the rest of the pipeline remains shared.

This is what makes `merging_diag()` easily removable later:

- when the project is ready to drop it, the future change is only a default switch plus cleanup of the isolated merged-Hough path.

## A4. Prepare the Cython boundary around the isolated merge module

Because `merging_diag()` is a known Python-side bottleneck and will remain part of the default path for now, the right Cython stance is:

- isolate the exact merge behavior first,
- accelerate that isolated behavior second,
- keep the raw-Hough alternate path free of any dependency on the merge accelerator.

That preserves exact current results in the default path while also keeping future removal simple.

## A5. Use this refactor for maintainability and future removability

The benefits of this approach are:

- smaller Hough module,
- much clearer separation between:
  - preprocessing,
  - raw probabilistic Hough,
  - post-Hough greedy segment merge,
  - handoff selection into true-IoU filtering,
- easier profiling and documentation,
- exact current default output preserved,
- future `merging_diag()` removal becomes a contained change instead of a cross-cutting rewrite.

---

## Inventory: Active, Optional, Dormant, Legacy

## 1. Active production modules

These are on the active shell-entry path and should be treated as core production code:

- `text_metrics_report.py`
- `run_text_metrics_report.sh`
- `pipeline/run_text_metrics_pipeline.py`
- `pipeline/parse_text_metrics_report_args.py`
- `pipeline/resolve_text_metrics_input_sources.py`
- `pipeline/load_or_compute_score_matrices.py`
- `pipeline/process_single_document_metrics.py`
- `parallelisation/execute_document_tasks.py`
- `parallelisation/record_parallel_progress.py`
- `parallelisation/write_parallel_report_files.py`
- `line_alignment_pipeline.py`
- `hough_line_transform_endpoints_line_direction_30_to_60_degrees.py`
- `line_endpoint_records.py`
- `line_filtering_v2_12_IoU.py`
- `line_metric_bundle.py`
- `line_coverage_subtract.py`
- `levenshtein_metric.py`
- `score_matrix_builder.py`
- `score_stream_index.py`
- `runfile_records.py`
- `pipeline/report_item_views.py`

## 2. Optional active modules

These are not always imported, but they are part of the supported runtime behavior.

### `--debug`

- `debug/per_document_stage_timing.py`
- `debug/run_timing_telemetry.py`

### `--with-visuals`

- `visualisation/render_text_metrics_visualisations.py`
- `visualisation/render_alignment_matrix_views.py`
- `visualisation/render_line_coverage_views.py`

### Per-document tuned Hough mode

- `pipeline/load_hough_params_per_document.py`

These are not dead code. They are optional runtime branches.

## 3. Debug module that should be lazy-imported under `--debug`

### `debug/line_filtering_v2_12_detailed_iou_analysis.py`

This file is explicitly documented as debug-only:

- [debug/line_filtering_v2_12_detailed_iou_analysis.py:1](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/debug/line_filtering_v2_12_detailed_iou_analysis.py:1)

Within `text_metrics_v2_12_parallel/` itself, the normal production runtime does not import it today.

That is the right basic direction, but the refactor plan should tighten it:

- the module should remain in `debug/`,
- it should be **lazy-imported only after `--debug` is known to be enabled**,
- it should reuse the shared production filter helpers exactly as it already does,
- it should never duplicate filtering logic.

### How it helps when `--debug` is enabled

This module is valuable because it can produce detailed filter diagnostics that the main production path intentionally does not carry:

- `candidate_lines`
- `candidate_coverages`
- full `pairwise_iou` payloads
- connected `components`
- `merged_coverages`

That makes it the correct place to explain difficult filtering behavior on pathological documents.

### Important operational constraint

The module can be extremely heavy because `pairwise_iou` can become very large.
So the plan should be:

- import it lazily only in debug mode,
- keep the production non-debug path free of it,
- and wire its execution carefully so debug users can request or receive detailed analysis without forcing that cost into normal runs.

This keeps the only public entrypoint stable while still making deep filter diagnostics available from that same entrypoint when `--debug` is enabled.

## 4. Runtime artifacts currently sitting inside the source tree

These are not source code and should not live next to maintained code:

- `.score_index_cache/*.index.pkl`
- `__pycache__/...`

Recommended action:

- remove them from the source tree,
- keep caching as a supported runtime feature,
- relocate cache storage outside the source tree,
- key the cache by the relevant runtime inputs so reruns of the same documents stay fast,
- ensure `.gitignore` excludes the runtime cache location and `__pycache__`.

These changes are safe and do not affect logic as long as the cache content is treated as a runtime optimization rather than a source artifact.

## 5. Static documentation artifacts with stale naming

The `diagrams/` directory contains artifacts named `v2_1_parallel` even though this directory is `v2_12_parallel`.

Examples:

- `diagrams/pipeline_parallel_text_metrics_v2_1_parallel.*`
- `diagrams/pipeline_sequential_text_metrics_v2_1_parallel.*`

These are not runtime code, but they are stale and confusing.

Recommended action:

- remove `diagrams/` from the source tree during the refactor,
- add it to `.gitignore` together with all runtime cache artifacts,
- regenerate the diagrams only after the full refactor is complete.

---

## Legacy Or Compatibility Surfaces Still Present

These are the most important cleanup candidates that are still in active source.

## A. Remove the legacy CLI alias: `--scores-pkl`

In [pipeline/parse_text_metrics_report_args.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/parse_text_metrics_report_args.py), this is explicitly marked as a legacy alias:

- [parse_text_metrics_report_args.py:28](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/parse_text_metrics_report_args.py:28)

and handled in:

- [resolve_text_metrics_input_sources.py:54](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/resolve_text_metrics_input_sources.py:54)
- [resolve_text_metrics_input_sources.py:55](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/resolve_text_metrics_input_sources.py:55)

The chosen direction is to remove this alias and keep the explicit score-input flags only:

- `--scores-pkl-ref-to-ref`
- `--scores-pkl-ref-to-pred`
- `--scores-pkl-ref-to-adjusted-pred`

Recommended action:

- remove `--scores-pkl`,
- wire the explicit flags through the resolver in the most direct and maintainable way,
- preserve current `.pkl`-based behavior exactly,
- keep `run_text_metrics_report.sh` as the only supported public entrypoint for those inputs.

## B. Legacy coverage fields in `line_metric_bundle.py`

These fields are explicitly marked legacy:

- `x_char_intervals_coverage_legacy`
- `y_char_intervals_coverage_legacy`

at:

- [line_metric_bundle.py:93](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_metric_bundle.py:93)
- [line_metric_bundle.py:98](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_metric_bundle.py:98)
- [line_metric_bundle.py:104](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_metric_bundle.py:104)
- [line_metric_bundle.py:120](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_metric_bundle.py:120)
- [line_metric_bundle.py:121](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_metric_bundle.py:121)

These are still actively consumed by:

- [line_coverage_subtract.py:100](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_coverage_subtract.py:100)
- [line_coverage_subtract.py:104](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_coverage_subtract.py:104)
- [line_coverage_subtract.py:108](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_coverage_subtract.py:108)

So they are not removable yet.

Recommended action:

- replace the current legacy coverage representation with an equivalent non-legacy representation,
- preserve the exact current coverage semantics,
- verify exact output equality against the current implementation before removing the legacy fields.

Until that exact-equality replacement is implemented and verified, these fields are active and must be treated as behavior-bearing.

## C. `v2_1` labels still embedded in outputs and visuals

Examples:

- [process_single_document_metrics.py:220](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/process_single_document_metrics.py:220)
- [process_single_document_metrics.py:235](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/process_single_document_metrics.py:235)
- [process_single_document_metrics.py:323](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/process_single_document_metrics.py:323)
- [run_text_metrics_pipeline.py:279](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/run_text_metrics_pipeline.py:279)
- [visualisation/render_text_metrics_visualisations.py:122](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/visualisation/render_text_metrics_visualisations.py:122)

These are compatibility labels, not algorithmic requirements.

Recommended action:

- rename these labels consistently to `v2_12`,
- clean the stale `v2_1` references in outputs, visuals, comments, and docstrings,
- preserve the underlying logic and metrics exactly while updating the naming.

## D. Stale v2.1 references in comments and docstrings

Examples:

- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:4](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:4)
- diagram filenames and titles in `diagrams/`

Recommended action:

- safe documentation-only cleanup.

## E. Unnecessary import-path hack in `line_endpoint_records.py`

This file still mutates `sys.path` locally:

- [line_endpoint_records.py:1](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_endpoint_records.py:1)

This is a classic maintenance smell. The current pipeline imports this module from inside the package tree already.

Recommended action:

- remove the local `sys.path` insertion after one smoke pass confirms nothing relies on it.

This should not affect results.

## F. Shell/Python default drift

The shell wrapper defaults and direct Python parser defaults do not fully match.

Examples:

- shell defaults:
  - [run_text_metrics_report.sh:37](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/run_text_metrics_report.sh:37)
  - [run_text_metrics_report.sh:41](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/run_text_metrics_report.sh:41)
  - [run_text_metrics_report.sh:45](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/run_text_metrics_report.sh:45)
- Python parser defaults:
  - [parse_text_metrics_report_args.py:60](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/parse_text_metrics_report_args.py:60)
  - [parse_text_metrics_report_args.py:89](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/parse_text_metrics_report_args.py:89)
  - [parse_text_metrics_report_args.py:99](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/parse_text_metrics_report_args.py:99)

This is a maintainability hazard because direct `python3 text_metrics_report.py` and `run_text_metrics_report.sh` are not equivalent unless explicit params are passed.

Recommended action:

- treat `run_text_metrics_report.sh` as the only supported public entrypoint,
- align the internal Python argument handling to that shell entrypoint,
- treat direct `python3 text_metrics_report.py` invocation as an internal/developer path rather than a public contract.

This keeps one authoritative surface for users while still preserving the internal Python structure.

---

## Safe Cleanup Opportunities That Should Preserve Exact Results

## 1. Remove runtime artifacts from the source tree

Safe, no logic change:

- remove `.score_index_cache/` from the source directory,
- remove `__pycache__/` from the source directory,
- keep caches under `results/.score_index_cache` or another runtime-only location.

## 2. Remove `sys.path` mutation from `line_endpoint_records.py`

Safe after smoke validation.

## 3. Fix stale comments, docstrings, and diagram names

Safe, no logic change.

## 4. Keep optional debug logic under `debug/` and make that boundary explicit

This is already mostly true in v2.12 and is a strength of the current layout.

Recommended refinement:

- keep production logic in production modules,
- keep manual heavy analysis in `debug/`,
- lazy-import `debug/line_filtering_v2_12_detailed_iou_analysis.py` only when `--debug` is enabled,
- keep its filtering math shared with production helpers rather than duplicated.

## 5. Reduce non-debug success-spool payload size

Current behavior:

- `process_item()` still builds a rich internal result object,
- `record_parallel_progress.py` writes most of it into the success spool,
- then `report_item_views.py` projects a smaller non-debug public view only at final report-writing time.

Relevant files:

- [process_single_document_metrics.py:281](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/process_single_document_metrics.py:281)
- [record_parallel_progress.py:75](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/parallelisation/record_parallel_progress.py:75)
- [report_item_views.py:35](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/report_item_views.py:35)

This creates extra JSONL I/O and larger temporary spools even when:

- `--debug` is off,
- `--with-visuals` is off.

Recommended no-result-change optimization:

- split internal success payloads into:
  - a **minimal non-debug spool schema**, and
  - a **rich debug/visual schema** only when needed.

This can reduce disk I/O and peak JSON parsing cost without changing final outputs.

This needs careful validation because:

- run-level averages still depend on `normalized_levenshtein_before` and `average_normalized_levenshtein_along_lines`,
- final report projection still needs its current input fields.

## 6. Keep package structure but add clearer internal module boundaries

The directory can become more descriptive without changing behavior by using internal subpackages and compatibility wrappers.

Example target structure:

- `pipeline/`
- `parallelisation/`
- `debug/`
- `visualisation/`
- `shared/`
- `alignment_utils/`
- `hough_detection/`
- `filtering/`
- `metrics/`
- `coverage/`
- `io/`
- `reporting/`

Then keep thin compatibility wrapper files at the current top-level import paths until migration is complete.

This would improve readability without forcing immediate caller changes.

---

## Performance And Memory Opportunities That Can Preserve Results

## 1. Keep optimizing the Hough side, not the true-IoU filter first

The v2.12 true-IoU filter is already substantially cleaner and more optimized than older versions.
The likely remaining runtime hotspot in many difficult documents is the Hough stage and the post-Hough merge stage, not the current production true-IoU filter.

Recommended action:

- benchmark Hough preprocessing,
- benchmark the full Hough-to-handoff path other than the already-known `merging_diag()` bottleneck,
- benchmark line conversion and downstream filter separately,
- benchmark the Python-side pieces around the compiled `skimage` probabilistic Hough call rather than expecting to speed up `skimage` itself.

There is no need to spend time proving that `merging_diag()` is already a current bottleneck. The more useful profiling work is to discover what comes next around it and after it.

## 2. Consider worker-local score-stream handle reuse

Current matrix loading opens and seeks into `.pkl` files per fetch:

- [load_or_compute_score_matrices.py:70](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/pipeline/load_or_compute_score_matrices.py:70)
- [score_stream_index.py:94](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/score_stream_index.py:94)

Potential optimization:

- one file handle per worker per score-stream path,
- reused across many items.

This can improve I/O overhead without changing matrix contents.

This must be implemented worker-locally to avoid cross-process file-handle issues.

## 3. Carefully vectorize geometry sampling only if exact rounding semantics are preserved

Potential candidates:

- `alignment_utils/line_geometry_support.py`
- `_build_line_coverage()` in `line_filtering_v2_12_IoU.py`

But this is subtle because:

- rounding semantics,
- clipping semantics,
- empty-span semantics,
- fit behavior

must remain bit-for-bit or at least output-equivalent.

So this is a benchmark-driven optimization candidate, not an immediate cleanup.

## 4. Keep the current one-path executor design

This is already a strong property of v2.12:

- one worker and many workers share the same scheduler path.

That logic lives in:

- [parallelisation/execute_document_tasks.py:1](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/parallelisation/execute_document_tasks.py:1)

This should be preserved.

---

## Highly Detailed Recommended Implementation Order

## Phase -1: Create the rollback snapshot

Before the first code edit:

1. create the full snapshot under `/scratch/project_2017385/dorian/Churro_copy/_unused_backup_20260423_151633/`,
2. use a timestamped snapshot directory name,
3. verify that the full `text_metrics_v2_12_parallel/` tree is restorable from that snapshot.

## Phase 0: Freeze default-path behavior

Before touching code:

1. one-document exact regression,
2. 20-document tuned-Hough regression,
3. one debug+visual regression,
4. one pathological Hough-heavy regression.

Artifacts to compare:

- `report.json`
- `report_skipped_empty_prediction.json`
- `report_failed_items.json`
- `report_timings.json` when `--debug`

## Phase 1: Make the shell script the explicit public contract

1. document `run_text_metrics_report.sh` as the only supported public entrypoint,
2. keep direct Python entrypoints internal,
3. align the internal argument plumbing with the shell contract.

## Phase 2: Repository hygiene only

Safe, zero-logic work:

1. move cache storage outside the source tree,
2. remove `__pycache__` from the source tree,
3. ensure ignore rules cover caches, `__pycache__`, and `diagrams/`,
4. remove `diagrams/` from the source tree,
5. defer diagram regeneration until the refactor is complete.

## Phase 3: Documentation and naming cleanup only

Safe, zero-logic work:

1. rename stale `v2_1` labels to `v2_12`,
2. fix stale comments and docstrings,
3. document that `merging_diag()` is active default production logic,
4. document the future dual-handoff design,
5. document the debug-only role of `debug/line_filtering_v2_12_detailed_iou_analysis.py`.

## Phase 4: Remove no-op import/path hacks

1. remove `sys.path` manipulation from `line_endpoint_records.py`,
2. run a smoke pass,
3. verify that results and reports remain unchanged.

## Phase 5: Input-surface cleanup

1. remove `--scores-pkl`,
2. keep the explicit `.pkl` flags,
3. wire them through the resolver in the cleanest direct way,
4. preserve current `.pkl`-based behavior exactly.

## Phase 6: Non-debug payload slimming

1. define a minimal internal success schema for non-debug runs,
2. keep current rich schema for debug/visual runs,
3. preserve report output exactly,
4. rerun exact regressions.

This is one of the safest meaningful performance cleanups available today.

## Phase 7: Structural refactor with compatibility wrappers

1. introduce clearer subpackages,
2. move large modules behind them,
3. keep old import paths as thin wrappers during migration,
4. preserve the single production entrypoint,
5. rerun exact regressions.

## Phase 8: Isolate `merging_diag()` and introduce the dual handoff

1. move the exact current merge heuristic into its own internal module,
2. preserve the current merged-Hough default path exactly,
3. add the alternate raw-Hough handoff path into the same true-IoU filter,
4. keep the default as current behavior,
5. make later removal of the merged-Hough path a contained future change.

## Phase 9: Replace the legacy coverage representation

1. introduce a non-legacy internal coverage representation,
2. keep the exact current coverage semantics,
3. prove exact equality before deleting the legacy fields.

## Phase 10: Add exact-result Cython accelerators

1. start with isolated `merging_diag()` helper kernels,
2. continue with true-IoU path sampling / assignment kernels,
3. then move to the agreed medium-priority kernels,
4. verify exact default-path equality after each accelerator layer.

---

## What I Would Actually Do

If I were implementing the refactor under your stated constraints, I would do this:

1. create the rollback snapshot first,
2. freeze current default-path outputs with regression fixtures,
3. make `run_text_metrics_report.sh` the explicit and documented public contract,
4. move cache and diagrams out of the source tree,
5. clean stale names and comments,
6. remove the `sys.path` hack,
7. remove `--scores-pkl` and keep the explicit `.pkl` flags,
8. slim non-debug success payloads,
9. isolate `merging_diag()` into a clearly named internal Hough postprocessing module,
10. add the alternate raw-Hough handoff path under the same entrypoint while keeping the current merged-Hough path as the default,
11. replace the legacy coverage representation with an exact-equivalent non-legacy form,
12. add exact-result Cython accelerators in the agreed priority order.

That gives:

- exact output preservation in the default path,
- one public entrypoint,
- easier future removal of `merging_diag()`,
- cleaner runtime/debug separation,
- likely lower I/O overhead,
- better structure for future Cython acceleration.

---

## Final Recommendation

Treat the work as **one exact-results-preserving refactor with two supported handoff modes**, not as a cleanup track plus a removal track.

### Default supported mode

- current merged-Hough handoff into true-IoU
- exact current logic and exact current results preserved

### Alternate supported mode

- raw-Hough handoff into the same true-IoU filter
- present for comparison, validation, and future migration work
- not the default during this refactor

### Refactor rule

- all cleanup, restructuring, cache relocation, naming cleanup, non-debug payload slimming, and Cython work must preserve the exact default-mode results.


---

## Cython Analysis: Where It Can Help Without Changing Results

This section answers a separate question:

- if we keep the **exact same logic**,
- return the **exact same results**,
- and only want speed / RAM / memory-management improvements,

where would Cython be worth using inside `text_metrics_v2_12_parallel`?

## First principle: do **not** Cythonize everything

The current pipeline contains several different kinds of work:

1. heavy external-library work already done in compiled code,
2. Python orchestration and JSON/report I/O,
3. tight Python numeric loops over many small objects,
4. nested Python loops that repeatedly allocate dictionaries/lists/sets.

Cython helps most in category 3 and sometimes category 4.
It helps much less in category 1 and category 2.

That means the best strategy is:

- keep the Python modules as the readable reference implementation,
- add **small, carefully chosen Cython kernels** behind stable helper APIs,
- require exact regression equality against the Python reference path.

---

## Where Cython would likely help the most

## Tier 1: Highest-value exact-result Cython candidates

These are the strongest candidates.

### A. `merging_diag()` and its helper chain

File:

- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py)

Relevant functions:

- `line_magnitude`
- `point_line_distance`
- `count_points_in_range`
- `sample_line_pixels`
- `longest_false_run`
- `segment_length`
- `segment_angle`
- `nearest_endpoints`
- `bridge_stats`
- `merging_diag`

### Why this is a strong Cython target

This code is exactly the kind of workload Cython is good at:

- many short numeric helper calls,
- repeated Euclidean-distance math,
- nested loops over points and segments,
- repeated branching on floats and ints,
- repeated Python tuple unpacking,
- repeated temporary-object creation.

When difficult documents create many raw Hough segments, the Python overhead of these loops can become significant.

### Why exact-result preservation is realistic here

This is a good exact-preserving target because the algorithm is already explicit and local.
A Cython version can preserve:

- the same sort order,
- the same candidate order,
- the same strict `>` tie handling,
- the same merge gate thresholds,
- the same distance threshold `20`,
- the same endpoint pairing order,
- the same return schema.

The implementation challenge is not conceptual ambiguity. It is mostly careful transcription.

### Memory/RAM benefits

A Cython implementation can reduce:

- temporary Python tuple creation,
- Python call overhead between helpers,
- Python object churn in tight loops.

If implemented with typed local variables and contiguous arrays for points, it should also reduce pressure on Python’s allocator.

### Recommended structure

Do **not** replace the readable Python file directly.
Instead:

1. keep the current Python implementation as the reference,
2. create a small accelerator module such as:
   - `accelerators/hough_merge_diag_exact.pyx`
3. expose one wrapper function with the exact same input/output behavior,
4. default to the Cython path only when the compiled extension is available,
5. keep the Python path as the authoritative fallback,
6. keep the raw-Hough alternate handoff path independent from the merge accelerator so future `merging_diag()` removal stays simple.

This gives:

- exact reference semantics,
- clearer benchmarking,
- easier rollback,
- easier debugging.

### Overall recommendation

If only **one** part of the current Hough-side Python logic were to be Cythonized first, this is the first place I would choose, because it remains part of the exact-result default path and is already understood to be a current bottleneck.

---

### B. Line sampling kernels used by the true-IoU filter

Files:

- [alignment_utils/line_geometry_support.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/alignment_utils/line_geometry_support.py)
- [line_filtering_v2_12_IoU.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_filtering_v2_12_IoU.py)

Relevant functions:

- `line_y_at_x`
- `mean_line_support`
- `_build_line_coverage`
- `_build_coverage_indices_by_prediction_column`
- `_compute_final_assignment`
- parts of `_merge_component`

### Why this is a strong Cython target

The current filter builds and consumes many per-column paths.
This involves:

- per-column interpolation,
- rounding and clipping,
- per-column score reads,
- repeated dictionary lookups,
- repeated small loops across many coverages.

This is already much cleaner in v2.12 than in older versions, but the low-level operations are still Python-heavy.

### What Cython could improve

A carefully designed internal representation could reduce:

- `dict[int, int]` and `dict[int, float]` overhead,
- Python loop overhead for path sampling,
- repeated creation of tiny tuples and temporary keys.

### Exact-result challenge

This is more delicate than `merging_diag()` because the filter’s correctness depends on exact local tie-break behavior.
To preserve exact results, the Cython implementation must preserve:

- exact `round(...)` semantics,
- exact `np.clip`-equivalent clipping,
- exact traversal order,
- exact `_local_path_key(...)` ordering,
- exact `>` / `<` comparison behavior,
- exact stable ordering of surviving lines.

So this is feasible, but only if the Cython layer is treated as an exact reimplementation, not as a new data structure experiment.

### Best target subset

The best entry point is not the whole filter module.
It is the most repetitive kernels:

1. path construction in `_build_line_coverage`,
2. local winner accumulation in `_merge_component`,
3. per-column owner scan in `_compute_final_assignment`.

### Overall recommendation

This is the second-best Cython target after `merging_diag()`.
It is especially attractive because the agreed refactor targets all of the high and medium recommendation kernels as long as exact results remain unchanged.

---

## Tier 2: Medium-value Cython candidates

### C. `line_metric_bundle.build_line_metric_bundle`

File:

- [line_metric_bundle.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/line_metric_bundle.py)

### Why it may help

This function does a lot of per-line list building and interval construction:

- scans `mapped_line_id`,
- rounds `mapped_y`,
- constructs ordered row sequences,
- builds merged intervals.

If there are many lines and many columns, this can add overhead.

### Why it is not the first choice

Compared with Hough and filtering, this code is less likely to dominate total runtime in the hardest documents.
Also, much of the logic is list/dict/interval assembly, where Cython helps less unless the whole data representation is redesigned.

### Overall recommendation

Worth considering later, but not a first-wave Cython target.

---

### D. `shared/project_line_to_text_windows.py`

File:

- [shared/project_line_to_text_windows.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/shared/project_line_to_text_windows.py)

### Why it may help

This code performs repeated sampling and interval merging for line coverage projection.
Those operations are simple and numeric.

### Why it is not top priority

It is active, but it is part of the current “legacy coverage” path rather than the main true-IoU decision core.
Unless profiling shows it dominates time, it is a secondary optimization target.

---

## Tier 3: Low-value or poor Cython targets

### E. `probabilistic_hough_line(...)` call site itself

Files:

- [hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:379](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/hough_line_transform_endpoints_line_direction_30_to_60_degrees.py:379)
- installed `skimage` wrapper:
  - [hough_transform.py:250](/users/dobeli/.local/lib/python3.12/site-packages/skimage/transform/hough_transform.py:250)

This is **not** a good Cython target in this project because the heavy work is already happening in `skimage`’s compiled implementation.
Wrapping that call in Cython will not materially speed it up.

### Recommendation

Do not spend time Cythonizing the call site.
If raw probabilistic Hough itself becomes the bottleneck, the only meaningful change would be:

- modifying or replacing the underlying `skimage` implementation,
- which is a much larger and riskier project.

---

### F. `levenshtein_metric.py`

File:

- [levenshtein_metric.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/levenshtein_metric.py)

This is already using the rapidfuzz C backend:

- `LEVENSHTEIN_BACKEND = "c"`

The expensive part is already in compiled code.

### Recommendation

Do not Cythonize this.
There is almost no upside.

---

### G. `score_matrix_builder.compute_score_matrix`

File:

- [score_matrix_builder.py](/scratch/project_2017385/dorian/Churro_copy/text_metrics_v2_12_parallel/score_matrix_builder.py)

### Why it looks tempting

It has nested Python loops over all ref/pred window pairs.

### Why Cython may disappoint here

Each inner iteration calls:

- `sacrebleu.sentence_chrf(...)`

That is a Python-level external function call.
So even if the surrounding loops were in Cython, the hot cost would still mostly be the repeated Python function invocation.

### Recommendation

Do not prioritize this for Cython unless profiling shows the loop overhead itself is large enough to matter.
A much bigger gain would likely require:

- batching,
- a lower-level chrf implementation,
- or a redesigned matrix-computation strategy.

That would be a different project.

---

### H. JSON/report writing and shell/arg parsing

Files:

- `parallelisation/write_parallel_report_files.py`
- `pipeline/parse_text_metrics_report_args.py`
- `run_text_metrics_report.sh`

These are not worthwhile Cython targets.
They are I/O-bound, serialization-bound, or shell-bound.

---

### I. Visualisation modules

Files:

- `visualisation/render_alignment_matrix_views.py`
- `visualisation/render_line_coverage_views.py`
- `visualisation/render_text_metrics_visualisations.py`

These are optional, dominated by Matplotlib and file I/O, and are not good Cython targets for this pipeline.

---

## Where Cython Can Improve RAM / Memory Behavior

Cython helps memory behavior most when it lets us replace large numbers of Python objects with typed arrays or compact loop-local primitives.

In this codebase, the strongest opportunities are:

### 1. `merging_diag()` point/segment math

Current behavior allocates and traverses many Python tuples/lists.
A Cython kernel could instead use:

- typed `double` coordinates,
- typed `Py_ssize_t` loop indices,
- typed views over point arrays,
- boolean or integer arrays for mask samples.

This can reduce:

- Python allocator pressure,
- intermediate object churn,
- GC pressure.

### 2. True-IoU line path sampling

Current behavior stores per-line path state in Python dictionaries:

- `x_to_y`
- `x_to_score`

If a Cython internal path kernel used compact arrays during construction before converting back to the exact Python output shape, it could reduce peak temporary allocations.

### Important caution

If the public/internal contract still expects Python dicts and sets afterward, the memory gain is only partial.
The biggest RAM improvement would require a deeper internal representation change.
That is feasible, but it increases implementation risk.

---

## Exactness Requirements For Any Cython Reimplementation

If the goal is **exact same results**, then every Cython kernel must preserve:

1. sort order,
2. traversal order,
3. tie-break order,
4. rounding behavior,
5. clipping behavior,
6. float comparison behavior,
7. empty-input behavior,
8. public dict/list ordering where callers rely on it,
9. the current fallback branches.

### Very important examples

- `merging_diag()` must preserve the exact candidate ordering:
  - `(prev_p0, prev_p1)`
  - `(prev_p0, cur_p1)`
  - `(cur_p0, prev_p1)`
  - `(cur_p0, cur_p1)`
- `_merge_component()` must preserve “first coverage that reaches best local key wins” semantics.
- `_compute_final_assignment()` must preserve its exact owner key ordering, including the final line-length and negative-index tie-break terms.
- `line_y_at_x()` and `_build_line_coverage()` must preserve the current interpolation + `round(...)` + clipping semantics exactly.

If any of those change, the outputs can drift.

---

## Best Architectural Pattern For Cython In This Project

If Cython is introduced, the cleanest structure would be:

- keep pure Python reference modules,
- add a small `accelerators/` package,
- keep each accelerator narrowly scoped,
- expose a stable Python wrapper that chooses:
  - compiled fast path when available,
  - Python reference path otherwise.

Example possible layout:

- `accelerators/hough_merge_diag_exact.pyx`
- `accelerators/filter_path_sampling_exact.pyx`
- `accelerators/filter_assignment_exact.pyx`
- `accelerators/__init__.py`

The public production modules would remain readable and could call these helpers through a thin compatibility layer.

This preserves:

- maintainability,
- readability,
- portability,
- exact regression comparison against Python reference code.

---

## Recommended Cython Order

If I had to prioritize exact-result Cython work in this codebase, I would do it in this order:

1. `merging_diag()` and its helper chain
2. true-IoU path sampling / local winner kernels
3. final ownership assignment kernel
4. bundle-building interval helpers
5. `shared/project_line_to_text_windows.py`
6. only then consider anything else

This ordering is also the current agreed implementation scope:

- Tier 1 targets are in scope,
- Tier 2 targets are in scope,
- Tier 3 targets are intentionally not part of the planned implementation unless later evidence forces a revisit.

I would **not** start with:

- `probabilistic_hough_line` wrapper code,
- Levenshtein,
- JSON/report writing,
- visuals,
- CLI parsing.

---

## Final Cython Recommendation

If the objective is:

- exact same logic,
- exact same outputs,
- cleaner organization,
- real speed/RAM gains,

then the best Cython strategy is:

1. leave the public contract of `run_text_metrics_report.sh` alone,
2. keep Python modules as the exact readable reference implementation,
3. isolate the merged-Hough and raw-Hough handoff selection behind shared pipeline logic,
4. add Cython only for the tight numeric kernels that are currently Python-loop-heavy,
5. enforce regression equality against the Python reference path for every accelerator,
6. start with `merging_diag()` because it remains part of the exact-results default production algorithm,
7. continue with the agreed high and medium priority kernels, but do not spend effort trying to out-optimize `skimage` PPHT or rapidfuzz Levenshtein.

That is the highest-precision, lowest-risk path.

