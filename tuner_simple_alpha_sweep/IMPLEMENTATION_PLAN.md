# tuner_simple_alpha_sweep Implementation Plan

## Goal

Create a behavior-preserving copy of `tuner_simple` that tries several score-floor alpha values per document, computes the normal Hough/filtering/scoring pipeline for each alpha, selects the best alpha by the harmonic metric below, and writes/plots only the selected result.

The original `tuner_simple` directory must remain unchanged. Existing single-alpha behavior must remain reproducible in this copied tree when the alpha sweep is disabled or reduced to one alpha.

Selection metric:

```text
harmonic_selection_score =
    3 / (
        (1 / weighted_along_lines_normalised_levenshtein)
        + (1 / correct_ref_coverage)
        + (1 / (1 - hallucination))
    )
```

Alpha grid:

```text
1.0, 1.2, 1.4, ..., 3.8, 4.0
```

Build it with integer ticks, not cumulative float addition:

```python
alpha_values = [tick / 10.0 for tick in range(10, 41, 2)]
```

This gives exactly 16 candidates and avoids values like `1.5999999999999999`.

## Current Behavior

The current copied code is a direct copy of `tuner_simple`, but it is not yet an independent runnable fork because most imports still refer to the package name `tuner_simple`.

Important current flow:

1. `run_tunner.sh` locates `run_tuner_simple.py`, builds Cython accelerators with `cython_accel/build.py`, then runs Python.
2. `run_tuner_simple.py` inserts the project parent directory into `sys.path`, imports `tuner_simple.config.cli_arguments`, and calls `run_simple_tuner`.
3. `serial_runner/pipeline_runner.py` selects documents, builds score-matrix indexes once, then processes each document with `process_one_document`.
4. `serial_runner/document_runner.py` loads the ref-to-pred and ref-to-ref matrices once per document.
5. For a single configured `score_floor_alpha`, it computes:
   - ref-to-pred score-floor mask;
   - ref-to-ref score-floor mask;
   - ref-to-pred Hough/filtering;
   - line-level NLS filtering;
   - ref-to-ref Hough/filtering only if ref-to-pred has surviving lines;
   - final metrics.
6. It builds one result row and one plot payload from that single result.
7. `plotting/stitched_language_panels.py` currently renders one hidden per-document PNG immediately, stores the hidden PNG path, stitches those hidden PNGs at the end, then deletes the hidden panel directory unless document-grid mode is enabled.
8. `results_writing/flat_csv_tables.py` writes one result row per processed document.

Current result semantics must remain unchanged for any chosen alpha. The sweep changes only which alpha result is chosen for each document.

## Import Isolation Guard

Before implementing sweep logic, make this copied directory run its own modules.

Problem:

`run_tuner_simple.py` currently imports `tuner_simple...`. When run from `tuner_simple_alpha_sweep`, `PROJECT_DIR` is `/scratch/project_2017385/dorian/Churro_copy`, so Python will find the original `/scratch/project_2017385/dorian/Churro_copy/tuner_simple` package first. Adding `tuner_simple_alpha_sweep` itself to `sys.path` does not make `import tuner_simple` resolve to this copy.

Recommended safe fix:

1. Rename the copied Python package imports from `tuner_simple` to `tuner_simple_alpha_sweep`.
2. Update every copied source import in this directory only.
3. Update Cython extension names in `cython_accel/build.py` from:

```python
cython_accel.ownership_core
cython_accel.filter_core
```

to package-qualified names:

```python
tuner_simple_alpha_sweep.cython_accel.ownership_core
tuner_simple_alpha_sweep.cython_accel.filter_core
```

4. Update optional accelerator imports accordingly.
5. Update `run_tunner.sh` fallback path from the original `tuner_simple` directory to this copied directory.

Do this before any behavior changes. Then run a single-alpha parity check against the original with the same command and compare `compact_combination_metrics.csv` rows.

## New Configuration

Add sweep configuration fields to `config/pipeline_config.py`:

```python
alpha_sweep_enabled: bool = True
alpha_sweep_min: float = 1.0
alpha_sweep_max: float = 4.0
alpha_sweep_step: float = 0.2
alpha_selection_metric: str = "harmonic_line_nls_ref_coverage_nonhallucination"
```

Keep `score_floor_alpha` for compatibility:

- If sweep is disabled, use the existing single-alpha behavior exactly.
- If sweep is enabled, `score_floor_alpha` may be treated as informational/default only, and the per-document selected row will store the winning alpha in the existing `score_floor_alpha` column.

Add CLI flags in `config/cli_arguments.py`:

```text
--alpha-sweep
--no-alpha-sweep
--alpha-sweep-min
--alpha-sweep-max
--alpha-sweep-step
```

Default should be decided deliberately:

- For maximum backward compatibility, default `--no-alpha-sweep` and provide a new wrapper or documented command for sweep.
- For this new copied directory's purpose, default sweep can be enabled, but then single-alpha parity tests must use `--no-alpha-sweep`.

I recommend defaulting to sweep enabled only in this copied variant, while retaining `--no-alpha-sweep` for exact old behavior.

## Per-Document Sweep Design

Refactor `process_one_document` into two layers:

1. A matrix-loading shell that runs once per document.
2. A per-alpha evaluation function that receives already loaded matrices and windows.

New helper:

```python
@dataclass
class AlphaEvaluationResult:
    alpha: float
    selection_score: float
    result_row: dict[str, Any]
    plot_payload: dict[str, Any] | None
    ref_to_pred_floor: ScoreFloorResult
    ref_to_ref_floor: ScoreFloorResult
    ref_to_pred_hough: HoughFilteredPayload
    ref_to_ref_hough: HoughFilteredPayload | None
    scored: ScoredDocumentResult
```

New function:

```python
def evaluate_document_for_alpha(
    *,
    document,
    config,
    alpha: float,
    ref_to_pred_loaded,
    ref_to_ref_loaded,
    reference_windows,
    prediction_windows,
    keep_plot_payload: bool,
    log,
) -> AlphaEvaluationResult:
```

This function should contain the current lines around:

- score-floor mask computation;
- ref-to-pred Hough/filtering;
- line NLS filtering;
- optional ref-to-ref Hough/filtering;
- `score_document_alignment`;
- `build_result_row`;
- plot-payload construction.

The only semantic difference inside this helper is that it uses `alpha` rather than `config.score_floor_alpha`. The easiest safe pattern is:

```python
alpha_config = replace(config, score_floor_alpha=float(alpha))
```

and then pass `alpha_config` into `build_result_row`.

Selection should happen after the normal metrics are computed. Do not choose from partial evidence.

## Harmonic Selection Function

Add a small function, preferably near `metrics_to_row_fields` or in a new `scoring/selection_metrics.py`:

```python
def harmonic_selection_score(metrics: DocumentAlignmentMetrics) -> float:
    line_nls = metrics.weighted_along_lines_normalised_levenshtein
    coverage = metrics.correct_ref_coverage
    non_hallucination = None if metrics.hallucination is None else 1.0 - float(metrics.hallucination)

    values = (line_nls, coverage, non_hallucination)
    if any(value is None for value in values):
        return float("-inf")
    if any(float(value) <= 0.0 for value in values):
        return float("-inf")
    return 3.0 / sum(1.0 / float(value) for value in values)
```

Tie-breakers must be deterministic. Recommended ordering:

1. Higher harmonic score.
2. Higher `correct_ref_coverage`.
3. Higher `weighted_along_lines_normalised_levenshtein`.
4. Lower `hallucination`.
5. Fewer final ref-to-pred lines, to avoid fragmentation when metrics are identical.
6. Lower alpha, to keep the earliest/simple option on exact ties.

This tie-breaker is stable and auditable.

## Memory Strategy

Per document:

1. Load both matrices once.
2. Build `reference_windows` and `prediction_windows` once.
3. Loop over 16 alpha values.
4. Keep only:
   - the current alpha result;
   - the best alpha result so far.
5. When a new alpha wins, replace the stored best result.
6. Explicitly drop losing alpha payloads before the next alpha.

Do not store all 16 plot payloads. Each plot payload includes matrices and line lists, so keeping all of them would multiply memory use.

Pseudo-flow:

```python
best_result = None
for alpha in alpha_values:
    candidate = evaluate_document_for_alpha(...)
    if best_result is None or candidate_key(candidate) > candidate_key(best_result):
        best_result = candidate
    else:
        del candidate
```

At the end of the document, return a normal `DocumentRunResult` built from `best_result`.

The main output CSV remains one row per processed document. That row contains the selected alpha in the existing `score_floor_alpha` column.

## Output Files

Main output should stay compatible:

- `best_combination_per_document.csv`: one selected result per document.
- `compact_combination_metrics.csv`: same selected rows.
- `document_type_summary.csv`: computed from selected rows only.
- `loadable_documents.csv`, `loaded_documents.csv`, `runfile_documents.csv`, `skipped_documents.csv`: unchanged semantics.
- `run_summary.json`: add sweep metadata.

Recommended new optional audit file:

```text
alpha_sweep_decisions.csv
```

One row per processed document:

```text
document_index
fname
main_language
document_type
selected_alpha
selected_harmonic_score
selected_weighted_along_lines_nls
selected_correct_ref_coverage
selected_hallucination
alpha_sweep_min
alpha_sweep_max
alpha_sweep_step
alpha_candidate_count
```

Do not write all 16 alpha rows into the main CSV outputs. The main CSVs stay best-result-only; the full set of alpha candidates is persisted in the per-document sweep pickle described below.

If strict "only save the selected result" is interpreted literally, skip `alpha_sweep_decisions.csv` and add only the selected harmonic score to `run_summary.json`. I still recommend the decisions CSV because it is tiny and makes debugging possible without storing all candidates.


## Full Sweep Pickle Output

Because this variant will be run with multiple atomic workers, and because previous `tuner_parallel_v2_2` runs already saved larger combination result sets, this copied tuner should persist the full per-document alpha sweep in a clean pickle file.

The main CSV outputs must still contain only the selected best alpha result. The pickle is an audit/debug artifact, not the public selected-result table.

Recommended path layout:

```text
<output_dir>/alpha_sweep_pickles/<language>/<document_safe_name>.pkl
```

Each worker writes exactly one pickle per processed document. The pickle contains all evaluated alpha candidates for that document, plus the selected best candidate id.

Recommended pickle schema:

```python
{
    "schema_version": "tuner_simple_alpha_sweep_document_v1",
    "document": {
        "document_index": int,
        "fname": str,
        "main_language": str,
        "document_type": str,
        "reference_text_length": int,
        "prediction_text_length": int,
        "reference_window_count": int,
        "prediction_window_count": int,
    },
    "fixed_parameters": {
        "window_size": int,
        "window_stride": int,
        "hough_threshold": int,
        "hough_line_length": int,
        "hough_line_gap": int,
        "hough_seed": int,
        "align_min_iou_threshold": float,
        "min_surviving_line_nls": float | None,
    },
    "alpha_values": [float, ...],
    "selection_metric": "harmonic_line_nls_ref_coverage_nonhallucination",
    "selected_alpha": float,
    "selected_candidate_index": int,
    "selected_harmonic_score": float,
    "candidates": [
        {
            "alpha": float,
            "harmonic_score": float,
            "result_row": dict,
            "metrics": dict,
            "coverage_diagnostics": dict,
            "ref_to_pred": {
                "score_floor": float,
                "score_mean": float,
                "score_standard_deviation": float,
                "active_cell_count": int,
                "active_fraction": float,
                "raw_line_count": int,
                "candidate_line_count": int,
                "used_line_count": int,
                "raw_hough_lines": list,
                "final_lines": list,
            },
            "ref_to_ref": {
                "score_floor": float,
                "score_mean": float,
                "score_standard_deviation": float,
                "active_cell_count": int,
                "active_fraction": float,
                "raw_line_count": int,
                "candidate_line_count": int,
                "used_line_count": int,
                "raw_hough_lines": list,
                "final_lines": list,
            } | None,
            "timings": dict,
        },
    ],
    "selected_plot_payload": {
        "result_row": dict,
        "ref_to_pred_score_matrix": numpy.ndarray,
        "ref_to_ref_score_matrix": numpy.ndarray,
        "ref_to_pred_hough_input_mask": numpy.ndarray,
        "raw_ref_to_pred_hough_lines": list,
        "final_surviving_ref_to_pred_lines": list,
    },
}
```

Store matrices only once in the pickle, preferably under `selected_plot_payload`, because plotting only needs the selected alpha. Do not duplicate full matrices inside every candidate. For non-selected candidates, keep line lists, counts, metrics, and diagnostics. If pickle size becomes too large, omit non-selected masks first because they can be reconstructed from the matrix, alpha, mean, and standard deviation.

Atomic write rule:

1. Worker writes to a temporary file in the target pickle directory.
2. Worker calls `Path.replace()` to publish the final pickle path atomically.
3. Worker records `alpha_sweep_pickle_path` in the progress CSV.
4. Worker marks the document done only after both the progress row and pickle are safely written.

The final aggregator reads the pickle for plotting/debugging. It does not choose the best alpha again unless an explicit validation mode is requested.

## Plotting Design

Requirement:

Do not plot every document and then stitch it. Keep the final language image layout as it is currently.

Current behavior renders a hidden PNG per document immediately and stitches those hidden files at the end. That still creates one per-document image internally. To satisfy the stricter requirement, introduce an in-memory stitched manager:

```python
class InMemoryStitchedPlotManager:
    def add_document_payload(self, plot_payload): ...
    def finish(self) -> list[Path]: ...
```

Implementation options:

### Option A: store payloads and render at finish

`pipeline_runner.py` calls:

```python
plotter.add_document_payload(document_result.plot_payload)
```

At `finish()`, group payloads by language, render each payload to an in-memory image, paste it into the stitched contact sheet, and save only the final stitched PNG.

Pros:

- Most faithful to "only plot at the end".
- No hidden per-document PNG files.
- Final layout can reuse the existing row/column/border math.

Cons:

- Stores all selected plot payloads until the run ends. For 30 documents this is probably fine; for very large runs it may be memory-heavy because each payload includes matrices.

### Option B: render in-memory panel immediately and store image objects

For each best document, render a panel to an in-memory PNG buffer or `PIL.Image`, keep that image object/pathless buffer, then stitch at finish.

Pros:

- Does not keep score matrices for the whole run.
- Still saves only the final stitched image.

Cons:

- It still "plots" per document internally, just not to disk. If the user's wording is interpreted strictly as no document rendering until the end, use Option A.

Recommended:

Use Option A first because selected document count is currently small and the requirement explicitly says everything needed can be kept in memory until the document is finished. If memory becomes a problem, switch to Option B while still saving only the stitched image.

Required renderer change:

`document_panel_renderer.render_document_panel` currently saves to an `output_path`. Add:

```python
def render_document_panel_to_image(...):
    ...
    return PIL.Image
```

or render to `io.BytesIO` and reopen as `PIL.Image`. Then `save_stitched_language_image` should accept either paths or already-open panel images. Preserve:

- border width;
- gap;
- panel column count;
- stitched filename;
- language grouping;
- document order.

## Cython Acceleration Plan

The existing copy already has Cython accelerators for important inner loops:

- `cython_accel/ownership_core.pyx` accelerates candidate-column ownership scans.
- `cython_accel/filter_core.pyx` accelerates line path sampling, final assignment from coverages, set IoU, and mean line support.

The new alpha loop adds repeated work. The biggest repeated loops are:

1. Score-floor mask creation for each alpha.
2. Hough detection over each alpha mask.
3. Filtering/coverage construction over each alpha's raw Hough results.
4. Coverage metrics and hallucination count accumulation.

Hough itself is inside `skimage.transform.probabilistic_hough_line`, so we cannot Cythonize that without replacing the core algorithm. The practical Cython targets are the Python/NumPy loops around it.

### Target 1: score-floor mask generation across all alphas

Current code calls `compute_score_floor_mask(matrix, alpha)` once per alpha and recomputes:

```text
mean, standard deviation, floor, mask
```

For a sweep, compute matrix mean/std once, then generate all masks from those statistics.

Add a helper:

```python
def compute_score_floor_masks_for_alphas(matrix, alpha_values) -> dict[float, ScoreFloorResult]:
```

Cython option:

Create `cython_accel/score_floor_sweep.pyx` that takes:

```text
double[:, ::1] matrix
double[::1] alpha_values
double mean
double std
```

and returns active counts and boolean masks. This is a simple matrix-size x alpha-count loop and is a good Cython candidate.

Behavior guard:

For each alpha, the result must exactly match the existing `compute_score_floor_mask(matrix, alpha)` output.

### Target 2: harmonic score calculation and candidate selection

This is tiny and does not need Cython. Keep it in Python for clarity.

### Target 3: coverage-count accumulation

`coverage_count_metrics.py` has interval accumulation loops. They are small compared with Hough/filtering, but they run once per alpha and can be moved into Cython later if profiling shows they matter.

Do not start here. It is easy to preserve behavior but unlikely to dominate runtime.

### Target 4: final assignment and path sampling

Already accelerated. Keep the existing accelerator boundary and verify that the copied package imports its own `.so` files after package renaming.

## Ref-To-Ref Work Reuse

For each alpha, ref-to-ref currently gets its own score-floor mask and Hough/filtering result. That must remain alpha-specific because:

- score-floor threshold changes with alpha;
- active ref-to-ref cells change;
- Hough lines change;
- coverage subtraction metrics depend on ref-to-ref self-coverage.

Do not reuse ref-to-ref Hough across alpha values unless alpha and mask are identical.

Safe reuse:

- loaded ref-to-ref matrix;
- ref-to-ref matrix mean/std;
- reference windows;
- document metadata.

Unsafe reuse:

- ref-to-ref mask;
- ref-to-ref raw Hough lines;
- ref-to-ref final lines.

## Detailed Implementation Steps

1. **Package isolation**
   - Rename imports in the copied tree from `tuner_simple` to `tuner_simple_alpha_sweep`.
   - Update `run_tunner.sh` fallback path.
   - Update Cython extension names and optional imports.
   - Build accelerators.
   - Run `python3 -m py_compile` on the copied tree.

2. **Single-alpha parity test**
   - Run original `tuner_simple` and copied `tuner_simple_alpha_sweep` with `--no-alpha-sweep --score-floor-alpha X`.
   - Compare `compact_combination_metrics.csv`.
   - Differences must be zero except paths/timings if those are included.

3. **Add alpha config and CLI**
   - Add sweep fields.
   - Add parser flags.
   - Add validation for positive step and min <= max.
   - Generate alpha values by integer ticks.

4. **Extract one-alpha evaluation function**
   - Move current alpha-dependent body from `process_one_document` into `evaluate_document_for_alpha`.
   - Make the current single-alpha path call this helper once.
   - Re-run parity test.

5. **Add harmonic selection**
   - Implement score function.
   - Add deterministic tie-break.
   - Add best-result selection loop.
   - Ensure returned `DocumentRunResult` contains only best row and best plot payload.

6. **Add sweep metadata**
   - Add selected alpha to existing `score_floor_alpha` column through `build_result_row`.
   - Add `alpha_sweep_enabled`, min/max/step, and metric name to `run_summary.json`.
   - Optionally add tiny `alpha_sweep_decisions.csv`.

7. **Change plotting to final-only stitched rendering**
   - Replace `SimplePlotManager.render_document_payload` immediate file rendering with payload collection.
   - At `finish()`, render only selected payloads to in-memory panels and stitch.
   - Save only `plots/stitched_best_combination_<language>_documents.png`.
   - Preserve final layout exactly: panel order, columns, border, gap, file naming.

8. **Add score-floor sweep acceleration**
   - Implement pure Python `compute_score_floor_masks_for_alphas`.
   - Add Cython implementation.
   - Keep Python fallback.
   - Add tests comparing per-alpha masks against repeated existing `compute_score_floor_mask`.

9. **Validation runs**
   - Single doc: `europeana_00675344.jpeg`.
   - Full Finnish: confirm selected alpha for Europeana matches the harmonic score logic.
   - Arabic preservation: run with alpha grid and compare chosen results where expected, but original Arabic result set remains untouched because this is a copied variant.

## Expected Europeana Behavior

For `europeana_00675344.jpeg`, based on the already inspected result rows:

- alpha `1.5` harmonic score: about `0.7720007895`
- alpha `3.5` harmonic score: about `0.8096739563`

So the sweep should pick alpha `3.5` over alpha `1.5` for that document, assuming the full run uses the same Hough/filtering settings:

```text
hough_threshold=3
hough_line_length=3
hough_line_gap=4
hough_seed=0
align_min_iou_threshold=0.10
min_surviving_line_nls=0.45
```

## Main Risks

1. **Importing the original package by accident**
   - This is the highest-risk issue because the copied tree still imports `tuner_simple`.
   - Fix package isolation first.

2. **Changing behavior while refactoring**
   - Extract one-alpha evaluation first and prove parity before adding sweep selection.

3. **Memory during final-only plotting**
   - Keeping plot payloads for every selected document stores matrices until the end.
   - Fine for 30-document language runs, but add a memory note if scaling up.

4. **Hidden output schema drift**
   - Keep the main CSV schema compatible unless a separate audit CSV is explicitly desired.

5. **Invalid metrics**
   - If `weighted_along_lines_normalised_levenshtein`, `correct_ref_coverage`, or `1 - hallucination` is `None` or <= 0, the harmonic score should be `-inf`.
   - If every alpha is invalid, fall back deterministically to the lowest alpha result so the document still has a clear output row if the old pipeline would have produced one.

## Done Criteria

The implementation is complete when:

1. The original `tuner_simple` directory is unchanged.
2. The copied `tuner_simple_alpha_sweep` can run independently.
3. `--no-alpha-sweep --score-floor-alpha X` matches old single-alpha outputs.
4. Sweep mode evaluates alpha `1.0..4.0` by `0.2` for each document.
5. Only the selected best result is written to the main CSV outputs.
6. Only the selected best result is plotted.
7. The final stitched language image layout matches the current one.
8. Cython acceleration is used where it can preserve exact behavior, with Python fallback.
