# tuner_simple v2.12 Coverage Metric Implementation Plan

## Goal

Implement the coverage and hallucination metrics in `tuner_simple/` using the
same geometric character-coverage logic used by
`text_metrics_v2_12_parallel/`, while keeping the current `tuner_simple`
handling for impossible reference-axis subtraction values.

The target behavior is:

```text
1. Keep the simple fixed-parameter tuner.
2. Keep the current Hough input and Hough filtering behavior.
3. Keep the current line-level Levenshtein filter.
4. Compute coverage metrics from final line geometry, not from column ownership.
5. Compare ref-to-pred reference coverage against ref-to-ref reference coverage.
6. Compute hallucination from prediction characters not crossed by final ref-to-pred lines.
7. Preserve the current tuner_simple invalid handling when y_diff < -1.
8. Avoid importing any code from text_metrics_v2_12_parallel.
9. Keep computation small, local, and easy to explain.
```

The final metric names stay unchanged:

```text
document_normalised_levenshtein
weighted_along_lines_normalised_levenshtein
correct_ref_coverage
missing_ref_coverage
repetition_on_reference
hallucination
```

No `alignment_selection_score`, `score_matrix_support`, `line_guided_fraction`,
`selection_objective`, or `tuning_score` should be calculated or written.


## Source Of Truth From text_metrics_v2_12_parallel

The original metric pipeline separates two ideas that must not be mixed:

```text
line ownership       -> used for line-guided Levenshtein text extraction
line geometry        -> used for coverage, repetition, missing reference, hallucination
```

The important original files are:

```text
text_metrics_v2_12_parallel/pipeline/process_single_document_metrics.py
text_metrics_v2_12_parallel/line_metric_bundle.py
text_metrics_v2_12_parallel/shared/project_line_to_text_windows.py
text_metrics_v2_12_parallel/line_coverage_subtract.py
```

The original data flow is:

```text
final ref-to-pred lines
  -> sample each line across matrix coordinates
  -> convert crossed x/y window ids into character intervals
  -> accumulate per-character counts
  -> produce other_y and other_x

final ref-to-ref lines
  -> sample each line across matrix coordinates
  -> convert crossed y window ids into reference-character intervals
  -> accumulate per-character counts
  -> produce refref_y

y_diff = other_y - refref_y
```

The original metric categories are:

```text
missing_ref_coverage      = count(y_diff == -1) / reference_character_count
correct_ref_coverage      = count(y_diff == 0)  / reference_character_count
repetition_on_reference   = count(y_diff > 0)   / reference_character_count
hallucination             = count(other_x == 0) / prediction_character_count
```

The critical point is that these ratios are character-level ratios. They are not
window-level ratios and they are not simple counts of assigned matrix columns.


## Current tuner_simple Situation

The current `tuner_simple/scoring/coverage_count_metrics.py` is already close to
the required idea because it:

```text
1. samples final line endpoints;
2. converts crossed windows into merged character intervals;
3. accumulates character counts with a difference-array method;
4. subtracts ref-to-ref reference counts from ref-to-pred reference counts;
5. treats y_diff < -1 as invalid instead of raising.
```

However, the implementation still needs to be made cleaner and more explicit
because the surrounding payload names and comments can mislead a reader into
thinking coverage is based on ownership arrays.

The risky points are:

```text
1. DirectionScoringPayload says "local ownership payload", but coverage must not
   be understood as ownership-based.

2. build_direction_scoring_payload() stores column_assignment because line-text
   scoring needs it, but coverage_count_metrics.py should ignore it.

3. refref_y is currently stored on DirectionScoringPayload but is not used by
   compute_coverage_count_metrics().

4. Existing comments say "reference windows" and "prediction windows" in places
   where the metric is actually character-level.

5. The current tests are too small to protect the distinction between line
   geometry and column ownership.
```

The implementation work should therefore be a careful cleanup and hardening, not
a large redesign.


## Required Mathematical Semantics

For one document, define:

```text
reference_text_length = number of characters in the reference text
prediction_text_length = number of characters in the prediction text
window_size = sliding score-window size
window_stride = sliding score-window stride
```

The number of windows on one text axis is:

```text
window_count = 0, if text_length < window_size
window_count = floor((text_length - window_size) / window_stride) + 1, otherwise
```

For each final line:

```text
1. Read its matrix endpoints:
   (x0, y0), (x1, y1)

2. Compute the number of sampling steps:
   step_count = max(abs(x1 - x0), abs(y1 - y0)) + 1

3. Sample evenly spaced x positions and y positions along the line.

4. Round sampled x and y positions to nearest integer matrix-window ids.

5. Clip x ids to [0, prediction_window_count - 1].

6. Clip y ids to [0, reference_window_count - 1].

7. Remove duplicate ids.

8. Convert each window id into a character interval:
   start = window_id * window_stride
   end = min(start + window_size, text_length)

9. Merge overlapping intervals for the same line.
```

One line contributes at most one count to a character, even if overlapping
windows from that same line cover the same character. Multiple different lines
can each contribute one count to the same character.

The per-character count arrays are:

```text
other_y:
  length = reference_text_length
  value at character i = how many final ref-to-pred lines geometrically cover
  reference character i

other_x:
  length = prediction_text_length
  value at character i = how many final ref-to-pred lines geometrically cover
  prediction character i

refref_y:
  length = reference_text_length
  value at character i = how many final ref-to-ref lines geometrically cover
  reference character i
```

The reference-axis subtraction is:

```text
y_diff = other_y - refref_y
```

Interpretation:

```text
y_diff == -1:
  ref-to-ref covered this reference character once, but ref-to-pred did not.
  This contributes to missing_ref_coverage.

y_diff == 0:
  ref-to-pred coverage matches the ref-to-ref baseline for this reference
  character. This contributes to correct_ref_coverage.

y_diff > 0:
  ref-to-pred covered this reference character more often than ref-to-ref did.
  This contributes to repetition_on_reference.

y_diff < -1:
  ref-to-ref covered this reference character two or more times more than
  ref-to-pred did. tuner_simple must preserve the current invalid handling for
  this case.
```

The `y_diff < -1` rule is important. The original v2.12 helper raises a
`ValueError` when values outside `-1`, `0`, and `>0` appear. In `tuner_simple`,
the current behavior is better for long automated runs because it records a
clean invalid result instead of crashing the whole worker.

Preserve this current behavior:

```text
if any y_diff < -1:
    correct_ref_coverage = None
    missing_ref_coverage = None
    repetition_on_reference = None
    hallucination = None
    invalid_reason = "coverage_y_diff_below_minus_one"
    diagnostics record exactly how many values were below -1
```


## Low-Computation Design

Do not build full v2.12 line bundles inside `tuner_simple`.

The full bundle contains fields needed by both text scoring and coverage. The
simple tuner only needs a small subset for coverage:

```text
line endpoints
reference text length
prediction text length
window size
window stride
```

The coverage code should therefore compute only:

```text
ref-to-pred y character intervals
ref-to-pred x character intervals
ref-to-ref y character intervals
other_y
other_x
refref_y
y_diff
diagnostics
four public coverage metrics
```

This avoids:

```text
1. building large score-matrix-sized helper arrays;
2. storing per-document JSON bundles;
3. recomputing text-window ownership;
4. keeping matrices alive after Hough and line filtering are finished;
5. importing or depending on text_metrics_v2_12_parallel.
```

The main computational cost is linear in text length plus the number of line
intervals:

```text
O(reference_text_length + prediction_text_length + number_of_final_lines)
```

The difference-array accumulation should stay because it is simple and fast:

```text
1. create an integer array of length text_length + 1;
2. for each interval [start, end):
       diff[start] += 1
       diff[end] -= 1
3. cumulative sum over diff[:-1] gives per-character counts.
```

This is much cheaper than touching every character for every line.


## Proposed File-Level Changes

### 1. scoring/coverage_count_metrics.py

Keep this as the only coverage metric implementation file.

Refactor it so its public structure is visibly aligned with v2.12:

```text
CoverageCountMetricResult
CoverageAxisCounts
count_text_windows()
line_window_ids_from_endpoint()
window_ids_to_merged_character_intervals()
accumulate_character_counts_from_interval_groups()
build_coverage_intervals_from_lines()
build_ref_to_pred_axis_counts()
build_ref_to_ref_reference_axis_counts()
compute_reference_axis_subtraction_diagnostics()
compute_coverage_count_metrics()
```

The function names should make the data flow obvious:

```text
line geometry -> window ids -> character intervals -> character counts -> y_diff -> metrics
```

`compute_coverage_count_metrics()` should remain the public entry point used by
`scoring_pipeline.py`.

It must ignore `column_assignment` for coverage. Column assignment belongs to
line-text Levenshtein scoring, not coverage counting.


### 2. scoring/scoring_pipeline.py

Clean up the naming around `DirectionScoringPayload`.

Recommended change:

```text
DirectionScoringPayload
  hough_payload
  metric_payload
```

or keep `scoring_payload` but update comments so they clearly say:

```text
The payload contains final lines, assignment arrays, text lengths, and window
settings. Coverage metrics use only final line geometry and text/window sizes.
Line-text metrics use the assignment arrays.
```

Remove `refref_y` from `DirectionScoringPayload` if no other code uses it. It is
currently misleading because the real coverage code computes `refref_y` inside
`compute_coverage_count_metrics()` from final ref-to-ref line geometry.

If removing it creates too much churn, leave it temporarily but mark it for
deletion in the same phase. The better final code should not carry unused
coverage arrays in the main scoring payload.


### 3. scoring/line_text_similarity.py

Do not change the metric formulas here unless tests reveal a direct dependency
on renamed payload fields.

This file should remain responsible for:

```text
weighted_along_lines_normalised_levenshtein
min_surviving_line_nls filtering
line text extraction from assignment arrays
```

This is separate from geometric coverage.


### 4. serial_runner/document_runner.py

Do not change the output metric names.

Only update wording in logs if necessary so logs do not call character-level
coverage "window coverage".

The result row should still include:

```text
coverage_invalid_reason
coverage_invalid_error_message
coverage_y_diff_size
coverage_y_diff_min
coverage_y_diff_max
coverage_y_diff_le_minus_one_count
coverage_y_diff_lt_minus_one_count
coverage_y_diff_below_minus_one_counts_json
```

These diagnostics are necessary to preserve the current `-2 or lower` handling.


### 5. results_writing/

Do not add new scientific columns.

The current final CSV should continue to expose only the approved public metrics
plus diagnostics needed for invalid coverage cases.

If any current output column says or implies window-level coverage, rename only
if the rename does not break existing downstream plots. Otherwise leave the
column name but document internally that values are character-level ratios.


### 6. plotting/

No mandatory plotting changes are needed.

If a plot displays metric descriptions, make sure descriptions say:

```text
correct_ref_coverage is a reference-character ratio
missing_ref_coverage is a reference-character ratio
repetition_on_reference is a reference-character ratio
hallucination is a prediction-character ratio
```

Do not add coverage-array plots in this phase. The goal is metric correctness
with minimal computation and minimal output growth.


## Test Plan

### Unit Tests For Character Interval Construction

Add tests that prove window ids become character intervals correctly.

Example:

```text
window_size = 10
window_stride = 5
text_length = 25
window ids = [0, 1, 2]

raw intervals:
  [0, 10)
  [5, 15)
  [10, 20)

merged interval:
  [0, 20)
```

This protects the overlapping-window behavior.


### Unit Tests For One-Line Correct Coverage

Create matching ref-to-ref and ref-to-pred diagonal lines.

Expected:

```text
correct_ref_coverage = 1.0
missing_ref_coverage = 0.0
repetition_on_reference = 0.0
hallucination = 0.0
invalid_reason = None
```


### Unit Tests For Missing Reference Coverage

Use a ref-to-ref line and no ref-to-pred line.

Expected:

```text
y_diff = -1 for covered reference characters
missing_ref_coverage = 1.0 when the baseline covers the full reference
hallucination = 1.0 when no prediction character is covered
```


### Unit Tests For Repetition On Reference

Use one ref-to-ref line and two ref-to-pred lines covering the same reference
characters.

Expected:

```text
y_diff > 0 for repeated reference characters
repetition_on_reference = fraction of repeated reference characters
```

The test must confirm that repetition is counted per character, not by the
magnitude of the positive difference.


### Unit Tests For y_diff Below Minus One

Use two ref-to-ref lines and no ref-to-pred line.

Expected:

```text
y_diff = -2 on covered reference characters
correct_ref_coverage = None
missing_ref_coverage = None
repetition_on_reference = None
hallucination = None
invalid_reason = "coverage_y_diff_below_minus_one"
coverage_y_diff_lt_minus_one_count > 0
coverage_y_diff_below_minus_one_counts_json includes "-2"
```

This test preserves the current tuner_simple behavior.


### Unit Tests Separating Ownership From Coverage

Add a test where `column_assignment` is deliberately empty or misleading, but
`lines_used` contains a valid line.

Coverage metrics should still be computed from the line geometry.

This test directly protects the conceptual distinction:

```text
column assignment is not coverage
line geometry is coverage
```


### Integration Smoke Test

Run a tiny `tuner_simple` command with:

```text
one or two documents
fixed Hough parameters
plotting disabled
one direct serial run
one atomic worker run
```

Check:

```text
1. both modes produce the same metric values;
2. invalid y_diff cases are written as rows, not worker crashes;
3. document-level Levenshtein remains unchanged;
4. weighted along-lines Levenshtein remains unchanged;
5. final CSV column names remain stable.
```


## Implementation Phases

### Phase 1: Snapshot And Baseline

Create a snapshot of the current `tuner_simple/` directory before touching code.

Run:

```text
python3 -m pytest /scratch/project_2017385/dorian/Churro_copy/tuner_simple/tests
python3 -m py_compile on all tuner_simple Python files
```

Record the current test state before refactoring.


### Phase 2: Refactor coverage_count_metrics.py Names Only

Rename internal helpers so the code reads like the v2.12 flow.

No formula changes in this phase.

Expected result:

```text
same tests pass
same output values
clearer function names
no imports from text_metrics_v2_12_parallel
```


### Phase 3: Remove Or Isolate Misleading Payload State

Update `scoring_pipeline.py` so `DirectionScoringPayload` no longer implies that
coverage is assignment-based.

Preferred change:

```text
remove unused refref_y
update comments around scoring_payload / metric_payload
```

Do not change line-text similarity behavior.


### Phase 4: Add Coverage-Specific Tests

Add the tests listed above before making any formula-sensitive edits.

These tests should use tiny synthetic line dictionaries so they are fast and
explainable.

The tests should not need score matrices, Hough, Slurm, plotting, or pickle
files.


### Phase 5: Verify v2.12 Metric Equivalence

Confirm that the local implementation matches the v2.12 formula:

```text
ref-to-pred line geometry -> other_y and other_x
ref-to-ref line geometry  -> refref_y
y_diff = other_y - refref_y
metrics from y_diff and other_x
```

This phase should compare against hand-calculated expected arrays, not by
importing v2.12 modules.

The production package must remain self-contained.


### Phase 6: Preserve Invalid y_diff Handling

Keep the current `tuner_simple` rule:

```text
y_diff < -1 means invalid coverage result, not a crashed process
```

Ensure all result rows still include:

```text
invalid_reason
invalid_error_message
diagnostic counts
```

This is where `tuner_simple` intentionally differs from the original v2.12
helper, which raises an exception for undefined categories.


### Phase 7: Direct And Atomic Smoke Tests

Run a tiny direct serial smoke test and a tiny atomic-worker smoke test.

Check:

```text
1. no Region of Interest calculation is reintroduced;
2. no Hough grid is introduced;
3. score-floor Hough input still works as before;
4. coverage metrics are character-level;
5. direct and atomic output rows agree for the same documents;
6. invalid coverage rows are visible in CSV output.
```


### Phase 8: Documentation Cleanup

Update `tuner_simple/README.md` only after tests pass.

The documentation should say:

```text
Coverage metrics are computed from final line geometry.
Reference coverage compares ref-to-pred against a ref-to-ref baseline.
Hallucination is the fraction of prediction characters not crossed by final
ref-to-pred line geometry.
y_diff < -1 is preserved as an invalid coverage diagnostic in tuner_simple.
```

Avoid saying the metrics are window ratios.


## Critical Review Of The Design

This design is intentionally conservative.

What it does well:

```text
1. It restores the scientific meaning from text_metrics_v2_12_parallel.
2. It keeps tuner_simple self-contained.
3. It avoids rebuilding full v2.12 bundles.
4. It keeps the current invalid -2 handling.
5. It keeps computation low.
6. It keeps output metric names stable.
```

Potential weak points:

```text
1. If ref-to-ref lines repeat strongly, y_diff < -1 can still invalidate the
   coverage metrics. This is intentional for now, but it means some documents
   will have None for coverage fields.

2. Coverage is line-geometry based, so a geometrically long but text-poor line
   can cover many characters. The line-level Levenshtein filter helps, but the
   coverage metric itself does not inspect text quality.

3. Ref-to-pred uses final lines after the tuner_simple line-level Levenshtein
   filter. Original v2.12 did not have exactly the same simple-tuner text filter,
   so the formula can be v2.12-compatible while the selected final line set is
   tuner_simple-specific.

4. Character-level arrays can still be long for very large documents. The
   difference-array method is the lightest simple implementation, but the memory
   cost is still proportional to text length.
```

These are acceptable tradeoffs for this pipeline because the goal is a simple,
fixed-parameter scientific tool rather than a full reproduction of the old
parallel report system.


## Definition Of Done

The metric correction is complete when:

```text
1. coverage_count_metrics.py computes coverage from final line geometry;
2. column_assignment is not used for coverage counts;
3. ref-to-pred creates other_y and other_x;
4. ref-to-ref creates refref_y;
5. y_diff = other_y - refref_y;
6. y_diff == -1 gives missing_ref_coverage;
7. y_diff == 0 gives correct_ref_coverage;
8. y_diff > 0 gives repetition_on_reference;
9. other_x == 0 gives hallucination;
10. y_diff < -1 returns invalid metrics with diagnostics, not a crash;
11. tests cover correct, missing, repetition, hallucination, and invalid cases;
12. direct serial and atomic worker modes still run;
13. public CSV metric names remain unchanged;
14. no code imports from text_metrics_v2_12_parallel;
15. no Region of Interest logic is reintroduced;
16. no Hough grid or multiprocessing is introduced.
```

## 2026-06-09 Implementation Update: v2.12 Coverage And Cython Ownership

The current `tuner_simple` implementation now keeps the v2.12-style coverage-count behavior inside the local pipeline instead of importing it from another tuner. The coverage metrics are calculated from final ref-to-pred and ref-to-ref line geometry, converted back to character spans through `window_size` and `window_stride`. A `y_diff < -1` situation is still treated as invalid coverage evidence, but the document result and final recognized lines are preserved so plotting and audit output can still show what the Hough stage found.

The Hough column-ownership hot loop now has a local optional Cython accelerator in `cython_accel/`. The accelerated function assigns each matrix column to the strongest final candidate line that crosses an active Hough voter cell. If the compiled extension is available, the Hough filtering code uses it. If it is not available, the code falls back to the readable Python implementation with the same output contract.

Both `run_tunner.sh` and `run_tunner_atomic.sh` attempt to build the Cython accelerator before starting the pipeline. This keeps the normal user command simple while still allowing safe fallback to Python when Cython is unavailable. Set `TUNER_SIMPLE_SKIP_CYTHON_BUILD=1` only when the caller deliberately wants to skip the build step.

Validation after implementation:

- `bash -n run_tunner.sh` passed.
- `bash -n run_tunner_atomic.sh` passed.
- `bash -n run_tunner_atomic_worker.sbatch` passed.
- `python3 -m compileall -q tuner_simple` passed.
- `python3 -m pytest --confcutdir=tuner_simple tuner_simple/tests -q` passed with 11 tests.
- A one-document Finnish smoke run completed successfully with the requested Hough and score-floor settings.
- A synthetic ownership-loop comparison showed identical Python and Cython assignments, identical owned-column counts, and about 42x faster runtime for the Cython ownership loop on the synthetic matrix.
