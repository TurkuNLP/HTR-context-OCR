# Region of Interest, Median Absolute Deviation, and `alignment_evidence` Selection

This document explains what happens in `tuner_parallel_v2_2` when the tuner is run with:

```bash
--selection-objective alignment_evidence
```

The goal is to explain the implementation as if the reader has no previous experience with this codebase.

Important terms used throughout this document:

- Region of Interest means the part of the score matrix that the preprocessing believes is worth searching for Hough lines.
- Median Absolute Deviation means a robust measure of spread around the median. It is used to build an adaptive score floor when the user enables it.
- Hough input means the final binary image passed to the Hough line detector. A cell value of 1 means "this matrix cell may vote for a line"; a cell value of 0 means "ignore this cell".
- Normalized Levenshtein Similarity means a text similarity score in the range 0..1, where 1 means identical text and 0 means no similarity.
- Intersection over Union means the overlap ratio used by the line filter when deciding which line owns which score-matrix cells.

## High-Level Flow

The pipeline has two different stages that are easy to confuse:

1. Score-matrix preprocessing happens once per document and per matrix direction.
2. Hough parameter sweeping happens many times per document.

The important design decision is that Region of Interest and Median Absolute Deviation preprocessing are not inside the hot Hough loop. They are computed once for the reference-to-prediction matrix and once for the reference-to-reference matrix. The hot loop then reuses the same binary Hough input while changing only Hough parameters such as threshold, line length, line gap, and seed.

The two matrix directions are:

- `ref_to_pred`: reference windows on the y-axis, prediction windows on the x-axis.
- `ref_to_ref`: reference windows on both axes. This is used as the reference self-alignment baseline for coverage.

In the final all-document run, the target document:

```text
newseye-fin_576458_0002_23676306.jpeg
```

had these preprocessing facts in `document_metadata.json`:

```text
ref_to_pred matrix shape: 396 rows x 385 columns
score median: 13.916204602894805
scaled Median Absolute Deviation: 2.711698466395645
adaptive score floor: 13.916204602894805
final score floor: 20.0
strong match cells: 645
Region of Interest cells: 2415
final Hough input active cells: 529
final Hough input active fraction: 0.0034697625606716514
```

Because the run used `median_absolute_deviation_multiplier = 0.0`, the adaptive Median Absolute Deviation term was calculated but did not raise the score floor. The final score floor became `max(20.0, matrix_median)`, which was `20.0`.

## Where The Settings Are Defined

Implementation file:

```text
tuner_parallel_v2_2/hough_preprocessing/config.py
```

The user-facing preprocessing settings are stored in `HoughPreprocessingConfig`:

```python
@dataclass(frozen=True)
class HoughPreprocessingConfig:
    """User-facing controls for Region of Interest Hough preprocessing."""

    minimum_score_floor: float = 20.0
    median_absolute_deviation_multiplier: float = 0.0
    median_absolute_deviation_backend: str = MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY
    near_peak_ratio: float = 0.90
    near_peak_margin: float | None = None
    minimum_component_cells: int = 2
    minimum_component_rows: int = 1
    minimum_component_columns: int = 1
    connected_component_backend: str = CONNECTED_COMPONENT_BACKEND_CYTHON
    region_dilation_radius: int = 1
    minimum_active_cells: int = 5
    minimum_active_rows: int = 2
    minimum_active_columns: int = 2
    minimum_x_span: int = 2
    minimum_y_span: int = 2
    maximum_active_fraction: float = 0.08
```

What these settings mean:

- `minimum_score_floor`: the lowest score that can be considered strong enough to vote.
- `median_absolute_deviation_multiplier`: how much the adaptive floor should rise above the matrix median.
- `median_absolute_deviation_backend`: which implementation computes Median Absolute Deviation. The default is the small NumPy implementation. SciPy is optional.
- `near_peak_ratio`: keeps cells that are close to the best score in their row or column.
- `near_peak_margin`: optional absolute score margin around the row or column peak.
- `minimum_component_cells`: smallest connected component that can become part of the Region of Interest.
- `minimum_component_rows`: smallest number of reference rows a connected component must cover.
- `minimum_component_columns`: smallest number of prediction columns a connected component must cover.
- `connected_component_backend`: connected-component labeler. Cython is default, SciPy is optional, Python is fallback.
- `region_dilation_radius`: how much the kept Region of Interest is expanded around kept components.
- `minimum_active_cells`: final Hough input must have at least this many active cells.
- `minimum_active_rows`: final Hough input must cover at least this many reference rows.
- `minimum_active_columns`: final Hough input must cover at least this many prediction columns.
- `minimum_x_span`: final Hough input must span at least this many prediction columns from left to right.
- `minimum_y_span`: final Hough input must span at least this many reference rows from top to bottom.
- `maximum_active_fraction`: final Hough input must not keep too large a fraction of the whole matrix.

The validation in `__post_init__` rejects settings that would make no sense, such as negative floors or a `near_peak_ratio` outside the interval 0..1.

The command-line arguments are defined in:

```text
tuner_parallel_v2_2/run_hough_parameter_sweep.py
```

Relevant code:

```python
parser.add_argument(
    "--minimum-score-floor",
    "--min-score",
    type=float,
    default=20.0,
    help="Minimum chrF score a matrix cell must reach before it can vote in Hough preprocessing",
)
parser.add_argument(
    "--median-absolute-deviation-multiplier",
    "--mad-k",
    type=float,
    default=0.0,
    help="Multiplier applied to the scaled Median Absolute Deviation when building the adaptive score floor",
)
parser.add_argument(
    "--median-absolute-deviation-backend",
    choices=(MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY, MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY),
    default=MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
    help="Implementation used for the scaled Median Absolute Deviation calculation",
)
parser.add_argument("--near-peak-ratio", type=float, default=0.90, help="Keep cells near the best score in their row or column")
parser.add_argument("--near-peak-margin", type=float, default=None, help="Optional score-distance margin for near-peak cells")
parser.add_argument("--maximum-active-fraction", "--max-active-fraction", type=float, default=0.08, help="Maximum fraction of matrix cells allowed to remain active after preprocessing")
```

Later in the same file, the parsed values are converted into a `HoughPreprocessingConfig`:

```python
hough_preprocessing_config = HoughPreprocessingConfig(
    minimum_score_floor=float(args.minimum_score_floor),
    median_absolute_deviation_multiplier=float(args.median_absolute_deviation_multiplier),
    median_absolute_deviation_backend=str(args.median_absolute_deviation_backend),
    near_peak_ratio=float(args.near_peak_ratio),
    near_peak_margin=None if args.near_peak_margin is None else float(args.near_peak_margin),
    minimum_component_cells=int(args.minimum_component_cells),
    minimum_component_rows=int(args.minimum_component_rows),
    minimum_component_columns=int(args.minimum_component_columns),
    connected_component_backend=str(args.connected_component_backend),
    region_dilation_radius=int(args.region_dilation_radius),
    minimum_active_cells=int(args.minimum_active_cells),
    minimum_active_rows=int(args.minimum_active_rows),
    minimum_active_columns=int(args.minimum_active_columns),
    minimum_x_span=int(args.minimum_x_span),
    minimum_y_span=int(args.minimum_y_span),
    maximum_active_fraction=float(args.maximum_active_fraction),
)
```

## Where Preprocessing Is Called

Implementation file:

```text
tuner_parallel_v2_2/matrices/document_prep.py
```

This file prepares each document before the Hough parameter sweep. It loads or computes the score matrices, builds text blocks, and builds Hough preprocessing contexts.

The reference-to-prediction context is built here:

```python
ref_to_pred_hough_ctx = build_region_of_interest_hough_context(
    ref_to_pred_matrix,
    config=active_hough_preprocessing_config,
    keep_debug_arrays=False,
)
if not bool(ref_to_pred_hough_ctx.get("hough_preprocessing_accepted", False)):
    skip_record = _hough_preprocessing_skip_record(
        item=item,
        fname=fname,
        pred=pred,
        ref=ref,
        matrix=ref_to_pred_matrix,
        matrix_source=ref_to_pred_source,
        matrix_direction="ref_to_pred",
        hough_context=ref_to_pred_hough_ctx,
    )
    ...
    continue
```

The reference-to-reference context is built here:

```python
ref_to_ref_hough_ctx = build_region_of_interest_hough_context(
    ref_to_ref_matrix,
    config=active_hough_preprocessing_config,
    keep_debug_arrays=False,
)
if not bool(ref_to_ref_hough_ctx.get("hough_preprocessing_accepted", False)):
    skip_record = _hough_preprocessing_skip_record(
        item=item,
        fname=fname,
        pred=pred,
        ref=ref,
        matrix=ref_to_ref_matrix,
        matrix_source=ref_to_ref_source,
        matrix_direction="ref_to_ref",
        hough_context=ref_to_ref_hough_ctx,
    )
    ...
    continue
```

This is where documents can be kicked out before the Hough hot loop. That is intentional. If preprocessing finds no usable Hough evidence, the tuner does not waste time evaluating thousands of Hough parameter combinations for that matrix.

The prepared document stores both matrices and both Hough contexts:

```python
prepared_document = SweepDocument(
    index=int(item["index"]),
    fname=Path(fname).name,
    window_size=int(window_size),
    window_stride=int(window_stride),
    pred=pred,
    ref=ref,
    ref_to_pred_matrix=ref_to_pred_matrix,
    ref_to_ref_matrix=ref_to_ref_matrix,
    whole_document_nls=whole_nls,
    pred_blocks=pred_blocks,
    ref_blocks=ref_blocks,
    ref_to_pred_hough_ctx=ref_to_pred_hough_ctx,
    ref_to_ref_hough_ctx=ref_to_ref_hough_ctx,
)
```

The important point is that `ref_to_pred_hough_ctx` and `ref_to_ref_hough_ctx` are computed once, then reused by every Hough parameter combination.

## Matrix Statistics And Median Absolute Deviation

Implementation file:

```text
tuner_parallel_v2_2/hough_preprocessing/matrix_statistics.py
```

The first step is to flatten the matrix into all finite score values:

```python
def finite_score_values(score_matrix: np.ndarray) -> np.ndarray:
    """Return the finite cells from a score matrix as one float array."""
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.size == 0:
        return np.asarray([], dtype=float)
    return matrix[np.isfinite(matrix)]
```

Why this matters:

- The score matrix should be numeric.
- Any non-finite value such as `NaN`, positive infinity, or negative infinity must not affect the threshold.
- The threshold is based only on real numeric score cells.

The matrix summary is built here:

```python
median_value = float(np.median(finite_values))
return ScoreMatrixStatistics(
    finite_value_count=int(finite_values.size),
    score_minimum=float(np.min(finite_values)),
    score_mean=float(np.mean(finite_values)),
    score_median=median_value,
    score_maximum=float(np.max(finite_values)),
    scaled_median_absolute_deviation=scaled_median_absolute_deviation(
        finite_values,
        median_value=median_value,
        backend=str(median_absolute_deviation_backend),
    ),
    median_absolute_deviation_backend=str(median_absolute_deviation_backend),
)
```

The median is the middle value of the finite score distribution. If the finite matrix cells were sorted from smallest to largest, the median is the value in the middle. If there is an even number of values, NumPy averages the two middle values.

Median Absolute Deviation is calculated in two steps:

1. For every finite score, compute how far it is from the median.
2. Take the median of those distances.

The manual implementation is:

```python
absolute_deviation_from_median = np.abs(finite_values - float(median_value))
raw_median_absolute_deviation = float(np.median(absolute_deviation_from_median))
return float(raw_median_absolute_deviation * NORMAL_DISTRIBUTION_MEDIAN_ABSOLUTE_DEVIATION_SCALE)
```

The constant is:

```python
NORMAL_DISTRIBUTION_MEDIAN_ABSOLUTE_DEVIATION_SCALE = 1.482602218505602
```

This value is the normal-distribution scaling factor. It equals:

```text
1 / inverse_standard_normal_cumulative_distribution_function(0.75)
```

Numerically:

```text
inverse_standard_normal_cumulative_distribution_function(0.75)
= 0.6744897501960817

1 / 0.6744897501960817
= 1.482602218505602
```

Why scale Median Absolute Deviation at all?

Raw Median Absolute Deviation is naturally smaller than standard deviation for normally distributed data. Multiplying by `1.482602218505602` puts it on the same scale as standard deviation if the values behave like a normal distribution. It is still more robust than standard deviation because it is based on medians instead of means.

Why is this acceptable for other datasets?

The constant is not learned from this dataset. It is a statistical convention used when a user wants Median Absolute Deviation to be comparable to standard deviation under a normal-distribution assumption. If a future dataset behaves differently, the user should not change this constant. The user should change `median_absolute_deviation_multiplier`, `minimum_score_floor`, or choose a different floor strategy. The constant only defines what "scaled Median Absolute Deviation" means.

In the current all-document run, the multiplier was `0.0`, so the scaled Median Absolute Deviation was reported but did not influence the final score floor.

## Building The Score Floor

Implementation file:

```text
tuner_parallel_v2_2/hough_preprocessing/region_of_interest.py
```

The adaptive score floor is computed like this:

```python
adaptive_score_floor = float(
    statistics.score_median
    + float(preprocessing_config.median_absolute_deviation_multiplier)
    * float(statistics.scaled_median_absolute_deviation)
)
final_score_floor = float(max(float(preprocessing_config.minimum_score_floor), adaptive_score_floor))
```

In words:

- Start at the matrix median.
- Optionally add `median_absolute_deviation_multiplier * scaled Median Absolute Deviation`.
- Compare that adaptive value with the user-defined minimum score floor.
- Use the larger one.

For the Finnish document in the current run:

```text
matrix median = 13.916204602894805
scaled Median Absolute Deviation = 2.711698466395645
median_absolute_deviation_multiplier = 0.0
minimum_score_floor = 20.0

adaptive score floor = 13.916204602894805 + 0.0 * 2.711698466395645
adaptive score floor = 13.916204602894805

final score floor = max(20.0, 13.916204602894805)
final score floor = 20.0
```

The score floor does not create the final Hough input by itself. It only creates the first mask:

```python
score_floor_mask = finite_score_matrix >= final_score_floor
```

This mask says:

```text
Keep cells whose score is at least the final score floor.
```

## Keeping Cells That Are Near Row Or Column Peaks

The next step asks a different question:

```text
Is this cell close to the best score in its own row or in its own column?
```

The code:

```python
row_peak_scores = np.max(finite_score_matrix, axis=1)
column_peak_scores = np.max(finite_score_matrix, axis=0)
row_near_peak_mask = finite_score_matrix >= (row_peak_scores[:, None] * float(preprocessing_config.near_peak_ratio))
column_near_peak_mask = finite_score_matrix >= (column_peak_scores[None, :] * float(preprocessing_config.near_peak_ratio))
```

With `near_peak_ratio = 0.90`, a cell is near the row peak if:

```text
cell score >= 0.90 * best score in that reference row
```

It is near the column peak if:

```text
cell score >= 0.90 * best score in that prediction column
```

Then the two peak masks are combined:

```python
near_peak_score_mask = finite_cell_mask & (row_near_peak_mask | column_near_peak_mask)
```

The vertical bar means logical OR. A cell can survive this test if it is near the row peak or near the column peak.

Why both row and column?

- Row peak protects the best prediction matches for each reference window.
- Column peak protects the best reference matches for each prediction window.
- Repeated text can produce more than one plausible high-scoring region. Using row or column support avoids forcing only one diagonal too early.

Then the floor and peak conditions are combined:

```python
strong_match_mask = score_floor_mask & near_peak_score_mask
```

The ampersand means logical AND. A cell must satisfy both:

```text
score is high enough
and
score is near a row or column peak
```

This `strong_match_mask` is still not the final Hough input. It is the evidence from which the Region of Interest is built.

## Connected Components

Implementation file:

```text
tuner_parallel_v2_2/hough_preprocessing/connected_components.py
```

Connected-component labeling groups touching active cells into components. The code supports three backends:

```python
CONNECTED_COMPONENT_BACKEND_CYTHON = "cython"
CONNECTED_COMPONENT_BACKEND_SCIPY = "scipy"
CONNECTED_COMPONENT_BACKEND_PYTHON = "python"
```

The default is Cython. If the Cython helper is not available, the code can fall back to SciPy or pure Python:

```python
def _label_with_requested_backend(mask: np.ndarray, requested_backend: str) -> tuple[np.ndarray, int, str]:
    """Label components with the requested backend and safe fallbacks."""
    if requested_backend == CONNECTED_COMPONENT_BACKEND_CYTHON:
        cython_result = _try_cython_label(mask)
        if cython_result is not None:
            labels, component_count = cython_result
            return labels, int(component_count), CONNECTED_COMPONENT_BACKEND_CYTHON
        scipy_result = _try_scipy_label(mask)
        if scipy_result is not None:
            labels, component_count = scipy_result
            return labels, int(component_count), CONNECTED_COMPONENT_BACKEND_SCIPY
```

The Python fallback uses eight-connected neighbors. Eight-connected means a cell is connected to its horizontal, vertical, and diagonal neighbors:

```python
for neighbour_row in range(max(0, active_row - 1), min(row_count, active_row + 2)):
    for neighbour_column in range(max(0, active_column - 1), min(column_count, active_column + 2)):
        if neighbour_row == active_row and neighbour_column == active_column:
            continue
        if bool(active_mask[neighbour_row, neighbour_column]) and labels[neighbour_row, neighbour_column] == 0:
            labels[neighbour_row, neighbour_column] = current_label
            stack.append((neighbour_row, neighbour_column))
```

After labeling, each component receives a summary:

```python
ComponentSummary(
    label=int(component_label),
    cell_count=int(component_rows.size),
    row_count=int(np.unique(absolute_rows).size),
    column_count=int(np.unique(absolute_columns).size),
    y_minimum=int(absolute_rows.min()),
    y_maximum=int(absolute_rows.max()),
    x_minimum=int(absolute_columns.min()),
    x_maximum=int(absolute_columns.max()),
)
```

This summary tells the preprocessing whether the component is large enough to be useful.

## Keeping Only Useful Components

Implementation file:

```text
tuner_parallel_v2_2/hough_preprocessing/region_of_interest.py
```

The component gate is:

```python
def _component_passes_region_gate(
    *,
    component,
    config: HoughPreprocessingConfig,
) -> bool:
    """Return True when a connected component is large enough to search."""
    return (
        int(component.cell_count) >= int(config.minimum_component_cells)
        and int(component.row_count) >= int(config.minimum_component_rows)
        and int(component.column_count) >= int(config.minimum_component_columns)
    )
```

In the current default settings, a component survives if:

```text
cell_count >= 2
row_count >= 1
column_count >= 1
```

For the Finnish target document:

```text
component count = 152
kept component count = 36
dropped component count = 116
```

So many small or isolated pieces of high score were removed from the possible Region of Interest.

## Region Of Interest Mask

After the useful connected components are selected, the code creates a component-region mask:

```python
if kept_component_labels:
    component_region_mask = np.isin(component_labels, list(kept_component_labels))
else:
    component_region_mask = np.zeros_like(strong_match_mask, dtype=bool)
```

This mask is true only for cells that belong to kept components.

Then the mask is dilated:

```python
region_of_interest_mask = dilate_mask(
    component_region_mask,
    radius=int(preprocessing_config.region_dilation_radius),
)
```

Dilation expands each kept component by a small radius while preserving the original matrix shape. With `region_dilation_radius = 1`, a kept cell can expand to its immediate 3 x 3 neighborhood.

Why dilate?

The score matrix is discrete. A real text line may not pass exactly through every strongest cell. Dilation gives the nearby evidence a little room, but it still limits the Hough detector to a narrow neighborhood around strong evidence.

The final binary Hough input is not the whole Region of Interest. It is the intersection of:

```text
strong match evidence
and
Region of Interest
```

The code:

```python
hough_input_mask = strong_match_mask & region_of_interest_mask
```

This is effectively a bitwise AND on two boolean masks.

Your intuition about bitwise AND is correct here:

- `strong_match_mask` says which cells are strong score evidence.
- `region_of_interest_mask` says which cells are inside a line-like region.
- `hough_input_mask` keeps only cells that satisfy both.

## Hough Input Acceptance Or Rejection

After the final binary Hough input is built, the code summarizes its geometry:

```python
geometry = active_mask_geometry(hough_input_mask)
```

Implementation file:

```text
tuner_parallel_v2_2/hough_preprocessing/connected_components.py
```

The geometry calculation is:

```python
active_rows, active_columns = np.nonzero(active_mask)
active_cell_count = int(active_rows.size)
...
return {
    "active_cell_count": int(active_cell_count),
    "active_fraction": float(active_cell_count / active_mask.size) if active_mask.size else 0.0,
    "active_row_count": int(np.unique(active_rows).size),
    "active_column_count": int(np.unique(active_columns).size),
    "x_span": int(active_columns.max() - active_columns.min() + 1),
    "y_span": int(active_rows.max() - active_rows.min() + 1),
}
```

These values answer:

- How many cells can vote?
- What fraction of the matrix can vote?
- How many unique reference rows are touched?
- How many unique prediction columns are touched?
- How wide is the active area on the prediction axis?
- How tall is the active area on the reference axis?

The rejection gate is:

```python
def _first_geometry_rejection_reason(
    *,
    geometry: dict[str, int | float],
    kept_component_count: int,
    config: HoughPreprocessingConfig,
) -> str:
    """Return the first reason that makes the final Hough input unusable."""
    if int(kept_component_count) <= 0:
        return "no_line_like_region_of_interest"
    if int(geometry["active_cell_count"]) < int(config.minimum_active_cells):
        return "insufficient_hough_evidence"
    if int(geometry["active_row_count"]) < int(config.minimum_active_rows):
        return "insufficient_active_rows"
    if int(geometry["active_column_count"]) < int(config.minimum_active_columns):
        return "insufficient_active_columns"
    if int(geometry["x_span"]) < int(config.minimum_x_span):
        return "insufficient_x_span"
    if int(geometry["y_span"]) < int(config.minimum_y_span):
        return "insufficient_y_span"
    if float(geometry["active_fraction"]) > float(config.maximum_active_fraction):
        return "ambiguous_or_too_dense"
    return ""
```

This is why a document can show a binary Hough input but still be rejected. A binary image can be non-empty and visually interesting, but the gate can still reject it if, for example, the active fraction is too high. That case is named:

```text
ambiguous_or_too_dense
```

This is especially important for small matrices. A clean diagonal in a small matrix can occupy more than `maximum_active_fraction = 0.08` of the matrix. Visually it may look line-like, but the current density gate can still reject it. That is an algorithmic behavior to evaluate carefully, not a plotting error.

## Hough Context Returned To The Tuner

The preprocessing returns a dictionary consumed by the Hough detector:

```python
binary_hough_image = np.asarray(hough_input_mask, dtype=bool).astype(np.uint8)
context = {
    "hough_image": binary_hough_image,
    "hough_mask_bool": np.asarray(hough_input_mask, dtype=bool),
    "mask": binary_hough_image,
    "threshold_start": float("nan"),
    "preprocessing_mode": "region_of_interest",
    "hough_preprocessing_accepted": bool(summary.get("accepted", False)),
    "hough_preprocessing_rejection_reason": str(summary.get("rejection_reason", "")),
    "hough_preprocessing_summary": dict(summary),
    "region_of_interest_mask_bool": np.asarray(region_of_interest_mask, dtype=bool),
    "strong_match_mask_bool": np.asarray(strong_match_mask, dtype=bool),
    "score_floor_mask_bool": np.asarray(score_floor_mask, dtype=bool),
    "near_peak_score_mask_bool": np.asarray(near_peak_score_mask, dtype=bool),
}
```

The key field for Hough is:

```text
hough_image
```

That is the binary image passed into `skimage.transform.probabilistic_hough_line`.

## Hough Detection

Implementation file:

```text
tuner_parallel_v2_2/alignment/line_alignment_pipeline_fast.py
```

The code uses only falling diagonals:

```python
FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES = 30.0
FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES = 60.0
FALLING_DIAGONAL_NORMAL_THETA_DEG = np.arange(-59.5, -30.0, 0.5)
FALLING_DIAGONAL_NORMAL_THETA_RAD = np.deg2rad(FALLING_DIAGONAL_NORMAL_THETA_DEG)
```

This means the detector looks for lines that move from upper-left to lower-right in matrix coordinates. That is the expected direction when reference and prediction progress together.

The detector runs here:

```python
raw_hough_segments_from_skimage = list(
    probabilistic_hough_line(
        thresholded_hough_image,
        threshold=int(threshold),
        line_length=int(line_length),
        line_gap=int(line_gap),
        theta=FALLING_DIAGONAL_NORMAL_THETA_RAD,
        rng=np.random.default_rng(int(seed)),
    )
)
```

The seed is made deterministic per document:

```python
det = detect_lines_only_from_hough_ctx(
    hough_ctx=hough_ctx,
    seed=int(hough_seed) + int(document_index),
    threshold=int(hough_threshold),
    line_length=int(hough_line_length),
    line_gap=int(hough_line_gap),
)
```

For document index `573` and `hough_seed = 1`, the effective seed is:

```text
573 + 1 = 574
```

The detector then applies an endpoint direction guard:

```python
raw_lines = keep_only_falling_diagonal_hough_segments(raw_hough_segments_from_skimage)
direction_rejected_line_count = int(skimage_raw_line_count - len(raw_lines))
```

This keeps the Hough theta restriction honest. If a segment still points the wrong way, it is rejected before filtering.

## Line Filtering

Implementation file:

```text
tuner_parallel_v2_2/alignment/line_alignment_pipeline_fast.py
```

Raw Hough segments are converted into line records and then filtered:

```python
candidate_segments = list(det_result.get("candidate_segments", []))
lines_for_filtering = line_records_from_raw_hough_segments(mat, candidate_segments)
```

Then ownership filtering decides which lines survive:

```python
lines_used, column_assignment = filter_lines_for_alignment_by_ownership(
    lines_for_filtering,
    mat,
    mask_bool,
    abs_min_len=float(align_abs_min_len),
    min_iou_threshold=float(align_min_iou_threshold),
    profile=profile,
)
```

`align_min_iou_threshold` is the minimum true Intersection over Union required by the geometry filter. In the final run, the value was:

```text
0.035
```

`align_abs_min_len` is present in the configuration, but the final run used:

```text
0.0
```

So there was no absolute minimum line length filter forcing short lines out at this stage.

The final run also used a line-level text filter:

```text
min_surviving_line_nls = 0.5
```

This means: after geometry filtering, a final line can be removed if its line-level Normalized Levenshtein Similarity is below 0.5.

The relevant evaluation code is in:

```text
tuner_parallel_v2_2/tuner/hough_eval.py
```

The filter is invoked here:

```python
if min_surviving_line_nls is not None:
    if line_nls_filter_ref_blocks is None or line_nls_filter_other_blocks is None:
        raise ValueError("line-NLS filtering requires reference and prediction text blocks")
    if levenshtein_backend is None:
        raise ValueError("line-NLS filtering requires a Levenshtein backend")

    filtered, precomputed_weighted_result, line_nls_filter_fields = (
        _filter_ref_to_pred_lines_by_minimum_nls(
            filtered=filtered,
            ref_blocks=line_nls_filter_ref_blocks,
            pred_blocks=line_nls_filter_other_blocks,
            n_ref_windows=int(matrix.shape[0]) if matrix.ndim == 2 else 0,
            min_surviving_line_nls=float(min_surviving_line_nls),
            levenshtein_backend=str(levenshtein_backend),
        )
    )
```

This filter is why the number of raw Hough lines, candidate lines, geometry-filtered lines, and final surviving lines can differ.

## The Normal `tuning_score`

Implementation file:

```text
tuner_parallel_v2_2/metrics/alignment_quality_score.py
```

The normal tuner score is still calculated even when `--selection-objective alignment_evidence` is active.

The code:

```python
def compute_harmonic_tuning_score(
    *,
    weighted_along_lines_nls,
    correct_ref_coverage,
    hallucination,
) -> float:
    """Compute the final harmonic tuner objective in ``[0, 1]``.

    A zero-quality component makes the harmonic score zero.  This avoids division
    by zero and ensures combinations with no meaningful alignment cannot win
    because of one strong component alone.
    """
    weighted_nls = clamp_unit_interval(weighted_along_lines_nls)
    coverage = clamp_unit_interval(correct_ref_coverage)
    hallucination_rate = clamp_unit_interval(hallucination)
    non_hallucination = clamp_unit_interval(1.0 - hallucination_rate)

    if weighted_nls <= 0.0 or coverage <= 0.0 or non_hallucination <= 0.0:
        return 0.0

    score = 3.0 / ((1.0 / weighted_nls) + (1.0 / coverage) + (1.0 / non_hallucination))
    return clamp_unit_interval(score)
```

The three inputs are:

- `weighted_along_lines_nls`: line-level text similarity along the final Hough lines.
- `correct_ref_coverage`: how much of the reference is correctly covered, computed by the v2.12 coverage logic.
- `hallucination`: how much prediction content is not supported by the reference alignment.

The score uses:

```text
non_hallucination = 1 - hallucination
```

Then it computes the harmonic mean of:

```text
weighted_along_lines_nls
correct_ref_coverage
non_hallucination
```

The harmonic mean is strict. A low value in any component pulls the whole score down. That is why the tuner cannot get a high `tuning_score` simply by doing well on one dimension while failing badly on another.

## The `alignment_evidence` Selection Score

Implementation file:

```text
tuner_parallel_v2_2/metrics/alignment_quality_score.py
```

`alignment_evidence` adds a second selection score:

```python
def compute_alignment_evidence_selection_score(
    *,
    weighted_along_lines_nls,
    score_matrix_support,
    line_guided_fraction,
    hallucination,
) -> float:
    """Score how strongly the matrix and final geometry support an alignment.

    This score is a selection objective only.  It does not replace the final
    scientific metrics, and it does not hide repetition or missing-reference
    penalties from the exported result.  It is useful when the Hough winner
    should prefer a matrix-supported repeated line over a geometrically neat but
    less faithful alternative.
    """
    non_hallucination = clamp_unit_interval(1.0 - clamp_unit_interval(hallucination))
    return compute_balanced_harmonic_mean(
        [
            clamp_unit_interval(weighted_along_lines_nls),
            clamp_unit_interval(score_matrix_support),
            clamp_unit_interval(line_guided_fraction),
            non_hallucination,
        ]
    )
```

This score uses four inputs:

1. `weighted_along_lines_nls`
2. `score_matrix_support`
3. `line_guided_fraction`
4. `non_hallucination`

It deliberately does not use `correct_ref_coverage` as directly as the normal `tuning_score` does. The purpose is different:

- `tuning_score` asks: how good is the final scientific score?
- `alignment_evidence` asks: how well supported is this Hough geometry by the score matrix and the final line-guided alignment?

The exported metrics still contain the normal scientific penalties. `alignment_evidence` only chooses which Hough parameter combination becomes the visual and per-document winner.

### Score Matrix Support

Implementation file:

```text
tuner_parallel_v2_2/metrics/alignment_quality_score.py
```

Code:

```python
def compute_score_matrix_support_from_lines(lines_used: Sequence[dict]) -> float:
    """Return average score-matrix support for final surviving lines.

    Each final line carries ``owned_score_mean`` from the score matrix cells that
    the final assignment gave to that line.  The score matrix is measured on the
    familiar 0..100 percentage scale, so this helper converts it to 0..1 before
    it is combined with other tuner signals.
    """
    weighted_support_sum = 0.0
    support_weight_sum = 0.0

    for line in lines_used:
        ...
        support_percent = float(line.get("owned_score_mean", 0.0))
        ...
        owned_column_count = float(line.get("owned_cols", 0.0))
        line_weight = owned_column_count if owned_column_count > 0.0 else euclidean_line_length(line)
        ...
        weighted_support_sum += clamp_unit_interval(support_percent / 100.0) * float(line_weight)
        support_weight_sum += float(line_weight)

    if support_weight_sum <= 0.0:
        return 0.0
    return clamp_unit_interval(weighted_support_sum / support_weight_sum)
```

In words:

- Each final surviving line owns some score-matrix cells.
- Those owned cells have a mean score called `owned_score_mean`.
- The matrix score is on a 0..100 scale.
- The code converts it to 0..1 by dividing by 100.
- Longer or more column-owning lines count more than tiny lines.
- The final result is a weighted average of matrix support across surviving lines.

This is the part that lets `alignment_evidence` prefer a line arrangement supported by the score matrix itself, even if a different parameter combination has a higher normal `tuning_score`.

### Line-Guided Fraction

Implementation file:

```text
tuner_parallel_v2_2/metrics/alignment_quality_score.py
```

Code:

```python
def compute_line_guided_fraction(*, line_guided_columns, fallback_columns) -> float:
    """Return the fraction of prediction columns explained by detected lines.

    ``line_guided_columns`` are prediction windows assigned through surviving
    Hough lines.  ``fallback_columns`` are prediction windows that had to be
    handled without a surviving line.  A value near one means the geometry is
    actually doing the alignment work.
    """
    guided_count = max(0.0, float(line_guided_columns))
    fallback_count = max(0.0, float(fallback_columns))
    total_columns = guided_count + fallback_count
    if total_columns <= 0.0:
        return 0.0
    return clamp_unit_interval(guided_count / total_columns)
```

In words:

- A prediction column is line-guided if a surviving Hough line explains it.
- A prediction column is fallback if no surviving Hough line explains it.
- The fraction is:

```text
line_guided_columns / (line_guided_columns + fallback_columns)
```

If this value is close to 1, most prediction windows are explained by detected lines. If it is lower, the metric had to rely more on fallback behavior.

## Where `alignment_evidence` Is Added To The Evaluation Row

Implementation file:

```text
tuner_parallel_v2_2/tuner/hough_eval.py
```

The evaluation row gets the extra selection fields here:

```python
def _selection_metric_fields(
    *,
    ref_to_pred_payload: dict,
    weighted_along_lines_nls,
    hallucination,
) -> dict:
    """Return scalar fields used by the optional alignment-evidence selector."""
    filtered_payload = ref_to_pred_payload.get("filtered", {}) if isinstance(ref_to_pred_payload, dict) else {}
    lines_used = filtered_payload.get("lines_used", []) if isinstance(filtered_payload, dict) else []
    line_guided_fraction = compute_line_guided_fraction(
        line_guided_columns=ref_to_pred_payload.get("line_guided_columns", 0),
        fallback_columns=ref_to_pred_payload.get("fallback_columns", 0),
    )
    score_matrix_support = compute_score_matrix_support_from_lines(lines_used if isinstance(lines_used, list) else [])
    alignment_selection_score = compute_alignment_evidence_selection_score(
        weighted_along_lines_nls=weighted_along_lines_nls,
        score_matrix_support=score_matrix_support,
        line_guided_fraction=line_guided_fraction,
        hallucination=hallucination,
    )
    return {
        "alignment_selection_score": float(alignment_selection_score),
        "score_matrix_support": float(score_matrix_support),
        "line_guided_fraction": float(line_guided_fraction),
    }
```

The final evaluation row includes both the normal scientific score and the selection score:

```python
eval_row = {
    "is_valid": True,
    ...
    "tuning_score": float(tuning_score),
    "weighted_along_lines_nls": None if weighted_along_lines_nls is None else float(weighted_along_lines_nls),
    "correct_ref_coverage": float(normalized_coverage_metrics["correct_ref_coverage"]),
    "missing_ref_coverage": float(normalized_coverage_metrics["missing_ref_coverage"]),
    "repetition_on_ref": float(normalized_coverage_metrics["repetition_on_ref"]),
    "hallucination": float(normalized_coverage_metrics["hallucination"]),
    **_selection_metric_fields(...),
    **_line_count_fields(...),
    **_line_nls_filter_fields_from_payload(ref_to_pred_payload),
    ...
}
```

This is why the CSV contains both:

```text
tuning_score
alignment_selection_score
score_matrix_support
line_guided_fraction
```

## How The Winning Combination Is Chosen

Implementation file:

```text
tuner_parallel_v2_2/tuner/hough_eval.py
```

The ranking function is:

```python
def evaluation_rank_key(row: dict, *, selection_objective: str = DEFAULT_SELECTION_OBJECTIVE) -> tuple:
    """Return a strict deterministic ranking tuple for best-evaluation selection."""
    objective = normalize_selection_objective(selection_objective)
    hallucination = _finite_float_for_rank(row.get("hallucination"), 1.0)

    if objective == SELECTION_OBJECTIVE_ALIGNMENT_EVIDENCE:
        return (
            _finite_float_for_rank(row.get("alignment_selection_score"), float("-inf")),
            _finite_float_for_rank(row.get("score_matrix_support"), float("-inf")),
            _finite_float_for_rank(row.get("line_guided_fraction"), float("-inf")),
            _finite_float_for_rank(row.get("weighted_along_lines_nls"), float("-inf")),
            -float(hallucination),
            _finite_float_for_rank(row.get("tuning_score"), float("-inf")),
            _finite_float_for_rank(row.get("correct_ref_coverage"), float("-inf")),
            int(row.get("line_guided_columns", 0)),
            -int(row.get("fallback_columns", 0)),
            -int(row.get(PARAM_HOUGH_THRESHOLD, 0)),
            -int(row.get(PARAM_HOUGH_LINE_LENGTH, 0)),
            -int(row.get(PARAM_HOUGH_LINE_GAP, 0)),
            -int(row.get(PARAM_HOUGH_SEED, 0)),
        )

    return (
        _finite_float_for_rank(row.get("tuning_score"), float("-inf")),
        _finite_float_for_rank(row.get("weighted_along_lines_nls"), float("-inf")),
        _finite_float_for_rank(row.get("correct_ref_coverage"), float("-inf")),
        -float(hallucination),
        int(row.get("line_guided_columns", 0)),
        -int(row.get("fallback_columns", 0)),
        -int(row.get(PARAM_HOUGH_THRESHOLD, 0)),
        -int(row.get(PARAM_HOUGH_LINE_LENGTH, 0)),
        -int(row.get(PARAM_HOUGH_LINE_GAP, 0)),
        -int(row.get(PARAM_HOUGH_SEED, 0)),
    )
```

This is the key behavioral change.

With the default strict objective, the first ranking value is:

```text
tuning_score
```

With `--selection-objective alignment_evidence`, the first ranking value is:

```text
alignment_selection_score
```

The full `alignment_evidence` ranking order is:

1. Higher `alignment_selection_score`.
2. Higher `score_matrix_support`.
3. Higher `line_guided_fraction`.
4. Higher `weighted_along_lines_nls`.
5. Lower `hallucination`.
6. Higher `tuning_score`.
7. Higher `correct_ref_coverage`.
8. More `line_guided_columns`.
9. Fewer `fallback_columns`.
10. Smaller `hough_threshold`.
11. Smaller `hough_line_length`.
12. Smaller `hough_line_gap`.
13. Smaller `hough_seed`.

The last four are deterministic tie-breakers. They do not dominate the metric. They only decide between rows that are otherwise tied.

The comparison function is:

```python
if evaluation_rank_key(candidate, selection_objective=selection_objective) > evaluation_rank_key(
    current,
    selection_objective=selection_objective,
):
    return candidate
return current
```

So the tuner compares tuples. Python compares tuple elements from left to right. The first differing element decides which row wins.

## Concrete Example: Finnish `newseye-fin_576458_0002_23676306.jpeg`

For the target Finnish document, the highest normal `tuning_score` winner was:

```text
hough_threshold = 34
hough_line_length = 16
hough_line_gap = 0
hough_seed = 1
tuning_score = 0.878550708351766
alignment_selection_score = 0.8926203182074048
score_matrix_support = 0.7747879314619837
line_guided_fraction = 0.9246753246753247
weighted_along_lines_nls = 0.9719644073022038
correct_ref_coverage = 0.7658947065592635
hallucination = 0.07425229493633402
used_line_count = 5
line_guided_columns = 356
fallback_columns = 29
```

The current stitched Finnish solution selected by `alignment_evidence` was:

```text
hough_threshold = 16
hough_line_length = 4
hough_line_gap = 1
hough_seed = 1
tuning_score = 0.8596984258094983
alignment_selection_score = 0.9234496846155938
score_matrix_support = 0.7769317793405506
line_guided_fraction = 0.9974025974025974
weighted_along_lines_nls = 0.9610474520001112
correct_ref_coverage = 0.6907364787111623
hallucination = 0.0013325436778205508
used_line_count = 8
line_guided_columns = 384
fallback_columns = 1
```

Why did `alignment_evidence` choose the second row even though its normal `tuning_score` is lower?

Because `alignment_evidence` is primarily ranking by:

```text
alignment_selection_score
```

The current stitched row has:

```text
0.9234496846155938
```

The strict tuning-score row has:

```text
0.8926203182074048
```

The current stitched row also has much stronger line-guided coverage of prediction windows:

```text
384 line-guided columns, 1 fallback column
line_guided_fraction = 0.9974025974025974
```

The strict tuning-score row has:

```text
356 line-guided columns, 29 fallback columns
line_guided_fraction = 0.9246753246753247
```

So the strict row scores better on normal final `tuning_score`, but the alignment-evidence row is more completely explained by surviving Hough lines and has far lower hallucination:

```text
strict hallucination = 0.07425229493633402
alignment_evidence hallucination = 0.0013325436778205508
```

That is exactly the reason this selector was added: the chosen Hough geometry should be supported by the score matrix and should explain the prediction columns through actual lines, not only optimize the old harmonic score.

## What `alignment_evidence` Does Not Do

`alignment_evidence` does not remove scientific penalties from the result.

It does not say:

```text
Ignore repetition.
Ignore hallucination.
Ignore missing reference coverage.
Ignore bad transcription.
```

It only changes which Hough parameter row is selected as the winner. The exported row still contains:

```text
tuning_score
weighted_along_lines_nls
correct_ref_coverage
missing_ref_coverage
repetition_on_ref
hallucination
score_matrix_support
line_guided_fraction
alignment_selection_score
```

So downstream analysis can still see the lower `tuning_score` and all coverage penalties.

## The Critical Point

The old strict objective answers this question:

```text
Which parameter combination has the highest harmonic score from text similarity,
reference coverage, and non-hallucination?
```

The new `alignment_evidence` objective answers this question:

```text
Which parameter combination gives a text-similar, low-hallucination alignment
whose surviving Hough lines are strongly supported by the score matrix and
explain most prediction windows?
```

Those are related but not identical questions.

This difference matters for repeated text. A model can repeat a correct-looking phrase. The score matrix may legitimately contain multiple high-scoring regions for that same reference text. A selector that only wants one clean global diagonal can make a false move. The `alignment_evidence` selector is designed to give the matrix-supported repeated alignment a chance to survive, while still exporting repetition and hallucination metrics so the model can be punished where it should be punished.

