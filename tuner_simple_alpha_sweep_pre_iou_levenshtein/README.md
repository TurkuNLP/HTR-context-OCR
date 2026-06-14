# tuner\_simple\_alpha\_sweep\_pre\_iou\_levenshtein — User Guide

This document is written for someone reading this codebase for the first time. Every option is explained from first principles, every output file is described, and every key source file is summarised so you know where to look when something needs changing.

---

## Table of Contents

1. [What this pipeline does](#1-what-this-pipeline-does)
2. [Prerequisites and first-time setup](#2-prerequisites-and-first-time-setup)
3. [Core concepts](#3-core-concepts)
4. [Running with alpha sweep (recommended mode)](#4-running-with-alpha-sweep)
5. [Running with a fixed Levenshtein threshold (--minimum-pre-hough-levenshtein)](#5-running-with-a-fixed-levenshtein-threshold)
6. [Harmonic mode: choosing a candidate scoring formula](#6-harmonic-mode)
7. [Output files and directory layout](#7-output-files-and-directory-layout)
8. [Running at scale with Slurm (run\_tunner\_atomic.sh)](#8-running-at-scale-with-slurm)
9. [How each script and module works](#9-how-each-script-and-module-works)
10. [Full CLI reference](#10-full-cli-reference)

---

## 1. What this pipeline does

This pipeline measures how well an OCR model's predicted text matches the reference (ground-truth) text for a collection of documents. It uses **probabilistic Hough detection** on character-level similarity matrices to find which predicted text regions align with which reference text regions, then computes five scientific metrics for each document:

| Metric | Meaning |
|---|---|
| `document_normalised_levenshtein` | Whole-document character-level similarity (no Hough, just raw text comparison) |
| `weighted_along_lines_normalised_levenshtein` (NLS) | Character similarity averaged across only the aligned Hough lines |
| `correct_ref_coverage` | Fraction of reference characters that are covered by at least one aligned line |
| `missing_ref_coverage` | Fraction of reference characters that no line covers |
| `repetition_on_reference` | Fraction of reference characters covered by more than one line (overlap) |
| `hallucination` | Fraction of prediction characters that do not correspond to any reference region |

The pipeline evaluates two text directions — reference-to-prediction and reference-to-reference — and subtracts the reference-to-reference signal from the reference-to-prediction signal to remove self-similarity noise. The final metrics therefore reflect how much real new content the prediction contains, not background noise.

---

## 2. Prerequisites and first-time setup

### Required inputs

Before running the pipeline you need three files on disk:

| File | Description |
|---|---|
| **Runfile JSON** (`outputs.json`) | A list of documents. Each entry contains the document filename, main language, document type, the full reference text, and the full predicted text. |
| **Ref-to-pred score matrix pickle** | A `.pkl` file mapping each document filename to a 2-D NumPy array. Each cell `[row, col]` is the Levenshtein similarity between reference window `row` and prediction window `col`. |
| **Ref-to-ref score matrix pickle** | The same structure but comparing each reference window to every other reference window (self-similarity). |

If a document is in the runfile but missing from either pickle, the pipeline automatically computes its score matrix from the raw text using a sliding window Levenshtein calculation. This is slower than loading from the pickle but guarantees every document can be processed.

### Building the Cython accelerators

The first time you run either shell script it automatically builds the compiled extensions:

```bash
cd /scratch/project_2017385/dorian/Churro_copy/tuner_simple_alpha_sweep_pre_iou_levenshtein
python3 cython_accel/build.py build_ext --inplace
```

Both `run_tunner.sh` and `run_tunner_atomic.sh` call this automatically before launching Python. If the build fails they fall back to pure-Python equivalents automatically — the results are identical, just slower.

Set `TUNER_SIMPLE_SKIP_CYTHON_BUILD=1` to skip this step when you know the extensions are already compiled.

---

## 3. Core concepts

### Score matrices

A score matrix is a 2-D array of shape `(n_reference_windows, n_prediction_windows)`. Each window represents a fixed-size chunk of text. The default is `window_size=50` characters, advancing by `window_stride=35` characters per step. So for a document with 1000 reference characters and 800 prediction characters you get roughly `(1000 - 50) / 35 + 1 ≈ 28` reference windows and `(800 - 50) / 35 + 1 ≈ 22` prediction windows.

### The pre-Hough mask

Hough detection requires a binary (black/white) input image. The pipeline converts the float score matrix into a binary mask by keeping only cells whose score is at or above a threshold. Two rules can build this mask:

- **Alpha sweep rule** (default): `threshold = mean(finite_cells) + alpha × std(finite_cells)`. Cells above this threshold become active. As alpha increases, fewer cells stay active.
- **Fixed Levenshtein rule** (`--minimum-pre-hough-levenshtein`): `threshold = the value you provide`. Cells with a raw Levenshtein score at or above this threshold become active. The value can be given in unit scale (0.0–1.0) or percent scale (0–100) — the pipeline detects the matrix scale automatically and normalises.

### Probabilistic Hough detection

Hough detection looks for straight diagonal lines in the binary mask. In the context of text alignment, a falling diagonal line crossing the matrix from upper-left to lower-right means the model predicted text in the same order as the reference — the hallmark of correct alignment. The pipeline uses `skimage.transform.probabilistic_hough_line` with configurable threshold, minimum line length, and maximum gap parameters.

### Pre-IoU Levenshtein filter

After Hough detection produces a set of raw line segments, the pipeline computes a text similarity score for each raw segment before the expensive IoU-based merge step. Segments whose text similarity falls below `--min-surviving-line-nls` are removed. This filter runs before true-IoU merging, making it faster than post-IoU filtering on weak segments.

### Alpha sweep and candidate selection

Rather than choosing a single alpha by hand, the pipeline evaluates every alpha candidate in a configurable range (default 1.0 to 4.0 in steps of 0.2). For each candidate it runs the full pipeline and computes a **harmonic selection score** from the three core metrics (NLS, coverage, non-hallucination). The candidate with the highest harmonic score is kept as the final result.

---

## 4. Running with alpha sweep

This is the recommended mode. The pipeline evaluates every alpha in the configured range and automatically selects the best one per document.

### Minimal example

```bash
bash /scratch/project_2017385/dorian/Churro_copy/tuner_simple_alpha_sweep_pre_iou_levenshtein/run_tunner.sh \
  --runfile-json   /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json \
  --output-dir     /scratch/project_2017385/dorian/Churro_copy/results/tuner_simple_alpha_sweep_pre_iou_levenshtein_all_languages_minIoU_0_10_line_length_3_hough_thresh_3_alpha_0_6_to_4_step_0_1_line_levenshtein_0_5 \
  --scores-pkl-ref-to-pred /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/old_scores_reference_prediction_ws50_st35_levenshtein.pkl \
  --scores-pkl-ref-to-ref  /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/old_scores_reference_self_ws50_st35_levenshtein.pkl \
  --all-languages \
  --alpha-sweep \
  --alpha-sweep-min 0.6 \
  --alpha-sweep-max 4.0 \
  --alpha-sweep-step 0.1 \
  --hough-threshold 3 \
  --hough-line-length 3 \
  --hough-line-gap 15 \
  --hough-seed 1 \
  --align-min-iou-threshold 0.10 \
  --min-surviving-line-nls 0.5 \
  --harmonic-mode balanced \
  --plot-mode stitched-language
```

Because `--harmonic-mode balanced` is used, the final CSV files and plots will be written to:
```
.../results/.../balanced/
```

### What each option does

| Option | Value in example | Meaning |
|---|---|---|
| `--runfile-json` | `outputs.json` | The document list. Every entry must have `fname`, `main_language`, `document_type`, `reference_text`, and `prediction_text`. |
| `--output-dir` | long path | Parent directory. A subdirectory named after `--harmonic-mode` is created automatically inside it. |
| `--scores-pkl-ref-to-pred` | `.pkl` path | Pre-computed reference-to-prediction score matrices. The pipeline falls back to computing matrices on the fly for any missing documents. |
| `--scores-pkl-ref-to-ref` | `.pkl` path | Pre-computed reference-to-reference score matrices. Same fallback applies. |
| `--all-languages` | flag | Process every language in the runfile. You can instead use `--language Finnish --language Swedish` to select specific ones. |
| `--alpha-sweep` | flag | Enable per-document alpha sweep. This is the default and can be omitted; it is shown here for clarity. |
| `--alpha-sweep-min 0.6` | 0.6 | The smallest alpha the sweep will try. A lower alpha means a lower threshold and more active cells in the binary mask, which produces more Hough lines but also more noise. |
| `--alpha-sweep-max 4.0` | 4.0 | The largest alpha the sweep will try. A higher alpha means a higher threshold and fewer active cells, which produces fewer but more confident Hough lines. |
| `--alpha-sweep-step 0.1` | 0.1 | How much alpha increases between candidates. `0.6` to `4.0` in steps of `0.1` produces 35 candidates per document. |
| `--hough-threshold 3` | 3 | Minimum number of Hough votes a line must collect to be retained. Lower values produce more lines; higher values require more evidence per line. |
| `--hough-line-length 3` | 3 | Minimum accepted line length in pixels (cells). Lines shorter than this are discarded immediately after detection. |
| `--hough-line-gap 15` | 15 | Maximum allowed gap inside a Hough line. Larger gaps let the algorithm bridge sparse regions of the binary mask. |
| `--hough-seed 1` | 1 | Integer seed passed to the probabilistic Hough algorithm. Fixing this makes runs reproducible. |
| `--align-min-iou-threshold 0.10` | 0.10 | Minimum intersection-over-union overlap between a Hough line and a score-matrix cell for the line to claim that cell. Lower values let lines claim more loosely overlapping cells. |
| `--min-surviving-line-nls 0.5` | 0.5 | Minimum line-level normalised Levenshtein similarity for a raw Hough segment to survive the pre-IoU filter. Set to `0` or omit to disable this filter entirely. |
| `--harmonic-mode balanced` | balanced | Which formula to use when scoring alpha candidates (see [Section 6](#6-harmonic-mode)). |
| `--plot-mode stitched-language` | stitched-language | Create one stitched overview image per language. Use `none` to skip plotting entirely. |

### Running multiple harmonic modes on the same data

Because the harmonic-mode subdirectory is created automatically, you can compare all three modes on the same dataset by running the same command three times with different `--harmonic-mode` values and the same parent `--output-dir`:

```bash
for MODE in balanced coverage-hallucination-priority coverage-hallucination-only; do
  bash run_tunner.sh \
    --output-dir /scratch/.../results/my_experiment \
    --harmonic-mode "${MODE}" \
    [... other options ...]
done
```

This produces three separate result directories, each with its own CSVs and plots:
```
my_experiment/balanced/
my_experiment/coverage-hallucination-priority/
my_experiment/coverage-hallucination-only/
```

---

## 5. Running with a fixed Levenshtein threshold

When you pass `--minimum-pre-hough-levenshtein`, the pipeline builds the pre-Hough binary mask by keeping every cell whose score is at or above your threshold. There is no mean, no standard deviation, and no alpha — just a single fixed cutoff. The alpha sweep is skipped entirely; only one candidate runs per document.

### Example command

```bash
bash /scratch/project_2017385/dorian/Churro_copy/tuner_simple_alpha_sweep_pre_iou_levenshtein/run_tunner.sh \
  --runfile-json   /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json \
  --output-dir     /scratch/project_2017385/dorian/Churro_copy/results/tuner_simple_alpha_sweep_pre_iou_levenshtein_all_languages_minIoU_0_10_line_length_3_hough_thresh_3_alpha_0_6_to_4_step_0_1_line_levenshtein_0_5 \
  --scores-pkl-ref-to-pred /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/old_scores_reference_prediction_ws50_st35_levenshtein.pkl \
  --scores-pkl-ref-to-ref  /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/old_scores_reference_self_ws50_st35_levenshtein.pkl \
  --all-languages \
  --minimum-pre-hough-levenshtein 0.5 \
  --hough-threshold 3 \
  --hough-line-length 3 \
  --hough-line-gap 15 \
  --hough-seed 1 \
  --align-min-iou-threshold 0.10 \
  --min-surviving-line-nls 0.5 \
  --harmonic-mode balanced \
  --plot-mode stitched-language
```

### What `--minimum-pre-hough-levenshtein` does

The value you provide is interpreted as the minimum Levenshtein similarity a cell must have to be included in the binary Hough input mask. The scale is detected automatically:

- If the score matrix stores values in the 0–100 range (percent scale), a value of `0.5` is converted to `50.0` internally, and a value of `50.0` stays as `50.0`.
- If the score matrix stores values in the 0–1 range (unit scale), a value of `0.5` stays as `0.5`, and a value of `50.0` is converted to `0.50`.

This means `0.30` and `30.0` always produce the same result regardless of matrix scale, making the parameter intuitive without needing to know your pickle's internal format.

### What changes compared to alpha sweep

| Behaviour | Alpha sweep | Fixed Levenshtein |
|---|---|---|
| Number of candidates evaluated per document | 1 per alpha (e.g. 35) | 1 (always) |
| Score statistics (mean, std) computed | Yes — needed for threshold | No — threshold is fixed; skipped for speed |
| Alpha sweep pickle written | Yes | No |
| `score_mean_ref_to_pred` in CSV | Computed value | `NaN` (not needed) |
| `score_standard_deviation_ref_to_pred` in CSV | Computed value | `NaN` (not needed) |
| All six scientific metric values | Identical computation | Identical computation |
| Speed | Slower (many candidates) | Faster (one candidate, statistics skipped) |

The five scientific metrics (`document_normalised_levenshtein`, `weighted_along_lines_normalised_levenshtein`, `correct_ref_coverage`, `missing_ref_coverage`, `repetition_on_reference`, `hallucination`) are computed with exactly the same code in both modes — only the mask-building step differs.

---

## 6. Harmonic mode

### Purpose

After computing all metrics for one alpha candidate, the pipeline needs a single number to decide whether this candidate is better or worse than others. It uses a **harmonic mean** of the three core metrics: NLS, coverage, and non-hallucination (`1 - hallucination`). The harmonic mean is chosen because it penalises extreme imbalances — a candidate with perfect coverage but zero NLS scores poorly, as it should.

Three formulas are available, selectable via `--harmonic-mode`:

### Available modes

#### `balanced` (default)

```
score = 3 / (1/NLS + 1/coverage + 1/(1 - hallucination))
```

All three terms carry equal weight. This is the original formula. Use it when you want NLS, coverage, and hallucination to contribute equally to alpha selection.

#### `coverage-hallucination-priority`

```
score = 5 / (1/NLS + 2/coverage + 2/(1 - hallucination))
```

Coverage and non-hallucination each carry twice the weight of NLS. Use this when you care more about whether the reference is fully covered and the prediction is clean than about the character-level similarity of the aligned text.

#### `coverage-hallucination-only`

```
score = 2 / (1/coverage + 1/(1 - hallucination))
```

NLS is excluded entirely. Alpha selection is driven purely by how much of the reference is covered and how little is hallucinated. Use this when the model's text quality is known to be good and you want selection to focus entirely on spatial alignment quality.

### How to pass the mode

```bash
# Default — equal weight on all three terms
--harmonic-mode balanced

# Double weight on coverage and non-hallucination
--harmonic-mode coverage-hallucination-priority

# NLS excluded; coverage and non-hallucination only
--harmonic-mode coverage-hallucination-only
```

### How the mode affects the output directory

The chosen mode is appended as a subdirectory to `--output-dir` automatically. You do **not** need to add it yourself:

```
--output-dir /path/to/parent  --harmonic-mode balanced
  → actual output: /path/to/parent/balanced/

--output-dir /path/to/parent  --harmonic-mode coverage-hallucination-priority
  → actual output: /path/to/parent/coverage-hallucination-priority/

--output-dir /path/to/parent  --harmonic-mode coverage-hallucination-only
  → actual output: /path/to/parent/coverage-hallucination-only/
```

The `harmonic_mode` column is also written to every row of `best_combination_per_document.csv` so you can always reconstruct which formula produced a given result.

### Environment variable alternative

```bash
export HARMONIC_MODE=coverage-hallucination-priority
bash run_tunner.sh --output-dir /path/to/parent [... other options ...]
```

---

## 7. Output files and directory layout

After a successful run the directory `<output-dir>/<harmonic-mode>/` contains:

```
<harmonic-mode>/
├── best_combination_per_document.csv      # One row per document; the best alpha's full metrics
├── compact_combination_metrics.csv        # Subset of columns from the above for quick inspection
├── document_type_summary.csv             # Per-document-type aggregated statistics
├── loadable_documents.csv                 # Documents that were in the runfile and had matrix data
├── loaded_documents.csv                   # Documents actually processed (after filters/limits)
├── runfile_documents.csv                  # All documents from the runfile
├── skipped_documents.csv                  # Documents skipped (too small, exception, etc.)
├── run_summary.json                       # Metadata: run duration, counts, config snapshot
├── plots/                                 # Only when --plot-mode is not "none"
│   └── <language>/                        # One directory per language
│       └── stitched_<language>.png        # Overview of all documents for that language
└── alpha_sweep_pickles/                   # Only in alpha-sweep mode (not with --minimum-pre-hough-levenshtein)
    └── <language>/
        └── <document_fname>.pkl           # Per-document audit pickle with all alpha candidates
```

### Key CSV columns

`best_combination_per_document.csv` contains all metrics and audit columns. The most important ones:

| Column | Description |
|---|---|
| `fname` | Document filename |
| `main_language` | Language label |
| `document_normalised_levenshtein` | Whole-document character similarity |
| `weighted_along_lines_normalised_levenshtein` | NLS on aligned Hough lines |
| `correct_ref_coverage` | Coverage (0–1) |
| `hallucination` | Hallucination fraction (0–1) |
| `selection_harmonic_score` | The harmonic score used to select this alpha |
| `harmonic_mode` | Which formula was used (`balanced`, etc.) |
| `score_floor_alpha` | The alpha chosen by the sweep, or 0.0 in fixed-mask mode |
| `pre_hough_mask_kind` | `score_mean_plus_alpha_standard_deviation` or `minimum_levenshtein` |
| `used_line_count` | Number of lines used in the final alignment |
| `score_mean_ref_to_pred` | Mean of finite cells in the ref-to-pred matrix (NaN in fixed-mask mode) |

---

## 8. Running at scale with Slurm

`run_tunner_atomic.sh` distributes the document pool across multiple Slurm jobs. Each worker picks documents from a shared queue, processes them, and writes results atomically. A final aggregation job merges all worker outputs after all workers finish.

### Example Slurm submission

```bash
bash /scratch/project_2017385/dorian/Churro_copy/tuner_simple_alpha_sweep_pre_iou_levenshtein/run_tunner_atomic.sh \
  --worker-count 20 \
  --account project_2017385 \
  --partition medium \
  --time 02:00:00 \
  --cpus-per-task 4 \
  --mem 24G \
  --runfile-json   /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json \
  --output-dir     /scratch/project_2017385/dorian/Churro_copy/results/tuner_simple_alpha_sweep_pre_iou_levenshtein_all_languages_minIoU_0_10_line_length_3_hough_thresh_3_alpha_0_6_to_4_step_0_1_line_levenshtein_0_5 \
  --scores-pkl-ref-to-pred /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/old_scores_reference_prediction_ws50_st35_levenshtein.pkl \
  --scores-pkl-ref-to-ref  /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/old_scores_reference_self_ws50_st35_levenshtein.pkl \
  --all-languages \
  --alpha-sweep \
  --alpha-sweep-min 0.6 \
  --alpha-sweep-max 4.0 \
  --alpha-sweep-step 0.1 \
  --hough-threshold 3 \
  --hough-line-length 3 \
  --hough-line-gap 15 \
  --align-min-iou-threshold 0.10 \
  --min-surviving-line-nls 0.5 \
  --harmonic-mode balanced \
  --plot-mode stitched-language
```

### Launcher-specific options

| Option | Default | Meaning |
|---|---|---|
| `--worker-count` | 1 | Number of Slurm worker jobs to submit |
| `--account` | `project_2017385` | Slurm account |
| `--partition` | `medium` | Slurm partition for worker jobs |
| `--time` | `01:30:00` | Wall-clock time limit per worker |
| `--cpus-per-task` | 4 | CPUs allocated per worker job |
| `--mem` | `24G` | Memory per worker job |
| `--aggregate-partition` | `medium` | Partition for the final aggregation job |
| `--aggregate-time` | `01:30:00` | Time limit for the aggregation job |
| `--slurm-log-dir` | `<output-dir>/slurm_logs/` | Where worker and aggregation logs go |
| `--resume` | flag | Reuse an existing document pool instead of initialising a fresh one |
| `--requeue-claimed` | flag | During resume, move stale claimed documents back to available |
| `--retry-failed` | flag | During resume, move previously failed documents back for retry |
| `--skip-final-aggregation` | flag | Submit workers only; do not submit the aggregation job |

All tuner options (`--alpha-sweep`, `--harmonic-mode`, etc.) are forwarded verbatim to every worker.

### What the document pool is

The document pool is a directory (`<output-dir>/dynamic_document_pool/`) containing one small JSON file per document. Each file records the document's status: `available`, `claimed` (by a worker), `completed`, or `failed`. Workers atomically claim documents, process them, and mark them complete. This design allows any number of workers to process documents in parallel without coordination overhead.

---

## 9. How each script and module works

### `run_tunner.sh` — single-node wrapper

Intended for running on one machine or as a Slurm single-task job. It:

1. Builds the Cython accelerators if not already built.
2. Applies environment-variable defaults (e.g. `OUTPUT_DIR`, `HARMONIC_MODE`) for any options not already on the command line.
3. Requires `--output-dir` (either as CLI argument or via `OUTPUT_DIR`).
4. Calls `python3 run_tuner_simple.py` with the assembled argument list.

### `run_tunner_atomic.sh` — Slurm distributed launcher

Intended for large runs spanning many documents across multiple Slurm jobs. It:

1. Builds the Cython accelerators once before submitting.
2. Calls `dynamic_pool/initialize_document_pool.py` to set up (or resume) the document queue.
3. Submits `--worker-count` worker jobs via `run_tunner_atomic_worker.sbatch`. Each worker runs the same `run_tuner_simple.py` with `--dynamic-document-pool-dir` pointing at the shared queue.
4. Submits a final aggregation job via `run_tunner_atomic_aggregate.sbatch` that runs after all workers succeed. This job merges progress CSVs, writes final CSVs, and stitches language plots.

### `run_tuner_simple.py` — Python entry point

Parses CLI arguments, builds a `PipelineConfig`, and calls `pipeline_runner.py`. This is what `run_tunner.sh` calls directly.

### `config/pipeline_config.py` — configuration dataclass

A frozen dataclass holding every user-controlled setting. It is validated with `config.validate()` after construction. Adding a new CLI option always involves adding a field here.

### `config/cli_arguments.py` — argument parsing

Defines the argparse parser and maps parsed values into a `PipelineConfig`. The `--harmonic-mode` option automatically appends its value as a subdirectory to `--output-dir` here, before the config object is built.

### `document_selection/` — runfile loading and filtering

`runfile_loader.py` reads `outputs.json` and returns a list of `RunfileDocument` objects. Filters by `--language`, `--document-type`, and `--target-fname` are applied here. `--max-items` limits the total count after filtering.

### `matrix_operations/matrix_loader.py` — score matrix loading

For each selected document, this module tries to load pre-computed score matrices from the pickle files. If a document is missing from a pickle it calls `matrix_fallback_computation.py` to build the matrix from raw text using a sliding window Levenshtein computation. The result (from whichever source) is a standard NumPy float64 2-D array.

### `matrix_operations/score_floor.py` — pre-Hough mask construction

This module converts the float score matrix into a binary mask in two ways:

- `compute_score_floor_statistics()`: computes the mean and standard deviation of all finite cells. These are used by `compute_score_floor_mask_from_statistics()` to build the `mean + alpha × std` threshold mask. Only called during alpha sweep mode — skipped in fixed-mask mode.
- `compute_minimum_levenshtein_mask()`: builds the mask directly from a fixed threshold. Called when `--minimum-pre-hough-levenshtein` is set. Uses `infer_score_matrix_scale()` to normalise the threshold to the matrix's actual scale (unit or percent).
- `build_boolean_threshold_mask()`: the final step that turns a numeric threshold into a boolean NumPy array. Uses a compiled Cython function when available.

### `probabilistic_hough/hough_detection.py` — Hough line detection and filtering

Calls `skimage.transform.probabilistic_hough_line` on the binary mask, then filters the results:
1. Keeps only **falling** diagonal lines (lines where the reference axis increases as the prediction axis increases — the expected direction of correct alignment).
2. Runs the pre-IoU Levenshtein filter (`raw_hough_line_text_filter.py`) on the surviving raw segments.
3. Runs the true IoU merge: segments are merged when their IoU overlap exceeds `--align-min-iou-threshold`.
4. Returns a `HoughFilteredPayload` containing the raw lines, candidate lines, used lines, and timing breakdown.

### `scoring/raw_hough_line_text_filter.py` — pre-IoU text filter

For each raw falling Hough segment, this module computes the text similarity between the prediction windows the segment crosses and the reference windows it maps to. Segments below `--min-surviving-line-nls` are removed before IoU merging. This is the hot path with the Cython fast-path: the Cython `sample_line_path` and `unique_reference_rows_from_path_slice` functions bypass the full `build_line_coverage` → `build_single_raw_line_assignment` → `compute_line_text_record` chain, giving roughly 2.4× faster execution.

### `scoring/scoring_pipeline.py` — final document metrics

After Hough and filtering, this module computes all six public metrics from the surviving lines, the text windows, and the ref-to-ref Hough result. The reference-to-reference signal is subtracted from the reference-to-prediction signal to remove self-similarity noise.

### `serial_runner/document_runner.py` — per-document orchestration

This is the central processing module. For each document it:

1. Loads the score matrices.
2. In alpha-sweep mode: computes score statistics (mean, std) once, then loops over every alpha candidate. In fixed-mask mode: skips statistics and runs one candidate directly.
3. For each alpha candidate: builds the pre-Hough mask, runs Hough detection, runs the pre-IoU Levenshtein filter, runs scoring, and computes the harmonic selection score using the configured `--harmonic-mode` formula.
4. Selects the candidate with the highest harmonic score (alpha-sweep mode) or uses the single fixed-mask result directly.
5. In alpha-sweep mode: writes a per-document audit pickle containing all candidate results.
6. Returns a `DocumentRunResult` with the final result row, plot payload, and pickle path.

### `serial_runner/pipeline_runner.py` — serial run loop

Iterates over selected documents, calls `process_one_document()` for each, and writes rows to CSV files in configurable batches. Also handles the dynamic worker protocol when `--dynamic-document-pool-dir` is set.

### `dynamic_pool/` — distributed document queue

`initialize_document_pool.py` creates the document pool directory and one status file per document. `document_pool.py` provides atomic claim/complete/fail operations used by workers.

### `cython_accel/` — compiled extensions

Contains `.pyx` source files and a `build.py` script. The compiled functions accelerate:
- `filter_core.pyx`: line path sampling, set IoU, coverage index building, final ownership assignment, and unique reference row deduplication.
- `threshold_mask_core.pyx`: binary mask construction (at-or-above threshold).

All Cython functions have pure-Python fallbacks in `optional_filtering.py` and `optional_threshold_mask.py`. The compiled versions are loaded at import time; if they are absent the Python equivalents run transparently.

### `results_writing/` — CSV and JSON output

Handles writing the final CSV files and `run_summary.json`. Progress CSVs (`document_completion_attempts.csv`) are written incrementally during the run so you can inspect partial results while the pipeline is still running.

### `plotting/` — stitched language plots

Imported lazily — only when `--plot-mode` is not `none`. Produces per-document 2×3 panel images (reference text / prediction text / reference score matrix with lines / prediction score matrix with lines / reference alignment / prediction alignment), then stitches all panels for one language into a single overview image.

---

## 10. Full CLI reference

The table below covers every option accepted by both `run_tunner.sh` and the Python entry point. Options marked **required** must be present.

### Input / output

| Option | Env var | Default | Required | Description |
|---|---|---|---|---|
| `--runfile-json` | `RUNFILE_JSON` | (hardcoded dev path) | Yes | Path to the `outputs.json` document list |
| `--output-dir` | `OUTPUT_DIR` | — | **Yes** | Parent directory. Actual results go to `<output-dir>/<harmonic-mode>/` |
| `--scores-pkl-ref-to-pred` | `SCORES_PKL_REF_TO_PRED` | (hardcoded dev path) | Yes | Ref-to-pred score matrix pickle |
| `--scores-pkl-ref-to-ref` | `SCORES_PKL_REF_TO_REF` | (hardcoded dev path) | Yes | Ref-to-ref score matrix pickle |

### Document selection

| Option | Env var | Default | Description |
|---|---|---|---|
| `--language <value>` | — | all | Main language to process; repeat for multiple |
| `--all-languages` | — | — | Select every language (default behaviour when no `--language` is given) |
| `--document-type <value>` | — | all | Document type to process; repeat for multiple |
| `--all-document-types` | — | — | Select every document type (default) |
| `--target-fname <value>` | — | all | Exact filename to process; repeat for multiple |
| `--max-items <n>` | `MAX_ITEMS` | unlimited | Maximum number of documents to process |

### Matrix and text windows

| Option | Env var | Default | Description |
|---|---|---|---|
| `--window-size <n>` | `WINDOW_SIZE` | 50 | Characters per score-matrix text window |
| `--window-stride <n>` | `WINDOW_STRIDE` | 35 | Character offset between neighbouring windows |
| `--minimum-matrix-rows <n>` | `MINIMUM_MATRIX_ROWS` | 4 | Skip documents with fewer reference windows than this |
| `--minimum-matrix-columns <n>` | `MINIMUM_MATRIX_COLUMNS` | 4 | Skip documents with fewer prediction windows than this |

### Alpha sweep and pre-Hough mask

| Option | Env var | Default | Description |
|---|---|---|---|
| `--alpha-sweep` | — | on | Enable per-document alpha sweep (default) |
| `--no-alpha-sweep` | — | — | Disable sweep; use `--score-floor-alpha` exactly |
| `--score-floor-alpha <v>` | `SCORE_FLOOR_ALPHA` | 1.0 | Alpha used when `--no-alpha-sweep` is set |
| `--alpha-sweep-min <v>` | `ALPHA_SWEEP_MIN` | 1.0 | Inclusive minimum alpha |
| `--alpha-sweep-max <v>` | `ALPHA_SWEEP_MAX` | 4.0 | Inclusive maximum alpha |
| `--alpha-sweep-step <v>` | `ALPHA_SWEEP_STEP` | 0.2 | Step between alpha candidates |
| `--minimum-pre-hough-levenshtein <v>` | `MINIMUM_PRE_HOUGH_LEVENSHTEIN` | off | Fixed Levenshtein threshold mask; disables alpha sweep |

### Harmonic mode

| Option | Env var | Default | Choices |
|---|---|---|---|
| `--harmonic-mode <value>` | `HARMONIC_MODE` | `balanced` | `balanced`, `coverage-hallucination-priority`, `coverage-hallucination-only` |

### Hough detection

| Option | Env var | Default | Description |
|---|---|---|---|
| `--hough-threshold <n>` | `HOUGH_THRESHOLD` | 25 | Minimum vote count per Hough line |
| `--hough-line-length <n>` | `HOUGH_LINE_LENGTH` | 35 | Minimum line length (cells) |
| `--hough-line-gap <n>` | `HOUGH_LINE_GAP` | 15 | Maximum gap inside a Hough line |
| `--hough-seed <n>` | `HOUGH_SEED` | 1 | Reproducibility seed for probabilistic Hough |
| `--align-min-iou-threshold <v>` | `ALIGN_MIN_IOU_THRESHOLD` | 0.035 | Minimum IoU for a line to claim a cell |
| `--min-surviving-line-nls <v>` | `MIN_SURVIVING_LINE_NLS` | 0.5 | Minimum pre-IoU line text similarity; `0` disables |

### Plotting

| Option | Env var | Default | Description |
|---|---|---|---|
| `--plot-mode <value>` | `PLOT_MODE` | `stitched-language` | `none`, `stitched-language`, or `stitched-language-and-document-grids` |
| `--show-line-ids` | — | off | Label raw and surviving lines on plot overlays |
| `--stitched-panel-columns <n>` | `STITCHED_PANEL_COLUMNS` | 3 | Document panels per row in stitched images |
| `--saved-figure-dpi <n>` | `SAVED_FIGURE_DPI` | 140 | PNG resolution |

### Worker / atomic mode (advanced)

These are set automatically by `run_tunner_atomic.sh` and do not need to be passed manually:

| Option | Description |
|---|---|
| `--dynamic-document-pool-dir <path>` | Activates dynamic worker mode; workers claim from this pool |
| `--dynamic-worker-id <value>` | Written into progress CSV rows and pool status files |
| `--atomic-output-dir <path>` | Where workers write progress CSV files |
| `--result-bucket-size <n>` | Rows flushed per CSV append (default 20) |
| `--result-bucket-seconds <v>` | Also flush after this many seconds (default 60) |
