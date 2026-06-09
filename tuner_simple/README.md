# tuner_simple

`tuner_simple` is a serial Hough-alignment pipeline for running one fixed parameter set over selected documents. It keeps the run structure deliberately small: load score matrices, build one binary Hough input, run probabilistic Hough once per direction, filter the detected lines, compute the final document metrics, and write flat output files.

## Main Command

```bash
python3 /scratch/project_2017385/dorian/Churro_copy/tuner_simple/run_tuner_simple.py \
  --runfile-json /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json \
  --output-dir /scratch/project_2017385/dorian/Churro_copy/results/tuner_simple_run \
  --scores-pkl-ref-to-pred /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/old_scores_reference_prediction_ws50_st35_levenshtein.pkl \
  --scores-pkl-ref-to-ref /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/old_scores_reference_self_ws50_st35_levenshtein.pkl \
  --window-size 50 \
  --window-stride 35 \
  --minimum-matrix-rows 4 \
  --minimum-matrix-columns 4 \
  --score-floor-alpha 1.0 \
  --hough-threshold 25 \
  --hough-line-length 35 \
  --hough-line-gap 15 \
  --hough-seed 1 \
  --align-min-iou-threshold 0.035 \
  --min-surviving-line-nls 0.5 \
  --plot-mode stitched-language
```

## Shell Command

```bash
bash /scratch/project_2017385/dorian/Churro_copy/tuner_simple/run_tunner.sh \
  --runfile-json /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json \
  --output-dir /scratch/project_2017385/dorian/Churro_copy/results/tuner_simple_run \
  --scores-pkl-ref-to-pred /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/old_scores_reference_prediction_ws50_st35_levenshtein.pkl \
  --scores-pkl-ref-to-ref /scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/old_scores_reference_self_ws50_st35_levenshtein.pkl \
  --window-size 50 \
  --window-stride 35 \
  --minimum-matrix-rows 4 \
  --minimum-matrix-columns 4 \
  --score-floor-alpha 1.0 \
  --hough-threshold 25 \
  --hough-line-length 35 \
  --hough-line-gap 15 \
  --hough-seed 1 \
  --align-min-iou-threshold 0.035 \
  --min-surviving-line-nls 0.5 \
  --plot-mode stitched-language
```

## Pipeline

1. `document_selection/` reads the runfile JSON and applies optional document filters.
2. `matrix_operations/` loads score matrices from the `.pkl` files. If a selected document is missing from the `.pkl`, it computes a Levenshtein score matrix from text windows.
3. `matrix_operations/score_floor.py` computes `score_floor = score_mean + score_floor_alpha * score_standard_deviation` for each matrix.
4. Cells at or above the score floor become the binary Hough input.
5. `probabilistic_hough/` runs `skimage.transform.probabilistic_hough_line` with one fixed parameter set.
6. `scoring/line_text_similarity.py` removes surviving lines whose line-level normalised Levenshtein similarity is below the configured minimum.
7. `scoring/coverage_count_metrics.py` computes the same coverage-count subtraction used by `tuner_parallel_v2_2`: reference-to-prediction character counts minus reference-to-reference character counts on the reference axis, plus prediction-axis zero counts for hallucination.
8. `scoring/scoring_pipeline.py` exposes only the final public metrics: document normalised Levenshtein, weighted along-lines normalised Levenshtein, correct reference coverage, missing reference coverage, repetition on reference, and hallucination.
9. `results_writing/` writes flat CSV and JSON files directly under the selected output directory.
9. `plotting/` is imported only when plotting is enabled.

## Output Files

The run writes these files directly under `--output-dir`:

- `best_combination_per_document.csv`
- `compact_combination_metrics.csv`
- `document_type_summary.csv`
- `loadable_documents.csv`
- `loaded_documents.csv`
- `runfile_documents.csv`
- `skipped_documents.csv`
- `run_summary.json`

When plotting is enabled, stitched language plots are written under `--output-dir/plots/`.

## Slurm Use

Submit the shell runner from your own Slurm script when cluster scheduling is needed. The runner itself intentionally contains no `#SBATCH` directives.

## Plot Modes

- `--plot-mode none`: do not import plotting libraries and do not create plots.
- `--plot-mode stitched-language`: create stitched language plots and remove temporary document panels.
- `--plot-mode stitched-language-and-document-grids`: create stitched language plots and keep each individual 2x3 document panel.

Line identifiers are hidden by default. Add `--show-line-ids` when line labels are needed on raw and surviving line overlays.
