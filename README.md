# HTR-context-OCR

This README documents how to run **every** script in:

- `python_scripts/`
- `shell_scripts/`

Important note:
- **Run Python scripts with `python3`** (not `python`).
- This repo targets **Python ≥ 3.12** (see `pixi.toml` + `pyproject.toml`). Many scripts will not run on older Python.

---

## Conventions

Run commands from the repo root (recommended, because many default paths are relative):

```bash
cd /scratch/project_2017385/dorian/HTR-context-OCR
```

Check your Python:

```bash
python3 --version
```

If your checkout path differs from `/scratch/project_2017385/...`:
- Some scripts contain **hard-coded absolute `/scratch/...` paths**.
- Update those constants, or search for them:

```bash
rg -n "/scratch/" python_scripts shell_scripts
```

---

## Common File Types Used Across Scripts

You’ll see these files referenced repeatedly:

### `outputs.json` (inference runfile)
A JSON list of per-image inference results (typically produced by a Churro evaluation pipeline). Many scripts expect fields like:
- `file_name`
- `normalized_gold_text`
- `normalized_predicted_text`

### `scores.pkl` (pickle stream; NOT one big list)
Produced by `python_scripts/compare.py`. It is a **pickle stream** (many sequential `pickle.dump(...)` calls), so downstream scripts read it using repeated `pickle.load(...)` until EOF.

Each record typically contains:
- `fname` (basename of the image file)
- `scores` (chrF matrix)
- `ref` (gold text)
- `pred` (predicted text)

### `endpoints.pkl` (pickle stream of Hough endpoints)
Produced by `python_scripts/hough_line_transform_endpoints_no_angle_all.py`. Stores per-item merged line segments so alignment can reuse them.

### `*_adjusted_pred.txt` (aligned predictions)
Produced by `python_scripts/align_graph_text_blocks_all.py` (and some smaller target-only aligners). These are used by `python_scripts/compare_aligned_texts.py`.

---

## Pipeline Overview (Recommended “All-in-one” Alignment Pipeline)

If you have:
- an image directory (`--img-dir`)
- an inference runfile `outputs.json` (`--runfile-json`)

…the shell wrapper `shell_scripts/run_evaluation_improvement.sh` runs this full pipeline:

1. `compare.py` → creates `scores.pkl`
2. `visualise_dorian_dense_matrices_style_no_angle_all.py` → dense-matrices visualisations
3. `hough_line_transform_endpoints_no_angle_all.py` → creates `endpoints.pkl`
4. `align_graph_text_blocks_all.py` → writes aligned `*_adjusted_pred.txt` + reports
5. `compare_aligned_texts.py` → creates aligned-score `scores_aligned.pkl`
6. `visualise_scores_heatmap_only.py` → heatmaps for aligned score matrices

See the `shell_scripts/run_evaluation_improvement.sh` section below for details.

---

# Python Scripts (`python_scripts/`)

> All commands in this section are run using `python3`.

---

## `python_scripts/align_graph_text_blocks.py`

Purpose:
- Align prediction text blocks for a **fixed list of target cases** using line endpoints detected from chrF score matrices.

How to run:
```bash
python3 python_scripts/align_graph_text_blocks.py
```

Configuration (NO CLI flags):
- This script is configured via **in-file constants**, including:
  - `PROJECT_ROOT` (currently hard-coded)
  - `SCORES_PKL` (input score matrix stream)
  - `IMG_DIR` (directory containing images; used for some visual checks)
  - `TARGET_GRAPHS` (which cases to process)
  - `WINDOW_SIZE`, `WINDOW_STRIDE` (must match what was used to generate `scores.pkl`)
  - `OUTPUT_DIR` (where outputs are written)
- Optional environment:
  - `ALLOW_MISSING=1` to skip missing targets rather than raising.

Inputs:
- `scores.pkl` produced by `python_scripts/compare.py`

Outputs (under `OUTPUT_DIR`):
- `<target>_adjusted_pred.txt`
- `<target>_alignment_report.json`
- `summary.json`

Notes / gotchas:
- This script contains absolute paths; it’s best treated as an experiment script.
- For batch processing across **all** cases using CLI flags, prefer `align_graph_text_blocks_all.py`.

---

## `python_scripts/align_graph_text_blocks_all.py`

Purpose:
- Align prediction text blocks for **all** cases in a `scores.pkl` stream, using precomputed Hough endpoints from an `endpoints.pkl` stream.

How to run (example):
```bash
python3 python_scripts/align_graph_text_blocks_all.py \
  --scores-pkl /path/to/scores.pkl \
  --endpoints-pkl /path/to/endpoints.pkl \
  --output-dir /path/to/aligned_outputs \
  --window-size 100 \
  --window-stride 50
```

Flags (all CLI flags):
- `--scores-pkl` (required): Path to compare output `scores.pkl` (pickle stream).
- `--endpoints-pkl` (required): Path to endpoint records (pickle stream) from `hough_line_transform_endpoints_no_angle_all.py`.
- `--output-dir` (required): Directory for writing aligned text + JSON reports.
- `--window-size` (default: `100`): Must match the compare window size used upstream (kept for metadata).
- `--window-stride` (default: `50`): **Used** for overlap-aware reassembly; must match compare stride.
- `--max-items` (default: unset): Stop after N items.
- `--visualise-full-dir` (optional): If provided, stored in reports as a cross-reference path.
- `--visualise-graph-dir` (optional): Same idea.
- `--visualise-mask-dir` (optional): Same idea.

Inputs:
- `scores.pkl` (from `compare.py`)
- `endpoints.pkl` (from `hough_line_transform_endpoints_no_angle_all.py`)

Outputs (under `--output-dir`):
- For each index `i` (0-based), using a stable prefix:
  - `0000_<safe_name>_adjusted_pred.txt`
  - `0000_<safe_name>_alignment_report.json`
- Summary:
  - `summary.json`

Notes / gotchas:
- If the endpoint stream does not contain a record for every `fname` in `scores.pkl`, this script errors.
- The script expects endpoint records keyed by the exact `fname` stored in the `scores.pkl` records.

---

## `python_scripts/align_two_graph_text_blocks.py`

Purpose:
- Align prediction text blocks for a **small list of target “graph” cases** using merged lines detected from chrF matrices.

How to run:
```bash
python3 python_scripts/align_two_graph_text_blocks.py
```

Configuration (NO CLI flags):
- Configure via in-file constants:
  - `PROJECT_ROOT`
  - `SCORES_PKL`
  - `TARGET_GRAPHS`
  - `WINDOW_SIZE`, `WINDOW_STRIDE`
  - `OUTPUT_DIR`

Inputs:
- `scores.pkl` produced by `python_scripts/compare.py`

Outputs:
- Under `results/aligned_text_blocks_two_cases/` (relative to `PROJECT_ROOT`):
  - `<target>_adjusted_pred.txt`
  - `<target>_alignment_report.json`
  - `summary.json`

---

## `python_scripts/churchbook_churro_infer.py`

Purpose:
- Run OCR on a **local folder of churchbook images** and write markdown outputs (XML + extracted plain text).

How to run (vLLM backend example):
```bash
python3 python_scripts/churchbook_churro_infer.py \
  --backend vllm \
  --input-dir /scratch/project_2017385/dorian/Churro_churchbooks/churchbook_images \
  --output-root /scratch/project_2017385/dorian/Churro_churchbooks/results/churchbook_results \
  --model-id stanford-oval/churro-3B \
  --system-message "Transcribe the entiretly of this historical documents to XML format." \
  --max-new-tokens 20000 \
  --temperature 0.6 \
  --device auto \
  --vllm-base-url http://localhost:8000/v1 \
  --vllm-model churro \
  --vllm-api-key "${OPENAI_API_KEY:-EMPTY}" \
  --vllm-timeout-seconds 600 \
  --max-concurrency 1 \
  --max-images 0 \
  --skip-existing
```

Flags (high level):
- `--backend`: `transformers` or `vllm`
- `--input-dir`: image directory
- `--output-root`: output root directory
- `--model-id`, `--system-message`, `--max-new-tokens`, `--temperature`, `--device`
- vLLM options: `--vllm-base-url`, `--vllm-model`, `--vllm-api-key`, `--vllm-timeout-seconds`, `--max-concurrency`
- controls: `--max-images`, `--skip-existing`

Outputs (under `--output-root`):
- `xml_results/<image_stem>_xml.md`
- `_pure_text_results/<image_stem>_pure_text.md`

---

## `python_scripts/compare.py`

Purpose:
- Sliding-window chrF comparison between gold and predicted text from an inference runfile (`outputs.json`).
- Produces a **pickle stream** `scores.pkl` used by visualisers and aligners.

How to run:
```bash
python3 python_scripts/compare.py \
  --window-size 100 \
  --window-stride 50 \
  --runfile-json /path/to/outputs.json \
  --output /path/to/scores.pkl
```

Flags (all CLI flags):
- `--window-size` (default: `100`)
- `--window-stride` (default: `50`)
- `--runfile-json` (default is a relative path; run from repo root if relying on defaults)
- `--output` (default: `scores.pkl`)
- `--max-items` (optional): process only first N entries

Outputs:
- `scores.pkl` (pickle stream) containing records like:
  - `{fname, scores, ref, pred}`

Notes:
- Downstream scripts expect `fname` to be the image basename (compare uses `os.path.basename(file_name)`).

---

## `python_scripts/compare_aligned_texts.py`

Purpose:
- Recompute chrF matrices by comparing the **gold text** in `outputs.json` to the **aligned predictions** stored in `*_adjusted_pred.txt` files.

How to run:
```bash
python3 python_scripts/compare_aligned_texts.py \
  --runfile-json /path/to/outputs.json \
  --aligned-dir /path/to/alignment_outputs \
  --txt-glob "*_adjusted_pred.txt" \
  --output /path/to/scores_aligned.pkl \
  --window-size 100 \
  --window-stride 50
```

Flags (all CLI flags):
- `--window-size` (default: `100`)
- `--window-stride` (default: `50`)
- `--runfile-json` (default relative path)
- `--aligned-dir` (default is an absolute path; override it)
- `--txt-glob` (default: `*_adjusted_pred.txt`)
- `--output` (default: `aligned_scores.pkl`)
- `--allow-missing` (optional): skip aligned txts that don’t match a runfile entry

Inputs:
- `outputs.json` runfile
- aligned text files produced by `align_graph_text_blocks_all.py` or similar

Outputs:
- `aligned_scores.pkl` (pickle stream) with records including:
  - `fname` (original image name)
  - `aligned_txt` (path)
  - `scores`, `ref`, `pred`

Notes / gotchas:
- This script maps aligned-txt names back to runfile entries using a safe-name key; if naming differs, use `--allow-missing` and inspect warnings.

---

## `python_scripts/custom_python_script.py`

Purpose:
- A custom entrypoint that delegates to `churro.cli.main.main`.
- Designed to be used like:
  - `python3 -m custom_python_script benchmark ...`

How to run (module-style):
```bash
python3 -m custom_python_script --help
```

Benchmark example:
```bash
python3 -m custom_python_script benchmark \
  --system finetuned \
  --engine churro \
  --dataset-split test \
  --input-size 0 \
  --max-concurrency 4
```

Notes:
- This module is typically invoked by `shell_scripts/run_finetuned_benchmark_existing_vllm.sh`.
- For `python3 -m custom_python_script ...` to work, your environment must be able to import `custom_python_script` as a module.
  (In many setups this is handled externally by environment configuration or wrapper scripts.)

---

## `python_scripts/download_churro_finnish.py`

Purpose:
- Stream `stanford-oval/churro-dataset` and export Finnish samples to disk.

How to run:
```bash
python3 python_scripts/download_churro_finnish.py
```

Configuration (NO CLI flags):
- Edit in-file constants if needed:
  - `DATASET_ID`
  - `SPLITS`
  - `LANGUAGE_FILTER`
  - `OUTPUT_ROOT`

Outputs (under `OUTPUT_ROOT/<split>/`):
- saved images (filename derived from dataset `file_name`)
- `<image_stem>_fields.md` (metadata dump)
- `manifest.jsonl` (one record per saved sample)
- `OUTPUT_ROOT/summary.json`

---

## `python_scripts/finnish_custom_churro_infer.py`

Purpose:
- Run inference on the HF Churro dataset, filter to Finnish, and write:
  - markdown outputs per sample
  - benchmark-style metrics outputs (`outputs.json`, `all_metrics.json`)

How to run (vLLM example):
```bash
python3 python_scripts/finnish_custom_churro_infer.py \
  --backend vllm \
  --dataset-id stanford-oval/churro-dataset \
  --dataset-split all \
  --max-samples-per-split 0 \
  --model-id stanford-oval/churro-3B \
  --system-message "Transcribe the entiretly of this historical documents to XML format." \
  --max-new-tokens 20000 \
  --temperature 0.6 \
  --device auto \
  --vllm-base-url http://localhost:8000/v1 \
  --vllm-model churro \
  --vllm-api-key "${OPENAI_API_KEY:-EMPTY}" \
  --vllm-timeout-seconds 600 \
  --max-concurrency 1 \
  --output-dir /scratch/project_2017385/dorian/HTR-context-OCR/responses \
  --metrics-output-root /scratch/project_2017385/dorian/HTR-context-OCR/responses \
  --skip-existing
```

Flags (summary):
- backend + model:
  - `--backend` (`transformers` or `vllm`)
  - `--model-id` (transformers backend)
- dataset:
  - `--dataset-id`
  - `--dataset-split` (split name or `all`)
  - `--max-samples-per-split`
- generation:
  - `--system-message`
  - `--max-new-tokens`
  - `--temperature`
  - `--device`
- vLLM:
  - `--vllm-base-url`, `--vllm-model`, `--vllm-api-key`, `--vllm-timeout-seconds`
  - `--max-concurrency`
- outputs:
  - `--output-dir`
  - `--metrics-output-root`
  - `--skip-existing`

Outputs:
- Markdown:
  - `<output_dir>/<split>/model_results/<index>_<name>.md`
  - `<output_dir>/<split>/gold/gold_<name>.md`
- Metrics:
  - `<metrics_output_root>/<backend>/<split>/outputs.json`
  - `<metrics_output_root>/<backend>/<split>/all_metrics.json`

Tip:
- For a fully wrapped vLLM + repeat-run workflow, see `shell_scripts/run_finnish_custom_churro_infer_existing_vllm.sh`.

---

## `python_scripts/hough_line_transform_endpoints_no_angle_all.py`

Purpose:
- Read `scores.pkl` (pickle stream), run dense-matrices style detection, and write ONLY merged line endpoints (pickle stream).
- Intended to be used before `align_graph_text_blocks_all.py`.

How to run:
```bash
python3 python_scripts/hough_line_transform_endpoints_no_angle_all.py \
  --scores-pkl /path/to/scores.pkl \
  --output /path/to/endpoints.pkl
```

Flags (all CLI flags):
- `--scores-pkl` (required): input `scores.pkl` pickle stream
- `--output` (required): output endpoint pickle stream
- `--max-items` (optional): limit processed items

Outputs:
- `endpoints.pkl` (pickle stream), each record includes:
  - `fname`
  - `threshold_start`
  - `merged_lines`
  - `raw_line_count`, `selected_line_count`, `merged_line_count`

---

## `python_scripts/plot_custom_churro_run_metrics.py`

Purpose:
- Discover run folders and plot summaries + heatmaps from `all_metrics.json`.

How to run:
```bash
python3 python_scripts/plot_custom_churro_run_metrics.py
```

Configuration (NO CLI flags):
- Edit these constants near the bottom of the file:
  - `SEARCH_ROOTS` (defaults to `[Path.cwd()]`)
  - `SAVE_FIGURES` (`False` by default)
  - `FIGURE_OUTPUT_DIR` (default: `results/plots/custom_churro_runs`)

Outputs:
- Prints run summary table to stdout.
- If `SAVE_FIGURES=True`, writes PNGs into `FIGURE_OUTPUT_DIR`.

---

## `python_scripts/single_finetuned_vllm_infer.py`

Purpose:
- Run a single image request against a local vLLM OpenAI-compatible endpoint and write a markdown report.

How to run (example):
```bash
python3 python_scripts/single_finetuned_vllm_infer.py \
  --image /scratch/project_2017385/dorian/Churro_copy/tests/churro_dataset_sample_1.jpeg \
  --engine churro \
  --system-message "..." \
  --timeout-seconds 600 \
  --local-vllm-port 8000 \
  --openai-api-key "${OPENAI_API_KEY:-EMPTY}" \
  --output-dir /scratch/project_2017385/dorian/HTR-context-OCR/responses \
  --output-file /scratch/project_2017385/dorian/HTR-context-OCR/responses/single_finetuned_vllm.md \
  --strip-xml
```

Flags (summary):
- input: `--image`
- model selection: `--engine`
- request: `--system-message`, `--timeout-seconds`, `--local-vllm-port`, `--openai-api-key`
- outputs: `--output-dir`, `--output-file`
- processing: `--strip-xml`

Output:
- Markdown at `--output-file` (or under `--output-dir` if `--output-file` omitted)

---

## `python_scripts/visualise_dorian_component_fit_no_hough_churro30.py`

Purpose:
- Visualize chrF matrices using a “component fit” approach (no Hough transform).

How to run:
```bash
VIZ_NOTEBOOK_OUTPUT=1 python3 python_scripts/visualise_dorian_component_fit_no_hough_churro30.py
```

Configuration (NO CLI flags):
- Controlled via in-file constants (paths + output dirs).
- Notebook/text-pane output is controlled via:
  - `VIZ_NOTEBOOK_OUTPUT=1` (default)
  - `VIZ_NOTEBOOK_OUTPUT=0` for headless runs

Outputs:
- Writes figures under its configured `RESULTS_DIR` (see constants in file)

---

## `python_scripts/visualise_dorian_component_fit_no_hough_churro30_v2.py`

Purpose:
- Same as v1, with improved handling for small matrices.

How to run:
```bash
VIZ_NOTEBOOK_OUTPUT=1 python3 python_scripts/visualise_dorian_component_fit_no_hough_churro30_v2.py
```

Configuration:
- Same pattern as v1 (`VIZ_NOTEBOOK_OUTPUT` + in-file constants)

Outputs:
- Writes figures under its configured `RESULTS_DIR`

---

## `python_scripts/visualise_dorian_dense_matrices_style_no_angle_all.py`

Purpose:
- Visualise chrF score matrices and overlay merged Hough lines (dense-matrices style, no angle filtering) for **all** cases in a `scores.pkl` stream.

How to run (example):
```bash
python3 python_scripts/visualise_dorian_dense_matrices_style_no_angle_all.py \
  --img-dir /scratch/project_2017385/dorian/churro_finnish_dataset/dataset_splits/dev \
  --scores-pkl /path/to/scores.pkl \
  --results-dir /path/to/visualise_outputs \
  --max-items 50
```

Flags (all CLI flags):
- `--img-dir`: directory containing images (looked up by `fname` from `scores.pkl`)
- `--scores-pkl`: input `scores.pkl` (pickle stream)
- `--results-dir`: output directory
- `--max-items`: optional limit
- `--render-notebook-output`: render extra output using IPython (only if available)
- `--show`: call `plt.show()` per item (usually for notebooks)

Outputs (under `--results-dir`):
- `full_figures/`
- `graph_only/`
- `detection_masks/`

---

## `python_scripts/visualise_dorian_dense_matrices_style_single_no_angle.py`

Purpose:
- Dense-matrices style detection + visualisation for **one** target case.

How to run:
```bash
python3 python_scripts/visualise_dorian_dense_matrices_style_single_no_angle.py
```

Configuration (NO CLI flags):
- Edit in-file constants:
  - `IMG_DIR`, `PROJECT_ROOT`, `SCORES_PKL`
  - `RESULTS_DIR` (+ its subdirectories)
  - `TARGET_NAME`
  - `RENDER_NOTEBOOK_OUTPUT`

Outputs:
- Writes PNGs under the configured `RESULTS_DIR`

---

## `python_scripts/visualise_dorian_simple.py`

Purpose:
- Experimental visualiser with extra intermediate masks.

How to run:
```bash
VIZ_NOTEBOOK_OUTPUT=1 VIZ_SAVE_OUTPUTS=1 python3 python_scripts/visualise_dorian_simple.py
```

Configuration:
- Controlled by in-file constants + env vars:
  - `VIZ_NOTEBOOK_OUTPUT=1|0`
  - `VIZ_SAVE_OUTPUTS=1|0`

Outputs:
- When `VIZ_SAVE_OUTPUTS=1`, saves figures under its configured output folder.

---

## `python_scripts/visualise_scores_heatmap_only.py`

Purpose:
- Render score matrices from a `scores.pkl` pickle stream as simple heatmaps (NO line detection).
- Intended for visualising aligned score matrices produced by `compare_aligned_texts.py`, but can also be used on original `scores.pkl`.

How to run:
```bash
python3 python_scripts/visualise_scores_heatmap_only.py \
  --img-dir /scratch/project_2017385/dorian/churro_finnish_dataset/dataset_splits/dev \
  --scores-pkl /path/to/scores_aligned.pkl \
  --results-dir /path/to/heatmap_outputs \
  --max-items 50
```

Flags (all CLI flags):
- `--img-dir` (optional, but recommended)
- `--scores-pkl` (required): pickle stream of `{fname, scores, ...}`
- `--results-dir` (optional): output directory
- `--max-items` (optional): limit processed items
- `--show` (optional): show figures interactively

Outputs (under `--results-dir`):
- `full_figures/`
- `graph_only/`

---

## `python_scripts/visualise_xmls.py`

Purpose:
- Render XML payloads embedded in markdown files into standalone HTML pages.

How to run (example):
```bash
python3 python_scripts/visualise_xmls.py \
  --results-dir /scratch/project_2017385/dorian/Churro_churchbooks/results/churchbook_results/xml_results \
  --output-dir /scratch/project_2017385/dorian/Churro_churchbooks/results/churchbook_results/xml_html_renders \
  --limit 0 \
  --stylesheet /scratch/project_2017385/dorian/HTR-context-OCR/python_scripts/visualise_xmls.css \
  --images-dir /scratch/project_2017385/dorian/Churro_churchbooks/churchbook_images \
  --copy-images-into-output
```

Key flags:
- `--results-dir`: input markdown folder
- `--output-dir`: output HTML folder
- `--limit`: number of files (0 = all)
- `--stylesheet`: CSS file path
- `--images-dir`: optional image directory for linking
- `--copy-images-into-output`: copy images into output folder

Outputs:
- One HTML per markdown file + `index.html` in `--output-dir`

---

## `python_scripts/visualise_xmls.css`

Purpose:
- Stylesheet used by `visualise_xmls.py`
- Not executable

---

## Jupyter notebooks (listed; not fully documented here)

- `python_scripts/custom_churro_run_metrics_analysis.ipynb`
- `python_scripts/visualise_dorian.ipynb`

---

# Shell Scripts (`shell_scripts/`)

Notes:
- These scripts include `#SBATCH ...` directives and can be run via:
  - `sbatch shell_scripts/<script>.sh ...` (recommended on CSC/HPC)
  - `bash shell_scripts/<script>.sh ...` (manual run; `#SBATCH` lines are ignored)
- Some scripts use `module load ...` and assume an HPC module environment.

---

## `shell_scripts/run_compare.sh`

Purpose:
- Wrapper for `python_scripts/compare.py`

How to run:
```bash
bash shell_scripts/run_compare.sh \
  --runfile-json /path/to/outputs.json \
  --output /path/to/scores.pkl \
  --window-size 100 \
  --window-stride 50
```

Flags:
- `--runfile-json`
- `--output`
- `--window-size`
- `--window-stride`
- `-h`, `--help`

Environment overrides:
- `RUNFILE_JSON`, `OUTPUT`, `WINDOW_SIZE`, `WINDOW_STRIDE`

Notes:
- This script calls `module purge/load ...` unconditionally; on systems without `module`, run it on the cluster or remove/guard those lines.

---

## `shell_scripts/run_custom_churro_infer_existing_vllm.sh`

Purpose:
- Start local vLLM, wait for readiness, then run a custom inference script repeatedly.

How to run (example):
```bash
bash shell_scripts/run_custom_churro_infer_existing_vllm.sh \
  --metrics-output-root /scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_run1 \
  --output-dir /scratch/project_2017385/dorian/Churro_copy/responses \
  --max-concurrency 32 \
  --vllm-timeout-seconds 3600 \
  --max-new-tokens 22000 \
  --dataset-split test \
  --max-model-len 125000 \
  --repeat-count 1 \
  --gpu-memory-utilization 0.3
```

Important path note:
- This script currently calls:
  - `python3 /scratch/project_2017385/dorian/HTR-context-OCR/tests/custom_churro_infer.py`
- That file does **not** exist in this repo. To run this script you must update that path to a real `custom_churro_infer.py` entrypoint (for example one in a sibling repository).

Flags:
- `--metrics-output-root`
- `--output-dir`
- `--max-concurrency`
- `--vllm-timeout-seconds`
- `--max-new-tokens`
- `--dataset-split` (`all|dev|test`)
- `--max-model-len`
- `--repeat-count`
- `--gpu-memory-utilization`
- `-h`, `--help`

---

## `shell_scripts/run_evaluation_improvement.sh`

Purpose:
- Run a full evaluation/alignment/visualisation pipeline in one command.

How to run:
```bash
bash shell_scripts/run_evaluation_improvement.sh \
  --img-dir /path/to/images \
  --runfile-json /path/to/outputs.json \
  --project-root-results /path/to/results_root \
  --window-size 100 \
  --window-stride 50 \
  --max-items 100
```

Required flags:
- `--img-dir`: directory containing the document images (filenames must match `fname` from `scores.pkl`)
- `--runfile-json`: path to `outputs.json`

Optional flags:
- `--project-root-results`: root output directory for the entire pipeline
- `--window-size`: chrF compare window
- `--window-stride`: chrF compare stride
- `--max-items`: cap number of items for all stages

Outputs:
- A run directory is created under:
  - `<project-root-results>/window_<WINDOW_SIZE>_stride_<WINDOW_STRIDE>/<timestamp>/`
- Inside that directory:
  - `compare/scores.pkl`
  - `visualise_dorian_dense_matrices_style_no_angle_all/...`
  - `hough_endpoints/endpoints.pkl`
  - `align_graph_text_blocks_all/...`
  - `compare_aligned/scores_aligned.pkl`
  - `aligned_graphs/...`

Notes:
- This script attempts to find the `HTR-context-OCR` repo root automatically so it can run the python scripts regardless of Slurm `--chdir`.

---

## `shell_scripts/run_finetuned_benchmark_existing_vllm.sh`

Purpose:
- Start local vLLM and run the finetuned benchmark via:
  - `python3 -m custom_python_script benchmark ...`

How to run (configured via environment variables):
```bash
ENGINE=churro \
DATASET_SPLIT=test \
INPUT_SIZE=0 \
MAX_CONCURRENCY=4 \
MODEL_REPO=stanford-oval/churro-3B \
MAX_MODEL_LEN=125000 \
WAIT_SECONDS=1200 \
SLEEP_SECONDS=2 \
bash shell_scripts/run_finetuned_benchmark_existing_vllm.sh
```

Environment options:
- `ENGINE`
- `DATASET_SPLIT`
- `INPUT_SIZE`
- `MAX_CONCURRENCY`
- `MODEL_REPO`
- `MAX_MODEL_LEN`
- `WAIT_SECONDS`
- `SLEEP_SECONDS`

Notes:
- Requires `vllm` available in PATH.
- Uses `SLURM_JOB_ID` when available to name log files.

---

## `shell_scripts/run_finnish_custom_churro_infer_existing_vllm.sh`

Purpose:
- Wrapper for `python_scripts/finnish_custom_churro_infer.py` with optional local vLLM startup and repeat runs.

How to run (example):
```bash
bash shell_scripts/run_finnish_custom_churro_infer_existing_vllm.sh \
  --backend vllm \
  --dataset-id stanford-oval/churro-dataset \
  --dataset-split test \
  --max-samples-per-split 0 \
  --model-id stanford-oval/churro-3B \
  --system-message "Transcribe the entiretly of this historical documents to XML format." \
  --max-new-tokens 22000 \
  --temperature 0.6 \
  --device auto \
  --max-concurrency 32 \
  --vllm-base-url http://localhost:8000/v1 \
  --vllm-model churro \
  --vllm-api-key "${OPENAI_API_KEY:-EMPTY}" \
  --vllm-timeout-seconds 3600 \
  --output-dir /scratch/project_2017385/dorian/HTR-context-OCR/responses/finnish_custom_infer_run1 \
  --metrics-output-root /scratch/project_2017385/dorian/HTR-context-OCR/results/finnish_custom_infer_run1 \
  --skip-existing \
  --repeat-count 1 \
  --start-local-vllm 1 \
  --local-vllm-port 8000 \
  --local-vllm-model-name churro \
  --model-repo stanford-oval/churro-3B \
  --max-model-len 125000 \
  --gpu-memory-utilization 0.3 \
  --wait-seconds 1200 \
  --sleep-seconds 2
```

Flags:
- Pass-through to python:
  - `--backend`, `--dataset-id`, `--dataset-split`, `--max-samples-per-split`
  - `--model-id`, `--system-message`, `--max-new-tokens`, `--temperature`, `--device`
  - `--max-concurrency`, `--vllm-base-url`, `--vllm-model`, `--vllm-api-key`, `--vllm-timeout-seconds`
  - `--output-dir`, `--metrics-output-root`, `--skip-existing`
- Wrapper/runtime:
  - `--repeat-count`
  - `--start-local-vllm`, `--local-vllm-port`, `--local-vllm-model-name`
  - `--model-repo`, `--max-model-len`, `--gpu-memory-utilization`
  - `--wait-seconds`, `--sleep-seconds`
  - `-h`, `--help`

---

## `shell_scripts/run_hugh_line_transform.sh`

Purpose:
- Submit/run a Hough-line transform job by calling:
  - `python3 /scratch/project_2017385/dorian/Churro_copy/hugh_line_transform_dev_schuro.py`

How to run:
```bash
bash shell_scripts/run_hugh_line_transform.sh
```

Optional:
- You can pass a results directory as the first argument; it exports it as `RESULTS_DIR` for the Python script:
```bash
bash shell_scripts/run_hugh_line_transform.sh /path/to/results_dir
```

Notes:
- This script is tightly coupled to `/scratch/project_2017385/dorian/Churro_copy`.
- It uses HPC modules and Slurm resource directives; best run with `sbatch` on the cluster:
```bash
sbatch shell_scripts/run_hugh_line_transform.sh
```
