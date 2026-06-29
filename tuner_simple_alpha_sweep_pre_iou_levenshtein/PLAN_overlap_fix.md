# Plan: Fix overlapping window double-counting in pre-IoU NLS filter

## Context

When the pre-IoU NLS filter scores raw Hough segments, it concatenates text windows along the segment's geometric path and computes one Levenshtein NLS on the result. With `window_size=50` and `window_stride=35`, adjacent windows overlap by **15 characters**. The current code uses a plain `"".join(windows)` with no deduplication, so those 15 chars are counted **twice** per adjacent window pair. This inflates both string lengths and distorts the NLS, making some short middle-section diagonal segments appear to reach ~0.49–0.51 NLS when they would fall below 0.45 with unique text. Those spurious survivors then pass IoU and collapse into one final bridging diagonal for documents like `tarima_bulac_res_mon_8_7199_0124.jpeg` where the correct result is two separate lines.

The fix: when concatenating consecutive windows, take the full text of the **first** window and only the non-overlapping suffix (`win[overlap:]`) of each subsequent **consecutive** window. Non-consecutive windows (gap in the diagonal) always include the full text.

`window_overlap = window_size − window_stride` (e.g. 50 − 35 = 15)

---

## Files to modify

Root: `/scratch/project_2017385/dorian/Churro_copy/tuner_simple_alpha_sweep_pre_iou_levenshtein/`

### 1. `scoring/raw_hough_line_text_filter.py`

**Fast path** — add `window_overlap: int = 0` parameter to `filter_raw_hough_segments_by_line_levenshtein` and fix the two concatenation lines (~168–171):

```python
# prediction columns are always consecutive
cols = list(range(col_start, col_end + 1))
if window_overlap > 0 and len(cols) > 1:
    pred_text = str(prediction_windows[cols[0]]) + "".join(
        str(prediction_windows[c])[window_overlap:] for c in cols[1:]
    )
else:
    pred_text = "".join(str(prediction_windows[c]) for c in cols)

# reference rows may have gaps
ref_parts = []
prev_r = None
for r in unique_rows:
    win = str(reference_windows[r])
    if window_overlap > 0 and prev_r is not None and r == prev_r + 1:
        ref_parts.append(win[window_overlap:])
    else:
        ref_parts.append(win)
    prev_r = r
ref_text = "".join(ref_parts)
```

### 2. `scoring/line_text_similarity.py`

**Slow path** — `join_text_windows_without_separators` (lines ~118–123) has the same issue. Add `window_overlap: int = 0` and strip for consecutive indices:

```python
def join_text_windows_without_separators(windows, indices, window_overlap: int = 0):
    parts = []
    prev = None
    for idx in indices:
        win = str(windows[int(idx)])
        if window_overlap > 0 and prev is not None and int(idx) == prev + 1:
            parts.append(win[window_overlap:])
        else:
            parts.append(win)
        prev = int(idx)
    return "".join(parts)
```

Thread `window_overlap` through: `compute_line_text_record` → `join_text_windows_without_separators` (both calls at ~lines 234 and 236). Also thread through `filter_lines_by_minimum_normalised_levenshtein` and `compute_line_text_records`.

### 3. `probabilistic_hough/hough_detection.py`

Add `window_overlap: int = 0` to `run_probabilistic_hough_and_filter` signature and pass it down to `filter_raw_hough_segments_by_line_levenshtein` (~line 366).

### 4. `serial_runner/document_runner.py`

In `run_alpha_candidate` (~line 876), compute overlap once and pass to both callers:

```python
window_overlap = max(0, config.window_size - config.window_stride)
# pass to run_probabilistic_hough_and_filter (line ~876)
# pass to filter_lines_by_minimum_normalised_levenshtein (line ~894)
```

---

## What does NOT change

- `PipelineConfig` — `window_size` and `window_stride` already stored; no new config fields needed
- CLI / shell scripts — no new flags; overlap is derived automatically
- Score matrix pkl files — untouched
- Any Hough, IoU, or scoring pipeline code outside the concatenation path

---

## Sync to HTR-context-OCR

After implementing in Churro_copy, copy the same 4 changed files to:
`/scratch/project_2017385/dorian/HTR-context-OCR/tuner_simple_alpha_sweep_pre_iou_levenshtein/`

---

## Verification

Run the tarima document with `--min-surviving-line-nls 0.45` and check that the 3 middle-section spurious survivors (~0.49–0.51 pre-fix) now score below 0.45 and are rejected:

```bash
/appl/soft/ai/wrap/pytorch-2.9/bin/python3 \
  /scratch/project_2017385/dorian/Churro_copy/results/tarima_debug_plots/plot_pre_iou_survivors.py
```

Expected: survivors count drops (ideally to 0 or only the legitimate endpoint-cluster lines remain).

Then run the full alpha-sweep to confirm the final result improves:

```bash
/appl/soft/ai/wrap/pytorch-2.9/bin/python3 \
  /scratch/project_2017385/dorian/Churro_copy/tuner_simple_alpha_sweep_pre_iou_levenshtein/run_tuner_simple.py \
  --output-dir /scratch/project_2017385/dorian/Churro_copy/results/tarima_debug_plots/overlap_fix_run \
  --target-fname tarima_bulac_res_mon_8_7199_0124.jpeg \
  --all-document-types \
  --window-size 50 --window-stride 35 \
  --minimum-matrix-rows 10 --minimum-matrix-columns 10 \
  --alpha-sweep --alpha-sweep-min 0.6 --alpha-sweep-max 4.0 --alpha-sweep-step 0.1 \
  --hough-threshold 3 --hough-line-length 3 --hough-line-gap 3 \
  --hough-num-runs 10 \
  --align-min-iou-threshold 0.10 --min-surviving-line-nls 0.45 \
  --plot-mode stitched-language --stitched-panel-columns 1 --saved-figure-dpi 100 \
  --harmonic-mode balanced
```

Expected: `used_lines=2` at the selected alpha (correct 2-line result) instead of the current `used_lines=1`.
