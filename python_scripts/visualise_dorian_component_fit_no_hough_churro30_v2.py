import html
import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.transforms import Bbox
from skimage import measure, morphology

"""
Visualize charF score matrices for the churro_finnish dev set and overlay
detected diagonal alignment lines (component-fit, no Hough transform).

This script processes up to CHURRO_MATCH_LIMIT examples from scores.pkl and,
for each example, saves:
1) a full figure (source image + heatmap with fitted lines),
2) a graph-only crop (heatmap area + colorbar),
3) the binary threshold mask used for component extraction,
4) a component mask showing only connected components used for final lines.

Important: this script is intentionally "analysis-first". It preserves
intermediate artifacts to help debug why a line was or was not detected.
"""


# Root paths for this experiment run.
PROJECT_ROOT = Path("/scratch/project_2017385/dorian/Churro_copy")
SCORES_PKL = PROJECT_ROOT / "results/custom_churro_infer_dev_run1/vllm/dev/scores.pkl"
IMG_DIR = PROJECT_ROOT / "churro_finnish_dataset/dev"

# Output folder structure:
# - full_figures: source image + heatmap + fitted lines
# - graph_only: cropped heatmap region (for fast visual inspection)
# - binary_masks: thresholded masks before component filtering
# - component_masks: only components that survived geometric filters
RESULTS_DIR = PROJECT_ROOT / "results/visualise_dorian_component_fit_no_hough_churro30_v2"
FULL_DIR = RESULTS_DIR / "full_figures"
GRAPH_DIR = RESULTS_DIR / "graph_only"
MASK_DIR = RESULTS_DIR / "binary_masks"
COMPONENT_DIR = RESULTS_DIR / "component_masks"

FULL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
COMPONENT_DIR.mkdir(parents=True, exist_ok=True)

# Process only a fixed subset for controlled evaluation/debug.
CHURRO_MATCH_LIMIT = 30
# If running inside a notebook, keep visual output by default.
# Set VIZ_NOTEBOOK_OUTPUT=0 to run in headless/batch mode.
RENDER_NOTEBOOK_OUTPUT = os.environ.get("VIZ_NOTEBOOK_OUTPUT", "1") == "1"


def safe_name(name: str) -> str:
    """
    Convert a filename into a filesystem-safe stem:
    - keep only [A-Za-z0-9._-]
    - collapse other chars to "_"
    - cap length to avoid path issues
    """
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]


def safe_matrix(scores) -> np.ndarray:
    """
    Convert raw scores to a valid numeric 2D matrix.

    Why this is needed:
    - scores.pkl may contain unexpected types/shapes;
    - NaN/Inf values break thresholding and fitting;
    - downstream code assumes a non-empty 2D array.
    """
    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros((1, 1), dtype=float)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def detect_lines_component_fit(matrix: np.ndarray):
    """
    Component-fit pipeline (no Hough) used to extract likely alignment lines.

    Steps:
    1) Threshold matrix at percentile p88 (keep strongest charF signal).
    2) Morphological cleanup to remove isolated noise and close tiny gaps.
    3) Label connected components in the binary mask.
    4) Fit one straight line per component using np.polyfit(xs, ys, 1).
    5) Keep only near-diagonal fitted lines (slope between 0.5 and 2.0).
    6) Score each line as mean_intensity * area and keep top 5.

    Small-matrix fix:
    - Tiny matrices can lose true diagonals during cleanup because signal
      regions are just a few pixels wide.
    - For these cases we lower object-size constraints and allow corner-based
      connectivity, so short but real diagonal chains survive.

    Returns:
    - thr: chosen threshold value (for reporting/debugging)
    - mask: cleaned binary signal mask
    - labels: connected-component label image
    - lines: top scored fitted lines
    - component_ids: labels of components used in "lines"
    - is_small_matrix: whether small-matrix branch was used
    """
    thr = float(np.percentile(matrix, 88))
    mask = matrix > thr

    h, w = matrix.shape
    min_dim = min(h, w)
    is_small_matrix = min_dim <= 20 or (h * w) <= 400

    if is_small_matrix:
        # Tiny matrices often contain diagonals connected only by corners.
        # Keep those by using 8-connectivity and a smaller min component size.
        mask = morphology.remove_small_objects(mask, min_size=3, connectivity=2)
        mask = morphology.binary_closing(mask, morphology.disk(1))
        min_region_area = 6
    else:
        mask = morphology.remove_small_objects(mask, min_size=15)
        mask = morphology.binary_closing(mask, morphology.disk(1))
        min_region_area = 20

    labels = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labels, intensity_image=matrix)

    lines = []
    component_ids = []

    for r in regions:
        # Skip tiny regions that are likely residual noise blobs.
        if r.area < min_region_area:
            continue

        coords = r.coords
        ys = coords[:, 0]
        xs = coords[:, 1]

        # Need at least 2 unique x positions for line fitting.
        if np.unique(xs).size < 2:
            continue

        # Fit y = slope*x + intercept through component pixels.
        slope, intercept = np.polyfit(xs, ys, 1)

        # Keep only near-diagonal line directions.
        if not (0.5 < slope < 2.0):
            continue

        x0 = float(xs.min())
        x1 = float(xs.max())
        y0 = float(slope * x0 + intercept)
        y1 = float(slope * x1 + intercept)

        # Clamp to image bounds for plotting.
        x0 = float(np.clip(x0, 0, w - 1))
        x1 = float(np.clip(x1, 0, w - 1))
        y0 = float(np.clip(y0, 0, h - 1))
        y1 = float(np.clip(y1, 0, h - 1))

        # Score combines brightness (mean intensity) and support (area).
        score = float(r.mean_intensity * r.area)
        lines.append((score, (x0, y0, x1, y1), int(r.label), float(slope)))
        component_ids.append(int(r.label))

    # Keep only strongest few lines to reduce overdraw and false positives.
    lines = sorted(lines, key=lambda x: x[0], reverse=True)[:5]

    return thr, mask, labels, lines, component_ids, is_small_matrix


print("scores exists:", SCORES_PKL.exists())
print("img dir exists:", IMG_DIR.exists())
print("saving to:", RESULTS_DIR)

with open(SCORES_PKL, "rb") as f:
    processed = 0

    while True:
        try:
            item = pickle.load(f)
        except EOFError:
            break

        # scores.pkl stores relative file names; normalize to dataset path.
        img_name = Path(item["fname"]).name
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            # Skip broken references silently to continue batch processing.
            continue

        matrix = safe_matrix(item["scores"])
        thr, mask, labels, lines, component_ids, is_small_matrix = detect_lines_component_fit(matrix)

        fig, (ax_img, ax_hm) = plt.subplots(1, 2, figsize=(12, 5))

        ax_img.imshow(plt.imread(img_path))
        ax_img.set_title(img_name)
        ax_img.axis("off")

        im = ax_hm.imshow(matrix, aspect="auto", cmap="viridis")
        cbar = plt.colorbar(im, ax=ax_hm, label="chrF")
        ax_hm.set_xlabel("pred segment")
        ax_hm.set_ylabel("ref segment")

        # Overlay fitted lines in black on top of the viridis heatmap.
        for score, (x0, y0, x1, y1), _label, _slope in lines:
            ax_hm.plot([x0, x1], [y0, y1], "-k", linewidth=2)

        mode = "small-matrix" if is_small_matrix else "default"
        ax_hm.set_title(
            f"component-fit-no-hough-v2 | {mode} | thr(p88)={thr:.2f} | kept={len(lines)}"
        )

        plt.tight_layout()

        base_name = safe_name(item["fname"])
        full_out = FULL_DIR / f"{processed:04d}_{base_name}_full.png"
        graph_out = GRAPH_DIR / f"{processed:04d}_{base_name}_graph.png"
        mask_out = MASK_DIR / f"{processed:04d}_{base_name}_mask.png"
        comp_out = COMPONENT_DIR / f"{processed:04d}_{base_name}_components.png"

        # 1) Save full side-by-side figure.
        fig.savefig(full_out, dpi=220, bbox_inches="tight", facecolor="white")

        # 2) Save graph-only crop: union of heatmap axis + colorbar axis.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        graph_bbox = Bbox.union([
            ax_hm.get_tightbbox(renderer),
            cbar.ax.get_tightbbox(renderer),
        ]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(graph_out, dpi=260, bbox_inches=graph_bbox, facecolor="white")

        # 3) Save cleaned binary signal mask.
        fig_mask, ax_mask = plt.subplots(figsize=(6, 6))
        ax_mask.imshow(mask, cmap="gray")
        ax_mask.set_title(f"binary mask | p88={thr:.2f}")
        ax_mask.set_axis_off()
        fig_mask.tight_layout()
        fig_mask.savefig(mask_out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig_mask)

        # 4) Save only the components that were actually used for fitted lines.
        selected_components = np.isin(labels, np.array(component_ids, dtype=int))
        fig_comp, ax_comp = plt.subplots(figsize=(6, 6))
        ax_comp.imshow(selected_components, cmap="gray")
        ax_comp.set_title("components used for fitted lines")
        ax_comp.set_axis_off()
        fig_comp.tight_layout()
        fig_comp.savefig(comp_out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig_comp)

        print(
            f"[{processed:04d}] {img_name} -> mode={mode} kept_lines={len(lines)}"
        )

        if RENDER_NOTEBOOK_OUTPUT:
            # Interactive notebook mode: show image pair + text panes.
            plt.show()
            pred_esc = html.escape(item.get("pred", ""))
            ref_esc = html.escape(item.get("ref", ""))
            display(HTML(f'''
            <div style="margin-top: 10px; display: flex; gap: 16px;">
                <div style="flex: 1;">
                    <div><strong>Predicted:</strong></div>
                    <div style="height: 300px; resize: vertical; overflow-y: auto; border: 1px solid #ccc; padding: 8px; font-family: monospace; white-space: pre-wrap;">{pred_esc}</div>
                </div>
                <div style="flex: 1;">
                    <div><strong>Reference:</strong></div>
                    <div style="height: 300px; resize: vertical; overflow-y: auto; border: 1px solid #ccc; padding: 8px; font-family: monospace; white-space: pre-wrap;">{ref_esc}</div>
                </div>
            </div>
            '''))
            display(HTML('<hr style="margin: 32px 0; border: none; border-top: 2px solid #999;">'))
        else:
            # Headless mode: close figures to avoid memory growth.
            plt.close(fig)

        processed += 1
        if processed >= CHURRO_MATCH_LIMIT:
            break

print(f"Done. Saved {processed} graph sets to: {RESULTS_DIR}")
