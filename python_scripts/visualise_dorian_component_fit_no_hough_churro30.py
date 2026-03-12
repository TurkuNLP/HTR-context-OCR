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


PROJECT_ROOT = Path("/scratch/project_2017385/dorian/Churro_copy")
SCORES_PKL = PROJECT_ROOT / "results/custom_churro_infer_dev_run1/vllm/dev/scores.pkl"
IMG_DIR = PROJECT_ROOT / "churro_finnish_dataset/dev"

RESULTS_DIR = PROJECT_ROOT / "results/visualise_dorian_component_fit_no_hough_churro30"
FULL_DIR = RESULTS_DIR / "full_figures"
GRAPH_DIR = RESULTS_DIR / "graph_only"
MASK_DIR = RESULTS_DIR / "binary_masks"
COMPONENT_DIR = RESULTS_DIR / "component_masks"

FULL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
COMPONENT_DIR.mkdir(parents=True, exist_ok=True)

CHURRO_MATCH_LIMIT = 30
RENDER_NOTEBOOK_OUTPUT = os.environ.get("VIZ_NOTEBOOK_OUTPUT", "1") == "1"


def safe_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]


def safe_matrix(scores) -> np.ndarray:
    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros((1, 1), dtype=float)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def detect_lines_component_fit(matrix: np.ndarray):
    """
    Fixed, simplified component-fit pipeline (no Hough):
    1) threshold at p95,
    2) remove small objects + closing,
    3) connected components,
    4) per-component linear fit,
    5) keep near-diagonal fitted lines.
    """
    thr = float(np.percentile(matrix, 95))
    mask = matrix > thr

    mask = morphology.remove_small_objects(mask, 15)
    mask = morphology.binary_closing(mask, morphology.disk(1))

    labels = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labels, intensity_image=matrix)

    lines = []
    component_ids = []

    for r in regions:
        if r.area < 20:
            continue

        coords = r.coords
        ys = coords[:, 0]
        xs = coords[:, 1]

        # Need at least 2 unique x positions for line fitting.
        if np.unique(xs).size < 2:
            continue

        slope, intercept = np.polyfit(xs, ys, 1)

        # Keep only near-diagonal line directions.
        if not (0.5 < slope < 2.0):
            continue

        x0 = float(xs.min())
        x1 = float(xs.max())
        y0 = float(slope * x0 + intercept)
        y1 = float(slope * x1 + intercept)

        # Clamp to image bounds for plotting.
        h, w = matrix.shape
        x0 = float(np.clip(x0, 0, w - 1))
        x1 = float(np.clip(x1, 0, w - 1))
        y0 = float(np.clip(y0, 0, h - 1))
        y1 = float(np.clip(y1, 0, h - 1))

        score = float(r.mean_intensity * r.area)
        lines.append((score, (x0, y0, x1, y1), int(r.label), float(slope)))
        component_ids.append(int(r.label))

    lines = sorted(lines, key=lambda x: x[0], reverse=True)[:5]

    return thr, mask, labels, lines, component_ids


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

        img_name = Path(item["fname"]).name
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            continue

        matrix = safe_matrix(item["scores"])
        thr, mask, labels, lines, component_ids = detect_lines_component_fit(matrix)

        fig, (ax_img, ax_hm) = plt.subplots(1, 2, figsize=(12, 5))

        ax_img.imshow(plt.imread(img_path))
        ax_img.set_title(img_name)
        ax_img.axis("off")

        im = ax_hm.imshow(matrix, aspect="auto", cmap="viridis")
        cbar = plt.colorbar(im, ax=ax_hm, label="chrF")
        ax_hm.set_xlabel("pred segment")
        ax_hm.set_ylabel("ref segment")

        for score, (x0, y0, x1, y1), _label, _slope in lines:
            ax_hm.plot([x0, x1], [y0, y1], "-k", linewidth=2)

        ax_hm.set_title(
            f"component-fit-no-hough | thr(p95)={thr:.2f} | kept={len(lines)}"
        )

        plt.tight_layout()

        base_name = safe_name(item["fname"])
        full_out = FULL_DIR / f"{processed:04d}_{base_name}_full.png"
        graph_out = GRAPH_DIR / f"{processed:04d}_{base_name}_graph.png"
        mask_out = MASK_DIR / f"{processed:04d}_{base_name}_mask.png"
        comp_out = COMPONENT_DIR / f"{processed:04d}_{base_name}_components.png"

        fig.savefig(full_out, dpi=220, bbox_inches="tight", facecolor="white")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        graph_bbox = Bbox.union([
            ax_hm.get_tightbbox(renderer),
            cbar.ax.get_tightbbox(renderer),
        ]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(graph_out, dpi=260, bbox_inches=graph_bbox, facecolor="white")

        fig_mask, ax_mask = plt.subplots(figsize=(6, 6))
        ax_mask.imshow(mask, cmap="gray")
        ax_mask.set_title(f"binary mask | p95={thr:.2f}")
        ax_mask.set_axis_off()
        fig_mask.tight_layout()
        fig_mask.savefig(mask_out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig_mask)

        selected_components = np.isin(labels, np.array(component_ids, dtype=int))
        fig_comp, ax_comp = plt.subplots(figsize=(6, 6))
        ax_comp.imshow(selected_components, cmap="gray")
        ax_comp.set_title("components used for fitted lines")
        ax_comp.set_axis_off()
        fig_comp.tight_layout()
        fig_comp.savefig(comp_out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig_comp)

        print(f"[{processed:04d}] {img_name} -> kept_lines={len(lines)}")

        if RENDER_NOTEBOOK_OUTPUT:
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
            plt.close(fig)

        processed += 1
        if processed >= CHURRO_MATCH_LIMIT:
            break

print(f"Done. Saved {processed} graph sets to: {RESULTS_DIR}")
