import html,math,pickle,re
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from IPython.display import HTML, display
from matplotlib.transforms import Bbox
from skimage.transform import probabilistic_hough_line

# --------------------------
# Paths and output folders
# --------------------------
IMG_DIR = Path("/scratch/project_2017385/dorian/churro_finnish_dataset/dataset_splits/dev/")
PROJECT_ROOT = Path("/scratch/project_2017385/dorian/Churro_copy")
SCORES_PKL = PROJECT_ROOT / "results/custom_churro_infer_dev_run1/vllm/dev/scores.pkl"

# All-cases output folder
RESULTS_DIR = PROJECT_ROOT / "results/visualise_dorian_dense_matrices_style_no_angle_all"
FULL_DIR = RESULTS_DIR / "full_figures"
GRAPH_DIR = RESULTS_DIR / "graph_only"
MASK_DIR = RESULTS_DIR / "detection_masks"

for out_dir in (FULL_DIR, GRAPH_DIR, MASK_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

# Set integer to limit runs, keep None for all.
MAX_ITEMS = None

# Controls whether notebook text panes are rendered.
RENDER_NOTEBOOK_OUTPUT = True


# --------------------------
# Dense-matrices style helpers
# --------------------------
def safe_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]


def safe_matrix(scores) -> np.ndarray:
    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros((1, 1), dtype=float)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_for_dense_style(mat: np.ndarray) -> np.ndarray:
    """
    Dense-matrices.ipynb uses test = 1/(1-matrix) where matrix is in [0,1).
    Our charF scores are typically 0..100, so we bring them into [0,1).
    """
    if mat.size == 0:
        return mat
    max_val = float(np.max(mat))
    if max_val <= 1.0:
        norm = mat.copy()
    elif max_val <= 100.0:
        norm = mat / 100.0
    else:
        norm = mat / max_val
    return np.clip(norm, 0.0, 0.999999)


def line_magnitude(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def point_line_distance(entry):
    """
    Same geometry as dense_matrices.ipynb distance() implementation.
    entry = (lp1, lp2, p3)
    """
    lp1, lp2, p3 = entry
    px, py = p3
    x1, y1 = lp1
    x2, y2 = lp2

    line_mag = line_magnitude(x1, y1, x2, y2)
    if line_mag < 1e-8:
        return 9999.0

    u1 = ((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1))
    u = u1 / (line_mag * line_mag)

    if (u < 0.00001) or (u > 1):
        ix = line_magnitude(px, py, x1, y1)
        iy = line_magnitude(px, py, x2, y2)
        return iy if ix > iy else ix

    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    return line_magnitude(px, py, ix, iy)


def count_points_in_range(segments, points, d):
    count = 0
    for seg in segments:
        p0, p1 = seg
        for p in points:
            if point_line_distance((p0, p1, p)) <= d:
                count += 1
    return count


def sample_line_pixels(p0, p1, shape):
    (x0, y0), (x1, y1) = p0, p1
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    n = max(n, 2)
    xs = np.clip(np.rint(np.linspace(x0, x1, n)).astype(int), 0, shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(y0, y1, n)).astype(int), 0, shape[0] - 1)
    return xs, ys


def longest_false_run(arr):
    best = 0
    run = 0
    for v in arr:
        if not v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def segment_length(seg):
    (x0, y0), (x1, y1) = seg
    return float(math.hypot(x1 - x0, y1 - y0))


def segment_angle(seg):
    (x0, y0), (x1, y1) = seg
    deg = math.degrees(math.atan2(y1 - y0, x1 - x0))
    # Fold to [0, 180) so opposite directions match.
    return float((deg + 180.0) % 180.0)


def nearest_endpoints(seg_a, seg_b):
    pts_a = [seg_a[0], seg_a[1]]
    pts_b = [seg_b[0], seg_b[1]]
    best_a, best_b = pts_a[0], pts_b[0]
    best_dist = float("inf")
    for pa in pts_a:
        for pb in pts_b:
            d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            if d < best_dist:
                best_dist = d
                best_a, best_b = pa, pb
    return best_a, best_b, best_dist


def bridge_stats(p0, p1, mask):
    xs, ys = sample_line_pixels(p0, p1, mask.shape)
    vals = mask[ys, xs]
    if vals.size == 0:
        return 0.0, 1.0
    support = float(vals.mean())
    gap = float(longest_false_run(vals) / len(vals))
    return support, gap


def merging_diag(lines, mask, points_glo):
    """
    Merge segments only if:
    - endpoints are close relative to segment length
    - angles are consistent
    - the bridge between segments is supported by the mask
    """
    res = []
    lines = sorted(lines, key=lambda x: x[1])
    lines = np.array(lines, dtype=object)
    for line in lines:
        if res == []:
            res.append((line[0], line[1]))
            continue
        p1, p2 = res[-1]
        p3, p4 = line

        len_last = segment_length((p1, p2))
        len_cur = segment_length((p3, p4))
        min_len = max(min(len_last, len_cur), 1.0)
        merge_dist = max(3.0, 0.25 * min_len)

        ang_last = segment_angle((p1, p2))
        ang_cur = segment_angle((p3, p4))
        ang_diff = abs(ang_last - ang_cur)
        if ang_diff > 90.0:
            ang_diff = 180.0 - ang_diff

        pa, pb, endpoint_dist = nearest_endpoints((p1, p2), (p3, p4))
        bridge_support, bridge_gap = bridge_stats(pa, pb, mask)

        # Require close endpoints, consistent angle, and supported bridge.
        if endpoint_dist <= merge_dist and ang_diff <= 12.0 and bridge_support >= 0.60 and bridge_gap <= 0.20:
            points = [(p1, p2), (p1, p4), (p3, p2), (p3, p4)]
            mer = ((0, 0), (0, 0))
            max_p = 0
            for pair in points:
                temp = count_points_in_range([pair], points_glo, 20)
                if temp > max_p:
                    max_p = temp
                    mer = pair
            res.pop()
            res.append(mer)
        else:
            res.append((p3, p4))
    return res


# --------------------------
# Dense-matrices style detection (no angle filter)
# --------------------------
def detect_lines_dense_style_no_angle(matrix):
    """
    Same as dense_matrices.ipynb logic, but WITHOUT angle filtering.
    """
    norm = normalize_for_dense_style(matrix)
    test = 1.0 / (1.0 - norm)

    start = 2.6
    enough = False
    # Use pixel count criterion (not sum of values).
    criteria = 1.4 * matrix.shape[0]

    test2 = test.copy()
    while not enough:
        if start < 0:
            break
        start -= 0.2
        test2 = test.copy()
        test2[test2 < start] = 0
        enough = ((test2 > 0).sum() > criteria)

    ys, xs = np.nonzero(test2)
    points_glo = [(int(x), int(y)) for y, x in zip(ys, xs)]

    # Probabilistic Hough on thresholded image.
    lines = probabilistic_hough_line(test2, threshold=16, line_length=2, line_gap=2)

    # No angle filtering here.
    res_lines = list(lines)

    merged = merging_diag(res_lines, test2 > 0, points_glo)

    return {
        "threshold_start": start,
        "mask": test2,
        "raw_lines": lines,
        "selected_lines": res_lines,
        "merged_lines": merged,
    }


def main():
    # --------------------------
    # Main execution (all cases)
    # --------------------------
    if not SCORES_PKL.exists():
        raise FileNotFoundError(f"Missing scores file: {SCORES_PKL}")

    saved_idx = 0
    seen = 0

    with open(SCORES_PKL, "rb") as f:
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break

            if MAX_ITEMS is not None and seen >= MAX_ITEMS:
                break
            seen += 1

            fig, (ax_img, ax_hm) = plt.subplots(1, 2, figsize=(12, 5))

            img_path = IMG_DIR / item["fname"]
            if img_path.exists():
                ax_img.imshow(plt.imread(img_path))
                ax_img.set_title(item["fname"])
                ax_img.axis("off")
            else:
                ax_img.text(0.5, 0.5, f"Missing image:\n{item['fname']}", ha="center", va="center")
                ax_img.axis("off")

            matrix = safe_matrix(item.get("scores"))
            det = detect_lines_dense_style_no_angle(matrix)

            im = ax_hm.imshow(matrix, aspect="auto", cmap="Greys")
            cbar = plt.colorbar(im, ax=ax_hm, label="chrF")
            ax_hm.set_xlabel("pred segment")
            ax_hm.set_ylabel("ref segment")

            for p0, p1 in det["merged_lines"]:
                ax_hm.plot((p0[0], p1[0]), (p0[1], p1[1]), color="yellow", linewidth=2)

            ax_hm.set_title(
                f"start={det['threshold_start']:.2f} | raw={len(det['raw_lines'])} | "
                f"selected={len(det['selected_lines'])} | merged={len(det['merged_lines'])}"
            )
            plt.tight_layout()

            base_name = safe_name(item["fname"])
            full_out = FULL_DIR / f"{saved_idx:04d}_{base_name}_full.png"
            graph_out = GRAPH_DIR / f"{saved_idx:04d}_{base_name}_graph.png"
            mask_out = MASK_DIR / f"{saved_idx:04d}_{base_name}_mask.png"

            fig.savefig(full_out, dpi=220, bbox_inches="tight", facecolor="white")

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            graph_bbox = Bbox.union(
                [ax_hm.get_tightbbox(renderer), cbar.ax.get_tightbbox(renderer)]
            ).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(graph_out, dpi=260, bbox_inches=graph_bbox, facecolor="white")

            plt.imsave(mask_out, (det["mask"] > 0).astype(np.uint8) * 255, cmap="gray")

            print(f"Saved: {full_out}")
            print(f"Saved: {graph_out}")
            print(f"Saved: {mask_out}")

            plt.show()
            saved_idx += 1

            if RENDER_NOTEBOOK_OUTPUT:
                pred_esc = html.escape(item["pred"])
                ref_esc = html.escape(item["ref"])
                display(
                    HTML(
                        f"""
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
        """
                    )
                )
                display(HTML('<hr style="margin: 32px 0; border: none; border-top: 2px solid #999;">'))

    print(f"Done. Saved {saved_idx} items to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
