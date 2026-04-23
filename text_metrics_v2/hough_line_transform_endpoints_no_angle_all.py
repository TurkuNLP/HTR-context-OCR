import argparse
import math
import pickle
from pathlib import Path

import numpy as np
from skimage.transform import probabilistic_hough_line


"""\
Extract probabilistic Hough-line endpoints from chrF score matrices.

This is a pipeline-friendly extractor that:
- Reads compare.py output (scores.pkl) which is a pickle *stream* of dicts.
- Runs the dense-matrices style detector (no angle filter).
- Persists ONLY the line endpoints (merged segments) + a bit of metadata,
  as another pickle *stream* so downstream alignment can reuse the endpoints.

The detector logic/thresholds are intentionally kept identical to
`visualise_dorian_dense_matrices_style_single_no_angle.py`.
"""


# Fixed theta range used by downstream alignment scripts.
# In Hough space this is the normal angle (not line direction angle).
# Keeping line-direction angles in [35 deg, 90 deg) maps to normal-angle
# bands [-55 deg, 0 deg) U (0 deg, 55 deg], which preserves both
# diagonal slants while excluding perfectly vertical 90 deg lines.
DIAGONAL_THETA_DEG = np.r_[
    np.arange(-55, 0, 0.5),
    np.arange(0.5, 55 + 0.5, 0.5),
]
DIAGONAL_THETA_RAD = np.deg2rad(DIAGONAL_THETA_DEG)


def safe_matrix(scores) -> np.ndarray:
    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros((1, 1), dtype=float)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_for_dense_style(mat: np.ndarray) -> np.ndarray:
    """Normalize chrF score matrix to [0, 1) for dense-matrices transform."""
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
    """Same geometry as dense_matrices.ipynb distance() implementation."""
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
    """Merge line segments using the same constraints as the notebook/script."""
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

        if (
            endpoint_dist <= merge_dist
            and ang_diff <= 12.0
            and bridge_support >= 0.60
            and bridge_gap <= 0.20
        ):
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


def detect_lines_dense_style(
    matrix,
    *,
    threshold: int = 26,
    line_length: int = 10,
    line_gap: int = 15,
    theta=None,
    rng=None,
    start_init: float = 2.6,
):
    """Dense-matrices style detector with configurable Hough settings."""
    norm = normalize_for_dense_style(matrix)
    test = 1.0 / (1.0 - norm)

    start = float(start_init)
    enough = False
    criteria = 1.4 * matrix.shape[0]

    test2 = test.copy()
    while not enough:
        if start < 0:
            break
        start -= 0.2
        test2 = test.copy()
        test2[test2 < start] = 0
        enough = (test2 > 0).sum() > criteria

    ys, xs = np.nonzero(test2)
    points_glo = [(int(x), int(y)) for y, x in zip(ys, xs)]

    lines = probabilistic_hough_line(
        test2,
        threshold=int(threshold),
        line_length=int(line_length),
        line_gap=int(line_gap),
        theta=theta,
        rng=rng,
    )
    res_lines = list(lines)
    merged = merging_diag(res_lines, test2 > 0, points_glo)

    return {
        "threshold_start": start,
        "mask": test2,
        "raw_lines": lines,
        "selected_lines": res_lines,
        "merged_lines": merged,
    }


def detect_lines_dense_style_no_angle(matrix, *, start_init: float = 2.6):
    """Dense-matrices style detector (no angle filter)."""
    return detect_lines_dense_style(matrix, start_init=start_init)


def detect_lines_dense_style_diagonal_fixed_theta(
    matrix,
    *,
    seed: int,
    threshold: int = 26,
    line_length: int = 10,
    line_gap: int = 15,
    start_init: float = 2.6,
):
    """Dense detector with fixed diagonal theta bands and seeded RNG."""
    return detect_lines_dense_style(
        matrix,
        threshold=threshold,
        line_length=line_length,
        line_gap=line_gap,
        theta=DIAGONAL_THETA_RAD,
        rng=np.random.default_rng(int(seed)),
        start_init=start_init,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract probabilistic Hough-line endpoints from chrF score matrices "
            "(dense-matrices style; no angle filter) and save them as a pickle stream."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scores-pkl",
        type=Path,
        required=True,
        help="Path to compare.py output scores.pkl (pickle stream).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output pickle stream containing per-item line endpoints.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit the number of processed items (None means all).",
    )
    p.add_argument(
        "--start-init",
        type=float,
        default=2.6,
        help="Initial adaptive threshold start value before decrementing by 0.2.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.scores_pkl.exists():
        raise FileNotFoundError(f"Missing scores file: {args.scores_pkl}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.scores_pkl, "rb") as in_f, open(args.output, "wb") as out_f:
        while True:
            try:
                item = pickle.load(in_f)
            except EOFError:
                break

            if args.max_items is not None and written >= args.max_items:
                break

            fname = item.get("fname")
            if not fname:
                fname = f"item_{written:04d}"

            matrix = safe_matrix(item.get("scores"))
            det = detect_lines_dense_style_no_angle(matrix, start_init=float(args.start_init))

            rec = {
                "index": written,
                "fname": str(fname),
                "threshold_start": float(det["threshold_start"]),
                "merged_lines": det["merged_lines"],
                "raw_line_count": int(len(det["raw_lines"])),
                "selected_line_count": int(len(det["selected_lines"])),
                "merged_line_count": int(len(det["merged_lines"])),
            }
            pickle.dump(rec, out_f)
            written += 1

            if written % 50 == 0:
                print(f"[hough] processed {written} items")

    print(f"[hough] Wrote {written} endpoint records to: {args.output}")


if __name__ == "__main__":
    main()
