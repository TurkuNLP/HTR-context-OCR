import html
import math
import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.transforms import Bbox
from skimage import measure, morphology
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = Path("/scratch/project_2017385/dorian/Churro_copy")
SCORES_PKL = PROJECT_ROOT / "results/custom_churro_infer_dev_run1/vllm/dev/scores.pkl"
IMG_DIR = PROJECT_ROOT / "churro_finnish_dataset/dev"

RESULTS_DIR = PROJECT_ROOT / "results/visualise_dorian_simple"
FULL_AFTER_RGB_DIR = RESULTS_DIR / "full_figures_after_rgb"
GRAPH_AFTER_RGB_DIR = RESULTS_DIR / "graph_only_after_rgb"
MASK_DIR = RESULTS_DIR / "ridge_masks"
EDGE_DIR = RESULTS_DIR / "edge_masks"
ENDPOINT_DIR = RESULTS_DIR / "endpoints_maps"

for out_dir in (FULL_AFTER_RGB_DIR, GRAPH_AFTER_RGB_DIR, MASK_DIR, EDGE_DIR, ENDPOINT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

MAX_ITEMS = None  # set int for debugging; keep None for full run
RENDER_NOTEBOOK_OUTPUT = os.environ.get("VIZ_NOTEBOOK_OUTPUT", "1") == "1"
SAVE_OUTPUTS = os.environ.get("VIZ_SAVE_OUTPUTS", "1") == "1"

DETECTOR_CONFIG = {
    # Candidate budget keeps full-run runtime bounded while preserving enough options
    # for fragmented graphs.
    "max_candidates_tiny": 32,
    "max_candidates_small": 72,
    "max_candidates_medium": 120,
    "max_candidates_large": 180,
    # Per-Hough-pass cap before global merge.
    "max_hough_lines_per_pass": 180,
    # Skip trivially short segments after support split.
    "min_segment_len_px": 2.0,
}


# -----------------------------
# Utility helpers
# -----------------------------
def safe_name(name):
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]


def safe_matrix(scores):
    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros((1, 1), dtype=float)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def robust_normalize(mat):
    flat = mat[np.isfinite(mat)]
    if flat.size == 0:
        return np.zeros_like(mat, dtype=float)

    lo, hi = np.percentile(flat, [2, 98])
    if hi <= lo:
        return np.zeros_like(mat, dtype=float)

    out = (mat - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def square_footprint(k):
    size = max(1, int(k))
    return morphology.footprint_rectangle((size, size))


def sample_line_pixels(p0, p1, shape):
    (x0, y0), (x1, y1) = p0, p1
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    n = max(2, n)
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


def canonical_segment(seg):
    p0, p1 = seg
    if (p0[0], p0[1]) <= (p1[0], p1[1]):
        return (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))
    return (int(p1[0]), int(p1[1])), (int(p0[0]), int(p0[1]))


def candidate_budget(min_dim):
    if min_dim <= 12:
        return DETECTOR_CONFIG["max_candidates_tiny"]
    if min_dim <= 40:
        return DETECTOR_CONFIG["max_candidates_small"]
    if min_dim <= 140:
        return DETECTOR_CONFIG["max_candidates_medium"]
    return DETECTOR_CONFIG["max_candidates_large"]


def line_angle_folded(seg):
    (x0, y0), (x1, y1) = seg
    deg = abs(math.degrees(math.atan2(y1 - y0, x1 - x0)))
    if deg > 90.0:
        deg = 180.0 - deg
    return float(deg)


def main_diagonal_shape_score(p0, p1, shape):
    h, w = shape
    dx = abs(float(p1[0] - p0[0]))
    dy = abs(float(p1[1] - p0[1]))
    if dx == 0.0 and dy == 0.0:
        return 0.0
    sx = dx * h
    sy = dy * w
    den = max(sx, sy, 1e-8)
    return float(min(sx, sy) / den)


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


def segments_are_similar(seg_a, seg_b, shape):
    len_a = segment_length(seg_a)
    len_b = segment_length(seg_b)
    if len_a <= 1e-8 or len_b <= 1e-8:
        return False

    ratio = max(len_a, len_b) / max(min(len_a, len_b), 1e-8)
    if ratio > 2.0:
        return False

    ang_a = line_angle_folded(seg_a)
    ang_b = line_angle_folded(seg_b)
    ang_diff = abs(ang_a - ang_b)
    if ang_diff > 90.0:
        ang_diff = 180.0 - ang_diff

    _, _, end_dist = nearest_endpoints(seg_a, seg_b)
    diag = max(1.0, math.hypot(shape[0], shape[1]))
    end_tol = max(2.0, 0.012 * diag)
    return ang_diff <= 12.0 and end_dist <= end_tol


# -----------------------------
# Per-graph profiling
# -----------------------------
def matrix_profile(norm):
    h, w = norm.shape
    global_std = float(np.std(norm)) + 1e-8

    row_means = np.mean(norm, axis=1)
    col_means = np.mean(norm, axis=0)
    stripe_strength = float(max(np.std(row_means), np.std(col_means)) / global_std)
    stripe_strength = float(np.clip(stripe_strength, 0.0, 3.0))

    band_w = max(1, int(round(min(h, w) * 0.03)))
    yy, xx = np.indices((h, w))
    diag_y = xx * (h - 1) / max(w - 1, 1)
    band = np.abs(yy - diag_y) <= band_w

    diag_mean = float(norm[band].mean()) if band.any() else 0.0
    off_band = ~band
    off_mean = float(norm[off_band].mean()) if off_band.any() else diag_mean
    diag_contrast = float(np.clip(diag_mean - off_mean, -1.0, 1.0))

    flat = norm[np.isfinite(norm)]
    hi = float(np.quantile(flat, 0.95)) if flat.size else 1.0
    high = norm >= hi
    high_total = int(high.sum())
    diag_density = float(high[band].sum() / high_total) if high_total > 0 else 0.0

    return {
        "h": int(h),
        "w": int(w),
        "min_dim": int(min(h, w)),
        "shape_ratio": float(max(h, w) / max(min(h, w), 1)),
        "stripe_strength": stripe_strength,
        "diag_contrast": diag_contrast,
        "diag_density": float(np.clip(diag_density, 0.0, 1.0)),
    }


# -----------------------------
# Adaptive masks and Hough params
# -----------------------------
def clean_mask(mask, min_dim):
    if not mask.any():
        return mask.astype(bool)

    mask = mask.astype(bool)
    if min_dim <= 10:
        # Tiny matrices are fragile; avoid aggressive cleanup that can erase the signal.
        return mask.copy()

    frac = max(mask.mean(), 1e-8)
    min_obj = max(2, int(round(mask.size * min(frac * 0.18, 0.005))))
    mask = morphology.remove_small_objects(mask, min_size=min_obj)

    k = int(np.clip(round(min_dim / 120), 1, 3))
    mask = morphology.binary_closing(mask, square_footprint(max(1, k)))
    mask = morphology.binary_opening(mask, square_footprint(2))
    return mask


def enhanced_signal(norm):
    # Remove row/column baseline to suppress stripe-like background responses.
    row_med = np.median(norm, axis=1, keepdims=True)
    col_med = np.median(norm, axis=0, keepdims=True)
    baseline = 0.5 * (row_med + col_med)
    residual = np.clip(norm - baseline, 0.0, None)
    return robust_normalize(residual)


def build_masks(norm, profile):
    min_dim = profile["min_dim"]
    flat = norm[np.isfinite(norm)]
    if flat.size == 0:
        empty = np.zeros_like(norm, dtype=bool)
        return empty, empty, empty, np.zeros_like(norm, dtype=float)

    signal_norm = enhanced_signal(norm)
    signal_flat = signal_norm[np.isfinite(signal_norm)]

    # Adaptive ridge threshold: lower threshold when diagonal is weak/fragmented.
    base_q = 0.92
    q = base_q - 0.12 * float(np.clip(0.16 - profile["diag_contrast"], 0.0, 0.16) / 0.16)
    q -= 0.08 * float(np.clip(0.62 - profile["diag_density"], 0.0, 0.62) / 0.62)
    q += 0.05 * float(np.clip(profile["stripe_strength"] / 1.4, 0.0, 1.0))
    if min_dim <= 12:
        q -= 0.10
    q = float(np.clip(q, 0.66, 0.97))

    ridge_thr = float(np.quantile(flat, q))
    sig_q = float(np.clip(q - 0.06 - 0.08 * np.clip(profile["stripe_strength"] / 1.2, 0.0, 1.0), 0.58, 0.94))
    sig_thr = float(np.quantile(signal_flat, sig_q)) if signal_flat.size else 1.0

    ridge_raw = norm >= ridge_thr
    ridge_sig = signal_norm >= sig_thr
    ridge = ridge_raw | ridge_sig

    sigma = float(np.clip(0.75 + min_dim / 230.0, 0.75, 1.9))
    low_q = float(np.clip(q - 0.18, 0.55, 0.90))
    high_q = float(np.clip(q - 0.04, 0.70, 0.98))
    low_thr = float(np.quantile(flat, low_q))
    high_thr = float(np.quantile(flat, high_q))
    if high_thr <= low_thr:
        high_thr = min(1.0, low_thr + 0.04)

    edges = canny(norm, sigma=sigma, low_threshold=low_thr, high_threshold=high_thr)

    support = clean_mask(ridge | edges, min_dim=min_dim)
    if not support.any():
        support = clean_mask(ridge, min_dim=min_dim)
    if not support.any():
        top_k = max(1, int(round(min_dim * 1.5)))
        seeds = np.zeros_like(norm, dtype=bool)
        idx = np.argpartition(norm.ravel(), -top_k)[-top_k:]
        seeds.ravel()[idx] = True
        support = clean_mask(seeds | ridge_sig, min_dim=min_dim)

    work = morphology.skeletonize(support)
    if edges.any():
        work = work | edges
    if not work.any():
        work = support.copy()

    return support, edges, work, signal_norm


def hough_param_grid(work_mask, profile):
    min_dim = profile["min_dim"]
    support_px = max(1, int(work_mask.sum()))

    if min_dim <= 10:
        return [
            {"threshold": 1, "line_length": 2, "line_gap": 1},
            {"threshold": 1, "line_length": 3, "line_gap": 1},
            {"threshold": 2, "line_length": 2, "line_gap": 2},
        ]

    base_len = int(np.clip(round(0.10 * min_dim), 3, max(6, int(round(0.75 * min_dim)))))
    base_gap = int(np.clip(round(0.02 * min_dim), 1, max(2, int(round(0.25 * base_len)))))
    base_vote = int(np.clip(round(np.sqrt(support_px) * 0.09), 2, max(10, min_dim)))

    return [
        {"threshold": base_vote, "line_length": base_len, "line_gap": base_gap},
        {
            "threshold": max(2, int(round(base_vote * 0.90))),
            "line_length": max(2, int(round(base_len * 0.88))),
            "line_gap": max(1, int(round(base_gap * 0.80))),
        },
        {
            "threshold": max(2, int(round(base_vote * 1.10))),
            "line_length": max(2, int(round(base_len * 1.08))),
            "line_gap": max(1, int(round(base_gap * 0.92))),
        },
        {
            "threshold": max(1, int(round(base_vote * 0.80))),
            "line_length": max(2, int(round(base_len * 0.72))),
            "line_gap": max(1, int(round(base_gap * 0.70))),
        },
    ]


# -----------------------------
# Line scoring (dense-matrices inspired)
# -----------------------------
def endpoint_support_score(seg, support_mask):
    h, w = support_mask.shape
    p0, p1 = seg
    pts = [(p0[0], p0[1]), (p1[0], p1[1])]

    rad = max(1, int(round(min(h, w) / 120.0)))
    scores = []
    for x, y in pts:
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        x0 = max(0, xi - rad)
        x1 = min(w, xi + rad + 1)
        y0 = max(0, yi - rad)
        y1 = min(h, yi + rad + 1)
        patch = support_mask[y0:y1, x0:x1]
        scores.append(float(patch.mean()) if patch.size else 0.0)
    return float(np.mean(scores))


def split_line_on_support(seg, support_mask):
    xs, ys = sample_line_pixels(seg[0], seg[1], support_mask.shape)
    vals = support_mask[ys, xs]
    if vals.size == 0:
        return []

    min_run = max(2, int(round(math.sqrt(len(vals)) * 0.55)))

    runs = []
    start = None
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(vals) - 1))

    pieces = []
    for lo, hi in runs:
        if (hi - lo + 1) < min_run:
            continue
        p0 = (int(xs[lo]), int(ys[lo]))
        p1 = (int(xs[hi]), int(ys[hi]))
        if p0 != p1:
            pieces.append((p0, p1))

    # If support is mostly continuous, keep original line.
    if not pieces and float(vals.mean()) > 0.70:
        return [seg]
    return pieces


def line_corridor_and_surround(seg, shape, radius):
    base = np.zeros(shape, dtype=bool)
    xs, ys = sample_line_pixels(seg[0], seg[1], shape)
    if xs.size == 0:
        return base, base
    base[ys, xs] = True

    inner = morphology.binary_dilation(base, square_footprint(2 * radius + 1))
    outer = morphology.binary_dilation(base, square_footprint(2 * (radius + 2) + 1))
    surround = outer & (~inner)
    return inner, surround


def score_line(seg, norm, signal_norm, support_mask, edge_mask, profile):
    shape = norm.shape
    min_dim = profile["min_dim"]
    radius = max(1, int(round(min_dim / 170.0)))
    line_mask, surround_mask = line_corridor_and_surround(seg, shape, radius)

    if not line_mask.any():
        return None

    line_vals = norm[line_mask]
    surround_vals = norm[surround_mask] if surround_mask.any() else line_vals

    line_mean = float(line_vals.mean())
    surround_mean = float(surround_vals.mean())
    contrast_ratio = line_mean / max(surround_mean, 1e-6)
    contrast_score = float(np.clip((contrast_ratio - 1.02) / 0.55, 0.0, 1.0))

    line_signal = signal_norm[line_mask]
    surround_signal = signal_norm[surround_mask] if surround_mask.any() else line_signal
    sig_ratio = (float(line_signal.mean()) + 1e-6) / (float(surround_signal.mean()) + 1e-6)
    signal_score = float(np.clip((sig_ratio - 1.05) / 0.80, 0.0, 1.0))

    xs, ys = sample_line_pixels(seg[0], seg[1], shape)
    support_trace = support_mask[ys, xs]
    edge_trace = edge_mask[ys, xs] if edge_mask.any() else support_trace
    support_frac = float(support_trace.mean()) if support_trace.size else 0.0
    edge_frac = float(edge_trace.mean()) if edge_trace.size else 0.0

    gap_ratio = float(longest_false_run(support_trace) / max(len(support_trace), 1))
    continuity = float(np.clip(1.0 - gap_ratio, 0.0, 1.0))

    endpoint_score = endpoint_support_score(seg, support_mask | edge_mask)
    shape_score = main_diagonal_shape_score(seg[0], seg[1], shape)
    length = segment_length(seg)
    diag = max(1.0, math.hypot(shape[0], shape[1]))
    len_score = float(np.clip(length / max(5.0, 0.35 * diag), 0.0, 1.0))

    # Dense-matrices inspired main signal: line corridor should be stronger than surroundings.
    score = (
        0.26 * contrast_score
        + 0.18 * signal_score
        + 0.19 * continuity
        + 0.12 * support_frac
        + 0.06 * edge_frac
        + 0.10 * endpoint_score
        + 0.07 * shape_score
        + 0.02 * len_score
    )

    # Strongly penalize bridge-like segments.
    bridge_weight = 0.16 + 0.14 * float(np.clip(profile["stripe_strength"] / 1.0, 0.0, 1.0))
    score -= bridge_weight * gap_ratio

    return {
        "line": seg,
        "score": float(score),
        "contrast_score": contrast_score,
        "contrast_ratio": contrast_ratio,
        "signal_score": signal_score,
        "signal_ratio": sig_ratio,
        "continuity": continuity,
        "support_frac": support_frac,
        "edge_frac": edge_frac,
        "endpoint_score": endpoint_score,
        "shape_score": shape_score,
        "gap_ratio": gap_ratio,
        "length": length,
    }


# -----------------------------
# Candidate generation and selection
# -----------------------------
def generate_candidate_lines(work_mask, support_mask, profile, cfg):
    min_dim = profile["min_dim"]
    global_cap = candidate_budget(min_dim)
    per_pass_cap = int(cfg["max_hough_lines_per_pass"])
    min_len = float(cfg["min_segment_len_px"])

    rng = np.random.default_rng(0)
    work_u8 = work_mask.astype(np.uint8, copy=False)

    lines = []
    seen = set()
    for params in hough_param_grid(work_mask, profile):
        found = probabilistic_hough_line(
            work_u8,
            threshold=params["threshold"],
            line_length=params["line_length"],
            line_gap=params["line_gap"],
            rng=rng,
        )
        if not found:
            continue

        # Prefer longer candidates within each pass; they carry more structure.
        found = sorted(found, key=segment_length, reverse=True)[:per_pass_cap]
        for seg in found:
            pieces = split_line_on_support(seg, support_mask)
            use = pieces if pieces else [seg]
            for piece in use:
                if segment_length(piece) < min_len:
                    continue
                key = canonical_segment(piece)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(piece)
                if len(lines) >= global_cap:
                    return lines
    return lines


def dedupe_lines_by_score(scored, shape):
    if not scored:
        return []
    ranked = sorted(scored, key=lambda d: d["score"], reverse=True)
    kept = []
    for meta in ranked:
        if any(segments_are_similar(meta["line"], k["line"], shape) for k in kept):
            continue
        kept.append(meta)
    return kept


def dominant_component_id(seg, labels):
    if labels.size == 0:
        return 0, 0.0
    xs, ys = sample_line_pixels(seg[0], seg[1], labels.shape)
    vals = labels[ys, xs]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0, 0.0
    ids, counts = np.unique(vals, return_counts=True)
    idx = int(np.argmax(counts))
    comp_id = int(ids[idx])
    frac = float(counts[idx] / max(vals.size, 1))
    return comp_id, frac


def select_lines_residual(scored, support_mask, profile):
    if not scored:
        return []

    shape = support_mask.shape
    min_dim = profile["min_dim"]
    if min_dim <= 12:
        labels = np.zeros_like(support_mask, dtype=np.int32)
        component_count = 1 if support_mask.any() else 0
    else:
        labels = measure.label(support_mask, connectivity=2)
        sizes = np.bincount(labels.ravel()) if labels.max() > 0 else np.array([support_mask.size])
        min_comp = max(3, int(round(0.008 * support_mask.sum())))
        component_count = int(np.sum(sizes[1:] >= min_comp)) if sizes.size > 1 else 0

    max_lines = 1
    if min_dim >= 20:
        max_lines = 2
    if component_count >= 3 and min_dim >= 80:
        max_lines = 3
    if profile["stripe_strength"] > 0.75:
        max_lines = min(max_lines, 2)
    if min_dim <= 12:
        max_lines = 1

    # Adaptive acceptance floor from score distribution.
    scores = np.array([d["score"] for d in scored], dtype=float)
    base_floor = float(np.quantile(scores, 0.60))
    floor = base_floor
    floor += 0.10 * float(np.clip(profile["stripe_strength"] / 1.2, 0.0, 1.0))
    floor -= 0.08 * float(np.clip(profile["diag_contrast"], 0.0, 1.0))
    if min_dim <= 12:
        floor -= 0.10
    floor = float(np.clip(floor, 0.18, 0.80))

    radius = max(1, int(round(min_dim / 180.0)))
    covered = np.zeros(shape, dtype=bool)
    selected = []
    used_components = set()

    for meta in sorted(scored, key=lambda d: d["score"], reverse=True):
        if meta["score"] < floor:
            continue

        seg = meta["line"]
        seg_mask, _ = line_corridor_and_surround(seg, shape, radius)
        hit = seg_mask & support_mask
        if not hit.any():
            continue

        new_hit = hit & (~covered)
        gain = float(new_hit.sum() / max(int(hit.sum()), 1))

        # Residual gain gating: allows second line only when it explains
        # genuinely new ridge support.
        min_gain = 0.20
        min_gain += 0.28 * float(np.clip(profile["stripe_strength"] / 1.1, 0.0, 1.0))
        min_gain -= 0.10 * float(np.clip(meta["endpoint_score"], 0.0, 1.0))
        min_gain -= 0.06 * float(np.clip(profile["diag_contrast"], 0.0, 1.0))
        if meta["gap_ratio"] > 0.18:
            min_gain += 0.10
        min_gain = float(np.clip(min_gain, 0.10, 0.58))

        comp_id, comp_frac = dominant_component_id(seg, labels)
        if comp_id in used_components and gain < (min_gain + 0.15):
            continue
        if comp_id > 0 and comp_frac < 0.52 and gain < (min_gain + 0.10):
            continue

        if gain < min_gain:
            continue

        selected.append(meta)
        covered |= seg_mask
        if comp_id > 0:
            used_components.add(comp_id)
        if len(selected) >= max_lines:
            break

    if not selected and scored:
        best = max(scored, key=lambda d: d["score"])
        relaxed = 0.18 if min_dim <= 12 else 0.28
        if best["score"] >= relaxed and best["continuity"] >= 0.40:
            selected = [best]

    return selected


def endpoints_map_from_lines(lines, shape):
    out = np.zeros(shape, dtype=bool)
    if not lines:
        return out
    h, w = shape
    for p0, p1 in lines:
        for x, y in (p0, p1):
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            out[yi, xi] = True
    rad = max(1, int(round(min(shape) / 250.0)))
    if rad > 1:
        out = morphology.binary_dilation(out, square_footprint(2 * rad + 1))
    return out


def detect_lines(scores):
    matrix = safe_matrix(scores)
    norm = robust_normalize(matrix)
    profile = matrix_profile(norm)

    support_mask, edge_mask, work_mask, signal_norm = build_masks(norm, profile)
    if not support_mask.any():
        return {
            "matrix": matrix,
            "profile": profile,
            "strategy": "none",
            "final_lines": [],
            "confidence": 0.0,
            "objective": 0.0,
            "mask": support_mask,
            "edge_mask": edge_mask,
            "endpoint_map": np.zeros_like(matrix, dtype=bool),
            "line_meta": [],
        }

    raw_lines = generate_candidate_lines(work_mask, support_mask, profile, DETECTOR_CONFIG)
    if not raw_lines:
        return {
            "matrix": matrix,
            "profile": profile,
            "strategy": "none",
            "final_lines": [],
            "confidence": 0.0,
            "objective": 0.0,
            "mask": support_mask,
            "edge_mask": edge_mask,
            "endpoint_map": np.zeros_like(matrix, dtype=bool),
            "line_meta": [],
        }

    scored = []
    for seg in raw_lines:
        meta = score_line(seg, norm, signal_norm, support_mask, edge_mask, profile)
        if meta is not None:
            scored.append(meta)

    scored = dedupe_lines_by_score(scored, matrix.shape)
    selected = select_lines_residual(scored, support_mask, profile)

    if not selected:
        return {
            "matrix": matrix,
            "profile": profile,
            "strategy": "none",
            "final_lines": [],
            "confidence": 0.0,
            "objective": 0.0,
            "mask": support_mask,
            "edge_mask": edge_mask,
            "endpoint_map": np.zeros_like(matrix, dtype=bool),
            "line_meta": scored,
        }

    final_lines = [m["line"] for m in selected]
    objective = float(np.mean([m["score"] for m in selected]))
    confidence = float(np.clip(objective, 0.0, 1.0))

    return {
        "matrix": matrix,
        "profile": profile,
        "strategy": "simple_v1",
        "final_lines": final_lines,
        "confidence": confidence,
        "objective": objective,
        "mask": support_mask,
        "edge_mask": edge_mask,
        "endpoint_map": endpoints_map_from_lines(final_lines, matrix.shape),
        "line_meta": selected,
    }


# -----------------------------
# Plotting / saving
# -----------------------------
def build_figure(item, det, img_path):
    fig, (ax_img, ax_hm) = plt.subplots(1, 2, figsize=(12, 5))

    if img_path.exists():
        ax_img.imshow(plt.imread(img_path))
        ax_img.set_title(item["fname"])
        ax_img.axis("off")
    else:
        ax_img.text(0.5, 0.5, f"Missing image:\n{item['fname']}", ha="center", va="center")
        ax_img.axis("off")

    matrix = det["matrix"]
    im = ax_hm.imshow(matrix, aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, ax=ax_hm, label="chrF")
    ax_hm.set_xlabel("pred segment")
    ax_hm.set_ylabel("ref segment")

    for p0, p1 in det["final_lines"]:
        ax_hm.plot((p0[0], p1[0]), (p0[1], p1[1]), color="black", linewidth=2.4)

    ax_hm.set_title(
        f"mode={det['strategy']} | conf={det['confidence']:.3f} | "
        f"lines={len(det['final_lines'])}"
    )

    plt.tight_layout()
    return fig, ax_hm, cbar


def save_figure_outputs(fig, ax_hm, cbar, full_out, graph_out, full_dpi=220, graph_dpi=260):
    fig.savefig(full_out, dpi=full_dpi, bbox_inches="tight", facecolor="white")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    graph_bbox = Bbox.union([ax_hm.get_tightbbox(renderer), cbar.ax.get_tightbbox(renderer)]).transformed(
        fig.dpi_scale_trans.inverted()
    )
    fig.savefig(graph_out, dpi=graph_dpi, bbox_inches=graph_bbox, facecolor="white")


# -----------------------------
# Main loop
# -----------------------------
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

        det = detect_lines(item.get("scores"))
        img_path = IMG_DIR / Path(item["fname"]).name
        base_name = safe_name(item["fname"])

        full_after_out = FULL_AFTER_RGB_DIR / f"{saved_idx:04d}_{base_name}_full_after_rgb.png"
        graph_after_out = GRAPH_AFTER_RGB_DIR / f"{saved_idx:04d}_{base_name}_graph_after_rgb.png"

        mask_out = MASK_DIR / f"{saved_idx:04d}_{base_name}_ridge.png"
        edge_out = EDGE_DIR / f"{saved_idx:04d}_{base_name}_edge.png"
        endpoint_out = ENDPOINT_DIR / f"{saved_idx:04d}_{base_name}_endpoints.png"

        fig, ax_hm, cbar = build_figure(item=item, det=det, img_path=img_path)

        if SAVE_OUTPUTS:
            save_figure_outputs(fig, ax_hm, cbar, full_after_out, graph_after_out)
            plt.imsave(mask_out, det["mask"].astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
            plt.imsave(edge_out, det["edge_mask"].astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
            plt.imsave(endpoint_out, det["endpoint_map"].astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)

            print(f"Saved: {full_after_out}")
            print(f"Saved: {graph_after_out}")
            print(f"Saved: {mask_out}")
            print(f"Saved: {edge_out}")
            print(f"Saved: {endpoint_out}")

        if RENDER_NOTEBOOK_OUTPUT:
            plt.show()

            pred_esc = html.escape(item.get("pred", ""))
            ref_esc = html.escape(item.get("ref", ""))
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

        plt.close(fig)
        saved_idx += 1

print(f"Done. Saved {saved_idx} items to: {RESULTS_DIR}")
