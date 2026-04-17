import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from align_text_blocks_from_endpoints_no_pkl_v2 import (
    compute_score_matrix,
    lines_from_merged_segments,
    load_run_items,
    same_file,
)
from hough_line_transform_endpoints_no_angle_all import detect_lines_dense_style_diagonal_fixed_theta
from line_filtering_v2_1_IoU import DEFAULT_MIN_IOU_THRESHOLD, filter_lines_for_alignment_by_ownership

# Expose only the public API function for other scripts.
__all__ = ["count_text_on_lne"]


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for coverage counting.

    Supports two execution modes:
    1) Direct mode: provide text_y/text_x and line endpoints JSON.
    2) Runfile mode: provide runfile JSON + target-fname, and optionally line endpoints JSON.
       If endpoints JSON is omitted, endpoints are derived via Hough + v2.1 IoU filtering.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-character line coverage for two texts using filtered line endpoints. "
            "Can run in direct text mode or auto-derive endpoints from outputs.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Direct mode text sources.
    text_y_group = parser.add_mutually_exclusive_group(required=False)
    text_y_group.add_argument("--text-y", type=str, default=None, help="First text (y-axis text)")
    text_y_group.add_argument("--text-y-path", type=Path, default=None, help="Path to first text file")

    text_x_group = parser.add_mutually_exclusive_group(required=False)
    text_x_group.add_argument("--text-x", type=str, default=None, help="Second text (x-axis text)")
    text_x_group.add_argument("--text-x-path", type=Path, default=None, help="Path to second text file")

    # Runfile mode.
    parser.add_argument(
        "--runfile-json",
        type=Path,
        default=None,
        help="Optional outputs.json path. Use with --target-fname to load one example.",
    )
    parser.add_argument(
        "--target-fname",
        type=str,
        default=None,
        help="Target file name for runfile mode (exact or basename match).",
    )

    parser.add_argument(
        "--line-endpoints-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON containing filtered lines. Accepted shapes: "
            "list of lines, or dict containing one of keys: lines_used, line_endpoints, lines, filtered_lines. "
            "If omitted in runfile mode, endpoints are auto-derived."
        ),
    )

    parser.add_argument("--window-size", type=int, default=100, help="Window size used by score matrices")
    parser.add_argument("--window-stride", type=int, default=50, help="Window stride used by score matrices")

    # Endpoint auto-derivation parameters (used only when endpoints JSON is omitted in runfile mode).
    parser.add_argument("--hough-threshold", type=int, default=26, help="Hough vote threshold")
    parser.add_argument("--hough-line-length", type=int, default=10, help="Minimum accepted line length")
    parser.add_argument("--hough-line-gap", type=int, default=15, help="Maximum gap to connect line pixels")
    parser.add_argument("--hough-seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--hough-start", type=float, default=2.6, help="Initial adaptive threshold start")
    parser.add_argument(
        "--align-abs-min-len",
        type=float,
        default=8.0,
        help="Absolute minimum line length kept before ownership resolution.",
    )
    parser.add_argument(
        "--align-min-iou-threshold",
        type=float,
        default=DEFAULT_MIN_IOU_THRESHOLD,
        help="Minimum true-IoU threshold used to merge overlapping line coverages in v2.1_true_IoU.",
    )

    parser.add_argument(
        "--strict-lines",
        action="store_true",
        help="Fail on malformed line endpoints. By default malformed lines are skipped.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate optional character coverage visualization for y and x axes.",
    )
    parser.add_argument(
        "--visual-output",
        type=Path,
        default=None,
        help="Optional PNG output path for visualization. If omitted with --visualize, an interactive plot is shown.",
    )
    parser.add_argument(
        "--visual-title",
        type=str,
        default=None,
        help="Optional figure title for the visualization.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path to save y_counts/x_counts arrays.",
    )
    parser.add_argument(
        "--print-arrays",
        action="store_true",
        help="Print full y_counts and x_counts arrays to stdout.",
    )
    return parser.parse_args()


def _read_text(raw_text: str | None, text_path: Path | None, *, label: str) -> str:
    """Resolve one text value from either inline input or file path."""
    if raw_text is not None:
        return str(raw_text)
    if text_path is None:
        raise ValueError(f"Missing {label}: provide either inline text or file path")
    if not text_path.exists():
        raise FileNotFoundError(f"Missing {label} file: {text_path}")
    return text_path.read_text(encoding="utf-8")


def _load_filtered_line_endpoints(line_endpoints_json: Path) -> list:
    """Load filtered line endpoints from JSON list or known wrapper keys."""
    if not line_endpoints_json.exists():
        raise FileNotFoundError(f"Missing line endpoints JSON: {line_endpoints_json}")

    payload = json.loads(line_endpoints_json.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("lines_used", "line_endpoints", "lines", "filtered_lines"):
            lines = payload.get(key)
            if isinstance(lines, list):
                return lines

    raise ValueError(
        "line-endpoints JSON must be a list, or a dict containing one of: "
        "lines_used, line_endpoints, lines, filtered_lines"
    )


def _load_target_item_from_runfile(runfile_json: Path, target_fname: str) -> dict:
    """Load one target item from outputs.json using exact/basename matching."""
    if not runfile_json.exists():
        raise FileNotFoundError(f"Missing runfile JSON: {runfile_json}")

    run_items = load_run_items(runfile_json)
    matches = [item for item in run_items if same_file(item["fname"], target_fname)]
    if not matches:
        raise KeyError(f"Target file not found in runfile: {target_fname!r}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple runfile items matched target {target_fname!r}; pass a more specific name. "
            f"Matches: {[m['fname'] for m in matches]}"
        )
    return matches[0]


def _derive_filtered_line_endpoints(
    *,
    ref_text: str,
    pred_text: str,
    item_index: int,
    window_size: int,
    window_stride: int,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    hough_start: float,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
) -> tuple[list[dict], dict]:
    """Derive filtered line endpoints with the same core logic as text_metrics_report.py."""
    matrix = compute_score_matrix(
        ref_text,
        pred_text,
        window_size=window_size,
        window_stride=window_stride,
    )

    if matrix.size > 0 and matrix.shape[0] > 0 and matrix.shape[1] > 0:
        det = detect_lines_dense_style_diagonal_fixed_theta(
            matrix,
            seed=int(hough_seed) + int(item_index),
            threshold=int(hough_threshold),
            line_length=int(hough_line_length),
            line_gap=int(hough_line_gap),
            start_init=float(hough_start),
        )
        merged_lines = det.get("merged_lines", [])
        lines_for_filtering = lines_from_merged_segments(matrix, merged_lines)
        mask_bool = np.asarray(det.get("mask", np.zeros_like(matrix))) > 0
        lines_used, _ = filter_lines_for_alignment_by_ownership(
            lines_for_filtering,
            matrix,
            mask_bool,
            abs_min_len=float(align_abs_min_len),
            min_iou_threshold=float(align_min_iou_threshold),
        )
        endpoint_debug = {
            "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "raw_line_count": int(len(det.get("raw_lines", []))),
            "merged_line_count": int(len(lines_for_filtering)),
            "used_line_count": int(len(lines_used)),
            "threshold_start": float(det.get("threshold_start", float("nan"))),
            "hough_threshold": int(hough_threshold),
            "hough_line_length": int(hough_line_length),
            "hough_line_gap": int(hough_line_gap),
            "hough_seed": int(hough_seed) + int(item_index),
            "hough_start": float(hough_start),
            "align_abs_min_len": float(align_abs_min_len),
            "align_min_iou_threshold": float(align_min_iou_threshold),
        }
        return lines_used, endpoint_debug

    return [], {
        "matrix_shape": [int(matrix.shape[0]) if matrix.ndim == 2 else 0, int(matrix.shape[1]) if matrix.ndim == 2 else 0],
        "raw_line_count": 0,
        "merged_line_count": 0,
        "used_line_count": 0,
        "threshold_start": float("nan"),
        "hough_threshold": int(hough_threshold),
        "hough_line_length": int(hough_line_length),
        "hough_line_gap": int(hough_line_gap),
        "hough_seed": int(hough_seed) + int(item_index),
        "hough_start": float(hough_start),
        "align_abs_min_len": float(align_abs_min_len),
        "align_min_iou_threshold": float(align_min_iou_threshold),
    }


def _num_windows(text_len: int, window_size: int, window_stride: int) -> int:
    """Return number of sliding windows for a text length and window config."""
    if text_len < window_size:
        return 0
    return ((text_len - window_size) // window_stride) + 1


def _parse_point_xy(point) -> tuple[float, float]:
    """Convert a point-like input into an `(x, y)` float pair."""
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Invalid point format: {point!r}")
    return float(arr[0]), float(arr[1])


def _normalize_line_endpoints(line) -> tuple[tuple[float, float], tuple[float, float]]:
    """Normalize one line into endpoint form `((x0, y0), (x1, y1))`."""
    if isinstance(line, dict):
        try:
            return (
                (float(line["x0"]), float(line["y0"])),
                (float(line["x1"]), float(line["y1"])),
            )
        except KeyError as exc:
            raise ValueError(f"Line dict missing endpoint key: {exc}") from exc

    if isinstance(line, (list, tuple)) and len(line) == 2:
        p0, p1 = line
        x0, y0 = _parse_point_xy(p0)
        x1, y1 = _parse_point_xy(p1)
        return (x0, y0), (x1, y1)

    raise ValueError(f"Unsupported line format: {line!r}")


def _line_window_indices(
    line, n_x_windows: int, n_y_windows: int
) -> tuple[set[int], set[int]]:
    """Sample a line and return covered window ids on x-axis and y-axis."""
    if n_x_windows <= 0 and n_y_windows <= 0:
        return set(), set()

    (x0, y0), (x1, y1) = _normalize_line_endpoints(line)

    # Use max(dx,dy) stepping to ensure coverage along both shallow and steep lines.
    n_steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    n_steps = max(n_steps, 1)

    xs = np.rint(np.linspace(x0, x1, n_steps)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n_steps)).astype(int)

    x_set: set[int] = set()
    y_set: set[int] = set()

    # Clamp to valid matrix ranges so partially out-of-bounds lines remain usable.
    if n_x_windows > 0:
        xs = np.clip(xs, 0, n_x_windows - 1)
        x_set = {int(v) for v in xs.tolist()}
    if n_y_windows > 0:
        ys = np.clip(ys, 0, n_y_windows - 1)
        y_set = {int(v) for v in ys.tolist()}

    return x_set, y_set


def _window_indices_to_merged_char_intervals(
    indices: set[int], text_len: int, window_size: int, window_stride: int
) -> list[tuple[int, int]]:
    """Map window ids to merged character intervals in the original text."""
    if not indices or text_len <= 0:
        return []

    raw: list[tuple[int, int]] = []
    for idx in sorted(indices):
        start = idx * window_stride
        end = min(start + window_size, text_len)
        if start >= text_len or end <= start:
            continue
        raw.append((start, end))

    if not raw:
        return []

    # Merge overlapping/adjacent windows so each character is incremented once per line.
    merged: list[tuple[int, int]] = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _increment_counts(counts: np.ndarray, intervals: list[tuple[int, int]]) -> None:
    """Add +1 coverage to each `[start:end)` interval in the counts array."""
    for start, end in intervals:
        counts[start:end] += 1


def _compute_line_character_coverage(
    text_y: str,
    text_x: str,
    line_endpoints: list,
    *,
    window_size: int,
    window_stride: int,
    strict_lines: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-character coverage arrays for y-text and x-text."""
    if window_size <= 0 or window_stride <= 0:
        raise ValueError("window_size and window_stride must be positive integers")

    y_counts = np.zeros(len(text_y), dtype=np.int32)
    x_counts = np.zeros(len(text_x), dtype=np.int32)

    n_y_windows = _num_windows(len(text_y), window_size, window_stride)
    n_x_windows = _num_windows(len(text_x), window_size, window_stride)

    for idx, line in enumerate(line_endpoints):
        try:
            x_indices, y_indices = _line_window_indices(line, n_x_windows, n_y_windows)
        except Exception:
            if strict_lines:
                raise ValueError(f"Malformed line endpoint at index {idx}: {line!r}")
            continue

        y_intervals = _window_indices_to_merged_char_intervals(
            y_indices,
            len(text_y),
            window_size,
            window_stride,
        )
        x_intervals = _window_indices_to_merged_char_intervals(
            x_indices,
            len(text_x),
            window_size,
            window_stride,
        )

        _increment_counts(y_counts, y_intervals)
        _increment_counts(x_counts, x_intervals)

    return y_counts, x_counts


def _plot_line_character_coverage(
    y_counts: np.ndarray,
    x_counts: np.ndarray,
    *,
    output_path: Path | None = None,
    title: str | None = None,
) -> Path | None:
    """Visualize y-axis and x-axis character coverage curves."""
    import matplotlib.pyplot as plt

    fig, (ax_y, ax_x) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    ax_y.plot(np.arange(y_counts.size), y_counts, color="tab:blue", linewidth=1.2)
    ax_y.set_title("Y-axis text character coverage")
    ax_y.set_xlabel("Character index")
    ax_y.set_ylabel("Coverage count")
    ax_y.grid(alpha=0.25)

    ax_x.plot(np.arange(x_counts.size), x_counts, color="tab:orange", linewidth=1.2)
    ax_x.set_title("X-axis text character coverage")
    ax_x.set_xlabel("Character index")
    ax_x.set_ylabel("Coverage count")
    ax_x.grid(alpha=0.25)

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    saved_path: Path | None = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
        saved_path = output_path
    else:
        plt.show()

    plt.close(fig)
    return saved_path


def _write_output_json(
    output_json: Path,
    *,
    text_y_len: int,
    text_x_len: int,
    line_count: int,
    window_size: int,
    window_stride: int,
    y_counts: np.ndarray,
    x_counts: np.ndarray,
) -> None:
    """Serialize coverage arrays and run metadata to a JSON file."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text_y_len": int(text_y_len),
        "text_x_len": int(text_x_len),
        "line_count": int(line_count),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "y_counts": y_counts.tolist(),
        "x_counts": x_counts.tolist(),
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def count_text_on_lne(
    text_y: str,
    text_x: str,
    line_endpoints: list,
    *,
    window_size: int,
    window_stride: int,
    visualize: bool = False,
    visual_output: Path | None = None,
    visual_title: str | None = None,
    strict_lines: bool = False,
    output_json: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Public entrypoint for line-based text coverage counting.

    This function computes y/x character coverage from filtered line endpoints,
    and can optionally render a plot and/or write a JSON artifact.
    """
    y_counts, x_counts = _compute_line_character_coverage(
        text_y=text_y,
        text_x=text_x,
        line_endpoints=line_endpoints,
        window_size=window_size,
        window_stride=window_stride,
        strict_lines=strict_lines,
    )

    if visualize:
        _plot_line_character_coverage(
            y_counts=y_counts,
            x_counts=x_counts,
            output_path=visual_output,
            title=visual_title,
        )

    if output_json is not None:
        _write_output_json(
            output_json,
            text_y_len=len(text_y),
            text_x_len=len(text_x),
            line_count=len(line_endpoints),
            window_size=window_size,
            window_stride=window_stride,
            y_counts=y_counts,
            x_counts=x_counts,
        )

    return y_counts, x_counts


def main() -> None:
    """CLI wrapper supporting direct mode and runfile+auto-endpoint mode."""
    args = parse_args()

    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("window-size and window-stride must be positive integers")
    if args.hough_threshold <= 0:
        raise ValueError("hough-threshold must be positive")
    if args.hough_line_length <= 0:
        raise ValueError("hough-line-length must be positive")
    if args.hough_line_gap < 0:
        raise ValueError("hough-line-gap must be non-negative")
    if args.hough_start <= 0:
        raise ValueError("hough-start must be positive")
    if args.align_abs_min_len <= 0:
        raise ValueError("align-abs-min-len must be positive")
    if not (0.0 <= args.align_min_iou_threshold <= 1.0):
        raise ValueError("align-min-iou-threshold must satisfy 0.0 <= value <= 1.0")

    has_runfile_mode = args.runfile_json is not None
    has_direct_text_y = args.text_y is not None or args.text_y_path is not None
    has_direct_text_x = args.text_x is not None or args.text_x_path is not None

    endpoint_debug: dict | None = None

    if has_runfile_mode:
        if has_direct_text_y or has_direct_text_x:
            raise ValueError("Do not combine runfile mode with direct text arguments")
        if args.target_fname is None:
            raise ValueError("--target-fname is required when using --runfile-json")

        target_item = _load_target_item_from_runfile(args.runfile_json, args.target_fname)
        text_y = str(target_item["ref"])
        text_x = str(target_item["pred"])
        item_index = int(target_item["index"])

        if args.line_endpoints_json is not None:
            line_endpoints = _load_filtered_line_endpoints(args.line_endpoints_json)
        else:
            line_endpoints, endpoint_debug = _derive_filtered_line_endpoints(
                ref_text=text_y,
                pred_text=text_x,
                item_index=item_index,
                window_size=int(args.window_size),
                window_stride=int(args.window_stride),
                hough_threshold=int(args.hough_threshold),
                hough_line_length=int(args.hough_line_length),
                hough_line_gap=int(args.hough_line_gap),
                hough_seed=int(args.hough_seed),
                hough_start=float(args.hough_start),
                align_abs_min_len=float(args.align_abs_min_len),
                align_min_iou_threshold=float(args.align_min_iou_threshold),
            )
    else:
        if not has_direct_text_y or not has_direct_text_x:
            raise ValueError(
                "Provide either runfile mode (--runfile-json + --target-fname) "
                "or direct text mode (--text-y/--text-y-path and --text-x/--text-x-path)."
            )
        if args.line_endpoints_json is None:
            raise ValueError(
                "In direct text mode, --line-endpoints-json is required. "
                "(Runfile mode can auto-derive endpoints when this is omitted.)"
            )

        text_y = _read_text(args.text_y, args.text_y_path, label="text-y")
        text_x = _read_text(args.text_x, args.text_x_path, label="text-x")
        line_endpoints = _load_filtered_line_endpoints(args.line_endpoints_json)

    y_counts, x_counts = count_text_on_lne(
        text_y=text_y,
        text_x=text_x,
        line_endpoints=line_endpoints,
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        visualize=bool(args.visualize),
        visual_output=args.visual_output,
        visual_title=args.visual_title,
        strict_lines=bool(args.strict_lines),
        output_json=args.output_json,
    )

    print(f"text_y_len={len(text_y)} text_x_len={len(text_x)}")
    print(f"line_count={len(line_endpoints)}")
    print(f"y_nonzero={int(np.count_nonzero(y_counts))} x_nonzero={int(np.count_nonzero(x_counts))}")

    if endpoint_debug is not None:
        print("auto_derived_line_endpoints=true")
        print(
            "derived_lines: "
            f"raw={endpoint_debug['raw_line_count']} "
            f"merged={endpoint_debug['merged_line_count']} "
            f"used={endpoint_debug['used_line_count']} "
            f"matrix_shape={endpoint_debug['matrix_shape'][0]}x{endpoint_debug['matrix_shape'][1]}"
        )

    if args.print_arrays:
        np.set_printoptions(threshold=np.inf, linewidth=180)
        print("y_counts:")
        print(y_counts)
        print("x_counts:")
        print(x_counts)


if __name__ == "__main__":
    main()
