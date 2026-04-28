"""Compute per-character line coverage arrays on reference and other axes.

The logic in this module is intentionally kept compatible with the existing v2.1
pipeline, while reusing shared helpers for line endpoint projection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from line_alignment_pipeline import derive_filtered_line_endpoints
from line_filtering_v2_1_IoU import DEFAULT_MIN_IOU_THRESHOLD
from runfile_records import load_run_items, same_file
from shared.project_line_to_text_windows import (
    line_window_ids_from_endpoint,
    num_windows_for_text_len,
    window_ids_to_merged_char_intervals,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone coverage counting."""
    parser = argparse.ArgumentParser(
        description=(
            "Count per-character text coverage induced by filtered alignment lines. "
            "This script can consume precomputed line endpoints or derive endpoints "
            "from runfile inputs using the shared alignment pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text-y", type=str, default=None, help="Reference text (Y axis).")
    parser.add_argument("--text-x", type=str, default=None, help="Other text (X axis).")
    parser.add_argument("--text-y-path", type=Path, default=None, help="Path to reference text file.")
    parser.add_argument("--text-x-path", type=Path, default=None, help="Path to other text file.")
    parser.add_argument(
        "--line-endpoints-json",
        type=Path,
        default=None,
        help="Optional JSON containing filtered line endpoints.",
    )
    parser.add_argument(
        "--runfile-json",
        type=Path,
        default=None,
        help="Optional outputs.json used to derive inputs for one target item.",
    )
    parser.add_argument(
        "--target-fname",
        type=str,
        default=None,
        help="Target file name used with --runfile-json.",
    )
    parser.add_argument("--window-size", type=int, default=100, help="Sliding window size.")
    parser.add_argument("--window-stride", type=int, default=50, help="Sliding window stride.")
    parser.add_argument("--hough-threshold", type=int, default=26, help="Hough vote threshold.")
    parser.add_argument("--hough-line-length", type=int, default=10, help="Minimum accepted line length.")
    parser.add_argument("--hough-line-gap", type=int, default=15, help="Maximum gap for connecting line pixels.")
    parser.add_argument("--hough-seed", type=int, default=0, help="Base Hough seed.")
    parser.add_argument("--hough-start", type=float, default=2.6, help="Initial adaptive threshold start.")
    parser.add_argument(
        "--align-abs-min-len",
        type=float,
        default=8.0,
        help="Absolute minimum line length kept before ownership filtering.",
    )
    parser.add_argument(
        "--align-min-iou-threshold",
        type=float,
        default=DEFAULT_MIN_IOU_THRESHOLD,
        help="Minimum true-IoU threshold used in v2.1 line filtering.",
    )
    parser.add_argument(
        "--strict-lines",
        action="store_true",
        help="Fail on malformed line endpoints. By default malformed lines are skipped.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Compatibility flag. Plot rendering is centralized elsewhere and ignored here.",
    )
    parser.add_argument("--visual-output", type=Path, default=None, help="Compatibility flag placeholder.")
    parser.add_argument("--visual-title", type=str, default=None, help="Compatibility flag placeholder.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    return parser.parse_args()


def _validate_numeric_inputs(
    *,
    window_size: int,
    window_stride: int,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_start: float,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
) -> None:
    """Validate numeric parameter ranges used by endpoint derivation."""
    if window_size <= 0 or window_stride <= 0:
        raise ValueError("window-size and window-stride must be positive integers")
    if hough_threshold <= 0:
        raise ValueError("hough-threshold must be positive")
    if hough_line_length <= 0:
        raise ValueError("hough-line-length must be positive")
    if hough_line_gap < 0:
        raise ValueError("hough-line-gap must be non-negative")
    if hough_start <= 0:
        raise ValueError("hough-start must be positive")
    if align_abs_min_len <= 0:
        raise ValueError("align-abs-min-len must be positive")
    if not (0.0 <= align_min_iou_threshold <= 1.0):
        raise ValueError("align-min-iou-threshold must satisfy 0.0 <= value <= 1.0")


def _validate_cli_inputs(args: argparse.Namespace) -> None:
    """Validate CLI argument combinations and numeric ranges."""
    _validate_numeric_inputs(
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        hough_threshold=int(args.hough_threshold),
        hough_line_length=int(args.hough_line_length),
        hough_line_gap=int(args.hough_line_gap),
        hough_start=float(args.hough_start),
        align_abs_min_len=float(args.align_abs_min_len),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
    )

    text_y_provided = args.text_y is not None or args.text_y_path is not None
    text_x_provided = args.text_x is not None or args.text_x_path is not None
    runfile_provided = args.runfile_json is not None

    if runfile_provided:
        if args.target_fname is None:
            raise ValueError("--target-fname is required when --runfile-json is used")
        if not Path(args.runfile_json).exists():
            raise FileNotFoundError(f"Missing runfile JSON: {args.runfile_json}")
    else:
        if not text_y_provided or not text_x_provided:
            raise ValueError(
                "Provide either --runfile-json/--target-fname or both Y/X texts via "
                "--text-y/--text-y-path and --text-x/--text-x-path"
            )


def read_text_input(raw_text: str | None, text_path: Path | None, *, label: str) -> str:
    """Read one text value either from inline argument or filesystem path."""
    if raw_text is not None:
        return str(raw_text)
    if text_path is None:
        raise ValueError(f"Missing {label} text: provide inline value or file path")
    if not Path(text_path).exists():
        raise FileNotFoundError(f"Missing {label} text file: {text_path}")
    return Path(text_path).read_text(encoding="utf-8")


def load_filtered_line_endpoints(line_endpoints_json: Path) -> list:
    """Load filtered line endpoints list from a JSON file."""
    if not Path(line_endpoints_json).exists():
        raise FileNotFoundError(f"Missing line endpoints JSON: {line_endpoints_json}")

    payload = json.loads(Path(line_endpoints_json).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "line_endpoints" in payload:
        endpoints = payload["line_endpoints"]
    else:
        endpoints = payload

    if not isinstance(endpoints, list):
        raise ValueError(f"Expected list of line endpoints in {line_endpoints_json}, got {type(endpoints).__name__}")
    return endpoints


def load_target_item_from_runfile(runfile_json: Path, target_fname: str) -> dict:
    """Select one runfile item by exact or basename match."""
    items = load_run_items(Path(runfile_json))
    matches = [item for item in items if same_file(item["fname"], target_fname)]
    if not matches:
        raise KeyError(f"Target file not found in runfile: {target_fname!r}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple runfile items matched target {target_fname!r}; "
            f"pass a more specific name. Matches: {[m['fname'] for m in matches]}"
        )
    return matches[0]


def derive_filtered_line_endpoints_from_text(
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
    """Derive filtered line endpoints with the shared alignment pipeline."""
    return derive_filtered_line_endpoints(
        ref_text=ref_text,
        pred_text=pred_text,
        item_index=int(item_index),
        window_size=int(window_size),
        window_stride=int(window_stride),
        hough_threshold=int(hough_threshold),
        hough_line_length=int(hough_line_length),
        hough_line_gap=int(hough_line_gap),
        hough_seed=int(hough_seed),
        hough_start=float(hough_start),
        align_abs_min_len=float(align_abs_min_len),
        align_min_iou_threshold=float(align_min_iou_threshold),
    )


def _increment_counts(counts: np.ndarray, intervals: list[tuple[int, int]]) -> None:
    """Add +1 coverage to each ``[start:end)`` interval in the count array."""
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
    """Compute Y/X per-character coverage arrays from filtered line endpoints."""
    if int(window_size) <= 0 or int(window_stride) <= 0:
        raise ValueError("window_size and window_stride must be positive integers")

    y_counts = np.zeros(len(text_y), dtype=np.int32)
    x_counts = np.zeros(len(text_x), dtype=np.int32)

    n_y_windows = num_windows_for_text_len(len(text_y), int(window_size), int(window_stride))
    n_x_windows = num_windows_for_text_len(len(text_x), int(window_size), int(window_stride))

    for idx, line in enumerate(line_endpoints):
        try:
            x_ids, y_ids = line_window_ids_from_endpoint(
                line,
                n_x_windows=int(n_x_windows),
                n_y_windows=int(n_y_windows),
            )
        except Exception:
            if strict_lines:
                raise ValueError(f"Malformed line endpoint at index {idx}: {line!r}")
            continue

        y_intervals = window_ids_to_merged_char_intervals(
            y_ids,
            text_len=len(text_y),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )
        x_intervals = window_ids_to_merged_char_intervals(
            x_ids,
            text_len=len(text_x),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )

        _increment_counts(y_counts, y_intervals)
        _increment_counts(x_counts, x_intervals)

    return y_counts, x_counts


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
    """Serialize coverage arrays and metadata to JSON."""
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text_y_len": int(text_y_len),
        "text_x_len": int(text_x_len),
        "line_count": int(line_count),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "y_counts": y_counts.tolist(),
        "x_counts": x_counts.tolist(),
    }
    Path(output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    """Compute y/x coverage arrays and optionally persist compatibility artifacts."""
    y_counts, x_counts = _compute_line_character_coverage(
        text_y=text_y,
        text_x=text_x,
        line_endpoints=line_endpoints,
        window_size=int(window_size),
        window_stride=int(window_stride),
        strict_lines=bool(strict_lines),
    )

    # Plotting is centralized in visualise_used_lines_from_report.py.
    # Keep compatibility flags accepted but intentionally unused here.
    _ = (visualize, visual_output, visual_title)

    if output_json is not None:
        _write_output_json(
            Path(output_json),
            text_y_len=len(text_y),
            text_x_len=len(text_x),
            line_count=len(line_endpoints),
            window_size=int(window_size),
            window_stride=int(window_stride),
            y_counts=y_counts,
            x_counts=x_counts,
        )

    return y_counts, x_counts


def main() -> None:
    """Run standalone CLI flow for line character coverage counting."""
    args = parse_args()
    _validate_cli_inputs(args)

    if args.runfile_json is not None:
        target = load_target_item_from_runfile(Path(args.runfile_json), str(args.target_fname))
        text_y = str(target["ref"])
        text_x = str(target["pred"])
        item_index = int(target["index"])
    else:
        text_y = read_text_input(args.text_y, args.text_y_path, label="Y")
        text_x = read_text_input(args.text_x, args.text_x_path, label="X")
        item_index = 0

    if args.line_endpoints_json is not None:
        line_endpoints = load_filtered_line_endpoints(Path(args.line_endpoints_json))
    else:
        line_endpoints, _ = derive_filtered_line_endpoints_from_text(
            ref_text=text_y,
            pred_text=text_x,
            item_index=int(item_index),
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

    payload = {
        "text_y_len": int(len(text_y)),
        "text_x_len": int(len(text_x)),
        "line_count": int(len(line_endpoints)),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "y_counts": y_counts.tolist(),
        "x_counts": x_counts.tolist(),
    }
    print(json.dumps(payload, ensure_ascii=False))


# Backward-compatible aliases retained for existing callsites.
_read_text = read_text_input
_load_filtered_line_endpoints = load_filtered_line_endpoints
_load_target_item_from_runfile = load_target_item_from_runfile
_derive_filtered_line_endpoints = derive_filtered_line_endpoints_from_text


if __name__ == "__main__":
    main()
