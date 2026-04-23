import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from line_metric_bundle import accumulate_counts_from_interval_groups
from runfile_records import load_run_items, same_file
from count_line_character_coverage import (
    _derive_filtered_line_endpoints,
    _load_filtered_line_endpoints,
    count_text_on_lne,
)

__all__ = [
    "compute_line_coverage_percentage_metrics_from_precomputed_endpoints",
    "compute_line_coverage_percentage_metrics_from_bundles",
    "compute_line_coverage_percentage_metrics",
    "compute_y_axis_coverage_percentage_metrics",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute line-coverage percentage metrics for one target document. "
            "By default, uses precomputed endpoints JSONs (no endpoint recomputation)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    parser.add_argument(
        "--target-fname",
        type=str,
        required=True,
        help="Target file name (exact or basename match).",
    )
    parser.add_argument(
        "--other-text-path",
        type=Path,
        default=None,
        help="Optional path to other text. If omitted, uses target prediction text from runfile.",
    )
    parser.add_argument(
        "--refref-line-endpoints-json",
        type=Path,
        default=None,
        help="Filtered line endpoints JSON for ref->ref call.",
    )
    parser.add_argument(
        "--other-line-endpoints-json",
        type=Path,
        default=None,
        help="Filtered line endpoints JSON for ref->other call.",
    )
    parser.add_argument(
        "--allow-derive-endpoints",
        action="store_true",
        help="Allow deriving endpoints if endpoint JSONs are not provided.",
    )
    parser.add_argument("--window-size", type=int, default=100, help="Window size")
    parser.add_argument("--window-stride", type=int, default=50, help="Window stride")
    parser.add_argument("--hough-threshold", type=int, default=26, help="Hough vote threshold")
    parser.add_argument("--hough-line-length", type=int, default=10, help="Hough minimum line length")
    parser.add_argument("--hough-line-gap", type=int, default=15, help="Hough maximum line gap")
    parser.add_argument("--hough-seed", type=int, default=0, help="Base Hough seed")
    parser.add_argument("--hough-start", type=float, default=2.6, help="Initial adaptive threshold start")
    parser.add_argument("--align-abs-min-len", type=float, default=8.0, help="Line filter absolute min length")
    parser.add_argument(
        "--align-min-iou-threshold",
        type=float,
        default=0.035,
        help="Line filter minimum true-IoU threshold",
    )
    parser.add_argument(
        "--strict-lines",
        action="store_true",
        help="Fail on malformed line endpoints. By default malformed lines are skipped.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path")
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

    if not bool(args.allow_derive_endpoints):
        if args.refref_line_endpoints_json is None or args.other_line_endpoints_json is None:
            raise ValueError(
                "Precomputed endpoints are required by default. Provide both "
                "--refref-line-endpoints-json and --other-line-endpoints-json, "
                "or use --allow-derive-endpoints to enable derivation."
            )


def _load_target_item(runfile_json: Path, target_fname: str) -> dict:
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


def _load_other_text(item: dict, other_text_path: Path | None) -> str:
    if other_text_path is None:
        return str(item["pred"])
    if not other_text_path.exists():
        raise FileNotFoundError(f"Missing other text file: {other_text_path}")
    return other_text_path.read_text(encoding="utf-8")


def _resolve_line_endpoints(
    *,
    endpoint_json: Path | None,
    ref_text: str,
    pred_text: str,
    item_index: int,
    args: argparse.Namespace,
) -> list:
    if endpoint_json is not None:
        return _load_filtered_line_endpoints(endpoint_json)

    if not bool(args.allow_derive_endpoints):
        raise ValueError(
            "Endpoint JSON not provided and endpoint derivation is disabled. "
            "Pass endpoint JSONs or enable --allow-derive-endpoints."
        )

    lines_used, _ = _derive_filtered_line_endpoints(
        ref_text=ref_text,
        pred_text=pred_text,
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
    return lines_used


def _compute_y_axis_percentage_metrics(y_diff: np.ndarray) -> dict:
    total_chars = int(y_diff.size)
    if total_chars == 0:
        return {
            "missing_percent": 0.0,
            "ok_percent": 0.0,
            "repetition_percent": 0.0,
        }

    missing_count = int(np.count_nonzero(y_diff == -1))
    ok_count = int(np.count_nonzero(y_diff == 0))
    repetition_count = int(np.count_nonzero(y_diff > 0))

    covered_count = missing_count + ok_count + repetition_count
    if covered_count != total_chars:
        unknown_count = total_chars - covered_count
        raise ValueError(
            "Found y-axis subtraction values outside defined categories "
            "(-1, 0, >0). "
            f"unknown_count={unknown_count}"
        )

    return {
        "missing_percent": float((missing_count / total_chars) * 100.0),
        "ok_percent": float((ok_count / total_chars) * 100.0),
        "repetition_percent": float((repetition_count / total_chars) * 100.0),
    }


def _compute_x_axis_hallucination_percent(other_x: np.ndarray) -> float:
    total_chars = int(other_x.size)
    if total_chars == 0:
        return 0.0
    hallucination_count = int(np.count_nonzero(other_x == 0))
    return float((hallucination_count / total_chars) * 100.0)


def _interval_groups_from_bundle(bundle: dict, key: str) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    for line in bundle.get("lines", []):
        intervals = line.get(key, [])
        groups.append([(int(start), int(end)) for start, end in intervals])
    return groups


def compute_line_coverage_percentage_metrics_from_bundles(
    *,
    refref_bundle: dict,
    other_bundle: dict,
    file_name: str | None = None,
) -> dict:
    ref_text_len = int(refref_bundle.get("ref_text_len", 0))
    other_text_len = int(other_bundle.get("other_text_len", 0))

    refref_y = accumulate_counts_from_interval_groups(
        text_len=ref_text_len,
        interval_groups=_interval_groups_from_bundle(refref_bundle, "y_char_intervals_coverage_legacy"),
    )
    other_y = accumulate_counts_from_interval_groups(
        text_len=ref_text_len,
        interval_groups=_interval_groups_from_bundle(other_bundle, "y_char_intervals_coverage_legacy"),
    )
    other_x = accumulate_counts_from_interval_groups(
        text_len=other_text_len,
        interval_groups=_interval_groups_from_bundle(other_bundle, "x_char_intervals_coverage_legacy"),
    )

    if refref_y.shape != other_y.shape:
        raise ValueError(
            f"Reference-axis counts must have same length, got {refref_y.shape[0]} and {other_y.shape[0]}"
        )

    y_diff = np.subtract(other_y, refref_y)
    metrics = _compute_y_axis_percentage_metrics(y_diff)
    metrics["hallucination_percent"] = _compute_x_axis_hallucination_percent(other_x)
    if file_name is not None:
        metrics["file_name"] = str(file_name)
    return metrics


def compute_line_coverage_percentage_metrics_from_precomputed_endpoints(
    *,
    ref_text: str,
    other_text: str,
    refref_line_endpoints: list,
    other_line_endpoints: list,
    window_size: int = 100,
    window_stride: int = 50,
    strict_lines: bool = False,
    file_name: str | None = None,
) -> dict:
    """Compute percentage metrics without recomputing endpoints.

    `refref_line_endpoints` and `other_line_endpoints` must be precomputed.
    """
    if window_size <= 0 or window_stride <= 0:
        raise ValueError("window-size and window-stride must be positive integers")

    refref_y, _ = count_text_on_lne(
        text_y=ref_text,
        text_x=ref_text,
        line_endpoints=refref_line_endpoints,
        window_size=int(window_size),
        window_stride=int(window_stride),
        strict_lines=bool(strict_lines),
    )
    other_y, other_x = count_text_on_lne(
        text_y=ref_text,
        text_x=other_text,
        line_endpoints=other_line_endpoints,
        window_size=int(window_size),
        window_stride=int(window_stride),
        strict_lines=bool(strict_lines),
    )

    refref_y = np.asarray(refref_y, dtype=np.int32)
    other_y = np.asarray(other_y, dtype=np.int32)
    other_x = np.asarray(other_x, dtype=np.int32)

    if refref_y.shape != other_y.shape:
        raise ValueError(
            f"Reference-axis counts must have same length, got {refref_y.shape[0]} and {other_y.shape[0]}"
        )

    y_diff = np.subtract(other_y, refref_y)

    metrics = _compute_y_axis_percentage_metrics(y_diff)
    metrics["hallucination_percent"] = _compute_x_axis_hallucination_percent(other_x)
    if file_name is not None:
        metrics["file_name"] = str(file_name)
    return metrics


def compute_line_coverage_percentage_metrics(
    *,
    runfile_json: Path,
    target_fname: str,
    other_text_path: Path | None = None,
    refref_line_endpoints_json: Path | None = None,
    other_line_endpoints_json: Path | None = None,
    allow_derive_endpoints: bool = False,
    window_size: int = 100,
    window_stride: int = 50,
    hough_threshold: int = 26,
    hough_line_length: int = 10,
    hough_line_gap: int = 15,
    hough_seed: int = 0,
    hough_start: float = 2.6,
    align_abs_min_len: float = 8.0,
    align_min_iou_threshold: float = 0.035,
    strict_lines: bool = False,
) -> dict:
    """Compute percentage metrics for one runfile target.

    By default, this does not derive endpoints.
    """
    _validate_numeric_inputs(
        window_size=int(window_size),
        window_stride=int(window_stride),
        hough_threshold=int(hough_threshold),
        hough_line_length=int(hough_line_length),
        hough_line_gap=int(hough_line_gap),
        hough_start=float(hough_start),
        align_abs_min_len=float(align_abs_min_len),
        align_min_iou_threshold=float(align_min_iou_threshold),
    )

    args = argparse.Namespace(
        runfile_json=Path(runfile_json),
        target_fname=str(target_fname),
        other_text_path=None if other_text_path is None else Path(other_text_path),
        refref_line_endpoints_json=(
            None if refref_line_endpoints_json is None else Path(refref_line_endpoints_json)
        ),
        other_line_endpoints_json=(
            None if other_line_endpoints_json is None else Path(other_line_endpoints_json)
        ),
        allow_derive_endpoints=bool(allow_derive_endpoints),
        window_size=int(window_size),
        window_stride=int(window_stride),
        hough_threshold=int(hough_threshold),
        hough_line_length=int(hough_line_length),
        hough_line_gap=int(hough_line_gap),
        hough_seed=int(hough_seed),
        hough_start=float(hough_start),
        align_abs_min_len=float(align_abs_min_len),
        align_min_iou_threshold=float(align_min_iou_threshold),
        strict_lines=bool(strict_lines),
    )

    item = _load_target_item(args.runfile_json, args.target_fname)
    ref_text = str(item["ref"])
    other_text = _load_other_text(item, args.other_text_path)

    refref_line_endpoints = _resolve_line_endpoints(
        endpoint_json=args.refref_line_endpoints_json,
        ref_text=ref_text,
        pred_text=ref_text,
        item_index=int(item["index"]),
        args=args,
    )
    other_line_endpoints = _resolve_line_endpoints(
        endpoint_json=args.other_line_endpoints_json,
        ref_text=ref_text,
        pred_text=other_text,
        item_index=int(item["index"]),
        args=args,
    )

    return compute_line_coverage_percentage_metrics_from_precomputed_endpoints(
        ref_text=ref_text,
        other_text=other_text,
        refref_line_endpoints=refref_line_endpoints,
        other_line_endpoints=other_line_endpoints,
        window_size=int(window_size),
        window_stride=int(window_stride),
        strict_lines=bool(strict_lines),
        file_name=str(item["fname"]),
    )


def compute_y_axis_coverage_percentage_metrics(**kwargs) -> dict:
    """Backward-compatible alias to the public metrics API."""
    return compute_line_coverage_percentage_metrics(**kwargs)


def _write_output_json(output_json: Path, *, payload: dict) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _validate_cli_inputs(args)

    metrics = compute_line_coverage_percentage_metrics(
        runfile_json=args.runfile_json,
        target_fname=args.target_fname,
        other_text_path=args.other_text_path,
        refref_line_endpoints_json=args.refref_line_endpoints_json,
        other_line_endpoints_json=args.other_line_endpoints_json,
        allow_derive_endpoints=bool(args.allow_derive_endpoints),
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        hough_threshold=int(args.hough_threshold),
        hough_line_length=int(args.hough_line_length),
        hough_line_gap=int(args.hough_line_gap),
        hough_seed=int(args.hough_seed),
        hough_start=float(args.hough_start),
        align_abs_min_len=float(args.align_abs_min_len),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
        strict_lines=bool(args.strict_lines),
    )

    print(json.dumps(metrics, ensure_ascii=False))

    if args.output_json is not None:
        _write_output_json(args.output_json, payload=metrics)


if __name__ == "__main__":
    main()
