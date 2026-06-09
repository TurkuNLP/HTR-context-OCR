"""CLI parsing and argument validation for ``text_metrics_report``.

The public contract of this pipeline is the shell wrapper
``run_text_metrics_report.sh``. The parser defaults here are intentionally kept
aligned with that shell entrypoint so the internal Python path and the public
shell path behave consistently.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hough_detection.line_handoff_modes import (
    DEFAULT_HOUGH_HANDOFF_MODE,
    HOUGH_HANDOFF_MODE_MERGED_TO_TRUE_IOU,
    HOUGH_HANDOFF_MODE_RAW_TO_TRUE_IOU,
)
from line_filtering_v2_12_IoU import DEFAULT_MIN_IOU_THRESHOLD

SELECTION_MODE_ALL_SELECTED_DOCS = "all_selected_docs"
SELECTION_MODE_ONLY_JSON_DOCS = "only_json_docs"
SUPPORTED_HOUGH_PARAMS_SELECTION_MODES = (
    SELECTION_MODE_ALL_SELECTED_DOCS,
    SELECTION_MODE_ONLY_JSON_DOCS,
)
SUPPORTED_HOUGH_HANDOFF_MODES = (
    HOUGH_HANDOFF_MODE_MERGED_TO_TRUE_IOU,
    HOUGH_HANDOFF_MODE_RAW_TO_TRUE_IOU,
)


# Parse the full CLI surface for the v2.12 text-metrics pipeline.
def parse_text_metrics_report_args() -> argparse.Namespace:
    """Parse CLI arguments for the main text-metrics report pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Align prediction text with fixed-diagonal probabilistic Hough lines "
            "from runfile JSON and/or precomputed scores.pkl matrices."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runfile-json", type=Path, default=None, help="Optional path to outputs.json")

    parser.add_argument(
        "--scores-pkl-ref-to-pred",
        type=Path,
        default=None,
        help="Optional ref->pred scores.pkl stream.",
    )
    parser.add_argument(
        "--scores-pkl-ref-to-ref",
        type=Path,
        default=None,
        help="Optional ref->ref scores.pkl stream.",
    )
    parser.add_argument(
        "--scores-pkl-ref-to-adjusted-pred",
        type=Path,
        default=None,
        help="Optional ref->adjusted-pred scores.pkl stream.",
    )
    parser.add_argument(
        "--scores-pkl-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing compare subfolders ref_to_pred, ref_to_ref, "
            "and ref_to_adjusted_pred with .pkl files."
        ),
    )

    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--window-size", type=int, default=50, help="Sliding window size")
    parser.add_argument("--window-stride", type=int, default=35, help="Sliding window stride")
    parser.add_argument("--target-fname", type=str, default=None, help="Optional exact/basename target file")
    parser.add_argument("--max-items", type=int, default=None, help="Optional maximum processed items")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of document workers. 1 keeps sequential behavior. "
            "Values greater than available CPUs fail fast by design."
        ),
    )
    parser.add_argument(
        "--skip-visuals",
        dest="skip_visuals",
        action="store_true",
        default=True,
        help=(
            "Skip visual outputs. When enabled with --with-visuals, the pipeline saves: "
            "before-Hough, raw-Hough, filtered-lines, optional after-reordering, and count_line_coverage (x/y)."
        ),
    )
    parser.add_argument(
        "--with-visuals",
        dest="skip_visuals",
        action="store_false",
        help=(
            "Generate visual outputs: before-Hough, raw-Hough, filtered-lines, optional after-reordering, "
            "and count_line_coverage (x/y)."
        ),
    )

    parser.add_argument("--hough-threshold", type=int, default=10, help="Hough vote threshold")
    parser.add_argument("--hough-line-length", type=int, default=8, help="Minimum accepted line length")
    parser.add_argument("--hough-line-gap", type=int, default=8, help="Maximum gap to connect line pixels")
    parser.add_argument("--hough-seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--hough-start", type=float, default=2.2, help="Initial adaptive threshold start before decrement loop")
    parser.add_argument(
        "--hough-handoff-mode",
        type=str,
        choices=SUPPORTED_HOUGH_HANDOFF_MODES,
        default=DEFAULT_HOUGH_HANDOFF_MODE,
        help=(
            "Which Hough segment set is handed into the production true-IoU filter. "
            "raw_hough_to_true_iou is the default and passes raw Hough segments directly "
            "into the exact same filter. merged_hough_to_true_iou remains available for "
            "explicit compatibility runs that still need greedy Hough postprocessing."
        ),
    )
    parser.add_argument(
        "--align-abs-min-len",
        type=float,
        default=6.0,
        help="Absolute minimum line length kept before ownership resolution.",
    )
    parser.add_argument(
        "--align-min-iou-threshold",
        type=float,
        default=DEFAULT_MIN_IOU_THRESHOLD,
        help="Minimum true-IoU threshold used to merge overlapping line coverages in the stable production filter.",
    )

    parser.add_argument(
        "--hough-params-per-document-json",
        type=Path,
        default=None,
        help=(
            "Optional path to tuner best_params_per_document.json. "
            "When set, matching documents use per-document Hough threshold/line_length/line_gap/seed overrides."
        ),
    )
    parser.add_argument(
        "--hough-params-selection-mode",
        type=str,
        choices=SUPPORTED_HOUGH_PARAMS_SELECTION_MODES,
        default=SELECTION_MODE_ONLY_JSON_DOCS,
        help=(
            "How to select documents when --hough-params-per-document-json is provided. "
            "only_json_docs: process only docs listed in JSON. "
            "all_selected_docs: keep normal selection and override only docs found in JSON."
        ),
    )
    parser.add_argument(
        "--hough-params-strict",
        action="store_true",
        help=(
            "Enable strict consistency checks for --hough-params-per-document-json. "
            "Fails on JSON/doc selection mismatches instead of falling back silently."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Write run-level report_timings.json with per-document timing telemetry. "
            "When disabled, timing collection is skipped to reduce overhead."
        ),
    )
    return parser.parse_args()


# Validate worker count against visible CPU resources.
def validate_workers_or_raise(workers: int) -> int:
    """Validate workers against visible CPU count and return available CPUs."""
    available_cpus = int(os.cpu_count() or 1)
    if int(workers) <= 0:
        raise ValueError("workers must be a positive integer")
    if int(workers) > available_cpus:
        raise ValueError(
            f"workers must not exceed available CPUs. requested={workers}, available={available_cpus}"
        )
    return available_cpus


# Validate normalized CLI argument values and required input paths.
def validate_text_metrics_report_args(
    args: argparse.Namespace,
    *,
    scores_pkl_paths_by_kind: dict[str, Path | None],
) -> None:
    """Validate normalized argument values and required input files."""
    if args.runfile_json is None and not any(scores_pkl_paths_by_kind.values()):
        raise ValueError("Provide at least one input source: --runfile-json or any --scores-pkl-* option")

    if args.runfile_json is not None and not Path(args.runfile_json).exists():
        raise FileNotFoundError(f"Missing runfile JSON: {args.runfile_json}")

    for kind, path in scores_pkl_paths_by_kind.items():
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"Missing {kind} scores file: {path}")

    if args.hough_params_per_document_json is not None and not Path(args.hough_params_per_document_json).exists():
        raise FileNotFoundError(
            f"Missing per-document Hough params JSON: {args.hough_params_per_document_json}"
        )
    if args.hough_params_per_document_json is None and bool(args.hough_params_strict):
        raise ValueError("--hough-params-strict requires --hough-params-per-document-json")

    if int(args.window_size) <= 0 or int(args.window_stride) <= 0:
        raise ValueError("window-size and window-stride must be positive")
    if args.max_items is not None and int(args.max_items) <= 0:
        raise ValueError("max-items must be positive")
    if int(args.hough_threshold) <= 0:
        raise ValueError("hough-threshold must be positive")
    if int(args.hough_line_length) <= 0:
        raise ValueError("hough-line-length must be positive")
    if int(args.hough_line_gap) < 0:
        raise ValueError("hough-line-gap must be non-negative")
    if float(args.hough_start) <= 0:
        raise ValueError("hough-start must be positive")
    if float(args.align_abs_min_len) <= 0:
        raise ValueError("align-abs-min-len must be positive")
    if not (0.0 <= float(args.align_min_iou_threshold) <= 1.0):
        raise ValueError("align-min-iou-threshold must satisfy 0.0 <= value <= 1.0")
