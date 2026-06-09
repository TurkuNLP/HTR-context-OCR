#!/usr/bin/env python3
from __future__ import annotations

"""CLI entry point for the Hough parameter tuner.

Users provide input/output paths, optional inclusive Hough ranges, and
parallelism settings.  The runner executes the sweep, writes CSV/JSON artifacts,
and generates visualisation artifacts only when visual output is requested.
"""

import argparse
from pathlib import Path

try:
    from .runtime.runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime.runtime_paths import ensure_tuner_runtime_paths  # type: ignore

_SCRIPT_DIR, _PROJECT_ROOT, _SHARED_METRICS_DIR = ensure_tuner_runtime_paths()

try:
    from .logging_utils.timestamped_logging import build_timestamped_logger
    from .tuner.hough_eval import DEFAULT_SELECTION_OBJECTIVE, SUPPORTED_SELECTION_OBJECTIVES
    from .tuner.tuner_config import (
        DEFAULT_REF_TO_REF_COMBO_CACHE_DIR,
        DEFAULT_TEXT_METRICS_V212_DIR,
        HOUGH_LINE_GAP_MAX,
        HOUGH_LINE_GAP_MIN,
        HOUGH_LINE_LENGTH_MAX,
        HOUGH_LINE_LENGTH_MIN,
        HOUGH_SEED_MAX,
        HOUGH_SEED_MIN,
        HOUGH_THRESHOLD_MAX,
        HOUGH_THRESHOLD_MIN,
        HoughBaselineConfig,
        build_hough_sweep_ranges,
    )
    from .tuner.tuner_core import (
        BACKEND_C,
        DEFAULT_SCORE_INDEX_CACHE_DIR,
        SUPPORTED_BACKENDS,
        run_hough_parameter_sweeps,
    )
    from .matrices.runfile_selection import select_run_items_for_tuning
    from .hough_preprocessing import (
        ADAPTIVE_BUDGET_MASK_STRONG_MATCH,
        CONNECTED_COMPONENT_BACKEND_CYTHON,
        CONNECTED_COMPONENT_BACKEND_PYTHON,
        CONNECTED_COMPONENT_BACKEND_SCIPY,
        FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST,
        HoughPreprocessingConfig,
        MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
        MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY,
        SCORE_FLOOR_METHOD_MEAN_PLUS_STANDARD_DEVIATION,
        SUPPORTED_ADAPTIVE_BUDGET_MASKS,
        SUPPORTED_FINAL_HOUGH_INPUT_MODES,
        SUPPORTED_SCORE_FLOOR_METHODS,
    )
    from .dynamic_pool.document_pool import (
        DocumentLease,
        DocumentLeasePool,
        iter_claimed_selected_run_items_from_pool,
    )
except ImportError:
    from logging_utils.timestamped_logging import build_timestamped_logger  # type: ignore
    from tuner.hough_eval import DEFAULT_SELECTION_OBJECTIVE, SUPPORTED_SELECTION_OBJECTIVES  # type: ignore
    from tuner.tuner_config import (  # type: ignore
        DEFAULT_REF_TO_REF_COMBO_CACHE_DIR,
        DEFAULT_TEXT_METRICS_V212_DIR,
        HOUGH_LINE_GAP_MAX,
        HOUGH_LINE_GAP_MIN,
        HOUGH_LINE_LENGTH_MAX,
        HOUGH_LINE_LENGTH_MIN,
        HOUGH_SEED_MAX,
        HOUGH_SEED_MIN,
        HOUGH_THRESHOLD_MAX,
        HOUGH_THRESHOLD_MIN,
        HoughBaselineConfig,
        build_hough_sweep_ranges,
    )
    from tuner.tuner_core import (  # type: ignore
        BACKEND_C,
        DEFAULT_SCORE_INDEX_CACHE_DIR,
        SUPPORTED_BACKENDS,
        run_hough_parameter_sweeps,
    )
    from matrices.runfile_selection import select_run_items_for_tuning  # type: ignore
    from hough_preprocessing import (  # type: ignore
        ADAPTIVE_BUDGET_MASK_STRONG_MATCH,
        CONNECTED_COMPONENT_BACKEND_CYTHON,
        CONNECTED_COMPONENT_BACKEND_PYTHON,
        CONNECTED_COMPONENT_BACKEND_SCIPY,
        FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST,
        HoughPreprocessingConfig,
        MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
        MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY,
        SCORE_FLOOR_METHOD_MEAN_PLUS_STANDARD_DEVIATION,
        SUPPORTED_ADAPTIVE_BUDGET_MASKS,
        SUPPORTED_FINAL_HOUGH_INPUT_MODES,
        SUPPORTED_SCORE_FLOOR_METHODS,
    )
    from dynamic_pool.document_pool import (  # type: ignore
        DocumentLease,
        DocumentLeasePool,
        iter_claimed_selected_run_items_from_pool,
    )


def _range_metavar(label: str) -> tuple[str, str]:
    """Return argparse metavar pair for a two-integer inclusive range."""
    return (f"{label}_START", f"{label}_END")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tuner runner."""
    default_cache_dir = _SCRIPT_DIR / "_matrix_cache"

    parser = argparse.ArgumentParser(
        description=(
            "Document-level Hough tuner. Defaults: threshold 1..40, "
            "line_length 1..50, line_gap 1..30, fixed seed 1. "
            "Hough line direction is constrained to falling 30..60 degree diagonals."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON/CSV/plots")
    parser.add_argument("--window-size", type=int, default=50, help="Sliding score-matrix window size")
    parser.add_argument("--window-stride", type=int, default=35, help="Sliding score-matrix window stride")

    parser.add_argument("--matrix-cache-dir", type=Path, default=default_cache_dir, help="Reusable tuner-local matrix cache")
    parser.add_argument("--no-matrix-cache", action="store_true", help="Disable tuner-local matrix cache")

    parser.add_argument(
        "--scores-pkl-ref-to-pred",
        type=Path,
        default=None,
        help="Optional read-only scores_reference_prediction_ws*_st*.pkl source for ref-to-pred matrices",
    )
    parser.add_argument(
        "--scores-pkl-ref-to-ref",
        type=Path,
        default=None,
        help="Optional read-only scores_reference_self_ws*_st*.pkl source for ref-to-ref matrices",
    )
    parser.add_argument(
        "--score-index-cache-file",
        type=Path,
        default=None,
        help="Optional explicit read-only index cache for the ref-to-pred scores pkl",
    )
    parser.add_argument(
        "--score-index-cache-file-ref-to-ref",
        type=Path,
        default=None,
        help="Optional explicit read-only index cache for the ref-to-ref scores pkl",
    )
    parser.add_argument(
        "--score-index-cache-dir",
        type=Path,
        default=DEFAULT_SCORE_INDEX_CACHE_DIR,
        help="Directory containing read-only score-stream index caches",
    )
    parser.add_argument("--disable-pkl-matrix-source", action="store_true", help="Disable all read-only pkl matrix sources")

    parser.add_argument(
        "--text-metrics-v212-dir",
        type=Path,
        default=DEFAULT_TEXT_METRICS_V212_DIR,
        help=(
            "Optional external text_metrics_v2_12_parallel directory for audits "
            "and equivalence tests. Normal scoring uses the tuner-local v2.12 "
            "compatibility code."
        ),
    )
    parser.add_argument(
        "--ref-to-ref-cache-mode",
        type=str,
        default="auto",
        choices=("off", "auto", "read-only"),
        help=(
            "Persistent reference-self document-pack cache mode. auto reads existing exact "
            "refref_y coverage baselines and writes one cache file per completed document/grid; "
            "off preserves the recompute-every-combination path."
        ),
    )
    parser.add_argument(
        "--ref-to-ref-cache-dir",
        type=Path,
        default=DEFAULT_REF_TO_REF_COMBO_CACHE_DIR,
        help="Directory for exact ref_to_ref document-pack cache artifacts",
    )
    parser.add_argument(
        "--ref-to-ref-cache-warm-only",
        action="store_true",
        help=(
            "Fill the exact ref_to_ref document-pack cache and exit before the "
            "prediction-side tuner. Use with --ref-to-ref-cache-mode auto."
        ),
    )

    parser.add_argument("--target-fname", action="append", default=[], help="Exact/basename target file; repeat for many")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on number of selected documents")
    parser.add_argument(
        "--selection-index-range",
        "--item-index-range",
        "--document-index-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help=(
            "Zero-based inclusive range inside the selected document list. "
            "Applied after --target-fname filtering and --max-items capping."
        ),
    )
    parser.add_argument(
        "--levenshtein-backend",
        type=str,
        default=BACKEND_C,
        choices=tuple(SUPPORTED_BACKENDS),
        help="Levenshtein backend; 'c' uses rapidfuzz exact C-backed distance",
    )
    parser.add_argument("--workers", type=int, default=1, help="Threshold-worker count per document")
    parser.add_argument("--doc-workers", type=int, default=1, help="Documents to process in parallel")
    parser.add_argument(
        "--dynamic-document-pool-dir",
        type=Path,
        default=None,
        help=(
            "Optional scheduling-only shared document pool. When passed, this "
            "worker claims one replacement document whenever a local document slot frees."
        ),
    )
    parser.add_argument(
        "--dynamic-worker-id",
        type=str,
        default=None,
        help="Stable worker id used in dynamic-pool claim filenames and logs",
    )
    parser.add_argument(
        "--dynamic-cpus-per-task",
        type=int,
        default=None,
        help="CPU count used to cap dynamic doc-workers by active threshold count",
    )

    parser.add_argument("--hough-seed", type=int, default=HOUGH_SEED_MIN, help="Deprecated compatibility arg; fixed seed 1 is used")
    parser.add_argument("--align-min-iou-threshold", type=float, default=0.035, help="Minimum true-IoU threshold for line filtering")
    parser.add_argument(
        "--minimum-score-floor",
        "--min-score",
        type=float,
        default=20.0,
        help="Minimum chrF score used by the Median Absolute Deviation score-floor method",
    )
    parser.add_argument(
        "--score-floor-method",
        choices=tuple(SUPPORTED_SCORE_FLOOR_METHODS),
        default=SCORE_FLOOR_METHOD_MEAN_PLUS_STANDARD_DEVIATION,
        help=(
            "How the preprocessing score floor is calculated. "
            "mean_plus_standard_deviation uses the document mean plus one population "
            "standard deviation. median_plus_scaled_median_absolute_deviation keeps "
            "the previous Median Absolute Deviation path and applies --minimum-score-floor."
        ),
    )
    parser.add_argument(
        "--median-absolute-deviation-multiplier",
        "--mad-k",
        type=float,
        default=0.0,
        help="Multiplier applied to the scaled Median Absolute Deviation when that score-floor method is selected",
    )
    parser.add_argument(
        "--median-absolute-deviation-backend",
        choices=(MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY, MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY),
        default=MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
        help="Implementation used for the scaled Median Absolute Deviation calculation",
    )
    parser.add_argument("--near-peak-ratio", type=float, default=0.70, help="Keep cells near the best score in their row or column")
    parser.add_argument("--near-peak-margin", type=float, default=None, help="Optional score-distance margin for near-peak cells")
    parser.add_argument("--minimum-component-cells", "--min-component-cells", type=int, default=2, help="Minimum cell count for a connected Region of Interest component")
    parser.add_argument("--minimum-component-rows", "--min-component-rows", type=int, default=1, help="Minimum row count for a connected Region of Interest component")
    parser.add_argument("--minimum-component-columns", "--min-component-cols", type=int, default=1, help="Minimum column count for a connected Region of Interest component")
    parser.add_argument(
        "--connected-component-backend",
        choices=(
            CONNECTED_COMPONENT_BACKEND_CYTHON,
            CONNECTED_COMPONENT_BACKEND_SCIPY,
            CONNECTED_COMPONENT_BACKEND_PYTHON,
        ),
        default=CONNECTED_COMPONENT_BACKEND_CYTHON,
        help="Backend for connected Region of Interest labeling; Cython is the default, SciPy is optional",
    )
    parser.add_argument("--region-dilation-radius", "--dilation-radius", type=int, default=1, help="How many cells the kept Region of Interest may expand before the final Hough input is built")
    parser.add_argument(
        "--final-hough-input-mode",
        choices=tuple(SUPPORTED_FINAL_HOUGH_INPUT_MODES),
        default=FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST,
        help="Choose which mask becomes the final binary Hough input after the Region of Interest is built",
    )
    parser.add_argument(
        "--adaptive-budget-mask",
        choices=tuple(SUPPORTED_ADAPTIVE_BUDGET_MASKS),
        default=ADAPTIVE_BUDGET_MASK_STRONG_MATCH,
        help="Choose which mask is checked against the adaptive score-floor/100 Hough-voter budget",
    )
    parser.add_argument("--minimum-active-cells", "--min-active-cells", type=int, default=0, help="Minimum active cells required in the final binary Hough input; 0 disables this cell-count gate")
    parser.add_argument("--minimum-active-rows", "--min-active-rows", type=int, default=2, help="Minimum active reference rows required in the final binary Hough input")
    parser.add_argument("--minimum-active-columns", "--min-active-cols", type=int, default=2, help="Minimum active prediction columns required in the final binary Hough input")
    parser.add_argument("--minimum-x-span", "--min-x-span", type=int, default=2, help="Minimum prediction-axis span required in the final binary Hough input")
    parser.add_argument("--minimum-y-span", "--min-y-span", type=int, default=2, help="Minimum reference-axis span required in the final binary Hough input")
    parser.add_argument("--maximum-active-fraction", "--max-active-fraction", type=float, default=1.0, help="Optional fixed maximum fraction of matrix cells allowed to remain active after preprocessing")
    parser.add_argument("--minimum-matrix-rows", type=int, default=4, help="Reject matrices with fewer reference-window rows before Hough preprocessing")
    parser.add_argument("--minimum-matrix-columns", type=int, default=4, help="Reject matrices with fewer prediction-window columns before Hough preprocessing")

    parser.add_argument(
        "--min-surviving-line-nls",
        type=float,
        default=None,
        help=(
            "Optional post-geometry text-quality filter. When set, final "
            "ref_to_pred lines whose line-level normalized Levenshtein "
            "similarity is below this value are removed before coverage/scoring. "
            "If all final lines are removed, the combination is reported as a "
            "valid zero-score 100%% hallucination outcome with an explicit reason."
        ),
    )

    parser.add_argument(
        "--selection-objective",
        type=str,
        default=DEFAULT_SELECTION_OBJECTIVE,
        choices=tuple(SUPPORTED_SELECTION_OBJECTIVES),
        help=(
            "How the tuner chooses the winning Hough combination. strict_quality "
            "keeps the original harmonic tuning score. alignment_evidence prefers "
            "matrix-supported, line-guided geometry. non_hallucination_weighted "
            "uses the same harmonic components as strict_quality, but gives "
            "non-hallucination double weight."
        ),
    )

    parser.add_argument(
        "--hough-threshold-range",
        "--threshold",
        nargs=2,
        type=int,
        metavar=_range_metavar("THRESHOLD"),
        default=None,
        help=f"Inclusive threshold range; default {HOUGH_THRESHOLD_MIN}..{HOUGH_THRESHOLD_MAX}",
    )
    parser.add_argument(
        "--line-length-range",
        "--line_length",
        "--line-length",
        nargs=2,
        type=int,
        metavar=_range_metavar("LINE_LENGTH"),
        default=None,
        help=f"Inclusive Hough line_length range; default {HOUGH_LINE_LENGTH_MIN}..{HOUGH_LINE_LENGTH_MAX}",
    )
    parser.add_argument(
        "--line-gap-range",
        "--line_gap",
        "--line-gap",
        nargs=2,
        type=int,
        metavar=_range_metavar("LINE_GAP"),
        default=None,
        help=f"Inclusive Hough line_gap range; default {HOUGH_LINE_GAP_MIN}..{HOUGH_LINE_GAP_MAX}",
    )
    parser.add_argument(
        "--seed-range",
        "--seed",
        nargs=2,
        type=int,
        metavar=_range_metavar("SEED"),
        default=None,
        help=(
            "Deprecated compatibility arg. Seed sweep is disabled in this tuner; "
            f"fixed seed {HOUGH_SEED_MIN} is used."
        ),
    )
    parser.add_argument(
        "--combination-bundle-dir",
        type=Path,
        default=None,
        help="Optional directory for per-combination binary visualization bundles",
    )
    parser.add_argument(
        "--combination-bundle-scope",
        type=str,
        default="none",
        choices=("none", "all", "valid-only", "invalid-only", "winner-only"),
        help="Which evaluated combinations should be written to --combination-bundle-dir",
    )
    parser.add_argument(
        "--combination-bundle-include-candidate-lines",
        action="store_true",
        help=(
            "Accepted for compatibility. Lean pklstream visualization bundles "
            "store raw Hough lines and final surviving lines, not pre-filter "
            "candidate geometry."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Optional shard index written into per-combination bundle records",
    )
    parser.add_argument(
        "--with-visuals",
        action="store_true",
        help=(
            "Generate final colour visualisations after the sweep. If no explicit "
            "bundle scope is provided, only winner geometry is written."
        ),
    )
    parser.add_argument(
        "--hide-line-labels",
        action="store_true",
        help="Hide raw Hough and final surviving-line labels in stitched best-combination visuals.",
    )
    parser.add_argument(
        "--profile-combinations",
        action="store_true",
        help=(
            "Write a compact top-level combination_profile.csv with scalar "
            "per-combination timing/count diagnostics. Scientific scoring is unchanged."
        ),
    )
    parser.add_argument(
        "--no-combination-score-table",
        action="store_true",
        help=(
            "Disable the normal top-level combination_scores.csv.gz scalar table. "
            "Keeping it enabled makes parameter-range analysis possible without reading geometry bundles."
        ),
    )

    return parser.parse_args()


def _format_path(value: Path | None) -> str:
    """Return a stable human-readable string for optional paths."""
    return "None" if value is None else str(Path(value))


def _dynamic_worker_id(cli_worker_id: str | None) -> str:
    """Return the best stable worker id available in Slurm or local mode."""
    if cli_worker_id is not None and str(cli_worker_id).strip():
        return str(cli_worker_id).strip()
    import os

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return f"slurm_{slurm_job_id}"
    return f"local_pid_{os.getpid()}"


def _cap_doc_workers_for_dynamic_pool(
    *,
    requested_doc_workers: int,
    dynamic_cpus_per_task: int | None,
    threshold_value_count: int,
    log,
) -> int:
    """Cap dynamic document concurrency to avoid oversubscribing one node."""
    requested_doc_workers = max(1, int(requested_doc_workers))
    if dynamic_cpus_per_task is None:
        return requested_doc_workers

    threshold_value_count = max(1, int(threshold_value_count))
    cpu_capacity = max(1, int(dynamic_cpus_per_task) // threshold_value_count)
    resolved_doc_workers = min(requested_doc_workers, cpu_capacity)
    if resolved_doc_workers != requested_doc_workers:
        log(
            "[dynamic-pool] capping doc_workers "
            f"from {requested_doc_workers} to {resolved_doc_workers} "
            f"because cpus_per_task={int(dynamic_cpus_per_task)} "
            f"and threshold_count={threshold_value_count}"
        )
    else:
        log(
            "[dynamic-pool] doc_workers within node capacity "
            f"doc_workers={resolved_doc_workers} cpus_per_task={int(dynamic_cpus_per_task)} "
            f"threshold_count={threshold_value_count}"
        )
    return int(resolved_doc_workers)


def main() -> None:
    """Run the tuner and optionally generate visualisation artifacts."""
    args = parse_args()
    log = build_timestamped_logger(print)

    active_ranges = build_hough_sweep_ranges(
        threshold_range=args.hough_threshold_range,
        line_length_range=args.line_length_range,
        line_gap_range=args.line_gap_range,
        seed_range=args.seed_range,
    )

    if int(args.hough_seed) != int(HOUGH_SEED_MIN):
        log(f"[compat] --hough-seed is ignored; fixed hough_seed={HOUGH_SEED_MIN} is used.")
    if args.seed_range is not None:
        log(f"[compat] --seed-range is ignored while fixed hough_seed={HOUGH_SEED_MIN} mode is active.")

    baseline = HoughBaselineConfig(
        hough_threshold=active_ranges.threshold.minimum,
        hough_line_length=active_ranges.line_length.minimum,
        hough_line_gap=active_ranges.line_gap.minimum,
        hough_seed=active_ranges.seed.minimum,
        align_min_iou_threshold=float(args.align_min_iou_threshold),
    )
    hough_preprocessing_config = HoughPreprocessingConfig(
        minimum_score_floor=float(args.minimum_score_floor),
        score_floor_method=str(args.score_floor_method),
        median_absolute_deviation_multiplier=float(args.median_absolute_deviation_multiplier),
        median_absolute_deviation_backend=str(args.median_absolute_deviation_backend),
        near_peak_ratio=float(args.near_peak_ratio),
        near_peak_margin=None if args.near_peak_margin is None else float(args.near_peak_margin),
        minimum_component_cells=int(args.minimum_component_cells),
        minimum_component_rows=int(args.minimum_component_rows),
        minimum_component_columns=int(args.minimum_component_columns),
        connected_component_backend=str(args.connected_component_backend),
        region_dilation_radius=int(args.region_dilation_radius),
        final_hough_input_mode=str(args.final_hough_input_mode),
        adaptive_budget_mask=str(args.adaptive_budget_mask),
        minimum_active_cells=int(args.minimum_active_cells),
        minimum_active_rows=int(args.minimum_active_rows),
        minimum_active_columns=int(args.minimum_active_columns),
        minimum_x_span=int(args.minimum_x_span),
        minimum_y_span=int(args.minimum_y_span),
        maximum_active_fraction=float(args.maximum_active_fraction),
        minimum_matrix_rows=int(args.minimum_matrix_rows),
        minimum_matrix_columns=int(args.minimum_matrix_columns),
    )

    targets = [str(v) for v in args.target_fname if str(v).strip()]
    matrix_cache_dir = None if bool(args.no_matrix_cache) else Path(args.matrix_cache_dir)
    selection_index_range = (
        None
        if args.selection_index_range is None
        else (int(args.selection_index_range[0]), int(args.selection_index_range[1]))
    )

    log(
        "[grid-active] "
        f"{active_ranges.active_grid_label()} "
        f"threshold_workers={max(1, int(args.workers))} doc_workers={max(1, int(args.doc_workers))} "
        "line_angle=falling_diagonal_only(30..60)"
    )
    log(
        "[matrix-sources] "
        f"npz_cache={'enabled' if matrix_cache_dir is not None else 'disabled'} "
        f"scores_pkl_ref_to_pred={_format_path(args.scores_pkl_ref_to_pred)} "
        f"scores_pkl_ref_to_ref={_format_path(args.scores_pkl_ref_to_ref)} "
        f"score_index_cache_file={_format_path(args.score_index_cache_file)} "
        f"score_index_cache_file_ref_to_ref={_format_path(args.score_index_cache_file_ref_to_ref)} "
        f"score_index_cache_dir={_format_path(args.score_index_cache_dir)} "
        f"disable_pkl_matrix_source={bool(args.disable_pkl_matrix_source)}"
    )
    log(
        "[v2.12-metrics] "
        "runtime_source=tuner_local_compat "
        f"external_audit_dir={Path(args.text_metrics_v212_dir)}"
    )
    log(
        "[ref-to-ref-cache] "
        f"mode={args.ref_to_ref_cache_mode} dir={Path(args.ref_to_ref_cache_dir)} "
        f"warm_only={bool(args.ref_to_ref_cache_warm_only)}"
    )
    log(
        "[hough-preprocessing] "
        "mode=region_of_interest "
        f"score_floor_method={hough_preprocessing_config.score_floor_method} "
        f"minimum_score_floor={hough_preprocessing_config.minimum_score_floor:.6f} "
        f"median_absolute_deviation_multiplier={hough_preprocessing_config.median_absolute_deviation_multiplier:.6f} "
        f"median_absolute_deviation_backend={hough_preprocessing_config.median_absolute_deviation_backend} "
        f"near_peak_ratio={hough_preprocessing_config.near_peak_ratio:.6f} "
        f"final_hough_input_mode={hough_preprocessing_config.final_hough_input_mode} "
        f"adaptive_budget_mask={hough_preprocessing_config.adaptive_budget_mask} "
        f"connected_component_backend={hough_preprocessing_config.connected_component_backend} "
        f"maximum_active_fraction={hough_preprocessing_config.maximum_active_fraction:.6f} "
        f"minimum_matrix_size={hough_preprocessing_config.minimum_matrix_rows}x{hough_preprocessing_config.minimum_matrix_columns}"
    )
    if selection_index_range is not None:
        log(f"[selection] selection_index_range={selection_index_range[0]}..{selection_index_range[1]}")
    if bool(args.profile_combinations):
        log("[profiling] combination_profile.csv export enabled")
    if bool(args.no_combination_score_table):
        log("[combination-scores] disabled by --no-combination-score-table")
    if args.min_surviving_line_nls is not None:
        if not (0.0 <= float(args.min_surviving_line_nls) <= 1.0):
            raise ValueError("--min-surviving-line-nls must be between 0.0 and 1.0")

    dynamic_document_pool = None
    dynamic_active_leases_by_document_index: dict[int, DocumentLease] = {}
    dynamic_completed_leases_waiting_for_output: list[DocumentLease] = []
    selected_run_items_override = None
    selected_document_count_override = None
    on_document_completed = None
    on_document_skipped = None
    resolved_doc_workers = max(1, int(args.doc_workers))

    if args.dynamic_document_pool_dir is not None:
        if selection_index_range is not None:
            raise ValueError("--selection-index-range cannot be combined with --dynamic-document-pool-dir")

        resolved_worker_id = _dynamic_worker_id(args.dynamic_worker_id)
        resolved_doc_workers = _cap_doc_workers_for_dynamic_pool(
            requested_doc_workers=max(1, int(args.doc_workers)),
            dynamic_cpus_per_task=args.dynamic_cpus_per_task,
            threshold_value_count=(
                int(active_ranges.threshold.maximum) - int(active_ranges.threshold.minimum) + 1
            ),
            log=log,
        )
        dynamic_document_pool = DocumentLeasePool(
            pool_dir=Path(args.dynamic_document_pool_dir),
            worker_id=resolved_worker_id,
            log_fn=log,
        )
        selected_run_items_for_dynamic_pool = select_run_items_for_tuning(
            runfile_json=Path(args.runfile_json),
            target_fnames=targets,
            max_items=args.max_items,
            selection_index_range=None,
        )
        selected_document_count_override = int(len(selected_run_items_for_dynamic_pool))
        selected_run_items_override = iter_claimed_selected_run_items_from_pool(
            document_pool=dynamic_document_pool,
            selected_run_items=selected_run_items_for_dynamic_pool,
            active_lease_by_document_index=dynamic_active_leases_by_document_index,
            log_fn=log,
        )

        def _record_dynamic_document_completion(doc, _tuned_payload: dict) -> None:
            """Remember completed leases and finalize them after outputs exist."""
            document_index = int(doc.index)
            lease = dynamic_active_leases_by_document_index.pop(document_index, None)
            if lease is None:
                raise RuntimeError(
                    "Dynamic document finished without an active lease: "
                    f"index={document_index} fname={doc.fname}"
                )
            dynamic_completed_leases_waiting_for_output.append(lease)
            log(
                "[dynamic-pool-complete-in-memory] "
                f"worker={resolved_worker_id} index={document_index} fname={doc.fname} "
                "will mark done after normal tuner outputs are written"
            )

        def _record_dynamic_document_skip(skip_record: dict) -> None:
            """Remember skipped leases and finalize them after skip CSV/summary output exists."""
            document_index = int(skip_record["index"])
            lease = dynamic_active_leases_by_document_index.pop(document_index, None)
            if lease is None:
                raise RuntimeError(
                    "Dynamic document skipped without an active lease: "
                    f"index={document_index} fname={skip_record.get('fname')}"
                )
            dynamic_completed_leases_waiting_for_output.append(lease)
            log(
                "[dynamic-pool-skip-in-memory] "
                f"worker={resolved_worker_id} index={document_index} fname={skip_record.get('fname')} "
                f"reason={skip_record.get('skip_reason')} "
                "will mark done after skipped-document outputs are written"
            )

        on_document_completed = _record_dynamic_document_completion
        on_document_skipped = _record_dynamic_document_skip
        log(
            "[dynamic-pool] enabled "
            f"pool_dir={Path(args.dynamic_document_pool_dir)} worker_id={resolved_worker_id} "
            f"selected_document_count={selected_document_count_override} "
            f"resolved_doc_workers={resolved_doc_workers}"
        )

    combination_bundle_scope = str(args.combination_bundle_scope)
    combination_bundle_dir = args.combination_bundle_dir
    if bool(args.with_visuals):
        # The compact score table now carries scalar rows for all combinations.
        # Visual mode therefore needs geometry only for the selected winner
        # unless the caller explicitly requested a diagnostic all/valid/invalid
        # bundle scope.
        if combination_bundle_scope == "none":
            combination_bundle_scope = "winner-only"
        if combination_bundle_dir is None:
            combination_bundle_dir = Path(args.output_dir) / "combination_bundles"
        log(
            "[visuals] enabled "
            f"combination_bundle_scope={combination_bundle_scope} "
            f"bundle_dir={Path(combination_bundle_dir)} "
            f"hide_line_labels={bool(args.hide_line_labels)}"
        )

    try:
        result = run_hough_parameter_sweeps(
            runfile_json=Path(args.runfile_json),
            output_dir=Path(args.output_dir),
            window_size=int(args.window_size),
            window_stride=int(args.window_stride),
            baseline_cfg=baseline,
            hough_sweep_ranges=active_ranges,
            matrix_cache_dir=matrix_cache_dir,
            scores_pkl_ref_to_pred=args.scores_pkl_ref_to_pred,
            scores_pkl_ref_to_ref=args.scores_pkl_ref_to_ref,
            score_index_cache_file=args.score_index_cache_file,
            score_index_cache_file_ref_to_ref=args.score_index_cache_file_ref_to_ref,
            score_index_cache_dir=args.score_index_cache_dir,
            disable_pkl_matrix_source=bool(args.disable_pkl_matrix_source),
            text_metrics_v212_dir=Path(args.text_metrics_v212_dir),
            ref_to_ref_cache_mode=str(args.ref_to_ref_cache_mode),
            ref_to_ref_cache_dir=Path(args.ref_to_ref_cache_dir),
            ref_to_ref_cache_warm_only=bool(args.ref_to_ref_cache_warm_only),
            target_fnames=targets,
            max_items=args.max_items,
            selection_index_range=selection_index_range,
            combination_bundle_dir=combination_bundle_dir,
            combination_bundle_scope=combination_bundle_scope,
            combination_bundle_include_candidate_lines=bool(args.combination_bundle_include_candidate_lines),
            shard_index=args.shard_index,
            levenshtein_backend=str(args.levenshtein_backend),
            workers=max(1, int(args.workers)),
            doc_workers=int(resolved_doc_workers),
            selected_run_items_override=selected_run_items_override,
            selected_document_count_override=selected_document_count_override,
            on_document_completed=on_document_completed,
            on_document_skipped=on_document_skipped,
            hough_preprocessing_config=hough_preprocessing_config,
            min_surviving_line_nls=args.min_surviving_line_nls,
            profile_combinations=bool(args.profile_combinations),
            selection_objective=str(args.selection_objective),
            write_combination_score_table=not bool(args.no_combination_score_table),
            log_fn=log,
        )
    except Exception as exc:
        if dynamic_document_pool is not None:
            failure_reason = f"worker failed before normal tuner outputs completed: {type(exc).__name__}: {exc}"
            for lease in list(dynamic_completed_leases_waiting_for_output):
                dynamic_document_pool.mark_lease_failed(lease, reason=failure_reason)
            for lease in list(dynamic_active_leases_by_document_index.values()):
                dynamic_document_pool.mark_lease_failed(lease, reason=failure_reason)
        raise

    if dynamic_document_pool is not None:
        # Mark documents done only after run_hough_parameter_sweeps has written
        # the normal JSON/CSV/bundle outputs.  The pool remains scheduling-only
        # while avoiding a false "done" state if a worker dies before export.
        for lease in dynamic_completed_leases_waiting_for_output:
            dynamic_document_pool.mark_lease_done(lease)
        if dynamic_active_leases_by_document_index:
            dangling_names = ", ".join(
                f"{lease.runfile_index}:{lease.fname}" for lease in dynamic_active_leases_by_document_index.values()
            )
            raise RuntimeError(f"Dynamic worker finished with active leases still claimed: {dangling_names}")

    summary_path = Path(result["summary_path"])
    if bool(result.get("mode") == "ref_to_ref_cache_warm_only"):
        log(f"[warm-only-done] summary_json={summary_path}")
        log(f"Summary JSON: {summary_path}")
        return

    if bool(args.with_visuals):
        log(f"[visuals-start] output_dir={Path(args.output_dir)} summary_json={summary_path}")
        try:
            from .tools.language_hough_parameter_metric_analysis import generate_tuner_visualisation_outputs
        except ImportError:
            from tools.language_hough_parameter_metric_analysis import generate_tuner_visualisation_outputs  # type: ignore

        visualisation_kwargs = {}
        if args.scores_pkl_ref_to_pred is not None:
            visualisation_kwargs["ref_to_pred_scores_pkl"] = Path(args.scores_pkl_ref_to_pred)
        if args.scores_pkl_ref_to_ref is not None:
            visualisation_kwargs["ref_to_ref_scores_pkl"] = Path(args.scores_pkl_ref_to_ref)

        visualisation_manifest = generate_tuner_visualisation_outputs(
            runfile_json=Path(args.runfile_json),
            tuner_output_dir=Path(args.output_dir),
            shards_dir=Path(args.output_dir),
            documents_per_shard=max(1, int(args.max_items or 1)),
            hide_line_labels=bool(args.hide_line_labels),
            **visualisation_kwargs,
        )
        result["visualisation_manifest_path"] = visualisation_manifest.get("manifest_path")
        result["generated_plot_root_dir"] = str(Path(args.output_dir) / "plots")
        log(f"[visuals-done] manifest={visualisation_manifest.get('manifest_path')}")
    else:
        result["summary_path"] = str(summary_path)
        log("[visuals-skip] pass --with-visuals to create final plots and visualisation bundles")

    log("")
    log(f"Summary JSON: {result['summary_path']}")
    if result.get("best_config_per_document_csv_path"):
        log(f"Best per-document configs CSV: {result['best_config_per_document_csv_path']}")
    if result.get("best_params_per_document_json_path"):
        log(f"Best per-document params JSON: {result['best_params_per_document_json_path']}")
    if result.get("all_documents_parameter_influence_csv_path"):
        log(f"All-doc parameter influence CSV: {result['all_documents_parameter_influence_csv_path']}")
    if result.get("skipped_documents_csv_path"):
        log(
            f"Skipped documents CSV: {result['skipped_documents_csv_path']} "
            f"count={int(result.get('skipped_document_count', 0))}"
        )
    score_table = result.get("combination_score_table", {})
    if isinstance(score_table, dict) and score_table.get("enabled"):
        log(f"Combination score table: {score_table.get('csv_gz_path')} rows={score_table.get('row_count')}")
    profiling = result.get("combination_profiling", {})
    if isinstance(profiling, dict) and profiling.get("enabled"):
        log(f"Combination profile CSV: {profiling.get('csv_path')} rows={profiling.get('row_count')}")
    if result.get("generated_plot_root_dir"):
        log(f"Plot root directory: {result['generated_plot_root_dir']}")
    bundle_logging = result.get("combination_bundle_logging", {})
    if isinstance(bundle_logging, dict) and bundle_logging.get("enabled"):
        log(f"Combination bundle directory: {bundle_logging.get('root_dir')}")

    parallel = result.get("parallelism", {})
    if isinstance(parallel, dict):
        log("Parallelism used:")
        log(f"  - scheduler_mode: {parallel.get('scheduler_mode', 'unknown')}")
        log(f"  - requested_threshold_workers_per_doc: {int(parallel.get('requested_threshold_workers_per_doc', 1))}")
        log(f"  - doc_workers: {int(parallel.get('doc_workers', 1))}")
        log(f"  - threshold_workers_per_doc: {int(parallel.get('threshold_workers_per_doc', 1))}")

    log("Best mean tuning score by parameter:")
    for param, payload in result["global_summary"]["best_by_parameter"].items():
        if payload is None:
            log(f"  - {param}: None")
            continue
        log(
            f"  - {param}: value={payload['value']} "
            f"mean_tuning_score={payload['mean_tuning_score']} "
            f"valid_docs={payload['valid_doc_count']}/{payload['doc_count']}"
        )


if __name__ == "__main__":
    main()
