#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime_paths import ensure_tuner_runtime_paths  # type: ignore

_SCRIPT_DIR, _PROJECT_ROOT, _SHARED_METRICS_DIR = ensure_tuner_runtime_paths()

try:
    from .plot_hough_parameter_sweep import generate_plots_for_summary_json
    from .tuner_core import (
        BACKEND_C,
        DEFAULT_SCORE_INDEX_CACHE_DIR,
        HOUGH_LINE_GAP_MAX,
        HOUGH_LINE_GAP_MIN,
        HOUGH_LINE_LENGTH_MAX,
        HOUGH_LINE_LENGTH_MIN,
        HOUGH_THRESHOLD_MAX,
        HOUGH_THRESHOLD_MIN,
        HoughBaselineConfig,
        SUPPORTED_BACKENDS,
        run_hough_parameter_sweeps,
    )
except ImportError:
    from plot_hough_parameter_sweep import generate_plots_for_summary_json  # type: ignore
    from tuner_core import (  # type: ignore
        BACKEND_C,
        DEFAULT_SCORE_INDEX_CACHE_DIR,
        HOUGH_LINE_GAP_MAX,
        HOUGH_LINE_GAP_MIN,
        HOUGH_LINE_LENGTH_MAX,
        HOUGH_LINE_LENGTH_MIN,
        HOUGH_THRESHOLD_MAX,
        HOUGH_THRESHOLD_MIN,
        HoughBaselineConfig,
        SUPPORTED_BACKENDS,
        run_hough_parameter_sweeps,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the fixed-range tuner runner."""
    default_cache_dir = _SCRIPT_DIR / "_matrix_cache"

    parser = argparse.ArgumentParser(
        description=(
            "Fixed-range document tuner. Automatically loops: "
            "threshold 1..40, line_length 1..50, line_gap 1..30. "
            "Seed is fixed from --hough-seed. "
            "Hough line direction is constrained to strict 30<x<60 degree diagonal bands."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON/CSV/plots")

    parser.add_argument("--window-size", type=int, default=50, help="Sliding window size")
    parser.add_argument("--window-stride", type=int, default=35, help="Sliding window stride")

    parser.add_argument(
        "--matrix-cache-dir",
        type=Path,
        default=default_cache_dir,
        help="On-disk cache directory for score matrices (reused across runs).",
    )
    parser.add_argument(
        "--no-matrix-cache",
        action="store_true",
        help="Disable on-disk score-matrix caching for this run.",
    )

    # Optional read-only matrix source from text_metrics score streams.
    parser.add_argument(
        "--scores-pkl-ref-to-pred",
        type=Path,
        default=None,
        help=(
            "Optional path to scores_reference_prediction_ws*_st*.pkl. "
            "When provided, tuner loads matrices from this stream in read-only mode before recompute."
        ),
    )
    parser.add_argument(
        "--score-index-cache-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit read-only index cache file (*.index.pkl). "
            "If omitted, tuner tries --score-index-cache-dir with deterministic cache naming."
        ),
    )
    parser.add_argument(
        "--score-index-cache-dir",
        type=Path,
        default=DEFAULT_SCORE_INDEX_CACHE_DIR,
        help=(
            "Directory containing read-only score-stream index caches from text_metrics_v2_1_parallel."
        ),
    )
    parser.add_argument(
        "--disable-pkl-matrix-source",
        action="store_true",
        help="Disable read-only scores.pkl matrix loading even if --scores-pkl-ref-to-pred is provided.",
    )

    parser.add_argument(
        "--target-fname",
        action="append",
        default=[],
        help="Optional exact/basename target file. Repeat flag to include multiple files.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on number of documents")
    parser.add_argument(
        "--levenshtein-backend",
        type=str,
        default=BACKEND_C,
        choices=tuple(SUPPORTED_BACKENDS),
        help="Levenshtein backend. 'c' uses rapidfuzz exact C-backed distance.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Threshold-worker count per document (auto-forced to 40 when --doc-workers > 1).",
    )
    parser.add_argument(
        "--doc-workers",
        type=int,
        default=1,
        help="How many documents to process in parallel.",
    )

    # Fixed-grid settings for non-ranged parameters.
    parser.add_argument("--hough-seed", type=int, default=0, help="Fixed seed used for all evaluations")
    parser.add_argument("--hough-start", type=float, default=2.6)
    parser.add_argument("--align-abs-min-len", type=float, default=8.0)
    parser.add_argument("--align-min-iou-threshold", type=float, default=0.035)

    return parser.parse_args()


def _format_path(value: Path | None) -> str:
    """Human-readable path formatter for console summaries."""
    return "None" if value is None else str(Path(value))


def main() -> None:
    """Execute fixed-grid tuning, then build plots and print concise run summary."""
    args = parse_args()

    baseline = HoughBaselineConfig(
        # These three are overwritten by fixed loops in tuner_core.
        hough_threshold=10,
        hough_line_length=8,
        hough_line_gap=8,
        hough_seed=int(args.hough_seed),
        hough_start=float(args.hough_start),
        align_abs_min_len=float(args.align_abs_min_len),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
    )

    targets = [str(v) for v in args.target_fname if str(v).strip()]
    matrix_cache_dir = None if bool(args.no_matrix_cache) else Path(args.matrix_cache_dir)

    print(
        "[grid-fixed] "
        f"threshold={HOUGH_THRESHOLD_MIN}..{HOUGH_THRESHOLD_MAX} "
        f"line_length={HOUGH_LINE_LENGTH_MIN}..{HOUGH_LINE_LENGTH_MAX} "
        f"line_gap={HOUGH_LINE_GAP_MIN}..{HOUGH_LINE_GAP_MAX} "
        f"seed={baseline.hough_seed} threshold_workers={max(1, int(args.workers))} doc_workers={max(1, int(args.doc_workers))} "
        "line_angle=strict(30,60)"
    )
    print(
        "[matrix-sources] "
        f"npz_cache={'enabled' if matrix_cache_dir is not None else 'disabled'} "
        f"scores_pkl_ref_to_pred={_format_path(args.scores_pkl_ref_to_pred)} "
        f"score_index_cache_file={_format_path(args.score_index_cache_file)} "
        f"score_index_cache_dir={_format_path(args.score_index_cache_dir)} "
        f"disable_pkl_matrix_source={bool(args.disable_pkl_matrix_source)}"
    )

    result = run_hough_parameter_sweeps(
        runfile_json=Path(args.runfile_json),
        output_dir=Path(args.output_dir),
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        baseline_cfg=baseline,
        matrix_cache_dir=matrix_cache_dir,
        scores_pkl_ref_to_pred=args.scores_pkl_ref_to_pred,
        score_index_cache_file=args.score_index_cache_file,
        score_index_cache_dir=args.score_index_cache_dir,
        disable_pkl_matrix_source=bool(args.disable_pkl_matrix_source),
        target_fnames=targets,
        max_items=args.max_items,
        levenshtein_backend=str(args.levenshtein_backend),
        workers=max(1, int(args.workers)),
        doc_workers=max(1, int(args.doc_workers)),
        log_fn=print,
    )

    summary_path = Path(result["summary_path"])
    result = generate_plots_for_summary_json(summary_json=summary_path)
    result["summary_path"] = str(summary_path)

    print()
    print(f"Summary JSON: {result['summary_path']}")
    if result.get("best_config_per_document_csv_path"):
        print(f"Best per-document configs CSV: {result['best_config_per_document_csv_path']}")
    if result.get("best_params_per_document_json_path"):
        print(f"Best per-document params JSON: {result['best_params_per_document_json_path']}")
    if result.get("all_documents_parameter_influence_csv_path"):
        print(f"All-doc parameter influence CSV: {result['all_documents_parameter_influence_csv_path']}")
    if "all_documents_parameter_influence_row_count" in result:
        print(f"All-doc parameter influence row count: {int(result.get('all_documents_parameter_influence_row_count', 0))}")
    if result.get("generated_plot_root_dir"):
        print(f"Plot root directory: {result['generated_plot_root_dir']}")

    parallel = result.get("parallelism", {})
    if isinstance(parallel, dict):
        print("Parallelism used:")
        if "requested_threshold_workers_per_doc" in parallel:
            print(f"  - requested_threshold_workers_per_doc: {int(parallel.get('requested_threshold_workers_per_doc', 1))}")
        print(f"  - doc_workers: {int(parallel.get('doc_workers', 1))}")
        print(f"  - threshold_workers_per_doc: {int(parallel.get('threshold_workers_per_doc', 1))}")

    print("Best mean along-lines NLS by parameter:")
    for param, payload in result["global_summary"]["best_by_parameter"].items():
        if payload is None:
            print(f"  - {param}: None")
            continue
        print(
            f"  - {param}: value={payload['value']} "
            f"mean_along_lines_nls={payload['mean_along_lines_nls']} "
            f"valid_docs={payload['valid_doc_count']}/{payload['doc_count']}"
        )

    by_doc_paths = result.get("generated_plot_paths_by_document", {})
    if isinstance(by_doc_paths, dict) and by_doc_paths:
        print("Generated plots by document:")
        for doc_name, per_param in by_doc_paths.items():
            if not isinstance(per_param, dict):
                continue
            print(f"  - {doc_name}")
            for parameter, path in per_param.items():
                print(f"      {parameter}: {path}")

    timings = result.get("timings", {})
    if isinstance(timings, dict):
        print("Timing breakdown (seconds):")
        print(f"  - run_total: {float(timings.get('run_total_seconds', 0.0)):.3f}")
        print(f"  - grid_evaluation: {float(timings.get('grid_evaluation_seconds', 0.0)):.3f}")
        print(f"  - profile_aggregation: {float(timings.get('profile_aggregation_seconds', 0.0)):.3f}")
        print(f"  - sweep_total: {float(timings.get('sweep_total_seconds', 0.0)):.3f}")
        print(f"  - non_sweep_non_load: {float(timings.get('non_sweep_non_load_seconds', 0.0)):.3f}")

        load = timings.get("load_documents", {})
        if isinstance(load, dict):
            print("Load-documents timing and source stats:")
            print(f"  - load_total: {float(load.get('load_documents_total_seconds', 0.0)):.3f}")
            print(f"  - matrix_total: {float(load.get('matrix_total_seconds', 0.0)):.3f}")
            print(f"  - matrix_compute: {float(load.get('matrix_compute_seconds', 0.0)):.3f}")
            print(f"  - matrix_cache_load: {float(load.get('matrix_cache_load_seconds', 0.0)):.3f}")
            print(f"  - matrix_cache_store: {float(load.get('matrix_cache_store_seconds', 0.0)):.3f}")
            print(f"  - matrix_pkl_load: {float(load.get('matrix_pkl_load_seconds', 0.0)):.3f}")
            print(f"  - pkl_index_prepare: {float(load.get('pkl_index_prepare_seconds', 0.0)):.3f}")
            print(f"  - whole_doc_nls: {float(load.get('whole_document_nls_seconds', 0.0)):.3f}")
            print(f"  - precompute_blocks: {float(load.get('precompute_blocks_seconds', 0.0)):.3f}")

            print("Load-documents matrix source counts:")
            print(f"  - matrix_source_npz_hits: {int(load.get('matrix_source_npz_hits', 0))}")
            print(f"  - matrix_source_pkl_hits: {int(load.get('matrix_source_pkl_hits', 0))}")
            print(f"  - matrix_source_computed: {int(load.get('matrix_source_computed', 0))}")

            print("Read-only scores.pkl diagnostics:")
            print(f"  - scores_pkl_enabled: {bool(load.get('scores_pkl_enabled', False))}")
            print(f"  - scores_pkl_disabled_reason: {load.get('scores_pkl_disabled_reason')}")
            print(f"  - scores_pkl_index_source: {load.get('scores_pkl_index_source')}")
            print(f"  - scores_pkl_index_entry_count: {int(load.get('scores_pkl_index_entry_count', 0))}")
            print(f"  - scores_pkl_lookup_misses: {int(load.get('scores_pkl_lookup_misses', 0))}")
            print(
                "  - scores_pkl_ref_text_mismatch_count: "
                f"{int(load.get('scores_pkl_ref_text_mismatch_count', 0))}"
            )
            print(
                "  - scores_pkl_pred_text_mismatch_count: "
                f"{int(load.get('scores_pkl_pred_text_mismatch_count', 0))}"
            )
            print(
                "  - scores_pkl_shape_mismatch_count: "
                f"{int(load.get('scores_pkl_shape_mismatch_count', 0))}"
            )
            print(f"  - scores_pkl_other_failure_count: {int(load.get('scores_pkl_other_failure_count', 0))}")


if __name__ == "__main__":
    main()
