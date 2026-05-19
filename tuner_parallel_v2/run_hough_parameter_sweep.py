#!/usr/bin/env python3
from __future__ import annotations

"""CLI entry point for the Hough parameter tuner.

Users provide input/output paths, optional inclusive Hough ranges, and
parallelism settings.  The runner executes the sweep, writes CSV/JSON artifacts,
and generates per-document plots automatically.
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
except ImportError:
    from logging_utils.timestamped_logging import build_timestamped_logger  # type: ignore
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
        help="Read-only text_metrics_v2_12_parallel directory used for coverage/hallucination metrics",
    )
    parser.add_argument(
        "--ref-to-ref-cache-mode",
        type=str,
        default="auto",
        choices=("off", "auto", "read-only"),
        help=(
            "Persistent reference-self combination cache mode. auto reads/writes exact "
            "refref_y coverage baselines; off preserves the old recompute-every-combination path."
        ),
    )
    parser.add_argument(
        "--ref-to-ref-cache-dir",
        type=Path,
        default=DEFAULT_REF_TO_REF_COMBO_CACHE_DIR,
        help="Directory for exact ref_to_ref per-combination cache artifacts",
    )
    parser.add_argument(
        "--ref-to-ref-cache-warm-only",
        action="store_true",
        help=(
            "Fill the exact ref_to_ref combination cache and exit before the "
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

    parser.add_argument("--hough-seed", type=int, default=HOUGH_SEED_MIN, help="Deprecated compatibility arg; fixed seed 1 is used")
    parser.add_argument("--hough-start", type=float, default=2.6, help="Adaptive Hough mask threshold start")
    parser.add_argument("--align-abs-min-len", type=float, default=8.0, help="Minimum candidate line length before IoU filtering")
    parser.add_argument("--align-min-iou-threshold", type=float, default=0.035, help="Minimum true-IoU threshold for line filtering")

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
            "Deprecated compatibility arg. Seed sweep is temporarily disabled; "
            f"fixed seed {HOUGH_SEED_MIN} is used."
        ),
    )
    parser.add_argument(
        "--combination-bundle-dir",
        type=Path,
        default=None,
        help="Optional directory for per-combination JSONL geometry bundles used by visualization tools",
    )
    parser.add_argument(
        "--combination-bundle-scope",
        type=str,
        default="none",
        choices=("none", "all", "valid-only", "invalid-only"),
        help="Which evaluated combinations should be written to --combination-bundle-dir",
    )
    parser.add_argument(
        "--combination-bundle-include-candidate-lines",
        action="store_true",
        help="Include pre-filter candidate line records in every combination bundle",
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
            "Generate final colour visualisations after the sweep. This forces "
            "full per-combination bundle logging so every graph grid can be rebuilt."
        ),
    )
    parser.add_argument(
        "--hide-line-labels",
        action="store_true",
        help="Hide raw Hough and final surviving-line labels in stitched best-combination visuals.",
    )

    return parser.parse_args()


def _format_path(value: Path | None) -> str:
    """Return a stable human-readable string for optional paths."""
    return "None" if value is None else str(Path(value))


def main() -> None:
    """Run the tuner and immediately generate plots."""
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
        hough_start=float(args.hough_start),
        align_abs_min_len=float(args.align_abs_min_len),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
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
    log(f"[v2.12-metrics] text_metrics_v212_dir={Path(args.text_metrics_v212_dir)}")
    log(
        "[ref-to-ref-cache] "
        f"mode={args.ref_to_ref_cache_mode} dir={Path(args.ref_to_ref_cache_dir)} "
        f"warm_only={bool(args.ref_to_ref_cache_warm_only)}"
    )
    if selection_index_range is not None:
        log(f"[selection] selection_index_range={selection_index_range[0]}..{selection_index_range[1]}")

    combination_bundle_scope = str(args.combination_bundle_scope)
    combination_bundle_dir = args.combination_bundle_dir
    if bool(args.with_visuals):
        # Visual mode needs every evaluated combination so the language/type
        # analysis can build all per-document graph grids without recomputing.
        combination_bundle_scope = "all"
        if combination_bundle_dir is None:
            combination_bundle_dir = Path(args.output_dir) / "combination_bundles"
        log(
            "[visuals] enabled; forcing combination_bundle_scope=all "
            f"bundle_dir={Path(combination_bundle_dir)} "
            f"hide_line_labels={bool(args.hide_line_labels)}"
        )

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
        doc_workers=max(1, int(args.doc_workers)),
        log_fn=log,
    )

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
        log("[visuals-skip] pass --with-visuals to create final plots and full combination bundles")

    log("")
    log(f"Summary JSON: {result['summary_path']}")
    if result.get("best_config_per_document_csv_path"):
        log(f"Best per-document configs CSV: {result['best_config_per_document_csv_path']}")
    if result.get("best_params_per_document_json_path"):
        log(f"Best per-document params JSON: {result['best_params_per_document_json_path']}")
    if result.get("all_documents_parameter_influence_csv_path"):
        log(f"All-doc parameter influence CSV: {result['all_documents_parameter_influence_csv_path']}")
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
