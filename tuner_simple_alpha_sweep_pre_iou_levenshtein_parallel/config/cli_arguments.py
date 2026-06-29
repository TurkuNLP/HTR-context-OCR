from __future__ import annotations

"""Command-line parsing for the simple tuner."""

import argparse
from pathlib import Path

from .hough_parameters import ProbabilisticHoughParameters
from .pipeline_config import PipelineConfig, VALID_HARMONIC_MODES, VALID_PLOT_MODES


DEFAULT_RUNFILE_JSON = Path("/scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json")
DEFAULT_REF_TO_PRED_SCORES_PKL = Path(
    "/scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/old_scores_reference_prediction_ws50_st35_levenshtein.pkl"
)
DEFAULT_REF_TO_REF_SCORES_PKL = Path(
    "/scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/old_scores_reference_self_ws50_st35_levenshtein.pkl"
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without touching the filesystem."""

    parser = argparse.ArgumentParser(
        description="Run the simple Hough alignment pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runfile-json", type=Path, default=DEFAULT_RUNFILE_JSON)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scores-pkl-ref-to-pred", type=Path, default=DEFAULT_REF_TO_PRED_SCORES_PKL)
    parser.add_argument("--scores-pkl-ref-to-ref", type=Path, default=DEFAULT_REF_TO_REF_SCORES_PKL)
    parser.add_argument("--language", action="append", dest="languages", default=None)
    parser.add_argument("--document-type", action="append", dest="document_types", default=None)
    parser.add_argument("--all-languages", action="store_true")
    parser.add_argument("--all-document-types", action="store_true")
    parser.add_argument("--target-fname", action="append", dest="target_fnames", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-stride", type=int, default=35)
    parser.add_argument("--minimum-matrix-rows", type=int, default=4)
    parser.add_argument("--minimum-matrix-columns", type=int, default=4)
    parser.add_argument("--score-floor-alpha", type=float, default=1.0)
    parser.set_defaults(alpha_sweep_enabled=True)
    parser.add_argument("--alpha-sweep", dest="alpha_sweep_enabled", action="store_true")
    parser.add_argument("--no-alpha-sweep", dest="alpha_sweep_enabled", action="store_false")
    parser.add_argument("--alpha-sweep-min", type=float, default=1.0)
    parser.add_argument("--alpha-sweep-max", type=float, default=4.0)
    parser.add_argument("--alpha-sweep-step", type=float, default=0.2)
    parser.add_argument("--minimum-pre-hough-levenshtein", type=float, default=None)
    parser.add_argument("--hough-threshold", type=int, default=25)
    parser.add_argument("--hough-line-length", type=int, default=35)
    parser.add_argument("--hough-line-gap", type=int, default=15)
    parser.add_argument("--hough-seed", type=int, default=None,
        help="Integer seed for probabilistic Hough. When omitted, skimage's own PCG64 generator is used (non-deterministic).")
    parser.add_argument("--hough-num-runs", type=int, default=1,
        help=(
            "Number of independent ref-to-pred Hough runs per alpha candidate. "
            "Each run uses the same mask and parameters but a different seed; "
            "the union of all detected segments feeds the downstream filter. "
            "Default 1 preserves the original single-run behaviour."
        )
    )
    parser.add_argument("--align-min-iou-threshold", type=float, default=0.035)
    parser.add_argument("--min-surviving-line-nls", type=float, default=0.5)
    parser.add_argument("--plot-mode", choices=sorted(VALID_PLOT_MODES), default="stitched-language")
    parser.add_argument("--show-line-ids", action="store_true")
    parser.add_argument("--stitched-panel-columns", type=int, default=3)
    parser.add_argument("--saved-figure-dpi", type=int, default=140)
    parser.add_argument("--dynamic-document-pool-dir", type=Path, default=None)
    parser.add_argument("--dynamic-worker-id", type=str, default=None)
    parser.add_argument("--atomic-output-dir", type=Path, default=None)
    parser.add_argument("--result-bucket-size", type=int, default=20)
    parser.add_argument("--result-bucket-seconds", type=float, default=60.0)
    parser.add_argument(
        "--suppress-output-files",
        action="store_true",
        default=False,
        help=(
            "When set, no files or directories are written to --output-dir. "
            "The pipeline runs entirely in memory: no CSVs, no per-document PKL pickles, "
            "no plots, and no output directory creation. "
            "All scoring results are still computed and logged to stdout. "
            "Implies --plot-mode none."
        ),
    )
    parser.add_argument(
        "--harmonic-mode",
        choices=sorted(VALID_HARMONIC_MODES),
        default="balanced",
        help=(
            "Formula used to score and select the best alpha candidate. "
            "'balanced' applies equal weight to NLS, coverage, and non-hallucination (default). "
            "'coverage-hallucination-priority' doubles the weight on coverage and non-hallucination. "
            "'coverage-hallucination-only' excludes NLS and selects on coverage and non-hallucination alone. "
            "'nls-priority' doubles the weight on NLS text similarity vs. coverage and non-hallucination. "
            "A subdirectory named after the chosen mode is automatically created inside --output-dir."
        ),
    )
    parser.add_argument("--two-phase", dest="two_phase_enabled", action="store_true")
    parser.add_argument("--no-two-phase", dest="two_phase_enabled", action="store_false")
    parser.set_defaults(two_phase_enabled=True)
    parser.add_argument(
        "--scout-hough-runs",
        type=int,
        default=3,
        help="Hough runs per alpha in the scout pass (cheap ranking). Refine uses --hough-num-runs.",
    )
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=5,
        help="How many top scout alphas are re-run at full --hough-num-runs and selected from.",
    )
    parser.add_argument(
        "--alpha-parallel-workers",
        type=int,
        default=0,
        help=(
            "Worker processes for parallel alpha candidate evaluation. "
            "0 = auto = max(1, floor(cpus/2)). "
            "Set --cpus-per-task in the SLURM header to control the pool size."
        ),
    )
    return parser


def parse_pipeline_config(argv: list[str] | None = None) -> PipelineConfig:
    """Parse CLI arguments into one validated pipeline configuration object."""

    args = build_argument_parser().parse_args(argv)
    languages = tuple() if args.all_languages else tuple(args.languages or ())
    document_types = tuple() if args.all_document_types else tuple(args.document_types or ())
    min_line_nls = None if args.min_surviving_line_nls is None or args.min_surviving_line_nls <= 0.0 else float(args.min_surviving_line_nls)
    harmonic_mode = str(args.harmonic_mode)
    # The harmonic-mode subdirectory is only meaningful during alpha sweep, where the formula
    # determines which candidate is selected. When --minimum-pre-hough-levenshtein is set there
    # is exactly one candidate and no selection, so results go directly into --output-dir.
    if args.minimum_pre_hough_levenshtein is None:
        output_dir = Path(args.output_dir) / harmonic_mode
    else:
        output_dir = Path(args.output_dir)
    config = PipelineConfig(
        runfile_json=Path(args.runfile_json),
        output_dir=output_dir,
        scores_pkl_ref_to_pred=Path(args.scores_pkl_ref_to_pred),
        scores_pkl_ref_to_ref=Path(args.scores_pkl_ref_to_ref),
        languages=languages,
        document_types=document_types,
        target_fnames=tuple(args.target_fnames or ()),
        max_items=args.max_items,
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        minimum_matrix_rows=int(args.minimum_matrix_rows),
        minimum_matrix_columns=int(args.minimum_matrix_columns),
        score_floor_alpha=float(args.score_floor_alpha),
        alpha_sweep_enabled=bool(args.alpha_sweep_enabled),
        alpha_sweep_min=float(args.alpha_sweep_min),
        alpha_sweep_max=float(args.alpha_sweep_max),
        alpha_sweep_step=float(args.alpha_sweep_step),
        minimum_pre_hough_levenshtein=None if args.minimum_pre_hough_levenshtein is None else float(args.minimum_pre_hough_levenshtein),
        hough_parameters=ProbabilisticHoughParameters(
            hough_threshold=int(args.hough_threshold),
            hough_line_length=int(args.hough_line_length),
            hough_line_gap=int(args.hough_line_gap),
            hough_seed=None if args.hough_seed is None else int(args.hough_seed),
            hough_num_runs=int(args.hough_num_runs),
        ),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
        min_surviving_line_nls=min_line_nls,
        plot_mode=str(args.plot_mode),
        show_line_ids=bool(args.show_line_ids),
        stitched_panel_columns=int(args.stitched_panel_columns),
        saved_figure_dpi=int(args.saved_figure_dpi),
        dynamic_document_pool_dir=Path(args.dynamic_document_pool_dir) if args.dynamic_document_pool_dir is not None else None,
        dynamic_worker_id=str(args.dynamic_worker_id) if args.dynamic_worker_id is not None else None,
        atomic_output_dir=Path(args.atomic_output_dir) if args.atomic_output_dir is not None else None,
        result_bucket_size=int(args.result_bucket_size),
        result_bucket_seconds=float(args.result_bucket_seconds),
        alpha_selection_harmonic_mode=harmonic_mode,
        suppress_output_files=bool(args.suppress_output_files),
        two_phase_enabled=bool(args.two_phase_enabled),
        scout_hough_runs=int(args.scout_hough_runs),
        refine_top_k=int(args.refine_top_k),
        alpha_parallel_workers=int(args.alpha_parallel_workers),
    )
    return config.validate()


__all__ = ["build_argument_parser", "parse_pipeline_config"]
