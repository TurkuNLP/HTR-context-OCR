from __future__ import annotations

"""Command-line parsing for the simple tuner."""

import argparse
from pathlib import Path

from .hough_parameters import ProbabilisticHoughParameters
from .pipeline_config import PipelineConfig, VALID_PLOT_MODES


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
    parser.add_argument("--hough-threshold", type=int, default=25)
    parser.add_argument("--hough-line-length", type=int, default=35)
    parser.add_argument("--hough-line-gap", type=int, default=15)
    parser.add_argument("--hough-seed", type=int, default=1)
    parser.add_argument("--align-abs-min-len", type=float, default=0.0)
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
    return parser


def parse_pipeline_config(argv: list[str] | None = None) -> PipelineConfig:
    """Parse CLI arguments into one validated pipeline configuration object."""

    args = build_argument_parser().parse_args(argv)
    languages = tuple() if args.all_languages else tuple(args.languages or ())
    document_types = tuple() if args.all_document_types else tuple(args.document_types or ())
    min_line_nls = None if args.min_surviving_line_nls is None or args.min_surviving_line_nls <= 0.0 else float(args.min_surviving_line_nls)
    config = PipelineConfig(
        runfile_json=Path(args.runfile_json),
        output_dir=Path(args.output_dir),
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
        hough_parameters=ProbabilisticHoughParameters(
            hough_threshold=int(args.hough_threshold),
            hough_line_length=int(args.hough_line_length),
            hough_line_gap=int(args.hough_line_gap),
            hough_seed=int(args.hough_seed),
        ),
        align_abs_min_len=float(args.align_abs_min_len),
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
    )
    return config.validate()


__all__ = ["build_argument_parser", "parse_pipeline_config"]
