from __future__ import annotations

"""Command-line parsing for the simple serial tuner."""

import argparse
from pathlib import Path

from .hough_parameters import ProbabilisticHoughParameters
from .pipeline_config import PipelineConfig, VALID_PLOT_MODES

# Compute or store DEFAULT_RUNFILE_JSON so later code can reuse this named value clearly.
DEFAULT_RUNFILE_JSON = Path(
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "/scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/outputs.json"
)
# Compute or store DEFAULT_REF_TO_PRED_SCORES_PKL so later code can reuse this named value clearly.
DEFAULT_REF_TO_PRED_SCORES_PKL = Path(
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "/scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_pred/"
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "old_scores_reference_prediction_ws50_st35_levenshtein.pkl"
)
# Compute or store DEFAULT_REF_TO_REF_SCORES_PKL so later code can reuse this named value clearly.
DEFAULT_REF_TO_REF_SCORES_PKL = Path(
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "/scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/ref_to_ref/"
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "old_scores_reference_self_ws50_st35_levenshtein.pkl"
)


# Define the build_argument_parser function; its body below performs one named step of the pipeline.
def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without touching the filesystem."""
    # Compute or store parser so later code can reuse this named value clearly.
    parser = argparse.ArgumentParser(
        # Pass the description argument into the surrounding call so the callee receives that setting explicitly.
        description="Run the simple serial Hough alignment pipeline.",
        # Pass the formatter_class argument into the surrounding call so the callee receives that setting explicitly.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--runfile-json", type=Path, default=DEFAULT_RUNFILE_JSON)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--output-dir", type=Path, required=True)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--scores-pkl-ref-to-pred", type=Path, default=DEFAULT_REF_TO_PRED_SCORES_PKL)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--scores-pkl-ref-to-ref", type=Path, default=DEFAULT_REF_TO_REF_SCORES_PKL)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--language", action="append", dest="languages", default=None)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--document-type", action="append", dest="document_types", default=None)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--target-fname", action="append", dest="target_fnames", default=None)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--max-items", type=int, default=None)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--window-size", type=int, default=50)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--window-stride", type=int, default=35)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--minimum-matrix-rows", type=int, default=4)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--minimum-matrix-columns", type=int, default=4)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--score-floor-alpha", type=float, default=1.0)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--hough-threshold", type=int, default=25)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--hough-line-length", type=int, default=35)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--hough-line-gap", type=int, default=15)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--hough-seed", type=int, default=1)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--align-abs-min-len", type=float, default=0.0)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--align-min-iou-threshold", type=float, default=0.035)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--min-surviving-line-nls", type=float, default=0.5)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--plot-mode", choices=sorted(VALID_PLOT_MODES), default="stitched-language")
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--show-line-ids", action="store_true")
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--stitched-panel-columns", type=int, default=3)
    # Register one command-line option so users can control this pipeline behavior from the shell.
    parser.add_argument("--saved-figure-dpi", type=int, default=140)
    # Return this computed value to the caller so the next pipeline stage can use it.
    return parser


# Define the parse_pipeline_config function; its body below performs one named step of the pipeline.
def parse_pipeline_config(argv: list[str] | None = None) -> PipelineConfig:
    """Parse CLI arguments into one validated pipeline configuration object."""
    # Compute or store args so later code can reuse this named value clearly.
    args = build_argument_parser().parse_args(argv)
    # Compute or store min_line_nls so later code can reuse this named value clearly.
    min_line_nls = None if args.min_surviving_line_nls is None or args.min_surviving_line_nls <= 0.0 else float(args.min_surviving_line_nls)
    # Compute or store config so later code can reuse this named value clearly.
    config = PipelineConfig(
        # Pass runfile_json into the surrounding call; this supplies the JSON file that lists source documents, references, predictions, languages, and document types.
        runfile_json=Path(args.runfile_json),
        # Pass output_dir into the surrounding call; this supplies the directory where CSV, JSON, and optional plot files will be written.
        output_dir=Path(args.output_dir),
        # Pass scores_pkl_ref_to_pred into the surrounding call; this supplies the precomputed reference-to-prediction score matrix file.
        scores_pkl_ref_to_pred=Path(args.scores_pkl_ref_to_pred),
        # Pass scores_pkl_ref_to_ref into the surrounding call; this supplies the precomputed reference-to-reference score matrix file.
        scores_pkl_ref_to_ref=Path(args.scores_pkl_ref_to_ref),
        # Pass languages into the surrounding call; this supplies the optional language filter requested by the user.
        languages=tuple(args.languages or ()),
        # Pass document_types into the surrounding call; this supplies the optional document-type filter requested by the user.
        document_types=tuple(args.document_types or ()),
        # Pass target_fnames into the surrounding call; this supplies the optional exact filename filter requested by the user.
        target_fnames=tuple(args.target_fnames or ()),
        # Pass max_items into the surrounding call; this supplies the optional cap on how many selected documents are processed.
        max_items=args.max_items,
        # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
        window_size=int(args.window_size),
        # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
        window_stride=int(args.window_stride),
        # Pass minimum_matrix_rows into the surrounding call; this supplies the smallest allowed reference-window count before the document is skipped.
        minimum_matrix_rows=int(args.minimum_matrix_rows),
        # Pass minimum_matrix_columns into the surrounding call; this supplies the smallest allowed prediction-window count before the document is skipped.
        minimum_matrix_columns=int(args.minimum_matrix_columns),
        # Pass score_floor_alpha into the surrounding call; this supplies the user-selected multiplier that controls how strongly the standard deviation raises the floor.
        score_floor_alpha=float(args.score_floor_alpha),
        # Compute or store hough_parameters so later code can reuse this named value clearly.
        hough_parameters=ProbabilisticHoughParameters(
            # Pass the hough_threshold argument into the surrounding call so the callee receives that setting explicitly.
            hough_threshold=int(args.hough_threshold),
            # Pass the hough_line_length argument into the surrounding call so the callee receives that setting explicitly.
            hough_line_length=int(args.hough_line_length),
            # Pass the hough_line_gap argument into the surrounding call so the callee receives that setting explicitly.
            hough_line_gap=int(args.hough_line_gap),
            # Pass the hough_seed argument into the surrounding call so the callee receives that setting explicitly.
            hough_seed=int(args.hough_seed),
        ),
        # Pass align_abs_min_len into the surrounding call; this supplies the absolute line-length filter passed into the existing line filtering code.
        align_abs_min_len=float(args.align_abs_min_len),
        # Pass align_min_iou_threshold into the surrounding call; this supplies the overlap threshold used when assigning line coverage to text windows.
        align_min_iou_threshold=float(args.align_min_iou_threshold),
        # Pass min_surviving_line_nls into the surrounding call; this supplies the minimum line-level normalised Levenshtein similarity required after Hough filtering.
        min_surviving_line_nls=min_line_nls,
        # Pass plot_mode into the surrounding call; this supplies whether plots are skipped, stitched only, or stitched while keeping per-document panels.
        plot_mode=str(args.plot_mode),
        # Pass show_line_ids into the surrounding call; this supplies whether raw and final line labels are printed on plot overlays.
        show_line_ids=bool(args.show_line_ids),
        # Pass stitched_panel_columns into the surrounding call; this supplies how many document panels appear in one row of each stitched language image.
        stitched_panel_columns=int(args.stitched_panel_columns),
        # Pass saved_figure_dpi into the surrounding call; this supplies the resolution used when saving plot images.
        saved_figure_dpi=int(args.saved_figure_dpi),
    )
    # Return this computed value to the caller so the next pipeline stage can use it.
    return config.validate()


__all__ = ["build_argument_parser", "parse_pipeline_config"]
