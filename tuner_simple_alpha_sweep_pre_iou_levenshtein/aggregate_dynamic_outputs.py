#!/usr/bin/env python3
from __future__ import annotations

"""Command-line entry point for final aggregation after dynamic workers finish."""

import argparse
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
TUNER_SIMPLE_DIR = CURRENT_FILE.parent
PROJECT_DIR = TUNER_SIMPLE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(TUNER_SIMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(TUNER_SIMPLE_DIR))

from tuner_simple_alpha_sweep_pre_iou_levenshtein.logging_utils.timestamped_logging import build_timestamped_logger  # noqa: E402
from tuner_simple_alpha_sweep_pre_iou_levenshtein.results_writing.dynamic_result_aggregation import aggregate_dynamic_worker_outputs  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate tuner_simple dynamic worker outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-csv", type=Path, default=None)
    parser.add_argument("--plot-mode", choices=("none", "stitched-language", "stitched-language-and-document-grids"), default="stitched-language")
    parser.add_argument("--stitched-panel-columns", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    aggregate_dynamic_worker_outputs(
        output_dir=arguments.output_dir,
        progress_csv_path=arguments.progress_csv,
        plot_mode=str(arguments.plot_mode),
        stitched_panel_columns=int(arguments.stitched_panel_columns),
        log=build_timestamped_logger(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
