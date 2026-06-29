#!/usr/bin/env python3
from __future__ import annotations

"""Command-line entry point for the simple Hough tuner."""

from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
TUNER_SIMPLE_DIR = CURRENT_FILE.parent
PROJECT_DIR = TUNER_SIMPLE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(TUNER_SIMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(TUNER_SIMPLE_DIR))

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.config.cli_arguments import parse_pipeline_config  # noqa: E402
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.logging_utils.timestamped_logging import build_timestamped_logger  # noqa: E402
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.runtime.runtime_paths import ensure_runtime_paths  # noqa: E402
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.serial_runner.dynamic_worker_runner import run_atomic_document_worker  # noqa: E402
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.serial_runner.pipeline_runner import run_simple_tuner  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the selected pipeline mode, and return an exit code."""

    ensure_runtime_paths()
    config = parse_pipeline_config(argv)
    log = build_timestamped_logger()
    if config.dynamic_document_pool_dir is not None:
        run_atomic_document_worker(config, log=log)
    else:
        run_simple_tuner(config, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
