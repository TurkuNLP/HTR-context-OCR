#!/usr/bin/env python3
from __future__ import annotations

"""Command-line entry point for the simple serial Hough tuner."""

from pathlib import Path
import sys

# Compute or store CURRENT_FILE so later code can reuse this named value clearly.
CURRENT_FILE = Path(__file__).resolve()
# Compute or store TUNER_SIMPLE_DIR so later code can reuse this named value clearly.
TUNER_SIMPLE_DIR = CURRENT_FILE.parent
# Compute or store PROJECT_DIR so later code can reuse this named value clearly.
PROJECT_DIR = TUNER_SIMPLE_DIR.parent
# Check whether str(PROJECT_DIR) not in sys.path; the indented block handles that specific case.
if str(PROJECT_DIR) not in sys.path:
    # Execute this statement as the next small step in the surrounding pipeline logic.
    sys.path.insert(0, str(PROJECT_DIR))
# Check whether str(TUNER_SIMPLE_DIR) not in sys.path; the indented block handles that specific case.
if str(TUNER_SIMPLE_DIR) not in sys.path:
    # Execute this statement as the next small step in the surrounding pipeline logic.
    sys.path.insert(0, str(TUNER_SIMPLE_DIR))

from tuner_simple.config.cli_arguments import parse_pipeline_config  # noqa: E402
from tuner_simple.logging_utils.timestamped_logging import build_timestamped_logger  # noqa: E402
from tuner_simple.runtime.runtime_paths import ensure_runtime_paths  # noqa: E402
from tuner_simple.serial_runner.pipeline_runner import run_simple_tuner  # noqa: E402


# Define the main function; its body below performs one named step of the pipeline.
def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline, and return a process exit code."""
    # Execute this statement as the next small step in the surrounding pipeline logic.
    ensure_runtime_paths()
    # Compute or store config so later code can reuse this named value clearly.
    config = parse_pipeline_config(argv)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    run_simple_tuner(config, log=build_timestamped_logger())
    # Return this computed value to the caller so the next pipeline stage can use it.
    return 0


# Run the command-line entry point only when this file is executed directly.
if __name__ == "__main__":
    # Stop execution for this invalid state by raising an explicit exception.
    raise SystemExit(main())
