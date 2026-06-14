from __future__ import annotations

"""Import-path bootstrap for direct tuner_simple script execution."""

from pathlib import Path
import sys


def ensure_runtime_paths() -> tuple[Path, Path]:
    """Make tuner_simple importable when run_tuner_simple_alpha_sweep_pre_iou_levenshtein.py is executed directly."""
    # The runtime package lives inside tuner_simple/runtime, so this file gives us the runtime directory.
    runtime_dir = Path(__file__).resolve().parent
    # The tuner_simple package is the parent directory of runtime/.
    tuner_simple_dir = runtime_dir.parent
    # Churro_copy is the project directory that contains the tuner_simple package.
    project_dir = tuner_simple_dir.parent
    for candidate_path in (tuner_simple_dir, project_dir):
        # Add the path only when it exists and is not already present, preserving import order stability.
        if candidate_path.exists() and str(candidate_path) not in sys.path:
            # Insert at the front so this working tree wins over any installed package with the same name.
            sys.path.insert(0, str(candidate_path))
    # Return the two useful paths for callers that want to log or inspect them.
    return tuner_simple_dir, project_dir


# Declare the public helper that direct script entry points may import.
__all__ = ["ensure_runtime_paths"]
