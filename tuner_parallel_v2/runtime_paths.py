from __future__ import annotations

"""Runtime import-path bootstrap for tuner_parallel scripts/modules.

This helper keeps path handling in one place so every script can run both as:
1) module import (package context), and
2) direct script execution (`python some_script.py`).
"""

from pathlib import Path
import sys


def ensure_tuner_runtime_paths() -> tuple[Path, Path, Path]:
    """Ensure all required project paths are present in ``sys.path``.

    Returns:
        A tuple ``(script_dir, project_root, shared_metrics_dir)``.

    Added paths (if they exist):
    - tuner_parallel_v2 directory (local modules)
    - project root
    - text_metrics_v2_1_parallel (shared metric helpers)
    - legacy helper locations used by HTR-context-OCR (``text_metrics_v2_1`` and ``python_scripts``)
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    shared_metrics_dir = project_root / "text_metrics_v2_1_parallel"
    legacy_metrics_dir = project_root / "text_metrics_v2_1"
    legacy_helpers_dir = project_root / "python_scripts"

    for candidate in (
        shared_metrics_dir,
        legacy_metrics_dir,
        legacy_helpers_dir,
        project_root,
        script_dir,
    ):
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    return script_dir, project_root, shared_metrics_dir


__all__ = ["ensure_tuner_runtime_paths"]
