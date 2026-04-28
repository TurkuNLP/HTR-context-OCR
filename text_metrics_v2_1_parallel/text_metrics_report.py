"""Entrypoint wrapper for the text metrics pipeline.

The implementation lives in ``pipeline/run_text_metrics_pipeline.py``.
This file is intentionally minimal to keep backward-compatible invocation.
"""

from __future__ import annotations

from pipeline.run_text_metrics_pipeline import main


if __name__ == "__main__":
    main()
