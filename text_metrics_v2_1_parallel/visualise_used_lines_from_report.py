"""Compatibility wrapper for visualization helpers and CLI.

This file preserves the legacy import path while delegating implementation to
modules under the visualisation/ package.
"""

from __future__ import annotations

from visualisation.run_visualise_report_cli import build_visualisation, main
from visualisation.render_alignment_matrix_views import raw_segments_to_labeled_lines, save_matrix_visualisation
from visualisation.render_line_coverage_views import save_count_line_coverage_visualisations
from visualisation.render_text_metrics_visualisations import save_text_metrics_visualisations

__all__ = [
    "build_visualisation",
    "raw_segments_to_labeled_lines",
    "save_count_line_coverage_visualisations",
    "save_matrix_visualisation",
    "save_text_metrics_visualisations",
]


if __name__ == "__main__":
    main()
