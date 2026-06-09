from __future__ import annotations

from pathlib import Path
from typing import Any

from tuner_simple.plotting.document_panel_renderer import render_document_panel, safe_path_component


def render_atomic_document_panel(
    *,
    plot_payload: dict[str, Any] | None,
    panel_root_dir: Path,
    saved_figure_dpi: int,
    show_line_ids: bool,
) -> Path | None:
    """Render one worker-owned document panel and return the PNG path."""

    if plot_payload is None:
        return None
    document = plot_payload["document"]
    language_name = str(document.main_language or "UNKNOWN")
    document_filename = str(document.fname)
    language_directory = Path(panel_root_dir) / safe_path_component(language_name)
    output_path = language_directory / f"{safe_path_component(document_filename)}.png"
    render_document_panel(
        plot_payload=plot_payload,
        output_path=output_path,
        saved_figure_dpi=int(saved_figure_dpi),
        show_line_ids=bool(show_line_ids),
    )
    return output_path
