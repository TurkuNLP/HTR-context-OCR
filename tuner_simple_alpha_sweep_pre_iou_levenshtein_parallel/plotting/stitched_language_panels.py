from __future__ import annotations

"""Create stitched language-level PNGs from per-document diagnostic panels."""

from collections import defaultdict
from pathlib import Path
import math
import shutil
from typing import Any

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.config.pipeline_config import PipelineConfig
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.plotting.document_panel_renderer import render_document_panel, safe_path_component


STITCHED_PANEL_BORDER_PIXELS = 8
STITCHED_PANEL_GAP_PIXELS = 10
STITCHED_PANEL_BORDER_COLOR = (30, 30, 30, 255)
STITCHED_PANEL_BACKGROUND_COLOR = (255, 255, 255, 255)


class SimplePlotManager:
    """Render document panels immediately and stitch them after the serial run."""

    def __init__(self, *, config: PipelineConfig, log) -> None:
        """Prepare output directories and remember how plots should be drawn."""

        self.config = config
        self.log = log
        self.plots_dir = Path(config.output_dir) / "plots"
        self.keep_document_panels = str(config.plot_mode) == "stitched-language-and-document-grids"
        self.panel_dir = self.plots_dir / "document_panels" if self.keep_document_panels else self.plots_dir / ".temporary_document_panels"
        self.panel_paths_by_language: dict[str, list[Path]] = defaultdict(list)
        self.stitched_paths: list[Path] = []
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.panel_dir.mkdir(parents=True, exist_ok=True)

    def render_document_payload(self, plot_payload: dict[str, Any]) -> None:
        """Render one document tile and immediately drop the large in-memory payload."""

        document = plot_payload["document"]
        language_name = str(document.main_language or "UNKNOWN")
        document_stem = safe_path_component(str(document.fname))
        temporary_path = self.panel_dir / safe_path_component(language_name) / f"{document_stem}.png"
        render_document_panel(
            plot_payload=plot_payload,
            output_path=temporary_path,
            saved_figure_dpi=int(self.config.saved_figure_dpi),
            show_line_ids=bool(self.config.show_line_ids),
        )
        self.panel_paths_by_language[language_name].append(temporary_path)

    def finish(self) -> list[Path]:
        """Create one stitched PNG per language and remove temporary document tiles."""

        if not self.panel_paths_by_language:
            self._remove_temporary_panel_dir()
            return []
        from PIL import Image

        for language_name in sorted(self.panel_paths_by_language):
            panel_paths = self.panel_paths_by_language[language_name]
            stitched_path = self.plots_dir / f"stitched_best_combination_{safe_path_component(language_name)}_documents.png"
            saved_path = save_stitched_language_image(
                panel_paths=panel_paths,
                stitched_output_path=stitched_path,
                panel_columns=int(self.config.stitched_panel_columns),
                image_module=Image,
            )
            if saved_path is not None:
                self.stitched_paths.append(saved_path)
                self.log(f"wrote stitched language plot: {saved_path}")
        self._remove_temporary_panel_dir()
        return list(self.stitched_paths)

    def _remove_temporary_panel_dir(self) -> None:
        """Delete hidden temporary panel files after stitching has completed."""

        if self.keep_document_panels:
            return
        if self.panel_dir.exists():
            shutil.rmtree(self.panel_dir)


def build_bordered_document_tile(
    *,
    panel_image: Any,
    image_module: Any,
    max_panel_width: int,
    max_panel_height: int,
) -> Any:
    """Return one document panel centered inside a visible border tile."""

    border_pixels = int(STITCHED_PANEL_BORDER_PIXELS)
    tile_width = int(max_panel_width) + 2 * border_pixels
    tile_height = int(max_panel_height) + 2 * border_pixels
    bordered_tile = image_module.new("RGBA", (tile_width, tile_height), STITCHED_PANEL_BORDER_COLOR)
    inner_background = image_module.new("RGBA", (int(max_panel_width), int(max_panel_height)), STITCHED_PANEL_BACKGROUND_COLOR)
    bordered_tile.alpha_composite(inner_background, (border_pixels, border_pixels))
    panel_x = border_pixels + max(0, (int(max_panel_width) - int(panel_image.width)) // 2)
    panel_y = border_pixels + max(0, (int(max_panel_height) - int(panel_image.height)) // 2)
    bordered_tile.alpha_composite(panel_image, (panel_x, panel_y))
    inner_background.close()
    return bordered_tile


def save_stitched_language_image(
    *,
    panel_paths: list[Path],
    stitched_output_path: Path,
    panel_columns: int,
    image_module: Any,
) -> Path | None:
    """Paste document panel PNGs into one language-level contact sheet with clear document borders."""

    if not panel_paths:
        return None
    opened_images = []
    try:
        for panel_path in panel_paths:
            opened_images.append(image_module.open(panel_path).convert("RGBA"))
        max_panel_width = max(image.width for image in opened_images)
        max_panel_height = max(image.height for image in opened_images)
        safe_panel_columns = max(1, int(panel_columns))
        row_count = int(math.ceil(len(opened_images) / safe_panel_columns))
        bordered_tile_width = int(max_panel_width) + 2 * int(STITCHED_PANEL_BORDER_PIXELS)
        bordered_tile_height = int(max_panel_height) + 2 * int(STITCHED_PANEL_BORDER_PIXELS)
        gap_pixels = int(STITCHED_PANEL_GAP_PIXELS)
        stitched_width = safe_panel_columns * bordered_tile_width + max(0, safe_panel_columns - 1) * gap_pixels
        stitched_height = row_count * bordered_tile_height + max(0, row_count - 1) * gap_pixels
        stitched_image = image_module.new("RGBA", (stitched_width, stitched_height), STITCHED_PANEL_BACKGROUND_COLOR)
        try:
            for panel_index, panel_image in enumerate(opened_images):
                column_index = panel_index % safe_panel_columns
                row_index = panel_index // safe_panel_columns
                upper_left_corner = (
                    column_index * (bordered_tile_width + gap_pixels),
                    row_index * (bordered_tile_height + gap_pixels),
                )
                bordered_tile = build_bordered_document_tile(
                    panel_image=panel_image,
                    image_module=image_module,
                    max_panel_width=int(max_panel_width),
                    max_panel_height=int(max_panel_height),
                )
                try:
                    stitched_image.alpha_composite(bordered_tile, upper_left_corner)
                finally:
                    bordered_tile.close()
            stitched_output_path.parent.mkdir(parents=True, exist_ok=True)
            stitched_image.save(stitched_output_path, optimize=True)
        finally:
            stitched_image.close()
    finally:
        for image in opened_images:
            image.close()
    return stitched_output_path


__all__ = ["SimplePlotManager", "save_stitched_language_image"]
