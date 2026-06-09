from __future__ import annotations

"""Create stitched language-level PNGs from per-document diagnostic panels."""

from collections import defaultdict
from pathlib import Path
import math
import shutil
from typing import Any

from tuner_simple.config.pipeline_config import PipelineConfig
from tuner_simple.plotting.document_panel_renderer import render_document_panel, safe_path_component


# Define the SimplePlotManager class, which groups related state and behavior for this part of the pipeline.
class SimplePlotManager:
    """Render document panels immediately and stitch them after the serial run."""

    # Define the __init__ function; its body below performs one named step of the pipeline.
    def __init__(self, *, config: PipelineConfig, log) -> None:
        """Prepare output directories and remember how plots should be drawn."""
        # Compute or store self.config so later code can reuse this named value clearly.
        self.config = config
        # Compute or store self.log so later code can reuse this named value clearly.
        self.log = log
        # Compute or store self.plots_dir so later code can reuse this named value clearly.
        self.plots_dir = Path(config.output_dir) / "plots"
        # Compute or store self.keep_document_panels so later code can reuse this named value clearly.
        self.keep_document_panels = str(config.plot_mode) == "stitched-language-and-document-grids"
        # Compute or store self.panel_dir so later code can reuse this named value clearly.
        self.panel_dir = self.plots_dir / "document_panels" if self.keep_document_panels else self.plots_dir / ".temporary_document_panels"
        # Compute or store self.panel_paths_by_language: dict[str, list[Path]] so later code can reuse this named value clearly.
        self.panel_paths_by_language: dict[str, list[Path]] = defaultdict(list)
        # Compute or store self.stitched_paths: list[Path] so later code can reuse this named value clearly.
        self.stitched_paths: list[Path] = []
        # Ensure the target directory exists before later code tries to write files into it.
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        # Ensure the target directory exists before later code tries to write files into it.
        self.panel_dir.mkdir(parents=True, exist_ok=True)

    # Define the render_document_payload function; its body below performs one named step of the pipeline.
    def render_document_payload(self, plot_payload: dict[str, Any]) -> None:
        """Render one document tile and immediately drop the large in-memory payload."""
        # Compute or store document so later code can reuse this named value clearly.
        document = plot_payload["document"]
        # Compute or store language_name so later code can reuse this named value clearly.
        language_name = str(document.main_language or "UNKNOWN")
        # Compute or store document_stem so later code can reuse this named value clearly.
        document_stem = safe_path_component(str(document.fname))
        # Compute or store temporary_path so later code can reuse this named value clearly.
        temporary_path = self.panel_dir / safe_path_component(language_name) / f"{document_stem}.png"
        # Start a multi-line call or data structure so related arguments stay readable.
        render_document_panel(
            # Pass the plot_payload argument into the surrounding call so the callee receives that setting explicitly.
            plot_payload=plot_payload,
            # Pass the output_path argument into the surrounding call so the callee receives that setting explicitly.
            output_path=temporary_path,
            # Pass saved_figure_dpi into the surrounding call; this supplies the resolution used when saving plot images.
            saved_figure_dpi=int(self.config.saved_figure_dpi),
            # Pass show_line_ids into the surrounding call; this supplies whether raw and final line labels are printed on plot overlays.
            show_line_ids=bool(self.config.show_line_ids),
        )
        # Add this item to the list that is accumulating results for later output.
        self.panel_paths_by_language[language_name].append(temporary_path)

    # Define the finish function; its body below performs one named step of the pipeline.
    def finish(self) -> list[Path]:
        """Create one stitched PNG per language and remove temporary document tiles."""
        # Check whether not self.panel_paths_by_language; the indented block handles that specific case.
        if not self.panel_paths_by_language:
            # Execute this statement as the next small step in the surrounding pipeline logic.
            self._remove_temporary_panel_dir()
            # Return this computed value to the caller so the next pipeline stage can use it.
            return []
        from PIL import Image

        # Iterate over language_name in sorted(self.panel_paths_by_language) so each item is processed with the same logic.
        for language_name in sorted(self.panel_paths_by_language):
            # Compute or store panel_paths so later code can reuse this named value clearly.
            panel_paths = self.panel_paths_by_language[language_name]
            # Compute or store stitched_path so later code can reuse this named value clearly.
            stitched_path = self.plots_dir / f"stitched_best_combination_{safe_path_component(language_name)}_documents.png"
            # Compute or store saved_path so later code can reuse this named value clearly.
            saved_path = save_stitched_language_image(
                # Pass the panel_paths argument into the surrounding call so the callee receives that setting explicitly.
                panel_paths=panel_paths,
                # Pass the stitched_output_path argument into the surrounding call so the callee receives that setting explicitly.
                stitched_output_path=stitched_path,
                # Pass the panel_columns argument into the surrounding call so the callee receives that setting explicitly.
                panel_columns=int(self.config.stitched_panel_columns),
                # Pass the image_module argument into the surrounding call so the callee receives that setting explicitly.
                image_module=Image,
            )
            # Check whether saved_path is not None; the indented block handles that specific case.
            if saved_path is not None:
                # Add this item to the list that is accumulating results for later output.
                self.stitched_paths.append(saved_path)
                # Write a progress message so long runs are understandable from terminal or Slurm output.
                self.log(f"wrote stitched language plot: {saved_path}")
        # Execute this statement as the next small step in the surrounding pipeline logic.
        self._remove_temporary_panel_dir()
        # Return this computed value to the caller so the next pipeline stage can use it.
        return list(self.stitched_paths)

    # Define the _remove_temporary_panel_dir function; its body below performs one named step of the pipeline.
    def _remove_temporary_panel_dir(self) -> None:
        """Delete hidden temporary panel files after stitching has completed."""
        # Check whether self.keep_document_panels; the indented block handles that specific case.
        if self.keep_document_panels:
            # Exit the function here without returning a separate data value.
            return
        # Check whether self.panel_dir.exists(); the indented block handles that specific case.
        if self.panel_dir.exists():
            # Execute this statement as the next small step in the surrounding pipeline logic.
            shutil.rmtree(self.panel_dir)


# Define the save_stitched_language_image function; its body below performs one named step of the pipeline.
def save_stitched_language_image(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the panel_paths field so this data object records that value explicitly.
    panel_paths: list[Path],
    # Define the stitched_output_path field so this data object records that value explicitly.
    stitched_output_path: Path,
    # Define the panel_columns field so this data object records that value explicitly.
    panel_columns: int,
    # Define the image_module field so this data object records that value explicitly.
    image_module: Any,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> Path | None:
    """Paste document panel PNGs into one language-level contact sheet."""
    # Check whether not panel_paths; the indented block handles that specific case.
    if not panel_paths:
        # Return this computed value to the caller so the next pipeline stage can use it.
        return None
    # Compute or store opened_images so later code can reuse this named value clearly.
    opened_images = []
    # Define the try field so this data object records that value explicitly.
    try:
        # Iterate over panel_path in panel_paths so each item is processed with the same logic.
        for panel_path in panel_paths:
            # Add this item to the list that is accumulating results for later output.
            opened_images.append(image_module.open(panel_path).convert("RGBA"))
        # Compute or store max_panel_width so later code can reuse this named value clearly.
        max_panel_width = max(image.width for image in opened_images)
        # Compute or store max_panel_height so later code can reuse this named value clearly.
        max_panel_height = max(image.height for image in opened_images)
        # Compute or store safe_panel_columns so later code can reuse this named value clearly.
        safe_panel_columns = max(1, int(panel_columns))
        # Compute or store row_count so later code can reuse this named value clearly.
        row_count = int(math.ceil(len(opened_images) / safe_panel_columns))
        # Compute or store stitched_image so later code can reuse this named value clearly.
        stitched_image = image_module.new(
            # Provide this literal text value to the surrounding path, message, or argument definition.
            "RGBA",
            # Pass this value into the surrounding multi-line call or collection.
            (safe_panel_columns * max_panel_width, row_count * max_panel_height),
            # Pass this value into the surrounding multi-line call or collection.
            (255, 255, 255, 255),
        )
        # Define the try field so this data object records that value explicitly.
        try:
            # Iterate over panel_index, panel_image in enumerate(opened_images) so each item is processed with the same logic.
            for panel_index, panel_image in enumerate(opened_images):
                # Compute or store column_index so later code can reuse this named value clearly.
                column_index = panel_index % safe_panel_columns
                # Compute or store row_index so later code can reuse this named value clearly.
                row_index = panel_index // safe_panel_columns
                # Compute or store upper_left_corner so later code can reuse this named value clearly.
                upper_left_corner = (column_index * max_panel_width, row_index * max_panel_height)
                # Execute this statement as the next small step in the surrounding pipeline logic.
                stitched_image.alpha_composite(panel_image, upper_left_corner)
            # Ensure the target directory exists before later code tries to write files into it.
            stitched_output_path.parent.mkdir(parents=True, exist_ok=True)
            # Execute this statement as the next small step in the surrounding pipeline logic.
            stitched_image.save(stitched_output_path, optimize=True)
        # Define the finally field so this data object records that value explicitly.
        finally:
            # Execute this statement as the next small step in the surrounding pipeline logic.
            stitched_image.close()
    # Define the finally field so this data object records that value explicitly.
    finally:
        # Iterate over image in opened_images so each item is processed with the same logic.
        for image in opened_images:
            # Execute this statement as the next small step in the surrounding pipeline logic.
            image.close()
    # Return this computed value to the caller so the next pipeline stage can use it.
    return stitched_output_path


__all__ = ["SimplePlotManager", "save_stitched_language_image"]
