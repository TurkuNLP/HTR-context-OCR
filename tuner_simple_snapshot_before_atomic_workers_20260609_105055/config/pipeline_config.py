from __future__ import annotations

"""Configuration dataclasses shared by the simple serial runner."""

from dataclasses import dataclass, field
from pathlib import Path

from .hough_parameters import ProbabilisticHoughParameters


# Compute or store VALID_PLOT_MODES so later code can reuse this named value clearly.
VALID_PLOT_MODES = {
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "none",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "stitched-language",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "stitched-language-and-document-grids",
}


# Ask Python to generate common data-container methods for the class defined next.
@dataclass(frozen=True)
# Define the PipelineConfig class, which groups related state and behavior for this part of the pipeline.
class PipelineConfig:
    """All user-controlled settings needed by one serial tuner run."""

    # Define the runfile_json field; it stores the JSON file that lists source documents, references, predictions, languages, and document types.
    runfile_json: Path
    # Define the output_dir field; it stores the directory where CSV, JSON, and optional plot files will be written.
    output_dir: Path
    # Define the scores_pkl_ref_to_pred field; it stores the precomputed reference-to-prediction score matrix file.
    scores_pkl_ref_to_pred: Path
    # Define the scores_pkl_ref_to_ref field; it stores the precomputed reference-to-reference score matrix file.
    scores_pkl_ref_to_ref: Path
    # Compute or store languages: tuple[str, ...] so later code can reuse this named value clearly.
    languages: tuple[str, ...] = field(default_factory=tuple)
    # Compute or store document_types: tuple[str, ...] so later code can reuse this named value clearly.
    document_types: tuple[str, ...] = field(default_factory=tuple)
    # Compute or store target_fnames: tuple[str, ...] so later code can reuse this named value clearly.
    target_fnames: tuple[str, ...] = field(default_factory=tuple)
    # Compute or store max_items: int | None so later code can reuse this named value clearly.
    max_items: int | None = None
    # Compute or store window_size: int so later code can reuse this named value clearly.
    window_size: int = 50
    # Compute or store window_stride: int so later code can reuse this named value clearly.
    window_stride: int = 35
    # Compute or store minimum_matrix_rows: int so later code can reuse this named value clearly.
    minimum_matrix_rows: int = 4
    # Compute or store minimum_matrix_columns: int so later code can reuse this named value clearly.
    minimum_matrix_columns: int = 4
    # Compute or store score_floor_alpha: float so later code can reuse this named value clearly.
    score_floor_alpha: float = 1.0
    # Compute or store hough_parameters: ProbabilisticHoughParameters so later code can reuse this named value clearly.
    hough_parameters: ProbabilisticHoughParameters = field(
        # Compute or store default_factory so later code can reuse this named value clearly.
        default_factory=lambda: ProbabilisticHoughParameters(25, 35, 15, 1)
    )
    # Compute or store align_abs_min_len: float so later code can reuse this named value clearly.
    align_abs_min_len: float = 0.0
    # Compute or store align_min_iou_threshold: float so later code can reuse this named value clearly.
    align_min_iou_threshold: float = 0.035
    # Compute or store min_surviving_line_nls: float | None so later code can reuse this named value clearly.
    min_surviving_line_nls: float | None = 0.5
    # Compute or store plot_mode: str so later code can reuse this named value clearly.
    plot_mode: str = "stitched-language"
    # Compute or store show_line_ids: bool so later code can reuse this named value clearly.
    show_line_ids: bool = False
    # Compute or store stitched_panel_columns: int so later code can reuse this named value clearly.
    stitched_panel_columns: int = 3
    # Compute or store saved_figure_dpi: int so later code can reuse this named value clearly.
    saved_figure_dpi: int = 140

    # Define the validate function; its body below performs one named step of the pipeline.
    def validate(self) -> "PipelineConfig":
        """Return this object after validating settings that can be checked locally."""
        # Check whether int(self.window_size) <= 0; the indented block handles that specific case.
        if int(self.window_size) <= 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--window-size must be positive")
        # Check whether int(self.window_stride) <= 0; the indented block handles that specific case.
        if int(self.window_stride) <= 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--window-stride must be positive")
        # Check whether int(self.minimum_matrix_rows) < 0; the indented block handles that specific case.
        if int(self.minimum_matrix_rows) < 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--minimum-matrix-rows must be zero or positive")
        # Check whether int(self.minimum_matrix_columns) < 0; the indented block handles that specific case.
        if int(self.minimum_matrix_columns) < 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--minimum-matrix-columns must be zero or positive")
        # Check whether float(self.score_floor_alpha) < 0.0; the indented block handles that specific case.
        if float(self.score_floor_alpha) < 0.0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--score-floor-alpha must be zero or positive")
        # Check whether str(self.plot_mode) not in VALID_PLOT_MODES; the indented block handles that specific case.
        if str(self.plot_mode) not in VALID_PLOT_MODES:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError(f"--plot-mode must be one of {sorted(VALID_PLOT_MODES)!r}")
        # Check whether int(self.stitched_panel_columns) <= 0; the indented block handles that specific case.
        if int(self.stitched_panel_columns) <= 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--stitched-panel-columns must be positive")
        # Check whether int(self.saved_figure_dpi) <= 0; the indented block handles that specific case.
        if int(self.saved_figure_dpi) <= 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--saved-figure-dpi must be positive")
        # Check whether self.max_items is not None and int(self.max_items) <= 0; the indented block handles that specific case.
        if self.max_items is not None and int(self.max_items) <= 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--max-items must be positive when provided")
        # Execute this statement as the next small step in the surrounding pipeline logic.
        self.hough_parameters.validate()
        # Return this computed value to the caller so the next pipeline stage can use it.
        return self


__all__ = ["PipelineConfig", "VALID_PLOT_MODES"]
