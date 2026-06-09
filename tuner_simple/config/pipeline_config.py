from __future__ import annotations

"""Configuration dataclasses shared by the simple tuner runners."""

from dataclasses import dataclass, field
from pathlib import Path

from .hough_parameters import ProbabilisticHoughParameters


VALID_PLOT_MODES = {
    "none",
    "stitched-language",
    "stitched-language-and-document-grids",
}


@dataclass(frozen=True)
class PipelineConfig:
    """All user-controlled settings needed by one simple tuner run."""

    # The runfile JSON supplies document filenames, language labels, references, and predictions.
    runfile_json: Path
    # The output directory stores final CSV files, progress CSV files, logs, and optional plots.
    output_dir: Path
    # The reference-to-prediction pickle stores the score matrices used for model-output alignment.
    scores_pkl_ref_to_pred: Path
    # The reference-to-reference pickle stores the self-alignment score matrices used for coverage comparison.
    scores_pkl_ref_to_ref: Path
    # Empty language filters mean every language from the runfile is eligible.
    languages: tuple[str, ...] = field(default_factory=tuple)
    # Empty document-type filters mean every document type from the runfile is eligible.
    document_types: tuple[str, ...] = field(default_factory=tuple)
    # Empty filename filters mean no exact filename restriction is applied.
    target_fnames: tuple[str, ...] = field(default_factory=tuple)
    # A missing max item limit means the runner processes every selected document.
    max_items: int | None = None
    # The window size is the number of text characters represented by one score-matrix cell.
    window_size: int = 50
    # The window stride is how far the text window moves between neighboring score-matrix cells.
    window_stride: int = 35
    # Matrices with fewer rows than this are skipped before Hough detection.
    minimum_matrix_rows: int = 4
    # Matrices with fewer columns than this are skipped before Hough detection.
    minimum_matrix_columns: int = 4
    # The score floor is computed as mean + alpha * standard deviation.
    score_floor_alpha: float = 1.0
    # The Hough parameters control the exact probabilistic Hough call used for every document.
    hough_parameters: ProbabilisticHoughParameters = field(default_factory=lambda: ProbabilisticHoughParameters(25, 35, 15, 1))
    # The absolute line-length filter is kept for compatibility with the existing filtering code.
    align_abs_min_len: float = 0.0
    # The intersection-over-union threshold decides when a detected line covers a text window.
    align_min_iou_threshold: float = 0.035
    # The line-level normalised Levenshtein filter removes detected lines with weak text support.
    min_surviving_line_nls: float | None = 0.5
    # Plot mode controls whether workers keep no plots, stitched language plots, or document grids too.
    plot_mode: str = "stitched-language"
    # Line identifiers are hidden by default because they can clutter stitched language plots.
    show_line_ids: bool = False
    # This controls how many 2x3 document panels appear in each row of a stitched language image.
    stitched_panel_columns: int = 3
    # This controls the saved PNG resolution.
    saved_figure_dpi: int = 140
    # A pool directory activates dynamic worker mode; leaving it unset keeps the original serial runner.
    dynamic_document_pool_dir: Path | None = None
    # The worker identifier is written into progress CSV rows and pool events.
    dynamic_worker_id: str | None = None
    # Atomic mode writes progress files under this directory; when missing, output_dir is used.
    atomic_output_dir: Path | None = None
    # Workers flush this many completed documents as one locked CSV append.
    result_bucket_size: int = 20
    # Workers also flush after this many seconds so progress remains visible during slow documents.
    result_bucket_seconds: float = 60.0

    def validate(self) -> "PipelineConfig":
        """Return this object after validating settings that can be checked locally."""

        if int(self.window_size) <= 0:
            raise ValueError("--window-size must be positive")
        if int(self.window_stride) <= 0:
            raise ValueError("--window-stride must be positive")
        if int(self.minimum_matrix_rows) < 0:
            raise ValueError("--minimum-matrix-rows must be zero or positive")
        if int(self.minimum_matrix_columns) < 0:
            raise ValueError("--minimum-matrix-columns must be zero or positive")
        if float(self.score_floor_alpha) < 0.0:
            raise ValueError("--score-floor-alpha must be zero or positive")
        if str(self.plot_mode) not in VALID_PLOT_MODES:
            raise ValueError(f"--plot-mode must be one of {sorted(VALID_PLOT_MODES)!r}")
        if int(self.stitched_panel_columns) <= 0:
            raise ValueError("--stitched-panel-columns must be positive")
        if int(self.saved_figure_dpi) <= 0:
            raise ValueError("--saved-figure-dpi must be positive")
        if self.max_items is not None and int(self.max_items) <= 0:
            raise ValueError("--max-items must be positive when provided")
        if int(self.result_bucket_size) <= 0:
            raise ValueError("--result-bucket-size must be positive")
        if float(self.result_bucket_seconds) <= 0.0:
            raise ValueError("--result-bucket-seconds must be positive")
        self.hough_parameters.validate()
        return self


__all__ = ["PipelineConfig", "VALID_PLOT_MODES"]
