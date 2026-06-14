from __future__ import annotations

"""Configuration dataclasses shared by the simple tuner runners."""

import math
from dataclasses import dataclass, field
from pathlib import Path

from .hough_parameters import ProbabilisticHoughParameters


VALID_PLOT_MODES = {
    "none",
    "stitched-language",
    "stitched-language-and-document-grids",
}

# Each mode selects a different harmonic-mean formula used to score and compare alpha candidates.
#
#   balanced                      Equal weight on NLS, coverage, and non-hallucination.
#                                 Formula: 3 / (1/NLS + 1/coverage + 1/(1-hallucination))
#
#   coverage-hallucination-priority  Double weight on coverage and non-hallucination vs. NLS.
#                                 Formula: 5 / (1/NLS + 2/coverage + 2/(1-hallucination))
#
#   coverage-hallucination-only   NLS excluded entirely; selection driven by coverage and
#                                 non-hallucination only.
#                                 Formula: 2 / (1/coverage + 1/(1-hallucination))
VALID_HARMONIC_MODES = {
    "balanced",
    "coverage-hallucination-priority",
    "coverage-hallucination-only",
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
    # Alpha sweep mode evaluates several score-floor alphas and keeps only the best result in final CSVs/plots.
    alpha_sweep_enabled: bool = True
    # The inclusive lower bound for alpha sweep candidates.
    alpha_sweep_min: float = 1.0
    # The inclusive upper bound for alpha sweep candidates.
    alpha_sweep_max: float = 4.0
    # The alpha increment between neighboring sweep candidates.
    alpha_sweep_step: float = 0.2
    # When set, this fixed Levenshtein cutoff builds one pre-Hough mask and skips alpha sweep.
    minimum_pre_hough_levenshtein: float | None = None
    # The Hough parameters control the exact probabilistic Hough call used for every document.
    hough_parameters: ProbabilisticHoughParameters = field(default_factory=lambda: ProbabilisticHoughParameters(25, 35, 15, 1))
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
    # The harmonic mode controls which formula is used to score each alpha candidate during selection.
    # See VALID_HARMONIC_MODES for the full list of accepted values and their formulas.
    alpha_selection_harmonic_mode: str = "balanced"

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
        if self.minimum_pre_hough_levenshtein is not None:
            minimum_levenshtein = float(self.minimum_pre_hough_levenshtein)
            if not math.isfinite(minimum_levenshtein) or minimum_levenshtein < 0.0:
                raise ValueError("--minimum-pre-hough-levenshtein must be finite and zero or positive")
        if bool(self.alpha_sweep_enabled):
            if float(self.alpha_sweep_min) < 0.0:
                raise ValueError("--alpha-sweep-min must be zero or positive")
            if float(self.alpha_sweep_max) < float(self.alpha_sweep_min):
                raise ValueError("--alpha-sweep-max must be greater than or equal to --alpha-sweep-min")
            if float(self.alpha_sweep_step) <= 0.0:
                raise ValueError("--alpha-sweep-step must be positive")
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
        if str(self.alpha_selection_harmonic_mode) not in VALID_HARMONIC_MODES:
            raise ValueError(
                f"--harmonic-mode must be one of {sorted(VALID_HARMONIC_MODES)!r}; "
                f"got {self.alpha_selection_harmonic_mode!r}"
            )
        self.hough_parameters.validate()
        return self


__all__ = ["PipelineConfig", "VALID_HARMONIC_MODES", "VALID_PLOT_MODES"]
