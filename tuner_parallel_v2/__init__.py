"""Public API for the standalone Hough parameter tuner package.

This module deliberately re-exports only stable entry points so runner scripts,
notebooks, and future integrations can import the tuner without depending on the
internal folder layout.  Configuration constants live in ``tuner_config``;
execution functions live in ``tuner_core``.
"""

from .tuner.tuner_config import (
    DEFAULT_SCORE_INDEX_CACHE_DIR,
    DEFAULT_TEXT_METRICS_V212_DIR,
    HOUGH_LINE_GAP_MAX,
    HOUGH_LINE_GAP_MIN,
    HOUGH_LINE_LENGTH_MAX,
    HOUGH_LINE_LENGTH_MIN,
    HOUGH_SEED_MAX,
    HOUGH_SEED_MIN,
    HOUGH_THRESHOLD_MAX,
    HOUGH_THRESHOLD_MIN,
    HoughBaselineConfig,
    HoughSweepRanges,
    InclusiveIntegerRange,
    PARAM_HOUGH_LINE_GAP,
    PARAM_HOUGH_LINE_LENGTH,
    PARAM_HOUGH_SEED,
    PARAM_HOUGH_THRESHOLD,
    SUPPORTED_SWEEP_PARAMETERS,
    build_hough_sweep_ranges,
    default_hough_sweep_ranges,
    fixed_parameter_ranges,
)
from .tuner.tuner_core import (
    BACKEND_C,
    BACKEND_PYTHON,
    SUPPORTED_BACKENDS,
    load_documents,
    run_hough_parameter_sweeps,
)
from .matrices.document_prep import (
    iter_prepared_documents_from_items,
    select_run_items_for_tuning,
)
from .outputs.tuner_outputs import (
    write_best_configs_csv,
    write_parameter_curve_csv,
)
from .outputs.tuner_result_exports import (
    build_best_params_records,
    build_parameter_influence_rows,
    write_best_params_json,
    write_parameter_influence_csv,
)
from .outputs.plot_hough_parameter_sweep import (
    generate_plots_for_summary_json,
    render_plots_from_summary_dict,
)

__all__ = [
    "BACKEND_C",
    "BACKEND_PYTHON",
    "DEFAULT_SCORE_INDEX_CACHE_DIR",
    "DEFAULT_TEXT_METRICS_V212_DIR",
    "HOUGH_LINE_GAP_MAX",
    "HOUGH_LINE_GAP_MIN",
    "HOUGH_LINE_LENGTH_MAX",
    "HOUGH_LINE_LENGTH_MIN",
    "HOUGH_SEED_MAX",
    "HOUGH_SEED_MIN",
    "HOUGH_THRESHOLD_MAX",
    "HOUGH_THRESHOLD_MIN",
    "HoughBaselineConfig",
    "HoughSweepRanges",
    "InclusiveIntegerRange",
    "PARAM_HOUGH_LINE_GAP",
    "PARAM_HOUGH_LINE_LENGTH",
    "PARAM_HOUGH_SEED",
    "PARAM_HOUGH_THRESHOLD",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_SWEEP_PARAMETERS",
    "build_best_params_records",
    "build_hough_sweep_ranges",
    "build_parameter_influence_rows",
    "default_hough_sweep_ranges",
    "fixed_parameter_ranges",
    "generate_plots_for_summary_json",
    "iter_prepared_documents_from_items",
    "load_documents",
    "render_plots_from_summary_dict",
    "run_hough_parameter_sweeps",
    "select_run_items_for_tuning",
    "write_best_configs_csv",
    "write_best_params_json",
    "write_parameter_curve_csv",
    "write_parameter_influence_csv",
]
