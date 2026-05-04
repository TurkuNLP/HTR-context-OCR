"""High-level public API for text-alignment fixed-range parameter tuning."""

from .tuner_core import (
    BACKEND_C,
    BACKEND_PYTHON,
    HOUGH_LINE_GAP_MAX,
    HOUGH_LINE_GAP_MIN,
    HOUGH_LINE_LENGTH_MAX,
    HOUGH_LINE_LENGTH_MIN,
    HOUGH_THRESHOLD_MAX,
    HOUGH_THRESHOLD_MIN,
    HoughBaselineConfig,
    PARAM_HOUGH_LINE_GAP,
    PARAM_HOUGH_LINE_LENGTH,
    PARAM_HOUGH_SEED,
    PARAM_HOUGH_THRESHOLD,
    SUPPORTED_BACKENDS,
    SUPPORTED_SWEEP_PARAMETERS,
    fixed_parameter_ranges,
    load_documents,
    run_hough_parameter_sweeps,
)
from .tuner_outputs import (
    write_best_configs_csv,
    write_parameter_curve_csv,
)
from .tuner_result_exports import (
    build_best_params_records,
    build_parameter_influence_rows,
    write_best_params_json,
    write_parameter_influence_csv,
)
from .plot_hough_parameter_sweep import (
    generate_plots_for_summary_json,
    render_plots_from_summary_dict,
)

__all__ = [
    "BACKEND_C",
    "BACKEND_PYTHON",
    "SUPPORTED_BACKENDS",
    "HoughBaselineConfig",
    "PARAM_HOUGH_LINE_GAP",
    "PARAM_HOUGH_LINE_LENGTH",
    "PARAM_HOUGH_SEED",
    "PARAM_HOUGH_THRESHOLD",
    "SUPPORTED_SWEEP_PARAMETERS",
    "HOUGH_THRESHOLD_MIN",
    "HOUGH_THRESHOLD_MAX",
    "HOUGH_LINE_LENGTH_MIN",
    "HOUGH_LINE_LENGTH_MAX",
    "HOUGH_LINE_GAP_MIN",
    "HOUGH_LINE_GAP_MAX",
    "fixed_parameter_ranges",
    "generate_plots_for_summary_json",
    "load_documents",
    "render_plots_from_summary_dict",
    "run_hough_parameter_sweeps",
    "write_best_configs_csv",
    "write_parameter_curve_csv",
    "build_best_params_records",
    "build_parameter_influence_rows",
    "write_best_params_json",
    "write_parameter_influence_csv",
]
