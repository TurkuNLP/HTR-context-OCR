# Local v2.12 Metric Compatibility Code

This package contains the small subset of `text_metrics_v2_12_parallel` metric
code that the Hough tuner needs at runtime.

The goal is maintainability: a developer should be able to understand and run
the tuner from inside `tuner_parallel_v2_2/` without discovering a hidden metric
implementation in a sibling directory.

The files here preserve v2.12 semantics.  Do not "clean up" formulas, interval
rules, or category definitions unless the change is treated as a scientific
metric change and compared against the original v2.12 implementation.

Vendored source files:

- `line_metric_bundle.py` came from
  `text_metrics_v2_12_parallel/line_metric_bundle.py`.
- `line_coverage_arrays.py` came from
  `text_metrics_v2_12_parallel/line_coverage_subtract.py`.
- `ordered_sequence_helpers.py` came from
  `text_metrics_v2_12_parallel/shared/ordered_sequence_helpers.py`.
- `text_window_projection.py` came from
  `text_metrics_v2_12_parallel/shared/project_line_to_text_windows.py`.

Copied on: 2026-05-25.

Runtime boundary:

`metrics/v2_12_metric_adapter.py` is still the public adapter used by the
tuner hot loop.  The adapter now uses this local compatibility package by
default and can still load the external v2.12 tree for equivalence tests.

Current hot-loop rule:

- normal tuner scoring calls the local compatibility code;
- `build_compact_line_scoring_payload()` builds only the fields needed by
  coverage and along-lines scoring;
- the full bundle builder remains available for audits and equivalence tests;
- the external `text_metrics_v2_12_parallel` tree is not required for normal
  runtime scoring.

Maintenance rule:

Do not edit formulas, interval boundaries, y-difference categories, or text
window projection behavior as a cleanup task.  These files define metric
semantics.  Any semantic edit must be treated as a scientific metric change and
tested against the historical v2.12 implementation.
