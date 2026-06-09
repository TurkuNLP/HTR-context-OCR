# Optional exact-result Cython backend sources

This directory contains the `.pyx` source files for the optional compiled
backends used by `text_metrics_v2_12_parallel`.

## Important design rule

These sources are **not** allowed to change the pipeline algorithm.

They may only accelerate helper boundaries that already exist in the Python
reference implementation.

That means a compiled backend in this directory must preserve:

- the same ordering,
- the same tie-breaking,
- the same rounding behavior,
- the same thresholds,
- and the same returned report values.

## Why the `.pyx` files live in the source tree

The `.pyx` files are part of the maintained source code, so they belong here.
They are versioned, reviewed, and documented like the Python reference code.

## Why the compiled artifacts do **not** live here

The generated C files, build temp files, and compiled extension modules are all
runtime artifacts. They are built into the external runtime-artifact root under:

- `results/text_metrics_v2_12_parallel_runtime_artifacts/exact_result_cython_backends/`

This keeps the source tree clean and makes it safe to rerun the same pipeline
many times without polluting the repository with machine-specific build output.

## Current backends

At the moment there are two optional compiled backends:

1. `greedy_diagonal_segment_merging_backend.pyx`
   - accelerates the default greedy post-Hough merge stage.
2. `true_iou_filter_backend.pyx`
   - accelerates the narrow helper boundaries inside the true-IoU filter.

## Build flow

Users do not build these modules manually.

The supported public path is still:

- `run_text_metrics_report.sh`

That entrypoint initializes the Python environment, then the Python pipeline
performs a best-effort build-or-reuse step before document processing starts.
If the toolchain is unavailable or the build fails, the pure-Python reference
implementation remains active automatically.
