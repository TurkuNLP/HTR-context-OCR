# Default Cython Acceleration

This directory contains behavior-preserving Cython helpers for
`tuner_parallel_v2_1`.  Normal Slurm runs through
`run_hough_parameter_sweep_20nodes_10docs_each.sh` build and require these
compiled helpers before the sweep starts.

The pure-Python implementation remains the debugging fallback, but production
runs should keep:

```text
BUILD_CYTHON_EXTENSIONS=1
REQUIRE_CYTHON_EXTENSIONS=1
```

## Current Extensions

```text
along_lines_core.pyx
filter_core.pyx
```

`along_lines_core.pyx` accelerates:

```text
mapped_line_id -> prediction-column grouping by final line id
weighted mean from line scores and Euclidean line lengths
```

`filter_core.pyx` accelerates filtering support helpers:

```text
set IoU for filtering conflict checks
coverage-index construction by prediction column
mean support sampling for one line segment
line path sampling into the Python coverage objects used by the filter
final prediction-column ownership assignment after coverage merging
```

The line path sampler still returns ordinary Python dictionaries, sets, and
lists.  That is intentional: the expensive repeated interpolation and matrix
lookup loop is compiled, while the rest of the filter keeps the same readable
data shape that the Python implementation already used.

The extensions must not change:

```text
Hough detection
line filtering semantics
ownership assignment semantics
RapidFuzz usage
Levenshtein scoring semantics
v2.12 coverage/hallucination semantics
best-parameter ranking semantics
output JSON/CSV schemas
```

## Build

From the tuner directory:

```bash
cd /scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_1
source /usr/share/lmod/8.6.17/init/bash
module use /appl/modulefiles
module load pytorch
python3 cython_accel/build.py build_ext --inplace
```

The shell entry point runs this build automatically before a normal sweep.

## Runtime Import Path

Along-lines helpers:

```text
alignment/along_lines_fast.py
metrics/alignment_quality_score.py
  -> cython_accel.optional_line_grouping
      -> cython_accel.along_lines_core if compiled
      -> pure Python fallback otherwise
```

Filtering helpers:

```text
filtering/line_filtering_v2_1_IoU_fast.py
  -> filtering/filter_cython_accelerators.py
  -> cython_accel.optional_filtering
      -> cython_accel.filter_core if compiled
      -> pure Python fallback otherwise
```

Both paths must preserve the same Python data shapes and tie-break semantics as
the pure-Python reference path.
