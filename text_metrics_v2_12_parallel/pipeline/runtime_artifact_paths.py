"""Central runtime-artifact paths for the v2.12 text-metrics pipeline.

Runtime-generated artifacts such as caches, regenerated diagrams, and optional
compiled accelerators must live outside the source tree. Keeping those paths
centralized avoids hard-coded scatter across the pipeline and makes the
runtime/storage behavior much easier for new contributors to understand.
"""

from __future__ import annotations

from pathlib import Path


# Resolve the project root that contains ``text_metrics_v2_12_parallel`` and ``results``.
def project_workspace_root() -> Path:
    """Return the Churro_copy workspace root that owns this pipeline."""
    return Path(__file__).resolve().parents[2]


# Return the shared runtime-artifacts root used by this pipeline version.
def runtime_artifacts_root() -> Path:
    """Return the shared runtime-artifact root outside the source tree."""
    return project_workspace_root() / "results" / "text_metrics_v2_12_parallel_runtime_artifacts"


# Return the cache directory for score-stream byte-offset indexes.
def score_index_cache_root() -> Path:
    """Return the external cache directory for score-stream indexes."""
    return runtime_artifacts_root() / "score_index_cache"


# Return the external build/cache root for optional exact-result Cython backends.
def exact_result_cython_backend_root() -> Path:
    """Return the external directory for compiled exact-result backend modules."""
    return runtime_artifacts_root() / "exact_result_cython_backends"


# Return the future runtime diagram output root.
def regenerated_diagrams_root() -> Path:
    """Return the external directory where regenerated diagrams should live."""
    return runtime_artifacts_root() / "diagrams"
