from __future__ import annotations

"""Dynamic document-pool scheduling helpers for Slurm tuner workers.

The package is intentionally small and scheduling-only.  It decides which
document name/index is free for a worker to process, while all metrics and
combination bundles continue to be written by the existing tuner pipeline.
"""

from .document_pool import (
    DocumentLease,
    DocumentLeasePool,
    DocumentPoolInitializationSummary,
    initialize_document_pool,
    iter_claimed_selected_run_items_from_pool,
)

__all__ = [
    "DocumentLease",
    "DocumentLeasePool",
    "DocumentPoolInitializationSummary",
    "initialize_document_pool",
    "iter_claimed_selected_run_items_from_pool",
]
