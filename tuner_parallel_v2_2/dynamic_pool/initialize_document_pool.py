#!/usr/bin/env python3
from __future__ import annotations

"""Create the shared scheduling-only document pool for dynamic Slurm workers."""

import argparse
import json
from pathlib import Path
import sys

try:
    from ..matrices.runfile_selection import select_run_items_for_tuning
    from .document_pool import initialize_document_pool
except ImportError:
    # Direct execution via ``python dynamic_pool/initialize_document_pool.py``
    # does not provide a package context, so add the tuner directory explicitly.
    script_dir = Path(__file__).resolve().parents[1]
    project_root = script_dir.parent
    for candidate in (script_dir, project_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    from matrices.runfile_selection import select_run_items_for_tuning  # type: ignore
    from dynamic_pool.document_pool import initialize_document_pool  # type: ignore


def parse_args() -> argparse.Namespace:
    """Parse the lightweight pool-initialisation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a file-backed dynamic document pool. The pool stores only "
            "document ids/names for scheduling; metrics stay in tuner outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    parser.add_argument("--pool-dir", type=Path, required=True, help="Directory where scheduling files are created")
    parser.add_argument("--max-items", type=int, default=None, help="Same selected-document cap used by the tuner")
    parser.add_argument("--target-fname", action="append", default=[], help="Optional target file filter; repeatable")
    return parser.parse_args()


def main() -> int:
    """Select runfile items exactly once and write the initial available queue."""
    args = parse_args()
    selected_run_items = select_run_items_for_tuning(
        runfile_json=Path(args.runfile_json),
        target_fnames=[str(value) for value in args.target_fname if str(value).strip()],
        max_items=args.max_items,
        selection_index_range=None,
    )
    summary = initialize_document_pool(
        pool_dir=Path(args.pool_dir),
        selected_run_items=selected_run_items,
        runfile_json=Path(args.runfile_json),
        max_items=args.max_items,
        log_fn=print,
    )
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
