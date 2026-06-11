from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dynamic_pool.document_pool import DocumentPool
else:
    from .document_pool import DocumentPool


def main() -> int:
    parser = argparse.ArgumentParser(description="Print tuner_simple dynamic document pool counts.")
    parser.add_argument("--pool-dir", type=Path, required=True)
    arguments = parser.parse_args()
    counts = DocumentPool(arguments.pool_dir).state_counts()
    for state_name in sorted(counts):
        print(f"{state_name}\t{counts[state_name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
