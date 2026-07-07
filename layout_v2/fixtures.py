#!/usr/bin/env python3
"""Frozen fixtures + label-free scorer (IMPLEMENTATION_PLAN.md §11; expectations from CCM §12).

The fixtures are the canonical dev pages whose ground truth was established during the method
work. ``EXPECTED`` freezes those values; the scorer compares a layout_v2 run's records against
them and reports exact / bin / structure correctness plus flag behaviour. No human labeling is
involved anywhere (project decision #8/#9): these expectations are already settled.

Usage:
    # score a run that was produced with --only-basenames of the fixture pages (or a full run)
    python3 fixtures.py --run results/layout_v2_run1/dev
    # print the basename list for the runner's --only-basenames
    python3 fixtures.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# --------------------------------------------------------------------------------------
# The frozen expectations (CCM §12). Semantics:
#   category         expected document_category
#   parts            expected number of independent parts
#   dominant_cols    expected column count of the dominant part (None = no single right answer)
#   dominant_bin     the acceptable analysis bin (checked when exact count is None or as backup)
#   may_review       True  = the review flag is ACCEPTABLE on this page
#   must_review      True  = the review flag MUST fire (ill-defined page; a confident wrong
#                            integer is the failure mode being tested)
#   stream_dominant  expected stream.exists for the dominant part (None = don't check)
# --------------------------------------------------------------------------------------
EXPECTED: dict[str, dict] = {
    "europeana_00675495": {   # Suomalainen Kansa 1908 — the clean-hard 9-column page
        "category": "newspaper", "parts": 1, "dominant_cols": 9, "dominant_bin": "7+",
        "may_review": False, "must_review": False, "stream_dominant": True,
    },
    "newseye-fin_576474_0003_23676390": {  # Uusi Suometar 1889 — 7 cols + book-format insert
        "category": "newspaper", "parts": 2, "dominant_cols": 7, "dominant_bin": "7+",
        "may_review": True, "must_review": False, "stream_dominant": True,
    },
    "europeana_00675329": {   # Turun Sanomat 1906 — the 8-column ad mosaic (base grid)
        "category": "newspaper", "parts": 1, "dominant_cols": 8, "dominant_bin": "7+",
        "may_review": True, "must_review": False, "stream_dominant": False,
    },
    "europeana_00674544": {   # Fraktur paper with feuilleton — 3 cols, 2 parts
        "category": "newspaper", "parts": 2, "dominant_cols": 3, "dominant_bin": "2-3",
        "may_review": False, "must_review": False, "stream_dominant": None,
    },
    "europeana_00674591": {   # Wiipuri 1904 — cropped scan; truth itself 7-8
        "category": "newspaper", "parts": 1, "dominant_cols": None, "dominant_bin": "7+",
        "may_review": True, "must_review": False, "stream_dominant": None,
    },
    "newseye-fin_576485_0001_23676428": {  # Uusi Suometar 1894 front — ill-defined mosaic
        "category": "newspaper", "parts": 1, "dominant_cols": None, "dominant_bin": "",
        "may_review": True, "must_review": True, "stream_dominant": None,
    },
}


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def load_records(run_split_dir: Path) -> dict[str, dict]:
    """Index a run's records by extensionless basename (from layout_outputs.json)."""
    outputs = run_split_dir / "layout_outputs.json"
    records = json.loads(outputs.read_text(encoding="utf-8"))
    return {Path(str(r.get("file_name") or "")).stem: r for r in records}


def score_one(stem: str, expected: dict, record: dict | None) -> dict:
    """Compare one fixture page against its frozen expectations; return per-check verdicts."""
    if record is None:
        return {"stem": stem, "found": False, "checks": {}, "passed": False}
    derived = record.get("derived") or {}
    checks: dict[str, bool] = {}

    checks["category"] = derived.get("document_category") == expected["category"]
    checks["parts"] = derived.get("independent_parts") == expected["parts"]

    dominant = derived.get("column_count_dominant")
    if expected["dominant_cols"] is not None:
        checks["columns_exact"] = dominant == expected["dominant_cols"]
        checks["columns_tol1"] = isinstance(dominant, int) and abs(dominant - expected["dominant_cols"]) <= 1
    if expected["dominant_bin"]:
        checks["columns_bin"] = config.column_bin(dominant) == expected["dominant_bin"]

    if expected["stream_dominant"] is not None:
        # Find the dominant part's stream verdict among the derived per-part summaries.
        streams = {p.get("part_index"): p.get("stream_exists") for p in derived.get("parts", [])}
        dominant_stream = next(iter(streams.values()), None) if len(streams) == 1 else None
        if dominant_stream is None and streams:
            dominant_stream = streams.get(1)  # part 1 is the top/dominant part in our fixtures
        checks["stream"] = dominant_stream == expected["stream_dominant"]

    review = bool(derived.get("needs_review"))
    if expected["must_review"]:
        checks["review_fires"] = review          # ill-defined page MUST be flagged
    elif not expected["may_review"]:
        checks["review_silent"] = not review     # clean page must NOT be flagged

    pass_checks = {k: v for k, v in checks.items() if k != "columns_exact"}
    return {"stem": stem, "found": True, "checks": checks, "passed": all(pass_checks.values()),
            "observed": {"category": derived.get("document_category"),
                         "parts": derived.get("independent_parts"),
                         "dominant_cols": dominant,
                         "needs_review": review,
                         "review_reasons": derived.get("needs_review_reasons", [])[:4]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Run split dir (…/dev) containing layout_outputs.json")
    parser.add_argument("--list", action="store_true", help="Print --only-basenames for the runner.")
    args = parser.parse_args()

    if args.list:
        print(",".join(EXPECTED.keys()))
        return 0
    if not args.run:
        parser.error("provide --run or --list")

    records = load_records(args.run)
    results = [score_one(stem, expected, records.get(stem)) for stem, expected in EXPECTED.items()]
    passed = sum(1 for r in results if r["passed"])

    # Human-readable report to stdout + machine-readable JSON next to the run.
    for result in results:
        status = "PASS" if result["passed"] else ("MISSING" if not result["found"] else "FAIL")
        print(f"[{status}] {result['stem']}")
        for check, ok in result.get("checks", {}).items():
            print(f"    {'ok ' if ok else 'BAD'} {check}")
        if result["found"] and not result["passed"]:
            print(f"    observed: {json.dumps(result['observed'], ensure_ascii=False)}")
    print(f"\n[fixtures] {passed}/{len(results)} pages fully passed")
    report_path = args.run / "fixture_scores.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fixtures] wrote {report_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
