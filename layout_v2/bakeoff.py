#!/usr/bin/env python3
"""Bake-off comparison: score multiple layout_v2 runs (arms) against each other, label-free.

Implements plan §11.2 (no hand verification anywhere). Arms are ordinary runs produced with one
flag flipped (anchor mode, input strategy, model); this tool compares their artifacts on four
measurable axes:

1. FIXTURE ACCURACY   — the frozen canonical pages (fixtures.EXPECTED; truth already settled).
2. INTERNAL CONSISTENCY — route-agreement rates (enumeration vs width vs returns), validator-issue
   rate, needs_review rate, undercount signature (enumeration < width-implied on 7+ pages).
3. CROSS-ARM AGREEMENT — pairwise agreement on the dominant column count / bin over the shared
   document set (arms that agree where routes also agree are trusted; divergence localizes the
   weaker arm via the fixtures).
4. TOKEN COST         — mean completion tokens per pass.

Usage:
    python3 bakeoff.py --runs results/armA_run1/dev results/armB_run1/dev --report bakeoff_report.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import fixtures  # noqa: E402


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_run(run_split_dir: Path) -> dict:
    """Load one arm: its records (by stem) and a label from the run's config snapshot."""
    records = json.loads((run_split_dir / "layout_outputs.json").read_text(encoding="utf-8"))
    by_stem = {Path(str(r.get("file_name") or "")).stem: r for r in records}
    # The run label lives in run_config.json one level up (falls back to the directory name).
    label = run_split_dir.parent.name
    config_path = run_split_dir.parent / "run_config.json"
    if config_path.exists():
        snapshot = json.loads(config_path.read_text(encoding="utf-8"))
        pieces = [snapshot.get("run_label") or label]
        # Make the arm identity self-describing in the report.
        for key in ("pass1_anchor_mode", "pass1_input", "model_repo"):
            value = snapshot.get(key)
            if value:
                pieces.append(str(value).split("/")[-1])
        label = " / ".join(pieces)
    return {"dir": run_split_dir, "label": label, "by_stem": by_stem}


# --------------------------------------------------------------------------------------
# Axis 1: fixtures
# --------------------------------------------------------------------------------------
def fixture_score(run: dict) -> dict:
    """Pass rate over the frozen fixture pages present in this run."""
    results = []
    for stem, expected in fixtures.EXPECTED.items():
        record = run["by_stem"].get(stem)
        if record is not None:
            results.append(fixtures.score_one(stem, expected, record))
    passed = sum(1 for r in results if r["passed"])
    return {"present": len(results), "passed": passed}


# --------------------------------------------------------------------------------------
# Axis 2: internal consistency
# --------------------------------------------------------------------------------------
def consistency_stats(run: dict) -> dict:
    """Label-free health measures over all documents of one arm."""
    verdicts: Counter = Counter()
    n_multi = 0            # parts with >= 2 enumerated columns (where agreement is informative)
    n_docs = 0
    n_review = 0
    n_validator_issue = 0
    undercount_hits = 0    # enumeration < width-implied on 7+ width pages: the undercount signature
    undercount_denominator = 0
    completion_tokens = {"pass0": [], "pass1": [], "pass2": []}

    for record in run["by_stem"].values():
        derived = record.get("derived") or {}
        n_docs += 1
        n_review += 1 if derived.get("needs_review") else 0
        n_validator_issue += 1 if record.get("validation_issues") else 0
        for part in derived.get("parts") or []:
            routes = part.get("routes") or {}
            if part.get("column_count", 0) >= 2:
                n_multi += 1
                verdicts[part.get("verdict", "unknown")] += 1
            width = routes.get("width")
            if isinstance(width, int) and width >= 7:  # the label-free 7+ undercount check
                undercount_denominator += 1
                if routes.get("enumeration", 0) < width:
                    undercount_hits += 1
        for pass_name in completion_tokens:
            usage = ((record.get(pass_name) or {}).get("response_metadata") or {}).get("usage") or {}
            tokens = usage.get("completion_tokens")
            if isinstance(tokens, (int, float)):
                completion_tokens[pass_name].append(tokens)

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 3) if denominator else float("nan")

    total_verdicts = sum(verdicts.values())
    return {
        "n_docs": n_docs,
        "route_agree_rate_multicol": rate(verdicts.get("agree", 0), total_verdicts),
        "route_disagree_rate_multicol": rate(verdicts.get("disagree", 0), total_verdicts),
        "needs_review_rate": rate(n_review, n_docs),
        "validator_issue_rate": rate(n_validator_issue, n_docs),
        "undercount_signature_rate_7plus": rate(undercount_hits, undercount_denominator),
        "mean_completion_tokens": {
            name: round(sum(vals) / len(vals), 1) if vals else None
            for name, vals in completion_tokens.items()
        },
    }


# --------------------------------------------------------------------------------------
# Axis 3: cross-arm agreement
# --------------------------------------------------------------------------------------
def cross_agreement(run_a: dict, run_b: dict) -> dict:
    """Agreement between two arms on dominant column count / bin over their shared documents."""
    shared = set(run_a["by_stem"]) & set(run_b["by_stem"])
    exact = 0
    within_bin = 0
    n = 0
    for stem in shared:
        derived_a = (run_a["by_stem"][stem].get("derived") or {})
        derived_b = (run_b["by_stem"][stem].get("derived") or {})
        count_a = derived_a.get("column_count_dominant")
        count_b = derived_b.get("column_count_dominant")
        if not isinstance(count_a, int) or not isinstance(count_b, int):
            continue
        n += 1
        exact += 1 if count_a == count_b else 0
        within_bin += 1 if config.column_bin(count_a) == config.column_bin(count_b) else 0
    return {
        "n_shared_scored": n,
        "exact_agreement": round(exact / n, 3) if n else float("nan"),
        "bin_agreement": round(within_bin / n, 3) if n else float("nan"),
    }


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="Two or more run split dirs (…/dev), one per arm.")
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parent / "bakeoff_report.md")
    args = parser.parse_args()

    runs = [load_run(path) for path in args.runs]
    lines = ["# Bake-off report (label-free, plan §11.2)", ""]

    # Axis 1 + 2 per arm.
    lines += ["## Per-arm health", "",
              "| arm | docs | fixtures passed | route agree (multicol) | route disagree | "
              "review rate | validator rate | undercount@7+ | pass1 mean completion tokens |",
              "|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        fx = fixture_score(run)
        stats = consistency_stats(run)
        lines.append(
            f"| {run['label']} | {stats['n_docs']} | {fx['passed']}/{fx['present']} "
            f"| {stats['route_agree_rate_multicol']} | {stats['route_disagree_rate_multicol']} "
            f"| {stats['needs_review_rate']} | {stats['validator_issue_rate']} "
            f"| {stats['undercount_signature_rate_7plus']} "
            f"| {stats['mean_completion_tokens']['pass1']} |"
        )

    # Axis 3 pairwise.
    lines += ["", "## Cross-arm agreement (dominant column count)", "",
              "| arm A | arm B | shared docs | exact | same bin |", "|---|---|---|---|---|"]
    for run_a, run_b in itertools.combinations(runs, 2):
        agreement = cross_agreement(run_a, run_b)
        lines.append(
            f"| {run_a['label']} | {run_b['label']} | {agreement['n_shared_scored']} "
            f"| {agreement['exact_agreement']} | {agreement['bin_agreement']} |"
        )

    lines += ["", "## Reading guide", "",
              "- Fixtures are the truth anchor; an arm that fails them loses regardless of the rest.",
              "- Prefer high route-agreement and LOW undercount@7+ (enumeration below width-implied on",
              "  wide pages is the historical failure signature).",
              "- needs_review should be nonzero (an arm with zero flags is not looking) but modest.",
              "- Where two arms disagree in bulk, the fixtures + undercount column localize the weaker one."]

    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[bakeoff] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
