#!/usr/bin/env python3
from __future__ import annotations

"""Per-language result analysis for a balanced tuner run.

Reads best_combination_per_document.csv / runfile_documents.csv /
skipped_documents.csv from a balanced run directory and prints per-language
tables plus writes two PNGs into <results-dir>/analysis_plots:
  - repetition_distribution_per_language.png
  - result_drivers_per_language.png

Answers: what influenced the results, and why are they low for some languages.

Usage:
  /appl/soft/ai/wrap/pytorch-2.9/bin/python3 scratch_tools/language_result_analysis.py \\
    --results-dir results/<run>/balanced
"""

import argparse
import csv
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _raise_csv_field_size_limit() -> None:
    """Allow very large CSV fields (full document text / serialised metrics)."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit // 10)


_raise_csv_field_size_limit()


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_number(value) -> bool:
    return to_number(value) is not None


def column_values(rows: list[dict], name: str) -> list[float]:
    return [v for v in (to_number(r.get(name)) for r in rows) if v is not None]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def print_tables(run, skip, best) -> None:
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in best:
        by_language[row["main_language"]].append(row)

    selected = Counter(r["main_language"] for r in run)
    skipped = Counter(r["main_language"] for r in skip)
    processed = Counter(r["main_language"] for r in best)
    valid = Counter(
        r["main_language"] for r in best
        if is_number(r.get("weighted_along_lines_normalised_levenshtein"))
    )

    print(f"totals: selected={len(run)} skipped={len(skip)} processed={len(best)}")
    print()

    print("=== FULL FUNNEL per language (sorted by end-to-end success) ===")
    print(f'{"lang":12s} {"selected":>8s} {"skipped":>7s} {"proc":>5s} {"validNLS":>8s} {"%e2e":>5s}')
    for lang in sorted(selected, key=lambda L: valid[L] / selected[L]):
        print(f"{lang:12s} {selected[lang]:8d} {skipped[lang]:7d} {processed[lang]:5d} "
              f"{valid[lang]:8d} {100 * valid[lang] / selected[lang]:4.0f}%")
    print()

    print("=== quality + repetition per language (processed docs) ===")
    print(f'{"lang":12s} {"docNLS":>7s} {"wNLS":>7s} {"cover":>7s} {"halluc":>7s} '
          f'{"rep_mean":>8s} {"rep_max":>7s} {"noLine%":>7s}')
    for lang in sorted(by_language, key=lambda L: mean(column_values(by_language[L], "weighted_along_lines_normalised_levenshtein") or [0])):
        rs = by_language[lang]
        dnls = column_values(rs, "document_normalised_levenshtein")
        wnls = column_values(rs, "weighted_along_lines_normalised_levenshtein")
        cover = column_values(rs, "correct_ref_coverage")
        hall = column_values(rs, "hallucination")
        rep = column_values(rs, "repetition_on_reference")
        used = column_values(rs, "used_line_count")
        no_line = 100 * sum(1 for v in used if v == 0) / len(used) if used else float("nan")
        print(f"{lang:12s} {mean(dnls):7.3f} {mean(wnls):7.3f} {mean(cover):7.3f} "
              f"{mean(hall):7.3f} {mean(rep):8.3f} {(max(rep) if rep else 0):7.3f} {no_line:6.0f}%")
    print()

    print("=== skip reasons (all stages) ===")
    for reason, count in Counter(r["skip_reason"] for r in skip).most_common():
        print(f"{count:4d}  {reason}")


def write_plots(run, skip, best, plots_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir.mkdir(parents=True, exist_ok=True)
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in best:
        by_language[row["main_language"]].append(row)

    # ---- FIG 1: repetition distribution per language ----
    langs = sorted(
        by_language,
        key=lambda L: -(mean(column_values(by_language[L], "repetition_on_reference")) or 0),
    )
    data = [column_values(by_language[L], "repetition_on_reference") or [0] for L in langs]
    fig, ax = plt.subplots(figsize=(15, 8))
    box = ax.boxplot(data, vert=True, showfliers=True, patch_artist=True, widths=0.6)
    for patch in box["boxes"]:
        patch.set(facecolor="#74C0FC", alpha=0.7)
    ax.set_xticks(range(1, len(langs) + 1))
    ax.set_xticklabels(langs, rotation=60, ha="right")
    ax.set_ylabel("repetition_on_reference")
    ax.set_title("Repetition-on-reference distribution per language (best combination per document)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig1 = plots_dir / "repetition_distribution_per_language.png"
    fig.savefig(fig1, dpi=140)
    plt.close(fig)

    # ---- FIG 2: drivers ----
    selected = Counter(r["main_language"] for r in run)
    skipped = Counter(r["main_language"] for r in skip)
    processed = Counter(r["main_language"] for r in best)
    valid = Counter(
        r["main_language"] for r in best
        if is_number(r.get("weighted_along_lines_normalised_levenshtein"))
    )
    order = sorted(selected, key=lambda L: valid[L] / selected[L])

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    x = np.arange(len(order))
    ok = np.array([valid[L] for L in order])
    invalid = np.array([processed[L] - valid[L] for L in order])
    miss = np.array([skipped[L] for L in order])
    axes[0, 0].bar(x, ok, label="valid weighted-NLS", color="#2F9E44")
    axes[0, 0].bar(x, invalid, bottom=ok, label="processed, no usable line", color="#F08C00")
    axes[0, 0].bar(x, miss, bottom=ok + invalid, label="skipped (matrix too small)", color="#E03131")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(order, rotation=70, ha="right", fontsize=8)
    axes[0, 0].set_title("Document funnel per language")
    axes[0, 0].set_ylabel("documents")
    axes[0, 0].legend()

    dn = [mean(column_values(by_language[L], "document_normalised_levenshtein")) for L in by_language]
    wn = [mean(column_values(by_language[L], "weighted_along_lines_normalised_levenshtein") or [float("nan")]) for L in by_language]
    axes[0, 1].scatter(dn, wn, color="#1971C2")
    for lang, a, b in zip(by_language, dn, wn):
        axes[0, 1].annotate(lang, (a, b), fontsize=7)
    axes[0, 1].plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    axes[0, 1].set_xlabel("mean raw document_normalised_levenshtein (model quality)")
    axes[0, 1].set_ylabel("mean weighted_along_lines_normalised_levenshtein (final)")
    axes[0, 1].set_title("Final score vs raw model quality")
    axes[0, 1].grid(alpha=0.3)

    skip_groups = Counter()
    for r in skip:
        reason = r.get("skip_reason", "")
        key = "cols<min (empty prediction)" if reason.startswith("matrix_columns") else \
              "rows<min (short reference)" if reason.startswith("matrix_rows") else "other"
        skip_groups[key] += 1
    axes[1, 0].bar(list(skip_groups.keys()), list(skip_groups.values()), color="#E03131")
    axes[1, 0].set_title(f"Why {len(skip)} docs were skipped")
    axes[1, 0].set_ylabel("documents")

    no_line_rate = []
    for lang in order:
        used = column_values(by_language[lang], "used_line_count")
        no_line_rate.append(100 * sum(1 for v in used if v == 0) / len(used) if used else 0)
    axes[1, 1].bar(range(len(order)), no_line_rate, color="#F08C00")
    axes[1, 1].set_xticks(range(len(order)))
    axes[1, 1].set_xticklabels(order, rotation=70, ha="right", fontsize=8)
    axes[1, 1].set_title("% processed docs with NO surviving line (-> blank weighted-NLS)")
    axes[1, 1].set_ylabel("%")

    fig.suptitle("What influenced the results: model quality + matrix/line failures", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig2 = plots_dir / "result_drivers_per_language.png"
    fig.savefig(fig2, dpi=140)
    plt.close(fig)

    return [fig1, fig2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-language result analysis for a balanced run.")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Balanced run directory containing the *_documents.csv files.")
    parser.add_argument("--no-plots", action="store_true", help="Print tables only, skip PNGs.")
    args = parser.parse_args()

    results_dir = args.results_dir
    run = load_rows(results_dir / "runfile_documents.csv")
    skip = load_rows(results_dir / "skipped_documents.csv")
    best = load_rows(results_dir / "best_combination_per_document.csv")

    print_tables(run, skip, best)

    if not args.no_plots:
        paths = write_plots(run, skip, best, results_dir / "analysis_plots")
        print()
        print("wrote plots:")
        for p in paths:
            print(" ", p)


if __name__ == "__main__":
    main()
