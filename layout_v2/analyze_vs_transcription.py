#!/usr/bin/env python3
"""Phase F — the proof analysis: do layout / page-complexity measurements predict transcription NLS?

Joins layout_v2's per-document annotations to per-document transcription scores (normalized
Levenshtein similarity, "NLS") and produces the deliverables of IMPLEMENTATION_PLAN.md §12:
- per-category and per-column-bin NLS tables (printed / handwritten separated);
- scatter + box plots (NLS vs column bin, NLS vs items) with per-language views;
- Spearman correlations, overall and within language (rank-based: robust to the bounded metric);
- a language-controlled regression — fractional-logit GLM when statsmodels is available, OLS as
  the fallback/appendix — reported as the change in language-coefficient spread once layout
  terms enter ("how much of the apparent language effect is layout");
- the within-Finnish contrast (2-3 columns vs 7+; language held fixed by construction).

Defaults per project decisions: join target = the existing dev transcription run (decision #11,
overridable/repeatable via --nls-outputs); needs_review pages excluded with one sensitivity
re-run including them (decision #17); vertical-script pages excluded from column analyses
(decision #14).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# --------------------------------------------------------------------------------------
# Loading + joining
# --------------------------------------------------------------------------------------
def load_layout(run_split_dir: Path) -> pd.DataFrame:
    """Flatten layout_outputs.json records into one analysis row per document."""
    records = json.loads((run_split_dir / "layout_outputs.json").read_text(encoding="utf-8"))
    rows = []
    for record in records:
        derived = record.get("derived") or {}
        rows.append(
            {
                "stem": Path(str(record.get("file_name") or "")).stem,
                "main_language": record.get("main_language", "unknown"),
                "document_type": record.get("document_type", "unknown"),
                "dataset_source": record.get("dataset_id", "unknown"),
                "category": derived.get("document_category"),
                "parts": derived.get("independent_parts"),
                "columns": derived.get("column_count_dominant"),
                "article_count": derived.get("article_count"),
                "advertisement_count": derived.get("advertisement_count"),
                "entry_count": derived.get("entry_count"),
                "needs_review": bool(derived.get("needs_review")),
                "vertical_script": bool(derived.get("vertical_script")),
                "gold_chars": record.get("gold_chars", 0),
            }
        )
    frame = pd.DataFrame(rows)
    frame["column_bin"] = frame["columns"].map(config.column_bin)
    # Total items = articles + ads + entries, treating not-applicable (None) as 0 for the sum.
    for column in ("article_count", "advertisement_count", "entry_count"):
        frame[column + "_filled"] = frame[column].fillna(0)
    frame["total_items"] = (
        frame["article_count_filled"] + frame["advertisement_count_filled"] + frame["entry_count_filled"]
    )
    return frame


def load_nls(paths: list[Path]) -> pd.DataFrame:
    """Per-document NLS from one or more Churro transcription ``outputs.json`` files (averaged).

    Each file is a list of records carrying an image file name and
    ``normalized_levenshtein_similarity``; multiple files average per document (NLS noise).
    """
    per_doc: dict[str, list[float]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Accept both a bare list and a dict with a list under a conventional key.
        records = data if isinstance(data, list) else data.get("outputs") or data.get("results") or []
        for record in records:
            if not isinstance(record, dict):
                continue
            name = record.get("file_name") or record.get("image") or record.get("filename") or ""
            score = record.get("normalized_levenshtein_similarity")
            if name and isinstance(score, (int, float)):
                per_doc.setdefault(Path(str(name)).stem, []).append(float(score))
    return pd.DataFrame(
        {"stem": list(per_doc.keys()), "nls": [float(np.mean(v)) for v in per_doc.values()]}
    )


# --------------------------------------------------------------------------------------
# Tables + correlations
# --------------------------------------------------------------------------------------
def bin_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean NLS per column bin x document type — the headline descriptive table."""
    usable = frame[frame["column_bin"] != ""]
    table = (
        usable.groupby(["document_type", "column_bin"])
        .agg(n=("nls", "size"), mean_nls=("nls", "mean"), median_nls=("nls", "median"))
        .reset_index()
    )
    order = [label for _, _, label in config.COLUMN_BINS]
    table["column_bin"] = pd.Categorical(table["column_bin"], categories=order, ordered=True)
    return table.sort_values(["document_type", "column_bin"])


def spearman_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank correlations of NLS with each complexity measure, overall and per language (n>=15)."""
    measures = ("columns", "parts", "total_items", "gold_chars")
    rows = []

    def corr(sub: pd.DataFrame, scope: str) -> None:
        for measure in measures:
            pair = sub[["nls", measure]].dropna()
            if len(pair) >= 15 and pair[measure].nunique() > 2:
                rows.append(
                    {"scope": scope, "measure": measure, "n": len(pair),
                     "spearman_rho": round(pair["nls"].corr(pair[measure], method="spearman"), 3)}
                )

    corr(frame, "ALL")
    for language, sub in frame.groupby("main_language"):
        corr(sub, str(language))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# The language-controlled regression (the shrinkage statistic)
# --------------------------------------------------------------------------------------
def language_shrinkage(frame: pd.DataFrame) -> dict:
    """Fit NLS ~ language (model L) vs NLS ~ language + layout (model L+S); compare spreads.

    The headline number: how much the spread of per-language effects shrinks once layout terms
    enter — i.e. how much of the apparent "language effect" is document layout. Engine:
    fractional-logit GLM (NLS is bounded in [0,1]) via statsmodels when available; plain OLS as
    the fallback so the analysis never hard-fails on a lean environment.
    """
    usable = frame.dropna(subset=["nls", "columns", "parts"]).copy()
    usable = usable[usable["gold_chars"] > 0]
    if len(usable) < 100:
        return {"note": f"insufficient joined rows for regression (n={len(usable)})"}

    # Design matrices: language dummies (drop-first) +/- the layout terms.
    usable["log_gold_chars"] = np.log(usable["gold_chars"])
    lang = pd.get_dummies(usable["main_language"], prefix="lang", drop_first=True, dtype=float)
    bins = pd.get_dummies(usable["column_bin"], prefix="colbin", drop_first=True, dtype=float)
    layout = pd.concat(
        [bins, usable[["parts", "total_items", "log_gold_chars"]].astype(float)], axis=1
    )
    y = usable["nls"].clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)  # keep the logit finite

    def fit(design: pd.DataFrame) -> dict[str, float]:
        """Return {feature: coefficient}; GLM-Binomial (fractional logit) or OLS fallback."""
        X = design.to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])  # intercept
        names = ["const", *design.columns]
        try:
            import statsmodels.api as sm  # optional dependency; the honest engine

            model = sm.GLM(y, X, family=sm.families.Binomial())
            params = model.fit().params
        except Exception:  # pragma: no cover - lean env fallback
            params, *_ = np.linalg.lstsq(X, y, rcond=None)
        return dict(zip(names, [float(p) for p in params]))

    coefs_lang_only = fit(lang)
    coefs_full = fit(pd.concat([lang, layout], axis=1))

    def language_spread(coefs: dict[str, float]) -> float:
        values = [v for k, v in coefs.items() if k.startswith("lang_")]
        return float(np.std(values)) if values else float("nan")

    spread_before = language_spread(coefs_lang_only)
    spread_after = language_spread(coefs_full)
    return {
        "n": int(len(usable)),
        "language_coef_spread_language_only": round(spread_before, 4),
        "language_coef_spread_with_layout": round(spread_after, 4),
        "shrinkage_pct": round(100 * (1 - spread_after / spread_before), 1) if spread_before else None,
        "layout_coefficients": {
            k: round(v, 4) for k, v in coefs_full.items()
            if k.startswith("colbin_") or k in ("parts", "total_items", "log_gold_chars")
        },
    }


def within_finnish_contrast(frame: pd.DataFrame) -> dict:
    """Finnish pages, 2-3 columns vs 7+: language fixed by construction (the direct exhibit)."""
    finnish = frame[(frame["main_language"].str.lower() == "finnish") & frame["nls"].notna()]
    low = finnish[finnish["column_bin"].isin(["1", "2-3"])]["nls"]
    high = finnish[finnish["column_bin"] == "7+"]["nls"]
    if low.empty or high.empty:
        return {"note": "insufficient Finnish pages in one of the bins"}
    return {
        "n_low_bin": int(len(low)), "mean_nls_low_bin": round(float(low.mean()), 4),
        "n_high_bin": int(len(high)), "mean_nls_high_bin": round(float(high.mean()), 4),
        "gap": round(float(low.mean() - high.mean()), 4),
    }


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_nls_by_bin(frame: pd.DataFrame, out_dir: Path) -> None:
    """Box plot of NLS per column bin, printed documents (the mechanism's home turf)."""
    printed = frame[(frame["document_type"] == "print") & (frame["column_bin"] != "")]
    order = [label for _, _, label in config.COLUMN_BINS]
    data = [printed[printed["column_bin"] == b]["nls"].dropna() for b in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=[f"{b}\n(n={len(d)})" for b, d in zip(order, data)])
    ax.set_xlabel("column count bin (dominant part)")
    ax.set_ylabel("transcription NLS")
    ax.set_title("Printed documents: transcription quality vs column structure")
    fig.tight_layout()
    fig.savefig(out_dir / "nls_by_column_bin_printed.png", dpi=150)
    plt.close(fig)


def plot_items_scatter(frame: pd.DataFrame, out_dir: Path) -> None:
    """NLS vs total items (log x), colored by language — the fragmentation view."""
    usable = frame[(frame["total_items"] > 0) & frame["nls"].notna()]
    fig, ax = plt.subplots(figsize=(8, 5))
    for language, sub in usable.groupby("main_language"):
        if len(sub) >= 15:  # keep the legend readable: only languages with enough pages
            ax.scatter(sub["total_items"], sub["nls"], s=14, alpha=0.6, label=str(language))
    ax.set_xscale("log")
    ax.set_xlabel("total items on page (articles + advertisements + entries)")
    ax.set_ylabel("transcription NLS")
    ax.set_title("Transcription quality vs page fragmentation")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "nls_vs_total_items.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def run_analysis(frame: pd.DataFrame, out_dir: Path, tag: str) -> dict:
    """One full analysis pass over a (possibly filtered) joined frame; returns the headline dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_table(frame).to_csv(out_dir / f"nls_by_column_bin_{tag}.csv", index=False)
    spearman_summary(frame).to_csv(out_dir / f"spearman_{tag}.csv", index=False)
    plot_nls_by_bin(frame, out_dir)
    plot_items_scatter(frame, out_dir)
    return {
        "tag": tag,
        "n_joined": int(frame["nls"].notna().sum()),
        "regression": language_shrinkage(frame),
        "within_finnish": within_finnish_contrast(frame),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-run", type=Path, required=True,
                        help="layout_v2 run split dir (…/dev) with layout_outputs.json")
    parser.add_argument("--nls-outputs", type=Path, nargs="+",
                        default=[Path(config.DEFAULT_NLS_OUTPUTS)],
                        help="One or more transcription outputs.json files (averaged per doc).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <layout-run>/analysis")
    args = parser.parse_args()
    out_dir = args.out_dir or (args.layout_run / "analysis")

    layout = load_layout(args.layout_run)
    nls = load_nls(list(args.nls_outputs))
    joined = layout.merge(nls, on="stem", how="left")
    print(f"[analysis] layout rows={len(layout)} nls rows={len(nls)} joined with nls={joined['nls'].notna().sum()}")

    # Primary frame per decisions #14/#17: drop vertical-script pages and needs_review pages.
    primary = joined[~joined["vertical_script"] & ~joined["needs_review"]]
    headline = run_analysis(primary, out_dir, tag="primary")
    # Sensitivity: same analysis WITH the review-flagged pages (vertical stays excluded).
    sensitivity = run_analysis(joined[~joined["vertical_script"]], out_dir, tag="with_review_pages")

    report = {"primary": headline, "sensitivity_with_review_pages": sensitivity}
    (out_dir / "headline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Short human-readable summary — the numbers a reader quotes.
    lines = [
        "# Layout vs transcription quality — headline results", "",
        f"Joined documents (primary frame): {headline['n_joined']}", "",
        "## Language-effect shrinkage (fractional-logit, language dummies)",
        "```json", json.dumps(headline["regression"], indent=2), "```", "",
        "## Within-Finnish contrast (language fixed by construction)",
        "```json", json.dumps(headline["within_finnish"], indent=2), "```", "",
        "Sensitivity run including needs_review pages: see headline.json.",
    ]
    (out_dir / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[analysis] wrote {out_dir}/ANALYSIS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
