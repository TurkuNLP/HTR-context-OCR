#!/usr/bin/env python3
# %%
"""Notebook-friendly analysis of custom Churro inference run metrics.

This script discovers and compares run folders named:
- results/custom_churro_infer_run<N>
- results/custom_churro_infer_run_run<N>

For each run it reads:
- vllm/test/all_metrics.json

It then builds:
1) A run summary table (printed, handwritten, and overall scores)
2) A grouped bar chart for per-run averages
3) Language heatmaps for printed and handwritten documents

The values in all_metrics.json are already percentage scores (0-100).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR_PATTERN = re.compile(r"^custom_churro_infer_run(?:_run)?(\d+)$")
METRICS_RELATIVE_PATH = Path("vllm") / "test" / "all_metrics.json"
MAIN_SCORE_KEY = "normalized_levenshtein_similarity"

DOC_TYPE_ALIASES = {
    "print": "print",
    "printed": "print",
    "handwriting": "handwriting",
    "handwritten": "handwriting",
}


@dataclass
class RunMetrics:
    source_name: str
    run_name: str
    run_index: int
    variant: str
    run_path: Path
    metrics_path: Path
    printed_score: float | None
    handwritten_score: float | None
    overall_score: float | None
    print_language_scores: dict[str, float]
    handwriting_language_scores: dict[str, float]
    run_label: str = ""


def normalize_doc_type(value: str) -> str | None:
    return DOC_TYPE_ALIASES.get(value.strip().lower())


def to_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def discover_metrics_files(search_roots: Iterable[Path]) -> list[Path]:
    """Find all matching run folders and return existing all_metrics.json paths."""
    metrics_files: list[Path] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for run_dir in root.rglob("results/custom_churro_infer_run*"):
            if not run_dir.is_dir():
                continue
            if not RUN_DIR_PATTERN.match(run_dir.name):
                continue
            metrics_path = run_dir / METRICS_RELATIVE_PATH
            if metrics_path.is_file():
                metrics_files.append(metrics_path)

    # Keep order deterministic and remove duplicates.
    unique_metrics = sorted(set(metrics_files), key=lambda p: str(p))
    return unique_metrics


def parse_language_type_metrics(raw_metrics: Any) -> tuple[dict[str, float], dict[str, float]]:
    """Return (print_language_scores, handwriting_language_scores).

    Supports both known structures:
    1) Nested:
       {"print": {"English": 90.2}, "handwriting": {"English": 82.1}}
    2) Flat:
       {"English_print": 90.2, "English_handwriting": 82.1}
    """
    print_scores: dict[str, float] = {}
    handwriting_scores: dict[str, float] = {}

    if not isinstance(raw_metrics, dict):
        return print_scores, handwriting_scores

    has_nested_buckets = any(
        normalize_doc_type(key) in {"print", "handwriting"} and isinstance(value, dict)
        for key, value in raw_metrics.items()
    )

    if has_nested_buckets:
        for bucket_name, language_map in raw_metrics.items():
            doc_type = normalize_doc_type(bucket_name)
            if doc_type not in {"print", "handwriting"}:
                continue
            if not isinstance(language_map, dict):
                continue
            target = print_scores if doc_type == "print" else handwriting_scores
            for language, score in language_map.items():
                score_float = to_float(score)
                if score_float is not None:
                    target[str(language)] = score_float
        return print_scores, handwriting_scores

    for key, score in raw_metrics.items():
        if "_" not in str(key):
            continue
        language, raw_doc_type = str(key).rsplit("_", 1)
        doc_type = normalize_doc_type(raw_doc_type)
        score_float = to_float(score)
        if doc_type == "print" and score_float is not None:
            print_scores[language] = score_float
        elif doc_type == "handwriting" and score_float is not None:
            handwriting_scores[language] = score_float

    return print_scores, handwriting_scores


def parse_run_metrics(metrics_path: Path) -> RunMetrics:
    run_dir = metrics_path.parent.parent.parent  # .../custom_churro_infer_runN
    run_name = run_dir.name
    match = RUN_DIR_PATTERN.match(run_name)
    run_index = int(match.group(1)) if match else -1
    variant = "run_run" if "_run_run" in run_name else "run"

    source_name = run_dir.parent.parent.name if run_dir.parent.name == "results" else run_dir.parent.name

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    type_metrics = metrics.get("type_metrics", {})
    aggregate_metrics = metrics.get("aggregate_metrics", {})
    language_and_type = metrics.get("main_language_and_type_metrics", {})

    printed_score = None
    handwritten_score = None
    if isinstance(type_metrics, dict):
        for key, value in type_metrics.items():
            doc_type = normalize_doc_type(str(key))
            score = to_float(value)
            if doc_type == "print":
                printed_score = score
            elif doc_type == "handwriting":
                handwritten_score = score

    overall_score = None
    if isinstance(aggregate_metrics, dict):
        overall_score = to_float(aggregate_metrics.get(MAIN_SCORE_KEY))

    print_language_scores, handwriting_language_scores = parse_language_type_metrics(language_and_type)

    return RunMetrics(
        source_name=source_name,
        run_name=run_name,
        run_index=run_index,
        variant=variant,
        run_path=run_dir,
        metrics_path=metrics_path,
        printed_score=printed_score,
        handwritten_score=handwritten_score,
        overall_score=overall_score,
        print_language_scores=print_language_scores,
        handwriting_language_scores=handwriting_language_scores,
    )


def load_all_runs(search_roots: Iterable[Path]) -> list[RunMetrics]:
    metrics_files = discover_metrics_files(search_roots)
    runs: list[RunMetrics] = []
    for metrics_path in metrics_files:
        try:
            runs.append(parse_run_metrics(metrics_path))
        except Exception as exc:
            print(f"[skip] Could not parse {metrics_path}: {exc}")

    # Use source/run name only if duplicates exist.
    name_counts: dict[str, int] = {}
    for run in runs:
        name_counts[run.run_name] = name_counts.get(run.run_name, 0) + 1
    for run in runs:
        run.run_label = (
            f"{run.source_name}/{run.run_name}"
            if name_counts.get(run.run_name, 0) > 1
            else run.run_name
        )

    runs.sort(key=lambda r: (r.run_index, 0 if r.variant == "run" else 1, r.source_name, r.run_name))
    return runs


def build_summary_dataframe(runs: list[RunMetrics]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "run_label": run.run_label,
                "source": run.source_name,
                "run_name": run.run_name,
                "run_index": run.run_index,
                "printed_score": run.printed_score,
                "handwritten_score": run.handwritten_score,
                "overall_score": run.overall_score,
                "metrics_path": str(run.metrics_path),
            }
        )
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["run_index", "run_name", "source"], ascending=[True, True, True]
        ).reset_index(drop=True)
    return summary_df


def build_language_dataframe(runs: list[RunMetrics], doc_type: str) -> pd.DataFrame:
    if doc_type not in {"print", "handwriting"}:
        raise ValueError("doc_type must be either 'print' or 'handwriting'")

    language_maps = [
        run.print_language_scores if doc_type == "print" else run.handwriting_language_scores for run in runs
    ]
    languages = sorted({language for language_map in language_maps for language in language_map})
    if not languages:
        return pd.DataFrame(index=[run.run_label for run in runs])

    matrix_rows = []
    for run in runs:
        language_map = run.print_language_scores if doc_type == "print" else run.handwriting_language_scores
        row = {language: language_map.get(language, np.nan) for language in languages}
        row["run_label"] = run.run_label
        matrix_rows.append(row)

    df = pd.DataFrame(matrix_rows).set_index("run_label")
    return df


def plot_run_averages(summary_df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    metric_columns = {
        "printed_score": "Printed",
        "handwritten_score": "Handwritten",
        "overall_score": "Overall",
    }
    plot_df = (
        summary_df.set_index("run_label")[list(metric_columns)]
        .rename(columns=metric_columns)
        .astype(float)
    )

    fig_width = max(10, len(plot_df) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    plot_df.plot(
        kind="bar",
        ax=ax,
        width=0.8,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )

    ax.set_title("Average Scores Per Run", fontsize=14)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Run")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Metric", frameon=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    for patch in ax.patches:
        height = patch.get_height()
        if np.isfinite(height):
            ax.annotate(
                f"{height:.1f}",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    fig.tight_layout()
    return fig, ax


def plot_language_heatmap(
    language_df: pd.DataFrame,
    title: str,
) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
    if language_df.empty or language_df.shape[1] == 0:
        print(f"[info] Skipping heatmap: no data for {title}")
        return None, None

    values = language_df.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    cmap = plt.get_cmap("YlGnBu")
    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    if hasattr(cmap, "set_bad"):
        cmap.set_bad(color="#f0f0f0")

    fig_width = max(10, language_df.shape[1] * 0.65 + 4)
    fig_height = max(4, language_df.shape[0] * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Score (%)")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Language")
    ax.set_ylabel("Run")
    ax.set_xticks(np.arange(language_df.shape[1]))
    ax.set_xticklabels(language_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(language_df.shape[0]))
    ax.set_yticklabels(language_df.index)

    for i in range(language_df.shape[0]):
        for j in range(language_df.shape[1]):
            score = language_df.iat[i, j]
            if pd.notna(score):
                ax.text(j, i, f"{score:.1f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    return fig, ax


def maybe_save_figure(fig: plt.Figure | None, output_dir: Path, file_name: str) -> None:
    if fig is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"[saved] {output_path}")


# %%
# Configuration for notebook/script execution.
SEARCH_ROOTS = [Path.cwd()]
SAVE_FIGURES = False
FIGURE_OUTPUT_DIR = Path("results") / "plots" / "custom_churro_runs"


# %%
runs = load_all_runs(SEARCH_ROOTS)
if not runs:
    raise RuntimeError(
        "No metrics found. Expected files at: results/custom_churro_infer_run*/vllm/test/all_metrics.json"
    )

summary_df = build_summary_dataframe(runs)
print(f"Loaded {len(summary_df)} runs.")

summary_columns = ["run_label", "printed_score", "handwritten_score", "overall_score", "metrics_path"]
print("\nPer-run score summary:")
print(summary_df[summary_columns].to_string(index=False))

overall_mean = summary_df[["printed_score", "handwritten_score", "overall_score"]].mean(numeric_only=True)
print("\nMean scores across all discovered runs:")
print(overall_mean.rename({"printed_score": "printed", "handwritten_score": "handwritten"}).to_string())


# %%
fig1, _ = plot_run_averages(summary_df)
if SAVE_FIGURES:
    maybe_save_figure(fig1, FIGURE_OUTPUT_DIR, "run_average_scores.png")

print_lang_df = build_language_dataframe(runs, "print")
handwriting_lang_df = build_language_dataframe(runs, "handwriting")

fig2, _ = plot_language_heatmap(print_lang_df, "Printed Document Scores by Language and Run")
if SAVE_FIGURES:
    maybe_save_figure(fig2, FIGURE_OUTPUT_DIR, "language_heatmap_printed.png")

fig3, _ = plot_language_heatmap(
    handwriting_lang_df,
    "Handwritten Document Scores by Language and Run",
)
if SAVE_FIGURES:
    maybe_save_figure(fig3, FIGURE_OUTPUT_DIR, "language_heatmap_handwritten.png")

plt.show()
