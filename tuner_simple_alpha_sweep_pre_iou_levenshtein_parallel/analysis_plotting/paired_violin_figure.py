#!/usr/bin/env python3
from __future__ import annotations

"""Paper figure: paired horizontal violins of along-line quality vs reference coverage.

For every language (same set as `metric_distributions_violin_by_quality.png`, but
**grouped by dominant script** — scripts alphabetical, languages alphabetical within each
group) this draws TWO horizontal violins per language — weighted along-line normalised
Levenshtein similarity (light blue) and correct reference coverage (light orange). Script
groups are separated by an alternating grey background band, a gap, and a single script label
on the right edge; languages within a group are separated by a thin black dashed line. The
legend sits OUTSIDE the chart (above it).

Self-contained and repeatable: reuses only the data layer of `ocr_failure_decomposition.py`
and writes a NEW PNG; never modifies any existing figure. Deterministic.

  /appl/soft/ai/wrap/pytorch-2.9/bin/python3 scratch_tools/paired_violin_figure.py \
    --outputs-json results/custom_churro_infer_dev_run1/vllm/dev/outputs.json \
    --results-dir  results/<tuner-run>/balanced
  # -> writes <results-dir>/analysis_plots/paired_violin_weighted_correct_by_script.png
"""

import argparse
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scratch_tools"))
import ocr_failure_decomposition as ofd  # noqa: E402

FIGURE_DPI = 300
WEIGHTED_COLOR = "#74C0FC"   # light blue
CORRECT_COLOR = "#FDB462"    # light orange
VIOLIN_ALPHA = 0.50
WEIGHTED_LABEL = ("Weighted along-line normalised Levenshtein similarity "
                  "(character accuracy along recognized lines)")
CORRECT_LABEL = ("Correct reference coverage "
                 "(fraction of the reference transcribed exactly once)")

BAND_TINTS = ("#EAF2FB", "#FBF4E6", "#EAF6EE", "#F2EEF8")  # optional cycling tints
BAND_GREYS = ("#FFFFFF", "#F0F0F0")                        # default: alternating greys

LANG_STEP = 1.3            # vertical space per language (every language gets an EQUAL segment)
GROUP_GAP = 0.0           # no extra gap: equal spacing for all languages incl. no-violin ones
VIOLIN_OFFSET = 0.22      # half-distance between a language's two violins (gap between them)
VIOLIN_WIDTH = 0.33
X_STEP = 0.05             # gridline step (0.1 = labelled major, 0.05 = unlabelled dashed minor)
X_EDGE_PAD_PIXELS = 58    # rendered-pixel x-padding before 0.0 and after 1.0 (fixed in pixels)


def dominant_main_script_by_language(documents) -> dict[str, str]:
    """Per language, the most common raw `main_script` over its scored docs (verbatim)."""
    counters: dict[str, Counter] = {}
    for doc in documents:
        if not doc["scored"] or not doc.get("main_script"):
            continue
        counters.setdefault(doc["language"], Counter())[str(doc["main_script"])] += 1
    return {language: sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            for language, counter in counters.items()}


def _layout_positions(order, scripts) -> dict[str, float]:
    """Assign each language a y position, inserting a gap whenever the script group changes."""
    positions: dict[str, float] = {}
    y = 0.0
    previous_script = object()
    for language in order:
        script = scripts.get(language)
        if positions and script != previous_script:
            y += GROUP_GAP
        positions[language] = y
        y += LANG_STEP
        previous_script = script
    return positions


def _draw_horizontal_violin(axis, data, position, width, color) -> None:
    """Draw one horizontal violin with thin black median/min/max lines."""
    if len(data) >= 2 and (max(data) - min(data)) > 1e-9:
        try:
            parts = axis.violinplot([data], positions=[position], widths=width,
                                    showmedians=True, showextrema=True, orientation="horizontal")
        except TypeError:  # older matplotlib uses vert= instead of orientation=
            parts = axis.violinplot([data], positions=[position], widths=width,
                                    showmedians=True, showextrema=True, vert=False)
        for body in parts["bodies"]:
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(VIOLIN_ALPHA)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("black"); parts[key].set_linewidth(0.8)
    elif len(data) >= 1:  # all-equal / single value -> marker (e.g. Khmer correct = 0)
        value = sum(data) / len(data)
        axis.scatter([value], [position], marker="D", s=22, color=color,
                     edgecolors="black", linewidths=0.4, zorder=4)


def _edge_padding_in_data(fig, axis, n_pixels: int) -> tuple[float, float]:
    """Return (dx, dy) data-units equal to n_pixels rendered (savefig-dpi) pixels."""
    fig.canvas.draw()
    fig_px = n_pixels * fig.dpi / FIGURE_DPI   # convert rendered px -> figure-dpi px
    inverse = axis.transData.inverted()
    origin = inverse.transform((0.0, 0.0))
    dx = abs(inverse.transform((fig_px, 0.0))[0] - origin[0])
    dy = abs(inverse.transform((0.0, fig_px))[1] - origin[1])
    return dx, dy


def _place_title_above_legend(fig, axis, legend, title) -> None:
    """Put the title just above the legend: title-legend gap = (plot-legend gap) + 5 px."""
    try:
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        legend_box = legend.get_window_extent().transformed(inv)
        axes_box = axis.get_window_extent().transformed(inv)
        plot_legend_gap = legend_box.y0 - axes_box.y1
        five_px = 5.0 / (fig.get_size_inches()[1] * FIGURE_DPI)
        title_y = legend_box.y1 + plot_legend_gap + five_px
        fig.suptitle(title, y=title_y, va="bottom")
    except Exception:
        fig.suptitle(title, y=1.02)


def build_figure(documents, order, scripts, out_path: Path, *, band_style: str = "grey") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 9, "ytick.labelsize": 11, "legend.fontsize": 12,
    })
    margin = LANG_STEP / 2.0
    positions = _layout_positions(order, scripts)
    extent = (max(positions.values()) - min(positions.values())) + LANG_STEP
    fig, axis = plt.subplots(figsize=(8.0, max(8.0, 0.42 * extent)))

    # --- script-group background bands + one label per group (right edge) ---
    groups: "OrderedDict[object, list]" = OrderedDict()
    for language in order:
        groups.setdefault(scripts.get(language), []).append(positions[language])
    palette = BAND_TINTS if band_style == "tints" else BAND_GREYS
    group_centers, group_labels = [], []
    for index, (script, ys) in enumerate(groups.items()):
        axis.axhspan(min(ys) - margin, max(ys) + margin, color=palette[index % len(palette)], zorder=0)
        group_centers.append((min(ys) + max(ys)) / 2.0)
        group_labels.append(script if script else "")

    # --- gridlines: solid at 0.1 (major, labelled); dashed at 0.05 midpoints (unlabelled) ---
    for x in np.arange(0.0, 1.0 + 1e-9, 0.1):
        axis.axvline(x, color="#cccccc", linewidth=0.5, linestyle="-", zorder=0.5)
    for x in np.arange(0.05, 1.0, 0.1):
        axis.axvline(x, color="#cccccc", linewidth=0.5, linestyle="--", zorder=0.5)

    # --- thin black dashed separator between EVERY pair of adjacent languages ---
    for first, second in zip(order, order[1:]):
        axis.axhline((positions[first] + positions[second]) / 2.0, color="black",
                     linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.85, zorder=0.7)

    # --- the paired violins + annotations for empty/degenerate rows ---
    for language in order:
        y = positions[language]
        weighted = ofd.values(documents, language, "weighted_nls", scored_only=True)
        correct = ofd.values(documents, language, "correct", scored_only=True)
        _draw_horizontal_violin(axis, weighted, y - VIOLIN_OFFSET, VIOLIN_WIDTH, WEIGHTED_COLOR)
        _draw_horizontal_violin(axis, correct, y + VIOLIN_OFFSET, VIOLIN_WIDTH, CORRECT_COLOR)
        if not weighted and not correct:
            axis.text(0.5, y, "no scored documents", ha="center", va="center",
                      fontsize=8, style="italic", color="#666666", zorder=5)
        elif not weighted:
            axis.text(0.015, y - VIOLIN_OFFSET, "no along-line quality (coverage = 0, shown as marker)",
                      ha="left", va="center", fontsize=7, style="italic", color="#666666", zorder=5)

    # --- axes ---
    major_ticks = np.arange(0.0, 1.0 + 1e-9, 0.1)
    axis.set_xticks(major_ticks)
    axis.set_xticklabels([f"{t:.1f}" for t in major_ticks])
    axis.set_xlabel("Similarity / coverage  (0 = none, 1 = perfect)")
    axis.set_axisbelow(True)
    axis.set_yticks([positions[language] for language in order])
    axis.set_yticklabels(order)
    y_low = min(positions.values()) - margin
    y_high = max(positions.values()) + margin
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(y_low, y_high)
    axis.invert_yaxis()  # first group (Arabic) at the top

    # x padding: fixed in pixels (X_EDGE_PAD_PIXELS) so the pixel gap before 0.0 and after 1.0
    # stays identical regardless of figure width. No y-axis pixel padding.
    pad_x, _ = _edge_padding_in_data(fig, axis, X_EDGE_PAD_PIXELS)
    axis.set_xlim(0.0 - pad_x, 1.0 + pad_x)
    axis.set_ylim(y_high, y_low)  # inverted (first group on top), bands flush with plot edge

    right = axis.twinx()
    right.set_ylim(axis.get_ylim())
    right.set_yticks(group_centers)
    right.set_yticklabels(group_labels)
    right.tick_params(length=0)
    right.set_ylabel("Dominant script (from outputs.json)", rotation=270, labelpad=22)

    handles = [Patch(facecolor=WEIGHTED_COLOR, alpha=VIOLIN_ALPHA, label=WEIGHTED_LABEL),
               Patch(facecolor=CORRECT_COLOR, alpha=VIOLIN_ALPHA, label=CORRECT_LABEL)]
    legend = axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
                         ncol=1, frameon=False)

    _place_title_above_legend(
        fig, axis, legend,
        "Per-language along-line transcription quality and reference coverage\n"
        "(scored documents, grouped by dominant script; languages and scripts alphabetical)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired horizontal violin figure (paper-ready).")
    parser.add_argument("--outputs-json", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--band-style", choices=["grey", "tints"], default="grey")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.output or (args.results_dir / "analysis_plots"
                               / "paired_violin_weighted_correct_by_script.png")
    outputs = ofd.load_outputs_json(args.outputs_json)
    partition = ofd.load_balanced_partition(args.results_dir)
    documents, _ = ofd.build_documents(outputs, partition)
    scripts = dominant_main_script_by_language(documents)
    all_languages = sorted({doc["language"] for doc in documents})
    order = sorted(all_languages, key=lambda language: (scripts.get(language) or "￿", language))

    written = build_figure(documents, order, scripts, out_path, band_style=args.band_style)
    print("languages:", len(order), "| wrote:", written)


if __name__ == "__main__":
    main()
