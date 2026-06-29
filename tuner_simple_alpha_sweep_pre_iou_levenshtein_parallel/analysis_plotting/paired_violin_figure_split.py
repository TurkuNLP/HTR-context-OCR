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
WEIGHTED_COLOR = "#20B2AA"   # lightseagreen
CORRECT_COLOR  = "#E69F00"   # Okabe-Ito orange
VIOLIN_ALPHA = 0.50
WEIGHTED_LABEL = ("Weighted along-line normalised Levenshtein similarity "
                  "(character accuracy along recognized lines)")
CORRECT_LABEL = ("Correct reference coverage "
                 "(fraction of the reference transcribed exactly once)")

BAND_TINTS = ("#EAF2FB", "#FBF4E6", "#EAF6EE", "#F2EEF8")  # optional cycling tints
BAND_GREYS = ("#FFFFFF", "#F0F0F0")                        # default: alternating greys

LANG_STEP = 1.8            # vertical space per language (increased for larger violin bodies)
GROUP_GAP = 0.0           # no extra gap: equal spacing for all languages incl. no-violin ones
VIOLIN_OFFSET = 0.32      # half-distance between a language's two violins (increased)
VIOLIN_WIDTH = 0.52       # total violin width; each half uses VIOLIN_WIDTH/2 (increased)
X_STEP = 0.05             # gridline step (0.1 = labelled major, 0.05 = unlabelled dashed minor)
X_EDGE_PAD = 0.05         # x-padding on each side in data units (= one minor grid step)
SPLIT_GAP_PIXELS      = 0    # no gap between halves of the same violin; both share the center baseline
INTER_INNER_GAP_PIXELS = 3   # minimum rendered-pixel clearance between inner halves of adjacent violin pairs

VIOLIN_BORDER_LW = 0.96        # violin border linewidth in points (≈ 1 px thinner than 1.2 at 300 DPI)

SCRIPT_ISO_CODES: dict[str, str] = {
    "Arabic":                                          "Arab",
    "Bengali (Bangla)":                               "Beng",
    "Cyrillic":                                       "Cyrl",
    "Devanagari (Nagari)":                            "Deva",
    "Greek":                                          "Grek",
    "Han (Hànzi, Kanji, Hanja)":                      "Hani",
    "Hebrew":                                         "Hebr",
    "Japanese (alias for Han + Hiragana + Katakana)": "Jpan",
    "Khmer":                                          "Khmr",
    "Latin":                                          "Latn",
}


def _to_code(raw: str) -> str:
    """Map a verbatim main_script string to its ISO 15924 4-character code."""
    import unicodedata
    return SCRIPT_ISO_CODES.get(unicodedata.normalize("NFC", raw), raw[:4])


from matplotlib.patches import Patch as _Patch
from matplotlib.legend_handler import HandlerBase as _HandlerBase


class _HalfViolinProxy(_Patch):
    """Proxy legend handle carrying sign and fill colour for the mini-violin handler."""
    def __init__(self, sign: int, fill_color: str, **kwargs):
        super().__init__(facecolor="none", edgecolor="none", **kwargs)
        self._hv_sign = sign       # +1 = opens upward in legend (NLS), -1 = downward (Cov)
        self._hv_fill = fill_color


class HalfViolinLegendHandler(_HandlerBase):
    """Renders a miniature half-violin (Gaussian bump) in the legend icon box."""

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        import numpy as np
        import matplotlib.colors as mcolors
        from matplotlib.patches import Polygon as _Polygon

        sign       = getattr(orig_handle, '_hv_sign', +1)
        fill_color = getattr(orig_handle, '_hv_fill', "#aaaaaa")

        n  = 80
        xs = np.linspace(0, width, n)
        bump = np.exp(-0.5 * ((xs - width * 0.5) / (width / 5.0)) ** 2)
        bump_h = bump * (height * 0.46)   # fill ~46 % of icon height per half

        x0     = -xdescent
        y_base = -ydescent + height / 2   # midline of icon; y increases upward in legend coords

        verts = ([(x0, y_base)]
                 + [(x0 + xs[i], y_base + sign * bump_h[i]) for i in range(n)]
                 + [(x0 + width, y_base)])

        face_rgba = (*mcolors.to_rgb(fill_color), VIOLIN_ALPHA)
        p = _Polygon(verts, closed=True, facecolor=face_rgba,
                     edgecolor="black", linewidth=0.5, transform=trans)
        return [p]


class _TextIconProxy(_Patch):
    """Proxy legend handle that renders a short bold text label in the icon area."""
    def __init__(self, icon_text: str, **kwargs):
        super().__init__(facecolor="none", edgecolor="none", **kwargs)
        self._icon_text = icon_text


class TextIconLegendHandler(_HandlerBase):
    """Renders a short bold text string (e.g. 'NLS') in the legend icon box."""

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        from matplotlib.text import Text as _Text
        txt = getattr(orig_handle, '_icon_text', '')
        t = _Text(
            -xdescent + width / 2,
            -ydescent + height / 2,
            txt,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight='bold',
            transform=trans,
        )
        return [t]


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


def _values_by_type(documents, language, key, doc_type: str) -> list[float]:
    """Like ofd.values(scored_only=True) but filtered to one document_type value."""
    out = []
    for doc in documents:
        if doc["language"] != language:
            continue
        if not doc["scored"]:
            continue
        if doc.get("document_type") != doc_type:
            continue
        v = doc.get(key)
        if v is not None:
            out.append(float(v))
    return out


def _draw_split_violin(axis, data_print, data_hw, center_y, half_width, fill_color,
                       half_gap: float, *, side: str = "both") -> None:
    """Horizontal split violin.

    Top half  (y < center_y, visually above on inverted y-axis) = printed  → PRINTED_BORDER.
    Bottom half (y > center_y, visually below)                  = handwritten → HW_BORDER.

    Each half:
      - KDE body hard-clipped to [min, max] of that half's own data (no tail bleed).
      - Horizontal bar at its own flat edge (y_print or y_hw).
      - Median / min / max ticks pointing ONLY into its own half; never crossing the gap.
    Fill interior is semi-transparent (VIOLIN_ALPHA); border is fully opaque.

    half_gap: data-unit displacement of each half's flat edge from center_y,
              giving a total gap of 2 × half_gap between the two halves.
    """
    import numpy as np
    import matplotlib.colors as mcolors
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Polygon as MplPolygon

    face_rgba = (*mcolors.to_rgb(fill_color), VIOLIN_ALPHA)  # opaque edge, transparent fill

    y_print = center_y - half_gap   # flat base of printed half  (inverted y: visually above)
    y_hw    = center_y + half_gap   # flat base of hw half       (inverted y: visually below)

    def _draw_half(data, y_base, sign, border_color):
        """
        sign = -1 → printed (extends toward smaller y, i.e. visually upward).
        sign = +1 → handwritten (extends toward larger y, i.e. visually downward).
        All drawn elements stay on the sign side of y_base; nothing crosses into the other half.
        """
        if not data:
            return
        dmin = float(min(data))
        dmax = float(max(data))
        med  = float(np.median(data))
        tick_span  = half_width          # median tick: full half height
        tick_short = half_width * 0.45   # min / max ticks: shorter

        if len(data) >= 2 and (dmax - dmin) > 1e-9:
            kde    = gaussian_kde(data, bw_method="scott")
            # Hard clip: KDE evaluated only between data min and max — no tail bleed.
            x_grid = np.linspace(dmin, dmax, 300)
            dens   = kde(x_grid)
            dens   = dens / dens.max() * half_width   # shape only; not count-weighted
            verts  = ([(dmin, y_base)]
                      + list(zip(x_grid, y_base + sign * dens))
                      + [(dmax, y_base)])
            axis.add_patch(MplPolygon(verts, closed=True,
                                      facecolor=face_rgba,
                                      edgecolor=border_color,
                                      linewidth=VIOLIN_BORDER_LW, zorder=3))
        elif len(data) >= 1:
            axis.scatter([dmin], [y_base + sign * half_width * 0.5],
                         marker="D", s=18, color=fill_color,
                         edgecolors=border_color, linewidths=0.6, zorder=4)

        # Horizontal bar at the flat edge (this half only)
        axis.plot([dmin, dmax], [y_base, y_base],
                  color="black", linewidth=0.8, zorder=5, solid_capstyle="butt")
        # Median tick: from y_base into this half only (sign direction)
        axis.plot([med, med], [y_base, y_base + sign * tick_span],
                  color="black", linewidth=0.8, zorder=5, solid_capstyle="butt")
        # Min / max ticks: shorter, same direction
        for x in (dmin, dmax):
            axis.plot([x, x], [y_base, y_base + sign * tick_short],
                      color="black", linewidth=0.8, zorder=5, solid_capstyle="butt")

    if side in ("both", "print"):
        _draw_half(data_print, y_base=y_print, sign=-1, border_color=fill_color)
    if side in ("both", "handwriting"):
        _draw_half(data_hw,    y_base=y_hw,    sign=+1, border_color=fill_color)


def _draw_paired_metric_violin(axis, nls_data, cov_data, center_y, half_width):
    """One split violin: top half (opens up) = NLS (WEIGHTED_COLOR), bottom half
    (opens down) = Cov (CORRECT_COLOR). Both flat baselines lie on center_y, so the
    halves touch only along the center line. Border == fill colour; fill = VIOLIN_ALPHA."""
    import numpy as np
    import matplotlib.colors as mcolors
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Polygon as MplPolygon

    def _half(data, sign, color):
        if not data:
            return
        face_rgba = (*mcolors.to_rgb(color), VIOLIN_ALPHA)
        dmin, dmax = float(min(data)), float(max(data))
        med = float(np.median(data))
        tick_span, tick_short = half_width, half_width * 0.45
        if len(data) >= 2 and (dmax - dmin) > 1e-9:
            kde  = gaussian_kde(data, bw_method="scott")
            xg   = np.linspace(dmin, dmax, 300)
            dens = kde(xg); dens = dens / dens.max() * half_width
            verts = ([(dmin, center_y)]
                     + list(zip(xg, center_y + sign * dens))
                     + [(dmax, center_y)])
            axis.add_patch(MplPolygon(verts, closed=True, facecolor=face_rgba,
                                      edgecolor=color, linewidth=VIOLIN_BORDER_LW, zorder=3))
        elif len(data) >= 1:
            axis.scatter([dmin], [center_y + sign * half_width * 0.5], marker="D", s=18,
                         color=color, edgecolors=color, linewidths=0.6, zorder=4)
        axis.plot([dmin, dmax], [center_y, center_y], color="black",
                  linewidth=0.8, zorder=5, solid_capstyle="butt")
        axis.plot([med, med], [center_y, center_y + sign * tick_span], color="black",
                  linewidth=0.8, zorder=5, solid_capstyle="butt")
        for x in (dmin, dmax):
            axis.plot([x, x], [center_y, center_y + sign * tick_short], color="black",
                      linewidth=0.8, zorder=5, solid_capstyle="butt")

    _half(nls_data, sign=-1, color=WEIGHTED_COLOR)   # top half (visually up)
    _half(cov_data, sign=+1, color=CORRECT_COLOR)    # bottom half (visually down)


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

    # Preliminary y-limits (needed so the axis transform is valid when we compute half_gap).
    y_low  = min(positions.values()) - margin
    y_high = max(positions.values()) + margin
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(y_low, y_high)
    axis.invert_yaxis()   # first group (Arabic) at the top

    # half_gap: data-unit displacement of each violin half from the shared center_y.
    # Total gap between the two halves = 2 × half_gap ≈ SPLIT_GAP_PIXELS rendered pixels.
    _, half_gap = _edge_padding_in_data(fig, axis, SPLIT_GAP_PIXELS // 2)
    # Collision guard: the NLS handwritten bottom and coverage printed top face each other across
    # the 2×VIOLIN_OFFSET gap. Each extends VIOLIN_WIDTH/2 from its base, leaving:
    #   clearance = 2×VIOLIN_OFFSET − 2×half_gap − VIOLIN_WIDTH
    # Solving for half_gap to guarantee clearance ≥ INTER_INNER_GAP_PIXELS rendered pixels:
    #   half_gap ≤ VIOLIN_OFFSET − VIOLIN_WIDTH/2 − inner_gap_du/2
    _, inner_gap_du = _edge_padding_in_data(fig, axis, INTER_INNER_GAP_PIXELS)
    half_gap = min(half_gap, max(0.0, VIOLIN_OFFSET - VIOLIN_WIDTH / 2 - inner_gap_du / 2))

    # --- script-group background bands + one label per group (right edge) ---
    groups: "OrderedDict[object, list]" = OrderedDict()
    for language in order:
        groups.setdefault(scripts.get(language), []).append(positions[language])
    palette = BAND_TINTS if band_style == "tints" else BAND_GREYS
    group_centers, group_labels = [], []
    for index, (script, ys) in enumerate(groups.items()):
        axis.axhspan(min(ys) - margin, max(ys) + margin, color=palette[index % len(palette)], zorder=0)
        group_centers.append((min(ys) + max(ys)) / 2.0)
        group_labels.append(_to_code(script) if script else "")

    # --- gridlines: solid at 0.1 (major, labelled); dashed at 0.05 midpoints (unlabelled) ---
    for x in np.arange(0.0, 1.0 + 1e-9, 0.1):
        axis.axvline(x, color="#cccccc", linewidth=0.5, linestyle="-", zorder=0.5)
    for x in np.arange(0.05, 1.0, 0.1):
        axis.axvline(x, color="#cccccc", linewidth=0.5, linestyle="--", zorder=0.5)

    # --- thin black dashed separator between EVERY pair of adjacent languages ---
    for first, second in zip(order, order[1:]):
        axis.axhline((positions[first] + positions[second]) / 2.0, color="black",
                     linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.85, zorder=0.7)

    # --- split violins (top=printed, bottom=handwritten) + annotations ---
    half_w = VIOLIN_WIDTH / 2           # each half uses half the full-violin width
    for language in order:
        y = positions[language]
        weighted = ofd.values(documents, language, "weighted_nls", scored_only=True)
        correct  = ofd.values(documents, language, "correct", scored_only=True)
        w_print = _values_by_type(documents, language, "weighted_nls", "print")
        w_hw    = _values_by_type(documents, language, "weighted_nls", "handwriting")
        c_print = _values_by_type(documents, language, "correct",      "print")
        c_hw    = _values_by_type(documents, language, "correct",      "handwriting")
        _draw_split_violin(axis, w_print, w_hw, y - VIOLIN_OFFSET, half_w, WEIGHTED_COLOR, half_gap)
        _draw_split_violin(axis, c_print, c_hw, y + VIOLIN_OFFSET, half_w, CORRECT_COLOR,  half_gap)
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

    # x padding: one minor grid step on each side so NLS/Cov labels have room
    pad_x = X_EDGE_PAD
    axis.set_xlim(0.0 - pad_x, 1.0 + pad_x)
    axis.set_ylim(y_high, y_low)  # flush y edges (no extra pixel padding)

    # NLS / Cov orientation labels in the left x-padding zone (after final xlim is set)
    _x_left = axis.get_xlim()[0] + pad_x * 0.08
    for _lang in order:
        _y_lang = positions[_lang]
        _has_data = (bool(ofd.values(documents, _lang, "weighted_nls", scored_only=True)) or
                     bool(ofd.values(documents, _lang, "correct",       scored_only=True)))
        if _has_data:
            axis.text(_x_left, _y_lang - VIOLIN_OFFSET, "NLS", ha="left", va="center",
                      fontsize=7, style="italic", color="#999999", zorder=6)
            axis.text(_x_left, _y_lang + VIOLIN_OFFSET, "Cov", ha="left", va="center",
                      fontsize=7, style="italic", color="#999999", zorder=6)

    right = axis.twinx()
    right.set_ylim(axis.get_ylim())
    right.set_yticks(group_centers)
    right.set_yticklabels(group_labels)
    right.tick_params(length=0)
    right.set_ylabel("Script (ISO 15924)", rotation=270, labelpad=22)

    # Mirror x-axis to the top (R2 fix: created after right so legend can be pushed above both)
    ax_top = axis.twiny()
    ax_top.set_xlim(axis.get_xlim())
    ax_top.set_xticks(major_ticks)
    ax_top.set_xticklabels([f"{t:.1f}" for t in major_ticks])
    ax_top.tick_params(axis="x", labelsize=9, length=4, direction="out")

    from matplotlib.lines import Line2D
    # Grouped by metric: text icon → printed violin → handwritten violin
    nls_handles = [
        _TextIconProxy("NLS", label=WEIGHTED_LABEL),
        _HalfViolinProxy(sign=+1, fill_color=WEIGHTED_COLOR, label="NLS printed"),
        _HalfViolinProxy(sign=-1, fill_color=WEIGHTED_COLOR, label="NLS handwritten"),
    ]
    cov_handles = [
        _TextIconProxy("Cov", label=CORRECT_LABEL),
        _HalfViolinProxy(sign=+1, fill_color=CORRECT_COLOR, label="Cov printed"),
        _HalfViolinProxy(sign=-1, fill_color=CORRECT_COLOR, label="Cov handwritten"),
    ]
    stat_handles = [
        Line2D([0], [0], color="black", linewidth=0.8,
               label="Horizontal bar: data range (min → max)"),
        Line2D([0], [0], color="black", linewidth=0.8,
               label="Solid tall tick: median"),
        Line2D([0], [0], color="black", linewidth=0.8,
               label="Short solid ticks: min and max"),
    ]
    # 2 spacers pad the left column to 11 entries so the right column begins exactly at the ISO header
    spacer_handles = [Line2D([0], [0], color="none", label=" ") for _ in range(2)]
    iso_handles = [
        Line2D([0], [0], color="none", label=lbl)
        for lbl in [
            "— ISO 15924 codes —",
            "Arab = Arabic",    "Beng = Bengali",    "Cyrl = Cyrillic",
            "Deva = Devanagari","Grek = Greek",       "Hani = Han",
            "Hebr = Hebrew",    "Jpan = Japanese",   "Khmr = Khmer",
            "Latn = Latin",
        ]
    ]
    legend = axis.legend(
        handles=nls_handles + cov_handles + stat_handles + spacer_handles + iso_handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.05),
        ncol=2, frameon=False, prop={"size": 8},
        handler_map={_HalfViolinProxy: HalfViolinLegendHandler(),
                     _TextIconProxy:   TextIconLegendHandler()})

    _place_title_above_legend(
        fig, axis, legend,
        "Per-language along-line transcription quality and reference coverage\n"
        "(scored documents, grouped by dominant script; languages and scripts alphabetical)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_two_panel_figure(documents, order, scripts, out_path: Path, *,
                           band_style: str = "grey") -> Path:
    """Two side-by-side panels: left = printed only, right = handwritten only.

    Each panel shows only languages that have scored documents of that type.
    Independent y-axes (panels may have different language counts).
    Single shared legend above both panels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import OrderedDict as _OD

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 9, "ytick.labelsize": 11, "legend.fontsize": 12,
    })
    margin = LANG_STEP / 2.0

    order_print = [lang for lang in order
                   if (_values_by_type(documents, lang, "weighted_nls", "print")
                       or _values_by_type(documents, lang, "correct", "print"))]
    order_hw    = [lang for lang in order
                   if (_values_by_type(documents, lang, "weighted_nls", "handwriting")
                       or _values_by_type(documents, lang, "correct", "handwriting"))]

    pos_print = _layout_positions(order_print, scripts) if order_print else {}
    pos_hw    = _layout_positions(order_hw,    scripts) if order_hw    else {}

    def _extent(pos):
        return (max(pos.values()) - min(pos.values()) + LANG_STEP) if pos else LANG_STEP

    fig_height = max(8.0, 0.42 * max(_extent(pos_print), _extent(pos_hw)))
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16.0, fig_height))
    fig.subplots_adjust(wspace=0.50)

    def _draw_panel(ax, panel_order, positions, doc_type):
        if not positions:
            return [], []
        y_low  = min(positions.values()) - margin
        y_high = max(positions.values()) + margin
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(y_low, y_high)
        ax.invert_yaxis()

        groups = _OD()
        for language in panel_order:
            groups.setdefault(scripts.get(language), []).append(positions[language])
        palette = BAND_TINTS if band_style == "tints" else BAND_GREYS
        group_centers, group_labels = [], []
        for index, (script, ys) in enumerate(groups.items()):
            ax.axhspan(min(ys) - margin, max(ys) + margin,
                       color=palette[index % len(palette)], zorder=0)
            group_centers.append((min(ys) + max(ys)) / 2.0)
            group_labels.append(_to_code(script) if script else "")

        for x in np.arange(0.0, 1.0 + 1e-9, 0.1):
            ax.axvline(x, color="#cccccc", linewidth=0.5, linestyle="-", zorder=0.5)
        for x in np.arange(0.05, 1.0, 0.1):
            ax.axvline(x, color="#cccccc", linewidth=0.5, linestyle="--", zorder=0.5)

        for first, second in zip(panel_order, panel_order[1:]):
            ax.axhline((positions[first] + positions[second]) / 2.0,
                       color="black", linewidth=1.0, linestyle=(0, (3, 3)),
                       alpha=0.85, zorder=0.7)

        half_w = VIOLIN_WIDTH / 2
        for language in panel_order:
            y = positions[language]
            weighted = _values_by_type(documents, language, "weighted_nls", doc_type)
            correct  = _values_by_type(documents, language, "correct",      doc_type)
            _draw_paired_metric_violin(ax, weighted, correct, y, half_w)
            if not weighted and not correct:
                ax.text(0.5, y, "no scored documents", ha="center", va="center",
                        fontsize=8, style="italic", color="#666666", zorder=5)
            elif not weighted:
                ax.text(0.015, y - VIOLIN_OFFSET,
                        "no along-line quality (coverage = 0, shown as marker)",
                        ha="left", va="center", fontsize=7, style="italic",
                        color="#666666", zorder=5)

        major_ticks = np.arange(0.0, 1.0 + 1e-9, 0.1)
        pad_x = X_EDGE_PAD
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in major_ticks])
        ax.set_xlabel("Similarity / coverage  (0 = none, 1 = perfect)")
        ax.set_axisbelow(True)
        ax.set_xlim(0.0 - pad_x, 1.0 + pad_x)
        ax.set_ylim(y_high, y_low)
        ax.set_yticks([positions[lang] for lang in panel_order])
        ax.set_yticklabels(panel_order)

        _x_left = (0.0 - pad_x) + pad_x * 0.08
        for _lang in panel_order:
            _y = positions[_lang]
            ax.text(_x_left, _y - VIOLIN_OFFSET, "NLS", ha="left", va="center",
                    fontsize=7, style="italic", color="#999999", zorder=6)
            ax.text(_x_left, _y + VIOLIN_OFFSET, "Cov", ha="left", va="center",
                    fontsize=7, style="italic", color="#999999", zorder=6)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(major_ticks)
        ax_top.set_xticklabels([f"{t:.1f}" for t in major_ticks])
        ax_top.tick_params(axis="x", labelsize=9, length=4, direction="out")

        return group_centers, group_labels

    gc_print, gl_print = _draw_panel(ax_left,  order_print, pos_print, "print")
    gc_hw,    gl_hw    = _draw_panel(ax_right, order_hw,    pos_hw,    "handwriting")

    ax_left.set_title("Printed documents")
    ax_right.set_title("Handwritten documents")

    for _ax, _gc, _gl, _lpad in (
            (ax_left,  gc_print, gl_print, 8),
            (ax_right, gc_hw,    gl_hw,    8)):
        _right = _ax.twinx()
        _right.set_ylim(_ax.get_ylim())
        _right.set_yticks(_gc)
        _right.set_yticklabels(_gl)
        _right.tick_params(length=0)
        _right.set_ylabel("Script (ISO 15924)", rotation=270, labelpad=_lpad)

    from matplotlib.lines import Line2D
    nls_handles = [
        _TextIconProxy("NLS", label=WEIGHTED_LABEL),
        _HalfViolinProxy(sign=+1, fill_color=WEIGHTED_COLOR, label="NLS"),
        Line2D([0], [0], color="none", label=" "),            # was "NLS handwritten"
    ]
    cov_handles = [
        _TextIconProxy("Cov", label=CORRECT_LABEL),
        Line2D([0], [0], color="none", label=" "),            # was "Cov printed"
        _HalfViolinProxy(sign=-1, fill_color=CORRECT_COLOR, label="Cov"),
    ]
    stat_handles = [
        Line2D([0], [0], color="black", linewidth=0.8,
               label="Horizontal bar: data range (min → max)"),
        Line2D([0], [0], color="black", linewidth=0.8,
               label="Solid tall tick: median"),
        Line2D([0], [0], color="black", linewidth=0.8,
               label="Short solid ticks: min and max"),
    ]
    spacer_handles = [Line2D([0], [0], color="none", label=" ") for _ in range(2)]
    iso_handles = [
        Line2D([0], [0], color="none", label=lbl)
        for lbl in [
            "— ISO 15924 codes —",
            "Arab = Arabic",    "Beng = Bengali",    "Cyrl = Cyrillic",
            "Deva = Devanagari","Grek = Greek",       "Hani = Han",
            "Hebr = Hebrew",    "Jpan = Japanese",   "Khmr = Khmer",
            "Latn = Latin",
        ]
    ]
    # Center legend vertically between the panel content and the figure title.
    # Pass 1: draw to measure panel top and legend height.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    top_y_fig = max(
        ax.get_tightbbox(renderer).transformed(inv).y1
        for ax in fig.axes
        if ax.get_tightbbox(renderer) is not None
    )
    gap_fig = 120.0 / (fig.get_size_inches()[1] * FIGURE_DPI)  # 120 rendered px

    legend = fig.legend(
        handles=nls_handles + cov_handles + stat_handles + spacer_handles + iso_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.5),  # temporary position for height measurement
        bbox_transform=fig.transFigure,
        ncol=2, frameon=False, prop={"size": 8},
        handler_map={_HalfViolinProxy: HalfViolinLegendHandler(),
                     _TextIconProxy:   TextIconLegendHandler()})

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_h = legend.get_window_extent(renderer).transformed(inv).height

    # Pass 2: equal gap above and below legend — center it in the header zone.
    legend_bottom = top_y_fig + gap_fig
    legend_top    = legend_bottom + legend_h
    legend.set_bbox_to_anchor((0.5, legend_bottom), transform=fig.transFigure)

    fig.suptitle(
        "Per-language along-line transcription quality and reference coverage\n"
        "(scored documents, grouped by dominant script; languages and scripts alphabetical)",
        y=legend_top + gap_fig, va="bottom")
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
                               / "paired_violin_weighted_correct_split_by_doctype.png")
    outputs = ofd.load_outputs_json(args.outputs_json)
    partition = ofd.load_balanced_partition(args.results_dir)
    documents, _ = ofd.build_documents(outputs, partition)
    scripts = dominant_main_script_by_language(documents)
    all_languages = sorted({doc["language"] for doc in documents})
    order = sorted(all_languages, key=lambda language: (scripts.get(language) or "￿", language))

    written = build_figure(documents, order, scripts, out_path, band_style=args.band_style)
    two_panel_path = out_path.parent / "paired_violin_weighted_correct_two_panel.png"
    written2 = build_two_panel_figure(documents, order, scripts, two_panel_path,
                                      band_style=args.band_style)
    print("languages:", len(order), "| wrote:", written)
    print("two-panel  | wrote:", written2)


if __name__ == "__main__":
    main()
