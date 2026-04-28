"""Render alignment matrix visualisations and line overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def line_id_for_plot(line: dict, fallback_lid: int) -> int:
    """Resolve line id for plotting, falling back to list index."""
    return int(line.get("line_id", fallback_lid))


def line_label_for_plot(line: dict, fallback_lid: int) -> str:
    """Resolve line label text for plotting."""
    label = line.get("label")
    if label is not None:
        return str(label)
    return str(line_id_for_plot(line, fallback_lid))


def create_heatmap_figure(
    *,
    matrix: np.ndarray,
    title: str,
    figure_width: float,
    figure_height: float,
    cmap: str = "Greys",
):
    """Create a single heatmap figure for a score matrix."""
    fig, ax = plt.subplots(1, 1, figsize=(float(figure_width), float(figure_height)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, label="chrF")
    ax.set_xlabel("pred segment")
    ax.set_ylabel("ref segment")
    ax.set_title(title)
    return fig, ax


def draw_labeled_line(
    ax,
    *,
    line: dict,
    label: str,
    color: str,
    alpha: float,
    linewidth: float,
    zorder: float,
    show_label: bool,
    label_fontsize: float,
) -> None:
    """Draw one line segment and optional text label on an existing axis."""
    x0 = float(line["x0"])
    y0 = float(line["y0"])
    x1 = float(line["x1"])
    y1 = float(line["y1"])
    ax.plot((x0, x1), (y0, y1), color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

    if not show_label:
        return

    xm = float(line.get("label_x", (x0 + x1) / 2.0))
    ym = float(line.get("label_y", (y0 + y1) / 2.0))
    ax.text(
        xm,
        ym,
        str(label),
        color="yellow",
        fontsize=float(label_fontsize),
        weight="bold",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "black",
            "edgecolor": "yellow",
            "alpha": 0.75,
        },
        zorder=max(4.0, float(zorder) + 1.0),
    )


def raw_segments_to_labeled_lines(
    raw_segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[dict]:
    """Convert raw Hough segment tuples into labeled line dictionaries."""
    out: list[dict] = []
    for raw_line_id, (p0, p1) in enumerate(raw_segments):
        out.append(
            {
                "line_id": int(raw_line_id),
                "raw_line_id": int(raw_line_id),
                "label": str(raw_line_id),
                "x0": float(p0[0]),
                "y0": float(p0[1]),
                "x1": float(p1[0]),
                "y1": float(p1[1]),
            }
        )
    return out


def normalize_lines_for_labels(lines: list[dict]) -> list[dict]:
    """Ensure each line dict has stable ``line_id`` and ``label`` fields."""
    out: list[dict] = []
    for fallback_lid, line in enumerate(lines):
        normalized = dict(line)
        normalized.setdefault("line_id", int(fallback_lid))
        normalized.setdefault("label", str(line_id_for_plot(normalized, fallback_lid)))
        out.append(normalized)
    return out


def build_line_lookup(lines: list[dict]) -> dict[int, dict]:
    """Build ``line_id -> line`` lookup and validate uniqueness."""
    lookup: dict[int, dict] = {}
    for fallback_lid, line in enumerate(lines):
        lid = line_id_for_plot(line, fallback_lid)
        if lid in lookup:
            raise ValueError(f"Duplicate line id in report lines: {lid}")
        lookup[lid] = line
    return lookup


def save_matrix_visualisation(
    *,
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    lines: list[dict] | None = None,
    line_color: str = "red",
    line_width: float = 2.0,
    line_alpha: float = 1.0,
    show_labels: bool = False,
    label_fontsize: float = 8.0,
    figure_width: float = 8.0,
    figure_height: float = 5.0,
    dpi: int = 220,
) -> str:
    """Save one matrix visualisation image with optional line overlays."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = create_heatmap_figure(
        matrix=matrix,
        title=title,
        figure_width=float(figure_width),
        figure_height=float(figure_height),
        cmap="Greys",
    )

    for fallback_lid, line in enumerate(lines or []):
        draw_labeled_line(
            ax,
            line=line,
            label=line_label_for_plot(line, fallback_lid),
            color=line_color,
            alpha=float(line_alpha),
            linewidth=float(line_width),
            zorder=3.0,
            show_label=bool(show_labels),
            label_fontsize=float(label_fontsize),
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_path)
