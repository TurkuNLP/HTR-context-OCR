#!/usr/bin/env python3
"""Plot small (<10×10) documents: score matrix heatmap with a fitted alignment diagonal.

The bold blue diagonal is fitted to the Levenshtein alignment (see small_document_alignment) and is
a VISUAL guide only — no metric is derived from it. The faint stair path underneath is the raw
character-level alignment, drawn so you can see how the straight fit departs from it. All printed
metrics come from the small-document CSV.
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from rapidfuzz.distance import Levenshtein

# Allow `import small_document_alignment` regardless of the working directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from small_document_alignment import (  # noqa: E402
    fit_alignment_line,
    load_score_pkl_records,
)

_TAG_STYLE: dict[str, dict] = {
    "equal":   {"color": "#2f9e44", "linewidth": 2.0, "linestyle": "-"},
    "replace": {"color": "#e8590c", "linewidth": 2.0, "linestyle": "-"},
    "delete":  {"color": "#c92a2a", "linewidth": 2.0, "linestyle": "-"},
    "insert":  {"color": "#868e96", "linewidth": 1.5, "linestyle": "--"},
}
_DIAGONAL_COLOR = "#1971C2"  # bold fitted line (matches the pipeline's surviving-line colour)


# ── Opcode → window segments ───────────────────────────────────────────────────

def opcodes_to_window_segments(
    ops: list[tuple],
    stride: int,
    n_rows: int,
    n_cols: int,
) -> list[dict]:
    """Project Levenshtein opcodes onto score-matrix window coordinates."""
    def clamp_row(c: int) -> int:
        return min(c // stride, n_rows - 1)

    def clamp_col(c: int) -> int:
        return min(c // stride, n_cols - 1)

    segments = []
    for tag, s, e, dst_s, dst_e in ops:
        row_start = clamp_row(s)
        row_end   = clamp_row(max(s, e - 1))
        col_start = clamp_col(dst_s)
        col_end   = clamp_col(max(dst_s, dst_e - 1))
        segments.append({
            "tag":       tag,
            "row_start": row_start,
            "row_end":   row_end,
            "col_start": col_start,
            "col_end":   col_end,
        })
    return segments


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_score_matrix_heatmap(axis: Any, matrix: np.ndarray | None, title: str) -> Any | None:
    """Draw a score matrix heatmap on the given axis (0–100 viridis scale)."""
    if matrix is None:
        axis.text(0.5, 0.5, "Matrix missing", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        return None
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.size == 0:
        axis.text(0.5, 0.5, f"Empty matrix\nshape={m.shape}", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        return None
    image = axis.imshow(m, origin="upper", aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    axis.set_title(title)
    axis.set_xlabel("Prediction window index")
    axis.set_ylabel("Reference window index")
    axis.set_xlim(-0.5, m.shape[1] - 0.5)
    axis.set_ylim(m.shape[0] - 0.5, -0.5)
    return image


def draw_levenshtein_path(axis: Any, segments: list[dict], *, faint: bool = False) -> None:
    """Overlay the character-level alignment path on a score-matrix axis.

    With ``faint=True`` it is drawn as a thin, semi-transparent, unlabelled underlay so the bold
    fitted diagonal stays the focus; otherwise it is the bold, legended path (original behaviour).
    """
    alpha = 0.30 if faint else 1.0
    width_scale = 0.6 if faint else 1.0
    drawn_tags: set[str] = set()
    for seg in segments:
        tag = seg["tag"]
        style = _TAG_STYLE.get(tag, _TAG_STYLE["replace"])
        # Suppress the per-tag legend when faint: the diagonal owns the legend in that mode.
        label = None if faint else (tag if tag not in drawn_tags else None)
        drawn_tags.add(tag)
        axis.plot(
            [seg["col_start"], seg["col_end"]],
            [seg["row_start"], seg["row_end"]],
            color=style["color"],
            linewidth=style["linewidth"] * width_scale,
            linestyle=style["linestyle"],
            marker="" if faint else "o",
            markersize=4,
            alpha=alpha,
            label=label,
            zorder=2 if faint else 3,
        )


def draw_fitted_diagonal(axis: Any, line: tuple | None) -> None:
    """Draw the PCA-fitted alignment diagonal as a visual guide only (no metric attached)."""
    if line is None:
        return
    x0, y0, x1, y1 = line
    axis.plot(
        [x0, x1], [y0, y1],
        color=_DIAGONAL_COLOR, linewidth=2.6, marker="o", markersize=5,
        label="fitted diagonal (visual)", zorder=5,
    )


# ── Panel rendering ────────────────────────────────────────────────────────────

def _to_float(value: Any) -> float:
    """Parse a CSV cell to float; empty / non-numeric (e.g. an undefined metric) becomes nan."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def render_panel(
    *,
    matrix: np.ndarray,
    segments: list[dict],
    line: tuple | None,
    csv_row: dict,
    window_size: int,
    window_stride: int,
    output_path: Path,
    figure_dpi: int,
) -> None:
    """Render one document panel and save it as a PNG."""
    fname        = csv_row["fname"]
    language     = csv_row["main_language"]
    doc_type     = csv_row["document_type"]
    nls          = _to_float(csv_row["document_normalised_levenshtein"])
    correct      = _to_float(csv_row["correct_ref_coverage"])
    missing      = _to_float(csv_row["missing_ref_coverage"])
    hallucination = _to_float(csv_row.get("hallucination"))
    repetition   = _to_float(csv_row["repetition_on_reference"])
    ref_len      = csv_row["reference_text_length"]
    pred_len     = csv_row["prediction_text_length"]

    fig = plt.figure(figsize=(8, 7), constrained_layout=False)
    gs  = fig.add_gridspec(2, 1, height_ratios=[4.0, 0.8], hspace=0.20)
    ax_matrix  = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])

    fig.suptitle(
        f"{language} / {doc_type} | {fname}",
        fontsize=12,
        y=0.995,
    )

    title = (
        f"ref_to_pred score matrix  ({matrix.shape[0]}×{matrix.shape[1]} windows, "
        f"ws={window_size} st={window_stride})  +  fitted alignment diagonal (visual)"
    )
    image = draw_score_matrix_heatmap(ax_matrix, matrix, title)
    draw_levenshtein_path(ax_matrix, segments, faint=True)   # faint character-alignment underlay
    draw_fitted_diagonal(ax_matrix, line)                    # visual-only diagonal
    if image is not None:
        fig.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.04, label="Score (0–100)")
    if line is not None:
        ax_matrix.legend(loc="lower right", fontsize=8, framealpha=0.7)

    metrics_text = (
        f"fname={fname}  lang={language}  type={doc_type}\n"
        f"ref_len={ref_len}  pred_len={pred_len}  "
        f"matrix={matrix.shape[0]}×{matrix.shape[1]}  ws={window_size}  stride={window_stride}\n"
        f"docNLS={nls:.4f}  hallucination={hallucination:.4f}\n"
        f"correct_ref_coverage={correct:.4f}  "
        f"missing_ref_coverage={missing:.4f}  repetition_on_reference={repetition:.4f}"
    )
    ax_metrics.set_axis_off()
    ax_metrics.text(
        0.01, 0.98,
        metrics_text,
        transform=ax_metrics.transAxes,
        ha="left", va="top",
        fontsize=8,
        family="monospace",
        linespacing=1.4,
    )

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.04, top=0.93)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small-document-scores-csv", required=True,
                    help="CSV written by score_small_documents.py")
    ap.add_argument("--scores-pkl-ref-to-pred", required=True,
                    help="ref→pred score pickle (matrix + texts + geometry per record)")
    ap.add_argument("--output-dir", required=True,
                    help="root output directory; PNGs written to {output-dir}/{language}/{fname}.png")
    ap.add_argument("--figure-dpi", type=int, default=150)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)

    # Step 1: read the CSV
    with open(args.small_document_scores_csv) as f:
        csv_rows = list(csv.DictReader(f))
    fnames_to_plot = {row["fname"] for row in csv_rows}
    print(f"[csv] {len(csv_rows)} documents to plot")

    # Step 2: load PKL records for the relevant fnames only
    print(f"[pkl] scanning {args.scores_pkl_ref_to_pred}")
    pkl_records = load_score_pkl_records(Path(args.scores_pkl_ref_to_pred))
    print(f"[pkl] {len(pkl_records)} total records; "
          f"{sum(1 for f in fnames_to_plot if f in pkl_records)} match CSV")

    # Step 3: render one panel per document
    skipped = []
    for i, row in enumerate(csv_rows):
        fname = row["fname"]
        if fname not in pkl_records:
            skipped.append(fname)
            continue

        rec = pkl_records[fname]
        matrix     = np.asarray(rec["scores"], dtype=float)
        ref_text   = str(rec.get("ref", ""))
        pred_text  = str(rec.get("pred", ""))
        window_size   = int(rec.get("window_size",   50))
        window_stride = int(rec.get("window_stride", 35))
        n_rows, n_cols = matrix.shape

        ops      = Levenshtein.opcodes(ref_text, pred_text)
        segments = opcodes_to_window_segments(ops, window_stride, n_rows, n_cols)
        line     = fit_alignment_line(ops, window_stride, n_rows, n_cols)

        stem       = Path(fname).stem
        out_path   = output_dir / row["main_language"] / f"{stem}.png"

        render_panel(
            matrix=matrix,
            segments=segments,
            line=line,
            csv_row=row,
            window_size=window_size,
            window_stride=window_stride,
            output_path=out_path,
            figure_dpi=args.figure_dpi,
        )

        if (i + 1) % 20 == 0 or (i + 1) == len(csv_rows):
            print(f"[progress] {i + 1}/{len(csv_rows)}")

    print(f"[done] {len(csv_rows) - len(skipped)} PNGs written to {output_dir}")
    if skipped:
        print(f"[skip] {len(skipped)} had no PKL record: {skipped[:5]}")


if __name__ == "__main__":
    main()
