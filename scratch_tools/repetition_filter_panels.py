#!/usr/bin/env python3
from __future__ import annotations

"""Render per-document diagnostic panels for documents whose repetition_on_reference
exceeds a threshold and stitch them into per-language contact sheets.

Reads the balanced-run CSV for document metadata + metrics and the per-document
alpha_sweep pickles for the score matrices and detected Hough lines.
No alpha sweep is re-run: only pre-computed results are visualised.

Panel layout (2×2 GridSpec):
  [0,0]  ref_to_pred score matrix — clean heatmap
  [0,1]  ref_to_ref score matrix  — clean heatmap
  [1,0]  ref_to_pred + final surviving Hough lines (blue hollow boxes)
  [1,1]  All six metrics as monospace text

Usage:
  python3 scratch_tools/repetition_filter_panels.py \\
    --results-dir results/.../balanced \\
    --min-repetition 0.0 \\
    --panel-columns 3 \\
    --world-readable
"""

import argparse
import csv
import math
import os
import pickle
import stat
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

# The alpha-sweep pickles contain objects from tuner_simple_alpha_sweep_pre_iou_levenshtein.
# We only need the module on sys.path to deserialise them — no other imports are taken from it.
_CHURRO_COPY = Path(__file__).resolve().parent.parent
if str(_CHURRO_COPY) not in sys.path:
    sys.path.insert(0, str(_CHURRO_COPY))


def _raise_csv_field_size_limit() -> None:
    """Allow very large CSV fields (full document text / serialised metrics).

    Progress/result rows can hold fields well past Python's default 131072-byte
    limit, so raise it to the largest value this platform's C long accepts.
    """

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit // 10)


_raise_csv_field_size_limit()

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINAL_LINE_COLOR = "#1971C2"
FINAL_LINE_BOX_HALF_WIDTH_CELLS = 2.0

BORDER_PIXELS = 8
GAP_PIXELS = 10
BORDER_COLOR = (30, 30, 30, 255)
BACKGROUND_COLOR = (255, 255, 255, 255)


# ---------------------------------------------------------------------------
# Drawing helpers (inlined from document_panel_renderer.py)
# ---------------------------------------------------------------------------

def safe_path_component(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(value))
    return cleaned.strip("._") or "unknown"


def draw_score_matrix_heatmap(axis: Any, score_matrix: Any, title: str) -> Any:
    if score_matrix is None:
        axis.text(0.5, 0.5, "Matrix missing", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        return None
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        axis.text(0.5, 0.5, f"Empty matrix\nshape={matrix.shape}", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        return None
    image = axis.imshow(matrix, origin="upper", aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    axis.set_title(title)
    axis.set_xlabel("Prediction/self window index")
    axis.set_ylabel("Reference window index")
    axis.set_xlim(-0.5, matrix.shape[1] - 0.5)
    axis.set_ylim(matrix.shape[0] - 0.5, -0.5)
    return image


def _segment_unit_vectors(x0, y0, x1, y1):
    dx, dy = float(x1) - float(x0), float(y1) - float(y0)
    length = math.hypot(dx, dy)
    if length <= 0.0 or not math.isfinite(length):
        return (1.0, 0.0), (0.0, 1.0)
    return (dx / length, dy / length), (-dy / length, dx / length)


def _oriented_box_points(x0, y0, x1, y1, half_width, end_padding=0.75):
    (dx, dy), (nx, ny) = _segment_unit_vectors(x0, y0, x1, y1)
    hw = max(0.5, float(half_width))
    ep = max(0.0, float(end_padding))
    sx, sy = float(x0) - dx * ep, float(y0) - dy * ep
    ex, ey = float(x1) + dx * ep, float(y1) + dy * ep
    return [
        (sx + nx * hw, sy + ny * hw),
        (ex + nx * hw, ey + ny * hw),
        (ex - nx * hw, ey - ny * hw),
        (sx - nx * hw, sy - ny * hw),
    ]


def _draw_segment_box(axis, *, x0, y0, x1, y1, color, label, linewidth, alpha, linestyle, half_width):
    from matplotlib.patches import Polygon
    pts = _oriented_box_points(x0, y0, x1, y1, half_width)
    axis.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=color, linewidth=linewidth,
                           alpha=alpha, linestyle=linestyle, label=label, joinstyle="miter", zorder=6))


def _endpoint_from_line_record(line_record: Any):
    if not isinstance(line_record, dict):
        return None
    if any(line_record.get(k) is None for k in ("x0", "y0", "x1", "y1")):
        return None
    return float(line_record["x0"]), float(line_record["y0"]), float(line_record["x1"]), float(line_record["y1"])


def draw_final_line_overlay(axis: Any, final_lines: Sequence[Any]) -> int:
    drawn = 0
    for line_record in (final_lines or []):
        ep = _endpoint_from_line_record(line_record)
        if ep is None:
            continue
        x0, y0, x1, y1 = ep
        _draw_segment_box(
            axis, x0=x0, y0=y0, x1=x1, y1=y1,
            color=FINAL_LINE_COLOR,
            label="Surviving line" if drawn == 0 else None,
            linewidth=2.0, alpha=0.95, linestyle="-",
            half_width=FINAL_LINE_BOX_HALF_WIDTH_CELLS,
        )
        drawn += 1
    return drawn


# ---------------------------------------------------------------------------
# Metrics text builder
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def build_metrics_text(result_row: dict, min_repetition: float) -> str:
    rep = result_row.get("repetition_on_reference")
    rep_str = _fmt(rep)
    try:
        rep_flag = "  <<<" if rep is not None and float(rep) > min_repetition else ""
    except (TypeError, ValueError):
        rep_flag = ""

    lines = [
        f"document_normalised_levenshtein:             {_fmt(result_row.get('document_normalised_levenshtein'))}",
        f"weighted_along_lines_normalised_levenshtein: {_fmt(result_row.get('weighted_along_lines_normalised_levenshtein'))}",
        f"correct_ref_coverage:                        {_fmt(result_row.get('correct_ref_coverage'))}",
        f"missing_ref_coverage:                        {_fmt(result_row.get('missing_ref_coverage'))}",
        f"repetition_on_reference:                     {rep_str}{rep_flag}",
        f"hallucination:                               {_fmt(result_row.get('hallucination'))}",
        "",
        f"alpha={result_row.get('score_floor_alpha')}  |  used_lines={result_row.get('used_line_count')}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-document panel renderer
# ---------------------------------------------------------------------------

def render_panel(
    *,
    ref_to_pred_matrix: Any,
    ref_to_ref_matrix: Any,
    final_lines: list,
    result_row: dict,
    language: str,
    fname: str,
    document_type: str,
    output_path: Path,
    dpi: int,
    min_repetition: float,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 15), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.18)
    ax_rp_clean = fig.add_subplot(gs[0, 0])
    ax_rr_clean = fig.add_subplot(gs[0, 1])
    ax_rp_lines = fig.add_subplot(gs[1, 0])
    ax_metrics  = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"{language} / {document_type}  |  {fname}",
        fontsize=12, y=0.995,
    )

    img0 = draw_score_matrix_heatmap(ax_rp_clean, ref_to_pred_matrix, "ref_to_pred score matrix")
    img1 = draw_score_matrix_heatmap(ax_rr_clean, ref_to_ref_matrix,  "ref_to_ref score matrix")
    img2 = draw_score_matrix_heatmap(ax_rp_lines, ref_to_pred_matrix, "ref_to_pred + final lines")
    n = draw_final_line_overlay(ax_rp_lines, final_lines)
    if n > 0:
        ax_rp_lines.legend(loc="upper right", fontsize=8)

    for ax, img in [(ax_rp_clean, img0), (ax_rr_clean, img1), (ax_rp_lines, img2)]:
        if img is not None:
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Score")

    ax_metrics.set_axis_off()
    ax_metrics.text(
        0.03, 0.97,
        build_metrics_text(result_row, min_repetition),
        transform=ax_metrics.transAxes,
        ha="left", va="top",
        fontsize=10, family="monospace", linespacing=1.6,
    )

    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.04, top=0.95, hspace=0.34, wspace=0.22)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Stitching helpers (inlined from stitch_language_panels.py)
# ---------------------------------------------------------------------------

def _build_bordered_tile(panel_image, max_width: int, max_height: int):
    tile_w = max_width + 2 * BORDER_PIXELS
    tile_h = max_height + 2 * BORDER_PIXELS
    tile = Image.new("RGBA", (tile_w, tile_h), BORDER_COLOR)
    bg   = Image.new("RGBA", (max_width, max_height), BACKGROUND_COLOR)
    tile.alpha_composite(bg, (BORDER_PIXELS, BORDER_PIXELS))
    px = BORDER_PIXELS + max(0, (max_width  - panel_image.width)  // 2)
    py = BORDER_PIXELS + max(0, (max_height - panel_image.height) // 2)
    tile.alpha_composite(panel_image, (px, py))
    bg.close()
    return tile


def save_stitched_language_image(*, panel_paths: list[Path], stitched_output_path: Path, panel_columns: int) -> Path | None:
    if not panel_paths:
        return None
    opened = []
    try:
        for p in panel_paths:
            opened.append(Image.open(p).convert("RGBA"))
        max_w = max(im.width  for im in opened)
        max_h = max(im.height for im in opened)
        cols = max(1, int(panel_columns))
        rows = math.ceil(len(opened) / cols)
        tw = max_w + 2 * BORDER_PIXELS
        th = max_h + 2 * BORDER_PIXELS
        stitched_w = cols * tw + max(0, cols - 1) * GAP_PIXELS
        stitched_h = rows * th + max(0, rows - 1) * GAP_PIXELS
        canvas = Image.new("RGBA", (stitched_w, stitched_h), BACKGROUND_COLOR)
        try:
            for idx, img in enumerate(opened):
                col_i = idx % cols
                row_i = idx // cols
                x = col_i * (tw + GAP_PIXELS)
                y = row_i * (th + GAP_PIXELS)
                tile = _build_bordered_tile(img, max_w, max_h)
                try:
                    canvas.alpha_composite(tile, (x, y))
                finally:
                    tile.close()
            stitched_output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(stitched_output_path, optimize=True)
        finally:
            canvas.close()
    finally:
        for im in opened:
            im.close()
    return stitched_output_path


# ---------------------------------------------------------------------------
# World-readable helper
# ---------------------------------------------------------------------------

def make_world_readable(path: Path) -> None:
    current = stat.S_IMODE(path.stat().st_mode)
    extra = stat.S_IROTH | stat.S_IXOTH if path.is_dir() else stat.S_IROTH
    path.chmod(current | extra)


def apply_world_readable(output_dir: Path, stitched_paths: list[Path]) -> None:
    for p in stitched_paths:
        try:
            make_world_readable(p)
        except PermissionError:
            pass
    seen: set[Path] = set()
    for directory in [output_dir, *output_dir.parents]:
        if directory in seen:
            continue
        seen.add(directory)
        try:
            make_world_readable(directory)
        except PermissionError:
            pass
        if directory.name == "results":
            break


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def read_csv_filtered(csv_path: Path, min_repetition: float) -> list[dict]:
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    kept = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get("repetition_on_reference", "")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val > min_repetition:
                kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render repetition-filtered diagnostic panels from alpha-sweep balanced results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="The balanced/ directory containing best_combination_per_document.csv")
    parser.add_argument("--min-repetition", type=float, default=0.0,
                        help="Include documents where repetition_on_reference > this value")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for panels and stitched PNGs "
                             "(default: <results-dir>/plots/repetition_filter_min<N>/)")
    parser.add_argument("--panel-columns", type=int, default=3,
                        help="Columns in the stitched contact sheet")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Figure DPI for saved panels")
    parser.add_argument("--world-readable", action="store_true",
                        help="chmod output files o+r and directories o+x")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"[error] --results-dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    csv_path   = results_dir / "best_combination_per_document.csv"
    pickle_dir = results_dir / "balanced" / "alpha_sweep_pickles"

    min_rep_str = str(args.min_repetition).replace(".", "_")
    output_dir = args.output_dir or (results_dir / "plots" / f"repetition_filter_min{min_rep_str}")
    output_dir = output_dir.resolve()
    panel_dir  = output_dir / ".temporary_document_panels"

    rows = read_csv_filtered(csv_path, args.min_repetition)
    print(f"[filter] {len(rows)} document(s) with repetition_on_reference > {args.min_repetition}")
    if not rows:
        print("[done] nothing to render")
        return

    panel_paths_by_language: dict[str, list[Path]] = defaultdict(list)
    for i, row in enumerate(rows, 1):
        language = row.get("main_language", "UNKNOWN")
        fname    = row.get("fname", "")
        if not fname:
            print(f"[warn] row {i}: missing fname, skipped", file=sys.stderr)
            continue

        pkl_path = pickle_dir / language / (fname + ".pkl")
        if not pkl_path.exists():
            print(f"[warn] pickle not found: {pkl_path}", file=sys.stderr)
            continue

        with open(pkl_path, "rb") as fh:
            doc_pickle = pickle.load(fh)

        payload = doc_pickle.get("selected_plot_payload")
        if payload is None:
            print(f"[warn] no selected_plot_payload in {pkl_path}", file=sys.stderr)
            continue

        result_row = dict(payload.get("result_row") or {})
        for key in (
            "repetition_on_reference", "hallucination", "correct_ref_coverage",
            "missing_ref_coverage", "document_normalised_levenshtein",
            "weighted_along_lines_normalised_levenshtein",
        ):
            if key in row:
                result_row[key] = row[key]

        panel_path = panel_dir / safe_path_component(language) / f"{safe_path_component(fname)}.png"
        print(f"[{i}/{len(rows)}] rendering {fname} ({language})", flush=True)
        render_panel(
            ref_to_pred_matrix=payload.get("ref_to_pred_score_matrix"),
            ref_to_ref_matrix=payload.get("ref_to_ref_score_matrix"),
            final_lines=payload.get("final_surviving_ref_to_pred_lines") or [],
            result_row=result_row,
            language=language,
            fname=fname,
            document_type=row.get("document_type", ""),
            output_path=panel_path,
            dpi=args.dpi,
            min_repetition=args.min_repetition,
        )
        panel_paths_by_language[language].append(panel_path)

    stitched_paths: list[Path] = []
    for language in sorted(panel_paths_by_language):
        panels = panel_paths_by_language[language]
        out_png = output_dir / f"repetition_filter_min{min_rep_str}_{language}_documents.png"
        saved = save_stitched_language_image(
            panel_paths=panels,
            stitched_output_path=out_png,
            panel_columns=args.panel_columns,
        )
        if saved is not None:
            stitched_paths.append(saved)
            print(f"[stitch] {language}: {len(panels)} panel(s) → {saved.name}")

    if args.world_readable:
        apply_world_readable(output_dir, stitched_paths)
        print("[chmod] world-readable permissions applied")

    print(f"[done] {len(stitched_paths)} stitched sheet(s) → {output_dir}")


if __name__ == "__main__":
    main()
