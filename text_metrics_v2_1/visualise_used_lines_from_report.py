import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from runfile_records import load_run_items, same_file, safe_name
from score_matrix_builder import compute_score_matrix

__all__ = [
    "build_visualisation",
    "raw_segments_to_labeled_lines",
    "save_matrix_visualisation",
    "save_text_metrics_visualisations",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Visualize the kept alignment lines saved in a report JSON on top of the full chrF heatmap. "
            "The script is designed for reports produced by levenshtein_along_lines_metric.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--report-json", type=Path, required=True, help="Path to *_report.json")
    p.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to summary.json. Defaults to report_json.parent / 'summary.json'.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <report dir>/visualise_used_lines_only.",
    )
    p.add_argument("--window-size", type=int, default=None, help="Override window size from summary.json")
    p.add_argument("--window-stride", type=int, default=None, help="Override window stride from summary.json")
    p.add_argument("--figure-width", type=float, default=10.0, help="Figure width in inches")
    p.add_argument("--figure-height", type=float, default=6.5, help="Figure height in inches")
    p.add_argument("--dpi", type=int, default=220, help="PNG DPI")
    p.add_argument("--line-width", type=float, default=2.0, help="Plotted line width")
    p.add_argument("--label-fontsize", type=float, default=8.0, help="Line id label font size")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument(
        "--only-lines",
        type=str,
        default=None,
        help="Comma-separated line ids or ranges to render, for example '5' or '5,7-9'.",
    )
    selection.add_argument(
        "--highlight-lines",
        type=str,
        default=None,
        help="Comma-separated line ids or ranges to highlight in red while optionally keeping others in gray.",
    )
    p.add_argument(
        "--hide-other-lines",
        action="store_true",
        help="With --highlight-lines, hide non-selected lines instead of drawing them in gray.",
    )
    p.add_argument(
        "--label-selected-only",
        action="store_true",
        help="Only label the selected or highlighted lines.",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_summary_path(args) -> Path:
    if args.summary_json is not None:
        return args.summary_json
    return args.report_json.parent / "summary.json"


def _resolve_window_params(args, report: dict, summary: dict) -> tuple[int, int]:
    window_size = args.window_size
    window_stride = args.window_stride

    if window_size is None:
        window_size = report.get("window_size", summary.get("window_size"))
    if window_stride is None:
        window_stride = report.get("window_stride", summary.get("window_stride"))

    if window_size is None or window_stride is None:
        raise ValueError(
            "Could not determine window_size/window_stride. Provide --window-size and --window-stride, "
            "or a summary.json containing them."
        )

    window_size = int(window_size)
    window_stride = int(window_stride)
    if window_size <= 0 or window_stride <= 0:
        raise ValueError("window-size and window-stride must be positive")
    return window_size, window_stride


def _resolve_output_dir(args) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.report_json.parent / "visualise_used_lines_only"


def _extract_lines(report: dict) -> list[dict]:
    lines = report.get("lines")
    if lines is None:
        lines = report.get("lines_used")
    if lines is None:
        raise KeyError("Report JSON does not contain 'lines' or 'lines_used'")
    if not isinstance(lines, list):
        raise ValueError(f"Expected list of lines in report, got {type(lines).__name__}")
    return lines


def _line_id_for_plot(line: dict, fallback_lid: int) -> int:
    return int(line.get("line_id", fallback_lid))


def _line_label_for_plot(line: dict, fallback_lid: int) -> str:
    label = line.get("label")
    if label is not None:
        return str(label)
    return str(_line_id_for_plot(line, fallback_lid))


def _build_line_lookup(lines: list[dict]) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for fallback_lid, line in enumerate(lines):
        line_id = _line_id_for_plot(line, fallback_lid)
        if line_id in lookup:
            raise ValueError(f"Duplicate line id in report lines: {line_id}")
        lookup[line_id] = line
    return lookup


def _parse_line_id_spec(spec: str, *, valid_line_ids: set[int]) -> list[int]:
    if spec is None:
        return []

    selected: set[int] = set()
    for raw_part in str(spec).split(','):
        part = raw_part.strip()
        if not part:
            continue
        if '-' in part:
            left, right = part.split('-', 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"Invalid descending range: {part!r}")
            for value in range(start, end + 1):
                selected.add(int(value))
        else:
            selected.add(int(part))

    invalid = sorted(value for value in selected if value not in valid_line_ids)
    if invalid:
        valid_str = ', '.join(str(value) for value in sorted(valid_line_ids))
        raise ValueError(f"Unknown line ids {invalid}; available line ids are: {valid_str}")
    return sorted(selected)


def _resolve_selection(args, lines: list[dict]) -> tuple[list[int], list[int]]:
    line_lookup = _build_line_lookup(lines)
    all_line_ids = sorted(line_lookup)
    valid_ids = set(all_line_ids)

    if args.only_lines is not None:
        selected = _parse_line_id_spec(args.only_lines, valid_line_ids=valid_ids)
        return selected, selected

    if args.highlight_lines is not None:
        selected = _parse_line_id_spec(args.highlight_lines, valid_line_ids=valid_ids)
        visible = selected if args.hide_other_lines else all_line_ids
        return visible, selected

    return all_line_ids, all_line_ids


def _selection_suffix(args, *, visible_ids: list[int], highlighted_ids: list[int]) -> str:
    if args.only_lines is not None:
        return "_only_lines_" + "_".join(str(value) for value in visible_ids)
    if args.highlight_lines is not None:
        base = "_highlight_lines_" + "_".join(str(value) for value in highlighted_ids)
        if args.hide_other_lines:
            base += "_only"
        return base
    return "_used_lines_labeled_full"


def _should_label_line(args, line_id: int, highlighted_ids: set[int]) -> bool:
    if args.label_selected_only:
        return line_id in highlighted_ids
    return True


def _create_heatmap_figure(
    *,
    matrix: np.ndarray,
    title: str,
    figure_width: float,
    figure_height: float,
    cmap: str = "Greys",
):
    fig, ax = plt.subplots(1, 1, figsize=(float(figure_width), float(figure_height)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, label="chrF")
    ax.set_xlabel("pred segment")
    ax.set_ylabel("ref segment")
    ax.set_title(title)
    return fig, ax


def _draw_labeled_line(
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


def _normalize_lines_for_labels(lines: list[dict]) -> list[dict]:
    out: list[dict] = []
    for fallback_lid, line in enumerate(lines):
        normalized = dict(line)
        normalized.setdefault("line_id", int(fallback_lid))
        normalized.setdefault("label", str(_line_id_for_plot(normalized, fallback_lid)))
        out.append(normalized)
    return out


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _create_heatmap_figure(
        matrix=matrix,
        title=title,
        figure_width=figure_width,
        figure_height=figure_height,
        cmap="Greys",
    )

    for fallback_lid, line in enumerate(lines or []):
        _draw_labeled_line(
            ax,
            line=line,
            label=_line_label_for_plot(line, fallback_lid),
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


def save_text_metrics_visualisations(
    *,
    matrix_before: np.ndarray,
    raw_hough_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    pre_filter_lines: list[dict],
    filtered_lines: list[dict],
    matrix_after_reordering: np.ndarray | None,
    case_prefix: str,
    file_name: str,
    output_dir: Path,
    threshold_start: float | None = None,
    line_filter_label: str = "v2.1_true_IoU",
) -> dict:
    output_dir = Path(output_dir)
    vis_before_hough_dir = output_dir / "visualise_before_hough"
    vis_after_hough_dir = output_dir / "visualise_after_hough_line_transform"
    vis_after_filtering_dir = output_dir / "visualise_after_filtering"
    vis_after_reordering_dir = output_dir / "visualise_after_reordering"

    raw_hough_lines = raw_segments_to_labeled_lines(list(raw_hough_segments))
    normalized_filtered_lines = _normalize_lines_for_labels(list(filtered_lines))

    before_path = save_matrix_visualisation(
        matrix=matrix_before,
        title=f"{file_name} | before Hough",
        out_path=vis_before_hough_dir / f"{case_prefix}_before_hough.png",
    )

    hough_suffix = ""
    if threshold_start is not None and np.isfinite(float(threshold_start)):
        hough_suffix = f" | start={float(threshold_start):.2f}"
    after_hough_path = save_matrix_visualisation(
        matrix=matrix_before,
        title=(
            f"{file_name} | after Hough line transform{hough_suffix} "
            f"| raw={len(raw_hough_lines)}"
        ),
        out_path=vis_after_hough_dir / f"{case_prefix}_after_hough_line_transform.png",
        lines=raw_hough_lines,
        line_color="red",
        line_width=2.0,
        line_alpha=1.0,
        show_labels=True,
        label_fontsize=8.0,
    )

    after_filtering_path = save_matrix_visualisation(
        matrix=matrix_before,
        title=(
            f"{file_name} | after filtering ({line_filter_label}) "
            f"| kept={len(normalized_filtered_lines)}/{len(pre_filter_lines)}"
        ),
        out_path=vis_after_filtering_dir / f"{case_prefix}_after_filtering.png",
        lines=normalized_filtered_lines,
        line_color="limegreen",
        line_width=2.4,
        line_alpha=1.0,
        show_labels=True,
        label_fontsize=8.0,
    )

    after_reordering_path = None
    if matrix_after_reordering is not None:
        after_reordering_path = save_matrix_visualisation(
            matrix=matrix_after_reordering,
            title=f"{file_name} | after reordering",
            out_path=vis_after_reordering_dir / f"{case_prefix}_after_reordering.png",
        )

    return {
        "visualise_before_hough_path": before_path,
        "visualise_after_hough_line_transform_path": after_hough_path,
        "visualise_after_filtering_path": after_filtering_path,
        "visualise_after_reordering_path": after_reordering_path,
        "visualise_raw_hough_path": after_hough_path,
        "visualise_after_v2_1_true_iou_path": after_filtering_path,
        "visualise_after_reorder_path": after_reordering_path,
        "visualise_full_path": after_hough_path,
        "visualise_graph_path": after_hough_path,
        "visualise_mask_path": None,
    }


def build_visualisation(*, report: dict, matrix: np.ndarray, output_dir: Path, args) -> Path:
    lines = _extract_lines(report)
    line_lookup = _build_line_lookup(lines)
    visible_ids, highlighted_ids = _resolve_selection(args, lines)
    highlighted_id_set = set(highlighted_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_prefix = f"{int(report['index']):04d}_{safe_name(report['fname'])}"
    out_path = output_dir / f"{case_prefix}{_selection_suffix(args, visible_ids=visible_ids, highlighted_ids=highlighted_ids)}.png"

    fig, ax = _create_heatmap_figure(
        matrix=matrix,
        title=f"{report['fname']} | visible lines={len(visible_ids)}",
        figure_width=float(args.figure_width),
        figure_height=float(args.figure_height),
        cmap="Greys",
    )

    for line_id in visible_ids:
        line = line_lookup[line_id]
        is_highlighted = line_id in highlighted_id_set

        if is_highlighted:
            color = "red"
            alpha = 1.0
            linewidth = float(args.line_width)
            zorder = 3.0
        else:
            color = "#6b7280"
            alpha = 0.65
            linewidth = max(1.0, float(args.line_width) * 0.8)
            zorder = 2.0

        _draw_labeled_line(
            ax,
            line=line,
            label=str(line_id),
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
            show_label=_should_label_line(args, line_id, highlighted_id_set),
            label_fontsize=float(args.label_fontsize),
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    if not args.report_json.exists():
        raise FileNotFoundError(f"Missing report JSON: {args.report_json}")
    if not args.runfile_json.exists():
        raise FileNotFoundError(f"Missing runfile JSON: {args.runfile_json}")
    if args.hide_other_lines and args.highlight_lines is None:
        raise ValueError("--hide-other-lines requires --highlight-lines")

    summary_path = _resolve_summary_path(args)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary JSON: {summary_path}. Pass --summary-json or override --window-size/--window-stride."
        )

    report = _load_json(args.report_json)
    summary = _load_json(summary_path)
    window_size, window_stride = _resolve_window_params(args, report, summary)

    items = load_run_items(args.runfile_json)
    target = next((item for item in items if same_file(item["fname"], report["fname"])), None)
    if target is None:
        raise KeyError(f"Target file not found in provided input items: {report['fname']!r}")

    matrix = compute_score_matrix(
        target["ref"],
        target["pred"],
        window_size=window_size,
        window_stride=window_stride,
    )
    out_path = build_visualisation(
        report=report,
        matrix=matrix,
        output_dir=_resolve_output_dir(args),
        args=args,
    )
    print(out_path)


if __name__ == "__main__":
    main()
