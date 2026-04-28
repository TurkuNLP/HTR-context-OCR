"""Standalone CLI for visualising used lines from one report JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runfile_records import load_run_items, safe_name, same_file
from score_matrix_builder import compute_score_matrix
from visualisation.render_alignment_matrix_views import (
    build_line_lookup,
    create_heatmap_figure,
    draw_labeled_line,
)
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-report visualization."""
    p = argparse.ArgumentParser(
        description=(
            "Visualize kept alignment lines from a report JSON on top of the full "
            "chrF heatmap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--report-json", type=Path, required=True, help="Path to *_report.json")
    p.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to summary.json. Defaults to report_json.parent/summary.json.",
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
        help="Comma-separated line ids/ranges to render, e.g. '5' or '5,7-9'.",
    )
    selection.add_argument(
        "--highlight-lines",
        type=str,
        default=None,
        help="Comma-separated line ids/ranges to highlight in red.",
    )
    p.add_argument(
        "--hide-other-lines",
        action="store_true",
        help="With --highlight-lines, hide non-selected lines instead of drawing them in gray.",
    )
    p.add_argument(
        "--label-selected-only",
        action="store_true",
        help="Only label selected/highlighted lines.",
    )
    return p.parse_args()


def load_json(path: Path) -> dict:
    """Load and parse one JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_summary_path(args: argparse.Namespace) -> Path:
    """Resolve summary JSON path from args/default location."""
    if args.summary_json is not None:
        return Path(args.summary_json)
    return Path(args.report_json).parent / "summary.json"


def resolve_window_params(args: argparse.Namespace, report: dict, summary: dict) -> tuple[int, int]:
    """Resolve window_size/window_stride from overrides or report metadata."""
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


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve output directory for generated visualizations."""
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path(args.report_json).parent / "visualise_used_lines_only"


def extract_lines(report: dict) -> list[dict]:
    """Extract lines list from supported report keys."""
    lines = report.get("lines")
    if lines is None:
        lines = report.get("lines_used")
    if lines is None:
        raise KeyError("Report JSON does not contain 'lines' or 'lines_used'")
    if not isinstance(lines, list):
        raise ValueError(f"Expected list of lines in report, got {type(lines).__name__}")
    return lines


def parse_line_id_spec(spec: str, *, valid_line_ids: set[int]) -> list[int]:
    """Parse comma/range line-id selection syntax into sorted unique ids."""
    if spec is None:
        return []

    selected: set[int] = set()
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
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
        valid_str = ", ".join(str(value) for value in sorted(valid_line_ids))
        raise ValueError(f"Unknown line ids {invalid}; available line ids are: {valid_str}")
    return sorted(selected)


def resolve_selection(args: argparse.Namespace, lines: list[dict]) -> tuple[list[int], list[int]]:
    """Resolve visible/highlighted line-id sets from CLI options."""
    line_lookup = build_line_lookup(lines)
    all_line_ids = sorted(line_lookup)
    valid_ids = set(all_line_ids)

    if args.only_lines is not None:
        selected = parse_line_id_spec(args.only_lines, valid_line_ids=valid_ids)
        return selected, selected

    if args.highlight_lines is not None:
        selected = parse_line_id_spec(args.highlight_lines, valid_line_ids=valid_ids)
        visible = selected if args.hide_other_lines else all_line_ids
        return visible, selected

    return all_line_ids, all_line_ids


def selection_suffix(args: argparse.Namespace, *, visible_ids: list[int], highlighted_ids: list[int]) -> str:
    """Build output file suffix that describes the current line selection."""
    if args.only_lines is not None:
        return "_only_lines_" + "_".join(str(value) for value in visible_ids)
    if args.highlight_lines is not None:
        base = "_highlight_lines_" + "_".join(str(value) for value in highlighted_ids)
        if args.hide_other_lines:
            base += "_only"
        return base
    return "_used_lines_labeled_full"


def should_label_line(args: argparse.Namespace, line_id: int, highlighted_ids: set[int]) -> bool:
    """Return whether one visible line should display its text label."""
    if args.label_selected_only:
        return line_id in highlighted_ids
    return True


def build_visualisation(*, report: dict, matrix, output_dir: Path, args: argparse.Namespace) -> Path:
    """Render and save one matrix visualization with selected lines."""
    lines = extract_lines(report)
    line_lookup = build_line_lookup(lines)
    visible_ids, highlighted_ids = resolve_selection(args, lines)
    highlighted_id_set = set(highlighted_ids)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_prefix = f"{int(report['index']):04d}_{safe_name(report['fname'])}"
    out_path = output_dir / f"{case_prefix}{selection_suffix(args, visible_ids=visible_ids, highlighted_ids=highlighted_ids)}.png"

    fig, ax = create_heatmap_figure(
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

        draw_labeled_line(
            ax,
            line=line,
            label=str(line_id),
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
            show_label=should_label_line(args, line_id, highlighted_id_set),
            label_fontsize=float(args.label_fontsize),
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    """Run standalone report visualization CLI."""
    args = parse_args()
    if not Path(args.report_json).exists():
        raise FileNotFoundError(f"Missing report JSON: {args.report_json}")
    if not Path(args.runfile_json).exists():
        raise FileNotFoundError(f"Missing runfile JSON: {args.runfile_json}")
    if args.hide_other_lines and args.highlight_lines is None:
        raise ValueError("--hide-other-lines requires --highlight-lines")

    summary_path = resolve_summary_path(args)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary JSON: {summary_path}. Pass --summary-json or override --window-size/--window-stride."
        )

    report = load_json(Path(args.report_json))
    summary = load_json(summary_path)
    window_size, window_stride = resolve_window_params(args, report, summary)

    items = load_run_items(Path(args.runfile_json))
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
        output_dir=resolve_output_dir(args),
        args=args,
    )
    print(out_path)


if __name__ == "__main__":
    main()
