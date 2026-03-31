import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from align_text_blocks_from_endpoints_no_pkl_v2 import compute_score_matrix, load_run_items, same_file, safe_name


# Parse CLI options for report loading, rendering, and optional line selection.
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


# Load a small JSON file with UTF-8 decoding.
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# Pick the summary file path, defaulting to the report directory.
def _resolve_summary_path(args) -> Path:
    if args.summary_json is not None:
        return args.summary_json
    return args.report_json.parent / "summary.json"


# Resolve the matrix window parameters from CLI overrides or saved metadata.
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


# Choose the output directory, defaulting to the original visualization folder.
def _resolve_output_dir(args) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.report_json.parent / "visualise_used_lines_only"


# Pull the saved kept lines from either supported report key.
def _extract_lines(report: dict) -> list[dict]:
    lines = report.get("lines")
    if lines is None:
        lines = report.get("lines_used")
    if lines is None:
        raise KeyError("Report JSON does not contain 'lines' or 'lines_used'")
    if not isinstance(lines, list):
        raise ValueError(f"Expected list of lines in report, got {type(lines).__name__}")
    return lines


# Normalize the displayed line id so selection matches the labels on the plot.
def _line_id_for_plot(line: dict, fallback_lid: int) -> int:
    return int(line.get("line_id", fallback_lid))


# Build a unique lookup from displayed line ids to line dictionaries.
def _build_line_lookup(lines: list[dict]) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for fallback_lid, line in enumerate(lines):
        line_id = _line_id_for_plot(line, fallback_lid)
        if line_id in lookup:
            raise ValueError(f"Duplicate line id in report lines: {line_id}")
        lookup[line_id] = line
    return lookup


# Parse a comma-separated list of ids and inclusive ranges like '5,7-9'.
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


# Decide which lines are visible and which subset should be highlighted.
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


# Build a descriptive output suffix so filtered images do not overwrite the default view.
def _selection_suffix(args, *, visible_ids: list[int], highlighted_ids: list[int]) -> str:
    if args.only_lines is not None:
        return "_only_lines_" + "_".join(str(value) for value in visible_ids)
    if args.highlight_lines is not None:
        base = "_highlight_lines_" + "_".join(str(value) for value in highlighted_ids)
        if args.hide_other_lines:
            base += "_only"
        return base
    return "_used_lines_labeled_full"


# Decide whether a visible line should receive a numeric label on the plot.
def _should_label_line(args, line_id: int, highlighted_ids: set[int]) -> bool:
    if args.label_selected_only:
        return line_id in highlighted_ids
    return True


# Render the chosen line view on top of the full chrF heatmap.
def build_visualisation(*, report: dict, matrix, output_dir: Path, args) -> Path:
    lines = _extract_lines(report)
    line_lookup = _build_line_lookup(lines)
    visible_ids, highlighted_ids = _resolve_selection(args, lines)
    highlighted_id_set = set(highlighted_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_prefix = f"{int(report['index']):04d}_{safe_name(report['fname'])}"
    out_path = output_dir / f"{case_prefix}{_selection_suffix(args, visible_ids=visible_ids, highlighted_ids=highlighted_ids)}.png"

    fig, ax = plt.subplots(1, 1, figsize=(float(args.figure_width), float(args.figure_height)))
    im = ax.imshow(matrix, aspect="auto", cmap="Greys")
    plt.colorbar(im, ax=ax, label="chrF")
    ax.set_xlabel("pred segment")
    ax.set_ylabel("ref segment")
    ax.set_title(f"{report['fname']} | visible lines={len(visible_ids)}")

    for line_id in visible_ids:
        line = line_lookup[line_id]
        x0 = float(line["x0"])
        y0 = float(line["y0"])
        x1 = float(line["x1"])
        y1 = float(line["y1"])
        is_highlighted = line_id in highlighted_id_set

        if is_highlighted:
            color = "red"
            alpha = 1.0
            linewidth = float(args.line_width)
            zorder = 3
        else:
            color = "#6b7280"
            alpha = 0.65
            linewidth = max(1.0, float(args.line_width) * 0.8)
            zorder = 2

        ax.plot((x0, x1), (y0, y1), color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

        if not _should_label_line(args, line_id, highlighted_id_set):
            continue

        xm = (x0 + x1) / 2.0
        ym = (y0 + y1) / 2.0
        ax.text(
            xm,
            ym,
            str(line_id),
            color="yellow",
            fontsize=float(args.label_fontsize),
            weight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "black",
                "edgecolor": "yellow",
                "alpha": 0.75,
            },
            zorder=4,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# Load the saved report and source texts, rebuild the matrix, and write the requested view.
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
