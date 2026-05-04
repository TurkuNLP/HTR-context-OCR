#!/usr/bin/env python3
from __future__ import annotations

"""Plot utilities for Hough parameter sweeps.

Primary plotting path uses one long-format CSV with rows:
(doc, parameter, value, along_lines_nls, ...).
From that single table we generate per-document plots under:
<output-dir>/plots/<document-name>/<parameter>_vs_levenshtein_along_lines.png
"""

import argparse
import csv
import json
import math
from pathlib import Path

SUPPORTED_PLOT_PARAMETERS: tuple[str, ...] = (
    "hough_threshold",
    "hough_line_length",
    "hough_line_gap",
)


def _safe_float(value) -> float | None:
    """Convert value to finite float or return ``None`` when unavailable."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value) -> int | None:
    """Convert value to int or return ``None`` if conversion fails."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except Exception:
        return None


def _plot_one_parameter(*, parameter: str, rows: list[dict], output_png: Path) -> None:
    """Plot one parameter curve and per-point scatter for one document."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [int(row.get("value", 0)) for row in rows]
    ys = [
        float("nan") if row.get("mean_along_lines_nls") is None else float(row.get("mean_along_lines_nls"))
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter_x: list[int] = []
    scatter_y: list[float] = []
    for row in rows:
        x = int(row.get("value", 0))
        for doc_row in row.get("docs", []):
            y = _safe_float(doc_row.get("along_lines_nls")) if isinstance(doc_row, dict) else None
            if y is None:
                continue
            scatter_x.append(x)
            scatter_y.append(y)

    if scatter_x:
        ax.scatter(scatter_x, scatter_y, s=14, alpha=0.28, label="Along-lines NLS")

    ax.plot(xs, ys, marker="o", linewidth=2.0, label="Curve")

    best = None
    for row in rows:
        mean_val = _safe_float(row.get("mean_along_lines_nls"))
        if mean_val is None:
            continue
        if best is None or mean_val > float(best.get("mean_along_lines_nls", -1.0)):
            best = row

    if best is not None:
        bx = int(best["value"])
        by = float(best["mean_along_lines_nls"])
        ax.scatter([bx], [by], s=70, zorder=3, label=f"Best ({bx}, {by:.4f})")

    ax.set_xlabel(parameter)
    ax.set_ylabel("Levenshtein Along Lines (NLS)")
    ax.set_title(f"{parameter} vs Levenshtein Along Lines")
    ax.grid(alpha=0.25)
    ax.legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _load_long_csv_rows(*, csv_path: Path) -> list[dict]:
    """Load normalized long rows from CSV for supported parameters."""
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            parameter = str(raw.get("parameter", "")).strip()
            if parameter not in SUPPORTED_PLOT_PARAMETERS:
                continue

            value = _safe_int(raw.get("value"))
            if value is None:
                continue

            fname = str(raw.get("fname", "")).strip()
            doc_index = _safe_int(raw.get("doc_index"))
            if doc_index is None:
                doc_index = -1

            rows.append(
                {
                    "parameter": parameter,
                    "value": int(value),
                    "along_lines_nls": _safe_float(raw.get("along_lines_nls")),
                    "fname": fname,
                    "doc_index": int(doc_index),
                }
            )

    return rows


def _rows_for_parameter_from_long_rows(*, parameter: str, long_rows: list[dict]) -> list[dict]:
    """Build aggregated rows from long rows for one parameter (legacy fallback)."""
    grouped: dict[int, list[dict]] = {}

    for row in long_rows:
        if str(row.get("parameter")) != str(parameter):
            continue
        value = _safe_int(row.get("value"))
        if value is None:
            continue
        grouped.setdefault(int(value), []).append({"along_lines_nls": row.get("along_lines_nls")})

    out_rows: list[dict] = []
    for value in sorted(grouped.keys()):
        docs = grouped[int(value)]
        vals = [_safe_float(doc_row.get("along_lines_nls")) for doc_row in docs if isinstance(doc_row, dict)]
        finite_vals = [v for v in vals if v is not None]
        mean_val = None
        if finite_vals:
            mean_val = float(sum(finite_vals) / len(finite_vals))

        out_rows.append(
            {
                "parameter": str(parameter),
                "value": int(value),
                "mean_along_lines_nls": mean_val,
                "docs": docs,
            }
        )

    return out_rows


def _document_folder_name(*, fname: str, doc_index: int, used_names: set[str]) -> str:
    """Create deterministic per-document folder names under plots/."""
    base = Path(str(fname)).name.strip() or f"document_{doc_index}"
    candidate = base

    if candidate in used_names:
        candidate = f"{base}__idx{doc_index}"
        suffix = 1
        while candidate in used_names:
            candidate = f"{base}__idx{doc_index}_{suffix}"
            suffix += 1

    used_names.add(candidate)
    return candidate


def _build_per_document_parameter_values(long_rows: list[dict]) -> dict[tuple[int, str], dict[str, dict[int, float | None]]]:
    """Group long rows into per-document, per-parameter value maps."""
    grouped: dict[tuple[int, str], dict[str, dict[int, float | None]]] = {}

    for row in long_rows:
        parameter = str(row.get("parameter", ""))
        if parameter not in SUPPORTED_PLOT_PARAMETERS:
            continue

        value = _safe_int(row.get("value"))
        if value is None:
            continue

        fname = str(row.get("fname", ""))
        doc_index = _safe_int(row.get("doc_index"))
        if doc_index is None:
            doc_index = -1

        key = (int(doc_index), fname)
        param_map = grouped.setdefault(key, {})
        value_map = param_map.setdefault(parameter, {})

        along = _safe_float(row.get("along_lines_nls"))
        existing = value_map.get(int(value))

        if int(value) not in value_map:
            value_map[int(value)] = along
        elif existing is None and along is not None:
            value_map[int(value)] = along
        elif existing is not None and along is not None:
            value_map[int(value)] = max(float(existing), float(along))

    return grouped


def _rows_for_one_document_parameter(*, parameter: str, value_map: dict[int, float | None]) -> list[dict]:
    """Convert per-document value map into plotting row structure."""
    rows: list[dict] = []
    for value in sorted(int(v) for v in value_map.keys()):
        y = _safe_float(value_map.get(int(value)))
        rows.append(
            {
                "parameter": str(parameter),
                "value": int(value),
                "mean_along_lines_nls": y,
                "docs": [] if y is None else [{"along_lines_nls": y}],
            }
        )
    return rows


def _render_per_document_plots_from_long_csv(
    *,
    summary: dict,
    output_dir: Path,
    long_csv_path: Path,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Render per-document parameter plots from unified long CSV."""
    long_rows = _load_long_csv_rows(csv_path=long_csv_path)
    grouped = _build_per_document_parameter_values(long_rows)

    plot_paths_flat: dict[str, str] = {}
    plot_paths_by_document: dict[str, dict[str, str]] = {}

    used_folder_names: set[str] = set()

    for doc_key in sorted(grouped.keys(), key=lambda item: (int(item[0]), str(item[1]))):
        doc_index, fname = doc_key
        doc_param_values = grouped[doc_key]

        folder_name = _document_folder_name(fname=str(fname), doc_index=int(doc_index), used_names=used_folder_names)
        doc_plot_dir = output_dir / folder_name

        doc_label = str(fname) if str(fname) else f"document_{doc_index}"
        if doc_label in plot_paths_by_document:
            doc_label = f"{doc_label}__idx{doc_index}"

        per_parameter_paths: dict[str, str] = {}

        for parameter in SUPPORTED_PLOT_PARAMETERS:
            value_map = doc_param_values.get(parameter, {})
            if not value_map:
                continue

            rows = _rows_for_one_document_parameter(parameter=parameter, value_map=value_map)
            if not rows:
                continue

            plot_path = doc_plot_dir / f"{parameter}_vs_levenshtein_along_lines.png"
            _plot_one_parameter(parameter=str(parameter), rows=rows, output_png=plot_path)

            per_parameter_paths[str(parameter)] = str(plot_path)
            flat_key = f"{doc_label}::{parameter}"
            plot_paths_flat[flat_key] = str(plot_path)

        if per_parameter_paths:
            plot_paths_by_document[doc_label] = per_parameter_paths

    summary["plot_generation_source"] = "all_documents_parameter_influence_csv"
    summary["generated_plot_root_dir"] = str(output_dir)
    summary["generated_plot_count_total"] = int(sum(len(v) for v in plot_paths_by_document.values()))

    return plot_paths_flat, plot_paths_by_document


def _render_plots_from_summary_rows(*, summary: dict, output_dir: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Legacy fallback when long CSV is unavailable."""
    if "parameter_sweeps" not in summary or not isinstance(summary["parameter_sweeps"], dict):
        raise ValueError("Invalid summary format: missing parameter_sweeps dict")

    synthetic_dir = output_dir / "all_documents"
    flat: dict[str, str] = {}
    by_doc: dict[str, dict[str, str]] = {"all_documents": {}}

    for parameter in SUPPORTED_PLOT_PARAMETERS:
        payload = summary["parameter_sweeps"].get(parameter)
        if not isinstance(payload, dict):
            continue

        rows = payload.get("rows", [])
        if not isinstance(rows, list) or not rows:
            continue

        plot_path = synthetic_dir / f"{parameter}_vs_levenshtein_along_lines.png"
        _plot_one_parameter(parameter=str(parameter), rows=rows, output_png=plot_path)

        payload["plot_path"] = str(plot_path)
        by_doc["all_documents"][str(parameter)] = str(plot_path)
        flat[f"all_documents::{parameter}"] = str(plot_path)

    summary["plot_generation_source"] = "summary_parameter_rows"
    summary["generated_plot_root_dir"] = str(output_dir)
    summary["generated_plot_count_total"] = int(sum(len(v) for v in by_doc.values()))

    return flat, by_doc


def render_plots_from_summary_dict(summary: dict, *, output_dir: Path | None = None) -> dict[str, str]:
    """Render plots and return flat path mapping.

    For richer metadata, this function also mutates `summary` with
    `generated_plot_paths_by_document`.
    """
    if output_dir is None:
        root = Path(summary.get("output_dir", "."))
        output_dir = root / "plots"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path_value = summary.get("all_documents_parameter_influence_csv_path")
    if isinstance(csv_path_value, str) and csv_path_value.strip():
        csv_path = Path(csv_path_value)
        if csv_path.exists():
            flat, by_doc = _render_per_document_plots_from_long_csv(
                summary=summary,
                output_dir=output_dir,
                long_csv_path=csv_path,
            )
            summary["generated_plot_paths_by_document"] = by_doc
            return flat

    flat, by_doc = _render_plots_from_summary_rows(summary=summary, output_dir=output_dir)
    summary["generated_plot_paths_by_document"] = by_doc
    return flat


def generate_plots_for_summary_json(
    *,
    summary_json: Path,
    output_dir: Path | None = None,
    overwrite_summary: bool = True,
) -> dict:
    """Generate plots from summary JSON and optionally write metadata back."""
    summary_json = Path(summary_json)
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing summary JSON: {summary_json}")

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    flat_paths = render_plots_from_summary_dict(summary, output_dir=output_dir)

    summary["generated_plot_paths"] = flat_paths
    if "generated_plot_paths_by_document" not in summary:
        summary["generated_plot_paths_by_document"] = {}

    if overwrite_summary:
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Hough sweep plots from hough_parameter_sweep_summary.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary-json", type=Path, required=True, help="Path to summary JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional plot output directory (default: <summary.output_dir>/plots)",
    )
    parser.add_argument(
        "--no-overwrite-summary",
        action="store_true",
        help="Do not write plot paths back to summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_plots_for_summary_json(
        summary_json=Path(args.summary_json),
        output_dir=args.output_dir,
        overwrite_summary=not bool(args.no_overwrite_summary),
    )

    print(f"Summary JSON: {args.summary_json}")
    print(f"Plot root: {summary.get('generated_plot_root_dir')}")
    print("Generated plot files by document:")
    by_doc = summary.get("generated_plot_paths_by_document", {})
    if isinstance(by_doc, dict):
        for doc_name, per_param in by_doc.items():
            if not isinstance(per_param, dict):
                continue
            print(f"  - {doc_name}")
            for parameter, path in per_param.items():
                print(f"      {parameter}: {path}")


if __name__ == "__main__":
    main()
