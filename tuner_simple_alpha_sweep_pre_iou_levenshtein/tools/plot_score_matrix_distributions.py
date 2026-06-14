#!/usr/bin/env python3
from __future__ import annotations

"""Plot score-matrix value distributions from a streamed Churro score pickle."""

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DocumentMetadata:
    """Language and audit metadata joined from the runfile JSON."""

    document_index: int | None
    fname: str
    main_language: str
    document_type: str


@dataclass
class LanguageAccumulator:
    """Streaming histogram and summary state for one language."""

    histogram_counts: np.ndarray
    document_count: int = 0
    finite_value_count: int = 0
    matrix_cell_count: int = 0
    finite_sum: float = 0.0
    finite_sum_of_squares: float = 0.0
    finite_minimum: float | None = None
    finite_maximum: float | None = None


def safe_path_component(value: str) -> str:
    """Return a path component safe enough for language and document filenames."""

    cleaned = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in str(value))
    return cleaned.strip("._") or "unknown"


def load_runfile_metadata(runfile_json: Path) -> dict[str, DocumentMetadata]:
    """Index runfile metadata by basename so score pickle records can be labelled by language."""

    payload = json.loads(Path(runfile_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected runfile JSON list, got {type(payload).__name__}")

    metadata_by_fname: dict[str, DocumentMetadata] = {}
    for document_index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        file_name = item.get("file_name", item.get("fname", f"document_{document_index:06d}"))
        fname = Path(str(file_name)).name
        metadata_by_fname[fname] = DocumentMetadata(
            document_index=int(document_index),
            fname=fname,
            main_language=str(item.get("main_language", "UNKNOWN") or "UNKNOWN"),
            document_type=str(item.get("document_type", "UNKNOWN") or "UNKNOWN"),
        )
    return metadata_by_fname


def iter_score_pickle_records(scores_pkl: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield dictionary records from a streamed pickle file without loading the whole file."""

    with Path(scores_pkl).open("rb") as handle:
        stream_index = 0
        while True:
            try:
                record = pickle.load(handle)
            except EOFError:
                break
            if isinstance(record, dict):
                yield stream_index, record
            stream_index += 1


def coerce_score_matrix(record: dict[str, Any]) -> np.ndarray:
    """Return the record score matrix as a two-dimensional float array."""

    matrix = np.asarray(record.get("scores"), dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"score matrix must be two-dimensional, got shape={tuple(matrix.shape)!r}")
    return np.ascontiguousarray(matrix, dtype=float)


def finite_values_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return finite score values as a flat one-dimensional array."""

    matrix_values = np.asarray(matrix, dtype=float)
    return matrix_values[np.isfinite(matrix_values)]


def summarize_values(values: np.ndarray, *, matrix_shape: tuple[int, int]) -> dict[str, Any]:
    """Compute compact document-level distribution statistics."""

    finite_count = int(values.size)
    total_count = int(matrix_shape[0]) * int(matrix_shape[1])
    if finite_count <= 0:
        return {
            "finite_value_count": 0,
            "matrix_cell_count": total_count,
            "finite_fraction": 0.0 if total_count > 0 else 0.0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "median": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "p99": None,
        }

    return {
        "finite_value_count": finite_count,
        "matrix_cell_count": total_count,
        "finite_fraction": float(finite_count / total_count) if total_count > 0 else 0.0,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def csv_value(value: Any) -> Any:
    """Return a stable scalar for CSV output."""

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    return value


def histogram_range(values: np.ndarray, *, fixed_minimum: float | None, fixed_maximum: float | None) -> tuple[float, float]:
    """Choose the histogram range for one document."""

    if fixed_minimum is not None and fixed_maximum is not None:
        return float(fixed_minimum), float(fixed_maximum)
    if values.size <= 0:
        return 0.0, 1.0
    minimum = float(np.min(values)) if fixed_minimum is None else float(fixed_minimum)
    maximum = float(np.max(values)) if fixed_maximum is None else float(fixed_maximum)
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum == maximum:
        center = minimum if math.isfinite(minimum) else 0.0
        return center - 0.5, center + 0.5
    return minimum, maximum


def draw_distribution_plot(
    *,
    values: np.ndarray,
    output_path: Path,
    title: str,
    subtitle: str,
    bins: int,
    value_range: tuple[float, float],
    dpi: int,
) -> None:
    """Write one histogram PNG for finite score values."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    try:
        if values.size > 0:
            axis.hist(values, bins=int(bins), range=value_range, color="#3568A6", edgecolor="#FFFFFF", linewidth=0.4)
            mean_value = float(np.mean(values))
            median_value = float(np.median(values))
            axis.axvline(mean_value, color="#D9480F", linewidth=1.4, label=f"mean {mean_value:.2f}")
            axis.axvline(median_value, color="#2B8A3E", linewidth=1.4, linestyle="--", label=f"median {median_value:.2f}")
            axis.legend(loc="upper right", frameon=False)
        else:
            axis.text(0.5, 0.5, "No finite score values", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(f"{title}\n{subtitle}", fontsize=10)
        axis.set_xlabel("Score matrix value")
        axis.set_ylabel("Cell count")
        axis.set_xlim(value_range)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        figure.savefig(output_path, dpi=int(dpi))
    finally:
        plt.close(figure)


def update_language_accumulator(
    *,
    accumulator: LanguageAccumulator,
    values: np.ndarray,
    matrix_shape: tuple[int, int],
    language_bins: np.ndarray,
) -> None:
    """Update one language-level streaming histogram and summary."""

    accumulator.document_count += 1
    accumulator.matrix_cell_count += int(matrix_shape[0]) * int(matrix_shape[1])
    if values.size <= 0:
        return
    counts, _edges = np.histogram(values, bins=language_bins)
    accumulator.histogram_counts += counts.astype(np.int64)
    accumulator.finite_value_count += int(values.size)
    accumulator.finite_sum += float(np.sum(values))
    accumulator.finite_sum_of_squares += float(np.sum(values * values))
    current_minimum = float(np.min(values))
    current_maximum = float(np.max(values))
    accumulator.finite_minimum = current_minimum if accumulator.finite_minimum is None else min(accumulator.finite_minimum, current_minimum)
    accumulator.finite_maximum = current_maximum if accumulator.finite_maximum is None else max(accumulator.finite_maximum, current_maximum)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows using fixed field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: csv_value(row.get(fieldname)) for fieldname in fieldnames})


def build_language_summary_rows(language_accumulators: dict[str, LanguageAccumulator]) -> list[dict[str, Any]]:
    """Convert language accumulators into CSV summary rows."""

    rows: list[dict[str, Any]] = []
    for language_name in sorted(language_accumulators):
        accumulator = language_accumulators[language_name]
        if accumulator.finite_value_count > 0:
            mean_value = accumulator.finite_sum / accumulator.finite_value_count
            variance = max(0.0, accumulator.finite_sum_of_squares / accumulator.finite_value_count - mean_value * mean_value)
            standard_deviation = math.sqrt(variance)
        else:
            mean_value = None
            standard_deviation = None
        rows.append(
            {
                "main_language": language_name,
                "document_count": int(accumulator.document_count),
                "matrix_cell_count": int(accumulator.matrix_cell_count),
                "finite_value_count": int(accumulator.finite_value_count),
                "finite_fraction": (
                    float(accumulator.finite_value_count / accumulator.matrix_cell_count)
                    if accumulator.matrix_cell_count > 0
                    else 0.0
                ),
                "minimum": accumulator.finite_minimum,
                "maximum": accumulator.finite_maximum,
                "mean": mean_value,
                "standard_deviation": standard_deviation,
            }
        )
    return rows


def draw_language_distribution_plots(
    *,
    language_accumulators: dict[str, LanguageAccumulator],
    output_dir: Path,
    language_bins: np.ndarray,
    dpi: int,
) -> list[str]:
    """Write one pooled histogram per language."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_paths: list[str] = []
    plot_dir = Path(output_dir) / "language_distributions"
    plot_dir.mkdir(parents=True, exist_ok=True)
    bin_widths = np.diff(language_bins)
    bin_left_edges = language_bins[:-1]
    for language_name in sorted(language_accumulators):
        accumulator = language_accumulators[language_name]
        output_path = plot_dir / f"{safe_path_component(language_name)}.png"
        figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        try:
            axis.bar(bin_left_edges, accumulator.histogram_counts, width=bin_widths, align="edge", color="#3568A6", edgecolor="#FFFFFF", linewidth=0.3)
            axis.set_xlim(float(language_bins[0]), float(language_bins[-1]))
            axis.set_title(
                f"{language_name}\n"
                f"documents={accumulator.document_count} finite_values={accumulator.finite_value_count}",
                fontsize=10,
            )
            axis.set_xlabel("Score matrix value")
            axis.set_ylabel("Cell count")
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
            figure.savefig(output_path, dpi=int(dpi))
        finally:
            plt.close(figure)
        output_paths.append(str(output_path))
    return output_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Plot per-document and per-language score-matrix value distributions from a streamed score pickle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores-pkl", type=Path, required=True, help="Score matrix pickle stream to inspect.")
    parser.add_argument("--runfile-json", type=Path, required=True, help="outputs.json used to map fname to language and document type.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where PNG and CSV files are written.")
    parser.add_argument("--language", action="append", dest="languages", default=None, help="Optional main_language filter; repeatable.")
    parser.add_argument("--target-fname", action="append", dest="target_fnames", default=None, help="Optional exact filename filter; repeatable.")
    parser.add_argument("--bins", type=int, default=100, help="Histogram bin count.")
    parser.add_argument("--value-min", type=float, default=0.0, help="Minimum plotted score value.")
    parser.add_argument("--value-max", type=float, default=100.0, help="Maximum plotted score value.")
    parser.add_argument("--auto-document-range", action="store_true", help="Use each document's finite min/max for its own PNG instead of value-min/value-max.")
    parser.add_argument("--dpi", type=int, default=120, help="PNG resolution.")
    parser.add_argument("--skip-document-plots", action="store_true", help="Only write CSV summaries and language-level plots.")
    parser.add_argument("--max-documents", type=int, default=None, help="Optional debug cap after filters.")
    return parser.parse_args()


def main() -> None:
    """Run the plotting workflow."""

    args = parse_arguments()
    if int(args.bins) <= 0:
        raise ValueError("--bins must be positive")
    if int(args.dpi) <= 0:
        raise ValueError("--dpi must be positive")
    if float(args.value_max) <= float(args.value_min):
        raise ValueError("--value-max must be greater than --value-min")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_fname = load_runfile_metadata(Path(args.runfile_json))
    language_filter = {str(value) for value in (args.languages or [])}
    target_filter = {Path(str(value)).name for value in (args.target_fnames or [])}
    language_bins = np.linspace(float(args.value_min), float(args.value_max), int(args.bins) + 1)

    document_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    language_accumulators: dict[str, LanguageAccumulator] = defaultdict(
        lambda: LanguageAccumulator(histogram_counts=np.zeros(int(args.bins), dtype=np.int64))
    )

    processed_count = 0
    for stream_index, record in iter_score_pickle_records(Path(args.scores_pkl)):
        fname = Path(str(record.get("fname", f"record_{stream_index:06d}"))).name
        metadata = metadata_by_fname.get(
            fname,
            DocumentMetadata(document_index=None, fname=fname, main_language="UNKNOWN", document_type="UNKNOWN"),
        )
        if language_filter and metadata.main_language not in language_filter:
            continue
        if target_filter and fname not in target_filter:
            continue
        if args.max_documents is not None and processed_count >= int(args.max_documents):
            break

        try:
            matrix = coerce_score_matrix(record)
        except Exception as exc:
            skipped_rows.append(
                {
                    "stream_index": stream_index,
                    "fname": fname,
                    "main_language": metadata.main_language,
                    "document_type": metadata.document_type,
                    "skip_reason": repr(exc),
                }
            )
            continue

        values = finite_values_from_matrix(matrix)
        matrix_shape = tuple(int(value) for value in matrix.shape)
        summary = summarize_values(values, matrix_shape=matrix_shape)
        document_plot_path = ""
        if not bool(args.skip_document_plots):
            if bool(args.auto_document_range):
                value_range = histogram_range(values, fixed_minimum=None, fixed_maximum=None)
            else:
                value_range = (float(args.value_min), float(args.value_max))
            document_plot_path = str(
                output_dir
                / "document_distributions"
                / safe_path_component(metadata.main_language)
                / f"{safe_path_component(fname)}.png"
            )
            draw_distribution_plot(
                values=values,
                output_path=Path(document_plot_path),
                title=f"{metadata.main_language} / {fname}",
                subtitle=f"shape={matrix_shape[0]}x{matrix_shape[1]} finite={summary['finite_value_count']}/{summary['matrix_cell_count']}",
                bins=int(args.bins),
                value_range=value_range,
                dpi=int(args.dpi),
            )

        update_language_accumulator(
            accumulator=language_accumulators[metadata.main_language],
            values=values,
            matrix_shape=matrix_shape,
            language_bins=language_bins,
        )
        document_rows.append(
            {
                "stream_index": stream_index,
                "document_index": metadata.document_index,
                "fname": fname,
                "main_language": metadata.main_language,
                "document_type": metadata.document_type,
                "row_count": matrix_shape[0],
                "column_count": matrix_shape[1],
                "document_plot_path": document_plot_path,
                **summary,
            }
        )
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"processed {processed_count} documents")

    language_plot_paths = draw_language_distribution_plots(
        language_accumulators=language_accumulators,
        output_dir=output_dir,
        language_bins=language_bins,
        dpi=int(args.dpi),
    )

    document_fieldnames = [
        "stream_index",
        "document_index",
        "fname",
        "main_language",
        "document_type",
        "row_count",
        "column_count",
        "matrix_cell_count",
        "finite_value_count",
        "finite_fraction",
        "minimum",
        "maximum",
        "mean",
        "standard_deviation",
        "median",
        "p01",
        "p05",
        "p25",
        "p75",
        "p95",
        "p99",
        "document_plot_path",
    ]
    language_fieldnames = [
        "main_language",
        "document_count",
        "matrix_cell_count",
        "finite_value_count",
        "finite_fraction",
        "minimum",
        "maximum",
        "mean",
        "standard_deviation",
    ]
    skipped_fieldnames = ["stream_index", "fname", "main_language", "document_type", "skip_reason"]
    write_csv(output_dir / "document_distribution_summary.csv", document_rows, document_fieldnames)
    write_csv(output_dir / "language_distribution_summary.csv", build_language_summary_rows(language_accumulators), language_fieldnames)
    write_csv(output_dir / "skipped_score_matrix_records.csv", skipped_rows, skipped_fieldnames)

    manifest = {
        "scores_pkl": str(Path(args.scores_pkl)),
        "runfile_json": str(Path(args.runfile_json)),
        "output_dir": str(output_dir),
        "processed_document_count": int(len(document_rows)),
        "skipped_record_count": int(len(skipped_rows)),
        "language_count": int(len(language_accumulators)),
        "language_plot_paths": language_plot_paths,
        "document_plots_enabled": not bool(args.skip_document_plots),
        "bins": int(args.bins),
        "value_min": float(args.value_min),
        "value_max": float(args.value_max),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
