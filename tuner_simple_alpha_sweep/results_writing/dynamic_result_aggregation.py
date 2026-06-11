from __future__ import annotations

"""Aggregate dynamic worker progress rows into final tuner_simple outputs."""

import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from tuner_simple_alpha_sweep.plotting.stitched_language_panels import save_stitched_language_image
from tuner_simple_alpha_sweep.results_writing.flat_csv_tables import write_all_flat_outputs
from tuner_simple_alpha_sweep.results_writing.progress_rows import (
    progress_row_to_loadable_row,
    progress_row_to_loaded_row,
    progress_row_to_result_row,
    progress_row_to_runfile_row,
    progress_row_to_skipped_row,
)


def read_progress_csv(progress_csv_path: Path) -> list[dict[str, Any]]:
    """Load every worker progress row that has reached the shared CSV file."""

    path = Path(progress_csv_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_handle:
        return [dict(row) for row in csv.DictReader(input_handle)]


def row_completed_time(progress_row: Mapping[str, Any]) -> float:
    """Return the completion timestamp used to pick the latest attempt for one document."""

    try:
        return float(progress_row.get("completed_at_unix_seconds") or 0.0)
    except Exception:
        return 0.0


def deduplicate_progress_rows(progress_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest completed progress row for every pool ordinal."""

    latest_row_by_pool_ordinal: dict[int, dict[str, Any]] = {}
    for progress_row in progress_rows:
        try:
            pool_ordinal = int(float(progress_row.get("pool_ordinal") or -1))
        except Exception:
            continue
        if pool_ordinal < 0:
            continue
        previous_row = latest_row_by_pool_ordinal.get(pool_ordinal)
        if previous_row is None or row_completed_time(progress_row) >= row_completed_time(previous_row):
            latest_row_by_pool_ordinal[pool_ordinal] = progress_row
    return [latest_row_by_pool_ordinal[pool_ordinal] for pool_ordinal in sorted(latest_row_by_pool_ordinal)]


def collect_final_rows(progress_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Convert dynamic progress rows back into the serial runner's final CSV row groups."""

    runfile_rows: list[dict[str, Any]] = []
    loadable_rows: list[dict[str, Any]] = []
    loaded_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for progress_row in progress_rows:
        runfile_rows.append(progress_row_to_runfile_row(progress_row))
        loadable_row = progress_row_to_loadable_row(progress_row)
        if loadable_row is not None:
            loadable_rows.append(loadable_row)
        loaded_row = progress_row_to_loaded_row(progress_row)
        if loaded_row is not None:
            loaded_rows.append(loaded_row)
        skipped_row = progress_row_to_skipped_row(progress_row)
        if skipped_row is not None:
            skipped_rows.append(skipped_row)
        result_row = progress_row_to_result_row(progress_row)
        if result_row is not None:
            result_rows.append(result_row)
    return {
        "runfile_rows": runfile_rows,
        "loadable_rows": loadable_rows,
        "loaded_rows": loaded_rows,
        "skipped_rows": skipped_rows,
        "result_rows": result_rows,
    }


def stitch_language_panels_from_progress(
    *,
    progress_rows: list[dict[str, Any]],
    output_dir: Path,
    stitched_panel_columns: int,
    log,
) -> list[str]:
    """Create final stitched language images from panel paths recorded by workers."""

    panel_paths_by_language: dict[str, list[Path]] = defaultdict(list)
    for progress_row in progress_rows:
        panel_path = str(progress_row.get("panel_path") or "").strip()
        if not panel_path:
            continue
        path = Path(panel_path)
        if not path.exists():
            continue
        language_name = str(progress_row.get("main_language") or "UNKNOWN")
        panel_paths_by_language[language_name].append(path)
    if not panel_paths_by_language:
        return []
    from PIL import Image
    from tuner_simple_alpha_sweep.plotting.document_panel_renderer import safe_path_component

    stitched_paths: list[str] = []
    plots_dir = Path(output_dir) / "plots"
    for language_name in sorted(panel_paths_by_language):
        stitched_path = plots_dir / f"stitched_best_combination_{safe_path_component(language_name)}_documents.png"
        saved_path = save_stitched_language_image(
            panel_paths=panel_paths_by_language[language_name],
            stitched_output_path=stitched_path,
            panel_columns=int(stitched_panel_columns),
            image_module=Image,
        )
        if saved_path is not None:
            stitched_paths.append(str(saved_path))
            log(f"[dynamic-aggregate] wrote stitched language plot: {saved_path}")
    return stitched_paths


def aggregate_dynamic_worker_outputs(
    *,
    output_dir: Path,
    progress_csv_path: Path | None,
    plot_mode: str,
    stitched_panel_columns: int,
    log,
) -> dict[str, Any]:
    """Build final CSV files and optional stitched plots from dynamic worker progress."""

    aggregation_started_at = time.perf_counter()
    output_dir = Path(output_dir)
    progress_csv = Path(progress_csv_path) if progress_csv_path is not None else output_dir / "progress_csv" / "document_completion_attempts.csv"
    log(f"[dynamic-aggregate] progress csv: {progress_csv}")
    all_progress_rows = read_progress_csv(progress_csv)
    final_progress_rows = deduplicate_progress_rows(all_progress_rows)
    final_rows = collect_final_rows(final_progress_rows)
    stitched_plot_paths: list[str] = []
    if str(plot_mode) != "none":
        stitched_plot_paths = stitch_language_panels_from_progress(
            progress_rows=final_progress_rows,
            output_dir=output_dir,
            stitched_panel_columns=int(stitched_panel_columns),
            log=log,
        )
        if str(plot_mode) == "stitched-language":
            temporary_panel_dir = output_dir / "plots" / ".temporary_document_panels"
            if temporary_panel_dir.exists():
                shutil.rmtree(temporary_panel_dir)
                log(f"[dynamic-aggregate] removed temporary panel directory: {temporary_panel_dir}")
    run_summary = {
        "schema_version": "tuner_simple_dynamic_aggregation_v1",
        "elapsed_seconds": float(time.perf_counter() - aggregation_started_at),
        "progress_csv": str(progress_csv),
        "progress_row_count": int(len(all_progress_rows)),
        "deduplicated_document_count": int(len(final_progress_rows)),
        "selected_document_count": int(len(final_rows["runfile_rows"])),
        "processed_document_count": int(len(final_rows["result_rows"])),
        "skipped_document_count": int(len(final_rows["skipped_rows"])),
        "output_dir": str(output_dir),
        "stitched_plot_paths": stitched_plot_paths,
    }
    written_paths = write_all_flat_outputs(
        output_dir=output_dir,
        runfile_rows=final_rows["runfile_rows"],
        loadable_rows=final_rows["loadable_rows"],
        loaded_rows=final_rows["loaded_rows"],
        skipped_rows=final_rows["skipped_rows"],
        result_rows=final_rows["result_rows"],
        run_summary=run_summary,
    )
    run_summary["written_paths"] = written_paths
    log(f"[dynamic-aggregate] done summary={run_summary}")
    return run_summary
