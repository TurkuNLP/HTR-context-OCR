from __future__ import annotations

"""Top-level orchestration for fixed-range Hough parameter tuning.

This module intentionally stays thin and delegates detailed logic to smaller
modules:
- `document_prep.py` for loading docs and matrix source layering.
- `sweep_engine.py` for exhaustive nested-grid evaluation.
- `sweep_aggregation.py` for parameter influence curve aggregation.
- `tuner_result_exports.py` for unified JSON/CSV exports.

Keeping this file focused makes maintenance safer while preserving a stable
public API for runner scripts.
"""

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Iterable

try:
    from .runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime_paths import ensure_tuner_runtime_paths  # type: ignore

# Ensure shared project paths are importable for both package and script execution.
ensure_tuner_runtime_paths()

try:
    from .document_prep import load_documents as _load_documents
    from .sweep_aggregation import (
        best_curve_row,
        best_doc_nls_mean,
        build_curve_row,
        compact_best_curve_row,
    )
    from .sweep_engine import run_document_sweeps
    from .tuner_config import (
        DEFAULT_SCORE_INDEX_CACHE_DIR,
        HOUGH_LINE_GAP_MAX,
        HOUGH_LINE_GAP_MIN,
        HOUGH_LINE_LENGTH_MAX,
        HOUGH_LINE_LENGTH_MIN,
        HOUGH_THRESHOLD_MAX,
        HOUGH_THRESHOLD_MIN,
        HoughBaselineConfig,
        LogFn,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SUPPORTED_SWEEP_PARAMETERS,
        fixed_parameter_ranges as _fixed_parameter_ranges,
        fixed_values_for,
    )
    from .tuner_outputs import write_best_configs_csv, write_parameter_curve_csv
    from .tuner_result_exports import (
        build_parameter_influence_rows,
        write_best_params_json,
        write_parameter_influence_csv,
    )
except ImportError:
    from document_prep import load_documents as _load_documents  # type: ignore
    from sweep_aggregation import (  # type: ignore
        best_curve_row,
        best_doc_nls_mean,
        build_curve_row,
        compact_best_curve_row,
    )
    from sweep_engine import run_document_sweeps  # type: ignore
    from tuner_config import (  # type: ignore
        DEFAULT_SCORE_INDEX_CACHE_DIR,
        HOUGH_LINE_GAP_MAX,
        HOUGH_LINE_GAP_MIN,
        HOUGH_LINE_LENGTH_MAX,
        HOUGH_LINE_LENGTH_MIN,
        HOUGH_THRESHOLD_MAX,
        HOUGH_THRESHOLD_MIN,
        HoughBaselineConfig,
        LogFn,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SUPPORTED_SWEEP_PARAMETERS,
        fixed_parameter_ranges as _fixed_parameter_ranges,
        fixed_values_for,
    )
    from tuner_outputs import write_best_configs_csv, write_parameter_curve_csv  # type: ignore
    from tuner_result_exports import (  # type: ignore
        build_parameter_influence_rows,
        write_best_params_json,
        write_parameter_influence_csv,
    )

from levenshtein_metric import BACKEND_C, SUPPORTED_BACKENDS

try:
    from levenshtein_metric import BACKEND_PYTHON
except Exception:
    BACKEND_PYTHON = "python"


def _no_log(_: str) -> None:
    return


def fixed_parameter_ranges() -> dict[str, tuple[int, int]]:
    """Public accessor for hardcoded fixed sweep ranges."""
    return _fixed_parameter_ranges()


def load_documents(
    *,
    runfile_json: Path,
    window_size: int,
    window_stride: int,
    levenshtein_backend: str,
    matrix_cache_dir: Path | None = None,
    scores_pkl_ref_to_pred: Path | None = None,
    score_index_cache_file: Path | None = None,
    score_index_cache_dir: Path | None = None,
    disable_pkl_matrix_source: bool = False,
    target_fnames: Iterable[str] | None = None,
    max_items: int | None = None,
    timing_out: dict | None = None,
    log_fn: LogFn | None = None,
    hough_start: float = 2.6,
):
    """Public wrapper around document preparation.

    `hough_start` is exposed so callers can keep Hough precompute aligned with
    the same threshold-start strategy used during sweeps.
    """
    return _load_documents(
        runfile_json=Path(runfile_json),
        window_size=int(window_size),
        window_stride=int(window_stride),
        hough_start=float(hough_start),
        levenshtein_backend=str(levenshtein_backend),
        matrix_cache_dir=matrix_cache_dir,
        scores_pkl_ref_to_pred=scores_pkl_ref_to_pred,
        score_index_cache_file=score_index_cache_file,
        score_index_cache_dir=score_index_cache_dir,
        disable_pkl_matrix_source=bool(disable_pkl_matrix_source),
        target_fnames=target_fnames,
        max_items=max_items,
        timing_out=timing_out,
        log_fn=log_fn,
    )


def run_hough_parameter_sweeps(
    *,
    runfile_json: Path,
    output_dir: Path,
    window_size: int,
    window_stride: int,
    baseline_cfg: HoughBaselineConfig | None = None,
    matrix_cache_dir: Path | None = None,
    scores_pkl_ref_to_pred: Path | None = None,
    score_index_cache_file: Path | None = None,
    score_index_cache_dir: Path | None = None,
    disable_pkl_matrix_source: bool = False,
    target_fnames: Iterable[str] | None = None,
    max_items: int | None = None,
    levenshtein_backend: str = BACKEND_C,
    workers: int = 1,
    doc_workers: int = 1,
    log_fn: LogFn | None = None,
) -> dict:
    """Run fixed-grid per-document tuning and build parameter influence curves."""
    run_started_at = time.perf_counter()
    log = _no_log if log_fn is None else log_fn
    baseline = baseline_cfg if baseline_cfg is not None else HoughBaselineConfig()
    load_timing: dict = {}

    resolved_index_cache_dir = (
        DEFAULT_SCORE_INDEX_CACHE_DIR if score_index_cache_dir is None else Path(score_index_cache_dir)
    )

    docs = load_documents(
        runfile_json=Path(runfile_json),
        window_size=int(window_size),
        window_stride=int(window_stride),
        hough_start=float(baseline.hough_start),
        levenshtein_backend=str(levenshtein_backend),
        matrix_cache_dir=matrix_cache_dir,
        scores_pkl_ref_to_pred=scores_pkl_ref_to_pred,
        score_index_cache_file=score_index_cache_file,
        score_index_cache_dir=resolved_index_cache_dir,
        disable_pkl_matrix_source=bool(disable_pkl_matrix_source),
        target_fnames=target_fnames,
        max_items=max_items,
        timing_out=load_timing,
        log_fn=log,
    )

    threshold_values = fixed_values_for(PARAM_HOUGH_THRESHOLD)
    line_length_values = fixed_values_for(PARAM_HOUGH_LINE_LENGTH)
    line_gap_values = fixed_values_for(PARAM_HOUGH_LINE_GAP)

    combos_per_doc = int(len(threshold_values) * len(line_length_values) * len(line_gap_values))

    requested_doc_workers = max(1, int(doc_workers))
    requested_threshold_workers = max(1, int(workers))

    # User requirement: when parallelizing across multiple documents, each document
    # should use full threshold parallelism (40 thresholds -> 40 workers).
    if requested_doc_workers > 1:
        effective_threshold_workers = len(threshold_values)
        if requested_threshold_workers != effective_threshold_workers:
            log(
                f"[grid] overriding threshold_workers from {requested_threshold_workers} "
                f"to {effective_threshold_workers} because doc_workers={requested_doc_workers} > 1"
            )
    else:
        effective_threshold_workers = requested_threshold_workers

    log(
        f"[grid] docs={len(docs)} combos_per_doc={combos_per_doc} "
        f"threshold={HOUGH_THRESHOLD_MIN}..{HOUGH_THRESHOLD_MAX} "
        f"line_length={HOUGH_LINE_LENGTH_MIN}..{HOUGH_LINE_LENGTH_MAX} "
        f"line_gap={HOUGH_LINE_GAP_MIN}..{HOUGH_LINE_GAP_MAX} seed={baseline.hough_seed} "
        f"threshold_workers={effective_threshold_workers} doc_workers={requested_doc_workers}"
    )

    sweep_result = run_document_sweeps(
        docs=docs,
        baseline_cfg=baseline,
        levenshtein_backend=str(levenshtein_backend),
        threshold_values=threshold_values,
        line_length_values=line_length_values,
        line_gap_values=line_gap_values,
        workers=int(effective_threshold_workers),
        doc_workers=int(requested_doc_workers),
        log_fn=log,
    )

    profile_points = sweep_result["profile_points"]
    doc_best_records = sweep_result["doc_best_records"]
    grid_eval_seconds = float(sweep_result["grid_eval_seconds"])
    doc_grid_seconds_total = float(sweep_result["doc_grid_seconds_total"])

    output_dir = Path(output_dir)
    csv_dir = output_dir / "csv"

    profile_started_at = time.perf_counter()
    sweeps: dict[str, dict] = {}
    per_parameter_seconds: dict[str, float] = {}
    sweep_total_seconds = 0.0
    best_means: list[float] = []

    for param in SUPPORTED_SWEEP_PARAMETERS:
        param_started_at = time.perf_counter()
        values = sorted(int(v) for v in profile_points[param].keys())

        rows: list[dict] = []
        for value in values:
            row = build_curve_row(parameter=param, value=int(value), doc_rows=profile_points[param][int(value)])
            rows.append(row)

            mean_str = "None" if row["mean_along_lines_nls"] is None else f"{row['mean_along_lines_nls']:.6f}"
            log(
                f"SWEEP {param}={value} mean_along_lines={mean_str} "
                f"valid_docs={row['valid_doc_count']}/{row['doc_count']} "
                f"t_hough_detect_s={row['timing_hough_detect_seconds']:.3f} "
                f"t_filter_s={row['timing_filter_seconds']:.3f} "
                f"t_detect_filter_s={row['timing_detect_filter_seconds']:.3f} "
                f"t_lev_s={row['timing_levenshtein_seconds']:.3f} "
                f"t_total_s={row['timing_total_seconds']:.3f}"
            )

        best = best_curve_row(rows)
        best_compact = compact_best_curve_row(best)
        if best_compact is not None and best_compact.get("mean_along_lines_nls") is not None:
            best_means.append(float(best_compact["mean_along_lines_nls"]))

        csv_path = csv_dir / f"{param}_summary.csv"
        write_parameter_curve_csv(rows=rows, output_csv=csv_path)

        sweeps[param] = {
            "parameter": str(param),
            "values": values,
            "rows": rows,
            "best": best,
            "best_compact": best_compact,
            "csv_path": str(csv_path),
            "plot_path": None,
        }

        param_seconds = float(time.perf_counter() - param_started_at)
        per_parameter_seconds[param] = param_seconds
        sweep_total_seconds += param_seconds
        log(f"[timing] sweep_parameter={param} seconds={param_seconds:.3f}")

    profile_aggregation_seconds = float(time.perf_counter() - profile_started_at)

    # Existing aggregate CSV (one row per document, best combination only).
    best_configs_csv_path = csv_dir / "best_config_per_document.csv"
    write_best_configs_csv(best_records=doc_best_records, output_csv=best_configs_csv_path)

    # New human-readable JSON artifact: best full combination per document.
    best_params_json_path = output_dir / "best_params_per_document.json"
    write_best_params_json(best_records=doc_best_records, output_json=best_params_json_path)

    # New long-format CSV artifact: all docs x all parameter values influence rows.
    influence_rows = build_parameter_influence_rows(profile_points=profile_points)
    all_docs_influence_csv_path = csv_dir / "all_documents_parameter_influence.csv"
    write_parameter_influence_csv(rows=influence_rows, output_csv=all_docs_influence_csv_path)
    log(
        f"[exports] best_params_json={best_params_json_path} "
        f"influence_csv={all_docs_influence_csv_path} rows={len(influence_rows)}"
    )

    run_total_seconds = float(time.perf_counter() - run_started_at)
    load_total_seconds = float(load_timing.get("load_documents_total_seconds", 0.0))
    non_sweep_non_load_seconds = max(
        0.0,
        run_total_seconds - load_total_seconds - grid_eval_seconds - profile_aggregation_seconds,
    )
    log(
        f"[timing] run_total_s={run_total_seconds:.3f} load_total_s={load_total_seconds:.3f} "
        f"grid_eval_s={grid_eval_seconds:.3f} profile_s={profile_aggregation_seconds:.3f} "
        f"other_s={non_sweep_non_load_seconds:.3f}"
    )

    mean_of_parameter_best_means = None
    if best_means:
        mean_of_parameter_best_means = float(sum(best_means) / len(best_means))

    summary = {
        "runfile_json": str(Path(runfile_json)),
        "output_dir": str(output_dir),
        "target_fnames": [str(v) for v in (target_fnames or [])],
        "max_items": None if max_items is None else int(max_items),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "levenshtein_backend": str(levenshtein_backend),
        "matrix_cache_dir": None if matrix_cache_dir is None else str(Path(matrix_cache_dir)),
        "scores_pkl_ref_to_pred": None if scores_pkl_ref_to_pred is None else str(Path(scores_pkl_ref_to_pred)),
        "score_index_cache_file": None if score_index_cache_file is None else str(Path(score_index_cache_file)),
        "score_index_cache_dir": None if resolved_index_cache_dir is None else str(Path(resolved_index_cache_dir)),
        "disable_pkl_matrix_source": bool(disable_pkl_matrix_source),
        "baseline": asdict(baseline),
        "doc_count": int(len(docs)),
        "doc_names": [doc.fname for doc in docs],
        "grid_ranges": {
            PARAM_HOUGH_THRESHOLD: {"min": HOUGH_THRESHOLD_MIN, "max": HOUGH_THRESHOLD_MAX},
            PARAM_HOUGH_LINE_LENGTH: {"min": HOUGH_LINE_LENGTH_MIN, "max": HOUGH_LINE_LENGTH_MAX},
            PARAM_HOUGH_LINE_GAP: {"min": HOUGH_LINE_GAP_MIN, "max": HOUGH_LINE_GAP_MAX},
            PARAM_HOUGH_SEED: {"fixed": int(baseline.hough_seed)},
        },
        "combos_per_doc": int(combos_per_doc),
        "parallelism": {
            "requested_threshold_workers_per_doc": int(requested_threshold_workers),
            "threshold_workers_per_doc": int(effective_threshold_workers),
            "doc_workers": int(requested_doc_workers),
        },
        "best_config_per_document": doc_best_records,
        "best_config_per_document_csv_path": str(best_configs_csv_path),
        "best_params_per_document_json_path": str(best_params_json_path),
        "all_documents_parameter_influence_csv_path": str(all_docs_influence_csv_path),
        "all_documents_parameter_influence_row_count": int(len(influence_rows)),
        "parameter_sweeps": sweeps,
        "timings": {
            "run_total_seconds": run_total_seconds,
            "load_documents": load_timing,
            "grid_evaluation_seconds": grid_eval_seconds,
            "doc_grid_seconds_total": doc_grid_seconds_total,
            "profile_aggregation_seconds": profile_aggregation_seconds,
            "sweep_total_seconds": float(sweep_total_seconds),
            "sweep_per_parameter_seconds": per_parameter_seconds,
            "non_sweep_non_load_seconds": float(non_sweep_non_load_seconds),
        },
        "global_summary": {
            "best_by_parameter": {param: sweeps[param].get("best_compact") for param in SUPPORTED_SWEEP_PARAMETERS},
            "mean_of_parameter_best_means": mean_of_parameter_best_means,
            "mean_best_along_lines_across_docs": best_doc_nls_mean(doc_best_records),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "hough_parameter_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
