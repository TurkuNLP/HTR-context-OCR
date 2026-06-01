from __future__ import annotations

"""Top-level orchestration for Hough parameter tuning.

This module stays thin and delegates detailed logic to smaller modules:
- document preparation and matrix source layering live in ``matrices/``
- exhaustive scheduling lives in ``sweep_scheduler.py``
- curve aggregation lives in ``sweep_aggregation.py``
- output serialization lives in ``outputs/``
"""

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Callable, Iterable

try:
    from ..runtime.runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime.runtime_paths import ensure_tuner_runtime_paths  # type: ignore

ensure_tuner_runtime_paths()

try:
    from ..matrices.document_prep import (
        iter_prepared_documents_from_items as _iter_prepared_documents_from_items,
        load_documents as _load_documents,
        select_run_items_for_tuning as _select_run_items_for_tuning,
    )
    from ..metrics.v2_12_metric_adapter import configure_text_metrics_v212_dir, get_v212_metric_functions
    from ..hough_preprocessing import HoughPreprocessingConfig
    from ..cache.ref_to_ref_combo_cache import RefToRefCombinationCache
    from ..outputs.tuner_invalid_exports import write_invalid_combinations_csv
    from ..outputs.tuner_combination_score_exports import CombinationScoreTableWriter
    from ..outputs.tuner_outputs import write_best_configs_csv, write_parameter_curve_csv, write_skipped_documents_csv
    from ..outputs.tuner_profile_exports import write_combination_profile_csv
    from ..outputs.tuner_result_exports import (
        build_parameter_influence_rows,
        write_best_params_json,
        write_parameter_influence_csv,
    )
    from ..outputs.tuner_summary_export import build_public_tuner_summary
    from .hough_eval import DEFAULT_SELECTION_OBJECTIVE, SUPPORTED_SELECTION_OBJECTIVES, normalize_selection_objective
    from .sweep_aggregation import (
        best_curve_row,
        best_doc_nls_mean,
        best_doc_tuning_score_mean,
        build_curve_row,
        compact_best_curve_row,
    )
    from .sweep_grid import build_sweep_values, combinations_per_document
    from .ref_to_ref_cache_warmup import warm_ref_to_ref_cache_for_documents
    from .sweep_scheduler import run_document_sweeps
    from .tuner_config import (
        DEFAULT_SCORE_INDEX_CACHE_DIR,
        DEFAULT_REF_TO_REF_COMBO_CACHE_DIR,
        DEFAULT_TEXT_METRICS_V212_DIR,
        HoughBaselineConfig,
        HoughSweepRanges,
        LogFn,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SUPPORTED_SWEEP_PARAMETERS,
        default_hough_sweep_ranges,
        fixed_parameter_ranges as _fixed_parameter_ranges,
    )
except ImportError:
    from matrices.document_prep import (  # type: ignore
        iter_prepared_documents_from_items as _iter_prepared_documents_from_items,
        load_documents as _load_documents,
        select_run_items_for_tuning as _select_run_items_for_tuning,
    )
    from metrics.v2_12_metric_adapter import configure_text_metrics_v212_dir, get_v212_metric_functions  # type: ignore
    from hough_preprocessing import HoughPreprocessingConfig  # type: ignore
    from cache.ref_to_ref_combo_cache import RefToRefCombinationCache  # type: ignore
    from outputs.tuner_invalid_exports import write_invalid_combinations_csv  # type: ignore
    from outputs.tuner_combination_score_exports import CombinationScoreTableWriter  # type: ignore
    from outputs.tuner_outputs import write_best_configs_csv, write_parameter_curve_csv, write_skipped_documents_csv  # type: ignore
    from outputs.tuner_profile_exports import write_combination_profile_csv  # type: ignore
    from outputs.tuner_result_exports import (  # type: ignore
        build_parameter_influence_rows,
        write_best_params_json,
        write_parameter_influence_csv,
    )
    from outputs.tuner_summary_export import build_public_tuner_summary  # type: ignore
    from tuner.hough_eval import DEFAULT_SELECTION_OBJECTIVE, SUPPORTED_SELECTION_OBJECTIVES, normalize_selection_objective  # type: ignore
    from tuner.sweep_aggregation import (  # type: ignore
        best_curve_row,
        best_doc_nls_mean,
        best_doc_tuning_score_mean,
        build_curve_row,
        compact_best_curve_row,
    )
    from tuner.sweep_grid import build_sweep_values, combinations_per_document  # type: ignore
    from tuner.ref_to_ref_cache_warmup import warm_ref_to_ref_cache_for_documents  # type: ignore
    from tuner.sweep_scheduler import run_document_sweeps  # type: ignore
    from tuner.tuner_config import (  # type: ignore
        DEFAULT_SCORE_INDEX_CACHE_DIR,
        DEFAULT_REF_TO_REF_COMBO_CACHE_DIR,
        DEFAULT_TEXT_METRICS_V212_DIR,
        HoughBaselineConfig,
        HoughSweepRanges,
        LogFn,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SUPPORTED_SWEEP_PARAMETERS,
        default_hough_sweep_ranges,
        fixed_parameter_ranges as _fixed_parameter_ranges,
    )

try:
    from ..metrics.levenshtein_compat import BACKEND_C, BACKEND_PYTHON, SUPPORTED_BACKENDS
except ImportError:
    from metrics.levenshtein_compat import BACKEND_C, BACKEND_PYTHON, SUPPORTED_BACKENDS  # type: ignore


def _no_log(_: str) -> None:
    """No-op logger used when callers omit a log hook."""
    return


def _write_public_summary_json(*, summary: dict, summary_path: Path) -> dict:
    """Write the human-facing summary JSON and return the same public payload."""
    public_summary = build_public_tuner_summary(summary)
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(public_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    public_summary["summary_path"] = str(summary_path)
    return public_summary


def fixed_parameter_ranges() -> dict[str, tuple[int, int]]:
    """Public accessor for the default sweep ranges."""
    return _fixed_parameter_ranges()


def load_documents(
    *,
    runfile_json: Path,
    window_size: int,
    window_stride: int,
    levenshtein_backend: str,
    matrix_cache_dir: Path | None = None,
    scores_pkl_ref_to_pred: Path | None = None,
    scores_pkl_ref_to_ref: Path | None = None,
    score_index_cache_file: Path | None = None,
    score_index_cache_file_ref_to_ref: Path | None = None,
    score_index_cache_dir: Path | None = None,
    disable_pkl_matrix_source: bool = False,
    target_fnames: Iterable[str] | None = None,
    max_items: int | None = None,
    selection_index_range: tuple[int, int] | None = None,
    timing_out: dict | None = None,
    on_document_skipped: Callable[[dict], None] | None = None,
    skip_diagnostic_bundle_dir: Path | None = None,
    hough_preprocessing_config: HoughPreprocessingConfig | None = None,
    log_fn: LogFn | None = None,
):
    """Prepare document payloads once so the sweep can reuse expensive state."""
    return _load_documents(
        runfile_json=Path(runfile_json),
        window_size=int(window_size),
        window_stride=int(window_stride),
        levenshtein_backend=str(levenshtein_backend),
        matrix_cache_dir=matrix_cache_dir,
        scores_pkl_ref_to_pred=scores_pkl_ref_to_pred,
        scores_pkl_ref_to_ref=scores_pkl_ref_to_ref,
        score_index_cache_file=score_index_cache_file,
        score_index_cache_file_ref_to_ref=score_index_cache_file_ref_to_ref,
        score_index_cache_dir=score_index_cache_dir,
        disable_pkl_matrix_source=bool(disable_pkl_matrix_source),
        target_fnames=target_fnames,
        max_items=max_items,
        selection_index_range=selection_index_range,
        timing_out=timing_out,
        on_document_skipped=on_document_skipped,
        skip_diagnostic_bundle_dir=skip_diagnostic_bundle_dir,
        hough_preprocessing_config=hough_preprocessing_config,
        log_fn=log_fn,
    )


def run_hough_parameter_sweeps(
    *,
    runfile_json: Path,
    output_dir: Path,
    window_size: int,
    window_stride: int,
    baseline_cfg: HoughBaselineConfig | None = None,
    hough_sweep_ranges: HoughSweepRanges | None = None,
    matrix_cache_dir: Path | None = None,
    scores_pkl_ref_to_pred: Path | None = None,
    scores_pkl_ref_to_ref: Path | None = None,
    score_index_cache_file: Path | None = None,
    score_index_cache_file_ref_to_ref: Path | None = None,
    score_index_cache_dir: Path | None = None,
    disable_pkl_matrix_source: bool = False,
    text_metrics_v212_dir: Path | None = None,
    ref_to_ref_cache_mode: str = "auto",
    ref_to_ref_cache_dir: Path | None = None,
    ref_to_ref_cache_warm_only: bool = False,
    target_fnames: Iterable[str] | None = None,
    max_items: int | None = None,
    selection_index_range: tuple[int, int] | None = None,
    combination_bundle_dir: Path | None = None,
    combination_bundle_scope: str = "none",
    combination_bundle_include_candidate_lines: bool = False,
    shard_index: int | None = None,
    levenshtein_backend: str = BACKEND_C,
    workers: int = 1,
    doc_workers: int = 1,
    selected_run_items_override: Iterable[dict] | None = None,
    selected_document_count_override: int | None = None,
    on_document_completed: Callable[[object, dict], None] | None = None,
    on_document_skipped: Callable[[dict], None] | None = None,
    hough_preprocessing_config: HoughPreprocessingConfig | None = None,
    min_surviving_line_nls: float | None = None,
    profile_combinations: bool = False,
    selection_objective: str = DEFAULT_SELECTION_OBJECTIVE,
    write_combination_score_table: bool = True,
    log_fn: LogFn | None = None,
) -> dict:
    """Run per-document tuning and build summary/CSV/JSON artifacts."""
    run_started_at = time.perf_counter()
    log = _no_log if log_fn is None else log_fn
    baseline = baseline_cfg if baseline_cfg is not None else HoughBaselineConfig()
    active_hough_preprocessing_config = (
        HoughPreprocessingConfig() if hough_preprocessing_config is None else hough_preprocessing_config
    )
    active_selection_objective = normalize_selection_objective(selection_objective)
    active_ranges = default_hough_sweep_ranges() if hough_sweep_ranges is None else hough_sweep_ranges
    load_timing: dict = {}
    if min_surviving_line_nls is not None and not (0.0 <= float(min_surviving_line_nls) <= 1.0):
        raise ValueError("min_surviving_line_nls must be between 0.0 and 1.0 inclusive")
    if min_surviving_line_nls is not None:
        log(
            "[line-nls-filter] "
            f"min_surviving_line_nls={float(min_surviving_line_nls):.6f} "
            "scope=ref_to_pred_final_lines_after_geometry_filtering"
        )
    log(
        "[selection-objective] "
        f"active={active_selection_objective} supported={list(SUPPORTED_SELECTION_OBJECTIVES)}"
    )

    resolved_combination_bundle_dir = None
    if str(combination_bundle_scope) != "none":
        resolved_combination_bundle_dir = (
            Path(output_dir) / "combination_bundles"
            if combination_bundle_dir is None
            else Path(combination_bundle_dir)
        )

    resolved_index_cache_dir = (
        DEFAULT_SCORE_INDEX_CACHE_DIR if score_index_cache_dir is None else Path(score_index_cache_dir)
    )
    resolved_v212_dir = DEFAULT_TEXT_METRICS_V212_DIR if text_metrics_v212_dir is None else Path(text_metrics_v212_dir)
    resolved_ref_to_ref_cache_dir = (
        DEFAULT_REF_TO_REF_COMBO_CACHE_DIR if ref_to_ref_cache_dir is None else Path(ref_to_ref_cache_dir)
    )

    # Import and validate v2.12 metric functions before worker threads start so
    # the hot path never races on sys.path/sys.modules manipulation.
    configure_text_metrics_v212_dir(resolved_v212_dir)
    v212_functions = get_v212_metric_functions()
    log(
        f"[v2.12-metrics] dir={v212_functions.text_metrics_v212_dir} "
        f"bundle={v212_functions.line_metric_bundle_path} "
        f"coverage={v212_functions.line_coverage_subtract_path}"
    )
    ref_to_ref_cache = RefToRefCombinationCache(
        cache_dir=resolved_ref_to_ref_cache_dir,
        mode=str(ref_to_ref_cache_mode),
    )
    log(
        f"[ref-to-ref-cache] mode={ref_to_ref_cache.mode} "
        f"enabled={ref_to_ref_cache.enabled} dir={resolved_ref_to_ref_cache_dir}"
    )

    if selected_run_items_override is None:
        selected_run_items = _select_run_items_for_tuning(
            runfile_json=Path(runfile_json),
            target_fnames=target_fnames,
            max_items=max_items,
            selection_index_range=selection_index_range,
        )
        selected_doc_count = int(len(selected_run_items))
    else:
        # Dynamic-pool workers provide a lazy iterator that claims exactly one
        # document whenever the scheduler has a free local document slot.  The
        # preparation and sweep logic below are intentionally reused unchanged.
        selected_run_items = selected_run_items_override
        selected_doc_count = 0 if selected_document_count_override is None else int(selected_document_count_override)

    prepared_document_stream = _iter_prepared_documents_from_items(
        selected_run_items=selected_run_items,
        window_size=int(window_size),
        window_stride=int(window_stride),
        levenshtein_backend=str(levenshtein_backend),
        matrix_cache_dir=matrix_cache_dir,
        scores_pkl_ref_to_pred=scores_pkl_ref_to_pred,
        scores_pkl_ref_to_ref=scores_pkl_ref_to_ref,
        score_index_cache_file=score_index_cache_file,
        score_index_cache_file_ref_to_ref=score_index_cache_file_ref_to_ref,
        score_index_cache_dir=resolved_index_cache_dir,
        disable_pkl_matrix_source=bool(disable_pkl_matrix_source),
        prepare_ref_to_pred_artifacts=not bool(ref_to_ref_cache_warm_only),
        # Dynamic-pool workers may legitimately start after faster workers have
        # already claimed every document.  In that case the iterator should end
        # cleanly instead of turning an empty pool into a worker failure.
        raise_when_no_documents_selected=selected_run_items_override is None,
        timing_out=load_timing,
        on_document_skipped=on_document_skipped,
        skip_diagnostic_bundle_dir=resolved_combination_bundle_dir,
        hough_preprocessing_config=active_hough_preprocessing_config,
        log_fn=log,
    )

    sweep_values = build_sweep_values(active_ranges)
    threshold_values = sweep_values[PARAM_HOUGH_THRESHOLD]
    line_length_values = sweep_values[PARAM_HOUGH_LINE_LENGTH]
    line_gap_values = sweep_values[PARAM_HOUGH_LINE_GAP]
    seed_values = sweep_values[PARAM_HOUGH_SEED]
    combos_per_doc = combinations_per_document(active_ranges)

    requested_doc_workers = max(1, int(doc_workers))
    requested_threshold_workers = max(1, int(workers))

    if requested_doc_workers > 1:
        effective_threshold_workers = len(threshold_values)
        if requested_threshold_workers != effective_threshold_workers:
            log(
                f"[grid] overriding threshold_workers from {requested_threshold_workers} "
                f"to {effective_threshold_workers} because doc_workers={requested_doc_workers} > 1 "
                f"and active threshold_count={len(threshold_values)}"
            )
    else:
        effective_threshold_workers = requested_threshold_workers

    log(
        f"[grid] docs={selected_doc_count} combos_per_doc={combos_per_doc} "
        f"active_grid={active_ranges.active_grid_label()} "
        f"threshold_workers={effective_threshold_workers} doc_workers={requested_doc_workers}"
    )

    combination_bundle_logger = None
    combination_score_writer = None
    combination_score_table_path = Path(output_dir) / "combination_scores.csv.gz"
    if bool(write_combination_score_table) and not bool(ref_to_ref_cache_warm_only):
        combination_score_writer = CombinationScoreTableWriter(output_csv_gz=combination_score_table_path)
        log(f"[combination-scores] enabled path={combination_score_table_path}")

    if str(combination_bundle_scope) != "none":
        try:
            from ..outputs.combination_bundle_logger import CombinationBundleLogger
        except ImportError:
            from outputs.combination_bundle_logger import CombinationBundleLogger  # type: ignore

        combination_bundle_logger = CombinationBundleLogger(
            root_dir=resolved_combination_bundle_dir,
            scope=str(combination_bundle_scope),
            include_candidate_lines=bool(combination_bundle_include_candidate_lines),
            shard_index=shard_index,
            selection_index_range=selection_index_range,
            log_fn=log,
        )
        log(
            f"[combination-bundles] enabled root={resolved_combination_bundle_dir} "
            f"scope={combination_bundle_scope} "
            f"include_candidate_lines={bool(combination_bundle_include_candidate_lines)}"
        )

    if bool(ref_to_ref_cache_warm_only):
        warm_started_at = time.perf_counter()
        warm_summary = warm_ref_to_ref_cache_for_documents(
            docs=prepared_document_stream,
            total_docs=selected_doc_count,
            baseline_cfg=baseline,
            threshold_values=threshold_values,
            line_length_values=line_length_values,
            line_gap_values=line_gap_values,
            seed_values=seed_values,
            workers=int(effective_threshold_workers),
            doc_workers=int(requested_doc_workers),
            ref_to_ref_cache=ref_to_ref_cache,
            log_fn=log,
        )
        if hasattr(ref_to_ref_cache, "close"):
            # Warm-only runs also queue document-level cache writes; wait here so
            # the summary reports final cache counters and all files are durable.
            ref_to_ref_cache.close()
        run_total_seconds = float(time.perf_counter() - run_started_at)
        warm_seconds = float(time.perf_counter() - warm_started_at)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        skipped_document_records = list(load_timing.get("skipped_documents", []))
        skipped_documents_csv_path = output_dir / "csv" / "skipped_documents.csv"
        write_skipped_documents_csv(
            skipped_records=skipped_document_records,
            output_csv=skipped_documents_csv_path,
        )
        summary = {
            "runfile_json": str(Path(runfile_json)),
            "output_dir": str(output_dir),
            "mode": "ref_to_ref_cache_warm_only",
            "selected_doc_count": int(selected_doc_count),
            "target_fnames": [str(value) for value in (target_fnames or [])],
            "max_items": None if max_items is None else int(max_items),
            "selection_index_range": (
                None if selection_index_range is None else [int(selection_index_range[0]), int(selection_index_range[1])]
            ),
            "window_size": int(window_size),
            "window_stride": int(window_stride),
            "levenshtein_backend": str(levenshtein_backend),
            "matrix_cache_dir": None if matrix_cache_dir is None else str(Path(matrix_cache_dir)),
            "scores_pkl_ref_to_pred": None if scores_pkl_ref_to_pred is None else str(Path(scores_pkl_ref_to_pred)),
            "scores_pkl_ref_to_ref": None if scores_pkl_ref_to_ref is None else str(Path(scores_pkl_ref_to_ref)),
            "score_index_cache_dir": None if resolved_index_cache_dir is None else str(Path(resolved_index_cache_dir)),
            "disable_pkl_matrix_source": bool(disable_pkl_matrix_source),
            "hough_preprocessing": active_hough_preprocessing_config.as_dict(),
            "text_metrics_v212_dir": str(Path(v212_functions.text_metrics_v212_dir)),
            "requested_text_metrics_v212_dir": str(Path(resolved_v212_dir)),
            "metric_backend": {
                "name": "tuner_local_v2_12_compat",
                "semantics": "v2_12_line_coverage_exact",
                "external_dependency_required": False,
                "line_metric_bundle_path": str(Path(v212_functions.line_metric_bundle_path)),
                "line_coverage_subtract_path": str(Path(v212_functions.line_coverage_subtract_path)),
            },
            "ref_to_ref_cache": ref_to_ref_cache.stats.as_dict(),
            "baseline": asdict(baseline),
            "doc_count": int(selected_doc_count),
            "prepared_doc_count": int(load_timing.get("prepared_document_count", warm_summary.document_count)),
            "skipped_document_count": int(len(skipped_document_records)),
            "skipped_document_reason_counts": dict(load_timing.get("skipped_document_reason_counts", {})),
            "skipped_documents_csv_path": str(skipped_documents_csv_path),
            "grid_ranges": active_ranges.as_summary_dict(),
            "active_grid_label": active_ranges.active_grid_label(),
            "selection_objective": active_selection_objective,
            "combos_per_doc": int(combos_per_doc),
            "parallelism": {
                "scheduler_mode": "ref_to_ref_cache_warmup_global_threshold_queue",
                "requested_threshold_workers_per_doc": int(requested_threshold_workers),
                "threshold_workers_per_doc": int(effective_threshold_workers),
                "doc_workers": int(requested_doc_workers),
            },
            "ref_to_ref_cache_warmup": warm_summary.as_dict(),
            "timings": {
                "run_total_seconds": float(run_total_seconds),
                "load_documents": load_timing,
                "ref_to_ref_cache_warmup_seconds": float(warm_seconds),
            },
        }
        summary_path = output_dir / "hough_parameter_sweep_summary.json"
        return _write_public_summary_json(summary=summary, summary_path=summary_path)

    try:
        sweep_result = run_document_sweeps(
            docs=prepared_document_stream,
            total_docs=selected_doc_count,
            baseline_cfg=baseline,
            levenshtein_backend=str(levenshtein_backend),
            threshold_values=threshold_values,
            line_length_values=line_length_values,
            line_gap_values=line_gap_values,
            seed_values=seed_values,
            workers=int(effective_threshold_workers),
            doc_workers=int(requested_doc_workers),
            ref_to_ref_cache=ref_to_ref_cache,
            log_fn=log,
            combination_bundle_logger=combination_bundle_logger,
            combination_score_writer=combination_score_writer,
            min_surviving_line_nls=min_surviving_line_nls,
            profile_combinations=bool(profile_combinations),
            on_document_completed=on_document_completed,
            selection_objective=active_selection_objective,
        )
    finally:
        if combination_score_writer is not None:
            combination_score_writer.close()
        if combination_bundle_logger is not None:
            combination_bundle_logger.close()
        if hasattr(ref_to_ref_cache, "close"):
            # Ref-to-ref document cache writes run in the background after each
            # document finishes.  Close before exporting summaries so cache
            # stats and files reflect the completed run exactly.
            ref_to_ref_cache.close()

    profile_points = sweep_result["profile_points"]
    doc_best_records = sweep_result["doc_best_records"]
    skipped_document_records = list(load_timing.get("skipped_documents", []))
    invalid_combination_records = list(sweep_result.get("invalid_combination_records", []))
    combination_profile_records = list(sweep_result.get("combination_profile_records", []))
    invalid_combination_count = int(sweep_result.get("invalid_combination_count", len(invalid_combination_records)))
    invalid_y_diff_le_minus_one_total = int(sweep_result.get("invalid_y_diff_le_minus_one_total", 0))
    invalid_y_diff_lt_minus_one_total = int(sweep_result.get("invalid_y_diff_lt_minus_one_total", 0))
    line_nls_filter_all_removed_combination_count = int(
        sweep_result.get("line_nls_filter_all_removed_combination_count", 0)
    )
    grid_eval_seconds = float(sweep_result["grid_eval_seconds"])
    doc_grid_seconds_total = float(sweep_result["doc_grid_seconds_total"])
    evaluated_combination_count_total = int(sweep_result.get("evaluated_combination_count_total", 0))
    calculation_timing_sums_total = dict(sweep_result.get("calculation_timing_sums_total", {}))
    calculation_seconds_per_combination = dict(sweep_result.get("calculation_seconds_per_combination", {}))
    scheduler_mode = str(sweep_result.get("scheduler_mode", "serial_documents"))
    combination_score_table_summary = (
        {
            "enabled": False,
            "csv_gz_path": None,
            "row_count": 0,
            "write_seconds": 0.0,
            "field_count": 0,
            "format": "csv.gz",
        }
        if combination_score_writer is None
        else combination_score_writer.summary()
    )

    output_dir = Path(output_dir)
    csv_dir = output_dir / "csv"
    skipped_documents_csv_path = csv_dir / "skipped_documents.csv"
    write_skipped_documents_csv(
        skipped_records=skipped_document_records,
        output_csv=skipped_documents_csv_path,
    )

    profile_started_at = time.perf_counter()
    sweeps: dict[str, dict] = {}
    per_parameter_seconds: dict[str, float] = {}
    sweep_total_seconds = 0.0
    best_mean_scores: list[float] = []

    for param in SUPPORTED_SWEEP_PARAMETERS:
        param_started_at = time.perf_counter()
        values = sorted(int(value) for value in profile_points[param].keys())
        rows: list[dict] = []

        for value in values:
            row = build_curve_row(parameter=param, value=int(value), doc_rows=profile_points[param][int(value)])
            rows.append(row)
            mean_score = row.get("mean_tuning_score")
            mean_score_str = "None" if mean_score is None else f"{float(mean_score):.6f}"
            log(
                f"SWEEP {param}={value} mean_tuning_score={mean_score_str} "
                f"valid_docs={row['valid_doc_count']}/{row['doc_count']} "
                f"t_detect_refpred_s={row['timing_hough_detect_ref_to_pred_seconds']:.3f} "
                f"t_filter_refpred_s={row['timing_filter_ref_to_pred_seconds']:.3f} "
                f"t_detect_refref_s={row['timing_hough_detect_ref_to_ref_seconds']:.3f} "
                f"t_filter_refref_s={row['timing_filter_ref_to_ref_seconds']:.3f} "
                f"t_bundle_s={row['timing_build_bundle_seconds']:.3f} "
                f"t_coverage_s={row['timing_coverage_seconds']:.3f} "
                f"t_lev_s={row['timing_levenshtein_seconds']:.3f} "
                f"t_total_s={row['timing_total_seconds']:.3f}"
            )

        best = best_curve_row(rows)
        best_compact = compact_best_curve_row(best)
        if best_compact is not None and best_compact.get("mean_tuning_score") is not None:
            best_mean_scores.append(float(best_compact["mean_tuning_score"]))

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

    best_configs_csv_path = csv_dir / "best_config_per_document.csv"
    write_best_configs_csv(best_records=doc_best_records, output_csv=best_configs_csv_path)

    best_params_json_path = output_dir / "best_params_per_document.json"
    write_best_params_json(best_records=doc_best_records, output_json=best_params_json_path)

    influence_rows = build_parameter_influence_rows(profile_points=profile_points)
    all_docs_influence_csv_path = csv_dir / "all_documents_parameter_influence.csv"
    write_parameter_influence_csv(rows=influence_rows, output_csv=all_docs_influence_csv_path)

    invalid_combinations_csv_path = csv_dir / "invalid_combinations.csv"
    write_invalid_combinations_csv(rows=invalid_combination_records, output_csv=invalid_combinations_csv_path)
    combination_profile_csv_path = None
    if bool(profile_combinations):
        combination_profile_csv_path = output_dir / "combination_profile.csv"
        write_combination_profile_csv(
            rows=combination_profile_records,
            output_csv=combination_profile_csv_path,
        )
    log(
        f"[exports] best_params_json={best_params_json_path} "
        f"influence_csv={all_docs_influence_csv_path} rows={len(influence_rows)} "
        f"invalid_csv={invalid_combinations_csv_path} invalid_rows={len(invalid_combination_records)} "
        f"score_table={combination_score_table_summary.get('csv_gz_path')} "
        f"score_rows={combination_score_table_summary.get('row_count')} "
        f"profile_csv={combination_profile_csv_path if combination_profile_csv_path is not None else 'disabled'} "
        f"profile_rows={len(combination_profile_records) if bool(profile_combinations) else 0} "
        f"skipped_documents_csv={skipped_documents_csv_path} "
        f"skipped_documents={len(skipped_document_records)}"
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

    mean_of_parameter_best_scores = None
    if best_mean_scores:
        mean_of_parameter_best_scores = float(sum(best_mean_scores) / len(best_mean_scores))

    documents_with_any_line_nls_all_removed = [
        record
        for record in doc_best_records
        if int(record.get("line_nls_filter_all_removed_combination_count", 0) or 0) > 0
    ]
    documents_with_best_line_nls_all_removed = [
        record
        for record in doc_best_records
        if bool(record.get("best", {}).get("line_nls_filter_all_lines_removed", False))
    ]
    documents_with_all_combinations_line_nls_all_removed = [
        record
        for record in doc_best_records
        if int(record.get("evaluated_combination_count", 0) or 0) > 0
        and int(record.get("line_nls_filter_all_removed_combination_count", 0) or 0)
        == int(record.get("evaluated_combination_count", 0) or 0)
    ]

    summary = {
        "runfile_json": str(Path(runfile_json)),
        "output_dir": str(output_dir),
        "selected_doc_count": int(selected_doc_count),
        "target_fnames": [str(value) for value in (target_fnames or [])],
        "max_items": None if max_items is None else int(max_items),
        "selection_index_range": (
            None if selection_index_range is None else [int(selection_index_range[0]), int(selection_index_range[1])]
        ),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "levenshtein_backend": str(levenshtein_backend),
        "matrix_cache_dir": None if matrix_cache_dir is None else str(Path(matrix_cache_dir)),
        "scores_pkl_ref_to_pred": None if scores_pkl_ref_to_pred is None else str(Path(scores_pkl_ref_to_pred)),
        "scores_pkl_ref_to_ref": None if scores_pkl_ref_to_ref is None else str(Path(scores_pkl_ref_to_ref)),
        "score_index_cache_file": None if score_index_cache_file is None else str(Path(score_index_cache_file)),
        "score_index_cache_file_ref_to_ref": (
            None if score_index_cache_file_ref_to_ref is None else str(Path(score_index_cache_file_ref_to_ref))
        ),
        "score_index_cache_dir": None if resolved_index_cache_dir is None else str(Path(resolved_index_cache_dir)),
        "disable_pkl_matrix_source": bool(disable_pkl_matrix_source),
        "hough_preprocessing": active_hough_preprocessing_config.as_dict(),
        "line_nls_filter": {
            "enabled": min_surviving_line_nls is not None,
            "min_surviving_line_nls": None if min_surviving_line_nls is None else float(min_surviving_line_nls),
            "scope": "ref_to_pred_final_lines_after_geometry_filtering",
            "all_lines_removed_combination_count": int(line_nls_filter_all_removed_combination_count),
            "documents_with_any_all_lines_removed_combination_count": int(
                len(documents_with_any_line_nls_all_removed)
            ),
            "documents_where_best_combination_removed_all_lines_count": int(
                len(documents_with_best_line_nls_all_removed)
            ),
            "documents_where_all_combinations_removed_all_lines_count": int(
                len(documents_with_all_combinations_line_nls_all_removed)
            ),
        },
        "text_metrics_v212_dir": str(Path(v212_functions.text_metrics_v212_dir)),
        "requested_text_metrics_v212_dir": str(Path(resolved_v212_dir)),
        "metric_backend": {
            "name": "tuner_local_v2_12_compat",
            "semantics": "v2_12_line_coverage_exact",
            "external_dependency_required": False,
            "line_metric_bundle_path": str(Path(v212_functions.line_metric_bundle_path)),
            "line_coverage_subtract_path": str(Path(v212_functions.line_coverage_subtract_path)),
        },
        "ref_to_ref_cache_warm_only": bool(ref_to_ref_cache_warm_only),
        "ref_to_ref_cache": ref_to_ref_cache.stats.as_dict(),
        "metric_objective": "harmonic_mean_weighted_nls_correct_ref_coverage_non_hallucination_v1",
        "baseline": asdict(baseline),
        "doc_count": int(len(doc_best_records)),
        "prepared_doc_count": int(load_timing.get("prepared_document_count", len(doc_best_records))),
        "skipped_document_count": int(len(skipped_document_records)),
        "skipped_document_reason_counts": dict(load_timing.get("skipped_document_reason_counts", {})),
        "skipped_documents_csv_path": str(skipped_documents_csv_path),
        "doc_names": [str(record["fname"]) for record in doc_best_records],
        "grid_ranges": active_ranges.as_summary_dict(),
        "active_grid_label": active_ranges.active_grid_label(),
        "selection_objective": active_selection_objective,
        "combination_bundle_logging": {
            "enabled": bool(combination_bundle_logger is not None),
            "root_dir": None if resolved_combination_bundle_dir is None else str(Path(resolved_combination_bundle_dir)),
            "scope": str(combination_bundle_scope),
            "include_candidate_lines": bool(combination_bundle_include_candidate_lines),
            "record_format": "pickle_stream",
            "shard_index": None if shard_index is None else int(shard_index),
        },
        "combos_per_doc": int(combos_per_doc),
        "parallelism": {
            "scheduler_mode": str(scheduler_mode),
            "requested_threshold_workers_per_doc": int(requested_threshold_workers),
            "threshold_workers_per_doc": int(effective_threshold_workers),
            "doc_workers": int(requested_doc_workers),
        },
        "best_config_per_document": doc_best_records,
        "best_config_per_document_csv_path": str(best_configs_csv_path),
        "best_params_per_document_json_path": str(best_params_json_path),
        "all_documents_parameter_influence_csv_path": str(all_docs_influence_csv_path),
        "all_documents_parameter_influence_row_count": int(len(influence_rows)),
        "invalid_combinations_csv_path": str(invalid_combinations_csv_path),
        "combination_score_table": combination_score_table_summary,
        "combination_profiling": {
            "enabled": bool(profile_combinations),
            "csv_path": None if combination_profile_csv_path is None else str(combination_profile_csv_path),
            "row_count": int(len(combination_profile_records)) if bool(profile_combinations) else 0,
        },
        "invalid_combination_count": int(invalid_combination_count),
        "invalid_y_diff_le_minus_one_total": int(invalid_y_diff_le_minus_one_total),
        "invalid_y_diff_lt_minus_one_total": int(invalid_y_diff_lt_minus_one_total),
        "invalid_combination_examples": invalid_combination_records[:20],
        "parameter_sweeps": sweeps,
        "timings": {
            "run_total_seconds": run_total_seconds,
            "load_documents": load_timing,
            "grid_evaluation_seconds": grid_eval_seconds,
            "doc_grid_seconds_total": doc_grid_seconds_total,
            "calculation_only": {
                "description": (
                    "Sum of evaluator timing fields from completed combinations. "
                    "This excludes document bundle writing, cache writing, final CSV/JSON exports, and final visuals."
                ),
                "evaluated_combination_count": int(evaluated_combination_count_total),
                "timing_sums_seconds": calculation_timing_sums_total,
                "seconds_per_combination": calculation_seconds_per_combination,
            },
            "profile_aggregation_seconds": profile_aggregation_seconds,
            "sweep_total_seconds": float(sweep_total_seconds),
            "sweep_per_parameter_seconds": per_parameter_seconds,
            "non_sweep_non_load_seconds": float(non_sweep_non_load_seconds),
        },
        "global_summary": {
            "best_by_parameter": {param: sweeps[param].get("best_compact") for param in SUPPORTED_SWEEP_PARAMETERS},
            "mean_of_parameter_best_tuning_scores": mean_of_parameter_best_scores,
            "mean_best_tuning_score_across_docs": best_doc_tuning_score_mean(doc_best_records),
            "mean_best_weighted_along_lines_across_docs": best_doc_nls_mean(doc_best_records),
            "invalid_combination_count": int(invalid_combination_count),
            "invalid_y_diff_le_minus_one_total": int(invalid_y_diff_le_minus_one_total),
            "invalid_y_diff_lt_minus_one_total": int(invalid_y_diff_lt_minus_one_total),
            "line_nls_filter_all_removed_combination_count": int(
                line_nls_filter_all_removed_combination_count
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "hough_parameter_sweep_summary.json"
    return _write_public_summary_json(summary=summary, summary_path=summary_path)


__all__ = [
    "BACKEND_C",
    "BACKEND_PYTHON",
    "SUPPORTED_BACKENDS",
    "DEFAULT_SCORE_INDEX_CACHE_DIR",
    "DEFAULT_REF_TO_REF_COMBO_CACHE_DIR",
    "DEFAULT_TEXT_METRICS_V212_DIR",
    "HoughBaselineConfig",
    "HoughSweepRanges",
    "fixed_parameter_ranges",
    "load_documents",
    "run_hough_parameter_sweeps",
]
