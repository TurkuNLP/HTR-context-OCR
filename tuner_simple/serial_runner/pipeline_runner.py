from __future__ import annotations

"""Run the simple tuner over selected documents in one serial process."""

from pathlib import Path
import gc
import time
from typing import Any

from tuner_simple.config.pipeline_config import PipelineConfig
from tuner_simple.document_selection.document_filters import select_documents
from tuner_simple.document_selection.runfile_loader import load_runfile_documents
from tuner_simple.matrix_operations.matrix_loader import build_score_matrix_indexes
from tuner_simple.results_writing.flat_csv_tables import write_all_flat_outputs
from tuner_simple.serial_runner.document_runner import document_table_row, process_one_document


# Define the run_simple_tuner function; its body below performs one named step of the pipeline.
def run_simple_tuner(config: PipelineConfig, *, log) -> dict[str, Any]:
    """Execute one complete serial tuner run and return the output manifest."""
    # Compute or store run_started_at so later code can reuse this named value clearly.
    run_started_at = time.perf_counter()
    # Compute or store output_dir so later code can reuse this named value clearly.
    output_dir = Path(config.output_dir)
    # Ensure the target directory exists before later code tries to write files into it.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log("[run] simple serial tuner started")
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[run] output directory: {output_dir}")
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[run] runfile JSON: {config.runfile_json}")
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[run] ref-to-pred score pkl: {config.scores_pkl_ref_to_pred}")
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[run] ref-to-ref score pkl: {config.scores_pkl_ref_to_ref}")
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(
        f"[config] selection "
        f"languages={list(config.languages) if config.languages else 'ALL'} "
        f"document_types={list(config.document_types) if config.document_types else 'ALL'} "
        f"target_fnames={list(config.target_fnames) if config.target_fnames else 'ALL'} "
        f"max_items={config.max_items}"
    )
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(
        f"[config] matrix window_size={int(config.window_size)} "
        f"window_stride={int(config.window_stride)} "
        f"minimum_rows={int(config.minimum_matrix_rows)} "
        f"minimum_columns={int(config.minimum_matrix_columns)} "
        f"score_floor_alpha={float(config.score_floor_alpha):.6f}"
    )
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(
        f"[config] hough threshold={int(config.hough_parameters.hough_threshold)} "
        f"line_length={int(config.hough_parameters.hough_line_length)} "
        f"line_gap={int(config.hough_parameters.hough_line_gap)} "
        f"seed={int(config.hough_parameters.hough_seed)} "
        f"align_min_iou_threshold={float(config.align_min_iou_threshold):.6f} "
        f"min_surviving_line_nls={config.min_surviving_line_nls}"
    )
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(
        f"[config] plotting mode={config.plot_mode} "
        f"show_line_ids={bool(config.show_line_ids)} "
        f"stitched_columns={int(config.stitched_panel_columns)} "
        f"saved_figure_dpi={int(config.saved_figure_dpi)}"
    )

    # Compute or store runfile_started_at so later code can reuse this named value clearly.
    runfile_started_at = time.perf_counter()
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log("[runfile] load start")
    # Compute or store all_runfile_documents so later code can reuse this named value clearly.
    all_runfile_documents = load_runfile_documents(config.runfile_json)
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[runfile] load done documents={len(all_runfile_documents)} seconds={time.perf_counter() - runfile_started_at:.6f}")
    # Compute or store selection_started_at so later code can reuse this named value clearly.
    selection_started_at = time.perf_counter()
    # Compute or store selected_documents so later code can reuse this named value clearly.
    selected_documents = select_documents(
        # Pass the documents argument into the surrounding call so the callee receives that setting explicitly.
        documents=all_runfile_documents,
        # Pass languages into the surrounding call; this supplies the optional language filter requested by the user.
        languages=config.languages,
        # Pass document_types into the surrounding call; this supplies the optional document-type filter requested by the user.
        document_types=config.document_types,
        # Pass target_fnames into the surrounding call; this supplies the optional exact filename filter requested by the user.
        target_fnames=config.target_fnames,
        # Pass max_items into the surrounding call; this supplies the optional cap on how many selected documents are processed.
        max_items=config.max_items,
        # Pass the log argument into the surrounding call so the callee receives that setting explicitly.
        log=log,
    )
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[selection] final selected documents={len(selected_documents)} seconds={time.perf_counter() - selection_started_at:.6f}")
    # Compute or store runfile_rows so later code can reuse this named value clearly.
    runfile_rows = [
        # Execute this statement as the next small step in the surrounding pipeline logic.
        document_table_row(document, window_size=config.window_size, window_stride=config.window_stride)
        # Iterate over document in selected_documents so each item is processed with the same logic.
        for document in selected_documents
    ]

    # Compute or store index_started_at so later code can reuse this named value clearly.
    index_started_at = time.perf_counter()
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log("[matrix-index] build start")
    # Compute or store indexes so later code can reuse this named value clearly.
    indexes = build_score_matrix_indexes(
        # Pass the ref_to_pred_scores_pkl argument into the surrounding call so the callee receives that setting explicitly.
        ref_to_pred_scores_pkl=config.scores_pkl_ref_to_pred,
        # Pass the ref_to_ref_scores_pkl argument into the surrounding call so the callee receives that setting explicitly.
        ref_to_ref_scores_pkl=config.scores_pkl_ref_to_ref,
        # Pass the log argument into the surrounding call so the callee receives that setting explicitly.
        log=log,
    )
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[matrix-index] build done seconds={time.perf_counter() - index_started_at:.6f}")

    # Compute or store plotter so later code can reuse this named value clearly.
    plotter = None
    # Check whether config.plot_mode != "none"; the indented block handles that specific case.
    if config.plot_mode != "none":
        from tuner_simple.plotting.stitched_language_panels import SimplePlotManager

        # Compute or store plotter so later code can reuse this named value clearly.
        plotter = SimplePlotManager(config=config, log=log)
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(f"[plot] plot mode enabled: {config.plot_mode}")
    # Define the else field so this data object records that value explicitly.
    else:
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log("[plot] plot mode disabled; plotting libraries will not be imported")

    # Compute or store result_rows: list[dict[str, Any]] so later code can reuse this named value clearly.
    result_rows: list[dict[str, Any]] = []
    # Compute or store skipped_rows: list[dict[str, Any]] so later code can reuse this named value clearly.
    skipped_rows: list[dict[str, Any]] = []
    # Compute or store loadable_rows: list[dict[str, Any]] so later code can reuse this named value clearly.
    loadable_rows: list[dict[str, Any]] = []
    # Compute or store loaded_rows: list[dict[str, Any]] so later code can reuse this named value clearly.
    loaded_rows: list[dict[str, Any]] = []

    # Compute or store total_document_count so later code can reuse this named value clearly.
    total_document_count = len(selected_documents)
    # Iterate over position, document in enumerate(selected_documents, start=1) so each item is processed with the same logic.
    for position, document in enumerate(selected_documents, start=1):
        # Compute or store document_loop_started_at so later code can reuse this named value clearly.
        document_loop_started_at = time.perf_counter()
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(f"[document] {position}/{total_document_count} started: {document.fname}")
        # Compute or store document_result so later code can reuse this named value clearly.
        document_result = process_one_document(
            # Pass the document argument into the surrounding call so the callee receives that setting explicitly.
            document=document,
            # Pass the config argument into the surrounding call so the callee receives that setting explicitly.
            config=config,
            # Pass the indexes argument into the surrounding call so the callee receives that setting explicitly.
            indexes=indexes,
            # Pass the log argument into the surrounding call so the callee receives that setting explicitly.
            log=log,
            # Pass the keep_plot_payload argument into the surrounding call so the callee receives that setting explicitly.
            keep_plot_payload=plotter is not None,
        )

        # Check whether document_result.result_row is not None; the indented block handles that specific case.
        if document_result.result_row is not None:
            # Add this item to the list that is accumulating results for later output.
            result_rows.append(document_result.result_row)
        # Check whether document_result.skipped_row is not None; the indented block handles that specific case.
        if document_result.skipped_row is not None:
            # Add this item to the list that is accumulating results for later output.
            skipped_rows.append(document_result.skipped_row)
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(f"[document] skipped {document.fname}: {document_result.skipped_row.get('skip_reason')}")
        # Check whether document_result.loadable_row is not None; the indented block handles that specific case.
        if document_result.loadable_row is not None:
            # Add this item to the list that is accumulating results for later output.
            loadable_rows.append(document_result.loadable_row)
        # Check whether document_result.loaded_row is not None; the indented block handles that specific case.
        if document_result.loaded_row is not None:
            # Add this item to the list that is accumulating results for later output.
            loaded_rows.append(document_result.loaded_row)
        # Check whether plotter is not None and document_result.plot_payload is not None; the indented block handles that specific case.
        if plotter is not None and document_result.plot_payload is not None:
            # Compute or store plot_started_at so later code can reuse this named value clearly.
            plot_started_at = time.perf_counter()
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(f"[plot] render start document={document.fname}")
            # Execute this statement as the next small step in the surrounding pipeline logic.
            plotter.render_document_payload(document_result.plot_payload)
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(f"[plot] render done document={document.fname} seconds={time.perf_counter() - plot_started_at:.6f}")

        # Release the local reference to document_result so large intermediate data can be freed sooner.
        del document_result
        # Ask Python to reclaim unreachable objects after a document has finished processing.
        gc.collect()
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(
            f"[document] {position}/{total_document_count} finished: {document.fname} "
            f"seconds={time.perf_counter() - document_loop_started_at:.6f}"
        )

    # Compute or store stitched_plot_paths: list[str] so later code can reuse this named value clearly.
    stitched_plot_paths: list[str] = []
    # Check whether plotter is not None; the indented block handles that specific case.
    if plotter is not None:
        # Compute or store stitched_started_at so later code can reuse this named value clearly.
        stitched_started_at = time.perf_counter()
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log("[plot] stitched language plots start")
        # Compute or store stitched_plot_paths so later code can reuse this named value clearly.
        stitched_plot_paths = [str(path) for path in plotter.finish()]
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(f"[plot] stitched language plots done count={len(stitched_plot_paths)} seconds={time.perf_counter() - stitched_started_at:.6f}")

    # Compute or store elapsed_seconds so later code can reuse this named value clearly.
    elapsed_seconds = time.perf_counter() - run_started_at
    # Compute or store run_summary so later code can reuse this named value clearly.
    run_summary = {
        # Add the schema_version field to the surrounding dictionary so it appears in outputs or returned metadata.
        "schema_version": "tuner_simple_run_summary_v1",
        # Add the elapsed_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "elapsed_seconds": elapsed_seconds,
        # Add the selected_document_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "selected_document_count": len(selected_documents),
        # Add the processed_document_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "processed_document_count": len(result_rows),
        # Add the skipped_document_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "skipped_document_count": len(skipped_rows),
        # Add the output_dir field to the surrounding dictionary so it appears in outputs or returned metadata.
        "output_dir": str(output_dir),
        # Add the runfile_json field to the surrounding dictionary so it appears in outputs or returned metadata.
        "runfile_json": str(config.runfile_json),
        # Add the scores_pkl_ref_to_pred field to the surrounding dictionary so it appears in outputs or returned metadata.
        "scores_pkl_ref_to_pred": str(config.scores_pkl_ref_to_pred),
        # Add the scores_pkl_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "scores_pkl_ref_to_ref": str(config.scores_pkl_ref_to_ref),
        # Add the score_floor_formula field to the surrounding dictionary so it appears in outputs or returned metadata.
        "score_floor_formula": "score_mean + score_floor_alpha * score_standard_deviation",
        # Add the score_floor_alpha field to the surrounding dictionary so it appears in outputs or returned metadata.
        "score_floor_alpha": float(config.score_floor_alpha),
        # Add the hough_threshold field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_threshold": int(config.hough_parameters.hough_threshold),
        # Add the hough_line_length field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_line_length": int(config.hough_parameters.hough_line_length),
        # Add the hough_line_gap field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_line_gap": int(config.hough_parameters.hough_line_gap),
        # Add the hough_seed field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_seed": int(config.hough_parameters.hough_seed),
        # Add the minimum_matrix_rows field to the surrounding dictionary so it appears in outputs or returned metadata.
        "minimum_matrix_rows": int(config.minimum_matrix_rows),
        # Add the minimum_matrix_columns field to the surrounding dictionary so it appears in outputs or returned metadata.
        "minimum_matrix_columns": int(config.minimum_matrix_columns),
        # Add the align_min_iou_threshold field to the surrounding dictionary so it appears in outputs or returned metadata.
        "align_min_iou_threshold": float(config.align_min_iou_threshold),
        # Add the min_surviving_line_nls field to the surrounding dictionary so it appears in outputs or returned metadata.
        "min_surviving_line_nls": config.min_surviving_line_nls,
        # Add the plot_mode field to the surrounding dictionary so it appears in outputs or returned metadata.
        "plot_mode": str(config.plot_mode),
        # Add the show_line_ids field to the surrounding dictionary so it appears in outputs or returned metadata.
        "show_line_ids": bool(config.show_line_ids),
        # Add the stitched_plot_paths field to the surrounding dictionary so it appears in outputs or returned metadata.
        "stitched_plot_paths": stitched_plot_paths,
    }

    # Compute or store output_started_at so later code can reuse this named value clearly.
    output_started_at = time.perf_counter()
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log("[output] flat files write start")
    # Compute or store output_paths so later code can reuse this named value clearly.
    output_paths = write_all_flat_outputs(
        # Pass output_dir into the surrounding call; this supplies the directory where CSV, JSON, and optional plot files will be written.
        output_dir=output_dir,
        # Pass the runfile_rows argument into the surrounding call so the callee receives that setting explicitly.
        runfile_rows=runfile_rows,
        # Pass the loadable_rows argument into the surrounding call so the callee receives that setting explicitly.
        loadable_rows=loadable_rows,
        # Pass the loaded_rows argument into the surrounding call so the callee receives that setting explicitly.
        loaded_rows=loaded_rows,
        # Pass the skipped_rows argument into the surrounding call so the callee receives that setting explicitly.
        skipped_rows=skipped_rows,
        # Pass the result_rows argument into the surrounding call so the callee receives that setting explicitly.
        result_rows=result_rows,
        # Pass the run_summary argument into the surrounding call so the callee receives that setting explicitly.
        run_summary=run_summary,
    )
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[output] flat files write done file_count={len(output_paths)} seconds={time.perf_counter() - output_started_at:.6f}")
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[run] finished in {elapsed_seconds:.3f} seconds")
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[run] processed={len(result_rows)} skipped={len(skipped_rows)}")
    # Iterate over label, path in sorted(output_paths.items()) so each item is processed with the same logic.
    for label, path in sorted(output_paths.items()):
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(f"[output] {label}: {path}")
    # Return this computed value to the caller so the next pipeline stage can use it.
    return {"run_summary": run_summary, "output_paths": output_paths}


__all__ = ["run_simple_tuner"]
