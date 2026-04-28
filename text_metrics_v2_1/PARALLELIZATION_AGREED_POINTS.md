# Parallelization Agreed Points (text_metrics_v2_1)

## Scope
Parallelize at the document level only: one full pipeline per document in a worker process.
No metric-internal parallelization.

## CLI
- Add `--workers N` to:
  - `run_text_metrics_report.sh`
  - `text_metrics_report.py`
- Default remains sequential (`workers=1`).

## Worker Policy
- Hard fail if requested `--workers` exceeds available CPUs on the host.
- Preserve original input order in final outputs (not completion order).

## Output Files (single files only)
Keep exactly one of each:
- `report.json` (successfully processed items only)
- `report_skipped_empty_prediction.json`
- `report_failed_items.json`
- `report_timings.json` only when `--debug`

## Existing Skip/Failure Behavior
- Empty prediction documents are skipped and recorded.
- Non-empty documents that error are recorded as failed and do not stop the run.

## Memory Goals
- Keep per-document heavy data (score matrices, intermediate arrays) in worker-local scope only.
- Do not retain per-document matrices after document pipeline is done.
- Parent/orchestrator should avoid storing all full document payloads in memory for long runs.

## Portability
- Avoid environment-specific hardcoding.
- Logic must run across different environments (local/server/cluster) with the same code path.
