from __future__ import annotations

"""Custom module entrypoint for Churro commands.

Why this file exists:
- You wanted a command surface that is *not* `python -m churro.cli ...`.
- You still wanted to integrate with the existing repository implementation.

Design choice:
- This module delegates directly to `churro.cli.main.main`.
- That preserves behavior/parity with upstream benchmark/infer logic while giving
  you a custom module name for invocation.

Usage:
- `python3 -m custom_python_script benchmark ...`
- `python3 -m custom_python_script infer ...`
"""

import sys
from collections.abc import Sequence
from datetime import datetime, timezone
import os
from pathlib import Path

from churro.cli.main import main as churro_main
from variable_audit import VariableAuditTracer


def main(argv: Sequence[str] | None = None) -> int:
    """Run Churro commands through the custom module name.

    We forward argv as-is to the existing Typer-based CLI handler so this entrypoint
    remains a thin compatibility layer rather than a forked implementation.
    """
    # Typer expects a mutable list of argv tokens.
    # We normalize any Sequence to list so callers can pass tuples/tests safely.
    cli_args = list(argv) if argv is not None else list(sys.argv[1:])

    # Enable deep variable audit automatically for benchmark runs.
    # This traces project-local Python files and writes a single Markdown report
    # next to README in the repository root.
    repo_root = Path(__file__).resolve().parents[1]
    report_path = Path(
        os.environ.get(
            "CHURRO_VARIABLE_AUDIT_REPORT_PATH",
            str(repo_root / "VARIABLE_AUDIT_REPORT.md"),
        )
    ).resolve()
    should_audit = bool(cli_args) and cli_args[0] == "benchmark"

    tracer: VariableAuditTracer | None = None
    prev_exact_io_enabled = os.environ.get("CHURRO_EXACT_IO_LOG")
    prev_exact_io_path = os.environ.get("CHURRO_EXACT_IO_LOG_PATH")
    prev_exact_io_prefix = os.environ.get("CHURRO_EXACT_IO_LOG_PREFIX")
    prev_exact_io_raw = os.environ.get("CHURRO_EXACT_IO_INCLUDE_RAW_RESPONSE")
    exact_io_log_path: Path | None = None
    exact_io_log_prefix: Path | None = None
    if should_audit:
        tracer = VariableAuditTracer(
            repo_root=repo_root,
            output_path=report_path,
            command_tokens=cli_args,
        )
        tracer.start()
        # Benchmark-only exact vLLM I/O capture:
        # - enables request/response logging inside utils.llm.core
        # - keeps this behavior out of standalone scripts (e.g. custom_churro_infer.py)
        os.environ["CHURRO_EXACT_IO_LOG"] = "1"
        # Keep logs compact by default: omit full raw response JSON unless explicitly requested.
        os.environ.setdefault("CHURRO_EXACT_IO_INCLUDE_RAW_RESPONSE", "0")
        if prev_exact_io_path:
            exact_io_log_path = Path(prev_exact_io_path).resolve()
        else:
            run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            exact_io_log_path = repo_root / "logs" / f"BENCHMARK_VLLM_EXACT_IO_{run_stamp}.jsonl"
            os.environ["CHURRO_EXACT_IO_LOG_PATH"] = str(exact_io_log_path)
        if prev_exact_io_prefix:
            exact_io_log_prefix = Path(prev_exact_io_prefix).resolve()
        else:
            # Split exact I/O logs by content type:
            #   *_summary.jsonl, *_request_meta.jsonl, *_request_messages.jsonl,
            #   *_response_meta.jsonl, *_response_text.jsonl, *_errors.jsonl
            # (and *_response_raw.jsonl when CHURRO_EXACT_IO_INCLUDE_RAW_RESPONSE=1)
            exact_io_log_prefix = repo_root / "logs" / f"BENCHMARK_VLLM_EXACT_IO_{run_stamp}"
            os.environ["CHURRO_EXACT_IO_LOG_PREFIX"] = str(exact_io_log_prefix)

    try:
        return churro_main(cli_args)
    finally:
        # Restore environment to avoid leaking benchmark-only logging into other modes.
        if prev_exact_io_enabled is None:
            os.environ.pop("CHURRO_EXACT_IO_LOG", None)
        else:
            os.environ["CHURRO_EXACT_IO_LOG"] = prev_exact_io_enabled
        if prev_exact_io_path is None:
            os.environ.pop("CHURRO_EXACT_IO_LOG_PATH", None)
        else:
            os.environ["CHURRO_EXACT_IO_LOG_PATH"] = prev_exact_io_path
        if prev_exact_io_prefix is None:
            os.environ.pop("CHURRO_EXACT_IO_LOG_PREFIX", None)
        else:
            os.environ["CHURRO_EXACT_IO_LOG_PREFIX"] = prev_exact_io_prefix
        if prev_exact_io_raw is None:
            os.environ.pop("CHURRO_EXACT_IO_INCLUDE_RAW_RESPONSE", None)
        else:
            os.environ["CHURRO_EXACT_IO_INCLUDE_RAW_RESPONSE"] = prev_exact_io_raw

        if tracer is not None:
            tracer.stop()
            try:
                generated_path = tracer.write_report()
                print(f"[audit] Variable report written to: {generated_path}")
                print(f"[audit] Per-script variable reports dir: {tracer.per_script_reports_dir}")
                if exact_io_log_prefix is not None:
                    print(f"[audit] Exact vLLM I/O log prefix: {exact_io_log_prefix}")
                    print(
                        "[audit] Exact vLLM I/O files: "
                        f"{exact_io_log_prefix}_summary.jsonl, "
                        f"{exact_io_log_prefix}_request_meta.jsonl, "
                        f"{exact_io_log_prefix}_request_messages.jsonl, "
                        f"{exact_io_log_prefix}_response_meta.jsonl, "
                        f"{exact_io_log_prefix}_response_text.jsonl, "
                        f"{exact_io_log_prefix}_errors.jsonl"
                    )
                elif exact_io_log_path is not None:
                    print(f"[audit] Exact vLLM I/O log written to: {exact_io_log_path}")
            except Exception as exc:
                # Do not mask benchmark failures with report serialization issues.
                print(
                    f"[audit] Failed to write variable audit report: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    # Delegate command-line arguments after module name. Example:
    #   python3 -m custom_python_script benchmark --system finetuned ...
    raise SystemExit(main(sys.argv[1:]))
