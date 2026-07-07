#!/usr/bin/env python3
"""layout_v2 runner: stream the Churro split, run pass 0 → gate → pass 1 (+ pass 2), derive, write.

Per document (see IMPLEMENTATION_PLAN.md §2):
  1. pass 0 (full page)            -> document_category + independent parts (the shared frame)
  2. band crops cut per the parts  -> pass 1 (full page + bands): per-part column enumeration
  3. category gate                 -> pass 2 (full page): articles/advertisements/entries
  4. harness                       -> validators + derivations + needs_review
  5. one self-describing record    -> responses/<idx>_<slug>.json, aggregated to layout_outputs.json

Operational behaviour mirrors the proven old driver (bounded concurrency, targeted retries,
run-numbered output dirs, run_config.json snapshot, per-doc isolation: one bad image never kills
the run) — re-implemented here, not imported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run from anywhere (sbatch spool dirs)
import config  # noqa: E402
import imaging  # noqa: E402
from client import call_pass  # noqa: E402
from derive import derive_document  # noqa: E402
from gold_ref import extract_gold_fields, gold_line_stats  # noqa: E402
from passes import pass0_structure, pass1_columns, pass2_items  # noqa: E402
from validate import validate_pass1, validate_pass2  # noqa: E402


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Every knob has a default from config.py so the launcher stays a thin env-var shim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=config.DATASET_ID)
    parser.add_argument("--dataset-revision", default=config.DATASET_REVISION)
    parser.add_argument("--dataset-split", default="dev", choices=["all", *config.DATASET_SPLITS])
    parser.add_argument("--max-samples-per-split", type=int, default=0, help="0 = full split.")
    parser.add_argument(
        "--only-basenames", default="",
        help="Comma-separated image basenames (with or without extension) to process — used for "
        "fixture and bake-off runs. Empty = everything.",
    )
    # vLLM endpoint.
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="layoutv2", help="Must match --served-model-name.")
    parser.add_argument("--model-repo", default="", help="Exact HF repo id, recorded for attribution.")
    parser.add_argument("--vllm-api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--vllm-timeout-seconds", type=int, default=1200)
    # Sampling (decision #6: thinking-style guidance; identical across passes).
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    # Concurrency / retries.
    parser.add_argument("--max-concurrency", type=int, default=8, help="Documents in flight.")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    # Bookkeeping.
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results" / "layout_v2_run",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Resume: skip docs with a record.")
    parser.add_argument("--skip-pass2", action="store_true", help="Columns only (bake-off arms).")
    # Bake-off arms (plan §11.2). Defaults are the production configuration; the non-default
    # values exist so bakeoff runs are ordinary runs with one flag flipped (fully recorded in
    # run_config.json like everything else).
    parser.add_argument(
        "--pass1-anchor-mode", default="dual", choices=["dual", "x_only", "text_only"],
        help="Column anchor format arm: dual (default), positional-only, or text-only.",
    )
    parser.add_argument(
        "--pass1-input", default="full_bands", choices=["full_bands", "full_only", "bands_only"],
        help="Input strategy arm for pass 1: full page + bands (default), or either alone.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _slugify(value: str) -> str:
    """Filesystem-safe token from a filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "sample"


def resolve_next_run_path(path: Path) -> Path:
    """Pick a non-clobbering ``..._runN`` sibling (same behaviour as the proven old pipeline)."""
    match = re.match(r"^(.*_run)(\d+)$", str(path))
    if match:
        prefix, requested = match.group(1), int(match.group(2))
    elif str(path).endswith("_run"):
        prefix, requested = str(path), 1
    else:
        prefix, requested = f"{path}_run", 1
    parent, name = Path(prefix).parent, Path(prefix).name
    max_existing = 0
    if parent.exists():
        for candidate in parent.iterdir():  # find the highest existing ``<name><digits>`` sibling
            if candidate.is_dir() and candidate.name.startswith(name):
                suffix = candidate.name[len(name):]
                if suffix.isdigit():
                    max_existing = max(max_existing, int(suffix))
    requested_path = Path(f"{prefix}{requested}")
    if max_existing < requested or (max_existing == requested and requested_path.is_dir() and not any(requested_path.iterdir())):
        return requested_path
    return Path(f"{prefix}{max_existing + 1}")


def resolve_model_commit(repo: str) -> str:
    """Best-effort resolution of the locally cached snapshot hash for ``repo`` (attribution).

    Looks in ``$HF_HOME/hub/models--org--name/snapshots/``; returns "" when unresolvable so the
    runner works offline and never fails on bookkeeping.
    """
    if not repo:
        return ""
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_name = "models--" + repo.replace("/", "--")
    snapshots = Path(hf_home) / "hub" / cache_name / "snapshots"
    try:
        hashes = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
        return hashes[-1] if hashes else ""
    except OSError:
        return ""


def _basename_filter(arg: str) -> set[str]:
    """Normalize --only-basenames into a set of extensionless stems."""
    stems = set()
    for token in arg.split(","):
        token = token.strip()
        if token:
            stems.add(Path(token).stem)
    return stems


def _gold_metadata(example: dict, dataset_id: str) -> dict[str, str]:
    """The dataset's own labels — analysis context, never model input."""
    return {
        "main_language": str(example.get("main_language") or "unknown"),
        "main_script": str(example.get("main_script") or "unknown"),
        "document_type": str(example.get("document_type") or "unknown").lower(),
        "dataset_id": str(example.get("dataset_id") or dataset_id),
    }


# --------------------------------------------------------------------------------------
# Per-document pipeline
# --------------------------------------------------------------------------------------
def _run_pass(args: argparse.Namespace, *, system: str, user_parts: list[dict],
              schema: dict, max_tokens: int) -> dict:
    """Thin adapter: one pass call with the run's shared connection/sampling settings."""
    return call_pass(
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
        model=args.vllm_model,
        system=system,
        user_parts=user_parts,
        json_schema=schema,
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_seconds=args.vllm_timeout_seconds,
        max_attempts=args.max_attempts,
        backoff_seconds=args.retry_backoff_seconds,
    )


def _strip_bulky(pass_result: dict) -> dict:
    """Pass result as stored in the record: keep everything except the parsed duplicate.

    ``parsed`` is re-stored postprocessed under its own key by the caller; keeping the raw parse
    too would double every record for no analytical gain.
    """
    slim = dict(pass_result)
    slim.pop("parsed", None)
    return slim


def _finished_by_length(pass_result: dict) -> bool:
    """True when vLLM stopped because the generation cap was exhausted."""
    metadata = pass_result.get("response_metadata") or {}
    return metadata.get("finish_reason") == "length"


def _length_retry_note(pass_result: dict, *, from_tokens: int, to_tokens: int) -> dict:
    """Small audit note for a targeted larger retry, without copying raw content twice."""
    usage = (pass_result.get("response_metadata") or {}).get("usage") or {}
    return {
        "from_max_tokens": from_tokens,
        "to_max_tokens": to_tokens,
        "previous_parse_error": pass_result.get("parse_error", ""),
        "previous_finish_reason": (pass_result.get("response_metadata") or {}).get("finish_reason"),
        "previous_completion_tokens": usage.get("completion_tokens"),
    }


def process_example(*, sample_index: int, example: dict, split: str, args: argparse.Namespace) -> dict:
    """The full three-pass pipeline for one document; never raises (records its own failures)."""
    file_name = str(example.get("file_name") or f"{split}_{sample_index:06d}")
    record: dict[str, Any] = {
        "split": split,
        "sample_index": sample_index,
        "file_name": file_name,
        "model": args.vllm_model,
        "model_repo": args.model_repo,
        "run_label": args.run_label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pass0": None,
        "pass1": None,
        "pass2": None,
        "pass0_parsed": None,
        "pass1_parsed": None,
        "pass2_parsed": None,
        "validation_issues": [],
        "derived": None,
        "error": "",
    }
    record.update(_gold_metadata(example, args.dataset_id))
    record.update(extract_gold_fields(example.get("transcription")))   # reference text fields
    record.update(gold_line_stats(example.get("transcription")))       # model-free density covariates

    native = None
    try:
        # ---- Images: native once; full page capped by area; bands cut after pass 0 -----
        native = imaging.dataset_image_to_pil(example["image"])
        record["native_size"] = list(native.size)
        full = imaging.cap_area(native, config.MAX_FULL_PAGE_MP)
        record["processed_full_size"] = list(full.size)  # the resolution audit reads this
        full_url = imaging.pil_to_data_url(full)
        if full is not native:
            full.close()

        # ---- Pass 0: category + parts ---------------------------------------------------
        result0 = _run_pass(
            args,
            system=pass0_structure.SYSTEM_PROMPT,
            user_parts=pass0_structure.build_user_parts(full_url),
            schema=pass0_structure.SCHEMA,
            max_tokens=config.PASS0_MAX_TOKENS,
        )
        record["pass0"] = _strip_bulky(result0)
        if not result0["ok"]:
            # Without the frame nothing downstream is well-posed; derive a review-flagged stub.
            record["derived"] = derive_document(
                category="unknown", pass0_parts=[], pass1=None, pass2=None,
                validation_issues=["pass0 failed: " + (result0["error"] or result0["parse_error"])],
            )
            return record
        parsed0 = pass0_structure.postprocess(result0["parsed"])
        record["pass0_parsed"] = parsed0
        category = parsed0["document_category"]
        parts0 = parsed0["parts"]

        # ---- Pass 1: columns (input strategy per the selected arm) ----------------------
        band_meta: list[dict] = []
        if args.pass1_input != "full_only":
            bands = imaging.make_bands(native, parts0)  # native-resolution bands: max gutter pixels
            band_meta = [
                {"data_url": imaging.pil_to_data_url(b["image"]),
                 "top_frac": b["top_frac"], "bottom_frac": b["bottom_frac"]}
                for b in bands
            ]
            for band in bands:
                band["image"].close()
        record["band_ranges"] = [[b["top_frac"], b["bottom_frac"]] for b in band_meta]
        pass1_full_url = None if args.pass1_input == "bands_only" else full_url

        pass1_system = pass1_columns.get_prompt(args.pass1_anchor_mode)
        pass1_user_parts = pass1_columns.build_user_parts(pass1_full_url, band_meta, parts0)
        pass1_schema = pass1_columns.get_schema(args.pass1_anchor_mode)
        result1 = _run_pass(
            args,
            system=pass1_system,
            user_parts=pass1_user_parts,
            schema=pass1_schema,
            max_tokens=config.PASS1_MAX_TOKENS,
        )
        if not result1["ok"] and _finished_by_length(result1):
            retry_note = _length_retry_note(
                result1,
                from_tokens=config.PASS1_MAX_TOKENS,
                to_tokens=config.PASS1_LENGTH_RETRY_MAX_TOKENS,
            )
            result1 = _run_pass(
                args,
                system=pass1_system,
                user_parts=pass1_user_parts,
                schema=pass1_schema,
                max_tokens=config.PASS1_LENGTH_RETRY_MAX_TOKENS,
            )
            result1["length_retry"] = retry_note
        record["pass1"] = _strip_bulky(result1)
        parsed1 = None
        issues: list[str] = []
        if result1["ok"]:
            parsed1 = pass1_columns.postprocess(result1["parsed"])
            record["pass1_parsed"] = parsed1
            issues.extend(validate_pass1(parsed1, parts0))
        else:
            issues.append("pass1 failed: " + (result1["error"] or result1["parse_error"]))

        # ---- Pass 2: items — only where the question is well-posed (category gate) ------
        parsed2 = None
        requested = pass2_items.groups_for_category(category)
        if requested and not args.skip_pass2:
            columns_per_part = {
                p["part_index"]: len(p.get("columns") or [])
                for p in (parsed1.get("parts") if parsed1 else []) or []
                if isinstance(p.get("part_index"), int)
            }
            pass2_user_parts = pass2_items.build_user_parts(full_url, parts0, requested, columns_per_part)
            result2 = _run_pass(
                args,
                system=pass2_items.SYSTEM_PROMPT,
                user_parts=pass2_user_parts,
                schema=pass2_items.SCHEMA,
                max_tokens=config.PASS2_MAX_TOKENS,
            )
            if not result2["ok"] and _finished_by_length(result2):
                retry_note = _length_retry_note(
                    result2,
                    from_tokens=config.PASS2_MAX_TOKENS,
                    to_tokens=config.PASS2_LENGTH_RETRY_MAX_TOKENS,
                )
                result2 = _run_pass(
                    args,
                    system=pass2_items.SYSTEM_PROMPT,
                    user_parts=pass2_user_parts,
                    schema=pass2_items.SCHEMA,
                    max_tokens=config.PASS2_LENGTH_RETRY_MAX_TOKENS,
                )
                result2["length_retry"] = retry_note
            record["pass2"] = _strip_bulky(result2)
            if result2["ok"]:
                parsed2 = pass2_items.postprocess(result2["parsed"])
                record["pass2_parsed"] = parsed2
                issues.extend(validate_pass2(parsed2, parts0, requested))
            else:
                issues.append("pass2 failed: " + (result2["error"] or result2["parse_error"]))

        # ---- Harness: everything numeric is computed here, not asked --------------------
        record["validation_issues"] = issues
        record["derived"] = derive_document(
            category=category, pass0_parts=parts0, pass1=parsed1, pass2=parsed2,
            validation_issues=issues,
        )
    except Exception as exc:  # noqa: BLE001 — per-doc isolation: record and continue
        record["error"] = str(exc)
        if record["derived"] is None:
            record["derived"] = derive_document(
                category="unknown", pass0_parts=[], pass1=None, pass2=None,
                validation_issues=[f"exception: {exc}"],
            )
    finally:
        if native is not None:
            native.close()  # release decoded pixels promptly under concurrency
    return record


# --------------------------------------------------------------------------------------
# Summary over a finished split
# --------------------------------------------------------------------------------------
def summarize(records: list[dict]) -> dict:
    """Run-level distributions over the NEW primitives (replaces the old field summaries)."""
    derived = [r["derived"] for r in records if isinstance(r.get("derived"), dict)]

    def pass_ok(name: str) -> int:
        return sum(1 for r in records if isinstance(r.get(name), dict) and r[name].get("ok"))

    def mean_usage(name: str, key: str) -> float:
        values = [
            (r.get(name) or {}).get("response_metadata", {}).get("usage", {}).get(key)
            for r in records
        ]
        values = [v for v in values if isinstance(v, (int, float))]
        return round(sum(values) / len(values), 1) if values else 0.0

    dominant_by_category: dict[str, Counter] = {}
    for d in derived:
        count = d.get("column_count_dominant")
        if isinstance(count, int):
            dominant_by_category.setdefault(d["document_category"], Counter())[str(count)] += 1

    verdicts = Counter(
        part["verdict"] for d in derived for part in d.get("parts", []) if isinstance(part, dict)
    )
    review_reasons = Counter()
    for d in derived:
        for reason in d.get("needs_review_reasons", []):
            review_reasons[reason.split(":", 1)[0][:60]] += 1  # bucket by reason prefix

    return {
        "n_total": len(records),
        "pass0_ok": pass_ok("pass0"),
        "pass1_ok": pass_ok("pass1"),
        "pass2_ok": pass_ok("pass2"),
        "category_distribution": dict(Counter(d["document_category"] for d in derived).most_common()),
        "parts_histogram": dict(Counter(str(d["independent_parts"]) for d in derived).most_common()),
        "column_dominant_by_category": {k: dict(v.most_common()) for k, v in dominant_by_category.items()},
        "reconciliation_verdicts": dict(verdicts.most_common()),
        "needs_review_count": sum(1 for d in derived if d.get("needs_review")),
        "needs_review_reason_buckets": dict(review_reasons.most_common(20)),
        "vertical_script_count": sum(1 for d in derived if d.get("vertical_script")),
        "mean_usage": {
            name: {
                "prompt_tokens": mean_usage(name, "prompt_tokens"),
                "completion_tokens": mean_usage(name, "completion_tokens"),
            }
            for name in ("pass0", "pass1", "pass2")
        },
    }


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------
def run_split(split: str, args: argparse.Namespace, output_dir: Path) -> dict[str, int]:
    """Process one split with bounded concurrency; write per-doc records + aggregate files."""
    split_dir = output_dir / split
    responses_dir = split_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    wanted_stems = _basename_filter(args.only_basenames)

    print(f"[run] split={split} model={args.vllm_model} repo={args.model_repo} label={args.run_label!r}", flush=True)
    stream = load_dataset(args.dataset_id, split=split, streaming=True, revision=args.dataset_revision)

    records: list[dict] = []
    counts = {"seen": 0, "ok": 0, "fail": 0, "skipped": 0}
    start = time()

    def handle(record: dict) -> None:
        """Persist one finished record and log a one-line status."""
        stem = Path(record["file_name"]).stem or f"sample_{record['sample_index']:06d}"
        path = responses_dir / f"{record['sample_index']:06d}_{_slugify(stem)}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
        derived = record.get("derived") or {}
        healthy = not record.get("error") and (record.get("pass0") or {}).get("ok") and (record.get("pass1") or {}).get("ok")
        counts["ok" if healthy else "fail"] += 1
        flag = " REVIEW" if derived.get("needs_review") else ""
        print(
            f"[{'ok' if healthy else 'fail'}] {split} idx={record['sample_index']} {record['file_name']} "
            f"cat={derived.get('document_category')} parts={derived.get('independent_parts')} "
            f"cols={derived.get('column_count_dominant')}{flag}",
            flush=True,
        )

    # Bounded concurrency: at most max_concurrency documents in flight; each worker performs its
    # 2-3 HTTP calls sequentially, so the server sees a healthy mixed batch.
    pending: set[Future] = set()
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        for sample_index, example in enumerate(stream):
            if args.max_samples_per_split and counts["seen"] >= args.max_samples_per_split:
                break
            if not isinstance(example, dict):
                continue
            stem = Path(str(example.get("file_name") or "")).stem
            if wanted_stems and stem not in wanted_stems:
                continue  # fixture/bake-off mode: only the requested pages
            counts["seen"] += 1
            if args.skip_existing:
                existing = responses_dir / f"{sample_index:06d}_{_slugify(stem)}.json"
                if existing.exists():
                    counts["skipped"] += 1
                    continue
            pending.add(executor.submit(
                process_example, sample_index=sample_index, example=example, split=split, args=args
            ))
            while len(pending) >= args.max_concurrency:  # backpressure: drain the first finisher
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    handle(future.result())
        for future in pending:  # drain the tail after the stream ends
            handle(future.result())

    records.sort(key=lambda r: r["sample_index"])
    summary = summarize(records)
    summary.update({"split": split, "elapsed_seconds": round(time() - start, 1),
                    "model": args.vllm_model, "model_repo": args.model_repo,
                    "run_label": args.run_label})
    (split_dir / "layout_outputs.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (split_dir / "layout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {split} seen={counts['seen']} ok={counts['ok']} fail={counts['fail']} "
          f"skipped={counts['skipped']} -> {split_dir}", flush=True)
    return counts


def main() -> int:
    args = parse_args()
    output_dir = resolve_next_run_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot exactly how this run was configured, including the resolved model commit — the
    # old pipeline's unattributable "qwen3vl" alias is the failure this line exists to prevent.
    snapshot = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    snapshot["structured_output_mode"] = os.getenv("LAYOUT_V2_STRUCTURED_OUTPUT", "response_format")
    snapshot["model_commit"] = resolve_model_commit(args.model_repo)
    snapshot["config"] = {  # the numeric policy this run ran under, frozen into the artifact
        "MAX_FULL_PAGE_MP": config.MAX_FULL_PAGE_MP,
        "N_BANDS": config.N_BANDS,
        "BAND_HEIGHT_FRAC": config.BAND_HEIGHT_FRAC,
        "PASS_MAX_TOKENS": [config.PASS0_MAX_TOKENS, config.PASS1_MAX_TOKENS, config.PASS2_MAX_TOKENS],
        "PASS1_LENGTH_RETRY_MAX_TOKENS": config.PASS1_LENGTH_RETRY_MAX_TOKENS,
        "PASS2_LENGTH_RETRY_MAX_TOKENS": config.PASS2_LENGTH_RETRY_MAX_TOKENS,
        "ITEM_ENUM_MAX": config.ITEM_ENUM_MAX,
        "TEMPERATURE": config.TEMPERATURE,
        "TOP_P": config.TOP_P,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    splits = list(config.DATASET_SPLITS) if args.dataset_split == "all" else [args.dataset_split]
    total_fail = 0
    for split in splits:
        total_fail += run_split(split, args, output_dir)["fail"]
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
