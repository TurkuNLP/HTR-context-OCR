from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (str(_SCRIPT_DIR),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from line_alignment_pipeline import detect_and_filter_lines_from_matrix
from levenshtein_metric import (
    BACKEND_PYTHON,
    SUPPORTED_BACKENDS,
    compute_levenshtein_metrics_from_bundle,
)
from line_coverage_subtract import compute_line_coverage_percentage_metrics_from_bundles
from line_filtering_v2_1_IoU import DEFAULT_MIN_IOU_THRESHOLD
from line_metric_bundle import build_line_metric_bundle
from runfile_records import load_run_items, safe_name, same_file
from score_matrix_builder import coerce_score_matrix, compute_score_matrix
from score_stream_index import (
    build_score_stream_index_cached,
    load_run_items_from_score_index,
    load_score_item_by_offset,
)

KIND_REF_TO_PRED = "ref_to_pred"
KIND_REF_TO_REF = "ref_to_ref"
KIND_REF_TO_ADJUSTED_PRED = "ref_to_adjusted_pred"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Align prediction text with fixed-diagonal probabilistic Hough lines "
            "from runfile JSON and/or precomputed scores.pkl matrices."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runfile-json", type=Path, default=None, help="Optional path to outputs.json")

    # Backward-compatible legacy alias.
    p.add_argument(
        "--scores-pkl",
        type=Path,
        default=None,
        help="Legacy alias for --scores-pkl-ref-to-pred.",
    )

    p.add_argument(
        "--scores-pkl-ref-to-pred",
        type=Path,
        default=None,
        help="Optional ref->pred scores.pkl stream.",
    )
    p.add_argument(
        "--scores-pkl-ref-to-ref",
        type=Path,
        default=None,
        help="Optional ref->ref scores.pkl stream.",
    )
    p.add_argument(
        "--scores-pkl-ref-to-adjusted-pred",
        type=Path,
        default=None,
        help="Optional ref->adjusted-pred scores.pkl stream.",
    )
    p.add_argument(
        "--scores-pkl-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing compare subfolders ref_to_pred, ref_to_ref, "
            "and ref_to_adjusted_pred with .pkl files."
        ),
    )

    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--window-size", type=int, default=100, help="Sliding window size")
    p.add_argument("--window-stride", type=int, default=50, help="Sliding window stride")
    p.add_argument("--target-fname", type=str, default=None, help="Optional exact/basename target file")
    p.add_argument("--max-items", type=int, default=None, help="Optional maximum processed items")
    p.add_argument(
        "--skip-visuals",
        dest="skip_visuals",
        action="store_true",
        default=True,
        help="Skip before-Hough, after-Hough-line-transform, after-filtering, and after-reordering PNG outputs.",
    )
    p.add_argument(
        "--with-visuals",
        dest="skip_visuals",
        action="store_false",
        help="Generate before-Hough, after-Hough-line-transform, after-filtering, and after-reordering PNG outputs.",
    )

    p.add_argument("--hough-threshold", type=int, default=26, help="Hough vote threshold")
    p.add_argument("--hough-line-length", type=int, default=10, help="Minimum accepted line length")
    p.add_argument("--hough-line-gap", type=int, default=15, help="Maximum gap to connect line pixels")
    p.add_argument("--hough-seed", type=int, default=0, help="Base random seed")
    p.add_argument("--hough-start", type=float, default=2.6, help="Initial adaptive threshold start before decrement loop")
    p.add_argument(
        "--align-abs-min-len",
        type=float,
        default=8.0,
        help="Absolute minimum line length kept before ownership resolution.",
    )
    p.add_argument(
        "--align-min-iou-threshold",
        type=float,
        default=DEFAULT_MIN_IOU_THRESHOLD,
        help="Minimum true-IoU threshold used to merge overlapping line coverages in v2.1_true_IoU.",
    )
    p.add_argument(
        "--levenshtein-backend",
        type=str,
        default=BACKEND_PYTHON,
        choices=tuple(SUPPORTED_BACKENDS),
        help="Levenshtein backend. 'python' keeps current implementation; 'c' uses exact C-backed distance.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Write run-level report_timings.json with per-document timing telemetry.",
    )
    return p.parse_args()


def _discover_scores_pkl(
    root: Path,
    *,
    subdir: str,
    stem_hint: str,
    window_size: int,
    window_stride: int,
) -> Path | None:
    sub = root / subdir
    if not sub.exists() or not sub.is_dir():
        return None

    candidates = sorted(p for p in sub.glob("*.pkl") if p.is_file())
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    exact = sub / f"{stem_hint}_ws{int(window_size)}_st{int(window_stride)}.pkl"
    if exact.exists():
        return exact

    ws_st = f"ws{int(window_size)}_st{int(window_stride)}"
    ws_candidates = [p for p in candidates if ws_st in p.name]
    if len(ws_candidates) == 1:
        return ws_candidates[0]

    hinted = [p for p in ws_candidates if stem_hint in p.name]
    if len(hinted) == 1:
        return hinted[0]

    raise ValueError(
        f"Ambiguous .pkl selection in {sub}. Provide explicit --scores-pkl-* path. Candidates: {[p.name for p in candidates]}"
    )


def resolve_scores_pkl_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    legacy_ref_to_pred = args.scores_pkl
    ref_to_pred = args.scores_pkl_ref_to_pred if args.scores_pkl_ref_to_pred is not None else legacy_ref_to_pred

    paths: dict[str, Path | None] = {
        KIND_REF_TO_PRED: ref_to_pred,
        KIND_REF_TO_REF: args.scores_pkl_ref_to_ref,
        KIND_REF_TO_ADJUSTED_PRED: args.scores_pkl_ref_to_adjusted_pred,
    }

    if args.scores_pkl_root is not None:
        root = Path(args.scores_pkl_root)
        if paths[KIND_REF_TO_PRED] is None:
            paths[KIND_REF_TO_PRED] = _discover_scores_pkl(
                root,
                subdir="ref_to_pred",
                stem_hint="scores_reference_prediction",
                window_size=args.window_size,
                window_stride=args.window_stride,
            )
        if paths[KIND_REF_TO_REF] is None:
            paths[KIND_REF_TO_REF] = _discover_scores_pkl(
                root,
                subdir="ref_to_ref",
                stem_hint="scores_reference_self",
                window_size=args.window_size,
                window_stride=args.window_stride,
            )
        if paths[KIND_REF_TO_ADJUSTED_PRED] is None:
            paths[KIND_REF_TO_ADJUSTED_PRED] = _discover_scores_pkl(
                root,
                subdir="ref_to_adjusted_pred",
                stem_hint="scores_reference_adjusted_prediction",
                window_size=args.window_size,
                window_stride=args.window_stride,
            )

    return paths


class ItemMatrixProvider:
    def __init__(
        self,
        *,
        item: dict,
        window_size: int,
        window_stride: int,
        score_index_by_kind: dict[str, dict[str, dict]],
        scores_pkl_paths_by_kind: dict[str, Path | None],
    ) -> None:
        self.item = item
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.score_index_by_kind = score_index_by_kind
        self.scores_pkl_paths_by_kind = scores_pkl_paths_by_kind
        self._cache: dict[str, np.ndarray] = {}
        self._source: dict[str, str] = {}

    def _lookup_item(self, kind: str) -> dict | None:
        lookup = self.score_index_by_kind.get(kind, {})
        return lookup.get(Path(str(self.item["fname"])).name)

    def _coerce_and_store_from_pkl(
        self,
        *,
        kind: str,
        score_index_item: dict,
        expected_pred_text: str | None = None,
    ) -> np.ndarray:
        ref_text = str(self.item["ref"])
        pred_text = str(self.item["pred"])

        p = self.scores_pkl_paths_by_kind.get(kind)
        if p is None:
            raise ValueError(f"Internal error: missing scores path for kind={kind!r}")

        if score_index_item.get("has_ref", False) and str(score_index_item.get("ref", "")) != ref_text:
            raise ValueError(
                f"Reference text mismatch between item and {kind} scores.pkl for {self.item['fname']!r}"
            )

        if kind == KIND_REF_TO_PRED:
            if score_index_item.get("has_pred", False) and str(score_index_item.get("pred", "")) != pred_text:
                raise ValueError(
                    f"Prediction text mismatch between item and {kind} scores.pkl for {self.item['fname']!r}"
                )

        if expected_pred_text is not None and score_index_item.get("has_pred", False):
            if str(score_index_item.get("pred", "")) != expected_pred_text:
                raise ValueError(
                    f"Adjusted prediction text mismatch between item and {kind} scores.pkl for {self.item['fname']!r}"
                )

        raw_item = load_score_item_by_offset(Path(p), int(score_index_item["offset"]))
        self._cache[kind] = coerce_score_matrix(
            raw_item.get("scores"),
            source_desc=f"{p}:{score_index_item['stream_index']}:{score_index_item['fname']}",
        )
        self._source[kind] = f"pkl:{p}" if p is not None else "pkl"
        return self._cache[kind]

    def _fetch_or_compute(
        self,
        *,
        kind: str,
        build_fn,
        expected_pred_text: str | None = None,
    ) -> np.ndarray:
        if kind in self._cache:
            return self._cache[kind]

        score_index_item = self._lookup_item(kind)
        if score_index_item is not None:
            return self._coerce_and_store_from_pkl(
                kind=kind,
                score_index_item=score_index_item,
                expected_pred_text=expected_pred_text,
            )

        if self.scores_pkl_paths_by_kind.get(kind) is not None:
            raise KeyError(
                f"No {kind} scores.pkl entry found for fname={self.item['fname']!r} in "
                f"{self.scores_pkl_paths_by_kind[kind]}"
            )

        self._cache[kind] = build_fn()
        self._source[kind] = "computed"
        return self._cache[kind]

    def get_ref_to_pred_matrix(self) -> np.ndarray:
        return self._fetch_or_compute(
            kind=KIND_REF_TO_PRED,
            build_fn=lambda: compute_score_matrix(
                str(self.item["ref"]),
                str(self.item["pred"]),
                window_size=self.window_size,
                window_stride=self.window_stride,
            ),
        )

    def get_ref_to_ref_matrix(self) -> np.ndarray:
        return self._fetch_or_compute(
            kind=KIND_REF_TO_REF,
            build_fn=lambda: compute_score_matrix(
                str(self.item["ref"]),
                str(self.item["ref"]),
                window_size=self.window_size,
                window_stride=self.window_stride,
            ),
        )

    def get_ref_to_adjusted_pred_matrix(self, adjusted_pred_text: str) -> np.ndarray:
        if (
            self.scores_pkl_paths_by_kind.get(KIND_REF_TO_ADJUSTED_PRED) is None
            and adjusted_pred_text == str(self.item["pred"])
            and KIND_REF_TO_PRED in self._cache
        ):
            self._cache[KIND_REF_TO_ADJUSTED_PRED] = self._cache[KIND_REF_TO_PRED]
            self._source[KIND_REF_TO_ADJUSTED_PRED] = self._source[KIND_REF_TO_PRED] + " (reused_ref_to_pred)"
            return self._cache[KIND_REF_TO_ADJUSTED_PRED]

        return self._fetch_or_compute(
            kind=KIND_REF_TO_ADJUSTED_PRED,
            build_fn=lambda: compute_score_matrix(
                str(self.item["ref"]),
                adjusted_pred_text,
                window_size=self.window_size,
                window_stride=self.window_stride,
            ),
            expected_pred_text=adjusted_pred_text,
        )

    def source_for(self, kind: str) -> str:
        return self._source.get(kind, "not_needed")


def _line_report(lines_used: list[dict]) -> list[dict]:
    out = []
    for lid, ln in enumerate(lines_used):
        out.append(
            {
                "line_id": int(lid),
                "x0": float(ln.get("x0", 0.0)),
                "y0": float(ln.get("y0", 0.0)),
                "x1": float(ln.get("x1", 0.0)),
                "y1": float(ln.get("y1", 0.0)),
                "score": float(ln.get("score", 0.0)),
                "length": float(ln.get("length", 0.0)),
                "support": float(ln.get("support", 0.0)),
                "owned_cols": int(ln.get("owned_cols", 0)),
                "owned_fraction": float(ln.get("owned_fraction", 0.0)),
                "owned_score_mean": float(ln.get("owned_score_mean", 0.0)),
                "owned_mask_hits": int(ln.get("owned_mask_hits", 0)),
                "owned_mask_fraction": float(ln.get("owned_mask_fraction", 0.0)),
                "anchor_y": float(ln.get("anchor_y", min(ln.get("y0", 0.0), ln.get("y1", 0.0)))),
            }
        )
    return out


def process_item(
    item: dict,
    args: argparse.Namespace,
    visual_output_dir: Path,
    *,
    score_index_by_kind: dict[str, dict[str, dict]],
    scores_pkl_paths_by_kind: dict[str, Path | None],
    visuals_enabled: bool = False,
) -> dict:
    t_total_start = time.perf_counter()
    timings: dict[str, float] = {}

    pred = str(item["pred"])
    ref = str(item["ref"])

    matrix_provider = ItemMatrixProvider(
        item=item,
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        score_index_by_kind=score_index_by_kind,
        scores_pkl_paths_by_kind=scores_pkl_paths_by_kind,
    )

    t0 = time.perf_counter()
    matrix = matrix_provider.get_ref_to_pred_matrix()
    timings["matrix_ref_to_pred_s"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    ref_to_ref_matrix = matrix_provider.get_ref_to_ref_matrix()
    timings["matrix_ref_to_ref_s"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    pred_lines_payload = detect_and_filter_lines_from_matrix(
        matrix,
        item_index=int(item["index"]),
        hough_threshold=int(args.hough_threshold),
        hough_line_length=int(args.hough_line_length),
        hough_line_gap=int(args.hough_line_gap),
        hough_seed=int(args.hough_seed),
        hough_start=float(args.hough_start),
        align_abs_min_len=float(args.align_abs_min_len),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
    )
    timings["hough_filter_ref_to_pred_s"] = float(time.perf_counter() - t0)

    det = pred_lines_payload["det"]
    raw_hough_segments = pred_lines_payload["raw_hough_segments"]
    lines_for_filtering = pred_lines_payload["lines_for_filtering"]
    lines_used = pred_lines_payload["lines_used"]
    column_assignment = pred_lines_payload["column_assignment"]

    n_ref = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    n_other = int(matrix.shape[1]) if matrix.ndim == 2 else 0

    t0 = time.perf_counter()
    bundle_ref_to_pred = build_line_metric_bundle(
        lines_used=lines_used,
        column_assignment=column_assignment,
        n_ref_windows=n_ref,
        n_other_windows=n_other,
        ref_text_len=len(ref),
        other_text_len=len(pred),
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
    )
    timings["bundle_ref_to_pred_s"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    line_metric = compute_levenshtein_metrics_from_bundle(
        ref_text=ref,
        other_text=pred,
        lines_used=lines_used,
        bundle=bundle_ref_to_pred,
        backend=str(args.levenshtein_backend),
    )
    timings["levenshtein_metrics_s"] = float(time.perf_counter() - t0)

    before_nls = float(line_metric["whole_document_normalized_levenshtein_similarity"])
    along_lines_nls = line_metric.get("document_normalized_levenshtein_similarity_along_lines")
    after_nls = float(before_nls if along_lines_nls is None else along_lines_nls)
    adjusted_pred = pred

    case_prefix = f"{item['index']:04d}_{safe_name(Path(item['fname']).name)}"

    t0 = time.perf_counter()
    refref_lines_payload = detect_and_filter_lines_from_matrix(
        ref_to_ref_matrix,
        item_index=int(item["index"]),
        hough_threshold=int(args.hough_threshold),
        hough_line_length=int(args.hough_line_length),
        hough_line_gap=int(args.hough_line_gap),
        hough_seed=int(args.hough_seed),
        hough_start=float(args.hough_start),
        align_abs_min_len=float(args.align_abs_min_len),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
    )
    timings["hough_filter_ref_to_ref_s"] = float(time.perf_counter() - t0)

    lines_used_ref_to_ref = refref_lines_payload["lines_used"]
    column_assignment_ref_to_ref = refref_lines_payload["column_assignment"]

    n_ref_ref = int(ref_to_ref_matrix.shape[0]) if ref_to_ref_matrix.ndim == 2 else 0
    n_other_ref = int(ref_to_ref_matrix.shape[1]) if ref_to_ref_matrix.ndim == 2 else 0

    t0 = time.perf_counter()
    bundle_ref_to_ref = build_line_metric_bundle(
        lines_used=lines_used_ref_to_ref,
        column_assignment=column_assignment_ref_to_ref,
        n_ref_windows=n_ref_ref,
        n_other_windows=n_other_ref,
        ref_text_len=len(ref),
        other_text_len=len(ref),
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
    )
    timings["bundle_ref_to_ref_s"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    line_coverage_metrics = compute_line_coverage_percentage_metrics_from_bundles(
        refref_bundle=bundle_ref_to_ref,
        other_bundle=bundle_ref_to_pred,
        file_name=Path(item["fname"]).name,
    )
    timings["coverage_subtract_s"] = float(time.perf_counter() - t0)

    line_guided_columns = int(bundle_ref_to_pred.get("line_guided_columns", 0))
    fallback_columns = int(bundle_ref_to_pred.get("fallback_columns", 0))

    matrix_source_adjusted = "not_needed"
    if visuals_enabled:
        t0 = time.perf_counter()
        from visualise_used_lines_from_report import save_text_metrics_visualisations

        matrix_after_reordering = matrix_provider.get_ref_to_adjusted_pred_matrix(adjusted_pred)
        matrix_source_adjusted = matrix_provider.source_for(KIND_REF_TO_ADJUSTED_PRED)
        vis_paths = save_text_metrics_visualisations(
            matrix_before=matrix,
            raw_hough_segments=raw_hough_segments,
            pre_filter_lines=lines_for_filtering,
            filtered_lines=lines_used,
            matrix_after_reordering=matrix_after_reordering,
            case_prefix=case_prefix,
            file_name=Path(item["fname"]).name,
            output_dir=visual_output_dir,
            threshold_start=float(det.get("threshold_start", float("nan"))),
            line_filter_label=f"v2.1_true_IoU @ {float(args.align_min_iou_threshold):.3f}",
        )
        timings["visualisations_s"] = float(time.perf_counter() - t0)
    else:
        vis_paths = {
            "visualise_before_hough_path": None,
            "visualise_after_hough_line_transform_path": None,
            "visualise_after_filtering_path": None,
            "visualise_after_reordering_path": None,
            "visualise_raw_hough_path": None,
            "visualise_after_v2_1_true_iou_path": None,
            "visualise_after_reorder_path": None,
            "visualise_full_path": None,
            "visualise_graph_path": None,
            "visualise_mask_path": None,
        }

    matrix_source_ref_to_pred = matrix_provider.source_for(KIND_REF_TO_PRED)
    matrix_source_ref_to_ref = matrix_provider.source_for(KIND_REF_TO_REF)
    timings["total_item_s"] = float(time.perf_counter() - t_total_start)

    return {
        "index": int(item["index"]),
        "fname": Path(item["fname"]).name,
        "adjusted_pred": adjusted_pred,
        "initial_matrix_source": matrix_source_ref_to_pred,
        "matrix_source_ref_to_pred": matrix_source_ref_to_pred,
        "matrix_source_ref_to_ref": matrix_source_ref_to_ref,
        "matrix_source_ref_to_adjusted_pred": matrix_source_adjusted,
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_shape_ref_to_ref": [int(ref_to_ref_matrix.shape[0]), int(ref_to_ref_matrix.shape[1])],
        "visuals_enabled": bool(visuals_enabled),
        "normalized_levenshtein_before": float(before_nls),
        "average_normalized_levenshtein_along_lines": None if along_lines_nls is None else float(along_lines_nls),
        "ok_percent": float(line_coverage_metrics["ok_percent"]),
        "missing_percent": float(line_coverage_metrics["missing_percent"]),
        "repetition_percent": float(line_coverage_metrics["repetition_percent"]),
        "hallucination_percent": float(line_coverage_metrics["hallucination_percent"]),
        "before_normalized_levenshtein_similarity": float(before_nls),
        "after_normalized_levenshtein_similarity": float(after_nls),
        "delta": float(after_nls - before_nls),
        "whole_document_normalized_levenshtein_similarity": float(before_nls),
        "document_normalized_levenshtein_similarity_along_lines": along_lines_nls,
        "levenshtein_backend": str(args.levenshtein_backend),
        "line_metric_line_count": int(line_metric.get("line_count", 0)),
        "line_metric_lines": line_metric.get("lines", []),
        "line_coverage_metrics": line_coverage_metrics,
        "line_guided_columns": int(line_guided_columns),
        "fallback_columns": int(fallback_columns),
        "attached_between_columns": 0,
        "attached_between_runs": 0,
        "movable_components": 0,
        "raw_line_count": int(len(raw_hough_segments)),
        "merged_line_count": int(len(lines_for_filtering)),
        "used_line_count": int(len(lines_used)),
        "used_line_count_ref_to_ref": int(len(lines_used_ref_to_ref)),
        "threshold_start": float(det.get("threshold_start", float("nan"))),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed) + int(item["index"]),
        "hough_start": float(args.hough_start),
        "align_abs_min_len": float(args.align_abs_min_len),
        "line_filter_version": "v2_1_true_iou",
        "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
        "lines_used": _line_report(lines_used),
        "__timing": timings,
        **vis_paths,
    }


def _select_run_items_source_kind(
    *,
    runfile_json: Path | None,
    score_index_by_kind: dict[str, dict[str, dict]],
) -> str:
    if runfile_json is not None:
        return "runfile"

    for kind in (KIND_REF_TO_PRED, KIND_REF_TO_REF, KIND_REF_TO_ADJUSTED_PRED):
        if score_index_by_kind.get(kind):
            return kind

    raise ValueError("No input source available: provide --runfile-json and/or at least one --scores-pkl-* file")


def _avg_or_none(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def main() -> None:
    args = parse_args()
    scores_pkl_paths_by_kind = resolve_scores_pkl_paths(args)

    if args.runfile_json is None and not any(scores_pkl_paths_by_kind.values()):
        raise ValueError("Provide at least one input source: --runfile-json or any --scores-pkl-* option")

    if args.runfile_json is not None and not args.runfile_json.exists():
        raise FileNotFoundError(f"Missing runfile JSON: {args.runfile_json}")

    for kind, path in scores_pkl_paths_by_kind.items():
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"Missing {kind} scores file: {path}")

    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("window-size and window-stride must be positive")
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("max-items must be positive")
    if args.hough_threshold <= 0:
        raise ValueError("hough-threshold must be positive")
    if args.hough_line_length <= 0:
        raise ValueError("hough-line-length must be positive")
    if args.hough_line_gap < 0:
        raise ValueError("hough-line-gap must be non-negative")
    if args.hough_start <= 0:
        raise ValueError("hough-start must be positive")
    if args.align_abs_min_len <= 0:
        raise ValueError("align-abs-min-len must be positive")
    if not (0.0 <= args.align_min_iou_threshold <= 1.0):
        raise ValueError("align-min-iou-threshold must satisfy 0.0 <= value <= 1.0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals_enabled = not bool(args.skip_visuals)
    visual_output_dir = args.output_dir

    score_index_by_kind: dict[str, dict[str, dict]] = {}
    score_index_cache_root = _SCRIPT_DIR / ".score_index_cache"
    for kind, path in scores_pkl_paths_by_kind.items():
        if path is None:
            score_index_by_kind[kind] = {}
            continue
        score_index_by_kind[kind] = build_score_stream_index_cached(Path(path), score_index_cache_root)

    items_source_kind = _select_run_items_source_kind(
        runfile_json=args.runfile_json,
        score_index_by_kind=score_index_by_kind,
    )
    if items_source_kind == "runfile":
        run_items = load_run_items(args.runfile_json)
    else:
        source_path = scores_pkl_paths_by_kind[items_source_kind]
        run_items = load_run_items_from_score_index(score_index_by_kind[items_source_kind], Path(source_path))

    report_items: list[dict] = []
    timing_items: list[dict] = []
    matched = 0

    for item in run_items:
        if args.target_fname is not None and not same_file(item["fname"], args.target_fname):
            continue
        matched += 1

        if args.max_items is not None and len(report_items) >= args.max_items:
            break

        res = process_item(
            item,
            args,
            visual_output_dir,
            score_index_by_kind=score_index_by_kind,
            scores_pkl_paths_by_kind=scores_pkl_paths_by_kind,
            visuals_enabled=visuals_enabled,
        )

        timings = dict(res.pop("__timing", {}))
        report_items.append(res)

        timing_items.append(
            {
                "index": int(res["index"]),
                "fname": str(res["fname"]),
                "timings_seconds": timings,
            }
        )

        print(
            f"[{len(report_items)}] {res['fname']} | "
            f"before={res['normalized_levenshtein_before']:.6f} "
            f"along={res['average_normalized_levenshtein_along_lines']} "
            f"ok={res['ok_percent']:.4f} missing={res['missing_percent']:.4f} "
            f"repetition={res['repetition_percent']:.4f} hallucination={res['hallucination_percent']:.4f}"
        )

    if args.target_fname is not None and matched == 0:
        raise KeyError(f"Target file not found in provided input items: {args.target_fname!r}")
    if not report_items:
        raise RuntimeError("No items processed. Check target filters and inputs.")

    avg_before = _avg_or_none([item["normalized_levenshtein_before"] for item in report_items])
    avg_along = _avg_or_none([item["average_normalized_levenshtein_along_lines"] for item in report_items])

    report_payload = {
        "count": int(len(report_items)),
        "runfile_json": None if args.runfile_json is None else str(args.runfile_json),
        "scores_pkl": None if scores_pkl_paths_by_kind[KIND_REF_TO_PRED] is None else str(scores_pkl_paths_by_kind[KIND_REF_TO_PRED]),
        "scores_pkl_ref_to_pred": (
            None if scores_pkl_paths_by_kind[KIND_REF_TO_PRED] is None else str(scores_pkl_paths_by_kind[KIND_REF_TO_PRED])
        ),
        "scores_pkl_ref_to_ref": (
            None if scores_pkl_paths_by_kind[KIND_REF_TO_REF] is None else str(scores_pkl_paths_by_kind[KIND_REF_TO_REF])
        ),
        "scores_pkl_ref_to_adjusted_pred": (
            None
            if scores_pkl_paths_by_kind[KIND_REF_TO_ADJUSTED_PRED] is None
            else str(scores_pkl_paths_by_kind[KIND_REF_TO_ADJUSTED_PRED])
        ),
        "scores_pkl_root": None if args.scores_pkl_root is None else str(args.scores_pkl_root),
        "visuals_enabled": bool(visuals_enabled),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed),
        "hough_start": float(args.hough_start),
        "align_abs_min_len": float(args.align_abs_min_len),
        "line_filter_version": "v2_1_true_iou",
        "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
        "levenshtein_backend": str(args.levenshtein_backend),
        "debug": bool(args.debug),
        "run_average_normalized_levenshtein_before": avg_before,
        "run_average_normalized_levenshtein_along_lines": avg_along,
        "items": report_items,
    }

    out_report = args.output_dir / "report.json"
    out_report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_timing = None
    if bool(args.debug):
        timing_payload = {
            "count": int(len(timing_items)),
            "runfile_json": None if args.runfile_json is None else str(args.runfile_json),
            "scores_pkl_root": None if args.scores_pkl_root is None else str(args.scores_pkl_root),
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "hough_threshold": int(args.hough_threshold),
            "hough_line_length": int(args.hough_line_length),
            "hough_line_gap": int(args.hough_line_gap),
            "hough_seed": int(args.hough_seed),
            "hough_start": float(args.hough_start),
            "align_abs_min_len": float(args.align_abs_min_len),
            "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
            "levenshtein_backend": str(args.levenshtein_backend),
            "items": timing_items,
        }
        out_timing = args.output_dir / "report_timings.json"
        out_timing.write_text(json.dumps(timing_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Processed items: {len(report_items)}")
    if avg_before is None:
        print("Run avg normalized levenshtein before: <none>")
    else:
        print(f"Run avg normalized levenshtein before: {avg_before:.6f}")
    if avg_along is None:
        print("Run avg normalized levenshtein along lines: <none>")
    else:
        print(f"Run avg normalized levenshtein along lines: {avg_along:.6f}")
    print(f"Report: {out_report}")
    if out_timing is not None:
        print(f"Timings: {out_timing}")


if __name__ == "__main__":
    main()
