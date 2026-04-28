"""Resolve score-pkl paths and build selected run items for processing."""

from __future__ import annotations

from pathlib import Path

from runfile_records import load_run_items, same_file
from score_stream_index import load_run_items_from_score_index

KIND_REF_TO_PRED = "ref_to_pred"
KIND_REF_TO_REF = "ref_to_ref"
KIND_REF_TO_ADJUSTED_PRED = "ref_to_adjusted_pred"


def discover_scores_pkl(
    root: Path,
    *,
    subdir: str,
    stem_hint: str,
    window_size: int,
    window_stride: int,
) -> Path | None:
    """Discover a .pkl file in a compare subdirectory using stable rules."""
    sub = Path(root) / subdir
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
        f"Ambiguous .pkl selection in {sub}. Provide explicit --scores-pkl-* path. "
        f"Candidates: {[p.name for p in candidates]}"
    )


def resolve_scores_pkl_paths(args) -> dict[str, Path | None]:
    """Resolve explicit/legacy/root score-pkl options into canonical paths."""
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
            paths[KIND_REF_TO_PRED] = discover_scores_pkl(
                root,
                subdir="ref_to_pred",
                stem_hint="scores_reference_prediction",
                window_size=args.window_size,
                window_stride=args.window_stride,
            )
        if paths[KIND_REF_TO_REF] is None:
            paths[KIND_REF_TO_REF] = discover_scores_pkl(
                root,
                subdir="ref_to_ref",
                stem_hint="scores_reference_self",
                window_size=args.window_size,
                window_stride=args.window_stride,
            )
        if paths[KIND_REF_TO_ADJUSTED_PRED] is None:
            paths[KIND_REF_TO_ADJUSTED_PRED] = discover_scores_pkl(
                root,
                subdir="ref_to_adjusted_pred",
                stem_hint="scores_reference_adjusted_prediction",
                window_size=args.window_size,
                window_stride=args.window_stride,
            )

    return paths


def select_run_items_source_kind(
    *,
    runfile_json: Path | None,
    score_index_by_kind: dict[str, dict[str, dict]],
) -> str:
    """Choose which source provides the item list for processing."""
    if runfile_json is not None:
        return "runfile"

    for kind in (KIND_REF_TO_PRED, KIND_REF_TO_REF, KIND_REF_TO_ADJUSTED_PRED):
        if score_index_by_kind.get(kind):
            return kind

    raise ValueError("No input source available: provide --runfile-json and/or at least one --scores-pkl-* file")


def load_items_from_sources(
    *,
    items_source_kind: str,
    runfile_json: Path | None,
    score_index_by_kind: dict[str, dict[str, dict]],
    scores_pkl_paths_by_kind: dict[str, Path | None],
) -> list[dict]:
    """Load normalized run items from either runfile JSON or score stream index."""
    if items_source_kind == "runfile":
        return load_run_items(Path(runfile_json))

    source_path = scores_pkl_paths_by_kind[items_source_kind]
    return load_run_items_from_score_index(score_index_by_kind[items_source_kind], Path(source_path))


def build_selected_items(
    run_items: list[dict],
    *,
    target_fname: str | None,
    max_items: int | None,
) -> tuple[list[dict], int, int]:
    """Apply target/max selection while preserving existing counters."""
    selected_items: list[dict] = []
    matched = 0
    attempted = 0

    for item in run_items:
        if target_fname is not None and not same_file(item["fname"], target_fname):
            continue
        matched += 1

        if max_items is not None and attempted >= max_items:
            break

        selected_items.append(item)
        attempted += 1

    return selected_items, matched, attempted
