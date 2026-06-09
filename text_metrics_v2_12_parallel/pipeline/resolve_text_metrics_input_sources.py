"""Resolve score-pkl paths and build selected run items for processing."""

from __future__ import annotations

from pathlib import Path

from runfile_records import load_run_items, same_file
from score_stream_index import load_run_items_from_score_index

KIND_REF_TO_PRED = "ref_to_pred"
KIND_REF_TO_REF = "ref_to_ref"
KIND_REF_TO_ADJUSTED_PRED = "ref_to_adjusted_pred"


# Discover one scores.pkl file under a compare subdirectory using stable matching rules.
def discover_scores_pkl(
    root: Path,
    *,
    subdir: str,
    stem_hint: str,
    window_size: int,
    window_stride: int,
) -> Path | None:
    """Discover a .pkl file in a compare subdirectory using stable rules."""
    subdirectory = Path(root) / subdir
    if not subdirectory.exists() or not subdirectory.is_dir():
        return None

    candidate_files = sorted(path for path in subdirectory.glob("*.pkl") if path.is_file())
    if not candidate_files:
        return None
    if len(candidate_files) == 1:
        return candidate_files[0]

    exact_name_match = subdirectory / f"{stem_hint}_ws{int(window_size)}_st{int(window_stride)}.pkl"
    if exact_name_match.exists():
        return exact_name_match

    window_signature = f"ws{int(window_size)}_st{int(window_stride)}"
    matching_window_signature_files = [path for path in candidate_files if window_signature in path.name]
    if len(matching_window_signature_files) == 1:
        return matching_window_signature_files[0]

    matching_hint_files = [path for path in matching_window_signature_files if stem_hint in path.name]
    if len(matching_hint_files) == 1:
        return matching_hint_files[0]

    raise ValueError(
        f"Ambiguous .pkl selection in {subdirectory}. Provide explicit --scores-pkl-* path. "
        f"Candidates: {[path.name for path in candidate_files]}"
    )


# Resolve explicit score-pkl inputs and optional root discovery into canonical paths.
def resolve_scores_pkl_paths(args) -> dict[str, Path | None]:
    """Resolve explicit/root score-pkl options into canonical paths."""
    paths: dict[str, Path | None] = {
        KIND_REF_TO_PRED: args.scores_pkl_ref_to_pred,
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


# Choose which source provides the normalized item list for processing.
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


# Load normalized run items from the chosen source kind.
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


# Apply target/max item selection while preserving stable counting semantics.
def build_selected_items(
    run_items: list[dict],
    *,
    target_fname: str | None,
    max_items: int | None,
) -> tuple[list[dict], int, int]:
    """Apply target/max selection while preserving existing counters."""
    selected_items: list[dict] = []
    matched_count = 0
    attempted_count = 0

    for item in run_items:
        if target_fname is not None and not same_file(item["fname"], target_fname):
            continue
        matched_count += 1

        if max_items is not None and attempted_count >= max_items:
            break

        selected_items.append(item)
        attempted_count += 1

    return selected_items, matched_count, attempted_count
