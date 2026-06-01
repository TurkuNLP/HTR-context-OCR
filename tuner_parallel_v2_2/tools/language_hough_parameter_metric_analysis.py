#!/usr/bin/env python3
"""Create language/document-type Hough tuner diagnostics from bundle output.

The script reads existing tuner combination bundles and creates diagnostic plots
without recomputing any tuner metrics.  It is designed for selective inspection:
you choose one or more ``main_language`` values and one or more real
``document_type`` values from ``outputs.json``.  For each selected
language/document-type pair, the script builds one compact pandas DataFrame,
then writes:

* one line-graph grid per loaded document;
* one stitched best-combination matrix/Hough panel per language/document-type pair;
* one stitched best-combination metrics text file per language/document-type pair;
* CSV tables with compact metrics, best combinations, loaded documents, skipped
  documents, and document-type summaries;
* one top-level manifest JSON.

For static-shard output, the script uses the known sharding rule
``document_index // documents_per_shard`` to probe the expected shard directly.
For dynamic-pool output, it also checks each ``dynamic_*/combination_bundles``
directory because any worker may have claimed the document.  Documents whose
``ref_to_pred`` matrix has zero prediction columns are skipped before any
threshold record stream is loaded.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

# The script is meant to run from terminal or sbatch, so use a headless backend.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

# The plotting script can be run directly from tools/, so bootstrap the tuner
# paths before importing shared score-matrix helpers.
try:
    from runtime.runtime_paths import ensure_tuner_runtime_paths
except ImportError:  # pragma: no cover - direct package import fallback.
    tuner_root = Path(__file__).resolve().parents[1]
    if str(tuner_root) not in sys.path:
        sys.path.insert(0, str(tuner_root))
    from runtime.runtime_paths import ensure_tuner_runtime_paths  # type: ignore

ensure_tuner_runtime_paths()
from score_matrix_builder import compute_score_matrix  # type: ignore

try:
    from outputs.combination_bundle_records import (
        GZIP_JSONL_SUFFIX,
        JSONL_SUFFIX,
        PICKLE_STREAM_SUFFIX,
        iter_pickle_stream_records,
        read_pickle_stream_record_at_position,
    )
except ImportError:  # pragma: no cover - supports package-style execution.
    from ..outputs.combination_bundle_records import (  # type: ignore
        GZIP_JSONL_SUFFIX,
        JSONL_SUFFIX,
        PICKLE_STREAM_SUFFIX,
        iter_pickle_stream_records,
        read_pickle_stream_record_at_position,
    )


DEFAULT_RUNFILE_JSON = Path(
    "/scratch/project_2017385/dorian/Churro_copy/results/"
    "custom_churro_infer_dev_run1/vllm/dev/outputs.json"
)
DEFAULT_SHARDS_DIR = Path(
    "/scratch/project_2000539/dorian/results/"
    "tuner_parallel_v2_docs1000_th10_35_len5_35_gap0_15_cache_off_50docs_per_node/shards"
)
DEFAULT_REF_TO_PRED_SCORES_PKL = Path(
    "/scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/"
    "ref_to_pred/scores_reference_prediction_ws50_st35.pkl"
)
DEFAULT_REF_TO_REF_SCORES_PKL = Path(
    "/scratch/project_2017385/dorian/Churro_copy/results/compares_churro_dev/"
    "ref_to_ref/scores_reference_self_ws50_st35.pkl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_2/"
    "_language_hough_parameter_metric_visuals_script"
)
DEFAULT_DOCUMENTS_PER_SHARD = 50
DEFAULT_WINDOW_SIZE = 50
DEFAULT_WINDOW_STRIDE = 35

REFERENCE_TEXT_FOR_PLOTTING_KEY = "_reference_text_for_plotting"
PREDICTION_TEXT_FOR_PLOTTING_KEY = "_prediction_text_for_plotting"

PARAMETER_METRIC_LINE_GRAPH_GRID_FILENAME_PREFIX = "parameter_metric_line_graph_grid"
LEGACY_PARAMETER_METRIC_LINE_GRAPH_GRID_FILENAME = "parameter_metric_line_graph_grid.png"
PLOTS_DIRECTORY_NAME = "plots"
DOCUMENT_PLOTS_DIRECTORY_NAME = "documents"
STITCHED_BEST_COMBINATION_METRICS_JSON_FILENAME = "stitched_best_combination_metrics.json"
STITCHED_BEST_COMBINATION_METRICS_FILENAME = "stitched_best_combination_metrics.txt"
STITCHED_BEST_COMBINATION_PANEL_FILENAME = (
    "stitched_best_combination_no_hough_raw_hough_surviving_lines_roi_hough_input_panel.png"
)
PER_DOCUMENT_BEST_COMBINATION_METRICS_FILENAME = "best_combination_metrics.txt"
PER_DOCUMENT_BEST_COMBINATION_PANEL_FILENAME = (
    "best_combination_no_hough_raw_hough_surviving_lines_roi_hough_input_panel.png"
)
TEMPORARY_BEST_COMBINATION_PANEL_DIRECTORY_NAME = ".temporary_best_combination_panels"
TEMPORARY_BEST_COMBINATION_PANEL_FILENAME_PREFIX = (
    "best_combination_no_hough_raw_hough_surviving_lines_roi_hough_input_panel"
)
BEST_COMBINATION_PANEL_GRID_COLUMN_COUNT = 3

# Raw detections stay cyan so they remain visually distinct from the final filtered result.
RAW_HOUGH_LINE_COLOR = "#00B8D9"
RAW_HOUGH_LINE_WIDTH_POINTS = 2.0
RAW_HOUGH_LINE_BOX_HALF_WIDTH_CELLS = 1.8

# Final surviving lines are the key diagnostic output, so they use the requested red color.
SURVIVING_FILTERED_LINE_COLOR = "#DC2626"
SURVIVING_FILTERED_LINE_WIDTH_POINTS = 2.8
SURVIVING_FILTERED_LINE_BOX_HALF_WIDTH_CELLS = 2.4

HOUGH_PARAMETER_COLUMNS = [
    ("hough_threshold", "Hough threshold"),
    ("hough_line_length", "Hough line length"),
    ("hough_line_gap", "Hough line gap"),
]

HARMONIC_COMPONENT_COLUMNS = [
    ("weighted_along_lines_nls", "Weighted along-lines NLS"),
    ("correct_ref_coverage", "Correct reference coverage"),
    ("non_hallucination", "Non-hallucination (1 - hallucination)"),
]

BEST_COMBINATION_METRIC_COLUMNS = [
    "tuning_score",
    "alignment_selection_score",
    "score_matrix_support",
    "line_guided_fraction",
    "weighted_along_lines_nls",
    "whole_document_nls",
    "correct_ref_coverage",
    "missing_ref_coverage",
    "repetition_on_ref",
    "hallucination",
    "raw_line_count",
    "used_line_count",
]

PUBLIC_BEST_DOCUMENT_JSON_FIELDS = [
    "best_threshold",
    "best_line_length",
    "best_line_gap",
    "best_seed",
    "tuning_score",
    "selection_objective",
    "alignment_selection_score",
    "score_matrix_support",
    "line_guided_fraction",
    "normalized_levenshtein_similarity",
    "average_weighted_normalized_levenshtein_similarity",
    "normalised_document_levenshtein_similarity_after_alignment",
    "correct_reference_coverage",
    "missing_reference_coverage",
    "repetition_on_reference",
    "hallucination",
    "raw_line_count",
    "used_line_count",
    "raw_hough_lines",
    "surviving_lines",
]

PUBLIC_AVERAGE_METRIC_FIELDS = [
    "tuning_score",
    "alignment_selection_score",
    "score_matrix_support",
    "line_guided_fraction",
    "normalized_levenshtein_similarity",
    "average_weighted_normalized_levenshtein_similarity",
    "normalised_document_levenshtein_similarity_after_alignment",
    "correct_reference_coverage",
    "missing_reference_coverage",
    "repetition_on_reference",
    "hallucination",
]

COMPACT_METRIC_COLUMNS = [
    "main_language",
    "document_type",
    "document_index",
    "fname",
    "bundle_dir",
    "shard_index",
    "whole_document_nls",
    "hough_threshold",
    "hough_line_length",
    "hough_line_gap",
    "hough_seed",
    "effective_hough_seed",
    "tuning_score",
    "selection_objective",
    "alignment_selection_score",
    "score_matrix_support",
    "line_guided_fraction",
    "weighted_along_lines_nls",
    "correct_ref_coverage",
    "missing_ref_coverage",
    "repetition_on_ref",
    "hallucination",
    "non_hallucination",
    "raw_line_count",
    "candidate_line_count",
    "used_line_count",
    "line_guided_columns",
    "fallback_columns",
    "is_valid",
    "invalid_reason",
    "source_jsonl_path",
    "source_line_number",
]

COMBINATION_SCORE_TABLE_FILENAME = "combination_scores.csv.gz"

SCORE_TABLE_COLUMNS_REQUIRED_FOR_VISUALS = {
    "doc_index",
    "fname",
    "whole_document_nls",
    "hough_threshold",
    "hough_line_length",
    "hough_line_gap",
    "hough_seed",
    "tuning_score",
    "selection_objective",
    "alignment_selection_score",
    "score_matrix_support",
    "line_guided_fraction",
    "weighted_along_lines_nls",
    "correct_ref_coverage",
    "missing_ref_coverage",
    "repetition_on_ref",
    "hallucination",
    "raw_line_count",
    "candidate_line_count",
    "used_line_count",
    "line_guided_columns",
    "fallback_columns",
    "is_valid",
    "invalid_reason",
}

DOCUMENT_TABLE_COLUMNS = [
    "document_index",
    "fname",
    "main_language",
    "document_type",
    "main_script",
    "file_name",
    "shard_index",
    "bundle_dir",
    "diagnostic_bundle_dir",
    "bundle_record_format",
    "ref_to_pred_matrix_shape",
    "ref_to_ref_matrix_shape",
    "skip_reason",
    "skip_stage",
    "preprocessing_rejection_reason",
]


class ConfigurationError(ValueError):
    """Raised when command-line filters do not match outputs.json metadata."""


# ---------------------------------------------------------------------------
# Safe conversion and formatting helpers
# ---------------------------------------------------------------------------


def safe_float_or_nan(value: Any) -> float:
    """Return a float value, or NaN when the value is missing or non-numeric."""
    # Missing values should stay missing so pandas can ignore them in means.
    if value is None:
        return float("nan")

    # JSON values can be strings, integers, floats, or unexpected nested data.
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_int_or_none(value: Any) -> int | None:
    """Return an integer value, or None when the value is missing or non-numeric."""
    # Missing values should remain missing instead of becoming zero.
    if value is None:
        return None

    # int(float(...)) handles values that arrive as numeric strings.
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_path_component(value: Any) -> str:
    """Return a readable filesystem-safe path component."""
    # Convert anything to text before replacing unsafe path characters.
    text_value = str(value if value is not None else "missing")

    # Keep the component readable while avoiding whitespace and path separators.
    safe_text = re.sub(r"[^A-Za-z0-9._-]+", "_", text_value)

    # Avoid returning an empty component when the input was punctuation only.
    return safe_text.strip("._") or "missing"


def format_metric_for_text(value: Any) -> str:
    """Format a value for compact metric labels and text summaries."""
    # Keep missing values explicit in saved summaries.
    if value is None:
        return "None"

    # Pandas missing values should not silently look like real numbers.
    try:
        if pd.isna(value):
            return "NaN"
    except TypeError:
        pass

    # Fixed precision makes side-by-side metric comparisons easier.
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"

    # Non-floats are written plainly.
    return str(value)


def json_ready_value(value: Any) -> Any:
    """Convert common NumPy/Pandas/Path values into JSON-serializable values."""
    # Path values should be written as strings in JSON manifests.
    if isinstance(value, Path):
        return str(value)

    # NumPy integer and floating types are not serializable by the standard JSON encoder.
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)

    # Tuples are converted recursively so matrix shapes become simple lists.
    if isinstance(value, tuple):
        return [json_ready_value(item) for item in value]

    # NaN floats should be written as null for valid JSON semantics.
    if isinstance(value, float) and math.isnan(value):
        return None

    return value


# ---------------------------------------------------------------------------
# Runfile metadata and filter validation
# ---------------------------------------------------------------------------


def load_runfile_items(runfile_json_path: Path) -> list[dict[str, Any]]:
    """Load outputs.json and validate that it is a list of document dictionaries."""
    # The runfile is the authority for document index, language, and document_type.
    with Path(runfile_json_path).open("r", encoding="utf-8") as runfile_handle:
        runfile_items = json.load(runfile_handle)

    # This script expects the Churro outputs.json list format.
    if not isinstance(runfile_items, list):
        raise TypeError(f"Expected {runfile_json_path} to contain a list, found {type(runfile_items).__name__}")

    # Keep only dictionary entries so the rest of the code can access keys safely.
    return [item for item in runfile_items if isinstance(item, dict)]


def available_metadata_values(runfile_items: list[dict[str, Any]], field_name: str) -> list[str]:
    """Return sorted non-empty metadata values for one runfile field."""
    # Convert values to strings because argparse filters are strings.
    values = {str(item[field_name]) for item in runfile_items if item.get(field_name) not in (None, "")}
    return sorted(values)


def validate_requested_values(requested_values: list[str], available_values: list[str], field_label: str) -> None:
    """Raise a clear error if any requested language/type does not exist in outputs.json."""
    # Build a set for exact membership checks.
    available_value_set = set(available_values)

    # Find every unknown requested value so the user can fix all typos at once.
    unknown_values = [value for value in requested_values if value not in available_value_set]
    if not unknown_values:
        return

    # The error prints only real values from the runfile, never invented alternatives.
    raise ConfigurationError(
        f"Unknown {field_label} value(s): {', '.join(unknown_values)}. "
        f"Available {field_label} values: {', '.join(available_values)}"
    )


def selected_values_from_arguments(
    *,
    explicit_values: list[str] | None,
    select_all: bool,
    available_values: list[str],
    field_label: str,
) -> list[str]:
    """Resolve explicit or all-values CLI selection for one metadata field."""
    # The user must choose either explicit values or the all-values flag.
    if explicit_values and select_all:
        raise ConfigurationError(f"Use either --{field_label} or --all-{field_label}s, not both.")

    # The all-values flag expands to exactly the values present in outputs.json.
    if select_all:
        return list(available_values)

    # Explicit values must exist in outputs.json.
    if explicit_values:
        validate_requested_values(explicit_values, available_values, field_label)
        return list(dict.fromkeys(explicit_values))

    # Avoid accidental huge runs when the user forgets a filter.
    raise ConfigurationError(f"Pass at least one --{field_label} value or --all-{field_label}s.")


def build_selected_runfile_documents(
    runfile_items: list[dict[str, Any]],
    selected_languages: list[str],
    selected_document_types: list[str],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group runfile documents by selected (language, document_type) pairs."""
    # Initialize every selected pair so empty combinations still get a manifest entry.
    documents_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {
        (language_name, document_type): []
        for language_name in selected_languages
        for document_type in selected_document_types
    }

    # The list position in outputs.json is the tuner document index.
    for document_index, item in enumerate(runfile_items):
        # Read exact metadata values from the runfile.
        main_language = item.get("main_language")
        document_type = item.get("document_type")

        # Keep only selected language/type pairs.
        pair_key = (main_language, document_type)
        if pair_key not in documents_by_pair:
            continue

        # Bundle folders use the basename, while outputs.json often contains dev/<name>.
        original_file_name = str(item.get("file_name", ""))
        document_file_name = Path(original_file_name).name

        # Store compact metadata only; full text is not needed for these diagnostics.
        documents_by_pair[pair_key].append(
            {
                "document_index": int(document_index),
                "file_name": original_file_name,
                "fname": document_file_name,
                "main_language": main_language,
                "document_type": document_type,
                "main_script": item.get("main_script"),
                REFERENCE_TEXT_FOR_PLOTTING_KEY: str(item.get("normalized_gold_text", item.get("ref", ""))),
                PREDICTION_TEXT_FOR_PLOTTING_KEY: str(item.get("normalized_predicted_text", item.get("pred", ""))),
                "whole_document_nls_from_runfile": safe_float_or_nan(
                    item.get("normalized_levenshtein_similarity")
                ),
            }
        )

    # Ensure stable loading order within every pair.
    for documents in documents_by_pair.values():
        documents.sort(key=lambda document: int(document["document_index"]))

    return documents_by_pair


# ---------------------------------------------------------------------------
# Direct shard and bundle lookup
# ---------------------------------------------------------------------------


def expected_shard_index_for_document_index(document_index: int, documents_per_shard: int) -> int:
    """Return the shard index for one document based on fixed shard size."""
    # The current run uses 50 documents per shard, configurable for future runs.
    return int(document_index) // int(documents_per_shard)


def expected_shard_directory_for_document_index(
    shards_dir: Path,
    document_index: int,
    documents_per_shard: int,
) -> Path:
    """Return the exact expected shard directory for one document index."""
    # Compute the inclusive document range encoded in shard folder names.
    shard_index = expected_shard_index_for_document_index(document_index, documents_per_shard)
    shard_start = shard_index * documents_per_shard
    shard_end = shard_start + documents_per_shard - 1

    # Current shard names follow shard_009_docs_000450_000499.
    return Path(shards_dir) / f"shard_{shard_index:03d}_docs_{shard_start:06d}_{shard_end:06d}"


def load_document_metadata_json(metadata_json_path: Path) -> dict[str, Any] | None:
    """Load one document_metadata.json file, returning None when it is not ready."""
    try:
        with Path(metadata_json_path).open("r", encoding="utf-8") as metadata_handle:
            return json.load(metadata_handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # Still-running jobs may briefly expose missing or incomplete files.
        return None


def bundle_info_from_metadata_path(metadata_json_path: Path, shard_index: int) -> dict[str, Any] | None:
    """Create compact bundle metadata from one document_metadata.json file."""
    # Load lightweight document metadata before touching threshold record files.
    metadata_payload = load_document_metadata_json(metadata_json_path)
    if metadata_payload is None:
        return None

    # The bundle logger writes document-level metadata under the document key.
    document_payload = metadata_payload.get("document", {}) if isinstance(metadata_payload, dict) else {}
    document_index = safe_int_or_none(document_payload.get("index"))
    if document_index is None:
        return None

    # Store only fields needed for filtering, plotting, and table output.
    return {
        "document_index": document_index,
        "fname": document_payload.get("fname"),
        "bundle_dir": metadata_json_path.parent,
        "metadata_json_path": metadata_json_path,
        "shard_index": shard_index,
        "whole_document_nls": safe_float_or_nan(document_payload.get("whole_document_nls")),
        "bundle_record_format": metadata_payload.get("record_format"),
        "skip_record": metadata_payload.get("skip_record", {}) if isinstance(metadata_payload, dict) else {},
        "ref_to_pred_matrix_shape": tuple(document_payload.get("ref_to_pred_matrix_shape") or []),
        "ref_to_ref_matrix_shape": tuple(document_payload.get("ref_to_ref_matrix_shape") or []),
    }


def find_document_bundle_info(
    *,
    shards_dir: Path,
    document_index: int,
    document_file_name: str,
    documents_per_shard: int,
) -> dict[str, Any] | None:
    """Find bundle metadata for one document by probing direct and sharded layouts."""
    # Compute the shard directly; this avoids scanning all shard directories for
    # the normal multi-shard output layout.
    shard_index = expected_shard_index_for_document_index(document_index, documents_per_shard)
    shards_dir = Path(shards_dir)
    expected_shard_dir = expected_shard_directory_for_document_index(shards_dir, document_index, documents_per_shard)

    # Single-job tuner runs write <output_dir>/combination_bundles, while
    # multi-shard runs write <output_dir>/shards/shard_XXX.../combination_bundles.
    # Supporting both layouts lets the same visualisation code run from the
    # Python entrypoint and from the final afterok shard visualisation job.
    candidate_bundle_roots = []
    if shards_dir.name == "combination_bundles":
        candidate_bundle_roots.append(shards_dir)
    candidate_bundle_roots.append(shards_dir / "combination_bundles")
    candidate_bundle_roots.append(expected_shard_dir / "combination_bundles")
    # Dynamic-pool runs do not know ahead of time which Slurm worker will claim
    # a document.  Each worker still writes the normal combination_bundles
    # layout, so visualisation only needs to look through the worker folders.
    candidate_bundle_roots.extend(sorted(shards_dir.glob("dynamic_*/combination_bundles")))

    seen_bundle_roots: set[Path] = set()
    for combination_bundle_dir in candidate_bundle_roots:
        combination_bundle_dir = Path(combination_bundle_dir)
        if combination_bundle_dir in seen_bundle_roots or not combination_bundle_dir.is_dir():
            continue
        seen_bundle_roots.add(combination_bundle_dir)

        # First try the exact expected document folder name.
        exact_document_dir = combination_bundle_dir / f"document_{document_index:06d}_{document_file_name}"
        exact_metadata_path = exact_document_dir / "document_metadata.json"
        if exact_metadata_path.exists():
            return bundle_info_from_metadata_path(exact_metadata_path, shard_index)

        # Fallback only inside the expected bundle root in case filenames were
        # sanitized differently by the bundle logger.
        candidate_metadata_paths = sorted(
            combination_bundle_dir.glob(f"document_{document_index:06d}_*/document_metadata.json")
        )
        for candidate_metadata_path in candidate_metadata_paths:
            bundle_info = bundle_info_from_metadata_path(candidate_metadata_path, shard_index)
            if bundle_info is not None:
                return bundle_info

    # No metadata file was found for this document in its expected shard.
    return None


def iter_skipped_document_csv_paths(shards_dir: Path) -> list[Path]:
    """Return skipped-document CSV paths from single-run, shard, and dynamic layouts."""
    shards_dir = Path(shards_dir)
    candidate_paths = [
        shards_dir / "csv" / "skipped_documents.csv",
        shards_dir / "skipped_documents.csv",
    ]
    candidate_paths.extend(sorted(shards_dir.glob("*/csv/skipped_documents.csv")))
    candidate_paths.extend(sorted(shards_dir.glob("shards/*/csv/skipped_documents.csv")))

    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for candidate_path in candidate_paths:
        candidate_path = Path(candidate_path)
        if candidate_path in seen_paths or not candidate_path.exists():
            continue
        seen_paths.add(candidate_path)
        unique_paths.append(candidate_path)
    return unique_paths


def normalize_skipped_document_record(raw_record: dict[str, Any], skipped_csv_path: Path) -> dict[str, Any] | None:
    """Normalize one skipped-document CSV row for visualisation lookup."""
    document_index = safe_int_or_none(raw_record.get("document_index"))
    if document_index is None:
        document_index = safe_int_or_none(raw_record.get("index"))
    if document_index is None:
        return None

    normalized_record = {key: value for key, value in raw_record.items() if value not in (None, "")}
    normalized_record["document_index"] = int(document_index)
    normalized_record["fname"] = Path(str(normalized_record.get("fname", ""))).name
    normalized_record["skip_reason"] = str(normalized_record.get("skip_reason", "unknown"))
    normalized_record["skipped_csv_path"] = str(skipped_csv_path)

    diagnostic_bundle_dir = normalized_record.get("diagnostic_bundle_dir")
    if diagnostic_bundle_dir not in (None, ""):
        normalized_record["diagnostic_bundle_dir"] = str(diagnostic_bundle_dir)
    return normalized_record


def load_skipped_document_records_by_index(shards_dir: Path) -> dict[int, dict[str, Any]]:
    """Load skipped-document records written by tuner workers, keyed by document index."""
    skipped_records_by_index: dict[int, dict[str, Any]] = {}
    for skipped_csv_path in iter_skipped_document_csv_paths(shards_dir):
        try:
            skipped_dataframe = pd.read_csv(skipped_csv_path, dtype=str).fillna("")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue

        for raw_record in skipped_dataframe.to_dict(orient="records"):
            normalized_record = normalize_skipped_document_record(raw_record, skipped_csv_path)
            if normalized_record is None:
                continue
            skipped_records_by_index[int(normalized_record["document_index"])] = normalized_record
    return skipped_records_by_index


def skipped_document_from_tuner_record(
    *,
    runfile_document: dict[str, Any],
    skipped_record: dict[str, Any],
    documents_per_shard: int,
) -> dict[str, Any]:
    """Merge runfile metadata with a worker-written skipped-document record."""
    document_index = int(runfile_document["document_index"])
    skipped_document = dict(runfile_document)
    skipped_document.update(skipped_record)
    skipped_document["document_index"] = document_index
    skipped_document["fname"] = Path(str(runfile_document.get("fname", skipped_record.get("fname", "")))).name
    skipped_document["shard_index"] = safe_int_or_none(skipped_record.get("shard_index"))
    if skipped_document["shard_index"] is None:
        skipped_document["shard_index"] = expected_shard_index_for_document_index(
            document_index,
            documents_per_shard,
        )

    diagnostic_bundle_dir = skipped_record.get("diagnostic_bundle_dir")
    if diagnostic_bundle_dir not in (None, ""):
        skipped_document["diagnostic_bundle_dir"] = str(diagnostic_bundle_dir)
        skipped_document["bundle_dir"] = str(diagnostic_bundle_dir)
    return skipped_document


def skipped_document_from_diagnostic_bundle(
    *,
    runfile_document: dict[str, Any],
    bundle_info: dict[str, Any],
) -> dict[str, Any]:
    """Build a skipped-document row from a diagnostic bundle metadata file."""
    skipped_record = bundle_info.get("skip_record", {})
    if not isinstance(skipped_record, dict):
        skipped_record = {}

    skipped_document = dict(runfile_document)
    skipped_document.update(skipped_record)
    skipped_document.update(bundle_info)
    skipped_document["document_index"] = int(runfile_document["document_index"])
    skipped_document["fname"] = Path(str(runfile_document.get("fname", bundle_info.get("fname", "")))).name
    skipped_document["skip_reason"] = str(skipped_document.get("skip_reason", "skipped_document_diagnostic"))
    skipped_document["diagnostic_bundle_dir"] = str(bundle_info["bundle_dir"])
    skipped_document["bundle_dir"] = str(bundle_info["bundle_dir"])
    return skipped_document


def ref_to_pred_shape_has_prediction_windows(matrix_shape: Any) -> bool:
    """Return True when a ref_to_pred matrix shape has at least one prediction column."""
    # Missing shape metadata may come from older bundles; do not hide those documents.
    if matrix_shape is None:
        return True

    # Normalize tuple/list/NumPy-like shapes into a plain list.
    shape_values = list(matrix_shape)

    # Unexpected shape metadata should not cause accidental document removal.
    if len(shape_values) < 2:
        return True

    # In ref_to_pred matrices, the second dimension is the prediction/other-window axis.
    prediction_window_count = safe_int_or_none(shape_values[1])
    if prediction_window_count is None:
        return True

    # Zero columns means no prediction windows, so there is nothing useful to plot or score.
    return prediction_window_count > 0


def shape_values_from_record_value(matrix_shape: Any) -> list[int]:
    """Return integer matrix-shape values from tuples, lists, or CSV text."""
    if matrix_shape is None:
        return []

    if isinstance(matrix_shape, str):
        return [int(value) for value in re.findall(r"-?\d+", matrix_shape)]

    try:
        return [int(value) for value in list(matrix_shape)]
    except (TypeError, ValueError):
        return []


def prediction_window_count_from_document_record(document_record: dict[str, Any]) -> int | None:
    """Return the ref_to_pred prediction-window count when the record exposes it."""
    prediction_window_count = safe_int_or_none(document_record.get("ref_to_pred_matrix_cols"))
    if prediction_window_count is not None:
        return prediction_window_count

    shape_values = shape_values_from_record_value(document_record.get("ref_to_pred_matrix_shape"))
    if len(shape_values) >= 2:
        return int(shape_values[1])

    return None


def skipped_document_has_prediction_text_or_windows(skipped_document: dict[str, Any]) -> bool:
    """Return True only when a skipped diagnostic can still show a prediction-side matrix."""
    skip_reason = str(skipped_document.get("skip_reason", ""))
    if skip_reason in {"no_prediction_text", "no_ref_to_pred_prediction_windows"}:
        return False

    non_whitespace_prediction_characters = safe_int_or_none(
        skipped_document.get("prediction_non_whitespace_character_count")
    )
    if non_whitespace_prediction_characters is not None and non_whitespace_prediction_characters <= 0:
        return False

    prediction_window_count = prediction_window_count_from_document_record(skipped_document)
    if prediction_window_count is not None and prediction_window_count <= 0:
        return False

    return True


def split_documents_by_bundle_availability_and_prediction(
    *,
    runfile_documents: list[dict[str, Any]],
    shards_dir: Path,
    documents_per_shard: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split selected runfile documents into loadable and skipped documents."""
    # Loadable documents have a successful combination bundle and at least one prediction column.
    loadable_documents: list[dict[str, Any]] = []

    # Skipped documents may come from worker CSVs or from diagnostic bundles.
    skipped_documents: list[dict[str, Any]] = []
    skipped_records_by_index = load_skipped_document_records_by_index(shards_dir)

    # Probe only each selected document's expected shard.
    for runfile_document in runfile_documents:
        document_index = int(runfile_document["document_index"])
        document_file_name = str(runfile_document["fname"])
        bundle_info = find_document_bundle_info(
            shards_dir=shards_dir,
            document_index=document_index,
            document_file_name=document_file_name,
            documents_per_shard=documents_per_shard,
        )

        # Missing successful bundles can mean either a real skipped document or a run that has
        # not processed the document yet.  Prefer the worker-written skip reason when available.
        if bundle_info is None:
            skipped_record = skipped_records_by_index.get(document_index)
            if skipped_record is not None:
                skipped_documents.append(
                    skipped_document_from_tuner_record(
                        runfile_document=runfile_document,
                        skipped_record=skipped_record,
                        documents_per_shard=documents_per_shard,
                    )
                )
                continue

            skipped_document = dict(runfile_document)
            skipped_document["shard_index"] = expected_shard_index_for_document_index(
                document_index,
                documents_per_shard,
            )
            skipped_document["skip_reason"] = "bundle_folder_not_available_yet"
            skipped_documents.append(skipped_document)
            continue

        # Skipped diagnostic bundles are intentionally not loadable combination bundles.
        if str(bundle_info.get("bundle_record_format")) == "skipped_document_diagnostic":
            skipped_documents.append(
                skipped_document_from_diagnostic_bundle(
                    runfile_document=runfile_document,
                    bundle_info=bundle_info,
                )
            )
            continue

        # No-prediction documents cannot produce meaningful ref_to_pred plots or metrics tables.
        if not ref_to_pred_shape_has_prediction_windows(bundle_info.get("ref_to_pred_matrix_shape")):
            skipped_document = dict(runfile_document)
            skipped_document.update(bundle_info)
            skipped_document["skip_reason"] = "no_ref_to_pred_prediction_windows"
            skipped_documents.append(skipped_document)
            continue

        # Merge runfile metadata with bundle metadata for downstream loading.
        combined_document_info = dict(runfile_document)
        combined_document_info.update(bundle_info)
        loadable_documents.append(combined_document_info)

    return loadable_documents, skipped_documents


# ---------------------------------------------------------------------------
# Combination record streaming and compact DataFrame creation
# ---------------------------------------------------------------------------


def threshold_value_from_record_path(record_path: Path) -> int:
    """Extract the numeric threshold value from threshold bundle filenames."""
    # Support the new binary stream plus the older JSONL formats.
    match = re.search(r"threshold_(\d+)(?:\.pklstream|\.jsonl(?:\.gz)?)$", Path(record_path).name)
    if match is None:
        return 10**9
    return int(match.group(1))


def iter_threshold_record_paths_for_document_bundle(document_bundle_dir: Path) -> list[Path]:
    """Return one preferred threshold record file per threshold in stable order."""
    # Prefer the new binary stream when both new and legacy files exist, because
    # loading both would duplicate every combination in the metrics DataFrame.
    preferred_path_by_threshold: dict[int, Path] = {}
    preferred_rank_by_threshold: dict[int, int] = {}
    suffix_rank_by_suffix = {
        PICKLE_STREAM_SUFFIX: 0,
        JSONL_SUFFIX: 1,
        GZIP_JSONL_SUFFIX: 2,
    }

    for record_path in Path(document_bundle_dir).glob("threshold_*.pklstream"):
        threshold_value = threshold_value_from_record_path(record_path)
        preferred_path_by_threshold[threshold_value] = record_path
        preferred_rank_by_threshold[threshold_value] = suffix_rank_by_suffix[PICKLE_STREAM_SUFFIX]

    for record_path in Path(document_bundle_dir).glob("threshold_*.jsonl"):
        threshold_value = threshold_value_from_record_path(record_path)
        candidate_rank = suffix_rank_by_suffix[JSONL_SUFFIX]
        if candidate_rank < preferred_rank_by_threshold.get(threshold_value, 10):
            preferred_path_by_threshold[threshold_value] = record_path
            preferred_rank_by_threshold[threshold_value] = candidate_rank

    for record_path in Path(document_bundle_dir).glob("threshold_*.jsonl.gz"):
        threshold_value = threshold_value_from_record_path(record_path)
        candidate_rank = suffix_rank_by_suffix[GZIP_JSONL_SUFFIX]
        if candidate_rank < preferred_rank_by_threshold.get(threshold_value, 10):
            preferred_path_by_threshold[threshold_value] = record_path
            preferred_rank_by_threshold[threshold_value] = candidate_rank

    # Sort numerically by threshold so plotting and tie-breaking are reproducible.
    return [preferred_path_by_threshold[threshold_value] for threshold_value in sorted(preferred_path_by_threshold)]


def open_jsonl_text_for_reading(jsonl_path: Path):
    """Open a plain or gzipped JSONL file in text mode."""
    # Legacy bundle files may be plain JSONL or gzipped JSONL.  Current runs
    # prefer pickle streams and reach this helper only for older outputs.
    if str(jsonl_path).endswith(".gz"):
        return gzip.open(jsonl_path, mode="rt", encoding="utf-8")
    return Path(jsonl_path).open("r", encoding="utf-8")


def iter_combination_records_for_document(document_info: dict[str, Any]):
    """Yield full combination records for one document with their source location."""
    # Each document bundle contains one preferred record file per threshold value.
    document_bundle_dir = Path(document_info["bundle_dir"])

    # Stream records; do not load full geometry for all combinations at once.
    for record_path in iter_threshold_record_paths_for_document_bundle(document_bundle_dir):
        yield from iter_combination_records_from_path(record_path)


def iter_combination_records_from_path(record_path: Path):
    """Yield combination records from one threshold stream with stable record numbers."""
    # Pickle streams use one-based record numbers in the same role that JSONL
    # line numbers used before: a stable pointer to a selected geometry record.
    if str(record_path).endswith(PICKLE_STREAM_SUFFIX):
        for combination_record, record_number in iter_pickle_stream_records(record_path):
            yield combination_record, Path(record_path), int(record_number)
        return

    with open_jsonl_text_for_reading(record_path) as jsonl_handle:
        for line_number, line in enumerate(jsonl_handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Skip partial JSON lines that can exist while a job is still writing.
            try:
                combination_record = json.loads(stripped_line)
            except json.JSONDecodeError:
                continue

            yield combination_record, Path(record_path), int(line_number)


def combination_score_table_path_for_document(document_info: dict[str, Any]) -> Path:
    """Return the shard-local compact score-table path for one bundle document."""
    document_bundle_dir = Path(document_info["bundle_dir"])
    shard_output_dir = document_bundle_dir.parent.parent
    return shard_output_dir / COMBINATION_SCORE_TABLE_FILENAME


def read_score_table_for_visualization(
    score_table_path: Path,
    score_table_cache_by_path: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    """Load the scalar score table columns needed by visual diagnostics."""
    cached_score_table = score_table_cache_by_path.get(score_table_path)
    if cached_score_table is not None:
        return cached_score_table

    print(f"[score-table] source=combination_scores path={score_table_path}")
    score_table_dataframe = pd.read_csv(
        score_table_path,
        compression="gzip",
        low_memory=False,
        usecols=lambda column_name: column_name in SCORE_TABLE_COLUMNS_REQUIRED_FOR_VISUALS,
    )
    score_table_cache_by_path[score_table_path] = score_table_dataframe
    return score_table_dataframe


def extract_compact_metric_row_from_score_table_row(
    *,
    score_row: pd.Series,
    source_jsonl_path: Path | None,
    source_line_number: int | None,
    document_info: dict[str, Any],
) -> dict[str, Any]:
    """Convert one compact score-table row into the visual tool DataFrame schema."""
    hallucination = safe_float_or_nan(score_row.get("hallucination"))
    non_hallucination = float("nan") if math.isnan(hallucination) else max(0.0, min(1.0, 1.0 - hallucination))
    hough_seed = safe_int_or_none(score_row.get("hough_seed"))
    document_index = safe_int_or_none(score_row.get("doc_index"))
    effective_hough_seed = None if hough_seed is None else int(hough_seed)

    return {
        "main_language": document_info.get("main_language"),
        "document_type": document_info.get("document_type"),
        "document_index": document_index,
        "fname": score_row.get("fname") or document_info.get("fname"),
        "bundle_dir": str(document_info["bundle_dir"]),
        "shard_index": document_info.get("shard_index"),
        "whole_document_nls": safe_float_or_nan(score_row.get("whole_document_nls")),
        "hough_threshold": safe_int_or_none(score_row.get("hough_threshold")),
        "hough_line_length": safe_int_or_none(score_row.get("hough_line_length")),
        "hough_line_gap": safe_int_or_none(score_row.get("hough_line_gap")),
        "hough_seed": hough_seed,
        "effective_hough_seed": effective_hough_seed,
        "tuning_score": safe_float_or_nan(score_row.get("tuning_score")),
        "selection_objective": score_row.get("selection_objective"),
        "alignment_selection_score": safe_float_or_nan(score_row.get("alignment_selection_score")),
        "score_matrix_support": safe_float_or_nan(score_row.get("score_matrix_support")),
        "line_guided_fraction": safe_float_or_nan(score_row.get("line_guided_fraction")),
        "weighted_along_lines_nls": safe_float_or_nan(score_row.get("weighted_along_lines_nls")),
        "correct_ref_coverage": safe_float_or_nan(score_row.get("correct_ref_coverage")),
        "missing_ref_coverage": safe_float_or_nan(score_row.get("missing_ref_coverage")),
        "repetition_on_ref": safe_float_or_nan(score_row.get("repetition_on_ref")),
        "hallucination": hallucination,
        "non_hallucination": non_hallucination,
        "raw_line_count": safe_float_or_nan(score_row.get("raw_line_count")),
        "candidate_line_count": safe_float_or_nan(score_row.get("candidate_line_count")),
        "used_line_count": safe_float_or_nan(score_row.get("used_line_count")),
        "line_guided_columns": safe_float_or_nan(score_row.get("line_guided_columns")),
        "fallback_columns": safe_float_or_nan(score_row.get("fallback_columns")),
        "is_valid": bool(safe_int_or_none(score_row.get("is_valid"))),
        "invalid_reason": score_row.get("invalid_reason"),
        "source_jsonl_path": "" if source_jsonl_path is None else str(source_jsonl_path),
        "source_line_number": 0 if source_line_number is None else int(source_line_number),
    }


def extract_compact_metric_row_from_combination_record(
    *,
    combination_record: dict[str, Any],
    source_jsonl_path: Path,
    source_line_number: int,
    document_info: dict[str, Any],
) -> dict[str, Any]:
    """Convert one full combination record into one compact scalar DataFrame row."""
    # Pull nested blocks safely so missing values become NaN/None instead of crashes.
    document_payload = combination_record.get("document", {}) or {}
    hough_parameters = combination_record.get("hough_parameters", {}) or {}
    metrics = combination_record.get("metrics", combination_record.get("eval", {})) or {}

    # The tuner harmonic score uses 1 - hallucination, not hallucination itself.
    hallucination = safe_float_or_nan(metrics.get("hallucination"))
    non_hallucination = float("nan") if math.isnan(hallucination) else max(0.0, min(1.0, 1.0 - hallucination))

    # Keep only scalar values in the DataFrame; geometry is reloaded only for the best visual.
    return {
        "main_language": document_info.get("main_language"),
        "document_type": document_info.get("document_type"),
        "document_index": safe_int_or_none(document_payload.get("index")),
        "fname": document_payload.get("fname") or document_info.get("fname"),
        "bundle_dir": str(document_info["bundle_dir"]),
        "shard_index": document_info.get("shard_index"),
        "whole_document_nls": safe_float_or_nan(document_payload.get("whole_document_nls")),
        "hough_threshold": safe_int_or_none(hough_parameters.get("hough_threshold")),
        "hough_line_length": safe_int_or_none(hough_parameters.get("hough_line_length")),
        "hough_line_gap": safe_int_or_none(hough_parameters.get("hough_line_gap")),
        "hough_seed": safe_int_or_none(hough_parameters.get("hough_seed")),
        "effective_hough_seed": safe_int_or_none(hough_parameters.get("effective_hough_seed")),
        "tuning_score": safe_float_or_nan(metrics.get("tuning_score")),
        "selection_objective": metrics.get("selection_objective"),
        "alignment_selection_score": safe_float_or_nan(metrics.get("alignment_selection_score")),
        "score_matrix_support": safe_float_or_nan(metrics.get("score_matrix_support")),
        "line_guided_fraction": safe_float_or_nan(metrics.get("line_guided_fraction")),
        "weighted_along_lines_nls": safe_float_or_nan(metrics.get("weighted_along_lines_nls")),
        "correct_ref_coverage": safe_float_or_nan(metrics.get("correct_ref_coverage")),
        "missing_ref_coverage": safe_float_or_nan(metrics.get("missing_ref_coverage")),
        "repetition_on_ref": safe_float_or_nan(metrics.get("repetition_on_ref")),
        "hallucination": hallucination,
        "non_hallucination": non_hallucination,
        "raw_line_count": safe_float_or_nan(metrics.get("raw_line_count")),
        "candidate_line_count": safe_float_or_nan(metrics.get("candidate_line_count")),
        "used_line_count": safe_float_or_nan(metrics.get("used_line_count")),
        "line_guided_columns": safe_float_or_nan(metrics.get("line_guided_columns")),
        "fallback_columns": safe_float_or_nan(metrics.get("fallback_columns")),
        "is_valid": bool(metrics.get("is_valid", False)),
        "invalid_reason": metrics.get("invalid_reason"),
        "source_jsonl_path": str(source_jsonl_path),
        "source_line_number": int(source_line_number),
    }


def load_language_document_type_metrics_dataframe(
    *,
    runfile_documents: list[dict[str, Any]],
    shards_dir: Path,
    documents_per_shard: int,
    max_documents: int | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load one compact DataFrame for one selected language/document-type pair."""
    # Find loadable documents before reading any threshold record streams.
    loadable_documents, skipped_documents = split_documents_by_bundle_availability_and_prediction(
        runfile_documents=runfile_documents,
        shards_dir=shards_dir,
        documents_per_shard=documents_per_shard,
    )

    # The user-controlled max_documents applies to loadable documents only.
    documents_to_load = loadable_documents if max_documents is None else loadable_documents[: int(max_documents)]

    # Store compact scalar rows only; never retain all full geometry records.
    compact_rows: list[dict[str, Any]] = []
    score_table_cache_by_path: dict[Path, pd.DataFrame] = {}

    # Load each selected document in order.
    for document_position, document_info in enumerate(documents_to_load, start=1):
        print(
            f"[{document_info['main_language']} / {document_info['document_type']}] "
            f"loading {document_position}/{len(documents_to_load)}: "
            f"document_index={document_info['document_index']}, fname={document_info['fname']}"
        )

        score_table_path = combination_score_table_path_for_document(document_info)
        if score_table_path.exists():
            score_table_dataframe = read_score_table_for_visualization(
                score_table_path,
                score_table_cache_by_path,
            )

            if "doc_index" not in score_table_dataframe.columns:
                print(f"[score-table] missing doc_index column; falling back to bundle records: {score_table_path}")
            else:
                document_index_values = pd.to_numeric(score_table_dataframe["doc_index"], errors="coerce")
                document_score_rows = score_table_dataframe[
                    document_index_values == int(document_info["document_index"])
                ]
                if not document_score_rows.empty:
                    for _, score_row in document_score_rows.iterrows():
                        compact_rows.append(
                            extract_compact_metric_row_from_score_table_row(
                                score_row=score_row,
                                source_jsonl_path=None,
                                source_line_number=None,
                                document_info=document_info,
                            )
                        )
                    continue

        # Legacy fallback: stream every geometry record and reduce it to scalar fields.
        print(f"[score-table] source=combination_bundles_legacy document={document_info['document_index']}")
        for combination_record, source_jsonl_path, source_line_number in iter_combination_records_for_document(document_info):
            compact_rows.append(
                extract_compact_metric_row_from_combination_record(
                    combination_record=combination_record,
                    source_jsonl_path=source_jsonl_path,
                    source_line_number=source_line_number,
                    document_info=document_info,
                )
            )

    # Build exactly one DataFrame for the language/document-type pair.
    # Supplying columns keeps empty language/type pairs schema-stable.
    metrics_dataframe = pd.DataFrame.from_records(compact_rows, columns=COMPACT_METRIC_COLUMNS)

    # Stable ordering makes CSVs and best-row tie-breaking reproducible.
    if not metrics_dataframe.empty:
        metrics_dataframe = metrics_dataframe.sort_values(
            [
                "main_language",
                "document_type",
                "document_index",
                "hough_threshold",
                "hough_line_length",
                "hough_line_gap",
                "hough_seed",
            ],
            kind="stable",
        ).reset_index(drop=True)

    return metrics_dataframe, loadable_documents, skipped_documents, documents_to_load


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------


def language_document_type_output_dir(output_dir: Path, language_name: str, document_type: str) -> Path:
    """Return the output directory for one language/document-type pair."""
    # Folder names are derived from actual runfile values and sanitized only for filesystem safety.
    return Path(output_dir) / safe_path_component(language_name) / safe_path_component(document_type)


def plots_output_dir(output_dir: Path) -> Path:
    """Return the flat visualisation directory used by the tuner-integrated output."""
    return Path(output_dir) / PLOTS_DIRECTORY_NAME


def document_plots_output_dir(output_dir: Path) -> Path:
    """Return the flat directory for per-document parameter/metric graph grids."""
    return plots_output_dir(output_dir) / DOCUMENT_PLOTS_DIRECTORY_NAME


def document_output_stem_from_row(best_or_metric_row: pd.Series) -> str:
    """Return the stable document stem shared by folders and document-specific filenames."""
    # Keep the index first so lexicographic order matches the original runfile order.
    document_index = int(best_or_metric_row["document_index"])
    document_name = str(best_or_metric_row["fname"])
    return f"document_{document_index:06d}_{safe_path_component(document_name)}"


def document_output_dir_from_row(output_dir: Path, best_or_metric_row: pd.Series) -> Path:
    """Return the output directory for one document row."""
    # All per-document outputs are grouped by language and document_type first.
    language_name = str(best_or_metric_row["main_language"])
    document_type = str(best_or_metric_row["document_type"])

    # The leaf folder includes document index and filename for easy manual navigation.
    return language_document_type_output_dir(Path(output_dir), language_name, document_type) / (
        document_output_stem_from_row(best_or_metric_row)
    )


def parameter_metric_grid_output_path_from_row(output_dir: Path, metric_row: pd.Series) -> Path:
    """Return the per-document line-graph grid path, including the document name in the filename."""
    document_output_dir = document_plots_output_dir(output_dir)
    output_filename = (
        f"{safe_path_component(metric_row['main_language'])}_"
        f"{document_output_stem_from_row(metric_row)}_parameter_metric_line_graph_grid.png"
    )
    return document_output_dir / output_filename


def stitched_best_combination_metrics_json_path(output_dir: Path) -> Path:
    """Return the single human-readable JSON report path for all stitched best outputs."""
    return plots_output_dir(output_dir) / STITCHED_BEST_COMBINATION_METRICS_JSON_FILENAME


def stitched_best_combination_metrics_path_for_pair(output_dir: Path, language_name: str, document_type: str) -> Path:
    """Return the pair-level stitched metrics text path."""
    return (
        language_document_type_output_dir(Path(output_dir), language_name, document_type)
        / STITCHED_BEST_COMBINATION_METRICS_FILENAME
    )


def stitched_best_combination_panel_path_for_pair(output_dir: Path, language_name: str, document_type: str) -> Path:
    """Return the pair-level stitched best-combination panel path."""
    output_filename = (
        f"stitched_best_combination_{safe_path_component(language_name)}_"
        f"{safe_path_component(document_type)}_documents.png"
    )
    return plots_output_dir(output_dir) / output_filename


def temporary_best_combination_panel_dir_for_pair(output_dir: Path, language_name: str, document_type: str) -> Path:
    """Return the hidden directory used while building the stitched panel."""
    return (
        plots_output_dir(output_dir)
        / TEMPORARY_BEST_COMBINATION_PANEL_DIRECTORY_NAME
        / safe_path_component(language_name)
        / safe_path_component(document_type)
    )


def remove_stale_per_document_visual_analysis_artifacts_for_pair(
    output_dir: Path,
    language_name: str,
    document_type: str,
) -> int:
    """Remove legacy per-document best outputs that are now replaced by stitched outputs."""
    pair_output_dir = language_document_type_output_dir(Path(output_dir), language_name, document_type)
    if not pair_output_dir.is_dir():
        return 0

    stale_filename_set = {
        # These two files are now represented by pair-level stitched outputs.
        PER_DOCUMENT_BEST_COMBINATION_METRICS_FILENAME,
        PER_DOCUMENT_BEST_COMBINATION_PANEL_FILENAME,
        # The graph grid still exists per document, but its filename now includes the document name.
        LEGACY_PARAMETER_METRIC_LINE_GRAPH_GRID_FILENAME,
    }
    removed_file_count = 0
    for document_output_dir in sorted(path for path in pair_output_dir.glob("document_*") if path.is_dir()):
        for stale_filename in stale_filename_set:
            stale_output_path = document_output_dir / stale_filename
            if stale_output_path.is_file():
                stale_output_path.unlink()
                removed_file_count += 1

    return removed_file_count


# ---------------------------------------------------------------------------
# Parameter and metric line graph grids
# ---------------------------------------------------------------------------


def build_mean_line_series_for_parameter(
    document_metrics_dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    """Return mean metric values grouped by one discrete Hough parameter."""
    # Keep only the two columns needed for this graph.
    plotting_dataframe = document_metrics_dataframe[[x_column, y_column]].dropna()
    if plotting_dataframe.empty:
        return pd.DataFrame(columns=[x_column, y_column])

    # Hough parameters are discrete integer values, so exact grouping is meaningful.
    return (
        plotting_dataframe.groupby(x_column, dropna=True)[y_column]
        .mean()
        .reset_index()
        .sort_values(x_column, kind="stable")
    )


def build_mean_line_series_for_continuous_component(
    document_metrics_dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    max_bins: int,
) -> pd.DataFrame:
    """Return a readable line series for score versus one continuous component."""
    # Drop missing and infinite values before binning.
    plotting_dataframe = document_metrics_dataframe[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if plotting_dataframe.empty:
        return pd.DataFrame(columns=[x_column, y_column])

    # Exact grouping is clearest when the component has only a few distinct values.
    unique_x_values = np.sort(plotting_dataframe[x_column].unique())
    if len(unique_x_values) <= max_bins:
        return (
            plotting_dataframe.groupby(x_column, dropna=True)[y_column]
            .mean()
            .reset_index()
            .sort_values(x_column, kind="stable")
        )

    # Bin dense continuous values so the result is a readable line graph.
    x_min = float(plotting_dataframe[x_column].min())
    x_max = float(plotting_dataframe[x_column].max())
    if math.isclose(x_min, x_max):
        return pd.DataFrame({x_column: [x_min], y_column: [float(plotting_dataframe[y_column].mean())]})

    # Evenly spaced bins are enough here because this is an exploratory diagnostic plot.
    bin_edges = np.linspace(x_min, x_max, max_bins + 1)
    bin_labels = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    binned_component = pd.cut(
        plotting_dataframe[x_column],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        duplicates="drop",
    )

    # Average tuning_score within each component bin.
    line_series = (
        plotting_dataframe.assign(_component_bin=binned_component)
        .groupby("_component_bin", observed=True)[y_column]
        .mean()
        .reset_index()
        .rename(columns={"_component_bin": x_column})
    )
    line_series[x_column] = line_series[x_column].astype(float)
    return line_series.sort_values(x_column, kind="stable")


def draw_line_graph_on_axis(
    axis,
    line_series_dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    force_unit_y_axis: bool = True,
) -> None:
    """Draw one labeled line graph on a matplotlib axis."""
    # Make empty graphs explicit so missing metrics are not confused with code failure.
    if line_series_dataframe.empty:
        axis.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        return

    # Draw a line with markers so discrete parameter values are visible.
    axis.plot(
        line_series_dataframe[x_column],
        line_series_dataframe[y_column],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        color="#174A7C",
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25)

    # Most tuning and harmonic-component values are normalized into [0, 1].
    if force_unit_y_axis:
        axis.set_ylim(-0.02, 1.02)


def draw_hough_parameter_row(
    *,
    axes_row,
    document_metrics_dataframe: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    force_unit_y_axis: bool,
) -> None:
    """Draw one row of metric-vs-Hough-parameter line graphs."""
    # Every row has one graph per Hough parameter.
    for column_index, (parameter_column, parameter_label) in enumerate(HOUGH_PARAMETER_COLUMNS):
        line_series = build_mean_line_series_for_parameter(document_metrics_dataframe, parameter_column, metric_column)
        draw_line_graph_on_axis(
            axes_row[column_index],
            line_series,
            parameter_column,
            metric_column,
            f"Mean {metric_label} by {parameter_label}",
            parameter_label,
            f"Mean {metric_label}",
            force_unit_y_axis=force_unit_y_axis,
        )


def plot_single_document_parameter_metric_grid(
    document_metrics_dataframe: pd.DataFrame,
    output_dir: Path,
    max_continuous_bins: int,
    saved_figure_dpi: int,
) -> Path:
    """Create the 18-graph parameter/metric grid for one document."""
    # Use the first row for stable document labels and output paths.
    first_row = document_metrics_dataframe.iloc[0]
    language_name = str(first_row["main_language"])
    document_type = str(first_row["document_type"])
    document_index = int(first_row["document_index"])
    document_name = str(first_row["fname"])

    # Save per-document grids in one flat plots/documents directory so users can
    # browse all graph grids without drilling into language/type folders.
    document_output_dir = document_plots_output_dir(output_dir)
    document_output_dir.mkdir(parents=True, exist_ok=True)

    # The requested grid has 18 line graphs: 3 + 3 + 9 + 3 survived-line graphs.
    fig, axes = plt.subplots(6, 3, figsize=(21, 28), constrained_layout=False)
    fig.suptitle(
        f"{language_name} / {document_type} | document {document_index} | {document_name}\n"
        "Hough parameter influence, harmonic-component influence, and surviving-line counts",
        fontsize=16,
        y=0.995,
    )

    # Row 1: tuning_score versus each Hough parameter.
    draw_hough_parameter_row(
        axes_row=axes[0],
        document_metrics_dataframe=document_metrics_dataframe,
        metric_column="tuning_score",
        metric_label="tuning_score",
        force_unit_y_axis=True,
    )

    # Row 2: tuning_score versus each harmonic component.
    for column_index, (component_column, component_label) in enumerate(HARMONIC_COMPONENT_COLUMNS):
        line_series = build_mean_line_series_for_continuous_component(
            document_metrics_dataframe,
            component_column,
            "tuning_score",
            max_continuous_bins,
        )
        draw_line_graph_on_axis(
            axes[1, column_index],
            line_series,
            component_column,
            "tuning_score",
            f"Mean tuning_score by {component_label}",
            component_label,
            "Mean tuning_score",
            force_unit_y_axis=True,
        )

    # Rows 3-5: every harmonic component versus every Hough parameter.
    for row_index, (component_column, component_label) in enumerate(HARMONIC_COMPONENT_COLUMNS, start=2):
        draw_hough_parameter_row(
            axes_row=axes[row_index],
            document_metrics_dataframe=document_metrics_dataframe,
            metric_column=component_column,
            metric_label=component_label,
            force_unit_y_axis=True,
        )

    # Row 6: surviving line count after filtering versus each Hough parameter.
    draw_hough_parameter_row(
        axes_row=axes[5],
        document_metrics_dataframe=document_metrics_dataframe,
        metric_column="used_line_count",
        metric_label="surviving lines after filtering (used_line_count)",
        force_unit_y_axis=False,
    )

    # Surviving-line count should be visually anchored at zero when possible.
    for axis in axes[5]:
        current_bottom, current_top = axis.get_ylim()
        axis.set_ylim(0.0, max(current_top, 1.0))

    # Leave room for the document title.
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])

    # Save and close promptly so large language batches do not accumulate figures in memory.
    output_png_path = parameter_metric_grid_output_path_from_row(output_dir, first_row)
    fig.savefig(output_png_path, dpi=saved_figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_png_path


def plot_parameter_metric_grids_for_pair(
    metrics_dataframe: pd.DataFrame,
    output_dir: Path,
    max_continuous_bins: int,
    saved_figure_dpi: int,
) -> list[Path]:
    """Create one full line-graph grid per loaded document in a pair DataFrame."""
    if metrics_dataframe.empty:
        return []

    # Reuse the one pair DataFrame and slice it per document for plotting.
    document_indices = sorted(metrics_dataframe["document_index"].dropna().astype(int).unique())
    output_paths: list[Path] = []

    for document_position, document_index in enumerate(document_indices, start=1):
        document_metrics_dataframe = metrics_dataframe[metrics_dataframe["document_index"] == document_index]
        first_row = document_metrics_dataframe.iloc[0]
        print(
            f"[{first_row['main_language']} / {first_row['document_type']}] "
            f"graph grid {document_position}/{len(document_indices)}: {document_index} {first_row['fname']}"
        )
        output_paths.append(
            plot_single_document_parameter_metric_grid(
                document_metrics_dataframe=document_metrics_dataframe,
                output_dir=output_dir,
                max_continuous_bins=max_continuous_bins,
                saved_figure_dpi=saved_figure_dpi,
            )
        )

    return output_paths


# ---------------------------------------------------------------------------
# Best-combination matrix/Hough visual panels
# ---------------------------------------------------------------------------


def read_combination_record_at_position(record_path: Path, target_record_number: int) -> dict[str, Any]:
    """Read exactly one combination record from JSONL or pickle-stream storage."""
    if str(record_path).endswith(PICKLE_STREAM_SUFFIX):
        return read_pickle_stream_record_at_position(record_path, int(target_record_number))

    with open_jsonl_text_for_reading(record_path) as jsonl_handle:
        for current_line_number, line in enumerate(jsonl_handle, start=1):
            if current_line_number == int(target_record_number):
                return json.loads(line)
    raise ValueError(f"Could not find record {target_record_number} in {record_path}")


def source_record_pointer_from_metric_row(metric_row: pd.Series) -> tuple[Path, int] | None:
    """Return a stored geometry pointer from a metric row, or None when absent."""
    # Scalar rows loaded from combination_scores.csv.gz usually do not carry a
    # geometry pointer.  Winner-only rows or already-resolved best rows can.
    source_record_value = metric_row.get("source_jsonl_path")
    try:
        if pd.isna(source_record_value):
            return None
    except TypeError:
        pass
    if source_record_value in (None, ""):
        return None

    source_record_number = safe_int_or_none(metric_row.get("source_line_number"))
    if source_record_number is None or source_record_number <= 0:
        return None

    source_record_path = Path(str(source_record_value))
    if not source_record_path.exists():
        return None

    return source_record_path, int(source_record_number)


def combination_record_matches_metric_row(combination_record: dict[str, Any], metric_row: pd.Series) -> bool:
    """Return True when one geometry record matches one selected metric row."""
    # The bundle is document-local, but checking the document index catches stale
    # or accidentally cross-linked pointers before a wrong overlay is drawn.
    document_payload = combination_record.get("document", {}) or {}
    record_document_index = safe_int_or_none(document_payload.get("index"))
    row_document_index = safe_int_or_none(metric_row.get("document_index"))
    if (
        row_document_index is not None
        and record_document_index is not None
        and row_document_index != record_document_index
    ):
        return False

    # Hough parameters define the geometry.  The final visual panel must draw
    # the same threshold/length/gap/seed combination that won in the scalar table.
    hough_parameters = combination_record.get("hough_parameters", {}) or {}
    parameter_pairs = [
        ("hough_threshold", "hough_threshold"),
        ("hough_line_length", "hough_line_length"),
        ("hough_line_gap", "hough_line_gap"),
        ("hough_seed", "hough_seed"),
    ]
    for row_parameter_name, record_parameter_name in parameter_pairs:
        row_parameter_value = safe_int_or_none(metric_row.get(row_parameter_name))
        record_parameter_value = safe_int_or_none(hough_parameters.get(record_parameter_name))
        if row_parameter_value is None:
            continue
        if record_parameter_value is None or row_parameter_value != record_parameter_value:
            return False

    return True


def combination_record_matches_document(combination_record: dict[str, Any], metric_row: pd.Series) -> bool:
    """Return True when a geometry record belongs to the row's document."""
    document_payload = combination_record.get("document", {}) or {}
    record_document_index = safe_int_or_none(document_payload.get("index"))
    row_document_index = safe_int_or_none(metric_row.get("document_index"))
    if row_document_index is not None and record_document_index is not None:
        return row_document_index == record_document_index
    return True


def candidate_geometry_record_paths_for_best_row(best_row: pd.Series) -> list[Path]:
    """Return the smallest useful set of threshold streams for one best row."""
    document_bundle_dir = Path(str(best_row["bundle_dir"]))
    all_record_paths = iter_threshold_record_paths_for_document_bundle(document_bundle_dir)
    if not all_record_paths:
        return []

    # Normal all-scope bundles have one stream per threshold.  The best row's
    # threshold lets us avoid scanning every other threshold just to draw one panel.
    best_threshold = safe_int_or_none(best_row.get("hough_threshold"))
    if best_threshold is None:
        return all_record_paths

    threshold_record_paths = [
        record_path
        for record_path in all_record_paths
        if threshold_value_from_record_path(record_path) == best_threshold
    ]
    return threshold_record_paths or all_record_paths


def single_saved_geometry_record_for_best_row(best_row: pd.Series) -> tuple[dict[str, Any], Path, int] | None:
    """Return the only saved geometry record for a document, or None when ambiguous."""
    document_bundle_dir = Path(str(best_row["bundle_dir"]))
    only_record: tuple[dict[str, Any], Path, int] | None = None

    for record_path in iter_threshold_record_paths_for_document_bundle(document_bundle_dir):
        for combination_record, source_record_path, source_record_number in iter_combination_records_from_path(
            record_path
        ):
            if not combination_record_matches_document(combination_record, best_row):
                continue

            # A winner-only bundle stores exactly one geometry record.  If a bundle has
            # more than one document-matching record, the CSV row must identify the
            # winner by exact Hough parameters because there is no safe fallback.
            if only_record is not None:
                return None

            only_record = (
                combination_record,
                Path(source_record_path),
                int(source_record_number),
            )

    return only_record


def resolve_combination_record_for_best_row(best_row: pd.Series) -> tuple[dict[str, Any], Path, int]:
    """Load the geometry record that belongs to one selected best combination."""
    # First use a stored pointer when one exists.  This is the fastest path for
    # already-resolved rows and for legacy rows loaded directly from bundle records.
    stored_pointer = source_record_pointer_from_metric_row(best_row)
    if stored_pointer is not None:
        source_record_path, source_record_number = stored_pointer
        combination_record = read_combination_record_at_position(source_record_path, source_record_number)
        if combination_record_matches_metric_row(combination_record, best_row):
            return combination_record, source_record_path, source_record_number
        print(
            "[geometry] stored source pointer did not match selected best row; "
            f"falling back to threshold scan: document={best_row.get('document_index')}, "
            f"path={source_record_path}, record={source_record_number}"
        )

    # Scalar-first loading intentionally avoids reading geometry while building
    # the 18-plot grids.  When a stitched best panel is requested, scan only the
    # best threshold stream and stop as soon as the matching record is found.
    searched_record_count = 0
    for record_path in candidate_geometry_record_paths_for_best_row(best_row):
        for combination_record, source_record_path, source_record_number in iter_combination_records_from_path(
            record_path
        ):
            searched_record_count += 1
            if combination_record_matches_metric_row(combination_record, best_row):
                return combination_record, Path(source_record_path), int(source_record_number)

    # Winner-only visual bundles intentionally save just the chosen geometry.  The
    # compact score table can still contain many tied rows, so visualization falls
    # back to that single saved record when exact parameter matching is impossible.
    single_saved_record = single_saved_geometry_record_for_best_row(best_row)
    if single_saved_record is not None:
        combination_record, source_record_path, source_record_number = single_saved_record
        hough_parameters = combination_record.get("hough_parameters", {}) or {}
        print(
            "[geometry] using single saved geometry record from bundle: "
            f"document={best_row.get('document_index')}, "
            f"selected_threshold={best_row.get('hough_threshold')}, "
            f"saved_threshold={hough_parameters.get('hough_threshold')}, "
            f"path={source_record_path}, record={source_record_number}"
        )
        return combination_record, source_record_path, source_record_number

    raise ValueError(
        "Could not find geometry record for best combination "
        f"document_index={best_row.get('document_index')}, fname={best_row.get('fname')}, "
        f"threshold={best_row.get('hough_threshold')}, "
        f"line_length={best_row.get('hough_line_length')}, "
        f"line_gap={best_row.get('hough_line_gap')}, "
        f"seed={best_row.get('hough_seed')}, "
        f"searched_records={searched_record_count}"
    )


def document_info_from_metric_row(metric_row: pd.Series) -> dict[str, Any]:
    """Return the document metadata needed to rebuild a compact metric row."""
    return {
        "main_language": metric_row.get("main_language"),
        "document_type": metric_row.get("document_type"),
        "fname": metric_row.get("fname"),
        "bundle_dir": metric_row.get("bundle_dir"),
        "shard_index": metric_row.get("shard_index"),
    }


def compact_metric_values_from_resolved_record(
    *,
    metric_row: pd.Series,
    combination_record: dict[str, Any],
    source_record_path: Path,
    source_record_number: int,
) -> dict[str, Any]:
    """Return scalar metric values that match the resolved geometry record."""
    return extract_compact_metric_row_from_combination_record(
        combination_record=combination_record,
        source_jsonl_path=Path(source_record_path),
        source_line_number=int(source_record_number),
        document_info=document_info_from_metric_row(metric_row),
    )


def attach_best_geometry_source_pointers(best_rows_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return best rows with geometry source pointers resolved for visual panels."""
    if best_rows_dataframe.empty:
        return best_rows_dataframe

    resolved_best_rows = best_rows_dataframe.copy()
    for row_index, best_row in resolved_best_rows.iterrows():
        combination_record, source_record_path, source_record_number = resolve_combination_record_for_best_row(best_row)
        resolved_values = compact_metric_values_from_resolved_record(
            metric_row=best_row,
            combination_record=combination_record,
            source_record_path=source_record_path,
            source_record_number=source_record_number,
        )
        for column_name, resolved_value in resolved_values.items():
            resolved_best_rows.at[row_index, column_name] = resolved_value
    return resolved_best_rows


def load_score_matrix_from_pickle_if_it_matches_document(
    scores_pickle_path: Path,
    expected_document_name: str,
) -> np.ndarray | None:
    """Load a fallback pickle matrix only when its fname matches this document."""
    if not Path(scores_pickle_path).exists():
        return None

    # The known pkl format contains one payload with fname, scores, ref, and pred.
    with Path(scores_pickle_path).open("rb") as pickle_handle:
        score_payload = pickle.load(pickle_handle)

    # Avoid using a matrix from the wrong document.
    pickle_document_name = Path(str(score_payload.get("fname", ""))).name if isinstance(score_payload, dict) else ""
    if pickle_document_name != Path(str(expected_document_name)).name:
        return None

    return np.asarray(score_payload.get("scores"), dtype=float)


def load_document_score_matrix(
    document_bundle_dir: Path,
    matrix_filename: str,
    fallback_scores_pickle_path: Path,
    expected_document_name: str,
) -> np.ndarray | None:
    """Load a document score matrix from the bundle-local .npy file or matching pkl fallback."""
    # Bundle-local .npy files are fastest and guaranteed to match the bundle records.
    bundle_matrix_path = Path(document_bundle_dir) / matrix_filename
    if bundle_matrix_path.exists():
        return np.load(bundle_matrix_path)

    # The fallback is safe only when the pickle declares the same fname.
    return load_score_matrix_from_pickle_if_it_matches_document(fallback_scores_pickle_path, expected_document_name)


def reference_text_from_document_record(document_record: dict[str, Any]) -> str:
    """Return the normalized reference text stored only for plotting fallbacks."""
    return str(document_record.get(REFERENCE_TEXT_FOR_PLOTTING_KEY, ""))


def compute_ref_to_ref_score_matrix_for_plotting(
    *,
    document_record: dict[str, Any],
    window_size: int,
    window_stride: int,
) -> np.ndarray | None:
    """Compute a reference-self score matrix when no saved diagnostic matrix exists."""
    reference_text = reference_text_from_document_record(document_record)
    if not reference_text.strip():
        return None

    try:
        return np.asarray(
            compute_score_matrix(
                reference_text,
                reference_text,
                window_size=int(window_size),
                window_stride=int(window_stride),
            ),
            dtype=float,
        )
    except Exception as exc:
        document_name = document_record.get("fname", "unknown")
        print(f"[plot fallback] failed to compute ref_to_ref matrix for {document_name}: {exc!r}")
        return None


def load_document_binary_mask(
    document_bundle_dir: Path,
    mask_filename: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Load one bundle-local binary mask and keep only masks with a usable shape."""
    mask_path = Path(document_bundle_dir) / mask_filename
    if not mask_path.exists():
        return None

    mask_array = np.asarray(np.load(mask_path), dtype=bool)
    if mask_array.ndim != 2 or mask_array.shape[0] == 0 or mask_array.shape[1] == 0:
        return None

    # A mask with the wrong shape cannot be interpreted against the score matrix.
    if expected_shape is not None and tuple(mask_array.shape) != tuple(expected_shape):
        return None

    return mask_array


def draw_binary_mask_panel(axis, binary_mask: np.ndarray | None, title: str, active_label: str) -> None:
    """Draw a binary diagnostic mask where black pixels are active cells."""
    if binary_mask is None:
        axis.text(0.5, 0.5, "Mask missing", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel("Prediction window index")
        axis.set_ylabel("Reference window index")
        return

    mask_array = np.asarray(binary_mask, dtype=bool)
    if mask_array.size == 0 or mask_array.ndim != 2 or mask_array.shape[0] == 0 or mask_array.shape[1] == 0:
        axis.text(
            0.5,
            0.5,
            f"Empty mask\nshape={getattr(mask_array, 'shape', None)}",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title(title)
        axis.set_xlabel("Prediction window index")
        axis.set_ylabel("Reference window index")
        return

    axis.imshow(mask_array.astype(np.uint8), origin="upper", aspect="auto", cmap="gray_r", vmin=0, vmax=1)
    active_cell_count = int(np.count_nonzero(mask_array))
    total_cell_count = int(mask_array.size)
    axis.set_title(f"{title} ({active_label}: {active_cell_count}/{total_cell_count})")
    axis.set_xlabel("Prediction window index")
    axis.set_ylabel("Reference window index")
    axis.set_xlim(-0.5, mask_array.shape[1] - 0.5)
    axis.set_ylim(mask_array.shape[0] - 0.5, -0.5)


def draw_score_matrix_heatmap(axis, score_matrix: np.ndarray | None, title: str):
    """Draw a score-matrix heatmap without any Hough overlays."""
    # Missing matrices are shown explicitly instead of raising.
    if score_matrix is None:
        axis.text(0.5, 0.5, "Matrix missing", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel("Other/prediction window index")
        axis.set_ylabel("Reference window index")
        return None

    # Normalize to a NumPy array before checking shape.
    score_matrix = np.asarray(score_matrix)

    # No-prediction documents should already be skipped, but this keeps rendering robust.
    if score_matrix.size == 0 or score_matrix.ndim != 2 or score_matrix.shape[0] == 0 or score_matrix.shape[1] == 0:
        axis.text(
            0.5,
            0.5,
            f"Empty matrix\nshape={getattr(score_matrix, 'shape', None)}",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title(title)
        axis.set_xlabel("Other/prediction window index")
        axis.set_ylabel("Reference window index")
        return None

    # Score matrices are chrF-style percentages, so every colored panel uses
    # the same 0..100 scale instead of auto-scaling each document separately.
    image = axis.imshow(score_matrix, origin="upper", aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    axis.set_title(title)
    axis.set_xlabel("Other/prediction window index")
    axis.set_ylabel("Reference window index")
    axis.set_xlim(-0.5, score_matrix.shape[1] - 0.5)
    axis.set_ylim(score_matrix.shape[0] - 0.5, -0.5)
    return image


def endpoint_tuple_from_raw_hough_segment(raw_segment: Any) -> tuple[float, float, float, float] | None:
    """Convert raw Hough [[x0, y0], [x1, y1]] segments into numeric endpoints."""
    try:
        (x0, y0), (x1, y1) = raw_segment
        return float(x0), float(y0), float(x1), float(y1)
    except (TypeError, ValueError):
        return None


def endpoint_tuple_from_filtered_line_record(line_record: Any) -> tuple[float, float, float, float] | None:
    """Convert a filtered line dictionary into numeric endpoints."""
    if not isinstance(line_record, dict):
        return None
    required_keys = ("x0", "y0", "x1", "y1")
    if any(line_record.get(key) is None for key in required_keys):
        return None
    return (
        float(line_record["x0"]),
        float(line_record["y0"]),
        float(line_record["x1"]),
        float(line_record["y1"]),
    )


def compact_raw_line_id_sequence(raw_line_ids: Sequence[int]) -> str:
    """Return a compact label fragment for one or more raw Hough line IDs."""
    # Deduplicate and sort so labels are stable even if a merge record repeats a source ID.
    normalized_ids = sorted({int(raw_line_id) for raw_line_id in raw_line_ids})
    if not normalized_ids:
        return "unknown"

    # Consecutive runs with more than two IDs are easier to read as 5..7 than 5,6,7.
    if len(normalized_ids) > 2 and normalized_ids == list(range(normalized_ids[0], normalized_ids[-1] + 1)):
        return f"{normalized_ids[0]}..{normalized_ids[-1]}"

    return ",".join(str(raw_line_id) for raw_line_id in normalized_ids)


def raw_source_line_ids_from_filtered_line_record(line_record: Any) -> list[int]:
    """Return source raw Hough IDs stored on one surviving filtered line record."""
    if not isinstance(line_record, dict):
        return []

    source_raw_line_ids = line_record.get("source_raw_line_ids")
    if source_raw_line_ids is None:
        source_raw_line_ids = line_record.get("raw_line_id")
    if source_raw_line_ids is None:
        return []
    if isinstance(source_raw_line_ids, (list, tuple, set)):
        return [int(raw_line_id) for raw_line_id in source_raw_line_ids]
    return [int(source_raw_line_ids)]


def format_raw_hough_line_label(raw_hough_line_index: int) -> str:
    """Return the label drawn on the raw Hough panel for one raw line."""
    return str(int(raw_hough_line_index))


def format_surviving_line_label(
    final_surviving_line_index: int,
    source_raw_line_ids: Sequence[int],
) -> str:
    """Return the label drawn on the final surviving filtered-line panel."""
    source_label = compact_raw_line_id_sequence(source_raw_line_ids)
    return f"F{int(final_surviving_line_index)} <- {source_label}"


def segment_unit_vectors(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return unit direction and perpendicular vectors for one plotted segment."""
    delta_x = float(x1) - float(x0)
    delta_y = float(y1) - float(y0)
    segment_length = math.hypot(delta_x, delta_y)
    if segment_length <= 0.0 or not math.isfinite(segment_length):
        return (1.0, 0.0), (0.0, 1.0)
    direction_x = delta_x / segment_length
    direction_y = delta_y / segment_length
    return (direction_x, direction_y), (-direction_y, direction_x)


def oriented_segment_box_points(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    half_width_cells: float,
    end_padding_cells: float,
) -> list[tuple[float, float]]:
    """Return a hollow rotated rectangle that surrounds, but does not cover, a segment."""
    (direction_x, direction_y), (normal_x, normal_y) = segment_unit_vectors(x0=x0, y0=y0, x1=x1, y1=y1)
    half_width = max(0.5, float(half_width_cells))
    end_padding = max(0.0, float(end_padding_cells))
    start_x = float(x0) - direction_x * end_padding
    start_y = float(y0) - direction_y * end_padding
    end_x = float(x1) + direction_x * end_padding
    end_y = float(y1) + direction_y * end_padding
    return [
        (start_x + normal_x * half_width, start_y + normal_y * half_width),
        (end_x + normal_x * half_width, end_y + normal_y * half_width),
        (end_x - normal_x * half_width, end_y - normal_y * half_width),
        (start_x - normal_x * half_width, start_y - normal_y * half_width),
    ]


def draw_segment_box(
    axis,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    label: str | None,
    linewidth: float,
    alpha: float,
    linestyle: str,
    half_width_cells: float,
) -> None:
    """Draw a hollow box around a detected segment so the score ridge remains visible."""
    box_points = oriented_segment_box_points(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        half_width_cells=float(half_width_cells),
        end_padding_cells=0.75,
    )
    axis.add_patch(
        Polygon(
            box_points,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            label=label,
            joinstyle="miter",
            zorder=6,
        )
    )


def draw_line_label_near_segment(
    axis,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label_text: str,
    offset_cells: float,
) -> None:
    """Draw one readable label next to, not on top of, the detected segment."""
    _, (normal_x, normal_y) = segment_unit_vectors(x0=x0, y0=y0, x1=x1, y1=y1)
    midpoint_x = (float(x0) + float(x1)) / 2.0 + normal_x * float(offset_cells)
    midpoint_y = (float(y0) + float(y1)) / 2.0 + normal_y * float(offset_cells)
    axis.text(
        midpoint_x,
        midpoint_y,
        str(label_text),
        color="#FFE066",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "black",
            "edgecolor": "#FFE066",
            "linewidth": 0.45,
            "alpha": 0.72,
        },
        zorder=8,
    )


def draw_endpoint_segments(
    axis,
    endpoint_segments: Iterable[tuple[float, float, float, float]],
    color: str,
    label: str,
    linewidth: float,
    alpha: float,
    linestyle: str,
    *,
    segment_labels: Sequence[str] | None = None,
    show_line_labels: bool = True,
    half_width_cells: float = 2.0,
) -> int:
    """Draw hollow boxes around endpoint segments and return how many were drawn."""
    drawn_segment_count = 0
    endpoint_segment_list = list(endpoint_segments)
    label_list = list(segment_labels or [])
    for segment_index, (x0, y0, x1, y1) in enumerate(endpoint_segment_list):
        draw_segment_box(
            axis,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            color=color,
            label=label if drawn_segment_count == 0 else None,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            half_width_cells=float(half_width_cells),
        )
        if show_line_labels and segment_index < len(label_list):
            draw_line_label_near_segment(
                axis,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                label_text=label_list[segment_index],
                offset_cells=float(half_width_cells) + 2.0,
            )
        drawn_segment_count += 1
    return drawn_segment_count


def draw_raw_hough_overlay(
    axis,
    combination_record: dict[str, Any],
    *,
    show_line_labels: bool,
) -> int:
    """Draw raw ref_to_pred Hough lines on an existing score-matrix axis."""
    # Raw Hough segments live in ref_to_pred only for this diagnostic.
    raw_lines = (
        (combination_record.get("ref_to_pred", {}) or {})
        .get("hough_detection", {})
        .get("raw_lines", [])
        or []
    )
    endpoint_segments: list[tuple[float, float, float, float]] = []
    segment_labels: list[str] = []
    for raw_hough_line_index, raw_line in enumerate(raw_lines):
        endpoint_tuple = endpoint_tuple_from_raw_hough_segment(raw_line)
        if endpoint_tuple is None:
            continue
        endpoint_segments.append(endpoint_tuple)
        segment_labels.append(format_raw_hough_line_label(raw_hough_line_index))
    return draw_endpoint_segments(
        axis=axis,
        endpoint_segments=endpoint_segments,
        color=RAW_HOUGH_LINE_COLOR,
        label="Raw Hough segment",
        linewidth=RAW_HOUGH_LINE_WIDTH_POINTS,
        alpha=0.85,
        linestyle="-",
        segment_labels=segment_labels,
        show_line_labels=show_line_labels,
        half_width_cells=RAW_HOUGH_LINE_BOX_HALF_WIDTH_CELLS,
    )


def draw_surviving_filtered_hough_overlay(
    axis,
    combination_record: dict[str, Any],
    *,
    saved_figure_dpi: int,
    show_line_labels: bool,
) -> int:
    """Draw only final surviving ref_to_pred lines after filtering."""
    # Filtering payload contains both candidates and final used lines; only lines_used survives.
    filtering_payload = (combination_record.get("ref_to_pred", {}) or {}).get("filtering", {}) or {}

    # Used lines are the final filtered lines used by the alignment metric.
    surviving_endpoint_segments: list[tuple[float, float, float, float]] = []
    surviving_segment_labels: list[str] = []
    for final_surviving_line_index, line_record in enumerate(filtering_payload.get("lines_used", []) or []):
        endpoint_tuple = endpoint_tuple_from_filtered_line_record(line_record)
        if endpoint_tuple is None:
            continue
        source_raw_line_ids = raw_source_line_ids_from_filtered_line_record(line_record)
        surviving_endpoint_segments.append(endpoint_tuple)
        surviving_segment_labels.append(
            format_surviving_line_label(
                final_surviving_line_index=final_surviving_line_index,
                source_raw_line_ids=source_raw_line_ids,
            )
        )

    # Matplotlib line widths are in points.  Converting one output pixel into points
    # makes the requested "1 pixel thicker" behavior stable across saved DPI values.
    one_output_pixel_in_points = 72.0 / float(saved_figure_dpi)
    surviving_line_width_points = SURVIVING_FILTERED_LINE_WIDTH_POINTS + one_output_pixel_in_points

    # Plot only the final surviving lines, with no candidate-line clutter.
    return draw_endpoint_segments(
        axis=axis,
        endpoint_segments=surviving_endpoint_segments,
        color=SURVIVING_FILTERED_LINE_COLOR,
        label="Surviving line after filtering",
        linewidth=surviving_line_width_points,
        alpha=0.95,
        linestyle="-",
        segment_labels=surviving_segment_labels,
        show_line_labels=show_line_labels,
        half_width_cells=SURVIVING_FILTERED_LINE_BOX_HALF_WIDTH_CELLS,
    )


def best_selection_metric_column(pair_metrics_dataframe: pd.DataFrame) -> str:
    """Return the metric column that should choose each document's visual winner."""
    if "selection_objective" not in pair_metrics_dataframe.columns:
        return "tuning_score"

    objective_values = {
        str(value)
        for value in pair_metrics_dataframe["selection_objective"].dropna().unique().tolist()
    }
    if "alignment_evidence" not in objective_values:
        return "tuning_score"
    if "alignment_selection_score" not in pair_metrics_dataframe.columns:
        return "tuning_score"

    numeric_alignment_scores = pd.to_numeric(
        pair_metrics_dataframe["alignment_selection_score"],
        errors="coerce",
    )
    return "alignment_selection_score" if numeric_alignment_scores.notna().any() else "tuning_score"


def select_best_combination_rows_per_document(pair_metrics_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the best row for every loaded document using the run's objective."""
    # Empty language/document-type pairs have no combinations to rank.
    if pair_metrics_dataframe.empty:
        return pd.DataFrame(columns=pair_metrics_dataframe.columns)

    # Defensive guard for malformed or older tables that do not contain the score column.
    if "tuning_score" not in pair_metrics_dataframe.columns:
        return pd.DataFrame(columns=pair_metrics_dataframe.columns)

    # Only finite tuning scores can be ranked, even when an alternate selector is active.
    valid_score_dataframe = pair_metrics_dataframe.replace([np.inf, -np.inf], np.nan).dropna(subset=["tuning_score"])
    if valid_score_dataframe.empty:
        return pd.DataFrame(columns=pair_metrics_dataframe.columns)

    selection_metric_column = best_selection_metric_column(valid_score_dataframe)
    sort_columns = [
        "main_language",
        "document_type",
        "document_index",
        selection_metric_column,
    ]
    sort_ascending = [True, True, True, False]
    if selection_metric_column != "tuning_score":
        sort_columns.append("tuning_score")
        sort_ascending.append(False)
    sort_columns.extend(["hough_threshold", "hough_line_length", "hough_line_gap"])
    sort_ascending.extend([True, True, True])

    # Smaller threshold/length/gap are stable tie-breakers after the active objective.
    sorted_dataframe = valid_score_dataframe.sort_values(
        sort_columns,
        ascending=sort_ascending,
        kind="stable",
    )
    return sorted_dataframe.groupby("document_index", as_index=False, sort=False).head(1).reset_index(drop=True)


def build_best_combination_metrics_text_lines_for_document(best_row: pd.Series) -> list[str]:
    """Return the human-readable best-combination metrics block for one document."""
    # This function is intentionally shared by the stitched TXT and the PNG metrics band so
    # both final outputs report the same best Hough parameters and metric fields.
    lines = [
        f"main_language: {best_row['main_language']}",
        f"document_type: {best_row['document_type']}",
        f"document_index: {int(best_row['document_index'])}",
        f"fname: {best_row['fname']}",
        "",
        "Best Hough Parameters:",
        f"  hough_threshold: {format_metric_for_text(best_row.get('hough_threshold'))}",
        f"  hough_line_length: {format_metric_for_text(best_row.get('hough_line_length'))}",
        f"  hough_line_gap: {format_metric_for_text(best_row.get('hough_line_gap'))}",
        f"  hough_seed: {format_metric_for_text(best_row.get('hough_seed'))}",
        f"  effective_hough_seed: {format_metric_for_text(best_row.get('effective_hough_seed'))}",
        "",
        "Metrics:",
    ]
    for metric_name in BEST_COMBINATION_METRIC_COLUMNS:
        lines.append(f"  {metric_name}: {format_metric_for_text(best_row.get(metric_name))}")
    return lines


def write_stitched_best_combination_metrics_text_for_pair(
    *,
    best_rows_dataframe: pd.DataFrame,
    output_txt_path: Path,
) -> Path:
    """Write one stitched best-combination metrics text file for a language/type pair."""
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # Sorting here makes the text file independent of any upstream DataFrame index history.
    sorted_best_rows_dataframe = best_rows_dataframe.sort_values(["document_index"], kind="stable")

    lines = [
        "Stitched best combination metrics",
        f"source_directory: {output_txt_path.parent}",
        f"document_metrics_count: {len(sorted_best_rows_dataframe)}",
        "",
    ]
    for _, best_row in sorted_best_rows_dataframe.iterrows():
        # The section header mirrors the document folder naming, even though best-panel
        # artifacts are no longer written inside each per-document folder.
        lines.extend(
            [
                "=" * 80,
                f"document_folder: {document_output_stem_from_row(best_row)}",
                "=" * 80,
            ]
        )
        lines.extend(build_best_combination_metrics_text_lines_for_document(best_row))
        lines.append("")

    output_txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_txt_path


def first_available_row_value(row: pd.Series, candidate_column_names: Sequence[str]) -> Any:
    """Return the first present, non-missing value from a row."""
    for column_name in candidate_column_names:
        if column_name not in row.index:
            continue
        value = row.get(column_name)
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        return value
    return None


def best_row_to_public_json_record(best_row: pd.Series) -> dict[str, Any]:
    """Convert one best-combination row into the public metrics JSON schema."""
    raw_line_count = safe_int_or_none(best_row.get("raw_line_count"))
    used_line_count = safe_int_or_none(best_row.get("used_line_count"))
    public_record = {
        "best_threshold": safe_int_or_none(best_row.get("hough_threshold")),
        "best_line_length": safe_int_or_none(best_row.get("hough_line_length")),
        "best_line_gap": safe_int_or_none(best_row.get("hough_line_gap")),
        "best_seed": safe_int_or_none(best_row.get("hough_seed")),
        "tuning_score": safe_float_or_nan(best_row.get("tuning_score")),
        "normalized_levenshtein_similarity": safe_float_or_nan(best_row.get("whole_document_nls")),
        "average_weighted_normalized_levenshtein_similarity": safe_float_or_nan(
            best_row.get("weighted_along_lines_nls")
        ),
        "normalised_document_levenshtein_similarity_after_alignment": safe_float_or_nan(
            first_available_row_value(
                best_row,
                [
                    "normalised_document_levenshtein_similarity_after_alignment",
                    "after_normalized_levenshtein_similarity",
                    "after_normalised_levenshtein_similarity",
                ],
            )
        ),
        "correct_reference_coverage": safe_float_or_nan(best_row.get("correct_ref_coverage")),
        "missing_reference_coverage": safe_float_or_nan(best_row.get("missing_ref_coverage")),
        "repetition_on_reference": safe_float_or_nan(best_row.get("repetition_on_ref")),
        "hallucination": safe_float_or_nan(best_row.get("hallucination")),
        "raw_line_count": raw_line_count,
        "used_line_count": used_line_count,
        "raw_hough_lines": raw_line_count,
        "surviving_lines": used_line_count,
    }
    return {field_name: json_ready_value(public_record.get(field_name)) for field_name in PUBLIC_BEST_DOCUMENT_JSON_FIELDS}


def build_average_metric_record(document_metric_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return mean public metrics across a list of document metric records."""
    average_record: dict[str, Any] = {}
    for metric_name in PUBLIC_AVERAGE_METRIC_FIELDS:
        metric_values = [
            float(record[metric_name])
            for record in document_metric_records
            if record.get(metric_name) is not None and not math.isnan(float(record[metric_name]))
        ]
        average_record[metric_name] = None if not metric_values else float(sum(metric_values) / len(metric_values))
    return average_record


def write_stitched_best_combination_metrics_json(
    *,
    best_rows_dataframe: pd.DataFrame,
    output_json_path: Path,
) -> Path:
    """Write one human-readable JSON report for every stitched best panel."""
    public_document_records: list[dict[str, Any]] = []

    if not best_rows_dataframe.empty:
        sorted_best_rows_dataframe = best_rows_dataframe.sort_values(
            ["document_type", "main_language", "document_index"],
            kind="stable",
        )
        for _, best_row in sorted_best_rows_dataframe.iterrows():
            public_document_records.append(
                {
                    "main_language": str(best_row.get("main_language")),
                    "document_type": str(best_row.get("document_type")),
                    "document_index": safe_int_or_none(best_row.get("document_index")),
                    "fname": str(best_row.get("fname")),
                    **best_row_to_public_json_record(best_row),
                }
            )

    return write_stitched_best_combination_metrics_json_from_public_records(
        public_document_records=public_document_records,
        output_json_path=output_json_path,
    )


def write_stitched_best_combination_metrics_json_from_public_records(
    *,
    public_document_records: list[dict[str, Any]],
    output_json_path: Path,
) -> Path:
    """Write one human-readable JSON report from already-normalized records."""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    documents_by_language_and_type: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    summary_by_document_type: dict[str, dict[str, Any]] = {}

    for public_record in sorted(
        public_document_records,
        key=lambda record: (
            str(record.get("document_type")),
            str(record.get("main_language")),
            int(record.get("document_index") or 0),
            str(record.get("fname")),
        ),
    ):
        language_name = str(public_record.get("main_language"))
        document_type = str(public_record.get("document_type"))
        document_name = str(public_record.get("fname"))
        document_metric_record = {
            field_name: json_ready_value(public_record.get(field_name))
            for field_name in PUBLIC_BEST_DOCUMENT_JSON_FIELDS
        }
        documents_by_language_and_type.setdefault(language_name, {}).setdefault(document_type, {})[
            document_name
        ] = document_metric_record

    for document_type in sorted(
        {
            document_type
            for language_payload in documents_by_language_and_type.values()
            for document_type in language_payload.keys()
        }
    ):
        document_type_records: list[dict[str, Any]] = []
        language_summary: dict[str, Any] = {}
        for language_name in sorted(documents_by_language_and_type.keys()):
            document_records_for_language_type = list(
                documents_by_language_and_type.get(language_name, {}).get(document_type, {}).values()
            )
            if not document_records_for_language_type:
                continue
            document_type_records.extend(document_records_for_language_type)
            language_summary[language_name] = {
                "document_count": int(len(document_records_for_language_type)),
                "average_metrics": build_average_metric_record(document_records_for_language_type),
            }

        # Keep the summary focused on per-language averages inside each real
        # document_type.  The project owner intentionally does not want an
        # overall document_type average here because language-level comparisons
        # are the useful diagnostic unit for this visualisation report.
        summary_by_document_type[document_type] = {
            "languages": language_summary,
        }

    json_payload = {
        "schema_version": "tuner_stitched_best_combination_metrics_v1",
        "summary": {
            "by_document_type": summary_by_document_type,
        },
        "documents": documents_by_language_and_type,
    }
    output_json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_ready_value),
        encoding="utf-8",
    )
    return output_json_path


def format_labeled_values_as_two_column_text(
    labeled_values: list[tuple[str, str]],
    *,
    label_width: int = 30,
    left_column_width: int = 52,
) -> str:
    """Format labeled values as a compact two-column monospace block for PNG panels."""
    # A two-column block keeps every metric visible without turning the panel into a tall page.
    midpoint_index = math.ceil(len(labeled_values) / 2)
    left_labeled_values = labeled_values[:midpoint_index]
    right_labeled_values = labeled_values[midpoint_index:]
    formatted_lines: list[str] = []

    for row_index in range(midpoint_index):
        left_label, left_value = left_labeled_values[row_index]
        left_text = f"{left_label:<{label_width}} {left_value}"

        # The right column may be absent on the final row when the item count is odd.
        if row_index < len(right_labeled_values):
            right_label, right_value = right_labeled_values[row_index]
            right_text = f"{right_label:<{label_width}} {right_value}"
        else:
            right_text = ""

        formatted_lines.append(f"{left_text:<{left_column_width}}{right_text}")

    return "\n".join(formatted_lines)


def build_best_combination_metrics_text_for_panel(best_row: pd.Series) -> str:
    """Return the compact metrics text drawn into one temporary best-combination panel."""
    document_index = int(best_row["document_index"])
    document_name = str(best_row["fname"])
    identity_line = f"document {document_index:06d} | {document_name}"

    hough_parameter_values = [
        ("hough_threshold", format_metric_for_text(best_row.get("hough_threshold"))),
        ("hough_line_length", format_metric_for_text(best_row.get("hough_line_length"))),
        ("hough_line_gap", format_metric_for_text(best_row.get("hough_line_gap"))),
        ("hough_seed", format_metric_for_text(best_row.get("hough_seed"))),
        ("effective_hough_seed", format_metric_for_text(best_row.get("effective_hough_seed"))),
    ]
    metric_values = [
        (metric_name, format_metric_for_text(best_row.get(metric_name)))
        for metric_name in BEST_COMBINATION_METRIC_COLUMNS
    ]

    return "\n".join(
        [
            identity_line,
            "",
            "Best Hough parameters",
            format_labeled_values_as_two_column_text(hough_parameter_values),
            "",
            "Metrics",
            format_labeled_values_as_two_column_text(metric_values),
        ]
    )


def document_output_stem_from_record(document_record: dict[str, Any]) -> str:
    """Return the document output stem for a plain dictionary record."""
    document_index = int(document_record["document_index"])
    document_name = str(document_record["fname"])
    return f"document_{document_index:06d}_{safe_path_component(document_name)}"


def should_render_skipped_visual_panel(skipped_document: dict[str, Any]) -> bool:
    """Return True for real skipped documents that deserve a diagnostic panel."""
    if str(skipped_document.get("skip_reason")) == "bundle_folder_not_available_yet":
        return False
    return skipped_document_has_prediction_text_or_windows(skipped_document)


def build_skipped_document_metrics_text_for_panel(skipped_document: dict[str, Any]) -> str:
    """Return the skipped-document explanation drawn into one diagnostic panel."""
    document_index = int(skipped_document["document_index"])
    document_name = str(skipped_document["fname"])
    identity_line = f"document {document_index:06d} | {document_name}"
    skip_values = [
        ("skip_reason", format_metric_for_text(skipped_document.get("skip_reason"))),
        ("skip_stage", format_metric_for_text(skipped_document.get("skip_stage"))),
        ("preprocessing_rejection_reason", format_metric_for_text(skipped_document.get("preprocessing_rejection_reason"))),
        ("ref_to_pred_matrix_rows", format_metric_for_text(skipped_document.get("ref_to_pred_matrix_rows"))),
        ("ref_to_pred_matrix_cols", format_metric_for_text(skipped_document.get("ref_to_pred_matrix_cols"))),
        ("ref_to_pred_matrix_max", format_metric_for_text(skipped_document.get("ref_to_pred_matrix_max"))),
        ("preprocessing_score_floor", format_metric_for_text(skipped_document.get("preprocessing_score_floor"))),
        ("preprocessing_active_cells", format_metric_for_text(skipped_document.get("preprocessing_active_cells"))),
        ("preprocessing_active_fraction", format_metric_for_text(skipped_document.get("preprocessing_active_fraction"))),
        ("diagnostic_bundle_dir", format_metric_for_text(skipped_document.get("diagnostic_bundle_dir"))),
    ]
    message = str(skipped_document.get("message", ""))
    return "\n".join(
        [
            identity_line,
            "",
            "Skipped before Hough combination search",
            format_labeled_values_as_two_column_text(skip_values, label_width=34, left_column_width=72),
            "",
            "Message",
            message if message else "No message recorded.",
        ]
    )


def render_skipped_document_visual_panel(
    *,
    skipped_document: dict[str, Any],
    temporary_panel_output_dir: Path,
    ref_to_pred_scores_pkl: Path,
    ref_to_ref_scores_pkl: Path,
    saved_figure_dpi: int,
    window_size: int,
    window_stride: int,
    return_image: bool = False,
) -> Path | Any:
    """Render one skipped-document diagnostic panel for the stitched sheet."""
    language_name = str(skipped_document.get("main_language"))
    document_type = str(skipped_document.get("document_type"))
    document_index = int(skipped_document["document_index"])
    document_name = str(skipped_document["fname"])
    bundle_dir_value = skipped_document.get("diagnostic_bundle_dir") or skipped_document.get("bundle_dir")
    document_bundle_dir = Path(str(bundle_dir_value)) if bundle_dir_value not in (None, "") else Path("__missing_bundle__")

    ref_to_pred_score_matrix = load_document_score_matrix(
        document_bundle_dir=document_bundle_dir,
        matrix_filename="ref_to_pred_score_matrix.npy",
        fallback_scores_pickle_path=ref_to_pred_scores_pkl,
        expected_document_name=document_name,
    )
    ref_to_ref_score_matrix = load_document_score_matrix(
        document_bundle_dir=document_bundle_dir,
        matrix_filename="ref_to_ref_score_matrix.npy",
        fallback_scores_pickle_path=ref_to_ref_scores_pkl,
        expected_document_name=document_name,
    )
    if ref_to_ref_score_matrix is None:
        ref_to_ref_score_matrix = compute_ref_to_ref_score_matrix_for_plotting(
            document_record=skipped_document,
            window_size=int(window_size),
            window_stride=int(window_stride),
        )

    ref_to_pred_matrix_shape = None
    if ref_to_pred_score_matrix is not None and np.asarray(ref_to_pred_score_matrix).ndim == 2:
        ref_to_pred_matrix_shape = tuple(int(value) for value in np.asarray(ref_to_pred_score_matrix).shape)
    ref_to_pred_region_of_interest_mask = load_document_binary_mask(
        document_bundle_dir=document_bundle_dir,
        mask_filename="ref_to_pred_region_of_interest_mask.npy",
        expected_shape=ref_to_pred_matrix_shape,
    )
    ref_to_pred_hough_input_mask = load_document_binary_mask(
        document_bundle_dir=document_bundle_dir,
        mask_filename="ref_to_pred_hough_input_mask.npy",
        expected_shape=ref_to_pred_matrix_shape,
    )

    if not return_image:
        temporary_panel_output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20, 24.8), constrained_layout=False)
    grid_specification = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 1.0, 1.0, 0.42],
        hspace=0.30,
        wspace=0.18,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid_specification[0, 0]), fig.add_subplot(grid_specification[0, 1])],
            [fig.add_subplot(grid_specification[1, 0]), fig.add_subplot(grid_specification[1, 1])],
            [fig.add_subplot(grid_specification[2, 0]), fig.add_subplot(grid_specification[2, 1])],
        ]
    )
    metrics_axis = fig.add_subplot(grid_specification[3, :])
    fig.suptitle(
        f"{language_name} / {document_type} | document {document_index} | {document_name}\n"
        "Skipped document diagnostic: score matrices and preprocessing masks",
        fontsize=15,
        y=0.995,
    )

    image_0 = draw_score_matrix_heatmap(
        axes[0, 0],
        ref_to_pred_score_matrix,
        "ref_to_pred score matrix",
    )
    image_1 = draw_score_matrix_heatmap(
        axes[0, 1],
        ref_to_ref_score_matrix,
        "ref_to_ref score matrix",
    )
    image_2 = draw_score_matrix_heatmap(
        axes[1, 0],
        ref_to_pred_score_matrix,
        "Raw Hough lines were not computed",
    )
    image_3 = draw_score_matrix_heatmap(
        axes[1, 1],
        ref_to_pred_score_matrix,
        "Surviving lines were not computed",
    )
    draw_binary_mask_panel(
        axes[2, 0],
        ref_to_pred_region_of_interest_mask,
        "ref_to_pred Region of Interest mask",
        "inside Region of Interest",
    )
    draw_binary_mask_panel(
        axes[2, 1],
        ref_to_pred_hough_input_mask,
        "ref_to_pred final Hough input",
        "voters",
    )

    for axis, image in zip(axes[:2, :].ravel(), [image_0, image_1, image_2, image_3]):
        if image is not None:
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Score")

    metrics_axis.set_axis_off()
    metrics_axis.text(
        0.01,
        0.98,
        build_skipped_document_metrics_text_for_panel(skipped_document),
        transform=metrics_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        linespacing=1.25,
    )

    output_png_path = (
        temporary_panel_output_dir
        / f"{TEMPORARY_BEST_COMBINATION_PANEL_FILENAME_PREFIX}__skipped__{document_output_stem_from_record(skipped_document)}.png"
    )
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.035, top=0.93, hspace=0.34, wspace=0.20)
    if return_image:
        return render_figure_to_rgba_image(fig, saved_figure_dpi=saved_figure_dpi)
    fig.savefig(output_png_path, dpi=saved_figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_png_path


def render_best_combination_visual_panel(
    *,
    best_row: pd.Series,
    temporary_panel_output_dir: Path,
    ref_to_pred_scores_pkl: Path,
    ref_to_ref_scores_pkl: Path,
    saved_figure_dpi: int,
    show_line_labels: bool,
    return_image: bool = False,
) -> Path | Any:
    """Render one temporary best-combination panel that will later be stitched."""
    # Extract document and source-record identity from the compact DataFrame row.
    language_name = str(best_row["main_language"])
    document_type = str(best_row["document_type"])
    document_index = int(best_row["document_index"])
    document_name = str(best_row["fname"])
    document_bundle_dir = Path(str(best_row["bundle_dir"]))

    # Reload only the selected best combination record.  Scalar-first visual
    # loading keeps the 18-plot grids away from geometry streams, so this helper
    # resolves the geometry lazily only when a matrix/Hough panel is requested.
    combination_record, _, _ = resolve_combination_record_for_best_row(best_row)

    # Load base matrices for no-Hough plots and ref_to_pred overlays.
    ref_to_pred_score_matrix = load_document_score_matrix(
        document_bundle_dir=document_bundle_dir,
        matrix_filename="ref_to_pred_score_matrix.npy",
        fallback_scores_pickle_path=ref_to_pred_scores_pkl,
        expected_document_name=document_name,
    )
    ref_to_ref_score_matrix = load_document_score_matrix(
        document_bundle_dir=document_bundle_dir,
        matrix_filename="ref_to_ref_score_matrix.npy",
        fallback_scores_pickle_path=ref_to_ref_scores_pkl,
        expected_document_name=document_name,
    )

    ref_to_pred_matrix_shape = None
    if ref_to_pred_score_matrix is not None and np.asarray(ref_to_pred_score_matrix).ndim == 2:
        ref_to_pred_matrix_shape = tuple(int(value) for value in np.asarray(ref_to_pred_score_matrix).shape)
    ref_to_pred_region_of_interest_mask = load_document_binary_mask(
        document_bundle_dir=document_bundle_dir,
        mask_filename="ref_to_pred_region_of_interest_mask.npy",
        expected_shape=ref_to_pred_matrix_shape,
    )
    ref_to_pred_hough_input_mask = load_document_binary_mask(
        document_bundle_dir=document_bundle_dir,
        mask_filename="ref_to_pred_hough_input_mask.npy",
        expected_shape=ref_to_pred_matrix_shape,
    )

    # Temporary panels are deliberately isolated from the public per-document folders:
    # after the pair-level stitched PNG succeeds, these intermediate files are removed.
    if not return_image:
        temporary_panel_output_dir.mkdir(parents=True, exist_ok=True)

    # The top two rows keep the established 2x2 score-matrix layout.  The third
    # row adds the two binary preprocessing masks that decide which cells vote.
    fig = plt.figure(figsize=(20, 24.8), constrained_layout=False)
    grid_specification = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 1.0, 1.0, 0.42],
        hspace=0.30,
        wspace=0.18,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid_specification[0, 0]), fig.add_subplot(grid_specification[0, 1])],
            [fig.add_subplot(grid_specification[1, 0]), fig.add_subplot(grid_specification[1, 1])],
            [fig.add_subplot(grid_specification[2, 0]), fig.add_subplot(grid_specification[2, 1])],
        ]
    )
    metrics_axis = fig.add_subplot(grid_specification[3, :])
    fig.suptitle(
        f"{language_name} / {document_type} | document {document_index} | {document_name}\n"
        "No-Hough score matrices, best-combination ref_to_pred Hough overlays, and preprocessing masks",
        fontsize=15,
        y=0.995,
    )

    image_0 = draw_score_matrix_heatmap(
        axes[0, 0],
        ref_to_pred_score_matrix,
        "ref_to_pred score matrix without Hough transform",
    )
    image_1 = draw_score_matrix_heatmap(
        axes[0, 1],
        ref_to_ref_score_matrix,
        "ref_to_ref score matrix without Hough transform",
    )
    image_2 = draw_score_matrix_heatmap(
        axes[1, 0],
        ref_to_pred_score_matrix,
        "Best combination: raw Hough lines on ref_to_pred",
    )
    raw_line_count_drawn = draw_raw_hough_overlay(
        axes[1, 0],
        combination_record,
        show_line_labels=show_line_labels,
    )
    if raw_line_count_drawn > 0:
        axes[1, 0].legend(loc="upper right")

    image_3 = draw_score_matrix_heatmap(
        axes[1, 1],
        ref_to_pred_score_matrix,
        "Best combination: surviving lines after filtering on ref_to_pred",
    )
    surviving_line_count_drawn = draw_surviving_filtered_hough_overlay(
        axes[1, 1],
        combination_record,
        saved_figure_dpi=saved_figure_dpi,
        show_line_labels=show_line_labels,
    )
    if surviving_line_count_drawn > 0:
        axes[1, 1].legend(loc="upper right")

    draw_binary_mask_panel(
        axes[2, 0],
        ref_to_pred_region_of_interest_mask,
        "ref_to_pred Region of Interest mask",
        "inside Region of Interest",
    )
    draw_binary_mask_panel(
        axes[2, 1],
        ref_to_pred_hough_input_mask,
        "ref_to_pred final Hough input",
        "voters",
    )

    # Add colorbars only for score matrices; binary masks use black/white pixels directly.
    for axis, image in zip(axes[:2, :].ravel(), [image_0, image_1, image_2, image_3]):
        if image is not None:
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Score")

    # Draw the complete metric report inside the PNG itself, rather than relying on
    # a sidecar file that can become separated from the visual.
    metrics_axis.set_axis_off()
    metrics_axis.text(
        0.01,
        0.98,
        build_best_combination_metrics_text_for_panel(best_row),
        transform=metrics_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        linespacing=1.25,
    )

    output_png_path = (
        temporary_panel_output_dir
        / f"{TEMPORARY_BEST_COMBINATION_PANEL_FILENAME_PREFIX}__{document_output_stem_from_row(best_row)}.png"
    )
    # Explicit spacing avoids tight_layout warnings with colorbars plus the custom metrics axis.
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.035, top=0.93, hspace=0.34, wspace=0.20)
    if return_image:
        return render_figure_to_rgba_image(fig, saved_figure_dpi=saved_figure_dpi)
    fig.savefig(output_png_path, dpi=saved_figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_png_path


def remove_stale_temporary_best_combination_panel_files(temporary_panel_output_dir: Path) -> None:
    """Remove stale temporary panel PNGs from a previous interrupted run."""
    if not temporary_panel_output_dir.is_dir():
        return

    # Only remove files with the exact temporary prefix owned by this diagnostic script.
    temporary_panel_glob = f"{TEMPORARY_BEST_COMBINATION_PANEL_FILENAME_PREFIX}__*.png"
    for temporary_panel_path in temporary_panel_output_dir.glob(temporary_panel_glob):
        temporary_panel_path.unlink(missing_ok=True)


def remove_temporary_best_combination_panel_files(
    temporary_panel_paths: list[Path],
    temporary_panel_output_dir: Path,
) -> None:
    """Clean up temporary panel PNGs after the stitched pair-level PNG is complete."""
    for temporary_panel_path in temporary_panel_paths:
        temporary_panel_path.unlink(missing_ok=True)

    # The directory is hidden and purely transient; remove it when no unexpected files remain.
    try:
        temporary_panel_output_dir.rmdir()
    except OSError:
        pass


def render_figure_to_rgba_image(fig, *, saved_figure_dpi: int):
    """Render a Matplotlib figure to the same RGBA pixels that a temporary PNG would contain."""
    from PIL import Image

    image_buffer = io.BytesIO()
    try:
        fig.savefig(image_buffer, format="png", dpi=saved_figure_dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    image_buffer.seek(0)
    with Image.open(image_buffer) as source_image:
        return source_image.convert("RGBA")


def save_stitched_panel_images_for_pair(
    *,
    panel_images: list[Any],
    stitched_output_path: Path,
) -> Path | None:
    """Save already-rendered document panels into one pair-level contact sheet."""
    if not panel_images:
        return None

    from PIL import Image

    stitched_output_path.parent.mkdir(parents=True, exist_ok=True)
    max_panel_width = max(panel_image.width for panel_image in panel_images)
    max_panel_height = max(panel_image.height for panel_image in panel_images)
    row_count = math.ceil(len(panel_images) / BEST_COMBINATION_PANEL_GRID_COLUMN_COUNT)

    # Panels can differ by a few pixels because tight bounding boxes depend on labels.
    # Padding every tile to the largest width/height keeps the contact sheet aligned.
    stitched_image = Image.new(
        "RGBA",
        (
            BEST_COMBINATION_PANEL_GRID_COLUMN_COUNT * max_panel_width,
            row_count * max_panel_height,
        ),
        (255, 255, 255, 255),
    )
    for panel_index, panel_image in enumerate(panel_images):
        column_index = panel_index % BEST_COMBINATION_PANEL_GRID_COLUMN_COUNT
        row_index = panel_index // BEST_COMBINATION_PANEL_GRID_COLUMN_COUNT
        upper_left_corner = (column_index * max_panel_width, row_index * max_panel_height)
        stitched_image.alpha_composite(panel_image, upper_left_corner)

    try:
        stitched_image.save(stitched_output_path, optimize=True)
    finally:
        stitched_image.close()
    return stitched_output_path


def stitch_best_combination_visual_panels_for_pair(
    *,
    temporary_panel_paths: list[Path],
    stitched_output_path: Path,
) -> Path | None:
    """Stitch temporary per-document best panels into one pair-level contact sheet."""
    if not temporary_panel_paths:
        return None

    from PIL import Image

    panel_images = []
    try:
        for temporary_panel_path in temporary_panel_paths:
            with Image.open(temporary_panel_path) as source_image:
                panel_images.append(source_image.convert("RGBA"))
        return save_stitched_panel_images_for_pair(
            panel_images=panel_images,
            stitched_output_path=stitched_output_path,
        )
    finally:
        for panel_image in panel_images:
            panel_image.close()


def render_best_combination_visual_panels_for_pair(
    *,
    best_rows_dataframe: pd.DataFrame,
    skipped_documents: list[dict[str, Any]] | None,
    output_dir: Path,
    language_name: str,
    document_type: str,
    ref_to_pred_scores_pkl: Path,
    ref_to_ref_scores_pkl: Path,
    saved_figure_dpi: int,
    show_line_labels: bool,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
) -> Path | None:
    """Render successful and skipped document panels into one stitched sheet."""
    skipped_documents_for_panels = [
        skipped_document
        for skipped_document in (skipped_documents or [])
        if should_render_skipped_visual_panel(skipped_document)
    ]
    temporary_panel_output_dir = temporary_best_combination_panel_dir_for_pair(
        output_dir,
        language_name,
        document_type,
    )
    stitched_output_path = stitched_best_combination_panel_path_for_pair(output_dir, language_name, document_type)
    if best_rows_dataframe.empty and not skipped_documents_for_panels:
        stitched_output_path.unlink(missing_ok=True)
        return None

    # A previous failed run may have left hidden temporary PNGs behind; clear just those.
    remove_stale_temporary_best_combination_panel_files(temporary_panel_output_dir)

    panel_jobs: list[tuple[int, str, Any]] = []
    for _, best_row in best_rows_dataframe.iterrows():
        panel_jobs.append((int(best_row["document_index"]), "best", best_row))
    for skipped_document in skipped_documents_for_panels:
        panel_jobs.append((int(skipped_document["document_index"]), "skipped", skipped_document))
    panel_jobs.sort(key=lambda item: (item[0], item[1]))

    panel_images: list[Any] = []
    try:
        for row_position, (document_index, panel_kind, panel_payload) in enumerate(panel_jobs, start=1):
            if panel_kind == "best":
                best_row = panel_payload
                print(
                    f"[{best_row['main_language']} / {best_row['document_type']}] "
                    f"direct stitched best panel {row_position}/{len(panel_jobs)}: "
                    f"{document_index} {best_row['fname']}"
                )
                panel_images.append(
                    render_best_combination_visual_panel(
                        best_row=best_row,
                        temporary_panel_output_dir=temporary_panel_output_dir,
                        ref_to_pred_scores_pkl=ref_to_pred_scores_pkl,
                        ref_to_ref_scores_pkl=ref_to_ref_scores_pkl,
                        saved_figure_dpi=saved_figure_dpi,
                        show_line_labels=show_line_labels,
                        return_image=True,
                    )
                )
                continue

            skipped_document = panel_payload
            print(
                f"[{skipped_document.get('main_language')} / {skipped_document.get('document_type')}] "
                f"direct stitched skipped panel {row_position}/{len(panel_jobs)}: "
                f"{document_index} {skipped_document.get('fname')} reason={skipped_document.get('skip_reason')}"
            )
            panel_images.append(
                render_skipped_document_visual_panel(
                    skipped_document=skipped_document,
                    temporary_panel_output_dir=temporary_panel_output_dir,
                    ref_to_pred_scores_pkl=ref_to_pred_scores_pkl,
                    ref_to_ref_scores_pkl=ref_to_ref_scores_pkl,
                    saved_figure_dpi=saved_figure_dpi,
                    window_size=int(window_size),
                    window_stride=int(window_stride),
                    return_image=True,
                )
            )

        return save_stitched_panel_images_for_pair(
            panel_images=panel_images,
            stitched_output_path=stitched_output_path,
        )
    finally:
        for panel_image in panel_images:
            panel_image.close()


# ---------------------------------------------------------------------------
# CSV table and manifest writing
# ---------------------------------------------------------------------------


def dataframe_from_document_records(document_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a document table with stable columns, even when no rows exist."""
    # Pandas writes cleaner empty CSVs when columns are supplied explicitly.
    if not document_records:
        return pd.DataFrame(columns=DOCUMENT_TABLE_COLUMNS)

    # Convert Path/tuple values into readable strings/lists before CSV writing.
    normalized_records = [
        {
            key: json_ready_value(value)
            for key, value in record.items()
            if not str(key).startswith("_")
        }
        for record in document_records
    ]
    dataframe = pd.DataFrame(normalized_records)

    # Put the most useful columns first while preserving any extra metadata columns at the end.
    ordered_columns = [column for column in DOCUMENT_TABLE_COLUMNS if column in dataframe.columns]
    remaining_columns = [column for column in dataframe.columns if column not in ordered_columns]
    return dataframe[ordered_columns + remaining_columns]


def build_document_type_summary_dataframe(
    *,
    language_name: str,
    document_type: str,
    runfile_documents: list[dict[str, Any]],
    loadable_documents: list[dict[str, Any]],
    skipped_documents: list[dict[str, Any]],
    loaded_documents: list[dict[str, Any]],
    metrics_dataframe: pd.DataFrame,
    best_rows_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row summary table for one language/document-type pair."""
    # Count skipped reasons in separate columns so CSV sorting/filtering stays easy.
    skip_reason_counts = Counter(document.get("skip_reason", "unknown") for document in skipped_documents)

    # The one-row table mirrors the manifest but is convenient from spreadsheet tools.
    summary_row = {
        "main_language": language_name,
        "document_type": document_type,
        "runfile_document_count": len(runfile_documents),
        "loadable_document_count": len(loadable_documents),
        "loaded_document_count": len(loaded_documents),
        "skipped_document_count": len(skipped_documents),
        "compact_metric_row_count": int(len(metrics_dataframe)),
        "best_combination_count": int(len(best_rows_dataframe)),
    }
    for skip_reason, count in sorted(skip_reason_counts.items()):
        summary_row[f"skipped_{skip_reason}"] = count

    return pd.DataFrame([summary_row])


def write_pair_tables(
    *,
    output_dir: Path,
    language_name: str,
    document_type: str,
    metrics_dataframe: pd.DataFrame,
    best_rows_dataframe: pd.DataFrame,
    runfile_documents: list[dict[str, Any]],
    loadable_documents: list[dict[str, Any]],
    skipped_documents: list[dict[str, Any]],
    loaded_documents: list[dict[str, Any]],
) -> dict[str, str]:
    """Write compact CSV tables for one language/document-type pair."""
    pair_output_dir = language_document_type_output_dir(output_dir, language_name, document_type)
    pair_output_dir.mkdir(parents=True, exist_ok=True)

    # Save the compact scalar metrics table used for all graphs.
    metrics_csv_path = pair_output_dir / "compact_combination_metrics.csv"
    metrics_dataframe.to_csv(metrics_csv_path, index=False)

    # Save the best row per loaded document.
    best_csv_path = pair_output_dir / "best_combination_per_document.csv"
    best_rows_dataframe.to_csv(best_csv_path, index=False)

    # Save document-level audit tables.
    runfile_csv_path = pair_output_dir / "runfile_documents.csv"
    dataframe_from_document_records(runfile_documents).to_csv(runfile_csv_path, index=False)

    loadable_csv_path = pair_output_dir / "loadable_documents.csv"
    dataframe_from_document_records(loadable_documents).to_csv(loadable_csv_path, index=False)

    loaded_csv_path = pair_output_dir / "loaded_documents.csv"
    dataframe_from_document_records(loaded_documents).to_csv(loaded_csv_path, index=False)

    skipped_csv_path = pair_output_dir / "skipped_documents.csv"
    dataframe_from_document_records(skipped_documents).to_csv(skipped_csv_path, index=False)

    summary_csv_path = pair_output_dir / "document_type_summary.csv"
    build_document_type_summary_dataframe(
        language_name=language_name,
        document_type=document_type,
        runfile_documents=runfile_documents,
        loadable_documents=loadable_documents,
        skipped_documents=skipped_documents,
        loaded_documents=loaded_documents,
        metrics_dataframe=metrics_dataframe,
        best_rows_dataframe=best_rows_dataframe,
    ).to_csv(summary_csv_path, index=False)

    return {
        "compact_metrics_csv": str(metrics_csv_path),
        "best_combination_csv": str(best_csv_path),
        "runfile_documents_csv": str(runfile_csv_path),
        "loadable_documents_csv": str(loadable_csv_path),
        "loaded_documents_csv": str(loaded_csv_path),
        "skipped_documents_csv": str(skipped_csv_path),
        "document_type_summary_csv": str(summary_csv_path),
    }


def read_existing_pair_csv(csv_path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read an existing pair-level CSV table, returning an empty table when absent."""
    if not Path(csv_path).exists():
        return pd.DataFrame(columns=[] if columns is None else columns)
    try:
        return pd.read_csv(csv_path).fillna("")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=[] if columns is None else columns)


def existing_pair_table_paths(output_dir: Path, language_name: str, document_type: str) -> dict[str, Path]:
    """Return the existing CSV paths that are enough to rebuild stitched panels."""
    pair_output_dir = language_document_type_output_dir(output_dir, language_name, document_type)
    return {
        "best_combination_csv": pair_output_dir / "best_combination_per_document.csv",
        "skipped_documents_csv": pair_output_dir / "skipped_documents.csv",
        "document_type_summary_csv": pair_output_dir / "document_type_summary.csv",
    }


def merge_runfile_context_into_skipped_records(
    *,
    skipped_dataframe: pd.DataFrame,
    runfile_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach runfile text metadata to skipped rows loaded from existing CSV tables."""
    runfile_document_by_index = {
        int(document["document_index"]): dict(document)
        for document in runfile_documents
        if safe_int_or_none(document.get("document_index")) is not None
    }
    skipped_documents: list[dict[str, Any]] = []
    for raw_record in skipped_dataframe.to_dict(orient="records"):
        document_index = safe_int_or_none(raw_record.get("document_index"))
        if document_index is None:
            document_index = safe_int_or_none(raw_record.get("index"))
        if document_index is None:
            continue

        # Start with runfile context so plotting-only fallback code can rebuild
        # reference-self matrices when skipped diagnostics did not save them.
        merged_record = dict(runfile_document_by_index.get(int(document_index), {}))
        for key, value in raw_record.items():
            if value in (None, ""):
                continue
            merged_record[key] = value
        merged_record["document_index"] = int(document_index)
        merged_record["fname"] = Path(str(merged_record.get("fname", raw_record.get("fname", "")))).name
        skipped_documents.append(merged_record)
    return skipped_documents


def analyze_one_language_document_type_pair_from_existing_tables(
    *,
    language_name: str,
    document_type: str,
    runfile_documents: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Rebuild stitched panels from existing pair CSVs without rereading score tables."""
    print(f"\n=== {language_name} / {document_type} [restitch existing tables] ===")
    table_paths_by_name = existing_pair_table_paths(args.output_dir, language_name, document_type)

    best_rows_dataframe = read_existing_pair_csv(
        table_paths_by_name["best_combination_csv"],
        columns=COMPACT_METRIC_COLUMNS,
    )
    skipped_dataframe = read_existing_pair_csv(
        table_paths_by_name["skipped_documents_csv"],
        columns=DOCUMENT_TABLE_COLUMNS,
    )
    skipped_documents = merge_runfile_context_into_skipped_records(
        skipped_dataframe=skipped_dataframe,
        runfile_documents=runfile_documents,
    )

    if "document_index" in best_rows_dataframe.columns and not best_rows_dataframe.empty:
        best_rows_dataframe["document_index"] = pd.to_numeric(
            best_rows_dataframe["document_index"],
            errors="coerce",
        )
        best_rows_dataframe = best_rows_dataframe.dropna(subset=["document_index"]).copy()
        best_rows_dataframe["document_index"] = best_rows_dataframe["document_index"].astype(int)
        best_rows_dataframe = best_rows_dataframe.sort_values(["document_index"], kind="stable")

    skip_reason_counts = Counter(document.get("skip_reason", "unknown") for document in skipped_documents)
    print(f"[{language_name} / {document_type}] existing best rows: {len(best_rows_dataframe)}")
    print(f"[{language_name} / {document_type}] existing skipped rows: {len(skipped_documents)}")
    print(f"[{language_name} / {document_type}] skip reasons: {dict(skip_reason_counts)}")

    stale_artifact_count = remove_stale_per_document_visual_analysis_artifacts_for_pair(
        args.output_dir,
        language_name,
        document_type,
    )
    if stale_artifact_count > 0:
        print(f"[{language_name} / {document_type}] removed stale per-document artifacts: {stale_artifact_count}")

    stitched_best_visual_panel_path: Path | None = None
    if not args.skip_best_visual_panels:
        stitched_best_visual_panel_path = render_best_combination_visual_panels_for_pair(
            best_rows_dataframe=best_rows_dataframe,
            skipped_documents=skipped_documents,
            output_dir=args.output_dir,
            language_name=language_name,
            document_type=document_type,
            ref_to_pred_scores_pkl=args.ref_to_pred_scores_pkl,
            ref_to_ref_scores_pkl=args.ref_to_ref_scores_pkl,
            saved_figure_dpi=args.saved_figure_dpi,
            show_line_labels=not bool(args.hide_line_labels),
            window_size=int(args.window_size),
            window_stride=int(args.window_stride),
        )

    return {
        "language": language_name,
        "document_type": document_type,
        "runfile_document_count": len(runfile_documents),
        "loadable_document_count": int(len(best_rows_dataframe)),
        "loaded_document_count": int(len(best_rows_dataframe)),
        "skipped_document_count": len(skipped_documents),
        "skip_reason_counts": dict(skip_reason_counts),
        "compact_metric_row_count": 0,
        "best_combination_count": int(len(best_rows_dataframe)),
        "restitch_from_existing_tables": True,
        "stale_per_document_artifact_count_removed": int(stale_artifact_count),
        "graph_grid_count": 0,
        "graph_grid_paths": [],
        "stitched_best_visual_panel_count": 1 if stitched_best_visual_panel_path is not None else 0,
        "stitched_best_visual_panel_path": (
            str(stitched_best_visual_panel_path) if stitched_best_visual_panel_path is not None else None
        ),
        "best_document_metric_records": [
            {
                "main_language": str(best_row.get("main_language")),
                "document_type": str(best_row.get("document_type")),
                "document_index": safe_int_or_none(best_row.get("document_index")),
                "fname": str(best_row.get("fname")),
                **best_row_to_public_json_record(best_row),
            }
            for _, best_row in best_rows_dataframe.sort_values(["document_index"], kind="stable").iterrows()
        ],
        "tables": {key: str(value) for key, value in table_paths_by_name.items()},
    }


def analyze_one_language_document_type_pair(
    *,
    language_name: str,
    document_type: str,
    runfile_documents: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the complete workflow for one language/document-type pair."""
    print(f"\n=== {language_name} / {document_type} ===")

    # Load the one compact DataFrame for this language/document-type pair.
    metrics_dataframe, loadable_documents, skipped_documents, loaded_documents = load_language_document_type_metrics_dataframe(
        runfile_documents=runfile_documents,
        shards_dir=args.shards_dir,
        documents_per_shard=args.documents_per_shard,
        max_documents=args.max_documents,
    )

    # Report skip reasons before plotting.
    skip_reason_counts = Counter(document.get("skip_reason", "unknown") for document in skipped_documents)
    print(f"[{language_name} / {document_type}] runfile documents: {len(runfile_documents)}")
    print(f"[{language_name} / {document_type}] loadable documents: {len(loadable_documents)}")
    print(f"[{language_name} / {document_type}] loaded documents: {len(loaded_documents)}")
    print(f"[{language_name} / {document_type}] skipped documents: {len(skipped_documents)}")
    print(f"[{language_name} / {document_type}] skip reasons: {dict(skip_reason_counts)}")
    print(f"[{language_name} / {document_type}] compact metric rows: {len(metrics_dataframe):,}")

    # Reused output directories may contain artifacts from the older per-document layout.
    # Remove only those exact filenames so unrelated document-level diagnostics stay intact.
    stale_artifact_count = remove_stale_per_document_visual_analysis_artifacts_for_pair(
        args.output_dir,
        language_name,
        document_type,
    )
    if stale_artifact_count > 0:
        print(f"[{language_name} / {document_type}] removed stale per-document artifacts: {stale_artifact_count}")

    # Select best rows once so the text summary, stitched panel, CSV, and manifest agree exactly.
    best_rows_dataframe = select_best_combination_rows_per_document(metrics_dataframe)

    # Resolve geometry pointers only when stitched matrix/Hough panels are requested.
    # The 18-plot grids and scalar CSV/TXT/JSON outputs use the score table alone.
    if not args.skip_best_visual_panels and not best_rows_dataframe.empty:
        best_rows_dataframe = attach_best_geometry_source_pointers(best_rows_dataframe)

    # Generate graph grids unless explicitly disabled.
    graph_grid_paths: list[Path] = []
    if not args.skip_graph_grids:
        graph_grid_paths = plot_parameter_metric_grids_for_pair(
            metrics_dataframe=metrics_dataframe,
            output_dir=args.output_dir,
            max_continuous_bins=args.max_continuous_bins,
            saved_figure_dpi=args.saved_figure_dpi,
        )

    # Generate the final stitched best-combination panel unless explicitly disabled.
    stitched_best_visual_panel_path: Path | None = None
    if not args.skip_best_visual_panels:
        stitched_best_visual_panel_path = render_best_combination_visual_panels_for_pair(
            best_rows_dataframe=best_rows_dataframe,
            skipped_documents=skipped_documents,
            output_dir=args.output_dir,
            language_name=language_name,
            document_type=document_type,
            ref_to_pred_scores_pkl=args.ref_to_pred_scores_pkl,
            ref_to_ref_scores_pkl=args.ref_to_ref_scores_pkl,
            saved_figure_dpi=args.saved_figure_dpi,
            show_line_labels=not bool(args.hide_line_labels),
            window_size=int(args.window_size),
            window_stride=int(args.window_stride),
        )

    # Persist the tables after best rows are known.
    table_paths = write_pair_tables(
        output_dir=args.output_dir,
        language_name=language_name,
        document_type=document_type,
        metrics_dataframe=metrics_dataframe,
        best_rows_dataframe=best_rows_dataframe,
        runfile_documents=runfile_documents,
        loadable_documents=loadable_documents,
        skipped_documents=skipped_documents,
        loaded_documents=loaded_documents,
    )

    return {
        "language": language_name,
        "document_type": document_type,
        "runfile_document_count": len(runfile_documents),
        "loadable_document_count": len(loadable_documents),
        "loaded_document_count": len(loaded_documents),
        "skipped_document_count": len(skipped_documents),
        "skip_reason_counts": dict(skip_reason_counts),
        "compact_metric_row_count": int(len(metrics_dataframe)),
        "best_combination_count": int(len(best_rows_dataframe)),
        "stale_per_document_artifact_count_removed": int(stale_artifact_count),
        "graph_grid_count": len(graph_grid_paths),
        "stitched_best_visual_panel_count": 1 if stitched_best_visual_panel_path is not None else 0,
        "graph_grid_paths": [str(path) for path in graph_grid_paths],
        "stitched_best_visual_panel_path": (
            str(stitched_best_visual_panel_path) if stitched_best_visual_panel_path is not None else None
        ),
        "best_document_metric_records": [
            {
                "main_language": str(best_row.get("main_language")),
                "document_type": str(best_row.get("document_type")),
                "document_index": safe_int_or_none(best_row.get("document_index")),
                "fname": str(best_row.get("fname")),
                **best_row_to_public_json_record(best_row),
            }
            for _, best_row in best_rows_dataframe.sort_values(["document_index"], kind="stable").iterrows()
        ],
        "tables": table_paths,
    }


# ---------------------------------------------------------------------------
# CLI and main entry point
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the language/type diagnostic script."""
    parser = argparse.ArgumentParser(
        description="Create language/document-type Hough tuner metric graphs and best-combination visuals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        default=None,
        help="main_language value to analyze. Repeat for multiple languages.",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Analyze every main_language value present in outputs.json.",
    )
    parser.add_argument(
        "--document-type",
        action="append",
        dest="document_types",
        default=None,
        help="document_type value to analyze. Repeat for multiple types. Actual values are read from outputs.json.",
    )
    parser.add_argument(
        "--all-document-types",
        action="store_true",
        help="Analyze every document_type value present in outputs.json.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum number of loadable documents to load per language/document-type pair.",
    )
    parser.add_argument("--runfile-json", type=Path, default=DEFAULT_RUNFILE_JSON, help="Path to outputs.json with main_language and document_type metadata.")
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR, help="Path to the tuner output shards directory.")
    parser.add_argument("--documents-per-shard", type=int, default=DEFAULT_DOCUMENTS_PER_SHARD, help="Number of document indices stored in each shard directory.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="Sliding score-matrix window size used when plotting must rebuild a missing ref_to_ref matrix.")
    parser.add_argument("--window-stride", type=int, default=DEFAULT_WINDOW_STRIDE, help="Sliding score-matrix window stride used when plotting must rebuild a missing ref_to_ref matrix.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where CSVs and PNGs will be written.")
    parser.add_argument("--ref-to-pred-scores-pkl", type=Path, default=DEFAULT_REF_TO_PRED_SCORES_PKL, help="Fallback ref_to_pred score pkl path.")
    parser.add_argument("--ref-to-ref-scores-pkl", type=Path, default=DEFAULT_REF_TO_REF_SCORES_PKL, help="Fallback ref_to_ref score pkl path.")
    parser.add_argument("--skip-graph-grids", action="store_true", help="Skip the 18-line-graph grid per document.")
    parser.add_argument(
        "--skip-best-visual-panels",
        action="store_true",
        help="Skip temporary best-combination panels and the final stitched panel.",
    )
    parser.add_argument(
        "--hide-line-labels",
        action="store_true",
        help="Hide raw Hough and surviving filtered-line labels in stitched best-combination panels.",
    )
    parser.add_argument(
        "--restitch-from-existing-tables",
        action="store_true",
        help="Regenerate stitched panels from existing pair CSVs without rereading combination score tables.",
    )
    parser.add_argument("--max-continuous-bins", type=int, default=50, help="Maximum bins for continuous component-vs-score line graphs.")
    parser.add_argument("--saved-figure-dpi", type=int, default=140, help="DPI for saved PNG figures.")
    return parser.parse_args()


def validate_numeric_arguments(args: argparse.Namespace) -> None:
    """Validate numeric arguments that argparse cannot fully constrain."""
    # max_documents is optional, but when present it must be positive.
    if args.max_documents is not None and args.max_documents <= 0:
        raise ConfigurationError("--max-documents must be a positive integer when provided.")

    # documents_per_shard is part of path calculation, so zero/negative values are invalid.
    if args.documents_per_shard <= 0:
        raise ConfigurationError("--documents-per-shard must be a positive integer.")

    # Plotting bins and DPI must be positive to keep matplotlib happy.
    if args.max_continuous_bins <= 0:
        raise ConfigurationError("--max-continuous-bins must be a positive integer.")
    if args.saved_figure_dpi <= 0:
        raise ConfigurationError("--saved-figure-dpi must be a positive integer.")
    if args.window_size <= 0:
        raise ConfigurationError("--window-size must be a positive integer.")
    if args.window_stride <= 0:
        raise ConfigurationError("--window-stride must be a positive integer.")


def resolve_cli_selection(args: argparse.Namespace, runfile_items: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Resolve and validate requested languages and document types."""
    # Extract exactly the values present in outputs.json.
    available_languages = available_metadata_values(runfile_items, "main_language")
    available_document_types = available_metadata_values(runfile_items, "document_type")

    # Resolve language and document-type selections independently.
    selected_languages = selected_values_from_arguments(
        explicit_values=args.languages,
        select_all=bool(args.all_languages),
        available_values=available_languages,
        field_label="language",
    )
    selected_document_types = selected_values_from_arguments(
        explicit_values=args.document_types,
        select_all=bool(args.all_document_types),
        available_values=available_document_types,
        field_label="document-type",
    )

    return selected_languages, selected_document_types


def run_language_hough_parameter_metric_analysis_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Run requested analyses from an argparse-like namespace and return the manifest."""
    validate_numeric_arguments(args)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read metadata before resolving --all-languages or --all-document-types.
    runfile_items = load_runfile_items(args.runfile_json)
    selected_languages, selected_document_types = resolve_cli_selection(args, runfile_items)

    print(f"Runfile JSON: {args.runfile_json}")
    print(f"Shards directory: {args.shards_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Selected languages: {selected_languages}")
    print(f"Selected document types: {selected_document_types}")
    print(f"Documents per shard: {args.documents_per_shard}")
    print(f"Max loadable documents per language/document-type pair: {args.max_documents}")

    # Group runfile documents by every selected language/type pair.
    documents_by_pair = build_selected_runfile_documents(
        runfile_items=runfile_items,
        selected_languages=selected_languages,
        selected_document_types=selected_document_types,
    )

    # Run each pair independently so each pair has exactly one compact DataFrame.
    pair_manifests = []
    public_best_document_records: list[dict[str, Any]] = []
    for language_name in selected_languages:
        for document_type in selected_document_types:
            if bool(args.restitch_from_existing_tables):
                pair_manifest = analyze_one_language_document_type_pair_from_existing_tables(
                    language_name=language_name,
                    document_type=document_type,
                    runfile_documents=documents_by_pair[(language_name, document_type)],
                    args=args,
                )
            else:
                pair_manifest = analyze_one_language_document_type_pair(
                    language_name=language_name,
                    document_type=document_type,
                    runfile_documents=documents_by_pair[(language_name, document_type)],
                    args=args,
                )
            pair_manifests.append(pair_manifest)
            public_best_document_records.extend(pair_manifest.get("best_document_metric_records", []))

    metrics_json_path = write_stitched_best_combination_metrics_json_from_public_records(
        public_document_records=public_best_document_records,
        output_json_path=stitched_best_combination_metrics_json_path(args.output_dir),
    )

    # Write a top-level manifest for easy audit and downstream automation.
    manifest = {
        "runfile_json": str(args.runfile_json),
        "shards_dir": str(args.shards_dir),
        "output_dir": str(args.output_dir),
        "selected_languages": selected_languages,
        "selected_document_types": selected_document_types,
        "max_documents_per_language_document_type_pair": args.max_documents,
        "documents_per_shard": args.documents_per_shard,
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "skip_graph_grids": bool(args.skip_graph_grids),
        "skip_best_visual_panels": bool(args.skip_best_visual_panels),
        "hide_line_labels": bool(args.hide_line_labels),
        "restitch_from_existing_tables": bool(args.restitch_from_existing_tables),
        "stitched_best_combination_metrics_json": str(metrics_json_path),
        "language_document_type_results": pair_manifests,
    }
    manifest_path = args.output_dir / "language_hough_parameter_metric_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_ready_value), encoding="utf-8")
    print(f"\nManifest written: {manifest_path}")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def generate_tuner_visualisation_outputs(
    *,
    runfile_json: Path,
    tuner_output_dir: Path,
    shards_dir: Path | None = None,
    documents_per_shard: int = DEFAULT_DOCUMENTS_PER_SHARD,
    ref_to_pred_scores_pkl: Path = DEFAULT_REF_TO_PRED_SCORES_PKL,
    ref_to_ref_scores_pkl: Path = DEFAULT_REF_TO_REF_SCORES_PKL,
    max_documents: int | None = None,
    hide_line_labels: bool = False,
    max_continuous_bins: int = 50,
    saved_figure_dpi: int = 140,
) -> dict[str, Any]:
    """Generate the final tuner visual outputs after sweep bundles exist.

    This function is the lazy-import integration point used by the tuner.  It
    deliberately calls the same analysis implementation as the CLI so the
    single-job and final-after-shards workflows cannot drift apart.
    """
    resolved_tuner_output_dir = Path(tuner_output_dir)
    resolved_shards_dir = Path(shards_dir) if shards_dir is not None else resolved_tuner_output_dir
    args = argparse.Namespace(
        languages=None,
        all_languages=True,
        document_types=None,
        all_document_types=True,
        max_documents=max_documents,
        runfile_json=Path(runfile_json),
        shards_dir=resolved_shards_dir,
        documents_per_shard=int(documents_per_shard),
        output_dir=resolved_tuner_output_dir,
        ref_to_pred_scores_pkl=Path(ref_to_pred_scores_pkl),
        ref_to_ref_scores_pkl=Path(ref_to_ref_scores_pkl),
        skip_graph_grids=False,
        skip_best_visual_panels=False,
        hide_line_labels=bool(hide_line_labels),
        max_continuous_bins=int(max_continuous_bins),
        saved_figure_dpi=int(saved_figure_dpi),
    )
    return run_language_hough_parameter_metric_analysis_from_args(args)


def main() -> int:
    """Run requested language/document-type analyses and write a manifest JSON."""
    args = parse_arguments()
    run_language_hough_parameter_metric_analysis_from_args(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigurationError as error:
        print(f"[error] {error}")
        raise SystemExit(2)
