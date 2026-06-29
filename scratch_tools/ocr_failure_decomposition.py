#!/usr/bin/env python3
from __future__ import annotations

"""Decompose OCR error per language using ONLY the pipeline's recognized-line metrics.

Goal: explain *why* raw OCR quality (document_normalised_levenshtein, "docNLS") is
low for some languages, using only signals the pipeline already produced from its
Hough-detected lines (plus the model's own per-document outputs).  We deliberately
do NOT inspect the raw score matrices: taking the best match per reference row /
prediction column ignores geometric coherence and destroys the repetition signal,
which is the exact thing the line-detection pipeline exists to handle.  Any layout
attribution would need page images / bounding boxes and is out of scope here.

Failure modes, all from recognized-line metrics:
  - missing      : reference not covered by any recognized line  (omission)
  - repetition   : reference covered more than once               (repetition_on_reference)
  - substitution : covered correctly but characters wrong         (correct x (1 - weighted_nls))
  - good         : covered correctly and characters right         (correct x weighted_nls)
Companion (model/prediction-level, shown separately, never summed in):
  - hallucination     : predicted text covered by no line
  - self-repetition   : model decoding loop (outputs.json `repetition`)

Data sources, joined on document basename:
  1. balanced CSV best_combination_per_document.csv  (recognized-line metrics, 976 docs)
  2. outputs.json                                     (main_script, self-repetition, is_empty)

Deterministic (sorted iteration, no RNG) and fully parameterized; runs on any
same-format inputs.

Usage:
  /appl/soft/ai/wrap/pytorch-2.9/bin/python3 scratch_tools/ocr_failure_decomposition.py \\
    --outputs-json results/<run>/vllm/dev/outputs.json \\
    --results-dir  results/<tuner-run>/balanced
"""

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Presentation constants
# --------------------------------------------------------------------------- #
FIGURE_DPI = 200
FAILURE_DOC_NLS_CUTOFF = 0.5   # a doc is a "catastrophic failure" below this docNLS
WORST_DOCS_PER_LANGUAGE = 10   # how many worst docs to list in the drill-down CSV

# "Stuck-repeating" = the model loops on one small part of the reference. The faithful
# signal is the recognized-line re-cover DEPTH: how many times the same reference window is
# covered by recognized lines. A small region repeated many times has LOW breadth (small
# repetition_on_reference / missing, and possibly high hallucination) but VERY HIGH depth, so
# breadth-based gates miss it. Detection is two cheap+faithful stages:
#   1) CSV pre-screen (limits how many per-doc pickles we open): some reference repetition +
#      some coverage;
#   2) flag when the recognized-line max re-cover depth reaches STUCK_DEPTH_MIN.
STUCK_REP_MIN = 0.02        # pre-screen: more than this much repetition_on_reference
STUCK_CORRECT_MIN = 0.0     # pre-screen: strictly more than this much correct coverage
STUCK_DEPTH_MIN = 10        # flag: recognized-line max re-cover depth at/above this
STUCK_EXEMPLAR_COUNT = 8    # how many docs to show in the recognized-line exemplar figure

# Okabe-Ito colorblind-safe palette, fixed meaning across every figure.
COLOR = {
    "good": "#009E73",          # bluish green
    "substitution": "#0072B2",  # blue
    "missing": "#E69F00",       # orange
    "repetition": "#D55E00",    # vermilion
    "correct": "#BBBBBB",       # neutral grey (partition figure)
    "hallucination": "#CC79A7", # reddish purple (companion)
    "self_repetition": "#56B4E9",  # sky blue (companion)
    "violin": "#74C0FC",
}
SCRIPT_BAND_SHADES = ("#00000000", "#0000000D")  # transparent / very light grey, alternating

# Distribution panels: (label, doc-field, uses-line-metric?)
DISTRIBUTION_METRICS = [
    ("docNLS (raw quality)", "doc_nls", False),
    ("weighted-along-lines NLS", "weighted_nls", True),
    ("correct_ref_coverage", "correct", True),
    ("missing_ref_coverage", "missing", True),
    ("repetition_on_reference", "rep_on_ref", True),
    ("hallucination", "hallucination", True),
    ("model self-repetition", "self_repetition", False),
]


def raise_csv_field_size_limit() -> None:
    """Allow very large CSV fields (full document text / serialised metrics)."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit // 10)


raise_csv_field_size_limit()


def to_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def basename(name: Any) -> str:
    return os.path.basename(str(name))


def script_family(main_script: Any) -> str:
    """Collapse a granular main_script into a coarse, data-driven writing-system family."""
    text = str(main_script or "Unknown").strip()
    head = text.split("(")[0].split(",")[0].strip() or "Unknown"
    merge = {
        "Han": "CJK", "Japanese": "CJK",
        "Bengali": "Indic", "Devanagari": "Indic", "Newa": "Indic",
        "Arabic": "Arabic/RTL", "Hebrew": "Arabic/RTL",
    }
    return merge.get(head, head)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_outputs_json(path: Path) -> dict[str, dict[str, Any]]:
    with open(path) as handle:
        data = json.load(handle)
    return {basename(item.get("file_name")): item for item in data}


def load_balanced_partition(results_dir: Path) -> dict[str, dict[str, Any]]:
    path = results_dir / "best_combination_per_document.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing recognized-line metrics CSV: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows[basename(row.get("fname"))] = row
    return rows


# --------------------------------------------------------------------------- #
# Build the per-document table (line metrics only) + universe filter
# --------------------------------------------------------------------------- #
def _has_prediction(output_record: dict[str, Any] | None) -> bool:
    if output_record is None:
        return True  # present in the scored CSV but absent from outputs.json -> assume real
    if output_record.get("is_empty"):
        return False
    predicted = output_record.get("normalized_predicted_text")
    if predicted is None:
        predicted = output_record.get("predicted_text")
    if predicted is not None and not str(predicted).strip():
        return False
    return True


def build_documents(outputs, partition) -> tuple[list[dict[str, Any]], dict[str, int]]:
    keys = sorted(set(outputs) | set(partition))
    documents: list[dict[str, Any]] = []
    counts = {"all": len(keys), "excluded_no_prediction": 0, "included": 0, "scored": 0}

    for key in keys:
        output_record = outputs.get(key)
        partition_row = partition.get(key)
        if not _has_prediction(output_record):
            counts["excluded_no_prediction"] += 1
            continue

        correct = missing = rep_on_ref = hallucination = weighted_nls = None
        doc_nls = None
        language = document_type = None
        pkl_path = None
        scored = False
        if partition_row is not None:
            correct = to_number(partition_row.get("correct_ref_coverage"))
            missing = to_number(partition_row.get("missing_ref_coverage"))
            rep_on_ref = to_number(partition_row.get("repetition_on_reference"))
            hallucination = to_number(partition_row.get("hallucination"))
            weighted_nls = to_number(partition_row.get("weighted_along_lines_normalised_levenshtein"))
            doc_nls = to_number(partition_row.get("document_normalised_levenshtein"))
            language = partition_row.get("main_language")
            document_type = partition_row.get("document_type")
            pkl_path = partition_row.get("alpha_sweep_pickle_path")
            scored = None not in (correct, missing, rep_on_ref)

        if output_record is not None:
            if doc_nls is None:
                doc_nls = to_number(output_record.get("normalized_levenshtein_similarity"))
            language = language or output_record.get("main_language")
            document_type = document_type or output_record.get("document_type")

        documents.append({
            "fname": key,
            "language": language or "Unknown",
            "script_family": script_family((output_record or {}).get("main_script")),
            "main_script": (output_record or {}).get("main_script"),
            "document_type": document_type or "Unknown",
            "doc_nls": doc_nls,
            "self_repetition": to_number((output_record or {}).get("repetition")),
            "correct": correct, "missing": missing, "rep_on_ref": rep_on_ref,
            "hallucination": hallucination, "weighted_nls": weighted_nls,
            "pkl_path": pkl_path, "stuck": False,
            "scored": scored,
        })
        counts["included"] += 1
        if scored:
            counts["scored"] += 1
    return documents, counts


def decompose_reference(document: dict[str, Any]) -> dict[str, float] | None:
    """Exact reference-axis split (sums to 1) from recognized-line metrics only.

    correct = good + substitution, so good/substitution/missing/repetition sum to 1.
    """
    if not document["scored"]:
        return None
    correct = document["correct"] or 0.0
    weighted = document["weighted_nls"]
    weighted = 1.0 if weighted is None else max(0.0, min(1.0, weighted))
    return {
        "good": correct * weighted,
        "substitution": correct * (1.0 - weighted),
        "missing": document["missing"] or 0.0,
        "repetition": document["rep_on_ref"] or 0.0,
    }


def is_stuck_candidate(document, *, rep_min, correct_min) -> bool:
    """Cheap CSV pre-screen for stuck-repeating (limits how many pickles we open).

    Final flagging uses the recognized-line re-cover depth (see main); this only narrows
    the set to documents that have some reference repetition and some coverage.
    """
    if not document["scored"]:
        return False
    repetition = document["rep_on_ref"]; correct = document["correct"]
    if repetition is None or correct is None:
        return False
    return repetition > rep_min and correct > correct_min


def recognized_line_descriptors(pkl_path, *, keep_payload=False) -> dict | None:
    """Descriptors computed ONLY from the pipeline's recognized lines (no best-match).

    Loads selected_plot_payload.final_surviving_ref_to_pred_lines and reports how much of the
    reference (y) vs the prediction (x) axis the recognized lines span and the re-cover DEPTH
    (how many times reference windows are covered). With keep_payload=True it also returns the
    matrix + lines for the exemplar plot; otherwise those are dropped to keep memory low when
    scanning many candidates.
    """
    import pickle
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        with open(pkl_path, "rb") as handle:
            payload = pickle.load(handle)["selected_plot_payload"]
    except Exception:
        return None
    matrix = np.asarray(payload["ref_to_pred_score_matrix"], dtype=float)
    rows, cols = matrix.shape
    lines = payload.get("final_surviving_ref_to_pred_lines") or []
    ref_hits = np.zeros(rows, dtype=int)
    pred_hits = np.zeros(cols, dtype=int)
    for line in lines:
        y_low, y_high = sorted((line["y0"], line["y1"]))
        x_low, x_high = sorted((line["x0"], line["x1"]))
        ref_hits[int(max(0, y_low)):int(min(rows, y_high)) + 1] += 1
        pred_hits[int(max(0, x_low)):int(min(cols, x_high)) + 1] += 1
    covered = ref_hits > 0
    result = {
        "n_lines": len(lines),
        "reference_coverage": float(covered.mean()),
        "prediction_coverage": float((pred_hits > 0).mean()),
        "mean_recover_depth": float(ref_hits[covered].mean()) if covered.any() else 0.0,
        "max_recover_depth": int(ref_hits.max()) if ref_hits.size else 0,
    }
    if keep_payload:
        result["matrix"] = matrix
        result["lines"] = lines
    return result


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def values(documents, language, key, *, scored_only=False) -> list[float]:
    out = []
    for doc in documents:
        if doc["language"] != language:
            continue
        if scored_only and not doc["scored"]:
            continue
        value = doc.get(key)
        if value is not None:
            out.append(float(value))
    return out


def mean_or_nan(numbers) -> float:
    return sum(numbers) / len(numbers) if numbers else float("nan")


def mean_doc_nls(documents, language) -> float:
    vals = values(documents, language, "doc_nls")
    return mean_or_nan(vals) if vals else -1.0


def language_orderings(documents) -> dict[str, list[str]]:
    languages = sorted({doc["language"] for doc in documents})
    family_of = {}
    for doc in documents:
        family_of.setdefault(doc["language"], doc["script_family"])
    by_quality = sorted(languages, key=lambda L: (-mean_doc_nls(documents, L), L))
    by_script = sorted(languages, key=lambda L: (family_of.get(L, "Unknown"),
                                                 -mean_doc_nls(documents, L), L))
    return {"by_quality": by_quality, "by_script": by_script}


def family_boundaries(documents, order) -> list[tuple[str, int, int]]:
    family_of = {}
    for doc in documents:
        family_of.setdefault(doc["language"], doc["script_family"])
    spans, start = [], 0
    for index in range(1, len(order) + 1):
        if index == len(order) or family_of.get(order[index]) != family_of.get(order[start]):
            spans.append((family_of.get(order[start], "Unknown"), start, index))
            start = index
    return spans


# --------------------------------------------------------------------------- #
# Plotting scaffolding
# --------------------------------------------------------------------------- #
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "figure.titlesize": 18, "savefig.dpi": FIGURE_DPI,
    })
    return plt


def _decorate_language_axis(axis, order, documents, *, by_script, rotation=60) -> None:
    """Shared x decoration: tick labels + (for by_script) shaded bands and separators.

    Family *headers* are added separately by place_family_headers() AFTER plotting,
    so they can sit in dedicated headroom above the data and never overlap titles.
    """
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order, rotation=rotation, ha="right")
    axis.set_xlim(-0.8, len(order) - 0.2)
    if not by_script:
        return
    for band_index, (_family, start, end) in enumerate(family_boundaries(documents, order)):
        shade = SCRIPT_BAND_SHADES[band_index % len(SCRIPT_BAND_SHADES)]
        axis.axvspan(start - 0.5, end - 0.5, color=shade, zorder=0)
        if start > 0:
            axis.axvline(start - 0.5, color="black", linewidth=0.7, alpha=0.3)


def place_family_headers(axis, order, documents) -> None:
    """Add bold script-family headers in headroom above the data (no title overlap)."""
    ymin, ymax = axis.get_ylim()
    span = ymax - ymin
    axis.set_ylim(ymin, ymax + 0.20 * span)
    label_y = ymax + 0.06 * span
    for family, start, end in family_boundaries(documents, order):
        axis.text((start + end - 1) / 2.0, label_y, family, ha="center", va="bottom",
                  fontsize=10, fontweight="bold", clip_on=False,
                  bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0))


def _deterministic_jitter(count, width=0.30) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width, width, count)


# --------------------------------------------------------------------------- #
# Figure 1: metric distributions
# --------------------------------------------------------------------------- #
def figure_metric_distributions(documents, order, *, by_script, style, out_path,
                                metrics=None, show_basis=True) -> Path:
    plt = _setup_matplotlib()
    metrics = metrics if metrics is not None else DISTRIBUTION_METRICS
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(max(16, len(order) * 0.62), 4.6 * rows))
    flat = list(np.ravel(axes))
    for panel_index, (label, key, line_metric) in enumerate(metrics):
        axis = flat[panel_index]
        per_language = [values(documents, lang, key, scored_only=line_metric) for lang in order]
        for position, data in enumerate(per_language):
            if not data:
                continue
            if style == "violin" and len(data) >= 2:
                parts = axis.violinplot([data], positions=[position], widths=0.85,
                                        showmeans=False, showmedians=True)
                for body in parts["bodies"]:
                    body.set_facecolor(COLOR["violin"]); body.set_alpha(0.65)
                if "cmedians" in parts:
                    parts["cmedians"].set_color("black")
            else:
                axis.boxplot([data], positions=[position], widths=0.62, showfliers=False,
                             patch_artist=True,
                             boxprops=dict(facecolor=COLOR["violin"], alpha=0.6),
                             medianprops=dict(color="black"))
                axis.scatter(position + _deterministic_jitter(len(data)), sorted(data),
                             s=5, color="#222222", alpha=0.30, zorder=3)
        # Show the "(scored docs)" basis only; never render "(all docs w/ prediction)".
        if show_basis and line_metric:
            axis.set_title(f"{label}  (scored docs)")
        else:
            axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        _decorate_language_axis(axis, order, documents, by_script=by_script, rotation=70)
        if by_script and panel_index < columns:  # label families once, on the top row
            place_family_headers(axis, order, documents)
    for extra in range(len(metrics), len(flat)):
        flat[extra].set_axis_off()
    ordering = "grouped by script" if by_script else "ordered by quality (mean docNLS)"
    fig.suptitle(f"Per-language metric distributions — {style}, {ordering}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Figure 2: exact reference coverage partition
# --------------------------------------------------------------------------- #
def figure_coverage_partition(documents, order, *, by_script, out_path) -> Path:
    plt = _setup_matplotlib()
    fig, axis = plt.subplots(figsize=(max(16, len(order) * 0.62), 8.5))
    correct = [mean_or_nan(values(documents, L, "correct", scored_only=True)) for L in order]
    missing = [mean_or_nan(values(documents, L, "missing", scored_only=True)) for L in order]
    repetition = [mean_or_nan(values(documents, L, "rep_on_ref", scored_only=True)) for L in order]
    x = np.arange(len(order))
    correct_a = np.nan_to_num(np.array(correct))
    missing_a = np.nan_to_num(np.array(missing))
    repetition_a = np.nan_to_num(np.array(repetition))
    axis.bar(x, correct_a, color=COLOR["correct"], label="correct coverage")
    axis.bar(x, missing_a, bottom=correct_a, color=COLOR["missing"], label="missing (omission)")
    axis.bar(x, repetition_a, bottom=correct_a + missing_a, color=COLOR["repetition"],
             label="repetition on reference")
    axis.set_ylabel("fraction of reference (sums to 1)")
    axis.set_ylim(0, 1.02)
    ordering = "grouped by script" if by_script else "ordered by quality"
    axis.set_title(f"Exact reference coverage partition per language (scored docs, {ordering})")
    axis.legend(loc="lower left", ncol=3, framealpha=0.9)
    _decorate_language_axis(axis, order, documents, by_script=by_script)
    if by_script:
        place_family_headers(axis, order, documents)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Figure 3: defensible reference-axis error decomposition + companion panel
# --------------------------------------------------------------------------- #
def figure_reference_decomposition(documents, order, *, by_script, out_path) -> Path:
    plt = _setup_matplotlib()
    fig, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(max(16, len(order) * 0.62), 12),
        gridspec_kw={"height_ratios": [3, 1.3]}, sharex=True)

    modes = [("good", "good (covered + correct chars)"),
             ("substitution", "substitution (covered, wrong chars)"),
             ("missing", "missing (omission)"),
             ("repetition", "repetition on reference")]
    stacks = {mode: [] for mode, _ in modes}
    for language in order:
        per_mode = {mode: [] for mode, _ in modes}
        for doc in documents:
            if doc["language"] != language:
                continue
            parts = decompose_reference(doc)
            if parts is None:
                continue
            for mode, _ in modes:
                per_mode[mode].append(parts[mode])
        for mode, _ in modes:
            stacks[mode].append(mean_or_nan(per_mode[mode]) if per_mode[mode] else 0.0)

    x = np.arange(len(order))
    base = np.zeros(len(order))
    for mode, label in modes:
        heights = np.nan_to_num(np.array(stacks[mode]))
        axis_top.bar(x, heights, bottom=base, color=COLOR[mode], label=label)
        base = base + heights
    axis_top.set_ylabel("fraction of reference (sums to 1)")
    axis_top.set_ylim(0, 1.02)
    ordering = "grouped by script" if by_script else "ordered by quality"
    axis_top.set_title(f"Reference error decomposition per language (scored docs, {ordering})")
    axis_top.legend(loc="lower left", ncol=4, framealpha=0.9)

    # Companion panel (separate axis: prediction-level / model-level signals, NOT summed in)
    hallucination = [mean_or_nan(values(documents, L, "hallucination", scored_only=True)) for L in order]
    self_rep = [mean_or_nan(values(documents, L, "self_repetition")) for L in order]
    width = 0.4
    axis_bottom.bar(x - width / 2, np.nan_to_num(hallucination), width,
                    color=COLOR["hallucination"], label="hallucination (prediction not on page)")
    axis_bottom.bar(x + width / 2, np.nan_to_num(self_rep), width,
                    color=COLOR["self_repetition"], label="model self-repetition (decoding loop)")
    axis_bottom.set_ylabel("fraction")
    axis_bottom.set_title("Companion signals (separate axes — not part of the reference partition)")
    axis_bottom.legend(loc="upper left", ncol=2, framealpha=0.9)
    axis_bottom.grid(axis="y", alpha=0.25)

    _decorate_language_axis(axis_top, order, documents, by_script=by_script)
    axis_top.set_xticklabels([])
    _decorate_language_axis(axis_bottom, order, documents, by_script=by_script)
    if by_script:
        place_family_headers(axis_top, order, documents)
    fig.text(0.5, 0.005,
             "good/substitution split applies along-line character quality to the correctly-covered "
             "fraction; line metrics only — raw matrices never inspected.",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Figure 4: catastrophic-failure subset per language
# --------------------------------------------------------------------------- #
def figure_failure_subset(documents, order, *, cutoff, out_path) -> Path:
    plt = _setup_matplotlib()
    fig, axis = plt.subplots(figsize=(max(16, len(order) * 0.62), 7))
    fractions, counts = [], []
    for language in order:
        vals = values(documents, language, "doc_nls")
        if not vals:
            fractions.append(0.0); counts.append((0, 0)); continue
        failures = sum(1 for v in vals if v < cutoff)
        fractions.append(100.0 * failures / len(vals))
        counts.append((failures, len(vals)))
    x = np.arange(len(order))
    axis.bar(x, fractions, color=COLOR["missing"])
    axis.set_ylim(0, max(fractions + [1.0]) * 1.15 + 2)
    for xi, (failures, total) in zip(x, counts):
        if total:
            axis.text(xi, fractions[int(xi)] + 1, f"{failures}/{total}", ha="center",
                      va="bottom", fontsize=8)
    axis.set_ylabel(f"% of docs with docNLS < {cutoff:g}")
    axis.set_title(f"Catastrophic-failure subset per language (docNLS < {cutoff:g})\n"
                   "reveals bimodal languages whose mean is dragged down by a few failing pages")
    axis.grid(axis="y", alpha=0.25)
    _decorate_language_axis(axis, order, documents, by_script=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Stuck-repeating figures
# --------------------------------------------------------------------------- #
def figure_stuck_signature(documents, descriptors, *, depth_min, rep_min, out_path) -> Path:
    """Depth view: recognized-line max re-cover depth vs repetition, coloured by hallucination.

    Each point is a candidate (pre-screened doc that has some reference repetition). Y is the
    recognized-line max re-cover depth — the faithful "stuck on a part" signal. Points at/above
    the dashed depth threshold are flagged (ringed + labelled). Note hallucination is shown
    (colour) but does NOT gate the flag: a doc can repeat a region AND hallucinate.
    """
    plt = _setup_matplotlib()
    fig, axis = plt.subplots(figsize=(12, 9))
    by_fname = {d["fname"]: d for d in documents}
    points = [(by_fname[f], desc) for f, desc in descriptors.items() if f in by_fname]
    if not points:
        axis.text(0.5, 0.5, "no candidates", ha="center", va="center", transform=axis.transAxes)
    else:
        xs = [max(d["rep_on_ref"] or 0.0, 1e-4) for d, _ in points]
        ys = [max(desc["max_recover_depth"], 1) for _, desc in points]
        cs = [min(d["hallucination"] or 0.0, 0.5) for d, _ in points]
        scatter = axis.scatter(xs, ys, c=cs, cmap="viridis_r", s=40, alpha=0.85,
                               edgecolors="none", zorder=3)
        fig.colorbar(scatter, ax=axis, label="hallucination (does NOT gate the flag; shown for context)")
        axis.axhline(depth_min, color=COLOR["repetition"], linestyle="--", linewidth=1.5,
                     label=f"stuck threshold (max re-cover depth ≥ {depth_min})")
        flagged = [(d, desc) for d, desc in points if desc["max_recover_depth"] >= depth_min]
        for d, desc in flagged:
            axis.scatter([max(d["rep_on_ref"] or 0.0, 1e-4)], [desc["max_recover_depth"]],
                         s=150, facecolors="none", edgecolors=COLOR["repetition"],
                         linewidths=1.8, zorder=4)
            axis.annotate(d["fname"][:22], (max(d["rep_on_ref"] or 0.0, 1e-4), desc["max_recover_depth"]),
                          fontsize=7, xytext=(4, 3), textcoords="offset points")
        axis.set_yscale("log")
        axis.legend(loc="lower right")
    axis.set_xlabel("repetition_on_reference (breadth: fraction of reference covered >1×)")
    axis.set_ylabel("recognized-line max re-cover depth (log) — times one reference window is re-covered")
    axis.set_title(f"Stuck-repeating signature: depth, not breadth ({len(descriptors)} candidates)\n"
                   "stuck = a small reference region the recognized lines pile onto many times")
    axis.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def figure_stuck_line_exemplars(exemplars, *, out_path, dpi=300) -> Path | None:
    """Overlay the pipeline's recognized lines on a faint matrix for stuck-repeating docs.

    `exemplars` is a list of (document, descriptors) where descriptors carry matrix+lines.
    Display only — recognized lines, never best-match scanning. Rendered at high DPI so the
    stitched panel reads clearly.
    """
    if not exemplars:
        return None
    plt = _setup_matplotlib()
    from matplotlib.collections import LineCollection
    count = len(exemplars)
    columns = 2
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(9 * columns, 6.2 * rows))
    flat = list(np.ravel([axes]))
    for axis, (doc, desc) in zip(flat, exemplars):
        matrix = desc["matrix"]
        axis.imshow(matrix, origin="upper", aspect="auto", cmap="Greys",
                    vmin=0, vmax=100, alpha=0.45)
        segments = [[(ln["x0"], ln["y0"]), (ln["x1"], ln["y1"])] for ln in desc["lines"]]
        axis.add_collection(LineCollection(segments, colors=COLOR["repetition"],
                                           linewidths=1.1, alpha=0.9))
        axis.set_xlim(-0.5, matrix.shape[1] - 0.5)
        axis.set_ylim(matrix.shape[0] - 0.5, -0.5)
        axis.set_xlabel("prediction window")
        axis.set_ylabel("reference window")
        axis.set_title(
            f"{doc['language']} | {doc['fname'][:38]}\n"
            f"docNLS={doc['doc_nls']:.2f}  ref-cov={desc['reference_coverage']*100:.0f}%  "
            f"pred-cov={desc['prediction_coverage']*100:.0f}%",
            fontsize=11)
    for extra in range(count, len(flat)):
        flat[extra].set_axis_off()
    fig.suptitle("Stuck-repeating, seen in the recognized lines: lines pile onto a narrow "
                 "reference band (y) across the full prediction width (x)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# CSV exports
# --------------------------------------------------------------------------- #
def export_per_language_csv(documents, order, out_path) -> Path:
    headers = ["language", "script_family", "n_docs", "n_scored", "mean_docNLS",
               "good", "substitution", "missing", "repetition",
               "correct_coverage", "hallucination", "self_repetition", "mean_weighted_nls",
               "stuck_repeating_count"]
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for language in order:
            docs = [d for d in documents if d["language"] == language]
            scored = [d for d in docs if d["scored"]]
            decomposed = [decompose_reference(d) for d in scored]
            decomposed = [d for d in decomposed if d]
            family = docs[0]["script_family"] if docs else "Unknown"

            def dmean(mode):
                return mean_or_nan([d[mode] for d in decomposed]) if decomposed else float("nan")
            row = [
                language, family, len(docs), len(scored),
                round(mean_doc_nls(documents, language), 4),
                round(dmean("good"), 4), round(dmean("substitution"), 4),
                round(dmean("missing"), 4), round(dmean("repetition"), 4),
                round(mean_or_nan(values(documents, language, "correct", scored_only=True)), 4),
                round(mean_or_nan(values(documents, language, "hallucination", scored_only=True)), 4),
                round(mean_or_nan(values(documents, language, "self_repetition")), 4),
                round(mean_or_nan(values(documents, language, "weighted_nls", scored_only=True)), 4),
                sum(1 for d in docs if d["stuck"]),
            ]
            writer.writerow(row)
    return out_path


def export_stuck_documents_csv(documents, descriptors, out_path) -> Path:
    """List every stuck-repeating doc with line metrics + recognized-line descriptors."""
    headers = ["language", "script_family", "fname", "docNLS", "weighted_nls",
               "correct", "missing", "repetition_on_reference", "hallucination",
               "self_repetition", "n_recognized_lines", "reference_coverage",
               "prediction_coverage", "mean_recover_depth", "max_recover_depth", "document_type"]
    flagged = sorted((d for d in documents if d["stuck"]),
                     key=lambda d: (-(d["rep_on_ref"] or 0.0), d["fname"]))
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for doc in flagged:
            desc = descriptors.get(doc["fname"], {})
            writer.writerow([
                doc["language"], doc["script_family"], doc["fname"],
                round(doc["doc_nls"], 4) if doc["doc_nls"] is not None else "",
                round(doc["weighted_nls"], 4) if doc["weighted_nls"] is not None else "",
                round(doc["correct"], 4), round(doc["missing"], 4),
                round(doc["rep_on_ref"], 4), round(doc["hallucination"], 4),
                round(doc["self_repetition"], 4) if doc["self_repetition"] is not None else "",
                desc.get("n_lines", ""),
                round(desc["reference_coverage"], 4) if "reference_coverage" in desc else "",
                round(desc["prediction_coverage"], 4) if "prediction_coverage" in desc else "",
                round(desc["mean_recover_depth"], 2) if "mean_recover_depth" in desc else "",
                desc.get("max_recover_depth", ""),
                doc["document_type"],
            ])
    return out_path


def export_worst_documents_csv(documents, *, worst_k, out_path) -> Path:
    headers = ["language", "script_family", "fname", "docNLS", "weighted_nls",
               "correct", "missing", "repetition_on_reference", "hallucination",
               "self_repetition", "document_type"]
    by_language = defaultdict(list)
    for doc in documents:
        if doc["doc_nls"] is not None:
            by_language[doc["language"]].append(doc)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for language in sorted(by_language):
            ranked = sorted(by_language[language], key=lambda d: (d["doc_nls"], d["fname"]))
            for doc in ranked[:worst_k]:
                writer.writerow([
                    language, doc["script_family"], doc["fname"],
                    round(doc["doc_nls"], 4),
                    "" if doc["weighted_nls"] is None else round(doc["weighted_nls"], 4),
                    "" if doc["correct"] is None else round(doc["correct"], 4),
                    "" if doc["missing"] is None else round(doc["missing"], 4),
                    "" if doc["rep_on_ref"] is None else round(doc["rep_on_ref"], 4),
                    "" if doc["hallucination"] is None else round(doc["hallucination"], 4),
                    "" if doc["self_repetition"] is None else round(doc["self_repetition"], 4),
                    doc["document_type"],
                ])
    return out_path


def write_plots_explained_md(out_path, *, counts, stuck_count, stuck_params, cutoff) -> Path:
    """Write an in-depth, per-PNG reading guide into the output directory."""
    rep_min, correct_min, depth_min = stuck_params
    text = f"""# How to read these plots

All figures here come from `scratch_tools/ocr_failure_decomposition.py`. **Everything is
computed from the pipeline's recognized-line metrics** (the per-document CSV) plus the model's
own per-document outputs. We deliberately **never scan the raw score matrices for the best
match per row/column**: that would ignore line geometry and destroy the repetition signal,
which is exactly what the Hough line-detection pipeline exists to handle. The one place a raw
matrix appears is as a *faint background* under the pipeline's own recognized lines in the
stuck-repeating exemplars — display only, not measurement.

## Definitions used throughout
- **docNLS** (`document_normalised_levenshtein`): raw similarity of the model's full
  transcription to the reference, 0–1. The headline "quality". "Ordered by quality" = sorted
  by each language's mean docNLS.
- **Reference coverage partition** (exact, sums to 1, from recognized lines):
  `correct_ref_coverage` (covered once) + `missing_ref_coverage` (never covered) +
  `repetition_on_reference` (covered more than once) = 1.
- **weighted_along_lines_nls**: character similarity measured *along the recognized lines* —
  i.e. how correct the text is where a line was found.
- **hallucination**: fraction of the prediction covered by no line (text not on the page).
- **model self-repetition** (`repetition` from outputs.json): the model's own decoding-loop
  metric (prediction repeats itself).

## This run
- documents with a prediction (universe): **{counts['included']}** (excluded
  {counts['excluded_no_prediction']} with no prediction).
- scored documents (have recognized lines): **{counts['scored']}**.
- stuck-repeating documents flagged: **{stuck_count}**.

---

## metric_distributions_{{violin,boxpoints}}_{{by_script,by_quality}}.png
One panel per metric; each language summarised across its documents.
- **violin** = smoothed density (wide = many docs at that value). Two bulges = **bimodal** =
  two populations (a "good" cluster and a "failure" cluster) — the mean is then misleading.
- **boxpoints** = median/IQR box with every document as a dot (honest about sample size and
  outliers).
- `by_quality` orders languages best→worst docNLS; `by_script` groups them by writing system
  (shaded bands, bold family headers). docNLS / self-repetition panels use all docs with a
  prediction; the rest use scored docs only (noted in each panel title).
- **Question answered:** is a language uniformly mediocre, or excellent-with-some-disasters?

## reference_coverage_partition_{{by_script,by_quality}}.png
Per-language stacked bar of `correct / missing / repetition_on_reference` — **exact, sums to 1**.
- Tall grey = most of the reference transcribed once (good). Large orange = omission. Red =
  reference covered multiple times (repetition).
- **Question answered:** of the reference text, how much was covered once / skipped / repeated?
- Caveat: only scored docs appear; a language with no scored docs shows no bar.

## reference_error_decomposition_{{by_script,by_quality}}.png
The *defensible* "why is it wrong" view. Top panel splits the reference (sums to 1) into:
- **good** = `correct × weighted_nls` (covered and characters right),
- **substitution** = `correct × (1 − weighted_nls)` (covered but wrong glyphs),
- **missing** (omission), **repetition** (covered more than once).
Bottom panel = **companion signals on a separate axis** (NOT part of the sum): hallucination
and model self-repetition. These live on the prediction/model side, so mixing them into the
reference partition would double-count — they are shown separately on purpose.
- **Question answered:** is the error from skipping text, repeating text, wrong characters, or
  inventing text?
- Caveat: the good/substitution split applies the along-line character quality to the
  correctly-covered fraction — an approximation, but from pipeline outputs only.

## catastrophic_failure_subset_by_quality.png
Per language, % of documents with docNLS < {cutoff:g} (count annotated). Surfaces **bimodal**
languages whose mean is dragged down by a few failing pages (e.g. a language that is mostly
excellent but has a handful of disasters).
- **Question answered:** is "this language is bad" really "a few specific pages are bad"?

## stuck_repeating_signature.png
**Stuck-repeating** = the model loops on one *small* part of the reference. The faithful
signal is **re-cover depth** — how many times the recognized lines cover the same reference
window — NOT breadth (how *much* is missing/repeated). A small region repeated many times has
low breadth (small `repetition_on_reference`/`missing`, and possibly high hallucination) yet
very high depth, so breadth gates miss it (e.g. a page covered 63% but with one region hit 82×).
Detection is two stages: (1) a cheap CSV pre-screen — `repetition_on_reference > {rep_min:g}`
and `correct > {correct_min:g}` — to limit how many per-doc pickles we open; (2) flag when the
recognized-line **max re-cover depth ≥ {depth_min}**.
This scatter shows the candidates: x = `repetition_on_reference` (breadth), y = **max re-cover
depth** (log; the real signal), colour = hallucination (shown for context, does NOT gate).
Points above the dashed line are flagged and labelled.
- **Question answered:** which documents loop on a small part of the reference?

## stuck_repeating_lines_exemplars.png
The same phenomenon seen **directly in the pipeline's recognized lines**. Faint grey = score
matrix (reference rows × prediction columns); red segments = the recognized lines. In a
stuck-repeating doc the lines **pile onto a narrow band of the reference height (y)**, re-covering
it many times, while spanning much of the prediction width (x). Each title reports
reference-coverage %, prediction-coverage %, and mean re-cover depth — all from the recognized
lines only. Exemplars are ordered by deepest re-cover (pinned `--stuck-example` docs first).
- **Question answered:** what does "stuck repeating" look like geometrically, in confirmed lines?

## CSV tables
- `per_language_decomposition.csv` — every per-language mean behind the figures + a
  `stuck_repeating_count`.
- `worst_documents_per_language.csv` — the lowest-docNLS documents per language with metrics.
- `stuck_repeating_documents.csv` — every flagged doc with line metrics and recognized-line
  descriptors (reference/prediction coverage, re-cover depth) for direct inspection.
"""
    with open(out_path, "w") as handle:
        handle.write(text)
    return out_path


# --------------------------------------------------------------------------- #
# Console tables
# --------------------------------------------------------------------------- #
def print_tables(documents, counts, order) -> None:
    print("=== universe accounting (line-based) ===")
    print(f"join keys (CSV ∪ outputs)     : {counts['all']}")
    print(f"excluded (no prediction)      : {counts['excluded_no_prediction']}")
    print(f"included (universe)           : {counts['included']}")
    print(f"  scored (have recognized lines): {counts['scored']}")
    print()
    print("=== reference decomposition per language (means; ordered by quality) ===")
    print(f'{"lang":12s} {"n":>4s} {"docNLS":>7s} {"good":>5s} {"subst":>5s} {"miss":>5s} '
          f'{"rep":>5s} {"hall":>5s} {"selfRep":>7s}')
    for language in order:
        scored = [d for d in documents if d["language"] == language and d["scored"]]
        parts = [decompose_reference(d) for d in scored]
        parts = [p for p in parts if p]

        def dmean(mode):
            return mean_or_nan([p[mode] for p in parts]) if parts else float("nan")

        def f(value):
            return f"{value:5.3f}" if not math.isnan(value) else "  nan"
        n = len([d for d in documents if d["language"] == language])
        print(f"{language:12s} {n:4d} {mean_doc_nls(documents, language):7.3f} "
              f"{f(dmean('good'))} {f(dmean('substitution'))} {f(dmean('missing'))} "
              f"{f(dmean('repetition'))} {f(mean_or_nan(values(documents, language, 'hallucination', scored_only=True)))} "
              f"{mean_or_nan(values(documents, language, 'self_repetition')):7.3f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Line-based OCR error decomposition per language.")
    parser.add_argument("--outputs-json", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Balanced run dir with best_combination_per_document.csv.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write outputs (default: <results-dir>/analysis_plots).")
    parser.add_argument("--failure-cutoff", type=float, default=FAILURE_DOC_NLS_CUTOFF)
    parser.add_argument("--worst-k", type=int, default=WORST_DOCS_PER_LANGUAGE)
    parser.add_argument("--stuck-rep-min", type=float, default=STUCK_REP_MIN,
                        help="Pre-screen: min repetition_on_reference to open a doc's pickle.")
    parser.add_argument("--stuck-correct-min", type=float, default=STUCK_CORRECT_MIN,
                        help="Pre-screen: min correct coverage.")
    parser.add_argument("--stuck-depth-min", type=int, default=STUCK_DEPTH_MIN,
                        help="Flag stuck when recognized-line max re-cover depth >= this.")
    parser.add_argument("--stuck-example", action="append", dest="stuck_examples", default=None,
                        help="Pin a document basename as a stuck-repeating exemplar (repeatable).")
    parser.add_argument("--stuck-exemplar-count", type=int, default=STUCK_EXEMPLAR_COUNT,
                        help="How many stuck-repeating docs to show in the recognized-line panel.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir / "analysis_plots")
    outputs = load_outputs_json(args.outputs_json)
    partition = load_balanced_partition(args.results_dir)
    documents, counts = build_documents(outputs, partition)

    # Stage 1 (cheap CSV pre-screen) -> Stage 2 (recognized-line re-cover depth from pickles).
    stuck_params = (args.stuck_rep_min, args.stuck_correct_min, args.stuck_depth_min)
    candidates = [d for d in documents
                  if is_stuck_candidate(d, rep_min=args.stuck_rep_min, correct_min=args.stuck_correct_min)]
    descriptors: dict[str, dict] = {}
    for doc in candidates:
        if doc["pkl_path"]:
            desc = recognized_line_descriptors(doc["pkl_path"])  # scalars only (low memory)
            if desc is not None:
                descriptors[doc["fname"]] = desc
    for doc in documents:
        desc = descriptors.get(doc["fname"])
        doc["stuck"] = bool(desc and desc["max_recover_depth"] >= args.stuck_depth_min)
    stuck_count = sum(1 for d in documents if d["stuck"])

    orderings = language_orderings(documents)
    print_tables(documents, counts, orderings["by_quality"])
    print(f"\nstuck-repeating: {len(candidates)} candidates (rep>{args.stuck_rep_min:g}) -> "
          f"{stuck_count} flagged (recognized-line max re-cover depth >= {args.stuck_depth_min})")

    if args.no_plots:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for order_name, order in orderings.items():
        by_script = order_name == "by_script"
        for style in ("violin", "boxpoints"):
            written.append(figure_metric_distributions(
                documents, order, by_script=by_script, style=style,
                out_path=output_dir / f"metric_distributions_{style}_{order_name}.png"))
        written.append(figure_coverage_partition(
            documents, order, by_script=by_script,
            out_path=output_dir / f"reference_coverage_partition_{order_name}.png"))
        written.append(figure_reference_decomposition(
            documents, order, by_script=by_script,
            out_path=output_dir / f"reference_error_decomposition_{order_name}.png"))
    written.append(figure_failure_subset(
        documents, orderings["by_quality"], cutoff=args.failure_cutoff,
        out_path=output_dir / "catastrophic_failure_subset_by_quality.png"))
    written.append(figure_stuck_signature(
        documents, descriptors, depth_min=args.stuck_depth_min, rep_min=args.stuck_rep_min,
        out_path=output_dir / "stuck_repeating_signature.png"))

    # Exemplars: pinned --stuck-example first, then flagged by deepest re-cover (deterministic).
    flagged = [d for d in documents if d["stuck"]]
    pinned = [basename(name) for name in (args.stuck_examples or [])]
    ranked = sorted(flagged, key=lambda d: (-descriptors[d["fname"]]["max_recover_depth"], d["fname"]))
    ordered_fnames = [f for f in pinned if f in descriptors] + \
                     [d["fname"] for d in ranked if d["fname"] not in pinned]
    by_fname = {d["fname"]: d for d in documents}
    # Reload payload (matrix + lines) only for the exemplars actually drawn.
    exemplars = []
    for f in ordered_fnames[:args.stuck_exemplar_count]:
        doc = by_fname.get(f)
        if doc is None or not doc["pkl_path"]:
            continue
        full = recognized_line_descriptors(doc["pkl_path"], keep_payload=True)
        if full is not None:
            exemplars.append((doc, full))
    exemplar_path = figure_stuck_line_exemplars(
        exemplars, out_path=output_dir / "stuck_repeating_lines_exemplars.png")
    if exemplar_path is not None:
        written.append(exemplar_path)

    csv_paths = [
        export_per_language_csv(documents, orderings["by_quality"],
                                output_dir / "per_language_decomposition.csv"),
        export_worst_documents_csv(documents, worst_k=args.worst_k,
                                   out_path=output_dir / "worst_documents_per_language.csv"),
        export_stuck_documents_csv(documents, descriptors,
                                   output_dir / "stuck_repeating_documents.csv"),
    ]
    md_path = write_plots_explained_md(
        output_dir / "PLOTS_EXPLAINED.md", counts=counts, stuck_count=stuck_count,
        stuck_params=stuck_params, cutoff=args.failure_cutoff)

    print()
    print("wrote figures:")
    for path in written:
        print(" ", path)
    print("wrote tables:")
    for path in csv_paths:
        print(" ", path)
    print("wrote doc:", md_path)


if __name__ == "__main__":
    main()
