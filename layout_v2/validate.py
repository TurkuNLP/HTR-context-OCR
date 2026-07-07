"""Mechanical validators over the parsed pass outputs (plan §9.1).

Validators never crash a document and never fix data: each returns a list of issue strings
(empty = clean). The harness attaches the issues to the record and folds them into
``needs_review``; downstream analysis decides what to do with flagged documents.

Everything here is intentionally simple arithmetic on the model's own emissions — the point of
the anchored-enumeration output contract is precisely that these checks become possible.
"""

from __future__ import annotations

import re

import config


def _meaningful_anchor(value: object) -> str:
    """Normalize anchors for duplicate checks; punctuation-only marks are not evidence."""
    anchor = str(value or "").strip().lower()
    return anchor if re.search(r"[\w\d]", anchor, flags=re.UNICODE) else ""


# --------------------------------------------------------------------------------------
# Pass 1 (columns)
# --------------------------------------------------------------------------------------
def validate_pass1_part(part: dict) -> list[str]:
    """Check one pass-1 part: monotone anchors, plausible gaps, tiling sanity, route coherence."""
    issues: list[str] = []
    index = part.get("part_index", "?")
    columns = part.get("columns") or []

    # O3 self-audit, mechanized: x-centers must be strictly increasing left-to-right.
    # Variant-aware: in the text_only bake-off arm no column carries x_center_frac at all —
    # the geometric checks are then skipped (that arm deliberately trades them away). A MIX of
    # with/without-x entries is still an error in every arm.
    with_x = [c for c in columns if isinstance(c.get("x_center_frac"), (int, float))]
    if with_x and len(with_x) != len(columns):
        issues.append(f"part {index}: some column entries missing x_center_frac")
    xs = [c["x_center_frac"] for c in with_x]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        issues.append(f"part {index}: column x-centers not strictly increasing")
    # Two "columns" closer than the minimum gap are almost certainly one column double-counted.
    if any((b - a) < config.MIN_COL_GAP_FRAC for a, b in zip(xs, xs[1:])):
        issues.append(f"part {index}: adjacent column centers closer than {config.MIN_COL_GAP_FRAC}")

    duplicate_anchor_text = False
    anchors = [_meaningful_anchor(c.get("anchor_text", "")) for c in columns]
    non_empty = [a for a in anchors if a]
    if len(non_empty) != len(set(non_empty)):
        duplicate_anchor_text = True

    # O7 tiling sanity: n_cols * unit width should roughly reproduce the block width.
    width_check = part.get("width_check") or {}
    unit = width_check.get("unit_width_frac")
    block = width_check.get("block_width_frac")
    if isinstance(unit, (int, float)) and isinstance(block, (int, float)) and unit > 0 and block > 0 and columns:
        expected = unit * len(columns)
        if abs(expected - block) / block > config.TILING_TOLERANCE:
            issues.append(
                f"part {index}: tiling mismatch (unit {unit:.3f} x {len(columns)} cols vs block {block:.3f})"
            )

    # O8 internal coherence: a stream implies returns; no stream implies no returns.
    stream = part.get("stream") or {}
    if stream.get("exists") is True and not isinstance(stream.get("returns"), int):
        issues.append(f"part {index}: stream.exists but returns is null")
    if stream.get("exists") is False and isinstance(stream.get("returns"), int):
        issues.append(f"part {index}: no stream but returns given")
    if isinstance(stream.get("returns"), int) and isinstance(stream.get("implied_count"), int):
        if stream["implied_count"] != stream["returns"] + 1:
            issues.append(f"part {index}: stream implied_count != returns + 1")

    # Duplicate text alone is weak evidence: short headings can repeat across real columns.
    # Treat it as review-worthy when another geometric/route check already looks suspicious.
    if duplicate_anchor_text and issues:
        issues.append(f"part {index}: duplicate anchor_text values")

    return issues


def validate_pass1(pass1: dict, pass0_parts: list[dict]) -> list[str]:
    """Whole-output checks for pass 1: per-part checks + cross-pass part consistency."""
    issues: list[str] = []
    parts1 = pass1.get("parts") or []
    indices0 = {p["part_index"] for p in pass0_parts}
    indices1 = {p.get("part_index") for p in parts1}
    # The injected frame must be answered exactly: same parts, no inventions, no omissions.
    if indices1 != indices0:
        issues.append(f"pass1 parts {sorted(indices1)} != pass0 parts {sorted(indices0)}")
    for part in parts1:
        issues.extend(validate_pass1_part(part))
    return issues


# --------------------------------------------------------------------------------------
# Pass 2 (items)
# --------------------------------------------------------------------------------------
def validate_pass2(pass2: dict, pass0_parts: list[dict], requested_groups: tuple[str, ...]) -> list[str]:
    """Check pass 2: part consistency, mode/branch coherence, anchor duplication."""
    issues: list[str] = []
    parts2 = pass2.get("parts") or []
    indices0 = {p["part_index"] for p in pass0_parts}
    indices2 = {p.get("part_index") for p in parts2}
    if indices2 - indices0:
        issues.append(f"pass2 invented parts {sorted(indices2 - indices0)}")
    for part in parts2:
        index = part.get("part_index", "?")
        for group_name in ("articles", "advertisements", "entries"):
            group = part.get(group_name) or {}
            mode = group.get("mode")
            items = group.get("items") or []
            sample = group.get("sample") or {}
            # A kind we did not ask about must be marked not_applicable (and vice versa).
            if group_name not in requested_groups and mode not in ("not_applicable", None):
                issues.append(f"part {index}: {group_name} answered but not requested (mode={mode})")
            if group_name in requested_groups and mode == "not_applicable":
                issues.append(f"part {index}: requested {group_name} marked not_applicable")
            # Mode/branch coherence: the discriminator decides which branch may carry data.
            if mode == "enumerate":
                if not items:
                    issues.append(f"part {index}: {group_name} mode=enumerate with empty items")
                anchors = [_meaningful_anchor(i.get("anchor", "")) for i in items]
                non_empty = [a for a in anchors if a]
                if len(non_empty) != len(set(non_empty)):
                    issues.append(f"part {index}: {group_name} duplicate item anchors")
            if mode == "sample":
                if not sample.get("items_in_column") or not sample.get("columns_with_items"):
                    issues.append(f"part {index}: {group_name} mode=sample with zero sample numbers")
            if mode in ("none_present", "not_applicable") and items:
                issues.append(f"part {index}: {group_name} mode={mode} but items listed")
    return issues


# --------------------------------------------------------------------------------------
# Cross-pass sanity used by derive.py
# --------------------------------------------------------------------------------------
def sample_vs_columns_issue(sample: dict, n_columns: int | None) -> str:
    """Sampling arithmetic sanity: columns_with_items cannot exceed the part's column count."""
    if not n_columns:
        return ""
    claimed = sample.get("columns_with_items")
    if isinstance(claimed, int) and claimed > n_columns:
        return f"sample columns_with_items {claimed} > column count {n_columns}"
    return ""
