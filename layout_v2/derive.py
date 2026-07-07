"""Harness derivations: every count, verdict and flag computed FROM the model's primitives.

The primitives-only principle (CCM §8.2) implemented: the model emits enumerations and route
results; this module computes
- per-part ``column_count`` = len(columns), with the O9 reconciliation verdict against the
  width-arithmetic and returns routes;
- page-level conventions (``column_count_dominant`` = the tallest part's count);
- item counts (enumerated lengths, or the sampling multiplication the model was forbidden to do);
- prior-violation and vertical-script flags;
- ``needs_review`` — the single, computed uncertainty channel (no confidence fields exist).

Everything here is deterministic arithmetic; there is deliberately no model access.
"""

from __future__ import annotations

import config
from validate import sample_vs_columns_issue


# --------------------------------------------------------------------------------------
# Column reconciliation (O9)
# --------------------------------------------------------------------------------------
def reconcile_part_columns(part1: dict) -> dict:
    """Derive one part's column count and the agreement verdict across the emitted routes.

    Routes: enumeration (primary, len(columns)), width arithmetic (implied_count), returns
    (implied_count where a stream exists). Verdict: ``agree`` (all present routes equal),
    ``minor_disagree`` (max spread 1), ``disagree`` (spread > 1), ``single_route`` (nothing to
    compare against).
    """
    enumerated = len(part1.get("columns") or [])
    routes: dict[str, int] = {"enumeration": enumerated}

    width_implied = (part1.get("width_check") or {}).get("implied_count")
    if isinstance(width_implied, int) and width_implied > 0:
        routes["width"] = width_implied

    stream = part1.get("stream") or {}
    stream_implied = stream.get("implied_count")
    if stream.get("exists") is True and isinstance(stream_implied, int) and stream_implied > 0:
        routes["returns"] = stream_implied

    values = list(routes.values())
    spread = max(values) - min(values) if values else 0
    if len(routes) == 1:
        verdict = "single_route"
    elif spread == 0:
        verdict = "agree"
    elif spread == 1:
        verdict = "minor_disagree"
    else:
        verdict = "disagree"

    return {
        "part_index": part1.get("part_index"),
        "column_count": enumerated,  # enumeration is the primary route by design (O3)
        "routes": routes,
        "verdict": verdict,
        "counting_band": part1.get("counting_band", ""),
        "stream_exists": stream.get("exists"),
        "stream_reason": stream.get("reason", ""),
        "spanning_edges": part1.get("spanning_edges", ""),
        "second_band_alignment": part1.get("second_band_alignment", ""),
        "parts_disputed": bool(part1.get("parts_disputed", False)),
    }


# --------------------------------------------------------------------------------------
# Item derivation (pass 2)
# --------------------------------------------------------------------------------------
def derive_item_group(group: dict, n_columns: int | None) -> dict:
    """Derive one item-kind's count for one part.

    Enumerate mode: count = len(items) (exact). Sample mode: the harness multiplication
    ``items_in_column x columns_with_items`` (estimate; the model never multiplies), sanity-checked
    against pass 1's column count. Other modes: count 0 / not applicable.
    """
    mode = group.get("mode", "not_applicable")
    if mode == "enumerate":
        return {"count": len(group.get("items") or []), "is_estimate": False, "mode": mode, "issue": ""}
    if mode == "sample":
        sample = group.get("sample") or {}
        per_column = int(sample.get("items_in_column") or 0)
        n_cols_claimed = int(sample.get("columns_with_items") or 0)
        issue = sample_vs_columns_issue(sample, n_columns)
        if issue and n_columns:  # clamp an impossible claim to the validated column count
            n_cols_claimed = min(n_cols_claimed, n_columns)
        return {
            "count": per_column * n_cols_claimed,
            "is_estimate": True,
            "mode": mode,
            "issue": issue,
        }
    if mode == "none_present":
        return {"count": 0, "is_estimate": False, "mode": mode, "issue": ""}
    return {"count": None, "is_estimate": False, "mode": mode, "issue": ""}  # not_applicable


# --------------------------------------------------------------------------------------
# Review heuristics
# --------------------------------------------------------------------------------------
def _is_high_column_mixed_mosaic(category: str, dominant: object, totals: dict[str, dict]) -> bool:
    """Flag dense newspaper fronts where a clean integer is usually the risky outcome."""
    if category != "newspaper" or not isinstance(dominant, int) or dominant < 8:
        return False
    articles = totals.get("articles", {})
    ads = totals.get("advertisements", {})
    article_count = articles.get("count")
    ad_count = ads.get("count")
    if not isinstance(article_count, int) or not isinstance(ad_count, int):
        return False
    if article_count <= 0 or ad_count <= 0:
        return False
    return bool(articles.get("is_estimate") or ads.get("is_estimate") or article_count + ad_count >= 20)


def _running_text_marked_independent(category: str, part0: dict, summary: dict) -> bool:
    """Review-only prior: a regular newspaper text part should not silently become an item field."""
    if category != "newspaper":
        return False
    if part0.get("content_class") != "running_text":
        return False
    if summary.get("stream_reason") != "independent_items":
        return False
    count = summary.get("column_count")
    return isinstance(count, int) and count >= 4


def _can_infer_running_text_stream(
    category: str, part0: dict, summary: dict, totals: dict[str, dict]
) -> bool:
    """Infer stream only after item counts show text dominates over advertisements."""
    if not _running_text_marked_independent(category, part0, summary):
        return False
    articles = totals.get("articles", {})
    ads = totals.get("advertisements", {})
    article_count = articles.get("count")
    ad_count = ads.get("count")
    if not isinstance(article_count, int) or article_count <= 0:
        return False
    if not isinstance(ad_count, int):
        return False
    return ad_count <= max(1, article_count // 4)


# --------------------------------------------------------------------------------------
# Whole-document derivation
# --------------------------------------------------------------------------------------
def derive_document(
    *,
    category: str,
    pass0_parts: list[dict],
    pass1: dict | None,
    pass2: dict | None,
    validation_issues: list[str],
) -> dict:
    """Assemble every derived quantity and flag for one document.

    Inputs are the postprocessed pass outputs (or None where a pass failed / was gated off);
    output is the ``derived`` block of the per-document record.
    """
    derived: dict = {
        "document_category": category,
        "independent_parts": len(pass0_parts),  # = len(parts list); never asked of the model
        "vertical_script": any(p.get("writing_direction") == "vertical" for p in pass0_parts),
        "parts": [],
        "column_count_dominant": None,
        "article_count": None,
        "article_count_is_estimate": False,
        "advertisement_count": None,
        "advertisement_count_is_estimate": False,
        "entry_count": None,
        "entry_count_is_estimate": False,
        "prior_violation": "",
        "needs_review": False,
        "needs_review_reasons": [],
    }
    reasons: list[str] = list(validation_issues)  # validator issues feed straight into review

    # ---- Columns (pass 1) -------------------------------------------------------------
    columns_by_part: dict[int, int] = {}
    if pass1 is not None:
        part_summaries = [reconcile_part_columns(p) for p in (pass1.get("parts") or [])]
        derived["parts"] = part_summaries
        pass0_by_index = {p.get("part_index"): p for p in pass0_parts}
        for summary in part_summaries:
            if isinstance(summary["part_index"], int):
                columns_by_part[summary["part_index"]] = summary["column_count"]
            if summary["verdict"] == "disagree":
                reasons.append(f"part {summary['part_index']}: column routes disagree {summary['routes']}")
            if summary["second_band_alignment"] == "misaligned":
                reasons.append(f"part {summary['part_index']}: second band misaligned")
            if summary["parts_disputed"]:
                reasons.append(f"part {summary['part_index']}: model disputed the injected parts")
            if _running_text_marked_independent(
                category, pass0_by_index.get(summary["part_index"], {}), summary
            ):
                reasons.append(f"part {summary['part_index']}: running_text part marked independent_items")

        # Page-level convention: the DOMINANT part's count (tallest by vertical extent). Computed,
        # never asked — a page with parts has no single column count (CCM Definition 7).
        heights = {
            p["part_index"]: (p.get("bottom_frac", 0) - p.get("top_frac", 0)) for p in pass0_parts
        }
        if columns_by_part:
            dominant_index = max(columns_by_part, key=lambda i: heights.get(i, 0))
            derived["column_count_dominant"] = columns_by_part[dominant_index]

        # Prior rail (plan §9.2): flags only — the prior never overrides the enumeration.
        prior_max = config.CATEGORY_COLUMN_PRIOR_MAX.get(category)
        dominant = derived["column_count_dominant"]
        if prior_max is not None and isinstance(dominant, int) and dominant > prior_max:
            derived["prior_violation"] = f"{category} with {dominant} columns (prior max {prior_max})"
            reasons.append(derived["prior_violation"])

    # ---- Items (pass 2) ---------------------------------------------------------------
    if pass2 is not None:
        totals: dict[str, dict] = {}
        for group_name in ("articles", "advertisements", "entries"):
            total = 0
            any_applicable = False
            any_estimate = False
            for part in pass2.get("parts") or []:
                part_index = part.get("part_index")
                group = part.get(group_name) or {}
                result = derive_item_group(group, columns_by_part.get(part_index))
                if result["issue"]:
                    reasons.append(f"part {part_index} {group_name}: {result['issue']}")
                if result["count"] is None:
                    continue  # not_applicable for this part
                any_applicable = True
                any_estimate = any_estimate or result["is_estimate"]
                total += result["count"]
            totals[group_name] = {
                "count": total if any_applicable else None,
                "is_estimate": any_estimate,
            }
        derived["article_count"] = totals["articles"]["count"]
        derived["article_count_is_estimate"] = totals["articles"]["is_estimate"]
        derived["advertisement_count"] = totals["advertisements"]["count"]
        derived["advertisement_count_is_estimate"] = totals["advertisements"]["is_estimate"]
        derived["entry_count"] = totals["entries"]["count"]
        derived["entry_count_is_estimate"] = totals["entries"]["is_estimate"]

        if _is_high_column_mixed_mosaic(category, derived["column_count_dominant"], totals):
            reasons.append("high-column mixed item mosaic")

        for summary in derived["parts"]:
            part0 = pass0_by_index.get(summary.get("part_index"), {}) if pass1 is not None else {}
            if _can_infer_running_text_stream(category, part0, summary, totals):
                summary["stream_exists"] = True
                summary["stream_reason"] = "inferred_continuous_stream"
                summary["stream_inferred"] = True
                reasons.append(f"part {summary['part_index']}: inferred running_text stream")

    # ---- The single uncertainty channel ------------------------------------------------
    derived["needs_review_reasons"] = reasons
    derived["needs_review"] = bool(reasons)
    return derived
