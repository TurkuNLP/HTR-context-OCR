"""Pass 2 — item counting: articles / advertisements / register entries (category-gated).

Runs ONLY for item-gated categories (newspaper, periodical, register — the runner gates; the
question "how many articles" is ill-posed for a book and is never asked of one).

Definitions implement the locked section-layer decisions (IMPLEMENTATION_PLAN.md §8, #13):
- an item is counted at its own heading or dash-leader start (items, not rubric headings);
- continuation tails from a previous page (no marker on this page) are not counted;
- the feuilleton/insert installment is the content of its own part, not an article;
- advertisements are counted separately from articles;
- <= ITEM_ENUM_MAX items: enumerate each with a short anchor; more: SAMPLE one column carefully
  and report the per-column count — the HARNESS multiplies (the model never does arithmetic).

Schema notes: every item group is always present with a ``mode`` discriminator; unused branches
carry empty arrays / zero-valued sample objects (grammar-safe: no object-or-null unions).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow flat imports under sbatch
import config  # noqa: E402
from client import image_part, text_part  # noqa: E402

# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = f"""\
You are an expert analyst of historical document pages. Your ONLY task is to count the ITEMS in \
each given part of the page, for the item kinds you are asked about. You will be told the parts; \
do not redefine them. Do not transcribe the page.

Definitions:
- An ARTICLE is an editorial content unit that begins on this page at its own heading or at a \
dash-leader mark at a paragraph start. Rubric headings that merely group several notices are not \
themselves articles; count the individual items under them. Text that merely finishes an item \
begun on a previous page (no heading or dash on this page) is not counted. A serialized story or \
book-format insert is the content of its own part, not an article of the news part.
- An ADVERTISEMENT is a paid notice or display advertisement: typically boxed, or set off with \
its own typography, or one entry in a classified/notice field.
- An ENTRY is one record of a register, directory, index or list (one person, one property, one \
transaction).

Counting procedure per part and item kind:
- If there are at most {config.ITEM_ENUM_MAX} items: use mode "enumerate" and list every item \
with a two-to-four word anchor (its heading or first words). Form the list item by item; the \
list is the count.
- If there may be more than {config.ITEM_ENUM_MAX}: use mode "sample" immediately. Choose ONE \
representative column of the part, count its items carefully one by one, and report: \
sampled_column (its position counting from the left, 1-based), items_in_column (your careful \
count in that one column), and columns_with_items (how many columns of this part contain items \
of this kind). Do NOT multiply these numbers yourself and do not report a total. Long newspaper \
advertisement or notice fields should almost always use sample, not enumerate.
- If the part contains no items of the kind: mode "none_present".
- If you were not asked about a kind for this part: mode "not_applicable".

When mode is not "enumerate", leave items as an empty list. When mode is not "sample", set every \
sample number to 0. Keep anchors short; the JSON object is the answer, with no explanatory prose.

Return only the JSON object."""

# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------
# One reusable group shape for articles / advertisements / entries. ``mode`` is the discriminator;
# unused branches are empty containers, never nulls (keeps the guided-decoding grammar simple).
_ITEM_GROUP: dict = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["enumerate", "sample", "none_present", "not_applicable"]},
        "items": {
            "type": "array",
            "minItems": 0,
            "maxItems": config.ITEM_ENUM_MAX,
            "items": {
                "type": "object",
                "properties": {"anchor": {"type": "string", "maxLength": 40}},
                "required": ["anchor"],
                "additionalProperties": False,
            },
        },
        "sample": {
            "type": "object",
            "properties": {
                "sampled_column": {"type": "integer", "minimum": 0, "maximum": 20},
                "items_in_column": {"type": "integer", "minimum": 0, "maximum": 200},
                "columns_with_items": {"type": "integer", "minimum": 0, "maximum": 20},
            },
            "required": ["sampled_column", "items_in_column", "columns_with_items"],
            "additionalProperties": False,
        },
    },
    "required": ["mode", "items", "sample"],
    "additionalProperties": False,
}

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "part_index": {"type": "integer", "minimum": 1},
                    "articles": _ITEM_GROUP,
                    "advertisements": _ITEM_GROUP,
                    "entries": _ITEM_GROUP,
                },
                "required": ["part_index", "articles", "advertisements", "entries"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["parts"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------------------
# User turn + postprocess
# --------------------------------------------------------------------------------------
def groups_for_category(category: str) -> tuple[str, ...]:
    """Which item kinds are well-posed questions for this category (runner-level gating)."""
    if category in ("newspaper", "periodical"):
        return ("articles", "advertisements")
    if category == "register":
        return ("entries",)
    return ()


def _describe_parts(parts: list[dict], columns_per_part: dict[int, int]) -> str:
    """Inject the shared frame (parts) plus pass 1's column counts as sampling context."""
    lines = [f"This page has {len(parts)} part(s):"]
    for part in parts:
        n_cols = columns_per_part.get(part["part_index"])
        col_note = f"; it has {n_cols} columns" if n_cols else ""
        lines.append(
            f"  part {part['part_index']}: vertical range {part['top_frac']:.2f}-{part['bottom_frac']:.2f}; "
            f"{part['anchor'] or 'unnamed'}; content: {part['content_class']}{col_note}."
        )
    return "\n".join(lines)


def build_user_parts(
    full_data_url: str,
    parts: list[dict],
    requested_groups: tuple[str, ...],
    columns_per_part: dict[int, int],
) -> list[dict]:
    """User turn = parts frame + which item kinds to count + the full page image.

    Pass 2 sees the full page only: items are large-scale marks (headings, boxes, rules), so it
    needs global coverage, not gutter-level magnification.
    """
    asked = ", ".join(requested_groups)
    preamble = (
        _describe_parts(parts, columns_per_part)
        + f"\n\nCount these item kinds in every part: {asked}. "
        "Mark all other kinds as not_applicable.\n\nImage 1: the full page."
    )
    return [text_part(preamble), image_part(full_data_url)]


def postprocess(parsed: dict) -> dict:
    """Light normalization: coerce shapes, order parts, trim anchors."""
    raw_parts = parsed.get("parts")
    if not isinstance(raw_parts, list):
        raise ValueError("pass2 payload missing required parts list")
    parts = [p for p in raw_parts if isinstance(p, dict)]
    if len(parts) != len(raw_parts):
        raise ValueError("pass2 payload contains non-object part entries")
    for part in parts:
        for group_name in ("articles", "advertisements", "entries"):
            group = part.get(group_name)
            if not isinstance(group, dict):  # defensive; schema should prevent this
                part[group_name] = {"mode": "not_applicable", "items": [], "sample": {}}
                continue
            group["items"] = [
                {"anchor": str(item.get("anchor", ""))[:40]}
                for item in (group.get("items") or [])
                if isinstance(item, dict)
            ]
    parts.sort(key=lambda p: int(p.get("part_index", 0)))
    return {"parts": parts}
