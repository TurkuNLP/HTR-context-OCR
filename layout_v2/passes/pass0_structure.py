"""Pass 0 — document category + independent parts (the shared frame).

Cheapest pass; runs on every document; its output gates everything downstream:
- ``document_category`` decides whether pass 2 (items) is a well-posed question at all;
- ``parts`` scope pass 1's per-part column counting and pass 2's per-part item counting;
- ``writing_direction`` routes the vertical-CJK column rule (a vertical line is a LINE).

Design notes (IMPLEMENTATION_PLAN.md §6): closed category vocabulary; the parts test is
operational ("no text stream flows across a part boundary"), not visual ("there is a rule");
counts are never asked — ``independent_parts = len(parts)`` is derived by the harness.
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
SYSTEM_PROMPT = """\
You are an expert analyst of historical document page images. Your only task on this page is to \
identify WHAT KIND of document it is and its INDEPENDENT PARTS. Do not count columns. Do not \
transcribe.

Definitions:
- document_category: the kind of document, judged from visual evidence alone. Choose exactly one \
of: newspaper, periodical, book, letter, manuscript, register, form, map_or_plate, other. A \
newspaper is a dated newspaper issue/page or newspaper-like news/ad sheet, including historical \
broadsheets with headlines, issue numbers, advertisements, notices, classifieds, serial stories \
or feuilleton material. Use periodical only for magazine/journal/review pages that are not \
primarily newspaper issue pages.
- An INDEPENDENT PART is a region of the page that is read as its own separate stream, such that \
no text flows across the boundary into another region. Typical examples: a block of news above \
and a serialized story, feuilleton or book-format insert below. A full-width horizontal rule or \
an abrupt change of typography are EVIDENCE of a boundary, but the test is always: does any text \
stream continue across it? A newspaper page remains ONE part even when it contains many articles, \
notices, advertisements, classifieds, rubrics, thematic sections, vertical bands or local boxes. \
Do not split by headings, item clusters, ad rows or visual boxes.
- Most pages are ONE part. For newspapers, default to a single part unless a large separate \
reading stream is obvious. Scan the page top-to-bottom for a sustained lower or side insert, such \
as a serialized story, feuilleton or book-format section with its own typography or measure. A \
lower insert can be a separate part even without a heavy rule line: treat a sustained change to \
book-like line length, story typography or reading flow as enough evidence. Split such a \
sustained insert even when it belongs to the same newspaper issue. Do not confuse that with local \
advertisements, notices, rubrics, boxes or article clusters.

For each part report:
- top_frac and bottom_frac: its vertical extent as fractions of page height (0.0 = top edge, \
1.0 = bottom edge). Parts must not overlap and should cover the content area.
- anchor: two to four words identifying the part (a heading or its content kind).
- content_class: running_text (continuous body text), items_field (a field of independent items \
such as classified advertisements), mixed, or image_or_decoration.
- writing_direction: horizontal, vertical (lines written top-to-bottom, as in Chinese or \
Japanese), or mixed.

Ignore the masthead or title banner, page numbers and running headers; they belong to no part.

Return only the JSON object."""

# --------------------------------------------------------------------------------------
# Schema (property order = emission order; all fields required; no free-form extras)
# --------------------------------------------------------------------------------------
SCHEMA: dict = {
    "type": "object",
    "properties": {
        "document_category": {"type": "string", "enum": list(config.CATEGORIES)},
        "parts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,  # a page with >6 genuine reading streams does not exist in practice
            "items": {
                "type": "object",
                "properties": {
                    "top_frac": {"type": "number", "minimum": 0, "maximum": 1},
                    "bottom_frac": {"type": "number", "minimum": 0, "maximum": 1},
                    "anchor": {"type": "string", "maxLength": 60},
                    "content_class": {"type": "string", "enum": list(config.CONTENT_CLASSES)},
                    "writing_direction": {"type": "string", "enum": list(config.WRITING_DIRECTIONS)},
                },
                "required": ["top_frac", "bottom_frac", "anchor", "content_class", "writing_direction"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["document_category", "parts"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------------------
# User turn + postprocess
# --------------------------------------------------------------------------------------
def build_user_parts(full_data_url: str) -> list[dict]:
    """Pass 0 sees the full page only — global structure needs no gutter-level resolution."""
    return [
        text_part(
            "The image is the full page. Identify the document category and its independent "
            "parts. Apply the independent-part test strictly; do not split a newspaper into "
            "article groups, advertisements, notices or local visual sections."
        ),
        image_part(full_data_url),
    ]


def postprocess(parsed: dict) -> dict:
    """Normalize the parsed answer: clamp fractions, sort parts top-to-bottom, index them.

    Adds ``part_index`` (1-based, top-to-bottom) — the shared key that passes 1/2 and the harness
    use to refer to the same physical region.
    """
    parts = parsed.get("parts") or []
    cleaned: list[dict] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        top = min(max(float(part.get("top_frac", 0.0)), 0.0), 1.0)
        bottom = min(max(float(part.get("bottom_frac", 1.0)), 0.0), 1.0)
        if bottom < top:  # defensive: swap inverted extents rather than dropping the part
            top, bottom = bottom, top
        cleaned.append(
            {
                "top_frac": round(top, 3),
                "bottom_frac": round(bottom, 3),
                "anchor": str(part.get("anchor", ""))[:60],
                "content_class": str(part.get("content_class", "mixed")),
                "writing_direction": str(part.get("writing_direction", "horizontal")),
            }
        )
    cleaned.sort(key=lambda p: p["top_frac"])  # stable top-to-bottom order
    for index, part in enumerate(cleaned, start=1):
        part["part_index"] = index
    return {"document_category": str(parsed.get("document_category", "other")), "parts": cleaned}
