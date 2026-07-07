"""Pass 1 — column counting, and nothing else (COLUMN_COUNT_METHOD.md operationalized).

The model's exclusive task is the per-part column structure. Every design element traces to a
CCM operation:
- O1  gestalt never emitted: the schema has NO count field before the enumeration;
- O2  parts are INJECTED from pass 0 (the model uses them; disputes via a flag, never silently);
- O3  anchored enumeration: per column an x-position (script-free) and a short text anchor;
- O4  the model names the band/image it counted in;
- O5  cross-band alignment reported as its own field;
- O6  spanning-element edge consistency reported (ads occupy whole numbers of columns);
- O7  width arithmetic always emitted (two numbers + implied count);
- O8  stream check + returns (a column = a region where reading descends until forced back up);
- O9  reconciliation happens in the HARNESS from the emitted route results — the model never
      outputs a final ``column_count`` at all.

Vertical-script rule (project decision #14): a top-to-bottom text line is a LINE, never a column;
for vertical parts the countable unit is the horizontal register (dan).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow flat imports under sbatch
from client import image_part, text_part  # noqa: E402

# Bake-off arms (IMPLEMENTATION_PLAN.md §11.2): what each column entry must carry.
#   dual      x_center_frac + anchor_text (the default; both anchors, always)
#   x_only    positional anchor only (script-free arm)
#   text_only text anchor only (cheap-tokens arm; loses the geometric validators)
ANCHOR_MODES = ("dual", "x_only", "text_only")

# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert analyst of historical printed and handwritten pages. Your ONLY task is to \
establish the COLUMN structure of each given part of the page. You will be told the parts; do \
not redefine them. Do not transcribe the page.

Definitions:
- A COLUMN is a vertical stripe of the body-text area, set to the printer's fixed measure, \
separated from its neighbours by a printed vertical rule or a consistent white gutter. A column \
is a layout unit, not a content unit: one column may hold many separate items.
- On pages where advertisements or other elements span several columns, the column structure is \
the BASE GRID the page was set in. Spanning elements never reduce the grid: printers sold \
advertisement space in whole column-widths, so the left and right edges of a spanning element \
fall on gutter lines. Use those edges as evidence of the grid.
- Historical newspaper grids are often denser than the large visual blocks on the page. A clean \
body-text column may be narrow; 7-9 base columns are common. If a newspaper first looks like \
4-6 broad blocks, inspect whether those blocks contain narrower compositor columns before \
finalizing the enumeration.
- For advertisement mosaics and fields of notices, count the base grid in the same way. Small \
boxes and display ads may span one, two or more columns; they do not become columns themselves.
- Do not count margins, cropped page-edge slivers, gutter fragments or partial border text as \
columns. The first and last counted stripes must be real body-text columns with the same measure \
as the interior grid; if an edge stripe is visibly narrower or fragmentary, leave it out.
- The internal sub-columns of a data table, price list, timetable or directory are tabular \
structure, NOT body-text columns. A single line of text is not a column. Marginal bleed and \
printer's marks are not columns.
- In vertically written text (Chinese, Japanese), each top-to-bottom line is a LINE, never a \
column. For a vertical part, the countable unit is the horizontal register (band) of lines; a \
vertical part with one register has one unit.

You receive several images: image 1 is the full page; any further images are full-width \
horizontal bands of the same page at higher magnification, each labelled with its vertical range.

Work through each part in the given order, and for each part do ALL of the following:
1. Choose your counting region: the image and vertical range where the columns of this part run \
clean and unobstructed (on pages with large spanning elements this is usually a lower band). \
Report it as counting_band.
2. Enumerate the columns LEFT TO RIGHT, one entry per column. For each column give x_center_frac \
(the horizontal position of the column's center as a fraction of page width, 0.0 = left edge, \
1.0 = right edge) and anchor_text (two to four words read from that column, such as a heading; \
use an empty string if the script is not legible to you). Anchor texts should be distinct when \
legible; if the same phrase spans or repeats across columns, choose a different nearby phrase or \
use an empty string. Do not skip narrow columns; do not merge neighbours. This list is the \
primary result: form it carefully, column by column.
3. Width check: report the width of one clean narrow compositor column as a fraction of page \
width \
(unit_width_frac), the width of the part's whole text block (block_width_frac), and the implied \
column count (block divided by unit, rounded).
4. Stream check: decide whether one continuous text stream flows through the columns of this \
part (evidence: a sentence or hyphenated word breaking at a column's bottom and continuing at \
the top of the next column). If yes, count the RETURNS: the number of times reading must jump \
back to the top of the part to continue, and report returns and the implied column count \
(returns + 1). For a continuous stream across N columns, returns is normally N-1; do not report \
returns 0 for a multi-column continuous stream. If the part is a field of independent items \
(advertisements, notices), report reason independent_items, returns null and implied_count null. \
Do not copy the enumeration count into stream.implied_count when no continuous stream exists. If \
you cannot read the script well enough, report reason not_legible.
5. Spanning elements: if any element spans several columns, check that its edges align with the \
gutters of your enumeration (consistent / inconsistent / none_present).
6. Cross-band check: verify in a SECOND image or vertical range that the gutter positions match \
your enumeration (aligned / misaligned / not_checked).

If you believe the given part division itself is wrong, still answer for the given parts and set \
parts_disputed true.

Return only the JSON object."""

# --------------------------------------------------------------------------------------
# Schema (property order is load-bearing: enumeration BEFORE any implied count; no count field)
# --------------------------------------------------------------------------------------
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
                    # O4: where the counting happened (audit trail; absence = probably estimated).
                    "counting_band": {"type": "string", "maxLength": 100},
                    # O3: the primary artifact — one entry per column, left to right.
                    "columns": {
                        "type": "array",
                        "minItems": 0,   # 0 allowed for image_or_decoration parts (no running text)
                        "maxItems": 16,  # sanity bound; no historical page exceeds this grid
                        "items": {
                            "type": "object",
                            "properties": {
                                "x_center_frac": {"type": "number", "minimum": 0, "maximum": 1},
                                "anchor_text": {"type": "string", "maxLength": 40},
                            },
                            "required": ["x_center_frac", "anchor_text"],
                            "additionalProperties": False,
                        },
                    },
                    # O7: geometry-only route (independent of the enumeration above).
                    "width_check": {
                        "type": "object",
                        "properties": {
                            "unit_width_frac": {"type": "number", "minimum": 0, "maximum": 1},
                            "block_width_frac": {"type": "number", "minimum": 0, "maximum": 1},
                            "implied_count": {"type": "integer", "minimum": 0, "maximum": 20},
                        },
                        "required": ["unit_width_frac", "block_width_frac", "implied_count"],
                        "additionalProperties": False,
                    },
                    # O8: reading-flow route; returns are defined only where a stream exists.
                    "stream": {
                        "type": "object",
                        "properties": {
                            "exists": {"type": "boolean"},
                            "reason": {
                                "type": "string",
                                "enum": ["continuous_stream", "independent_items", "not_legible", "no_text"],
                            },
                            "returns": {"type": ["integer", "null"], "minimum": 0, "maximum": 20},
                            "implied_count": {"type": ["integer", "null"], "minimum": 1, "maximum": 21},
                        },
                        "required": ["exists", "reason", "returns", "implied_count"],
                        "additionalProperties": False,
                    },
                    # O6: spanning elements as grid evidence.
                    "spanning_edges": {
                        "type": "string",
                        "enum": ["consistent", "inconsistent", "none_present"],
                    },
                    # O5: verification by a different measurement, not by repetition.
                    "second_band_alignment": {
                        "type": "string",
                        "enum": ["aligned", "misaligned", "not_checked"],
                    },
                    # O2: the escape valve for a wrong injected frame (never silent deviation).
                    "parts_disputed": {"type": "boolean"},
                },
                "required": [
                    "part_index",
                    "counting_band",
                    "columns",
                    "width_check",
                    "stream",
                    "spanning_edges",
                    "second_band_alignment",
                    "parts_disputed",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["parts"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------------------
# Bake-off variants (anchor format arms). ``SYSTEM_PROMPT``/``SCHEMA`` above are the dual
# (default) arm; the getters below derive the x_only / text_only arms from them so there is
# exactly one master copy of the prompt and schema to maintain.
# --------------------------------------------------------------------------------------
# The dual-anchor instruction clause as it appears (single-line) inside SYSTEM_PROMPT.
_DUAL_CLAUSE = (
    "For each column give x_center_frac "
    "(the horizontal position of the column's center as a fraction of page width, 0.0 = left edge, "
    "1.0 = right edge) and anchor_text (two to four words read from that column, such as a heading; "
    "use an empty string if the script is not legible to you). Anchor texts should be distinct when "
    "legible; if the same phrase spans or repeats across columns, choose a different nearby phrase or "
    "use an empty string."
)
_ANCHOR_CLAUSES = {
    "dual": _DUAL_CLAUSE,
    "x_only": (
        "For each column give x_center_frac (the horizontal position of the column's center as a "
        "fraction of page width, 0.0 = left edge, 1.0 = right edge)."
    ),
    "text_only": (
        "For each column give anchor_text (two to four words read from that column, such as a "
        "heading; use an empty string if the script is not legible to you)."
    ),
}


def get_prompt(anchor_mode: str = "dual") -> str:
    """The system prompt for an anchor-mode arm (dual = the unmodified master prompt)."""
    if anchor_mode not in ANCHOR_MODES:
        raise ValueError(f"unknown anchor_mode {anchor_mode!r}")
    if anchor_mode == "dual":
        return SYSTEM_PROMPT
    prompt = SYSTEM_PROMPT.replace(_DUAL_CLAUSE, _ANCHOR_CLAUSES[anchor_mode])
    # Guard against silent divergence: if the master prompt is edited, the clause above must be
    # kept in sync or this assertion fires at import/first use rather than corrupting a run.
    assert prompt != SYSTEM_PROMPT, "dual-anchor clause not found in SYSTEM_PROMPT (edit drift)"
    return prompt


def get_schema(anchor_mode: str = "dual") -> dict:
    """The guided-JSON schema for an anchor-mode arm (drops the unused column field)."""
    if anchor_mode not in ANCHOR_MODES:
        raise ValueError(f"unknown anchor_mode {anchor_mode!r}")
    if anchor_mode == "dual":
        return SCHEMA
    schema = copy.deepcopy(SCHEMA)
    column_item = schema["properties"]["parts"]["items"]["properties"]["columns"]["items"]
    drop = "anchor_text" if anchor_mode == "x_only" else "x_center_frac"
    column_item["properties"].pop(drop)
    column_item["required"] = [k for k in column_item["required"] if k != drop]
    return schema


# --------------------------------------------------------------------------------------
# User turn + postprocess
# --------------------------------------------------------------------------------------
def _describe_parts(parts: list[dict]) -> str:
    """Render pass 0's parts as the injected frame the model must work within."""
    lines = [f"This page has {len(parts)} part(s):"]
    for part in parts:
        lines.append(
            f"  part {part['part_index']}: vertical range {part['top_frac']:.2f}-{part['bottom_frac']:.2f} "
            f"of page height; {part['anchor'] or 'unnamed'}; content: {part['content_class']}; "
            f"writing: {part['writing_direction']}."
        )
    lines.append("Analyse the column structure of each part separately, in this order.")
    return "\n".join(lines)


def _describe_images(full_present: bool, bands: list[dict]) -> str:
    """Name every image so counting_band references are auditable (numbering matches the order
    the images are attached in build_user_parts, whatever the input-strategy arm)."""
    lines: list[str] = []
    index = 1
    if full_present:
        lines.append("Image 1: the full page.")
        index = 2
    for band in bands:
        lines.append(
            f"Image {index}: full-width band of the page, vertical range "
            f"{band['top_frac']:.2f}-{band['bottom_frac']:.2f} of page height (higher magnification)."
        )
        index += 1
    return "\n".join(lines)


def build_user_parts(
    full_data_url: str | None, bands: list[dict], parts: list[dict]
) -> list[dict]:
    """User turn = injected parts + image manifest + the images.

    Input-strategy arms (plan §11.2) are expressed by what the runner passes in:
    full+bands (default), full only (``bands=[]``), or bands only (``full_data_url=None``).
    """
    if full_data_url is None and not bands:
        raise ValueError("pass 1 needs at least one image (full page and/or bands)")
    user: list[dict] = [
        text_part(_describe_parts(parts) + "\n\n" + _describe_images(full_data_url is not None, bands))
    ]
    if full_data_url is not None:
        user.append(image_part(full_data_url))
    for band in bands:
        user.append(image_part(band["data_url"]))
    return user


def postprocess(parsed: dict) -> dict:
    """Light normalization: clamp fractions, keep parts sorted by part_index.

    Variant-aware: only fields the arm's schema actually produced are normalized — injecting
    defaults for a dropped field would fabricate data (e.g. x=0.0 for every column in the
    text_only arm would trip the monotonicity validator on the model's behalf).
    """
    raw_parts = parsed.get("parts")
    if not isinstance(raw_parts, list):
        raise ValueError("pass1 payload missing required parts list")
    parts = [p for p in raw_parts if isinstance(p, dict)]
    if len(parts) != len(raw_parts):
        raise ValueError("pass1 payload contains non-object part entries")
    for part in parts:
        columns = [c for c in (part.get("columns") or []) if isinstance(c, dict)]
        for column in columns:
            if "x_center_frac" in column:  # present in dual / x_only arms
                column["x_center_frac"] = round(min(max(float(column["x_center_frac"]), 0.0), 1.0), 4)
            if "anchor_text" in column:  # present in dual / text_only arms
                column["anchor_text"] = str(column["anchor_text"])[:40]
        part["columns"] = columns
        stream = part.get("stream")
        if isinstance(stream, dict) and stream.get("exists") is False:
            # A no-stream branch has no reading returns. Keep the column routes in
            # enumeration/width only so validators do not flag stale branch data.
            stream["returns"] = None
            stream["implied_count"] = None
    parts.sort(key=lambda p: int(p.get("part_index", 0)))
    return {"parts": parts}
