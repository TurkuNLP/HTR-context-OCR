"""Gold-reference extraction from the Churro HistoricalDocument XML (analysis context only).

COPIED from ``qwen3vl_layout/gold_xml.py`` (read-only reference codebase) per the layout_v2
ground rule: no cross-imports from the old pipeline. The extracted fields ride along in each
record for later agreement analysis and are NEVER shown to the model.

Additions over the original copy: ``gold_line_stats`` — line/heading counts from the gold XML,
used by the analysis as model-free text-density covariates (gold char count controls the
metric-length coupling of NLS).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def _local_name(tag: object) -> str:
    """Return the local part of an ElementTree tag, dropping any ``{namespace}`` prefix."""
    text = str(tag)
    return text.split("}", 1)[1] if "}" in text else text


def _normalize_space(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _find_descendant_by_local(root: ET.Element, local_name: str) -> ET.Element | None:
    """First element anywhere under ``root`` whose local (namespace-stripped) tag matches."""
    for element in root.iter():  # depth-first walk over the whole tree
        if _local_name(element.tag) == local_name:
            return element
    return None


def get_field_text(root: ET.Element, local_name: str) -> str:
    """Whitespace-normalized full text of the first descendant with this local name ("" if absent)."""
    element = _find_descendant_by_local(root, local_name)
    if element is None:
        return ""
    # Join itertext so inline markup inside Description/PhysicalDescription is not lost.
    return _normalize_space("".join(element.itertext()))


def extract_gold_fields(xml_text: str | None) -> dict[str, str]:
    """Parse a HistoricalDocument XML string and return its gold reference text fields.

    Returns empty strings for absent fields and unparseable/empty input, so callers can rely on
    a stable shape regardless of XML quality.
    """
    empty = {"physical_description": "", "description": ""}
    if not xml_text or not xml_text.strip():
        return dict(empty)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return dict(empty)
    return {
        "physical_description": get_field_text(root, "PhysicalDescription"),
        "description": get_field_text(root, "Description"),
    }


def gold_line_stats(xml_text: str | None) -> dict[str, int]:
    """Model-free density/structure covariates from the gold XML (analysis only, never model input).

    - ``gold_chars``: total characters across ``<Line>`` elements (the NLS length control).
    - ``gold_lines``: number of physical lines (the source ground truth preserved line breaks).
    - ``gold_headings``: number of ``<Heading>`` elements (a weak section-count reference).
    """
    empty = {"gold_chars": 0, "gold_lines": 0, "gold_headings": 0}
    if not xml_text or not xml_text.strip():
        return dict(empty)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return dict(empty)
    chars = 0
    lines = 0
    headings = 0
    for element in root.iter():
        local = _local_name(element.tag)
        if local == "Line":
            lines += 1
            chars += len(_normalize_space("".join(element.itertext())))
        elif local == "Heading":
            headings += 1
    return {"gold_chars": chars, "gold_lines": lines, "gold_headings": headings}
