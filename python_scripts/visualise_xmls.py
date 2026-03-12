from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import os
from pathlib import Path
import re
import shutil
from typing import Any
import xml.etree.ElementTree as ET

try:
    from lxml import etree as _LXML_ETREE  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _LXML_ETREE = None


DEFAULT_XML_RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "churchbook_results"
    / "xml_results"
)
DEFAULT_HTML_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "churchbook_results"
    / "xml_html_renders"
)
DEFAULT_STYLESHEET_SOURCE = Path(__file__).resolve().with_name("visualise_xmls.css")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Markdown ATX heading matcher:
# - ^\s*      : allow indentation before heading markers
# - (#{1,6})  : capture heading level marker (1-6 hashes)
# - \s+(.+?)  : capture visible heading text
# - \s*$      : ignore trailing whitespace
_ATX_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")

# Generic fence matcher for both ``` and ~~~ code fences:
# - ([`~]{3,}) captures the exact opening/closing fence token (char + length)
# - (.*) captures optional info string (e.g., "xml")
_FENCE_RE = re.compile(r"^\s*([`~]{3,})(.*)$")

# Metadata bullet parser used for top markdown header fields:
# "- Time (UTC): `...`", "- Model: ..."
# Captures key and value around the first colon.
_METADATA_BULLET_RE = re.compile(r"^\s*-\s*([^:]+):\s*(.+?)\s*$")

# Tokenizer for heading scoring:
# extracts lowercase alphanumeric tokens only (drops punctuation).
_WORD_RE = re.compile(r"[a-z0-9]+")

# Primary XML extractor:
# - allows optional namespace prefix on tag name (e.g., ns:HistoricalDocument)
# - \b after tag name avoids partial matches
# - DOTALL allows matching across newlines
# - non-greedy .*? ensures we capture the first full document block
_HISTORICAL_DOCUMENT_RE = re.compile(
    r"(<(?:\w+:)?HistoricalDocument\b.*?</(?:\w+:)?HistoricalDocument>)",
    flags=re.DOTALL | re.IGNORECASE,
)

# Opening HistoricalDocument tag detector with optional prefix capture.
# Captured "prefix" is reused when auto-appending a missing close tag.
_HISTORICAL_DOCUMENT_OPEN_RE = re.compile(
    r"<(?P<prefix>(?:\w+:)?)HistoricalDocument\b",
    flags=re.IGNORECASE,
)

# Closing HistoricalDocument detector used to avoid double-closing.
_HISTORICAL_DOCUMENT_CLOSE_RE = re.compile(
    r"</(?:\w+:)?HistoricalDocument\s*>",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class MarkdownSection:
    index: int
    level: int
    heading: str
    body: str


@dataclass(slots=True)
class FencedCodeBlock:
    language: str
    content: str
    start_line: int
    end_line: int


@dataclass(slots=True)
class ParsedXmlResult:
    path: Path
    metadata: dict[str, str]
    selected_section: str | None
    xml_payload: str


@dataclass(slots=True)
class HtmlRenderArtifact:
    source_markdown_path: Path
    output_html_path: Path
    selected_section: str | None
    xml_well_formed: bool
    parse_warning: str | None
    source_image_path: Path | None


class NoXmlPayloadError(ValueError):
    """Raised when markdown content contains no extractable XML payload."""


def iter_xml_markdown_paths(results_dir: Path = DEFAULT_XML_RESULTS_DIR) -> list[Path]:
    """Return sorted markdown result files from an xml_results directory."""
    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    return sorted(path for path in results_dir.rglob("*.md") if path.is_file())


def parse_markdown_metadata(markdown_text: str) -> dict[str, str]:
    """Read top metadata bullets that appear before the first level-2 heading."""
    metadata: dict[str, str] = {}
    for line in markdown_text.splitlines():
        if line.lstrip().startswith("## "):
            break
        match = _METADATA_BULLET_RE.match(line)
        if not match:
            continue
        key = _normalize_metadata_key(match.group(1))
        value = match.group(2).strip().strip("`")
        metadata[key] = value
    return metadata


def split_markdown_sections(markdown_text: str) -> list[MarkdownSection]:
    """Split markdown into heading sections while ignoring headings inside code fences."""
    sections: list[MarkdownSection] = []
    current_heading = ""
    current_level = 0
    current_lines: list[str] = []
    section_index = 0

    # Track fence state so headings inside code fences are ignored.
    in_fence = False
    fence_char = ""
    fence_len = 0

    # Single-pass state machine:
    # - toggles in_fence when fence delimiters are seen
    # - treats headings as section boundaries only when not in a fence
    for line in markdown_text.splitlines(keepends=True):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            marker_char = marker[0]
            marker_len = len(marker)
            if not in_fence:
                in_fence = True
                fence_char = marker_char
                fence_len = marker_len
            elif marker_char == fence_char and marker_len >= fence_len:
                in_fence = False

        if not in_fence:
            heading_match = _ATX_HEADING_RE.match(line)
            if heading_match:
                body = "".join(current_lines)
                if current_heading or body.strip():
                    sections.append(
                        MarkdownSection(
                            index=section_index,
                            level=current_level,
                            heading=current_heading,
                            body=body,
                        )
                    )
                    section_index += 1
                current_level = len(heading_match.group(1))
                current_heading = heading_match.group(2).strip()
                current_lines = []
                continue

        current_lines.append(line)

    trailing_body = "".join(current_lines)
    if current_heading or trailing_body.strip():
        sections.append(
            MarkdownSection(
                index=section_index,
                level=current_level,
                heading=current_heading,
                body=trailing_body,
            )
        )

    return sections


def extract_fenced_code_blocks(text: str) -> list[FencedCodeBlock]:
    """Extract fenced code blocks from markdown text."""
    blocks: list[FencedCodeBlock] = []
    lines = text.splitlines()
    # Parse fences line-by-line to preserve code block order and language info.
    in_fence = False
    fence_char = ""
    fence_len = 0
    language = ""
    start_line = 0
    buffer: list[str] = []

    # Second state machine:
    # - start block on opening fence
    # - close block only on matching fence char and sufficient fence length
    # - preserves code content verbatim between fence lines
    for line_number, line in enumerate(lines, start=1):
        fence_match = _FENCE_RE.match(line)
        if not in_fence:
            if not fence_match:
                continue
            marker = fence_match.group(1)
            fence_char = marker[0]
            fence_len = len(marker)
            info = fence_match.group(2).strip()
            language = info.split(maxsplit=1)[0].casefold() if info else ""
            start_line = line_number
            buffer = []
            in_fence = True
            continue

        if fence_match:
            marker = fence_match.group(1)
            marker_char = marker[0]
            marker_len = len(marker)
            trailing = fence_match.group(2).strip()
            if marker_char == fence_char and marker_len >= fence_len and not trailing:
                blocks.append(
                    FencedCodeBlock(
                        language=language,
                        content="\n".join(buffer).strip("\n"),
                        start_line=start_line,
                        end_line=line_number,
                    )
                )
                in_fence = False
                fence_char = ""
                fence_len = 0
                language = ""
                start_line = 0
                buffer = []
                continue

        buffer.append(line)

    if in_fence:
        blocks.append(
            FencedCodeBlock(
                language=language,
                content="\n".join(buffer).strip("\n"),
                start_line=start_line,
                end_line=len(lines),
            )
        )

    return blocks


def select_preferred_xml_payload(markdown_text: str) -> tuple[str, str | None]:
    """Pick the most relevant XML block from a markdown artifact.

    The selection is driven by parsed section semantics and fence content, not a
    brittle exact-string lookup for a single heading.
    """
    sections = split_markdown_sections(markdown_text)
    # Candidate tuple:
    # (score, -section_index, -block_index, heading, xml_fragment)
    # We use negative indices so later sections/blocks win on ties when reverse-sorting.
    candidates: list[tuple[int, int, int, str, str]] = []

    for section in sections:
        # Prioritize sections by heading semantics instead of exact heading text.
        section_score = _score_section_heading(section.heading)
        code_blocks = extract_fenced_code_blocks(section.body)
        for block_index, block in enumerate(code_blocks):
            xml_fragment = _extract_xml_fragment(block.content)
            if not xml_fragment:
                continue

            # Fence language and payload shape influence confidence.
            block_score = section_score
            if block.language == "xml":
                block_score += 40
            elif not block.language:
                block_score += 15
            else:
                block_score += 5

            if "historicaldocument" in xml_fragment.casefold():
                block_score += 30

            candidates.append(
                (
                    block_score,
                    -section.index,
                    -block_index,
                    section.heading,
                    xml_fragment,
                )
            )

    if candidates:
        candidates.sort(reverse=True)
        _, _, _, heading, xml_payload = candidates[0]
        return xml_payload, heading or None

    # Last-resort fallback: scan full markdown content for a usable XML fragment.
    fallback = _extract_xml_fragment(markdown_text)
    if fallback:
        return fallback, None
    raise NoXmlPayloadError("No XML payload could be extracted from markdown content.")


def parse_xml_markdown_file(path: Path) -> ParsedXmlResult:
    """Parse one markdown XML result artifact."""
    markdown_text = path.read_text(encoding="utf-8")
    xml_payload, selected_section = select_preferred_xml_payload(markdown_text)
    metadata = parse_markdown_metadata(markdown_text)
    return ParsedXmlResult(
        path=path,
        metadata=metadata,
        selected_section=selected_section,
        xml_payload=xml_payload,
    )


def find_xml_in_markdown_text(markdown_text: str) -> str:
    """Return only the best XML payload found in markdown text."""
    xml_payload, _ = select_preferred_xml_payload(markdown_text)
    return xml_payload


def find_xml_in_markdown_file(path: Path) -> str:
    """Return only the best XML payload found in a markdown result file."""
    return find_xml_in_markdown_text(path.read_text(encoding="utf-8"))


def read_xml_results(results_dir: Path = DEFAULT_XML_RESULTS_DIR) -> list[ParsedXmlResult]:
    """Load and parse all markdown XML results from a directory."""
    parsed_results: list[ParsedXmlResult] = []
    skipped_non_xml = 0
    for path in iter_xml_markdown_paths(results_dir):
        try:
            parsed_results.append(parse_xml_markdown_file(path))
        except NoXmlPayloadError:
            skipped_non_xml += 1

    if not parsed_results and skipped_non_xml > 0:
        raise ValueError(
            "No XML payloads found under {} ({} markdown files scanned, none contained XML).".format(
                results_dir, skipped_non_xml
            )
        )

    if skipped_non_xml > 0:
        print(
            "[warn] Skipped {} markdown file(s) without XML payload under {}".format(
                skipped_non_xml, results_dir
            )
        )
    return parsed_results


def render_xml_results_to_html(
    results_dir: Path = DEFAULT_XML_RESULTS_DIR,
    output_dir: Path = DEFAULT_HTML_OUTPUT_DIR,
    limit: int = 0,
    stylesheet_source: Path = DEFAULT_STYLESHEET_SOURCE,
    images_dir: Path | None = None,
    copy_images_into_output: bool = False,
) -> tuple[list[HtmlRenderArtifact], Path]:
    """Render XML payloads stored in markdown artifacts to standalone HTML pages."""
    parsed_results = read_xml_results(results_dir)
    if limit > 0:
        parsed_results = parsed_results[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    stylesheet_output_path = _copy_stylesheet_to_output(stylesheet_source, output_dir)
    artifacts: list[HtmlRenderArtifact] = []
    image_lookup: dict[str, Path] = {}
    duplicate_image_names: set[str] = set()
    resolved_image_links = 0
    unresolved_image_links = 0
    copied_image_links = 0
    copied_images_root: Path | None = None

    if images_dir is not None:
        image_lookup, duplicate_image_names = _build_image_name_index(images_dir)
        if duplicate_image_names:
            print(
                "[warn] Found {} duplicate image filename(s) under {}. "
                "Ambiguous names will not be linked.".format(
                    len(duplicate_image_names),
                    images_dir,
                )
            )
        if copy_images_into_output:
            copied_images_root = output_dir / "_paired_images"
            copied_images_root.mkdir(parents=True, exist_ok=True)

    for parsed in parsed_results:
        relative_markdown = parsed.path.relative_to(results_dir)
        output_path = (output_dir / relative_markdown).with_suffix(".html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stylesheet_href = _relative_href(output_path.parent, stylesheet_output_path)
        image_metadata_value = _get_metadata_image_reference(parsed.metadata)
        source_image_path = _resolve_source_image_path(
            metadata_image=image_metadata_value,
            markdown_path=parsed.path,
            images_dir=images_dir,
            image_lookup=image_lookup,
            duplicate_image_names=duplicate_image_names,
        )
        image_link_target_path = source_image_path
        if copied_images_root is not None and source_image_path is not None:
            image_link_target_path = _copy_source_image_to_output(
                source_image_path=source_image_path,
                output_dir=output_dir,
                images_dir=images_dir,
                copied_images_root=copied_images_root,
            )
            copied_image_links += 1
        source_image_href = (
            _relative_href(output_path.parent, image_link_target_path)
            if image_link_target_path is not None
            else None
        )
        if images_dir is not None:
            if source_image_href is not None:
                resolved_image_links += 1
            else:
                unresolved_image_links += 1

        rendered_fragment, xml_well_formed, parse_warning = render_xml_payload_fragment(
            parsed.xml_payload
        )
        html_document = build_result_html_document(
            parsed=parsed,
            rendered_fragment=rendered_fragment,
            xml_well_formed=xml_well_formed,
            parse_warning=parse_warning,
            stylesheet_href=stylesheet_href,
            source_image_href=source_image_href,
        )
        output_path.write_text(html_document, encoding="utf-8")
        artifacts.append(
            HtmlRenderArtifact(
                source_markdown_path=parsed.path,
                output_html_path=output_path,
                selected_section=parsed.selected_section,
                xml_well_formed=xml_well_formed,
                parse_warning=parse_warning,
                source_image_path=image_link_target_path,
            )
        )

    index_path = output_dir / "index.html"
    index_path.write_text(
        build_index_html_document(
            artifacts=artifacts,
            output_dir=output_dir,
            results_dir=results_dir,
            stylesheet_href=_relative_href(output_dir, stylesheet_output_path),
            image_linking_enabled=images_dir is not None,
        ),
        encoding="utf-8",
    )
    if images_dir is not None:
        print(
            "Source image links -> resolved: {}, unresolved: {}".format(
                resolved_image_links,
                unresolved_image_links,
            )
        )
        if copy_images_into_output:
            print(
                "Copied paired images into output: {} file(s) at {}".format(
                    copied_image_links,
                    copied_images_root,
                )
            )
    return artifacts, index_path


def render_xml_payload_fragment(xml_payload: str) -> tuple[str, bool, str | None]:
    """Render one XML payload to an HTML body fragment."""
    root, xml_well_formed, parse_warning = _parse_xml_root(xml_payload)
    if root is None:
        fallback_fragment = _render_malformed_xml_fallback_fragment(xml_payload)
        if fallback_fragment is not None:
            fallback_warning = (
                "{} | Rendered with regex fallback parser.".format(parse_warning)
                if parse_warning
                else "Rendered with regex fallback parser."
            )
            return fallback_fragment, False, fallback_warning
        return _render_parse_failure_fragment(xml_payload, parse_warning), False, parse_warning
    return _render_historical_document_layout(root), xml_well_formed, parse_warning


def build_result_html_document(
    *,
    parsed: ParsedXmlResult,
    rendered_fragment: str,
    xml_well_formed: bool,
    parse_warning: str | None,
    stylesheet_href: str,
    source_image_href: str | None,
) -> str:
    """Build a full standalone HTML document for one markdown artifact."""
    image_metadata_value = _get_metadata_image_reference(parsed.metadata)
    document_title = image_metadata_value or parsed.path.stem
    status_label = "well-formed" if xml_well_formed else "recovered or malformed"
    metadata_rows = [
        ("Source markdown", str(parsed.path)),
        ("Image", image_metadata_value or "-"),
        ("Model", parsed.metadata.get("model", "-")),
        ("Time (UTC)", parsed.metadata.get("time_utc", "-")),
        ("Selected section", parsed.selected_section or "-"),
        ("XML status", status_label),
    ]
    metadata_html = "".join(
        "<dt>{}</dt><dd>{}</dd>".format(html.escape(key), html.escape(value))
        for key, value in metadata_rows
    )
    warning_html = ""
    if parse_warning:
        warning_html = (
            '<div class="warning-banner"><strong>Rendering note:</strong> {}</div>'.format(
                html.escape(parse_warning)
            )
        )
    source_image_link_html = ""
    if source_image_href:
        source_image_link_html = (
            '<section class="artifact-actions">'
            '<a class="source-image-link" href="{href}" target="_blank" rel="noopener noreferrer">'
            "Open source image in new tab"
            "</a>"
            "</section>"
        ).format(href=html.escape(source_image_href))

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="stylesheet" href="{stylesheet_href}" />
</head>
<body class="result-page">
  <div class="page-chrome">
    <header class="artifact-header">
      <h1>{title}</h1>
      <p>Layout-faithful HTML rendering from HistoricalDocument XML.</p>
    </header>
    <section class="artifact-meta">
      <dl>{metadata}</dl>
    </section>
    {source_image_link}
    {warning}
    <main class="rendered-layout">
      {fragment}
    </main>
    <details class="raw-xml-details">
      <summary>Raw XML payload</summary>
      <pre>{raw_xml}</pre>
    </details>
  </div>
</body>
</html>
""".format(
        title=html.escape(document_title),
        stylesheet_href=html.escape(stylesheet_href),
        metadata=metadata_html,
        source_image_link=source_image_link_html,
        warning=warning_html,
        fragment=rendered_fragment,
        raw_xml=html.escape(parsed.xml_payload),
    )


def build_index_html_document(
    *,
    artifacts: list[HtmlRenderArtifact],
    output_dir: Path,
    results_dir: Path,
    stylesheet_href: str,
    image_linking_enabled: bool,
) -> str:
    """Build an index HTML page linking all rendered XML documents."""
    well_formed_count = sum(1 for item in artifacts if item.xml_well_formed)
    recovered_count = len(artifacts) - well_formed_count
    paired_count = sum(1 for item in artifacts if item.source_image_path is not None)
    unpaired_count = len(artifacts) - paired_count
    image_summary = (
        "Image links: paired: <strong>{}</strong> | unpaired: <strong>{}</strong>".format(
            paired_count,
            unpaired_count,
        )
        if image_linking_enabled
        else "Image links: <strong>n/a</strong> (pass --images-dir to enable pairing)"
    )

    rows: list[str] = []
    for artifact in artifacts:
        link = artifact.output_html_path.relative_to(output_dir).as_posix()
        source_rel = artifact.source_markdown_path.relative_to(results_dir).as_posix()
        status = "well-formed" if artifact.xml_well_formed else "recovered/malformed"
        warning = artifact.parse_warning or "-"
        image_pairing_html = "n/a"
        if image_linking_enabled:
            if artifact.source_image_path is not None:
                image_href = _relative_href(output_dir, artifact.source_image_path)
                image_pairing_html = '<a href="{href}" target="_blank" rel="noopener noreferrer">paired (open)</a>'.format(
                    href=html.escape(image_href)
                )
            else:
                image_pairing_html = "unpaired"
        rows.append(
            (
                "<tr>"
                "<td><a href=\"{link}\">{name}</a></td>"
                "<td>{section}</td>"
                "<td>{status}</td>"
                "<td>{warning}</td>"
                "<td>{image_pairing}</td>"
                "</tr>"
            ).format(
                link=html.escape(link),
                name=html.escape(source_rel),
                section=html.escape(artifact.selected_section or "-"),
                status=html.escape(status),
                warning=html.escape(warning),
                image_pairing=image_pairing_html,
            )
        )

    table_rows = "\n".join(rows) if rows else "<tr><td colspan=\"5\">No documents rendered.</td></tr>"
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>XML HTML Render Index</title>
  <link rel="stylesheet" href="{stylesheet_href}" />
</head>
<body class="index-page">
  <div class="index-wrap">
    <header>
      <h1>XML Render Index</h1>
      <p>Source folder: <code>{source_dir}</code></p>
      <p>Rendered: <strong>{total}</strong> | Well-formed: <strong>{well}</strong> | Recovered/Malformed: <strong>{recovered}</strong></p>
      <p>{image_summary}</p>
    </header>
    <table>
      <thead>
        <tr>
          <th>Document</th>
          <th>Selected XML Section</th>
          <th>Status</th>
          <th>Warning</th>
          <th>Image Pairing</th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>
</body>
</html>
""".format(
        stylesheet_href=html.escape(stylesheet_href),
        source_dir=html.escape(str(results_dir)),
        total=len(artifacts),
        well=well_formed_count,
        recovered=recovered_count,
        image_summary=image_summary,
        rows=table_rows,
    )


def _render_historical_document_layout(root: Any) -> str:
    root_name = _local_name(getattr(root, "tag", ""))
    if root_name != "HistoricalDocument":
        return _render_generic_element(root, as_card=True)

    parts = ['<section class="historical-document">']

    metadata = _first_child_named(root, "Metadata")
    if metadata is not None:
        parts.append(_render_metadata_block(metadata))

    pages = _children_named(root, "Page")
    if pages:
        parts.append('<section class="page-list">')
        for page_index, page in enumerate(pages, start=1):
            parts.append(_render_page_block(page, page_index))
        parts.append("</section>")
    else:
        parts.append(_render_generic_element(root, as_card=True))

    parts.append("</section>")
    return "".join(parts)


def _render_metadata_block(metadata_element: Any) -> str:
    rows: list[tuple[str, str]] = []
    for child in _child_elements(metadata_element):
        key = _local_name(getattr(child, "tag", "")) or "Unknown"
        value = _normalize_space(_iter_text(child))
        rows.append((key, value or "-"))

    if not rows:
        return ""

    content = "".join(
        "<dt>{}</dt><dd>{}</dd>".format(html.escape(key), html.escape(value))
        for key, value in rows
    )
    return (
        '<section class="metadata-block">'
        '<h2>Document Metadata</h2>'
        '<dl class="metadata-grid">{}</dl>'
        "</section>".format(content)
    )


def _render_page_block(page: Any, page_index: int) -> str:
    page_number = _extract_page_number(page) or str(page_index)
    parts = [
        '<article class="page-block" data-page="{}">'.format(html.escape(page_number)),
        '<div class="page-label">Page {}</div>'.format(html.escape(page_number)),
    ]

    header = _first_child_named(page, "Header")
    if header is not None:
        parts.append(_render_header_block(header))

    body = _first_child_named(page, "Body")
    if body is not None:
        parts.append(_render_body_block(body))

    for child in _child_elements(page):
        name = _local_name(getattr(child, "tag", ""))
        if name in {"Header", "Body"}:
            continue
        parts.append(_render_generic_element(child, as_card=True))

    parts.append("</article>")
    return "".join(parts)


def _render_header_block(header: Any) -> str:
    parts = ['<header class="page-header">']

    headings = _children_named(header, "Heading")
    for heading in headings:
        heading_type = _slugify_token(heading.attrib.get("type", "generic"))
        lines = _collect_line_texts(heading, recursive=True)
        if not lines:
            lines = [_normalize_space(_iter_text(heading))]
        title_tag = "h2" if heading.attrib.get("type", "").casefold() == "main" else "h3"
        content = "<br>".join(_escape_or_nbsp(line) for line in lines if line or len(lines) == 1)
        parts.append(
            "<{tag} class=\"doc-heading type-{kind}\">{content}</{tag}>".format(
                tag=title_tag,
                kind=html.escape(heading_type),
                content=content or "&nbsp;",
            )
        )

    page_number_entries = _children_named(header, "PageNumber")
    if page_number_entries:
        page_number = _normalize_space(_iter_text(page_number_entries[0]))
        if page_number:
            parts.append(
                '<div class="header-page-number">Recorded Page Number: {}</div>'.format(
                    html.escape(page_number)
                )
            )

    for child in _child_elements(header):
        name = _local_name(getattr(child, "tag", ""))
        if name in {"Heading", "PageNumber"}:
            continue
        parts.append(_render_generic_element(child, as_card=False))

    parts.append("</header>")
    return "".join(parts)


def _render_body_block(body: Any) -> str:
    parts = ['<section class="page-body">']
    children = _child_elements(body)

    if not children:
        body_lines = _collect_line_texts(body, recursive=True)
        if body_lines:
            parts.append('<div class="body-lines">{}</div>'.format(_render_line_stack(body_lines, "body-line")))
        else:
            body_text = _normalize_space(_iter_text(body))
            if body_text:
                parts.append('<p class="body-text">{}</p>'.format(html.escape(body_text)))
    else:
        for child in children:
            name = _local_name(getattr(child, "tag", ""))
            if name == "Table":
                parts.append(_render_table_block(child))
            else:
                parts.append(_render_generic_element(child, as_card=True))

    parts.append("</section>")
    return "".join(parts)


def _render_table_block(table: Any) -> str:
    rows = _children_named(table, "TableRow")
    if not rows:
        return _render_generic_element(table, as_card=True)

    parts = [
        '<section class="table-block">',
        '<div class="table-scroll">',
        '<table class="xml-table">',
        "<tbody>",
    ]

    for row_index, row in enumerate(rows):
        parts.append("<tr>")
        cells = _children_named(row, "TableCell")
        if not cells:
            parts.append('<td class="table-cell empty-cell">&nbsp;</td>')
        for cell in cells:
            parts.append(_render_table_cell(cell, row_index))
        parts.append("</tr>")

    parts.extend(["</tbody>", "</table>", "</div>", "</section>"])
    return "".join(parts)


def _render_table_cell(cell: Any, row_index: int) -> str:
    role = (cell.attrib.get("role") or "").casefold()
    tag = "th" if role == "header" else "td"
    classes = ["table-cell"]
    if role:
        classes.append("role-{}".format(_slugify_token(role)))
    if row_index == 0 and tag == "th":
        classes.append("header-row-cell")

    span_attrs = []
    for span_name in ("colspan", "rowspan"):
        span_value = cell.attrib.get(span_name)
        if span_value and span_value.isdigit():
            span_attrs.append(' {}="{}"'.format(span_name, span_value))

    # Start with direct <Line> children for faithful table-cell content ordering.
    lines = _collect_line_texts(cell, recursive=False)
    if not lines:
        lines = _collect_line_texts(cell, recursive=True)

    if lines:
        content = _render_line_stack(lines, "cell-line")
    else:
        content_text = _normalize_space(_iter_text(cell))
        content = html.escape(content_text) if content_text else "&nbsp;"

    return "<{tag} class=\"{classes}\"{attrs}>{content}</{tag}>".format(
        tag=tag,
        classes=" ".join(classes),
        attrs="".join(span_attrs),
        content=content,
    )


def _render_generic_element(element: Any, *, as_card: bool) -> str:
    name = _local_name(getattr(element, "tag", "")) or "Node"
    wrapper_class = "generic-card" if as_card else "generic-inline"
    classes = "{} tag-{}".format(wrapper_class, _slugify_token(name))
    parts = ['<section class="{}">'.format(html.escape(classes))]
    parts.append('<div class="generic-label">{}</div>'.format(html.escape(name)))

    lines = _collect_line_texts(element, recursive=False)
    if lines:
        parts.append('<div class="generic-lines">{}</div>'.format(_render_line_stack(lines, "generic-line")))

    direct_text = _normalize_space(getattr(element, "text", "") or "")
    if direct_text:
        parts.append('<p class="generic-text">{}</p>'.format(html.escape(direct_text)))

    child_nodes = [
        child for child in _child_elements(element) if _local_name(getattr(child, "tag", "")) != "Line"
    ]
    if child_nodes:
        parts.append('<div class="generic-children">')
        for child in child_nodes:
            child_name = _local_name(getattr(child, "tag", ""))
            if child_name == "Table":
                parts.append(_render_table_block(child))
            else:
                parts.append(_render_generic_element(child, as_card=True))
        parts.append("</div>")
    elif not lines and not direct_text:
        parts.append('<div class="generic-empty">&nbsp;</div>')

    parts.append("</section>")
    return "".join(parts)


def _render_parse_failure_fragment(xml_payload: str, parse_warning: str | None) -> str:
    warning_message = parse_warning or "Unable to parse XML payload."
    return (
        '<section class="parse-failure">'
        '<h2>XML Parsing Failed</h2>'
        '<p>{}</p>'
        '<pre>{}</pre>'
        "</section>".format(
            html.escape(warning_message),
            html.escape(xml_payload),
        )
    )


def _render_malformed_xml_fallback_fragment(xml_payload: str) -> str | None:
    """Best-effort layout renderer for malformed XML text.

    This parser is intentionally shallow: it captures headings, page number, and
    table row/cell structure from tag-like patterns so users can still inspect
    layout when strict XML parsing fails.
    """
    # Regex is intentionally permissive and non-greedy:
    # - <TableRow\b[^>]*>  : row start, with optional attributes
    # - (.*?)              : row body
    # - </TableRow>        : row end
    # DOTALL is required because row bodies span many lines.
    row_matches = re.findall(
        r"<TableRow\b[^>]*>(.*?)</TableRow>",
        xml_payload,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not row_matches:
        return None

    # Extract a representative page number for page label UI.
    page_number_match = re.search(
        r"<PageNumber\b[^>]*>(.*?)</PageNumber>",
        xml_payload,
        flags=re.DOTALL | re.IGNORECASE,
    )
    page_number = (
        _normalize_space(_strip_tags(page_number_match.group(1)))
        if page_number_match
        else "?"
    )

    heading_rows: list[str] = []
    # Heading extraction:
    # - captures raw attribute text and heading body separately
    # - later we parse type="main|sub|..." from attributes
    for heading_match in re.finditer(
        r"<Heading\b([^>]*)>(.*?)</Heading>",
        xml_payload,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        attrs = heading_match.group(1)
        body = heading_match.group(2)
        # type attribute parser:
        # - type\s*=\s*['"]value['"]
        # - supports both single and double quotes
        heading_type_match = re.search(
            r"type\s*=\s*[\"']([^\"']+)[\"']",
            attrs,
            flags=re.IGNORECASE,
        )
        heading_type = (heading_type_match.group(1) if heading_type_match else "heading").casefold()
        heading_tag = "h2" if heading_type == "main" else "h3"
        heading_lines = _extract_line_values_from_fragment(body)
        heading_text = "<br>".join(_escape_or_nbsp(line) for line in heading_lines) if heading_lines else "&nbsp;"
        heading_rows.append(
            '<{tag} class="doc-heading type-{kind}">{text}</{tag}>'.format(
                tag=heading_tag,
                kind=html.escape(_slugify_token(heading_type)),
                text=heading_text,
            )
        )

    table_rows: list[str] = []
    for row_index, row_body in enumerate(row_matches):
        # Cell capture mirrors row capture:
        # - grabs each cell's attributes and body payload for role/line extraction.
        cell_matches = re.findall(
            r"<TableCell\b([^>]*)>(.*?)</TableCell>",
            row_body,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cells_html: list[str] = []
        for attrs, cell_body in cell_matches:
            # role attribute (e.g., role="header") controls th/td rendering.
            role_match = re.search(
                r"role\s*=\s*[\"']([^\"']+)[\"']",
                attrs,
                flags=re.IGNORECASE,
            )
            role = (role_match.group(1) if role_match else "").casefold()
            cell_tag = "th" if role == "header" else "td"
            classes = ["table-cell"]
            if role:
                classes.append("role-{}".format(_slugify_token(role)))
            if row_index == 0 and cell_tag == "th":
                classes.append("header-row-cell")
            lines = _extract_line_values_from_fragment(cell_body)
            cell_content = _render_line_stack(lines, "cell-line") if lines else "&nbsp;"
            cells_html.append(
                "<{tag} class=\"{classes}\">{content}</{tag}>".format(
                    tag=cell_tag,
                    classes=" ".join(classes),
                    content=cell_content,
                )
            )
        if cells_html:
            table_rows.append("<tr>{}</tr>".format("".join(cells_html)))

    if not table_rows:
        return None

    heading_html = "".join(heading_rows)
    return (
        '<section class="historical-document">'
        '<article class="page-block" data-page="{page}">'
        '<div class="page-label">Page {page}</div>'
        '<header class="page-header">{headings}'
        '<div class="header-page-number">Recorded Page Number: {page}</div>'
        "</header>"
        '<section class="page-body"><section class="table-block"><div class="table-scroll">'
        '<table class="xml-table"><tbody>{rows}</tbody></table>'
        "</div></section></section>"
        "</article>"
        "</section>"
    ).format(
        page=html.escape(page_number),
        headings=heading_html,
        rows="".join(table_rows),
    )


def _extract_line_values_from_fragment(fragment: str) -> list[str]:
    values: list[str] = []
    # Line parser handles both forms:
    # - self-closing: <Line/>
    # - explicit:     <Line>...</Line>
    # Pattern details:
    # - <Line\b[^>]*        : start tag with optional attrs
    # - (?:/>|>(.*?)</Line>): either self-close OR capture inner text in group 1
    for line_match in re.finditer(
        r"<Line\b[^>]*(?:/>|>(.*?)</Line>)",
        fragment,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        text_value = line_match.group(1) if line_match.group(1) is not None else ""
        normalized = _normalize_space(_strip_tags(text_value))
        values.append(normalized)

    if values:
        return values

    # If no explicit Line tags exist, strip tags and keep residual text as one line.
    fallback_text = _normalize_space(_strip_tags(fragment))
    return [fallback_text] if fallback_text else []


def _strip_tags(text: str) -> str:
    # Broad tag stripper for fallback mode only.
    # Replaces tags with spaces to avoid accidental token concatenation.
    return re.sub(r"<[^>]+>", " ", text)


def _normalize_metadata_key(raw_key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", raw_key.casefold())
    return normalized.strip("_")


def _get_metadata_image_reference(metadata: dict[str, str]) -> str | None:
    for key in ("image", "source_file", "source_image", "image_path"):
        value = metadata.get(key)
        if not value:
            continue
        cleaned = value.strip().strip("`").strip()
        if cleaned and cleaned != "-":
            return cleaned
    return None


def _heading_tokens(heading: str) -> set[str]:
    return {token for token in _WORD_RE.findall(heading.casefold())}


def _score_section_heading(heading: str) -> int:
    tokens = _heading_tokens(heading)
    # Prefer "raw model output" semantics, then XML payload sections.
    # Higher scores indicate higher confidence that section stores true source XML.
    if {"raw", "model", "output"} <= tokens:
        return 400
    if {"raw", "output"} <= tokens:
        return 320
    if {"extracted", "xml", "payload"} <= tokens:
        return 260
    if {"xml", "payload"} <= tokens:
        return 220
    if "xml" in tokens:
        return 180
    if {"system", "prompt"} <= tokens:
        return -60
    if {"plain", "text"} <= tokens:
        return -40
    return 0


def _extract_xml_fragment(text: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None

    # Remove outer markdown fences when a fenced block was passed in.
    # Remove an optional opening fence line like ```xml, ```XML, ```json, etc.
    candidate = re.sub(r"^\s*```(?:\w+)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```\s*$", "", candidate)

    # Prefer explicit HistoricalDocument blocks when available.
    match = _HISTORICAL_DOCUMENT_RE.search(candidate)
    if match:
        return match.group(1).strip()

    if candidate.lstrip().startswith("<") and "</" in candidate:
        return candidate.strip()

    return None


def _parse_xml_root(xml_payload: str) -> tuple[Any | None, bool, str | None]:
    xml_payload = xml_payload.strip()
    if not xml_payload:
        return None, False, "XML payload is empty."

    # 1) strict parse (fastest and most reliable for well-formed XML)
    try:
        return ET.fromstring(xml_payload), True, None
    except ET.ParseError as strict_error:
        # 2) targeted repair for truncated payloads missing closing HistoricalDocument tag
        repaired = _try_append_historicaldocument_close(xml_payload)
        if repaired is not None:
            try:
                return (
                    ET.fromstring(repaired),
                    False,
                    "XML was repaired by appending </HistoricalDocument>. "
                    f"Original parse error: {strict_error}",
                )
            except ET.ParseError:
                pass

        # 3) optional lxml recovery parser for malformed XML with salvageable structure
        if _LXML_ETREE is not None:
            try:
                parser = _LXML_ETREE.XMLParser(recover=True, huge_tree=True)
                root = _LXML_ETREE.fromstring(xml_payload.encode("utf-8"), parser=parser)
                if root is None:
                    return None, False, f"XML parse failed: {strict_error}"
                warning = (
                    "XML was not well-formed and was rendered using lxml recovery. "
                    f"Strict parse error: {strict_error}"
                )
                if getattr(parser, "error_log", None):
                    first = parser.error_log[0]
                    warning += " | lxml: {}".format(first.message)
                return root, False, warning
            except Exception as recover_error:
                return (
                    None,
                    False,
                    f"XML parse failed: {strict_error}. lxml recovery failed: {recover_error}",
                )

        # 4) unrecoverable
        return None, False, f"XML parse failed: {strict_error}"


def _try_append_historicaldocument_close(xml_payload: str) -> str | None:
    # Only repair when:
    # - an opening HistoricalDocument tag exists
    # - no closing HistoricalDocument tag exists
    if not _HISTORICAL_DOCUMENT_OPEN_RE.search(xml_payload):
        return None
    if _HISTORICAL_DOCUMENT_CLOSE_RE.search(xml_payload):
        return None

    open_match = _HISTORICAL_DOCUMENT_OPEN_RE.search(xml_payload)
    if not open_match:
        return None

    # Preserve namespace prefix: <ns:HistoricalDocument> -> </ns:HistoricalDocument>
    prefix = open_match.group("prefix") or ""
    return "{}\n</{}HistoricalDocument>\n".format(xml_payload.rstrip(), prefix)


def _children_named(element: Any, name: str) -> list[Any]:
    return [child for child in _child_elements(element) if _local_name(getattr(child, "tag", "")) == name]


def _first_child_named(element: Any, name: str) -> Any | None:
    children = _children_named(element, name)
    return children[0] if children else None


def _child_elements(element: Any) -> list[Any]:
    children: list[Any] = []
    for child in list(element):
        if _local_name(getattr(child, "tag", "")):
            children.append(child)
    return children


def _extract_page_number(page: Any) -> str | None:
    for node in page.iter():
        if _local_name(getattr(node, "tag", "")) != "PageNumber":
            continue
        text = _normalize_space(_iter_text(node))
        if text:
            return text
    return None


def _collect_line_texts(element: Any, *, recursive: bool) -> list[str]:
    lines: list[str] = []
    iterable = element.iter() if recursive else _child_elements(element)
    for node in iterable:
        if _local_name(getattr(node, "tag", "")) != "Line":
            continue
        text = _normalize_space(_iter_text(node))
        lines.append(text)
    return lines


def _iter_text(element: Any) -> str:
    try:
        return "".join(element.itertext())
    except Exception:
        return str(getattr(element, "text", "") or "")


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _slugify_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return token or "value"


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    if ":" in tag:
        return tag.split(":", 1)[-1]
    return tag


def _escape_or_nbsp(value: str) -> str:
    if not value:
        return "&nbsp;"
    return html.escape(value)


def _render_line_stack(lines: list[str], css_class: str) -> str:
    if not lines:
        return '<div class="{}">&nbsp;</div>'.format(html.escape(css_class))
    return "".join(
        '<div class="{css}">{value}</div>'.format(
            css=html.escape(css_class),
            value=_escape_or_nbsp(line),
        )
        for line in lines
    )


def _copy_stylesheet_to_output(stylesheet_source: Path, output_dir: Path) -> Path:
    if not stylesheet_source.exists() or not stylesheet_source.is_file():
        raise FileNotFoundError(f"Stylesheet file does not exist: {stylesheet_source}")
    stylesheet_target = output_dir / stylesheet_source.name
    stylesheet_target.write_text(stylesheet_source.read_text(encoding="utf-8"), encoding="utf-8")
    return stylesheet_target


def _relative_href(from_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, start=from_dir)).as_posix()


def _iter_source_image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _build_image_name_index(images_dir: Path) -> tuple[dict[str, Path], set[str]]:
    name_to_path: dict[str, Path] = {}
    duplicate_names: set[str] = set()
    for image_path in _iter_source_image_paths(images_dir):
        name_key = image_path.name.casefold()
        if name_key in name_to_path and name_to_path[name_key] != image_path:
            duplicate_names.add(name_key)
            continue
        name_to_path[name_key] = image_path
    return name_to_path, duplicate_names


def _copy_source_image_to_output(
    *,
    source_image_path: Path,
    output_dir: Path,
    images_dir: Path | None,
    copied_images_root: Path,
) -> Path:
    if images_dir is not None:
        try:
            relative = source_image_path.resolve().relative_to(images_dir.resolve())
        except ValueError:
            relative = Path(source_image_path.name)
    else:
        relative = Path(source_image_path.name)

    target_path = copied_images_root / relative
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        shutil.copy2(str(source_image_path), str(target_path))
    return target_path


def _resolve_source_image_path(
    *,
    metadata_image: str | None,
    markdown_path: Path | None,
    images_dir: Path | None,
    image_lookup: dict[str, Path],
    duplicate_image_names: set[str],
) -> Path | None:
    if images_dir is None:
        return None

    if metadata_image:
        image_value = metadata_image.strip().strip("`").strip()
        if image_value and image_value != "-":
            metadata_path = Path(image_value)
            if metadata_path.is_absolute():
                if metadata_path.exists() and metadata_path.is_file():
                    return metadata_path
            else:
                direct_candidate = images_dir / metadata_path
                if direct_candidate.exists() and direct_candidate.is_file():
                    return direct_candidate

                name_key = metadata_path.name.casefold()
                if name_key and name_key not in duplicate_image_names:
                    mapped = image_lookup.get(name_key)
                    if mapped is not None:
                        return mapped

    # Fallback: derive image name from markdown filename.
    # Example: 000122_siecle-imprime-16_bpt6k8711898c_f9.md -> siecle-imprime-16_bpt6k8711898c_f9.jpeg
    if markdown_path is not None:
        for candidate_name in _candidate_image_names_from_markdown(markdown_path):
            candidate_key = candidate_name.casefold()
            if candidate_key in duplicate_image_names:
                continue
            mapped = image_lookup.get(candidate_key)
            if mapped is not None:
                return mapped

    return None


def _candidate_image_names_from_markdown(markdown_path: Path) -> list[str]:
    stem = markdown_path.stem
    stems = []
    # Remove numeric prefix used by many benchmark-style outputs.
    stem_without_prefix = re.sub(r"^\d+_", "", stem)
    stems.append(stem_without_prefix)
    stems.append(stem)

    normalized_stems: list[str] = []
    seen_stems: set[str] = set()
    for base in stems:
        if not base:
            continue
        variants = [base]
        # Common result suffixes used by current pipelines.
        for suffix in ("_xml", "_pure_text"):
            if base.casefold().endswith(suffix):
                variants.append(base[: -len(suffix)])
        for variant in variants:
            clean = variant.strip()
            if not clean or clean in seen_stems:
                continue
            seen_stems.add(clean)
            normalized_stems.append(clean)

    candidates: list[str] = []
    for clean_stem in normalized_stems:
        for ext in sorted(SUPPORTED_IMAGE_EXTENSIONS):
            candidates.append("{}{}".format(clean_stem, ext))
    return candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render HistoricalDocument XML stored in markdown artifacts as standalone HTML pages."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_XML_RESULTS_DIR,
        help="Directory containing markdown files with XML payloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_HTML_OUTPUT_DIR,
        help="Directory where rendered HTML files will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of markdown files to render (0 means all).",
    )
    parser.add_argument(
        "--stylesheet",
        type=Path,
        default=DEFAULT_STYLESHEET_SOURCE,
        help="Path to stylesheet file used by generated HTML pages.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing source images. "
            "When provided, each rendered page includes an 'Open source image in new tab' link."
        ),
    )
    parser.add_argument(
        "--copy-images-into-output",
        action="store_true",
        help=(
            "Copy each paired source image into output_dir/_paired_images and link that copy. "
            "Useful when serving output_dir via a local web server."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts, index_path = render_xml_results_to_html(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        stylesheet_source=args.stylesheet,
        images_dir=args.images_dir,
        copy_images_into_output=args.copy_images_into_output,
    )
    recovered_count = sum(1 for item in artifacts if not item.xml_well_formed)
    print("Rendered {} HTML files.".format(len(artifacts)))
    print("Output directory: {}".format(args.output_dir))
    print("Index page: {}".format(index_path))
    print(
        "XML status -> well-formed: {}, recovered/malformed: {}".format(
            len(artifacts) - recovered_count,
            recovered_count,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
