#!/usr/bin/env python3
from __future__ import annotations

"""Stitch per-document score-matrix distribution PNGs into one image per language."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


BACKGROUND_COLOR = (255, 255, 255, 255)
BORDER_COLOR = (30, 30, 30, 255)
BORDER_PIXELS = 8
GAP_PIXELS = 10


def safe_path_component(value: str) -> str:
    """Return a filesystem-safe filename component."""

    cleaned = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in str(value))
    return cleaned.strip("._") or "unknown"


def sorted_png_paths(language_dir: Path) -> list[Path]:
    """Return document PNGs in stable filename order."""

    return sorted(Path(language_dir).glob("*.png"), key=lambda path: path.name.lower())


def load_panel_image(image_module: Any, panel_path: Path, *, max_panel_width: int | None) -> Any:
    """Open one panel as RGBA and optionally downscale it."""

    image = image_module.open(panel_path).convert("RGBA")
    if max_panel_width is not None and int(max_panel_width) > 0 and image.width > int(max_panel_width):
        scale = float(max_panel_width) / float(image.width)
        resized_size = (int(max_panel_width), max(1, int(round(image.height * scale))))
        resized = image.resize(resized_size, resample=image_module.Resampling.LANCZOS)
        image.close()
        image = resized
    return image


def build_bordered_tile(*, image_module: Any, panel_image: Any, max_panel_width: int, max_panel_height: int) -> Any:
    """Place one panel on a uniform bordered tile."""

    tile_width = int(max_panel_width) + 2 * BORDER_PIXELS
    tile_height = int(max_panel_height) + 2 * BORDER_PIXELS
    tile = image_module.new("RGBA", (tile_width, tile_height), BORDER_COLOR)
    inner = image_module.new("RGBA", (int(max_panel_width), int(max_panel_height)), BACKGROUND_COLOR)
    try:
        tile.alpha_composite(inner, (BORDER_PIXELS, BORDER_PIXELS))
        panel_x = BORDER_PIXELS + max(0, (int(max_panel_width) - int(panel_image.width)) // 2)
        panel_y = BORDER_PIXELS + max(0, (int(max_panel_height) - int(panel_image.height)) // 2)
        tile.alpha_composite(panel_image, (panel_x, panel_y))
    finally:
        inner.close()
    return tile


def stitch_language_panels(
    *,
    language_name: str,
    panel_paths: list[Path],
    output_path: Path,
    panel_columns: int,
    max_panel_width: int | None,
) -> dict[str, Any] | None:
    """Create one stitched language image and return manifest metadata."""

    if not panel_paths:
        return None

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    opened_images = []
    try:
        for panel_path in panel_paths:
            opened_images.append(load_panel_image(Image, panel_path, max_panel_width=max_panel_width))

        max_width = max(image.width for image in opened_images)
        max_height = max(image.height for image in opened_images)
        safe_columns = max(1, int(panel_columns))
        row_count = int(math.ceil(len(opened_images) / safe_columns))
        tile_width = int(max_width) + 2 * BORDER_PIXELS
        tile_height = int(max_height) + 2 * BORDER_PIXELS
        stitched_width = safe_columns * tile_width + max(0, safe_columns - 1) * GAP_PIXELS
        stitched_height = row_count * tile_height + max(0, row_count - 1) * GAP_PIXELS
        stitched_image = Image.new("RGBA", (stitched_width, stitched_height), BACKGROUND_COLOR)
        try:
            for panel_index, panel_image in enumerate(opened_images):
                column_index = panel_index % safe_columns
                row_index = panel_index // safe_columns
                x = column_index * (tile_width + GAP_PIXELS)
                y = row_index * (tile_height + GAP_PIXELS)
                bordered_tile = build_bordered_tile(
                    image_module=Image,
                    panel_image=panel_image,
                    max_panel_width=max_width,
                    max_panel_height=max_height,
                )
                try:
                    stitched_image.alpha_composite(bordered_tile, (x, y))
                finally:
                    bordered_tile.close()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stitched_image.save(output_path, optimize=True)
        finally:
            stitched_image.close()
    finally:
        for image in opened_images:
            image.close()

    return {
        "main_language": language_name,
        "document_plot_count": int(len(panel_paths)),
        "stitched_path": str(output_path),
        "stitched_width": int(stitched_width),
        "stitched_height": int(stitched_height),
        "panel_columns": int(max(1, int(panel_columns))),
        "max_panel_width": "" if max_panel_width is None else int(max_panel_width),
    }


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the stitched image manifest."""

    fieldnames = [
        "main_language",
        "document_plot_count",
        "stitched_path",
        "stitched_width",
        "stitched_height",
        "panel_columns",
        "max_panel_width",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create one stitched PNG per language from per-document score-matrix distribution plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--document-distributions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-columns", type=int, default=6)
    parser.add_argument("--max-panel-width", type=int, default=0, help="0 keeps original panel resolution.")
    parser.add_argument("--language", action="append", dest="languages", default=None, help="Optional language folder filter; repeatable.")
    return parser.parse_args()


def main() -> None:
    """Run the stitching workflow."""

    args = parse_arguments()
    source_dir = Path(args.document_distributions_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"document distribution directory does not exist: {source_dir}")
    if int(args.panel_columns) <= 0:
        raise ValueError("--panel-columns must be positive")

    language_filter = {str(language) for language in (args.languages or [])}
    output_dir = Path(args.output_dir)
    max_panel_width = None if int(args.max_panel_width) <= 0 else int(args.max_panel_width)
    manifest_rows: list[dict[str, Any]] = []

    language_dirs = [path for path in sorted(source_dir.iterdir(), key=lambda item: item.name.lower()) if path.is_dir()]
    for language_dir in language_dirs:
        language_name = language_dir.name
        if language_filter and language_name not in language_filter:
            continue
        panel_paths = sorted_png_paths(language_dir)
        if not panel_paths:
            continue
        output_path = output_dir / f"stitched_{safe_path_component(language_name)}_document_distributions.png"
        row = stitch_language_panels(
            language_name=language_name,
            panel_paths=panel_paths,
            output_path=output_path,
            panel_columns=int(args.panel_columns),
            max_panel_width=max_panel_width,
        )
        if row is not None:
            manifest_rows.append(row)
            print(f"wrote {output_path} documents={len(panel_paths)}")

    write_manifest_csv(output_dir / "stitched_language_distribution_manifest.csv", manifest_rows)
    summary = {
        "document_distributions_dir": str(source_dir),
        "output_dir": str(output_dir),
        "language_count": int(len(manifest_rows)),
        "stitched_image_count": int(len(manifest_rows)),
        "panel_columns": int(args.panel_columns),
        "max_panel_width": max_panel_width,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
