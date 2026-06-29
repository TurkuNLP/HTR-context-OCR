#!/usr/bin/env python3
from __future__ import annotations

"""Recover per-language stitched contact sheets from saved document panels.

Replicates tuner_simple_alpha_sweep_pre_iou_levenshtein/plotting/
stitched_language_panels.py exactly (same border/gap/colours, same output name
``stitched_best_combination_<language>_documents.png``) so the result is
identical to what the pipeline would have produced at the end of a run.

Unlike the pipeline's finish() step, this does NOT delete the temporary panels.

Usage:
  stitch_language_panels.py --panels-dir <.temporary_document_panels> \
      --plots-dir <plots dir> [--panel-columns 6] [--world-readable]
"""

import argparse
import math
import os
import stat
from pathlib import Path

from PIL import Image

# --- constants copied verbatim from the original module ---
STITCHED_PANEL_BORDER_PIXELS = 8
STITCHED_PANEL_GAP_PIXELS = 10
STITCHED_PANEL_BORDER_COLOR = (30, 30, 30, 255)
STITCHED_PANEL_BACKGROUND_COLOR = (255, 255, 255, 255)


def build_bordered_document_tile(*, panel_image, image_module, max_panel_width, max_panel_height):
    """Return one document panel centered inside a visible border tile."""
    border_pixels = int(STITCHED_PANEL_BORDER_PIXELS)
    tile_width = int(max_panel_width) + 2 * border_pixels
    tile_height = int(max_panel_height) + 2 * border_pixels
    bordered_tile = image_module.new("RGBA", (tile_width, tile_height), STITCHED_PANEL_BORDER_COLOR)
    inner_background = image_module.new("RGBA", (int(max_panel_width), int(max_panel_height)), STITCHED_PANEL_BACKGROUND_COLOR)
    bordered_tile.alpha_composite(inner_background, (border_pixels, border_pixels))
    panel_x = border_pixels + max(0, (int(max_panel_width) - int(panel_image.width)) // 2)
    panel_y = border_pixels + max(0, (int(max_panel_height) - int(panel_image.height)) // 2)
    bordered_tile.alpha_composite(panel_image, (panel_x, panel_y))
    inner_background.close()
    return bordered_tile


def save_stitched_language_image(*, panel_paths, stitched_output_path, panel_columns, image_module):
    """Paste document panel PNGs into one language-level contact sheet."""
    if not panel_paths:
        return None
    opened_images = []
    try:
        for panel_path in panel_paths:
            opened_images.append(image_module.open(panel_path).convert("RGBA"))
        max_panel_width = max(image.width for image in opened_images)
        max_panel_height = max(image.height for image in opened_images)
        safe_panel_columns = max(1, int(panel_columns))
        row_count = int(math.ceil(len(opened_images) / safe_panel_columns))
        bordered_tile_width = int(max_panel_width) + 2 * int(STITCHED_PANEL_BORDER_PIXELS)
        bordered_tile_height = int(max_panel_height) + 2 * int(STITCHED_PANEL_BORDER_PIXELS)
        gap_pixels = int(STITCHED_PANEL_GAP_PIXELS)
        stitched_width = safe_panel_columns * bordered_tile_width + max(0, safe_panel_columns - 1) * gap_pixels
        stitched_height = row_count * bordered_tile_height + max(0, row_count - 1) * gap_pixels
        stitched_image = image_module.new("RGBA", (stitched_width, stitched_height), STITCHED_PANEL_BACKGROUND_COLOR)
        try:
            for panel_index, panel_image in enumerate(opened_images):
                column_index = panel_index % safe_panel_columns
                row_index = panel_index // safe_panel_columns
                upper_left_corner = (
                    column_index * (bordered_tile_width + gap_pixels),
                    row_index * (bordered_tile_height + gap_pixels),
                )
                bordered_tile = build_bordered_document_tile(
                    panel_image=panel_image,
                    image_module=image_module,
                    max_panel_width=int(max_panel_width),
                    max_panel_height=int(max_panel_height),
                )
                try:
                    stitched_image.alpha_composite(bordered_tile, upper_left_corner)
                finally:
                    bordered_tile.close()
            stitched_output_path.parent.mkdir(parents=True, exist_ok=True)
            stitched_image.save(stitched_output_path, optimize=True)
        finally:
            stitched_image.close()
    finally:
        for image in opened_images:
            image.close()
    return stitched_output_path


def make_world_readable(path: Path) -> None:
    """Add other-read to a file (or other-read+execute to a directory)."""
    current = stat.S_IMODE(path.stat().st_mode)
    if path.is_dir():
        path.chmod(current | stat.S_IROTH | stat.S_IXOTH)
    else:
        path.chmod(current | stat.S_IROTH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch per-language document panels into contact sheets.")
    parser.add_argument("--panels-dir", type=Path, required=True,
                        help="The .temporary_document_panels directory containing per-language subfolders.")
    parser.add_argument("--plots-dir", type=Path, default=None,
                        help="Where to write the stitched PNGs (default: parent of --panels-dir).")
    parser.add_argument("--panel-columns", type=int, default=6)
    parser.add_argument("--world-readable", action="store_true",
                        help="chmod the stitched PNGs and directory chain so all users can read them.")
    args = parser.parse_args()

    panels_dir = args.panels_dir
    plots_dir = args.plots_dir if args.plots_dir is not None else panels_dir.parent
    plots_dir.mkdir(parents=True, exist_ok=True)

    language_dirs = sorted(d for d in panels_dir.iterdir() if d.is_dir())
    print(f"[stitch] {len(language_dirs)} language folder(s) under {panels_dir}")

    written: list[Path] = []
    for language_dir in language_dirs:
        language_name = language_dir.name
        panel_paths = sorted(language_dir.glob("*.png"))
        if not panel_paths:
            print(f"[stitch] {language_name}: no panels, skipped")
            continue
        stitched_path = plots_dir / f"stitched_best_combination_{language_name}_documents.png"
        saved = save_stitched_language_image(
            panel_paths=panel_paths,
            stitched_output_path=stitched_path,
            panel_columns=int(args.panel_columns),
            image_module=Image,
        )
        if saved is not None:
            written.append(saved)
            if args.world_readable:
                make_world_readable(saved)
            print(f"[stitch] {language_name}: {len(panel_paths)} panels -> {saved.name}")

    if args.world_readable:
        # Make the directory chain inside this result tree traversable by all users.
        seen = set()
        for directory in [plots_dir, *plots_dir.parents]:
            if directory in seen:
                continue
            seen.add(directory)
            try:
                make_world_readable(directory)
            except PermissionError:
                pass
            if directory.name == "results":
                break

    print(f"[stitch] done: {len(written)} stitched image(s) in {plots_dir}")


if __name__ == "__main__":
    main()
