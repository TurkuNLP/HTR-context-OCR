from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

DATASET_ID = "stanford-oval/churro-dataset"
SPLITS = ("train", "dev", "test")
LANGUAGE_FILTER = [("main_language", "=", "Finnish")]
OUTPUT_ROOT = Path("/scratch/project_2017385/dorian/churro_finnish_dataset")


def _unique_output_path(split_dir: Path, source_name: str, index: int) -> Path:
    """Return a split-local file path, avoiding collisions."""
    base_name = Path(source_name).name if source_name else f"{split_dir.name}_{index}.png"
    if not Path(base_name).suffix:
        base_name = f"{base_name}.png"

    candidate = split_dir / base_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    serial = 1
    while True:
        candidate = split_dir / f"{stem}_{serial}{suffix}"
        if not candidate.exists():
            return candidate
        serial += 1


def _format_field_value(key: str, value) -> str:
    """Render a sample field in a compact markdown-friendly form."""
    if key == "image":
        # The binary image payload is already saved to disk; avoid duplicating it in markdown.
        return "<image payload omitted; see saved image file>"

    if value is None:
        return "null"

    if isinstance(value, (str, int, float, bool)):
        return str(value)

    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return "\n".join(f"- {item}" for item in value)

    return json.dumps(value, ensure_ascii=False, indent=2)


def _write_fields_markdown(
    sample: dict,
    output_image_path: Path,
    split: str,
    split_dir: Path,
    field_order: Iterable[str] | None = None,
) -> Path:
    ordered_keys = list(field_order) if field_order is not None else list(sample.keys())

    md_path = split_dir / f"{output_image_path.stem}_fields.md"
    lines = [
        f"# CHURRO Fields: `{output_image_path.name}`",
        "",
        f"- Split: `{split}`",
        f"- Saved image: `{output_image_path}`",
        "",
    ]

    for key in ordered_keys:
        value = _format_field_value(key, sample.get(key))
        lines.append(f"## `{key}`")
        lines.append("")
        if "\n" in value:
            lines.append(value)
        else:
            lines.append(f"`{value}`")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def export_finnish_split(split: str, output_root: Path) -> dict[str, int]:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = split_dir / "manifest.jsonl"
    saved = 0
    skipped = 0

    dataset = load_dataset(
        DATASET_ID,
        split=split,
        streaming=True,
        filters=LANGUAGE_FILTER,
    )

    field_order = list(dataset.features.keys()) if dataset.features is not None else None

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, sample in enumerate(dataset):
            image = sample.get("image")
            if image is None:
                skipped += 1
                continue

            output_path = _unique_output_path(
                split_dir=split_dir,
                source_name=sample.get("file_name", ""),
                index=index,
            )
            image.save(output_path)
            fields_md_path = _write_fields_markdown(
                sample=sample,
                output_image_path=output_path,
                split=split,
                split_dir=split_dir,
                field_order=field_order,
            )

            manifest_record = {
                "split": split,
                "source_file_name": sample.get("file_name"),
                "saved_path": str(output_path),
                "fields_markdown_path": str(fields_md_path),
                "main_language": sample.get("main_language"),
                "document_type": sample.get("document_type"),
                "dataset_id": sample.get("dataset_id"),
            }
            manifest.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
            saved += 1

    return {"saved": saved, "skipped_no_image": skipped}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        print(f"Processing split '{split}' with filter {LANGUAGE_FILTER}...")
        split_summary = export_finnish_split(split=split, output_root=OUTPUT_ROOT)
        summary[split] = split_summary
        print(
            f"Finished split '{split}': saved={split_summary['saved']}, "
            f"skipped_no_image={split_summary['skipped_no_image']}"
        )

    total_saved = sum(item["saved"] for item in summary.values())
    total_skipped = sum(item["skipped_no_image"] for item in summary.values())

    full_summary = {
        "dataset": DATASET_ID,
        "filter": LANGUAGE_FILTER,
        "output_root": str(OUTPUT_ROOT),
        "splits": summary,
        "totals": {
            "saved": total_saved,
            "skipped_no_image": total_skipped,
        },
    }

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(full_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary written to: {summary_path}")
    print(json.dumps(full_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
