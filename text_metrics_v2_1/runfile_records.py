from __future__ import annotations

import json
import re
from pathlib import Path


def same_file(a: str, b: str) -> bool:
    return str(a) == str(b) or Path(str(a)).name == Path(str(b)).name


def safe_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]


def load_run_items(runfile_json: Path) -> list[dict]:
    data = json.loads(runfile_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected runfile JSON list, got: {type(data).__name__}")

    out: list[dict] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file_name", item.get("fname", f"item_{idx:04d}")))
        out.append(
            {
                "index": idx,
                "fname": Path(file_name).name,
                "pred": str(item.get("normalized_predicted_text", item.get("pred", ""))),
                "ref": str(item.get("normalized_gold_text", item.get("ref", ""))),
            }
        )
    return out
