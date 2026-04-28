from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from runfile_records import safe_name


def build_score_stream_index(scores_pkl: Path) -> dict[str, dict]:
    """Build a lightweight fname->offset index for a scores.pkl stream."""
    lookup: dict[str, dict] = {}
    with open(scores_pkl, "rb") as f:
        stream_index = 0
        while True:
            offset = f.tell()
            try:
                item = pickle.load(f)
            except EOFError:
                break

            if not isinstance(item, dict):
                stream_index += 1
                continue

            fname = Path(str(item.get("fname", f"item_{stream_index:04d}"))).name
            if fname in lookup:
                raise ValueError(f"Duplicate fname in scores.pkl stream: {fname!r}")

            has_pred = "pred" in item
            has_ref = "ref" in item
            lookup[fname] = {
                "stream_index": int(stream_index),
                "offset": int(offset),
                "fname": str(fname),
                "has_pred": bool(has_pred),
                "has_ref": bool(has_ref),
                "pred": str(item.get("pred", "")) if has_pred else None,
                "ref": str(item.get("ref", "")) if has_ref else None,
            }
            stream_index += 1
    return lookup


def score_index_cache_file(scores_pkl: Path, cache_dir: Path) -> Path:
    resolved = str(scores_pkl.resolve())
    path_hash = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{safe_name(scores_pkl.name)}.{path_hash}.index.pkl"


def build_score_stream_index_cached(scores_pkl: Path, cache_dir: Path) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = score_index_cache_file(scores_pkl, cache_dir)

    try:
        stat = scores_pkl.stat()
    except OSError:
        return build_score_stream_index(scores_pkl)

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                cache_path = str(payload.get("scores_pkl_path", ""))
                cache_size = int(payload.get("scores_pkl_size", -1))
                cache_mtime_ns = int(payload.get("scores_pkl_mtime_ns", -1))
                cache_index = payload.get("index", {})
                if (
                    cache_path == str(scores_pkl.resolve())
                    and cache_size == int(stat.st_size)
                    and cache_mtime_ns == int(stat.st_mtime_ns)
                    and isinstance(cache_index, dict)
                ):
                    return cache_index
        except Exception:
            pass

    index = build_score_stream_index(scores_pkl)
    payload = {
        "scores_pkl_path": str(scores_pkl.resolve()),
        "scores_pkl_size": int(stat.st_size),
        "scores_pkl_mtime_ns": int(stat.st_mtime_ns),
        "index": index,
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return index


def load_score_item_by_offset(scores_pkl: Path, offset: int) -> dict:
    with open(scores_pkl, "rb") as f:
        f.seek(int(offset))
        item = pickle.load(f)
    if not isinstance(item, dict):
        raise ValueError(f"Expected dict record at offset {offset} in {scores_pkl}")
    return item


def load_run_items_from_score_index(score_index: dict[str, dict], scores_pkl: Path) -> list[dict]:
    out: list[dict] = []
    ordered_items = sorted(score_index.values(), key=lambda entry: int(entry["stream_index"]))
    for item in ordered_items:
        if not item.get("has_pred", False) or not item.get("has_ref", False):
            raise ValueError(
                "scores.pkl item is missing pred/ref text and cannot replace runfile-json: "
                f"{item['fname']!r} in {scores_pkl}"
            )
        out.append(
            {
                "index": int(item["stream_index"]),
                "fname": Path(str(item["fname"])).name,
                "pred": str(item["pred"]),
                "ref": str(item["ref"]),
            }
        )
    return out
