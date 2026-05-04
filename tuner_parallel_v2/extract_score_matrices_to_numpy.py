#!/usr/bin/env python3
"""Extract selected precomputed score matrices from a score-stream pickle.

This is a one-off helper for exporting a few matrices as standalone .npy files.
It intentionally reuses the existing score-stream index loader so we do not
invent a second pickle parser here.
"""

import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEXT_METRICS_DIR = PROJECT_ROOT / "text_metrics_v2_1_parallel"

# Make the local helper modules importable when the script is executed directly.
for path in (SCRIPT_DIR, TEXT_METRICS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from score_matrix_builder import coerce_score_matrix  # type: ignore
from score_stream_index import build_score_stream_index, load_score_item_by_offset  # type: ignore


PKL_PATH = PROJECT_ROOT / "results/compares_churro_dev/ref_to_pred/scores_reference_prediction_ws50_st35.pkl"
OUTPUT_DIR = SCRIPT_DIR / "extracted_numpy_arrays"
WINDOW_SIZE = 50
WINDOW_STRIDE = 35

# Match by stem so the script works whether the pickle stores ".jpeg" suffixes
# or not.
TARGET_DOC_STEMS = {
    "slovensky_2386879_0113_70957868",
    "slovensky_2386879_0108_70957846",
    "ahisto_1069_68",
    "ahisto_822_189",
}


def matrix_output_name(doc_stem):
    return f"{doc_stem}_ws{WINDOW_SIZE}_st{WINDOW_STRIDE}.npy"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the offset index once, then read only the records we care about.
    score_index = build_score_stream_index(PKL_PATH)

    matches = {}
    for fname, meta in score_index.items():
        stem = Path(str(fname)).stem
        if stem in TARGET_DOC_STEMS:
            matches[stem] = meta

    missing = sorted(TARGET_DOC_STEMS - set(matches))
    if missing:
        raise SystemExit("Missing documents in pkl index: {}".format(", ".join(missing)))

    manifest_documents = []
    for doc_stem in sorted(TARGET_DOC_STEMS):
        meta = matches[doc_stem]
        raw = load_score_item_by_offset(PKL_PATH, int(meta["offset"]))
        if not isinstance(raw, dict):
            raise SystemExit("Unexpected non-dict record for {}".format(doc_stem))

        raw_fname = str(raw.get("fname", ""))
        if Path(raw_fname).stem != doc_stem:
            raise SystemExit(
                "Record fname mismatch for {}: expected stem {!r}, got {!r}".format(
                    doc_stem, doc_stem, raw_fname
                )
            )

        if "scores" not in raw:
            raise SystemExit("Missing scores field for {}".format(doc_stem))

        scores = coerce_score_matrix(raw.get("scores"), source_desc="{}:{}".format(PKL_PATH, raw_fname))
        out_path = OUTPUT_DIR / matrix_output_name(doc_stem)
        np.save(out_path, scores)

        manifest_documents.append(
            {
                "doc_name": doc_stem,
                "source_fname": raw_fname,
                "matrix_file": out_path.name,
                "shape": list(scores.shape),
                "dtype": str(scores.dtype),
            }
        )

        print("saved {}: shape={}, dtype={}, path={}".format(doc_stem, scores.shape, scores.dtype, out_path))

    manifest = {
        "source_pkl": str(PKL_PATH),
        "output_dir": str(OUTPUT_DIR),
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "documents": manifest_documents,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote manifest: {}".format(manifest_path))


if __name__ == "__main__":
    main()
