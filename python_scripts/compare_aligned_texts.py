
import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import sacrebleu
import tqdm


def safe_name(name: str) -> str:
    """
    Normalize filename stem to a filesystem-safe token.
    Keeps only [A-Za-z0-9._-], replaces other chars with "_", and truncates.
    Mirrors align_graph_text_blocks.py's behavior so keys match.
    """
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]


def compare(text1: str, text2: str, args) -> np.ndarray:
    """
    Sliding window all-against-all comparison of two texts using chrF.

    Returns a score matrix of all segments in text1 against all segments in text2.
    Logic is intentionally kept identical to compare.py for compatibility.
    """
    segments1 = [
        text1[i : i + args.window_size]
        for i in range(0, len(text1) - args.window_size + 1, args.window_stride)
    ]
    segments2 = [
        text2[i : i + args.window_size]
        for i in range(0, len(text2) - args.window_size + 1, args.window_stride)
    ]
    scores = np.zeros((len(segments1), len(segments2)))
    print("scores matrix size: ", scores.shape)
    total = len(segments1) * len(segments2)
    with tqdm.tqdm(total=total, unit="cmp") as pbar:
        for i in range(len(segments1)):
            for j in range(len(segments2)):
                score = sacrebleu.sentence_chrf(segments1[i], [segments2[j]])
                scores[i, j] = score.score
                pbar.update(1)
    return scores


def adjusted_txt_to_base_id(txt_path: Path) -> str:
    """
    Convert an adjusted prediction text filename into the base id used by runfile entries.

    Examples:
      0002_europeana_00674548_full_adjusted_pred.txt -> europeana_00674548
      newseye-fin_..._graph_adjusted_pred.txt       -> newseye-fin_...
    """
    stem = txt_path.stem
    if stem.endswith("_adjusted_pred"):
        stem = stem[: -len("_adjusted_pred")]

    m = re.match(r"^\d{4}_(.+)$", stem)
    if m:
        stem = m.group(1)

    stem = re.sub(r"_(graph|full)$", "", stem)
    return stem


def build_runfile_index(loaded_run_json: list[dict]) -> dict[str, dict]:
    """
    Index runfile entries by the same safe-name key used in alignment outputs.
    """
    index: dict[str, dict] = {}
    for rec in loaded_run_json:
        file_name = rec.get("file_name")
        if not file_name:
            continue
        key = safe_name(Path(file_name).name)
        # Keep the first occurrence if there are collisions.
        index.setdefault(key, rec)
    return index


def all_comparisons_aligned_txts(loaded_run_json: list[dict], args) -> None:
    run_index = build_runfile_index(loaded_run_json)
    aligned_dir = Path(args.aligned_dir)

    txt_paths = sorted(aligned_dir.glob(args.txt_glob))
    if not txt_paths:
        raise FileNotFoundError(f"No files matched {aligned_dir}/{args.txt_glob}")

    written = 0
    with open(args.output, "wb") as f:
        for txt_path in tqdm.tqdm(txt_paths, unit="txt"):
            base_id = adjusted_txt_to_base_id(txt_path)
            key = safe_name(base_id)
            rec = run_index.get(key)
            if rec is None:
                msg = (
                    f"No runfile entry found for {txt_path.name} "
                    f"(base_id={base_id!r}, key={key!r})"
                )
                if args.allow_missing:
                    print("WARNING:", msg)
                    continue
                raise KeyError(msg)

            ref = rec.get("normalized_gold_text")
            if ref is None:
                raise KeyError(
                    f"Missing normalized_gold_text for runfile entry: {rec.get('file_name')}"
                )

            pred_aligned = txt_path.read_text(encoding="utf-8", errors="replace")
            scores = compare(ref, pred_aligned, args)
            pickle.dump(
                {
                    "fname": os.path.basename(rec["file_name"]),
                    "aligned_txt": str(txt_path),
                    "key": key,
                    "scores": scores,
                    "ref": ref,
                    "pred": pred_aligned,
                },
                f,
            )
            written += 1

    print(f"Wrote {written} comparison matrices to: {args.output}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Sliding window character n-gram comparison using chrF, but using aligned "
            "prediction texts from *_adjusted_pred.txt files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--window-size", type=int, default=100, help="Size of sliding window in characters")
    p.add_argument("--window-stride", type=int, default=50, help="Stride between window positions")
    p.add_argument(
        "--runfile-json",
        default="../../dorian/churro_finnish_dataset/run_results/dev_split/outputs.json",
        help="Path to run results JSON file (source of normalized_gold_text)",
    )
    p.add_argument(
        "--aligned-dir",
        default="/scratch/project_2017385/dorian/Churro_copy/results/aligned_text_blocks_two_cases",
        help="Directory containing *_adjusted_pred.txt files",
    )
    p.add_argument("--txt-glob", default="*_adjusted_pred.txt", help="Glob of aligned .txt files to compare")
    p.add_argument("--output", default="aligned_scores.pkl", help="Progressively pickled comparison score matrices")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip .txt files that cannot be matched to a runfile entry (instead of erroring)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_comparisons_aligned_txts(json.load(open(args.runfile_json, "r", encoding="utf-8")), args)