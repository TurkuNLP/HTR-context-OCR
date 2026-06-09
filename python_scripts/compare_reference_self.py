from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import os
import pickle

import numpy as np
import sacrebleu
import tqdm


def compare(text1, text2, args):
    """Sliding window all-against-all comparison of two texts using chrF.

    Returns a score matrix of all segments against all segments in text1 by text2.
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


def all_comparisons(loaded_run_json, args):
    max_items = args.max_items

    total_items = None
    try:
        total_items = len(loaded_run_json)
    except Exception:
        total_items = None

    pbar_total = total_items
    if max_items is not None:
        if max_items <= 0:
            raise ValueError("--max-items must be a positive integer")
        pbar_total = max_items if pbar_total is None else min(pbar_total, max_items)

    with open(args.output, "wb") as f:
        for idx, img in enumerate(tqdm.tqdm(loaded_run_json, unit="img", total=pbar_total)):
            if max_items is not None and idx >= max_items:
                break
            fname = os.path.basename(img["file_name"])
            ref = img["normalized_gold_text"]
            pred = img["normalized_gold_text"]
            scores = compare(ref, pred, args)
            pickle.dump({"fname": fname, "scores": scores, "ref": ref, "pred": pred}, f)


def parse_args():
    p = argparse.ArgumentParser(
        description="Sliding window character n-gram self-comparison of reference text using chrF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--window-size", type=int, default=100, help="Size of sliding window in characters"
    )
    p.add_argument(
        "--window-stride", type=int, default=50, help="Stride between window positions"
    )
    p.add_argument(
        "--runfile-json",
        default=str(REPO_ROOT / "results" / "custom_churro_infer_dev_run1" / "vllm" / "dev" / "outputs.json"),
        help="Path to run results JSON file",
    )
    p.add_argument(
        "--output",
        default="scores_reference_self.pkl",
        help="Progressively pickled comparison score matrices",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Process only the first N entries from the runfile JSON",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_comparisons(json.load(open(args.runfile_json)), args)
