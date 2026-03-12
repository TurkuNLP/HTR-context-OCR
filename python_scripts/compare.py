import argparse
import numpy as np
import sacrebleu
import json
import os
import tqdm
import pickle

def compare(text1, text2, args):
    """
    Sliding window all-against-all comparison of two texts using chrf

    returns a score matrix of all segments against all segments in text1 by text2
    """
    segments1 = [text1[i:i+args.window_size] for i in range(0, len(text1)-args.window_size+1, args.window_stride)]
    segments2 = [text2[i:i+args.window_size] for i in range(0, len(text2)-args.window_size+1, args.window_stride)]
    scores=np.zeros((len(segments1), len(segments2)))
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
    with open(args.output, "wb") as f:
        for img in tqdm.tqdm(loaded_run_json, unit="img"):
            fname=os.path.basename(img['file_name'])
            ref=img['normalized_gold_text']
            pred=img['normalized_predicted_text']
            scores=compare(ref, pred, args)
            pickle.dump({"fname": fname, "scores": scores, "ref": ref, "pred": pred}, f)

def parse_args():
    p = argparse.ArgumentParser(
        description="Sliding window character n-gram comparison using chrF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--window-size", type=int, default=100,
                   help="Size of sliding window in characters")
    p.add_argument("--window-stride", type=int, default=50,
                   help="Stride between window positions")
    p.add_argument("--runfile-json", default="/scratch/project_2017385/dorian/churro_finnish_dataset/run_results/dev_split/outputs.json",
                   help="Path to run results JSON file")
    p.add_argument("--output", default="scores.pkl",
                   help="Progressively pickled comparison score matrices")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    all_comparisons(json.load(open(args.runfile_json)), args)
