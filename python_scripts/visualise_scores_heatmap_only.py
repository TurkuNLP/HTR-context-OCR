#!/usr/bin/env python3

import argparse
from pathlib import Path
import pickle
import re

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


def safe_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]


def safe_matrix(scores) -> np.ndarray:
    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros((1, 1), dtype=float)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Visualise chrF score matrices as simple heatmaps (no line detection). "
            "Reads a scores.pkl pickle-stream produced by compare.py or compare_aligned_texts.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--img-dir",
        type=Path,
        default=Path("../../dorian/churro_finnish_dataset/dataset_splits/dev/"),
        help="Directory containing document images. Files are looked up by the `fname` stored in scores.pkl.",
    )
    p.add_argument(
        "--scores-pkl",
        type=Path,
        required=True,
        help="Path to the progressively-pickled score matrices (pickle stream).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results/visualise_scores_heatmap_only",
        help="Output directory. Subfolders will be created under this path.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit the number of loaded items (None means all).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() per item (useful in notebooks; typically disabled for batch runs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    img_dir = args.img_dir
    scores_pkl = args.scores_pkl

    results_dir = args.results_dir
    full_dir = results_dir / "full_figures"
    graph_dir = results_dir / "graph_only"

    for out_dir in (full_dir, graph_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

    if not scores_pkl.exists():
        raise FileNotFoundError(f"Missing scores file: {scores_pkl}")

    saved_idx = 0
    seen = 0

    with open(scores_pkl, "rb") as f:
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break

            if args.max_items is not None and seen >= args.max_items:
                break
            seen += 1

            fname = item.get("fname") or f"item_{seen:04d}"
            matrix = safe_matrix(item.get("scores"))

            fig, (ax_img, ax_hm) = plt.subplots(1, 2, figsize=(12, 5))

            img_path = img_dir / fname
            if img_path.exists():
                ax_img.imshow(plt.imread(img_path))
                ax_img.set_title(fname)
                ax_img.axis("off")
            else:
                ax_img.text(0.5, 0.5, f"Missing image:\n{fname}", ha="center", va="center")
                ax_img.axis("off")

            im = ax_hm.imshow(matrix, aspect="auto", cmap="Greys")
            cbar = plt.colorbar(im, ax=ax_hm, label="chrF")
            ax_hm.set_xlabel("pred segment")
            ax_hm.set_ylabel("ref segment")
            ax_hm.set_title("chrF heatmap")
            plt.tight_layout()

            base_name = safe_name(fname)
            full_out = full_dir / f"{saved_idx:04d}_{base_name}_full.png"
            graph_out = graph_dir / f"{saved_idx:04d}_{base_name}_graph.png"

            fig.savefig(full_out, dpi=220, bbox_inches="tight", facecolor="white")

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            graph_bbox = Bbox.union(
                [ax_hm.get_tightbbox(renderer), cbar.ax.get_tightbbox(renderer)]
            ).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(graph_out, dpi=260, bbox_inches=graph_bbox, facecolor="white")

            print(f"Saved: {full_out}")
            print(f"Saved: {graph_out}")

            if args.show:
                plt.show()
            plt.close(fig)
            saved_idx += 1

    print(f"Done. Saved {saved_idx} items to: {results_dir}")


if __name__ == "__main__":
    main()
