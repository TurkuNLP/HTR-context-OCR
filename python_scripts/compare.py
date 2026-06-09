from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import glob
import json
import os
import pickle
import re

import numpy as np
import sacrebleu
import tqdm

from text_metrics_v2_12_parallel.levenshtein_metric import normalized_levenshtein_similarity

SINGLE_METRIC_NAMES = ("chrf", "bleu", "levenshtein")
COMPARISON_MODES = ("ref-pred", "ref-ref", "pred-pred")
COMPARISON_MODE_ALIASES = {
    "ref-pred": "ref-pred",
    "ref_pred": "ref-pred",
    "ref-to-pred": "ref-pred",
    "ref_to_pred": "ref-pred",
    "reference-prediction": "ref-pred",
    "reference_prediction": "ref-pred",
    "ref-ref": "ref-ref",
    "ref_ref": "ref-ref",
    "ref-to-ref": "ref-ref",
    "ref_to_ref": "ref-ref",
    "reference-reference": "ref-ref",
    "reference_reference": "ref-ref",
    "pred-pred": "pred-pred",
    "pred_pred": "pred-pred",
    "pred-to-pred": "pred-pred",
    "pred_to_pred": "pred-pred",
    "prediction-prediction": "pred-pred",
    "prediction_prediction": "pred-pred",
    "pre-pred": "pred-pred",
    "pre_pred": "pred-pred",
}


def metric_name(value):
    """Normalize and validate the metric name from the CLI."""
    metric = value.lower()
    if metric == "blue":
        metric = "bleu"
    if metric not in SINGLE_METRIC_NAMES and metric != "all":
        raise argparse.ArgumentTypeError(
            "metric must be one of: chrf, bleu, levenshtein, all"
        )
    return metric


def metric_names_to_run(metric):
    """Return the concrete metric names requested by the CLI."""
    if metric == "all":
        return SINGLE_METRIC_NAMES
    return (metric,)


def comparison_mode_name(value):
    """Normalize and validate which text pair is compared."""
    mode_key = str(value).strip().lower()
    mode = COMPARISON_MODE_ALIASES.get(mode_key)
    if mode is None:
        allowed_modes = ", ".join(COMPARISON_MODES)
        raise argparse.ArgumentTypeError(
            f"comparison mode must be one of: {allowed_modes}"
        )
    return mode


def select_comparison_texts(run_item, comparison_mode):
    """Return row/column texts and labels for the requested comparison mode."""
    gold_text = run_item["normalized_gold_text"]
    predicted_text = run_item["normalized_predicted_text"]

    if comparison_mode == "ref-pred":
        return gold_text, predicted_text, "reference", "prediction"
    if comparison_mode == "ref-ref":
        return gold_text, gold_text, "reference", "reference"
    if comparison_mode == "pred-pred":
        return predicted_text, predicted_text, "prediction", "prediction"
    raise ValueError(f"Unsupported comparison mode: {comparison_mode}")


def chrf_segment_score(reference_segment, predicted_segment):
    """Return the existing chrF score for one segment pair."""
    return sacrebleu.sentence_chrf(reference_segment, [predicted_segment]).score


def bleu_segment_score(reference_segment, predicted_segment):
    """Return the BLEU score for one prediction segment against one reference segment."""
    return sacrebleu.sentence_bleu(predicted_segment, [reference_segment]).score


def levenshtein_segment_score(reference_segment, predicted_segment):
    """Return normalized Levenshtein similarity on the same 0-100 scale as sacrebleu."""
    return normalized_levenshtein_similarity(predicted_segment, reference_segment) * 100.0


def get_segment_score_function(metric):
    """Choose the segment scoring function requested by the CLI."""
    if metric == "chrf":
        return chrf_segment_score
    if metric == "bleu":
        return bleu_segment_score
    if metric == "levenshtein":
        return levenshtein_segment_score
    raise ValueError(f"Unsupported metric: {metric}")


def compare(text1, text2, args, metric=None):
    """Sliding window all-against-all comparison of two texts.

    Returns a score matrix of all segments against all segments in text1 by text2.
    """
    selected_metric = args.metric if metric is None else metric

    segments1 = [
        text1[i : i + args.window_size]
        for i in range(0, len(text1) - args.window_size + 1, args.window_stride)
    ]
    segments2 = [
        text2[i : i + args.window_size]
        for i in range(0, len(text2) - args.window_size + 1, args.window_stride)
    ]
    segment_score = get_segment_score_function(selected_metric)
    scores = np.zeros((len(segments1), len(segments2)))
    print(f"{selected_metric} scores matrix size: ", scores.shape)
    total = len(segments1) * len(segments2)
    with tqdm.tqdm(total=total, unit="cmp") as pbar:
        for i in range(len(segments1)):
            for j in range(len(segments2)):
                scores[i, j] = segment_score(segments1[i], segments2[j])
                pbar.update(1)
    return scores


def output_path_for_metric(output_path, metric, metric_count):
    """Return the pickle path for one metric."""
    if metric_count == 1:
        return output_path

    path_without_extension, extension = os.path.splitext(output_path)
    if not extension:
        return f"{output_path}_{metric}"
    return f"{path_without_extension}_{metric}{extension}"


def default_graph_output_dir(output_path):
    """Return the default directory for score matrix graph PNGs."""
    output_dir = os.path.dirname(output_path) or "."
    return os.path.join(output_dir, "score_matrix_graphs")


def safe_graph_filename(file_name):
    """Return a short filesystem-safe name for graph files."""
    file_stem = os.path.splitext(os.path.basename(file_name))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", file_stem)
    return safe_stem[:120] or "document"


def display_name_for_graph(file_name, document_index):
    """Return a unique graph title that still shows the stored document name."""
    clean_name = str(file_name).strip() or "document"
    return f"{document_index + 1:04d}_{clean_name}"


def save_score_matrix_graph(
    file_name,
    scores_by_metric,
    graph_output_dir,
    document_index=0,
    row_text_role="reference",
    column_text_role="prediction",
):
    """Save one PNG showing the score matrix for each computed metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(graph_output_dir, exist_ok=True)
    metric_names = list(scores_by_metric)
    figure, axes = plt.subplots(
        1,
        len(metric_names),
        figsize=(5.0 * len(metric_names), 4.2),
        squeeze=False,
    )

    plotted_images = []
    for axis, metric in zip(axes[0], metric_names):
        score_matrix = np.asarray(scores_by_metric[metric], dtype=float)
        axis.set_title(metric)
        axis.set_xlabel(f"{column_text_role} window")
        axis.set_ylabel(f"{row_text_role} window")

        if score_matrix.size == 0 or score_matrix.shape[0] == 0 or score_matrix.shape[1] == 0:
            axis.text(
                0.5,
                0.5,
                f"Empty matrix\nshape={score_matrix.shape}",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            continue

        image = axis.imshow(
            score_matrix,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            vmin=0.0,
            vmax=100.0,
            )
        plotted_images.append(image)

    document_label = display_name_for_graph(file_name, document_index)
    figure.suptitle(document_label, fontsize=10)
    figure.subplots_adjust(left=0.06, right=0.88, bottom=0.12, top=0.82, wspace=0.32)
    if plotted_images:
        colorbar_axis = figure.add_axes([0.92, 0.18, 0.015, 0.58])
        figure.colorbar(
            plotted_images[-1],
            cax=colorbar_axis,
            label="score (0-100)",
        )

    metrics_label = "all_metrics" if len(metric_names) > 1 else metric_names[0]
    graph_path = os.path.join(
        graph_output_dir,
        f"{safe_graph_filename(document_label)}_{metrics_label}.png",
    )
    figure.savefig(graph_path, dpi=160)
    plt.close(figure)
    return graph_path


def all_comparisons(loaded_run_json, args):
    max_items = args.max_items
    comparison_mode = args.comparison_mode
    selected_metrics = metric_names_to_run(args.metric)
    output_paths_by_metric = {
        metric: output_path_for_metric(args.output, metric, len(selected_metrics))
        for metric in selected_metrics
    }
    should_save_graphs = args.graph_output_dir is not None or args.metric == "all"
    graph_output_dir = args.graph_output_dir or default_graph_output_dir(args.output)

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

    for output_path in output_paths_by_metric.values():
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    temporary_output_paths_by_metric = {
        metric: f"{output_path}.tmp"
        for metric, output_path in output_paths_by_metric.items()
    }
    output_files_by_metric = {
        metric: open(temporary_output_paths_by_metric[metric], "wb")
        for metric in selected_metrics
    }
    written_record_count = 0
    try:
        for idx, img in enumerate(tqdm.tqdm(loaded_run_json, unit="img", total=pbar_total)):
            if max_items is not None and idx >= max_items:
                break
            fname = os.path.basename(img["file_name"])
            # The runfile fields below are already normalized by the evaluation pipeline.
            original_ref = img["normalized_gold_text"]
            original_pred = img["normalized_predicted_text"]
            ref, pred, row_text_role, column_text_role = select_comparison_texts(
                img,
                comparison_mode,
            )
            scores_by_metric = {}
            for metric in selected_metrics:
                scores_by_metric[metric] = compare(ref, pred, args, metric=metric)
                pickle.dump(
                    {
                        "fname": fname,
                        "scores": scores_by_metric[metric],
                        "ref": ref,
                        "pred": pred,
                        "comparison_mode": comparison_mode,
                        "row_text_role": row_text_role,
                        "column_text_role": column_text_role,
                        "normalized_gold_text": original_ref,
                        "normalized_predicted_text": original_pred,
                        "metric": metric,
                        "window_size": args.window_size,
                        "window_stride": args.window_stride,
                    },
                    output_files_by_metric[metric],
                )
            written_record_count += 1

            if should_save_graphs:
                graph_path = save_score_matrix_graph(
                    fname,
                    scores_by_metric,
                    graph_output_dir,
                    document_index=idx,
                    row_text_role=row_text_role,
                    column_text_role=column_text_role,
                )
                print(f"saved score matrix graph: {graph_path}")
    finally:
        for output_file in output_files_by_metric.values():
            output_file.close()

    if written_record_count == 0:
        for temporary_output_path in temporary_output_paths_by_metric.values():
            if os.path.exists(temporary_output_path):
                os.remove(temporary_output_path)
        print("No records were written; no score pickle files were created.")
        return

    for metric, output_path in output_paths_by_metric.items():
        os.replace(temporary_output_paths_by_metric[metric], output_path)
        print(f"wrote {metric} scores to: {output_path}")


def load_score_pickle_records(score_pickle_path):
    """Load every record from one progressively pickled score stream."""
    records = []
    with open(score_pickle_path, "rb") as score_pickle:
        while True:
            try:
                records.append(pickle.load(score_pickle))
            except EOFError:
                break
    return records


def find_existing_score_pickle(score_pickle_dir, metric):
    """Find the non-empty score pickle for one metric in a directory."""
    candidates = sorted(glob.glob(os.path.join(score_pickle_dir, f"*_{metric}.pkl")))
    candidates = [path for path in candidates if os.path.getsize(path) > 0]
    if not candidates:
        raise FileNotFoundError(
            f"No non-empty *_{metric}.pkl file found in {score_pickle_dir}"
        )
    return max(candidates, key=os.path.getsize)


def plot_existing_score_pickles(args):
    """Create score matrix graph PNGs from already-written metric pickle streams."""
    selected_metrics = metric_names_to_run(args.metric)
    score_pickle_dir = args.plot_existing_score_dir
    graph_output_dir = args.graph_output_dir or os.path.join(score_pickle_dir, "score_matrix_graphs")
    records_by_metric = {}

    for metric in selected_metrics:
        score_pickle_path = find_existing_score_pickle(score_pickle_dir, metric)
        records_by_metric[metric] = load_score_pickle_records(score_pickle_path)
        print(f"loaded {len(records_by_metric[metric])} {metric} records from: {score_pickle_path}")

    record_counts = {metric: len(records) for metric, records in records_by_metric.items()}
    if len(set(record_counts.values())) != 1:
        raise ValueError(f"Metric pickle streams have different record counts: {record_counts}")

    document_count = next(iter(record_counts.values()), 0)
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max-items must be a positive integer")
        document_count = min(document_count, args.max_items)

    os.makedirs(graph_output_dir, exist_ok=True)
    for document_index in tqdm.tqdm(range(document_count), unit="img"):
        first_record = records_by_metric[selected_metrics[0]][document_index]
        file_name = os.path.basename(str(first_record.get("fname", f"document_{document_index + 1:04d}")))
        row_text_role = first_record.get("row_text_role", "reference")
        column_text_role = first_record.get("column_text_role", "prediction")
        scores_by_metric = {
            metric: records_by_metric[metric][document_index]["scores"]
            for metric in selected_metrics
        }
        graph_path = save_score_matrix_graph(
            file_name,
            scores_by_metric,
            graph_output_dir,
            document_index=document_index,
            row_text_role=row_text_role,
            column_text_role=column_text_role,
        )
        print(f"saved score matrix graph: {graph_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Sliding window score-matrix comparison using chrF, BLEU, Levenshtein, or all three",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--metric",
        type=metric_name,
        default="chrf",
        help="Segment similarity metric used to fill the score matrix",
    )
    p.add_argument(
        "--comparison-mode",
        type=comparison_mode_name,
        default="ref-pred",
        help="Text pair to compare: ref-pred, ref-ref, or pred-pred",
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
        default="scores.pkl",
        help="Progressively pickled comparison score matrices. With --metric all, metric names are added before the extension.",
    )
    p.add_argument(
        "--graph-output-dir",
        default=None,
        help="Directory for score matrix PNG graphs. Defaults to <output-dir>/score_matrix_graphs when --metric all is used.",
    )
    p.add_argument(
        "--plot-existing-score-dir",
        default=None,
        help="Plot existing metric score pickle files from this directory instead of recomputing matrices.",
    )
    p.add_argument(
        "--max-items",
        "--num-documents",
        dest="max_items",
        type=int,
        default=None,
        help="Process only the first N entries from the runfile JSON",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plot_existing_score_dir is not None:
        plot_existing_score_pickles(args)
    else:
        with open(args.runfile_json, "r", encoding="utf-8") as runfile:
            all_comparisons(json.load(runfile), args)
