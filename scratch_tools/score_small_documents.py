#!/usr/bin/env python3
"""Score documents skipped for matrix_size using character-level Levenshtein.

All metrics are character-level, computed from one edit alignment of the outputs.json text:
  correct_ref_coverage = good (covered AND correct)      = equal / len(ref)      (character recall)
  missing_ref_coverage = substitution + omission         = (replace+delete)/len(ref) = 1 - recall
  hallucination        = invented prediction characters  = insert / len(pred)
  weighted_along_lines_normalised_levenshtein            = docNLS (no lines exist for small docs)
  repetition_on_reference                                = 0.0 (one-to-one alignment cannot repeat)

NOTE: this `correct_ref_coverage` measures character correctness, which differs from the large-doc
pipeline meaning (geometric line coverage). Same name, different axis — not directly comparable.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

# Allow `import small_document_alignment` regardless of the working directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from small_document_alignment import load_score_pkl_records  # noqa: E402


def nls(pred: str, ref: str) -> float:
    denom = max(len(ref), len(pred))
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - Levenshtein.distance(pred, ref) / denom)


def reference_partition(ref: str, ops: list[tuple]) -> tuple[float, float]:
    """Split the reference into correct (equal) vs missing (substitution + omission).

    correct_ref_coverage = equal / len(ref)              -- covered AND correct (character recall)
    missing_ref_coverage = (replace + delete) / len(ref) -- substituted or omitted (= 1 - recall)
    The two sum to 1 because equal + replace + delete spans every reference character.
    """
    n = len(ref)
    if n == 0:
        return 1.0, 0.0
    correct = sum(e - s for tag, s, e, _, _ in ops if tag == "equal")
    return correct / n, (n - correct) / n


def char_hallucination(ops: list[tuple], prediction_length: int) -> float:
    """Fraction of the prediction aligned to a gap in the reference (invented text, insert opcodes).

    Narrow / insert-only definition. 0.0 for an empty prediction.
    """
    if prediction_length <= 0:
        return 0.0
    inserted = sum(de - ds for tag, _, _, ds, de in ops if tag == "insert")
    return inserted / prediction_length


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores-pkl-ref-to-pred", required=True,
                    help="ref→pred score pickle; used to confirm which fnames have a prediction")
    ap.add_argument("--results-dir", required=True,
                    help="balanced/ output directory containing skipped_documents.csv")
    ap.add_argument("--runfile-json", required=True,
                    help="outputs.json with normalized_gold_text and normalized_predicted_text")
    ap.add_argument("--output-csv", default=None,
                    help="output path (default: results-dir/small_document_scores.csv)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_csv = Path(args.output_csv) if args.output_csv else results_dir / "small_document_scores.csv"

    # Step 1: scan the score dump only to gate on which fnames have a non-empty prediction.
    print(f"[pkl] scanning {args.scores_pkl_ref_to_pred}")
    pkl_records = load_score_pkl_records(Path(args.scores_pkl_ref_to_pred))
    pkl_fnames = {fname for fname, rec in pkl_records.items() if rec.get("pred")}
    print(f"[pkl] {len(pkl_fnames)} fnames with non-empty prediction")

    # Step 2: read skipped_documents.csv, keep only matrix_size rows
    skipped_csv = results_dir / "skipped_documents.csv"
    matrix_size_rows: list[dict] = []
    with skipped_csv.open() as f:
        for row in csv.DictReader(f):
            if row["skip_stage"] == "matrix_size":
                matrix_size_rows.append(row)
    print(f"[skipped] {len(matrix_size_rows)} matrix_size rows")

    # Step 3: drop rows where the model produced no prediction text
    scoreable = [r for r in matrix_size_rows if int(r["prediction_text_length"]) > 0]
    print(f"[filter] {len(scoreable)} rows with prediction_text_length > 0")

    # Step 4: load outputs.json → fname → (ref_text, pred_text)
    with open(args.runfile_json) as f:
        runfile: list[dict] = json.load(f)
    text_by_fname: dict[str, tuple[str, str]] = {}
    for doc in runfile:
        fname = Path(str(doc.get("file_name", doc.get("fname", "")))).name
        ref = str(doc.get("normalized_gold_text", doc.get("ref", "")))
        pred = str(doc.get("normalized_predicted_text", doc.get("pred", "")))
        text_by_fname[fname] = (ref, pred)
    print(f"[runfile] {len(text_by_fname)} documents loaded")

    # Step 5: score each qualifying document
    fieldnames = [
        "fname", "main_language", "document_type",
        "reference_text_length", "prediction_text_length",
        "row_count", "column_count",
        "document_normalised_levenshtein", "weighted_along_lines_normalised_levenshtein",
        "correct_ref_coverage", "missing_ref_coverage", "hallucination",
        "repetition_on_reference",
    ]
    result_rows: list[dict] = []
    skipped_no_pkl: list[str] = []
    skipped_no_text: list[str] = []

    for row in scoreable:
        fname = row["fname"]

        if fname not in pkl_fnames:
            skipped_no_pkl.append(fname)
            continue

        if fname not in text_by_fname:
            skipped_no_text.append(fname)
            continue

        ref, pred = text_by_fname[fname]
        if not ref:
            skipped_no_text.append(fname)
            continue

        # All metrics share one character edit alignment of the outputs.json text.
        ops = Levenshtein.opcodes(ref, pred)
        doc_nls = nls(pred, ref)
        correct, missing = reference_partition(ref, ops)
        hallucination = char_hallucination(ops, len(pred))
        # Small docs have no lines, so the weighted-along-lines column carries the full-text NLS.
        weighted = doc_nls

        result_rows.append({
            "fname": fname,
            "main_language": row["main_language"],
            "document_type": row["document_type"],
            "reference_text_length": row["reference_text_length"],
            "prediction_text_length": row["prediction_text_length"],
            "row_count": row["row_count"],
            "column_count": row["column_count"],
            "document_normalised_levenshtein": round(doc_nls, 6),
            "weighted_along_lines_normalised_levenshtein": round(weighted, 6),
            "correct_ref_coverage": round(correct, 6),
            "missing_ref_coverage": round(missing, 6),
            "hallucination": round(hallucination, 6),
            "repetition_on_reference": 0.0,
        })

    # Step 6: write output CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    print(f"[done] {len(result_rows)} documents scored → {out_csv}")
    if skipped_no_pkl:
        print(f"[skip/no-pkl] {len(skipped_no_pkl)}: {skipped_no_pkl[:5]}")
    if skipped_no_text:
        print(f"[skip/no-text] {len(skipped_no_text)}: {skipped_no_text[:5]}")


if __name__ == "__main__":
    main()
