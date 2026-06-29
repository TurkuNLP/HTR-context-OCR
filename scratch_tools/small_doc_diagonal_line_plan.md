# Plan — Small-document metrics: character recall/precision, diagonal becomes visual-only

**Status: DESIGN / NOT YET IMPLEMENTED.** Supersedes all earlier "weighted-along-line" designs.

---

## 1. Context — why this change

The earlier work gave small docs a synthesized diagonal and a "weighted-along-line NLS". That metric
was problematic (it depended on a fitted line whose geometry drifts off the bright ridge on non-square
matrices), and on reflection it is unnecessary: for a document too small for the Hough pipeline, the
**full-text NLS is already the right quality number**, and coverage is best expressed directly from the
character edit alignment.

New decision (confirmed): keep the pipeline's metric **names**, but for small docs compute the
reference metrics from the character alignment, keep the fitted diagonal **purely as a visual aid**
(no metric attached), and set the weighted column equal to docNLS.

### The metric scheme (character edit alignment of the outputs.json text)
From `ops = Levenshtein.opcodes(reference_text, prediction_text)`, count characters per opcode
(`replace` is 1:1 for Levenshtein, verified):

- **`correct_ref_coverage` = `equal` / len(ref)** — reference characters covered **and** correct (character recall).
- **`missing_ref_coverage` = (`replace` + `delete`) / len(ref)** — substitution + omission (= 1 − recall).
  → `correct_ref_coverage + missing_ref_coverage = 1` exactly.
- **`hallucination` = `insert` / len(pred)** — prediction characters with no reference (narrow / insert-only; unchanged from current code).
- **`repetition_on_reference` = 0.0** — a one-to-one alignment cannot cover a reference char twice.
- **`document_normalised_levenshtein`** — full-text NLS (unchanged).
- **`weighted_along_lines_normalised_levenshtein` = docNLS** — small docs have no lines; the column exists only to match the pipeline schema and carries the docNLS value.

Only the split point moves vs the current code: `replace` (substitution) now counts as **missing**,
not as covered. So `correct_ref_coverage` becomes stricter (exact matches only). Sample values:

| doc | equal/rep/del/ins | correct (eq/ref) | missing ((rep+del)/ref) | hallucination (ins/pred) |
|---|---|--:|--:|--:|
| impact (ref234/pred213) | 207/2/25/4 | 0.885 | 0.115 | 0.019 |
| arabic (ref356/pred367) | 328/18/10/21 | 0.921 | 0.079 | 0.057 |
| aldicam (ref182/pred109) | 74/32/76/3 | 0.407 | 0.593 | 0.028 |

**Recorded caveat:** this redefines `correct_ref_coverage` to mean *character correctness* (recall),
which differs from the large-doc pipeline meaning (*geometric* line coverage). Same name, different
axis — small-doc and large-doc `correct_ref_coverage` are not directly comparable. Chosen knowingly.

---

## 2. Changes by file

### `scratch_tools/small_document_alignment.py` — strip to geometry + loader
- **Remove** the whole weighted-metric / text-reconstruction stack: `along_line_text_nls` and the
  vendored helpers `sliding_text_windows`, `ordered_unique`, `sequence_is_non_decreasing`,
  `reference_rows_for_mapped_columns`, `join_text_windows_without_separators`,
  `normalized_levenshtein_similarity`; plus the now-unused `DEFAULT_WINDOW_SIZE`, `DEFAULT_WINDOW_STRIDE`
  and the `math` / `Sequence` / `rapidfuzz` imports.
- **Keep** `load_score_pkl_records`, `alignment_anchor_points`, `_clamp`, `fit_alignment_line`
  (the fitted diagonal is still drawn — purely visual now).
- Update the module docstring: the diagonal is a visual aid only; no metric is derived from it.

### `scratch_tools/score_small_documents.py` — new reference partition, weighted = docNLS
- Imports: keep only `load_score_pkl_records` from the shared module (drop `fit_alignment_line`,
  `along_line_text_nls`, `DEFAULT_WINDOW_SIZE/STRIDE`, `numpy`). **Remove** `matrix_weighted_along_line_nls`.
- Replace `char_coverage` with a reference partition (reuse the opcodes already computed for hallucination):
  `correct = equal / n`, `missing = (replace + delete) / n` (n = len(ref); guard n==0 → 1.0, 0.0).
  Keep `char_hallucination` unchanged (`insert / len(pred)`).
- In the scoring loop: `ops = Levenshtein.opcodes(ref, pred)` once; `doc_nls = nls(pred, ref)`;
  `correct, missing = reference_partition(ref, ops)`; `hall = char_hallucination(ops, len(pred))`;
  `weighted = doc_nls`.
- `fieldnames` (rename the weighted column to the pipeline-exact name):
  `fname, main_language, document_type, reference_text_length, prediction_text_length, row_count,
  column_count, document_normalised_levenshtein, weighted_along_lines_normalised_levenshtein,
  correct_ref_coverage, missing_ref_coverage, hallucination, repetition_on_reference`.
  Row: `weighted_along_lines_normalised_levenshtein = round(doc_nls, 6)`, `correct_ref_coverage`,
  `missing_ref_coverage`, `hallucination` rounded; `repetition_on_reference = 0.0`.
- Keep the existing prediction gate (load pkl records → fnames with non-empty pred) and the
  `prediction_text_length > 0` filter, so the document selection stays identical (131 docs). The matrix
  itself is no longer read for any metric.

### `scratch_tools/plot_small_documents.py` — diagonal is decorative
- Keep drawing the fitted diagonal (`fit_alignment_line`) over the faint staircase, but relabel it
  **"fitted diagonal (visual)"** with **no NLS in the legend** (it no longer measures anything);
  drop the `nls_value` argument from `draw_fitted_diagonal`.
- Metrics strip + title: stop reading `weighted_along_line_nls`; show `document_normalised_levenshtein`,
  `correct_ref_coverage`, `missing_ref_coverage`, `hallucination`, `repetition_on_reference` (the
  renamed weighted column is omitted from the strip since it equals docNLS).

Out of scope: the diagonal's slope/clamp artifact stays (cosmetic only now). Can be tidied later
(clamp-along-direction) if desired.

---

## 3. Verification
1. Re-run `score_small_documents.py` to a scratch CSV; confirm 131 rows and:
   - `correct_ref_coverage` = equal/ref_len (impact ≈ 0.885, arabic ≈ 0.921, aldicam ≈ 0.407);
   - `correct_ref_coverage + missing_ref_coverage == 1` for all rows;
   - `hallucination` byte-identical to the current CSV (definition unchanged);
   - `weighted_along_lines_normalised_levenshtein == document_normalised_levenshtein`;
   - `repetition_on_reference == 0`.
2. Re-render a couple of panels (impact + a degenerate doc) → diagonal drawn and labelled "visual",
   strip shows the new correct/missing values.
3. `py_compile` all three files; grep for dangling refs to removed names.
4. Roll out: regenerate in place at
   `.../balanced/small_document_scores_diagonal.csv` and `.../balanced/small_document_plots_diagonal/`
   (using `results/custom_churro_infer_dev_run1/vllm/dev/outputs.json`), overwriting the previous version.
