from __future__ import annotations

import numpy as np
import sacrebleu


def sliding_segments(text: str, window_size: int, window_stride: int) -> list[str]:
    if len(text) < window_size:
        return []
    return [text[i : i + window_size] for i in range(0, len(text) - window_size + 1, window_stride)]


def compute_score_matrix(ref_text: str, pred_text: str, window_size: int, window_stride: int) -> np.ndarray:
    ref_segments = sliding_segments(ref_text, window_size, window_stride)
    pred_segments = sliding_segments(pred_text, window_size, window_stride)

    scores = np.zeros((len(ref_segments), len(pred_segments)), dtype=float)
    for i, ref_seg in enumerate(ref_segments):
        for j, pred_seg in enumerate(pred_segments):
            scores[i, j] = sacrebleu.sentence_chrf(ref_seg, [pred_seg]).score
    return scores


def coerce_score_matrix(scores, *, source_desc: str) -> np.ndarray:
    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D score matrix in {source_desc}, got shape={mat.shape!r}")
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
