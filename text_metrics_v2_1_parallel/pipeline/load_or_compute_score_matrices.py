"""Per-document score matrix provider with strict source validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.resolve_text_metrics_input_sources import (
    KIND_REF_TO_ADJUSTED_PRED,
    KIND_REF_TO_PRED,
    KIND_REF_TO_REF,
)
from score_matrix_builder import coerce_score_matrix, compute_score_matrix
from score_stream_index import load_score_item_by_offset


class ItemScoreMatrixProvider:
    """Load matrices from indexed pkl or compute them on demand per document."""

    def __init__(
        self,
        *,
        item: dict,
        window_size: int,
        window_stride: int,
        score_index_by_kind: dict[str, dict[str, dict]],
        scores_pkl_paths_by_kind: dict[str, Path | None],
    ) -> None:
        """Store source metadata needed to fetch matrices for one item."""
        self.item = item
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.score_index_by_kind = score_index_by_kind
        self.scores_pkl_paths_by_kind = scores_pkl_paths_by_kind
        self._cache: dict[str, np.ndarray] = {}
        self._source: dict[str, str] = {}

    def _lookup_item(self, kind: str) -> dict | None:
        """Return indexed score entry for this filename and matrix kind."""
        lookup = self.score_index_by_kind.get(kind, {})
        return lookup.get(Path(str(self.item["fname"])).name)

    def _coerce_and_store_from_pkl(
        self,
        *,
        kind: str,
        score_index_item: dict,
        expected_pred_text: str | None = None,
    ) -> np.ndarray:
        """Load one matrix record from pkl stream and validate ref/pred text."""
        ref_text = str(self.item["ref"])
        pred_text = str(self.item["pred"])

        p = self.scores_pkl_paths_by_kind.get(kind)
        if p is None:
            raise ValueError(f"Internal error: missing scores path for kind={kind!r}")

        if score_index_item.get("has_ref", False) and str(score_index_item.get("ref", "")) != ref_text:
            raise ValueError(
                f"Reference text mismatch between item and {kind} scores.pkl for {self.item['fname']!r}"
            )

        if kind == KIND_REF_TO_PRED:
            if score_index_item.get("has_pred", False) and str(score_index_item.get("pred", "")) != pred_text:
                raise ValueError(
                    f"Prediction text mismatch between item and {kind} scores.pkl for {self.item['fname']!r}"
                )

        if expected_pred_text is not None and score_index_item.get("has_pred", False):
            if str(score_index_item.get("pred", "")) != expected_pred_text:
                raise ValueError(
                    f"Adjusted prediction text mismatch between item and {kind} scores.pkl for {self.item['fname']!r}"
                )

        raw_item = load_score_item_by_offset(Path(p), int(score_index_item["offset"]))
        self._cache[kind] = coerce_score_matrix(
            raw_item.get("scores"),
            source_desc=f"{p}:{score_index_item['stream_index']}:{score_index_item['fname']}",
        )
        self._source[kind] = f"pkl:{p}" if p is not None else "pkl"
        return self._cache[kind]

    def _fetch_or_compute(
        self,
        *,
        kind: str,
        build_fn,
        expected_pred_text: str | None = None,
    ) -> np.ndarray:
        """Fetch matrix from cache/pkl or compute it when source is absent."""
        if kind in self._cache:
            return self._cache[kind]

        score_index_item = self._lookup_item(kind)
        if score_index_item is not None:
            return self._coerce_and_store_from_pkl(
                kind=kind,
                score_index_item=score_index_item,
                expected_pred_text=expected_pred_text,
            )

        if self.scores_pkl_paths_by_kind.get(kind) is not None:
            raise KeyError(
                f"No {kind} scores.pkl entry found for fname={self.item['fname']!r} in "
                f"{self.scores_pkl_paths_by_kind[kind]}"
            )

        self._cache[kind] = build_fn()
        self._source[kind] = "computed"
        return self._cache[kind]

    def get_ref_to_pred_matrix(self) -> np.ndarray:
        """Return reference->prediction score matrix for the current item."""
        return self._fetch_or_compute(
            kind=KIND_REF_TO_PRED,
            build_fn=lambda: compute_score_matrix(
                str(self.item["ref"]),
                str(self.item["pred"]),
                window_size=self.window_size,
                window_stride=self.window_stride,
            ),
        )

    def get_ref_to_ref_matrix(self) -> np.ndarray:
        """Return reference->reference score matrix for the current item."""
        return self._fetch_or_compute(
            kind=KIND_REF_TO_REF,
            build_fn=lambda: compute_score_matrix(
                str(self.item["ref"]),
                str(self.item["ref"]),
                window_size=self.window_size,
                window_stride=self.window_stride,
            ),
        )

    def get_ref_to_adjusted_pred_matrix(self, adjusted_pred_text: str) -> np.ndarray:
        """Return reference->adjusted-pred matrix with safe reuse fallback."""
        if (
            self.scores_pkl_paths_by_kind.get(KIND_REF_TO_ADJUSTED_PRED) is None
            and adjusted_pred_text == str(self.item["pred"])
            and KIND_REF_TO_PRED in self._cache
        ):
            self._cache[KIND_REF_TO_ADJUSTED_PRED] = self._cache[KIND_REF_TO_PRED]
            self._source[KIND_REF_TO_ADJUSTED_PRED] = self._source[KIND_REF_TO_PRED] + " (reused_ref_to_pred)"
            return self._cache[KIND_REF_TO_ADJUSTED_PRED]

        return self._fetch_or_compute(
            kind=KIND_REF_TO_ADJUSTED_PRED,
            build_fn=lambda: compute_score_matrix(
                str(self.item["ref"]),
                adjusted_pred_text,
                window_size=self.window_size,
                window_stride=self.window_stride,
            ),
            expected_pred_text=adjusted_pred_text,
        )

    def source_for(self, kind: str) -> str:
        """Return source label for one matrix kind."""
        return self._source.get(kind, "not_needed")
