from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from derive import derive_document
from passes import pass1_columns
from validate import validate_pass1_part, validate_pass2


class GuardrailTests(unittest.TestCase):
    def test_pass1_postprocess_canonicalizes_no_stream_branch(self) -> None:
        parsed = {
            "parts": [
                {
                    "part_index": 1,
                    "counting_band": "image 1",
                    "columns": [{"x_center_frac": 0.2, "anchor_text": "A"}],
                    "width_check": {"unit_width_frac": 0.2, "block_width_frac": 0.2, "implied_count": 1},
                    "stream": {
                        "exists": False,
                        "reason": "independent_items",
                        "returns": 0,
                        "implied_count": 7,
                    },
                    "spanning_edges": "consistent",
                    "second_band_alignment": "aligned",
                    "parts_disputed": False,
                }
            ]
        }

        out = pass1_columns.postprocess(parsed)
        stream = out["parts"][0]["stream"]

        self.assertIsNone(stream["returns"])
        self.assertIsNone(stream["implied_count"])

    def test_duplicate_pass1_anchor_alone_is_not_review_issue(self) -> None:
        part = {
            "part_index": 2,
            "columns": [
                {"x_center_frac": 0.33, "anchor_text": "Kolmas"},
                {"x_center_frac": 0.67, "anchor_text": "Kolmas"},
            ],
            "width_check": {"unit_width_frac": 0.5, "block_width_frac": 0.95, "implied_count": 2},
            "stream": {"exists": True, "reason": "continuous_stream", "returns": 1, "implied_count": 2},
        }

        self.assertEqual(validate_pass1_part(part), [])

    def test_pass2_punctuation_duplicate_anchors_are_ignored(self) -> None:
        pass2 = {
            "parts": [
                {
                    "part_index": 1,
                    "articles": {
                        "mode": "enumerate",
                        "items": [{"anchor": "—"}, {"anchor": "—"}],
                        "sample": {"sampled_column": 0, "items_in_column": 0, "columns_with_items": 0},
                    },
                    "advertisements": {
                        "mode": "none_present",
                        "items": [],
                        "sample": {"sampled_column": 0, "items_in_column": 0, "columns_with_items": 0},
                    },
                    "entries": {
                        "mode": "not_applicable",
                        "items": [],
                        "sample": {"sampled_column": 0, "items_in_column": 0, "columns_with_items": 0},
                    },
                }
            ]
        }

        self.assertEqual(validate_pass2(pass2, [{"part_index": 1}], ("articles", "advertisements")), [])

    def test_running_text_independent_items_gets_inferred_when_articles_dominate(self) -> None:
        pass1 = {
            "parts": [
                {
                    "part_index": 1,
                    "columns": [{"x_center_frac": i / 10, "anchor_text": str(i)} for i in range(1, 8)],
                    "width_check": {"implied_count": 7},
                    "stream": {"exists": False, "reason": "independent_items"},
                    "spanning_edges": "none_present",
                    "second_band_alignment": "aligned",
                    "parts_disputed": False,
                }
            ]
        }
        pass2 = {
            "parts": [
                {
                    "part_index": 1,
                    "articles": {
                        "mode": "sample",
                        "items": [],
                        "sample": {"items_in_column": 5, "columns_with_items": 7},
                    },
                    "advertisements": {
                        "mode": "enumerate",
                        "items": [{"anchor": "single ad"}],
                        "sample": {},
                    },
                    "entries": {"mode": "not_applicable", "items": [], "sample": {}},
                }
            ]
        }

        derived = derive_document(
            category="newspaper",
            pass0_parts=[
                {
                    "part_index": 1,
                    "top_frac": 0.0,
                    "bottom_frac": 1.0,
                    "content_class": "running_text",
                }
            ],
            pass1=pass1,
            pass2=pass2,
            validation_issues=[],
        )

        self.assertTrue(derived["parts"][0]["stream_exists"])
        self.assertEqual(derived["parts"][0]["stream_reason"], "inferred_continuous_stream")
        self.assertTrue(derived["parts"][0]["stream_inferred"])
        self.assertIn("part 1: running_text part marked independent_items", derived["needs_review_reasons"])
        self.assertIn("part 1: inferred running_text stream", derived["needs_review_reasons"])

    def test_running_text_independent_items_stays_false_when_ads_dominate(self) -> None:
        pass1 = {
            "parts": [
                {
                    "part_index": 1,
                    "columns": [{"x_center_frac": i / 10, "anchor_text": str(i)} for i in range(1, 8)],
                    "width_check": {"implied_count": 7},
                    "stream": {"exists": False, "reason": "independent_items"},
                    "spanning_edges": "none_present",
                    "second_band_alignment": "aligned",
                    "parts_disputed": False,
                }
            ]
        }
        pass2 = {
            "parts": [
                {
                    "part_index": 1,
                    "articles": {
                        "mode": "enumerate",
                        "items": [{"anchor": "article"}],
                        "sample": {},
                    },
                    "advertisements": {
                        "mode": "sample",
                        "items": [],
                        "sample": {"items_in_column": 10, "columns_with_items": 7},
                    },
                    "entries": {"mode": "not_applicable", "items": [], "sample": {}},
                }
            ]
        }

        derived = derive_document(
            category="newspaper",
            pass0_parts=[
                {
                    "part_index": 1,
                    "top_frac": 0.0,
                    "bottom_frac": 1.0,
                    "content_class": "running_text",
                }
            ],
            pass1=pass1,
            pass2=pass2,
            validation_issues=[],
        )

        self.assertFalse(derived["parts"][0]["stream_exists"])
        self.assertEqual(derived["parts"][0]["stream_reason"], "independent_items")
        self.assertNotIn("stream_inferred", derived["parts"][0])

    def test_high_column_mixed_mosaic_needs_review(self) -> None:
        pass1 = {
            "parts": [
                {
                    "part_index": 1,
                    "columns": [{"x_center_frac": i / 10, "anchor_text": str(i)} for i in range(1, 11)],
                    "width_check": {"implied_count": 10},
                    "stream": {"exists": True, "reason": "continuous_stream", "implied_count": 10},
                    "spanning_edges": "consistent",
                    "second_band_alignment": "aligned",
                    "parts_disputed": False,
                }
            ]
        }
        pass2 = {
            "parts": [
                {
                    "part_index": 1,
                    "articles": {
                        "mode": "enumerate",
                        "items": [{"anchor": f"a{i}"} for i in range(4)],
                        "sample": {},
                    },
                    "advertisements": {
                        "mode": "sample",
                        "items": [],
                        "sample": {"items_in_column": 15, "columns_with_items": 2},
                    },
                    "entries": {"mode": "not_applicable", "items": [], "sample": {}},
                }
            ]
        }

        derived = derive_document(
            category="newspaper",
            pass0_parts=[{"part_index": 1, "top_frac": 0.0, "bottom_frac": 1.0}],
            pass1=pass1,
            pass2=pass2,
            validation_issues=[],
        )

        self.assertTrue(derived["needs_review"])
        self.assertIn("high-column mixed item mosaic", derived["needs_review_reasons"])


if __name__ == "__main__":
    unittest.main()
