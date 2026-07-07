from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures import score_one


class FixtureScoringTests(unittest.TestCase):
    def test_exact_column_miss_by_one_passes_tolerance(self) -> None:
        expected = {
            "category": "newspaper",
            "parts": 1,
            "dominant_cols": 8,
            "dominant_bin": "7+",
            "may_review": True,
            "must_review": False,
            "stream_dominant": None,
        }
        record = {
            "derived": {
                "document_category": "newspaper",
                "independent_parts": 1,
                "column_count_dominant": 9,
                "needs_review": True,
                "needs_review_reasons": ["high-column mixed item mosaic"],
            }
        }

        result = score_one("demo", expected, record)

        self.assertFalse(result["checks"]["columns_exact"])
        self.assertTrue(result["checks"]["columns_tol1"])
        self.assertTrue(result["passed"])

    def test_exact_column_miss_by_two_fails_tolerance(self) -> None:
        expected = {
            "category": "newspaper",
            "parts": 1,
            "dominant_cols": 8,
            "dominant_bin": "7+",
            "may_review": True,
            "must_review": False,
            "stream_dominant": None,
        }
        record = {
            "derived": {
                "document_category": "newspaper",
                "independent_parts": 1,
                "column_count_dominant": 10,
                "needs_review": False,
                "needs_review_reasons": [],
            }
        }

        result = score_one("demo", expected, record)

        self.assertFalse(result["checks"]["columns_exact"])
        self.assertFalse(result["checks"]["columns_tol1"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
