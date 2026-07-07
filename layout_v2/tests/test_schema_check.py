from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import schema_check  # noqa: E402
from passes import pass1_columns, pass2_items  # noqa: E402


class SchemaCheckTests(unittest.TestCase):
    def test_valid_pass1_payload_passes(self) -> None:
        payload = {
            "parts": [
                {
                    "part_index": 1,
                    "counting_band": "image 2",
                    "columns": [{"x_center_frac": 0.5, "anchor_text": "main text"}],
                    "width_check": {
                        "unit_width_frac": 0.8,
                        "block_width_frac": 0.8,
                        "implied_count": 1,
                    },
                    "stream": {
                        "exists": True,
                        "reason": "continuous_stream",
                        "returns": 0,
                        "implied_count": 1,
                    },
                    "spanning_edges": "none_present",
                    "second_band_alignment": "aligned",
                    "parts_disputed": False,
                }
            ]
        }

        self.assertEqual(schema_check.validate(payload, pass1_columns.SCHEMA), [])

    def test_old_style_pass1_payload_fails(self) -> None:
        payload = {
            "part_1": {
                "counting_band": "image 1",
                "columns": [{"x_center_frac": 0.5, "anchor_text": "main text"}],
                "width_check": {"unit_width_frac": 1.0, "block_width_frac": 1.0, "implied_column_count": 1},
                "stream_check": {"returns": 0, "implied_count": 1},
            }
        }

        errors = schema_check.validate(payload, pass1_columns.SCHEMA)

        self.assertTrue(any("missing required key parts" in error for error in errors))
        self.assertTrue(any("additional property not allowed" in error for error in errors))

    def test_wrong_pass1_inner_keys_fail(self) -> None:
        payload = {
            "parts": [
                {
                    "part_index": 1,
                    "counting_band": "image 1",
                    "columns": [{"x_center_frac": 0.5, "anchor_text": "main text"}],
                    "width_check": {"unit_width_frac": 1.0, "block_width_frac": 1.0, "implied_count": 1},
                    "stream_check": {"returns": 0, "implied_count": 1},
                    "spanning_elements": "none_present",
                    "cross_band_check": "aligned",
                }
            ]
        }

        errors = schema_check.validate(payload, pass1_columns.SCHEMA)

        self.assertTrue(any("$.parts[0]: missing required key stream" in error for error in errors))
        self.assertTrue(any("$.parts[0].stream_check: additional property not allowed" in error for error in errors))

    def test_pass2_top_level_articles_fail(self) -> None:
        payload = {
            "articles": {"mode": "none_present", "items": [], "sample": {}},
            "advertisements": {"mode": "none_present", "items": [], "sample": {}},
        }

        errors = schema_check.validate(payload, pass2_items.SCHEMA)

        self.assertTrue(any("missing required key parts" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
