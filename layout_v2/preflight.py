#!/usr/bin/env python3
"""Preflight probes against a live vLLM server (plan §13 step 3 — run BEFORE the first real run).

Four cheap checks, each guarding one design assumption with a known fallback:

1. MULTI-IMAGE  — pass 1 sends full page + band crops in ONE request; requires the server's
   ``--limit-mm-per-prompt`` to allow it. Fallback if it fails: PASS1_INPUT=bands_only or
   full_only (launcher env), or raise the serve flag.
2. PROPERTY ORDER — the evidence-before-conclusion contract relies on structured output emitting
   JSON properties in schema order. Fallback if it fails: accept unordered emission and rely on
   the schema having no count field at all (the stronger half of the contract survives).
3. PASS-1 CONTRACT — a miniature column pass must return the exact nested ``parts`` schema, not
   old-style keys such as ``part_1`` or ``stream_check``. Fallback if it fails: do not run fixtures.
4. TOKENS<->PIXELS CALIBRATION — pins the vision tile area and constant text tokens for the
   resolution audit (delegates to audit_resolution.calibrate).

Usage (with a server up, e.g. inside an interactive node or alongside the launcher):
    python3 preflight.py --vllm-base-url http://localhost:8000/v1 --vllm-model layoutv2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat_completion, image_part, parse_json_object, text_part  # noqa: E402
import schema_check  # noqa: E402
from passes import pass1_columns  # noqa: E402


def _tiny_image_url(width: int = 96, height: int = 96, color=(200, 200, 200)) -> str:
    """A minimal synthetic image for probe requests (Pillow available via the module stack)."""
    from PIL import Image

    import imaging

    image = Image.new("RGB", (width, height), color=color)
    url = imaging.pil_to_data_url(image)
    image.close()
    return url


def probe_multi_image(args: argparse.Namespace) -> bool:
    """Send TWO images in one user turn; any successful completion means multi-image works."""
    try:
        response = chat_completion(
            base_url=args.vllm_base_url, api_key=args.vllm_api_key, model=args.vllm_model,
            messages=[
                {"role": "system", "content": [text_part("Reply with the single word: ok")]},
                {"role": "user", "content": [
                    text_part("Two images follow."),
                    image_part(_tiny_image_url(color=(220, 220, 220))),
                    image_part(_tiny_image_url(color=(120, 120, 120))),
                ]},
            ],
            max_tokens=2000, temperature=0.6, top_p=0.95, json_schema=None, timeout_seconds=300,
        )
        ok = bool(response["content"] or response["reasoning_content"])
        print(f"[preflight] multi-image: {'PASS' if ok else 'FAIL'} (content={response['content']!r:.60})")
        return ok
    except Exception as exc:
        print(f"[preflight] multi-image: FAIL ({exc})")
        print("            fallback: PASS1_INPUT=full_only, or raise --limit-mm-per-prompt")
        return False


def probe_property_order(args: argparse.Namespace) -> bool:
    """Guided-JSON with 3 alphabetically-scrambled properties; check raw emission order.

    The schema deliberately orders properties zebra -> apple -> mango: only schema-order (not
    alphabetical, not model-whim) emission proves the contract the pass-1 design relies on.
    """
    schema = {
        "type": "object",
        "properties": {
            "zebra": {"type": "string", "maxLength": 8},
            "apple": {"type": "integer", "minimum": 0, "maximum": 9},
            "mango": {"type": "boolean"},
        },
        "required": ["zebra", "apple", "mango"],
        "additionalProperties": False,
    }
    try:
        response = chat_completion(
            base_url=args.vllm_base_url, api_key=args.vllm_api_key, model=args.vllm_model,
            messages=[
                {"role": "system", "content": [text_part(
                    "Describe the image color. Fill zebra with one word, apple with a digit, "
                    "mango with true or false.")]},
                {"role": "user", "content": [image_part(_tiny_image_url())]},
            ],
            max_tokens=2000, temperature=0.6, top_p=0.95, json_schema=schema, timeout_seconds=300,
        )
        raw = response["content"]
        parsed, err = parse_json_object(raw)
        if parsed is None:
            print(f"[preflight] property-order: FAIL (unparseable: {err})")
            return False
        schema_errors = schema_check.validate(parsed, schema)
        if schema_errors:
            print(f"[preflight] property-order: FAIL ({schema_check.short_error(schema_errors)})")
            return False
        # Compare the character positions of the keys in the RAW text (json.loads loses order).
        positions = {key: raw.find(f'"{key}"') for key in ("zebra", "apple", "mango")}
        ordered = positions["zebra"] < positions["apple"] < positions["mango"] and positions["zebra"] >= 0
        print(f"[preflight] property-order: {'PASS' if ordered else 'FAIL'} (raw={raw[:120]!r})")
        if not ordered:
            print("            fallback: evidence-first still holds structurally (no count field "
                  "exists in pass 1); optionally switch client.py to response_format mode")
        return ordered
    except Exception as exc:
        print(f"[preflight] property-order: FAIL ({exc})")
        return False


def probe_pass1_contract(args: argparse.Namespace) -> bool:
    """Check the real pass-1 nested schema, the contract that failed in Gate 1."""
    try:
        response = chat_completion(
            base_url=args.vllm_base_url, api_key=args.vllm_api_key, model=args.vllm_model,
            messages=[
                {"role": "system", "content": [text_part(
                    "Return one valid pass-1 JSON object for a single one-column page. Fill every required field.")]},
                {"role": "user", "content": [
                    text_part("This page has 1 part: part 1 covers 0.00-1.00, running_text, horizontal. Image 1 follows."),
                    image_part(_tiny_image_url()),
                ]},
            ],
            max_tokens=2000, temperature=0.6, top_p=0.95,
            json_schema=pass1_columns.SCHEMA, timeout_seconds=300,
        )
        raw = response["content"]
        parsed, err = parse_json_object(raw)
        if parsed is None:
            print(f"[preflight] pass1-contract: FAIL (unparseable: {err})")
            return False
        schema_errors = schema_check.validate(parsed, pass1_columns.SCHEMA)
        forbidden = [key for key in ("part_1", "stream_check", "spanning_elements", "cross_band_check") if f'"{key}"' in raw]
        if schema_errors or forbidden:
            details = schema_check.short_error(schema_errors) if schema_errors else ""
            if forbidden:
                details = (details + "; " if details else "") + "old-style keys: " + ", ".join(forbidden)
            print(f"[preflight] pass1-contract: FAIL ({details}) raw={raw[:160]!r}")
            return False
        print(f"[preflight] pass1-contract: PASS (raw={raw[:120]!r})")
        return True
    except Exception as exc:
        print(f"[preflight] pass1-contract: FAIL ({exc})")
        return False


def probe_calibration(args: argparse.Namespace) -> bool:
    """Pin tile area + text tokens for the audit (delegates to audit_resolution.calibrate)."""
    try:
        from audit_resolution import calibrate

        result = calibrate(args.vllm_base_url, args.vllm_model, args.vllm_api_key)
        print(f"[preflight] calibration: PASS {json.dumps(result)}")
        return True
    except Exception as exc:
        print(f"[preflight] calibration: FAIL ({exc})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="layoutv2")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    results = [probe_multi_image(args), probe_property_order(args), probe_pass1_contract(args)]
    if not args.skip_calibration:
        results.append(probe_calibration(args))
    print(f"[preflight] {sum(results)}/{len(results)} probes passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
