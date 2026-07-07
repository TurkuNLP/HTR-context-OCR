#!/usr/bin/env python3
"""Phase A: the resolution audit — what pixel sizes did the server actually process?

Runs OFFLINE over the per-document records of past runs (old pipeline or layout_v2): every record
stores ``usage.prompt_tokens``, and since the text side of a run's prompt is constant, the
variation in prompt tokens across documents is the image-token payload. From image tokens we
recover processed pixels via the vision tile area.

    image_tokens ≈ prompt_tokens - text_tokens          (text_tokens constant per run)
    processed_pixels ≈ image_tokens * tile_area          (tile_area ≈ 28x28 = 784 px/token)

``text_tokens`` is estimated as the per-run minimum prompt_tokens minus the smallest plausible
image payload — or pinned exactly with ``--calibrate`` against a live server (three synthetic
images of known size; the linear fit gives both tile_area and text_tokens).

Interpretation (COLUMN_COUNT_METHOD.md §3.4):
- past runs processed ~3 MP  -> optics were sufficient; the undercount was procedural;
- past runs processed ~1 MP  -> optics were binding; the resolution change must land first.

Usage:
    python3 audit_resolution.py --runs /path/to/old_run/dev [more runs...] --report audit_report.md
    python3 audit_resolution.py --calibrate --vllm-base-url http://localhost:8000/v1 --vllm-model layoutv2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Default vision tile area (Qwen-VL family: 28x28 px per merged visual token). The calibration
# mode measures the true value; this constant is only the uncalibrated fallback.
DEFAULT_TILE_AREA_PX = 28 * 28

# Fixture pages get called out individually in the report (the canonical hard cases).
FIXTURE_STEMS = (
    "europeana_00675495",
    "newseye-fin_576474_0003_23676390",
    "europeana_00675329",
    "europeana_00674544",
    "europeana_00674591",
    "newseye-fin_576485_0001_23676428",
)


# --------------------------------------------------------------------------------------
# Record walking
# --------------------------------------------------------------------------------------
def iter_records(run_dir: Path):
    """Yield (stem, prompt_tokens) from a run's per-document response files.

    Supports both record shapes: the old pipeline (usage at record.response_metadata) and
    layout_v2 (usage per pass; we take pass1 — the image-heavy call — falling back to pass0).
    """
    responses = run_dir / "responses"
    if not responses.is_dir():
        return
    for path in sorted(responses.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stem = Path(str(record.get("file_name") or path.stem)).stem
        usage = None
        if isinstance(record.get("response_metadata"), dict):  # old single-pass shape
            usage = record["response_metadata"].get("usage")
        else:  # layout_v2 shape: prefer the image-heaviest pass
            for pass_name in ("pass1", "pass0", "pass2"):
                pass_result = record.get(pass_name)
                if isinstance(pass_result, dict):
                    usage = (pass_result.get("response_metadata") or {}).get("usage")
                    if usage:
                        break
        tokens = (usage or {}).get("prompt_tokens")
        if isinstance(tokens, int):
            yield stem, tokens


def audit_run(run_dir: Path, tile_area: float, text_tokens: float | None) -> dict:
    """Compute the processed-megapixel distribution for one run directory."""
    pairs = list(iter_records(run_dir))
    if not pairs:
        return {"run": str(run_dir), "n": 0}
    tokens = [t for _, t in pairs]
    # Without calibration, estimate the constant text side as (min prompt) minus a small image:
    # the smallest realistic page (~0.5 MP) contributes ~640 tokens at the default tile area.
    estimated_text = text_tokens if text_tokens is not None else max(min(tokens) - 640, 0)

    def to_mp(prompt_tokens: int) -> float:
        image_tokens = max(prompt_tokens - estimated_text, 0)
        return round(image_tokens * tile_area / 1_000_000, 2)

    mps = sorted(to_mp(t) for t in tokens)
    fixture_rows = {stem: to_mp(t) for stem, t in pairs if stem in FIXTURE_STEMS}
    return {
        "run": str(run_dir),
        "n": len(tokens),
        "text_tokens_assumed": round(estimated_text, 0),
        "processed_mp": {
            "min": mps[0],
            "p25": mps[len(mps) // 4],
            "median": mps[len(mps) // 2],
            "p75": mps[(3 * len(mps)) // 4],
            "max": mps[-1],
            "mean": round(statistics.fmean(mps), 2),
        },
        "fixtures_mp": fixture_rows,
    }


# --------------------------------------------------------------------------------------
# Live calibration (optional; pins tile_area and text_tokens exactly)
# --------------------------------------------------------------------------------------
def calibrate(base_url: str, model: str, api_key: str) -> dict:
    """Send three known-size synthetic images; fit prompt_tokens = text + area/tile_area.

    A least-squares line through (pixels, prompt_tokens) gives slope = 1/tile_area and
    intercept = the constant text tokens for this exact prompt shape.
    """
    from PIL import Image  # local import: calibration is optional and needs Pillow only here

    from client import chat_completion, image_part, text_part

    sizes = [(800, 1000), (1400, 1800), (2000, 2500)]  # spans the realistic page range
    points: list[tuple[int, int]] = []
    for width, height in sizes:
        image = Image.new("RGB", (width, height), color=(230, 228, 220))  # blank "page"
        import imaging
        data_url = imaging.pil_to_data_url(image)
        image.close()
        response = chat_completion(
            base_url=base_url, api_key=api_key, model=model,
            messages=[
                {"role": "system", "content": [text_part("Reply with the single word: ok")]},
                {"role": "user", "content": [image_part(data_url)]},
            ],
            max_tokens=8, temperature=0.0, top_p=1.0, json_schema=None, timeout_seconds=300,
        )
        prompt_tokens = response["response_metadata"]["usage"].get("prompt_tokens", 0)
        points.append((width * height, prompt_tokens))
        print(f"[calibrate] {width}x{height} ({width*height/1e6:.1f} MP) -> prompt_tokens={prompt_tokens}")
    # Least-squares fit y = a*x + b over the three points.
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    tile_area = 1.0 / slope if slope > 0 else float("nan")
    result = {"tile_area_px_per_token": round(tile_area, 1), "text_tokens": round(intercept, 1)}
    print(f"[calibrate] fitted: {result}  (write these into your audit invocation / config)")
    return result


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def write_report(audits: list[dict], out_path: Path, tile_area: float) -> None:
    """Render the audit as a small markdown report with the CCM §3.4 verdict spelled out."""
    lines = [
        "# Resolution audit (Phase A)",
        "",
        f"Vision tile area used: **{tile_area:.0f} px/token**"
        f" ({'calibrated' if tile_area != DEFAULT_TILE_AREA_PX else 'default, uncalibrated'}).",
        "",
        "| run | n | text tokens (assumed) | min MP | median MP | max MP |",
        "|---|---|---|---|---|---|",
    ]
    for audit in audits:
        if audit.get("n", 0) == 0:
            lines.append(f"| {audit['run']} | 0 | - | - | - | - |")
            continue
        mp = audit["processed_mp"]
        lines.append(
            f"| {audit['run']} | {audit['n']} | {audit['text_tokens_assumed']:.0f} "
            f"| {mp['min']} | {mp['median']} | {mp['max']} |"
        )
    lines += ["", "## Canonical fixture pages (processed MP per run)", ""]
    for audit in audits:
        if audit.get("fixtures_mp"):
            lines.append(f"**{audit['run']}**")
            for stem, mp in audit["fixtures_mp"].items():
                lines.append(f"- {stem}: {mp} MP")
            lines.append("")
    lines += [
        "## Verdict guide (COLUMN_COUNT_METHOD.md §3.4)",
        "",
        "- Median ≈ 3 MP or above on the dense fixtures: optics were sufficient — the undercount",
        "  was procedural; the enumeration protocol carries the fix.",
        "- Median ≈ 1 MP or below: optics were binding — raise the serving-side pixel budget",
        "  (config.VLLM_MM_MAX_PIXELS) before crediting or blaming any prompt.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] report -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=[],
                        help="Run split dirs containing responses/ (old pipeline or layout_v2).")
    parser.add_argument("--tile-area", type=float, default=DEFAULT_TILE_AREA_PX,
                        help="Pixels per visual token (override with the calibrated value).")
    parser.add_argument("--text-tokens", type=float, default=None,
                        help="Known constant text tokens per request (from --calibrate).")
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parent / "audit_report.md")
    parser.add_argument("--calibrate", action="store_true", help="Run live calibration instead.")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="layoutv2")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    args = parser.parse_args()

    if args.calibrate:
        calibrate(args.vllm_base_url, args.vllm_model, args.vllm_api_key)
        return 0
    if not args.runs:
        parser.error("provide --runs (one or more .../<split> dirs with responses/) or --calibrate")
    audits = [audit_run(run, args.tile_area, args.text_tokens) for run in args.runs]
    write_report(audits, args.report, args.tile_area)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
