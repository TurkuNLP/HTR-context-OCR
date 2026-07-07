"""Image acquisition and preparation: full page + band crops, area-capped, JPEG data URLs.

Implements the imaging policy of IMPLEMENTATION_PLAN.md §5:
- input = the HF dataset image at native size (decision #12), resized only if its AREA exceeds
  ``config.MAX_FULL_PAGE_MP`` (no blanket long-edge constant — the old pipeline's 2500px cap is
  deliberately gone);
- up to ``config.N_BANDS`` full-width horizontal band crops at native width, giving every gutter
  several times the per-pixel density of the full page at the same token budget (CCM §3.3);
- JPEG (quality-pinned) data URLs — PNG inflates scanned-newsprint payloads ~3x for no gain.

All functions are pure (image in, image/URL out) so they are trivially unit-testable.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

import config

# Pillow >= 10 moved the resampling constants; support both without a version check at call time.
_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


# --------------------------------------------------------------------------------------
# Loading and area capping
# --------------------------------------------------------------------------------------
def dataset_image_to_pil(image_obj: Any) -> Image.Image:
    """Normalize whatever the HF ``datasets`` image feature yields into an RGB PIL image.

    The image feature can surface as a PIL image, a dict with raw ``bytes`` or a ``path``, or a
    plain path string. NOTE: unlike the old pipeline, no resize happens here — the caller applies
    the explicit area policy via ``cap_area`` so the policy is visible at the call site.
    """
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            with Image.open(BytesIO(image_obj["bytes"])) as src:
                return src.convert("RGB")
        if image_obj.get("path"):
            with Image.open(image_obj["path"]) as src:
                return src.convert("RGB")
    if isinstance(image_obj, (str, Path)):
        with Image.open(image_obj) as src:
            return src.convert("RGB")
    raise TypeError(f"Unsupported dataset image payload type: {type(image_obj)!r}")


def cap_area(image: Image.Image, max_megapixels: float) -> Image.Image:
    """Downscale (LANCZOS) so ``w*h <= max_megapixels``, preserving aspect ratio. No-op if under.

    Area-based (not edge-based) because the model's vision cost and the pixels-per-gutter budget
    both scale with area, not with the long edge.
    """
    width, height = image.size
    area_mp = (width * height) / 1_000_000
    if area_mp <= max_megapixels:
        return image
    scale = (max_megapixels / area_mp) ** 0.5  # linear scale factor that hits the area target
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, resample=_RESAMPLE)


# --------------------------------------------------------------------------------------
# Band crops (CCM O4/O5 made partially input-enforced)
# --------------------------------------------------------------------------------------
def band_centers_for_parts(parts: list[dict] | None) -> list[float]:
    """Choose vertical band centers: defaults, or one band per major part when parts are known.

    ``parts`` is pass 0's list (dicts with ``top_frac``/``bottom_frac``). With 2+ parts we center
    one band inside each of the two tallest parts so the column pass sees every grid regime; with
    0/1 parts we fall back to the configured defaults. Centers are clamped below the masthead zone.
    """
    centers: list[float]
    if parts and len(parts) >= 2:
        # Sort parts by height (tallest first) and take a mid-band inside each of the top two.
        by_height = sorted(parts, key=lambda p: p.get("bottom_frac", 0) - p.get("top_frac", 0), reverse=True)
        centers = []
        for part in by_height[: config.N_BANDS]:
            top = float(part.get("top_frac", 0.0))
            bottom = float(part.get("bottom_frac", 1.0))
            centers.append((top + bottom) / 2.0)
    else:
        centers = list(config.BAND_CENTERS)[: config.N_BANDS]
    # Clamp: no band center may put the band's top edge above the masthead skip zone.
    min_center = config.MASTHEAD_SKIP_FRAC + config.BAND_HEIGHT_FRAC / 2.0
    return [min(max(c, min_center), 1.0 - config.BAND_HEIGHT_FRAC / 2.0) for c in centers]


def make_bands(image: Image.Image, parts: list[dict] | None = None) -> list[dict]:
    """Cut full-width horizontal band crops at native width, one per chosen center.

    Returns dicts ``{"image": PIL, "center_frac": float, "top_frac": float, "bottom_frac": float}``
    so the prompt can name each band's vertical position (the model must report which band it
    counted in — O4's audit trail).
    """
    width, height = image.size
    bands: list[dict] = []
    for center in band_centers_for_parts(parts):
        top = int(max(0.0, center - config.BAND_HEIGHT_FRAC / 2.0) * height)
        bottom = int(min(1.0, center + config.BAND_HEIGHT_FRAC / 2.0) * height)
        if bottom - top < 32:  # degenerate band (tiny image); skip rather than send noise
            continue
        crop = image.crop((0, top, width, bottom))
        crop = cap_area(crop, config.BAND_MAX_MP)  # native width normally fits; cap defensively
        bands.append(
            {
                "image": crop,
                "center_frac": round(center, 3),
                "top_frac": round(top / height, 3),
                "bottom_frac": round(bottom / height, 3),
            }
        )
    return bands


# --------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------
def pil_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG ``data:`` URL (quality pinned in config)."""
    buffer = BytesIO()
    # JPEG has no alpha; inputs are already RGB via dataset_image_to_pil.
    image.save(buffer, format="JPEG", quality=config.JPEG_QUALITY)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def prepare_page_images(image_obj: Any, parts: list[dict] | None = None) -> dict:
    """One-stop preparation for a document: full page (area-capped) + bands, all as data URLs.

    Returns ``{"full": {...}, "bands": [{...}], "native_size": (w, h)}`` where each entry carries
    its data URL plus the geometry the prompts need to describe the images to the model.
    """
    native = dataset_image_to_pil(image_obj)
    native_size = native.size
    full = cap_area(native, config.MAX_FULL_PAGE_MP)
    bands = make_bands(native, parts)  # bands are cut from the NATIVE image for max gutter pixels
    prepared = {
        "native_size": native_size,
        "full": {
            "data_url": pil_to_data_url(full),
            "size": full.size,
        },
        "bands": [
            {
                "data_url": pil_to_data_url(band["image"]),
                "size": band["image"].size,
                "center_frac": band["center_frac"],
                "top_frac": band["top_frac"],
                "bottom_frac": band["bottom_frac"],
            }
            for band in bands
        ],
    }
    # Release decoded pixel buffers promptly (the runner processes many documents concurrently).
    if full is not native:
        full.close()
    for band in bands:
        band["image"].close()
    native.close()
    return prepared
