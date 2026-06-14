from __future__ import annotations

"""Convert score-floor masks into the local Hough context used by tuner_simple_alpha_sweep_pre_iou_levenshtein."""

import numpy as np


# Define the build_simple_hough_context function; its body below performs one named step of the pipeline.
def build_simple_hough_context(*, hough_input_mask: np.ndarray, score_floor: float) -> dict:
    """Return the small context consumed by probabilistic Hough detection."""
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    mask_bool = np.asarray(hough_input_mask, dtype=bool)
    # Return this computed value to the caller so the next pipeline stage can use it.
    return {
        # Add the mask field to the surrounding dictionary so it appears in outputs or returned metadata.
        "mask": mask_bool,
        # Add the hough_image field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_image": mask_bool,
        # Add the hough_mask_bool field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_mask_bool": mask_bool,
        # Add the threshold_start field to the surrounding dictionary so it appears in outputs or returned metadata.
        "threshold_start": float(score_floor),
    }


__all__ = ["build_simple_hough_context"]
