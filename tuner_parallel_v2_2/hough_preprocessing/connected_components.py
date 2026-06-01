from __future__ import annotations

"""Connected regions used to build the Hough Region of Interest."""

from dataclasses import dataclass

import numpy as np

from .config import (
    CONNECTED_COMPONENT_BACKEND_CYTHON,
    CONNECTED_COMPONENT_BACKEND_PYTHON,
    CONNECTED_COMPONENT_BACKEND_SCIPY,
)


@dataclass(frozen=True)
class ComponentSummary:
    """Shape summary for one connected active-cell component."""

    label: int
    cell_count: int
    row_count: int
    column_count: int
    y_minimum: int
    y_maximum: int
    x_minimum: int
    x_maximum: int


def _try_cython_label(mask: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Use the compiled Cython labeler when it has been built."""
    try:
        from ..cython_accel.roi_preprocessing_core import label_connected_components_uint8
    except ImportError:
        try:
            from cython_accel.roi_preprocessing_core import label_connected_components_uint8  # type: ignore
        except ImportError:
            return None

    active_mask_uint8 = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    labels, component_count = label_connected_components_uint8(active_mask_uint8)
    return np.asarray(labels, dtype=np.int32), int(component_count)


def _try_scipy_label(mask: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Use SciPy connected-component labeling when it is available."""
    try:
        from scipy import ndimage
    except ImportError:
        return None
    labels, component_count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8))
    return np.asarray(labels, dtype=np.int32), int(component_count)


def _label_components_with_python(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label eight-connected components without optional dependencies."""
    active_mask = np.asarray(mask, dtype=bool)
    row_count, column_count = active_mask.shape
    labels = np.zeros(active_mask.shape, dtype=np.int32)
    current_label = 0

    for row_index in range(row_count):
        for column_index in range(column_count):
            if not bool(active_mask[row_index, column_index]) or labels[row_index, column_index] != 0:
                continue

            current_label += 1
            labels[row_index, column_index] = current_label
            stack = [(row_index, column_index)]
            while stack:
                active_row, active_column = stack.pop()
                for neighbour_row in range(max(0, active_row - 1), min(row_count, active_row + 2)):
                    for neighbour_column in range(max(0, active_column - 1), min(column_count, active_column + 2)):
                        if neighbour_row == active_row and neighbour_column == active_column:
                            continue
                        if bool(active_mask[neighbour_row, neighbour_column]) and labels[neighbour_row, neighbour_column] == 0:
                            labels[neighbour_row, neighbour_column] = current_label
                            stack.append((neighbour_row, neighbour_column))

    return labels, int(current_label)


def _label_with_requested_backend(mask: np.ndarray, requested_backend: str) -> tuple[np.ndarray, int, str]:
    """Label components with the requested backend and safe fallbacks."""
    if requested_backend == CONNECTED_COMPONENT_BACKEND_CYTHON:
        cython_result = _try_cython_label(mask)
        if cython_result is not None:
            labels, component_count = cython_result
            return labels, int(component_count), CONNECTED_COMPONENT_BACKEND_CYTHON
        scipy_result = _try_scipy_label(mask)
        if scipy_result is not None:
            labels, component_count = scipy_result
            return labels, int(component_count), CONNECTED_COMPONENT_BACKEND_SCIPY

    if requested_backend == CONNECTED_COMPONENT_BACKEND_SCIPY:
        scipy_result = _try_scipy_label(mask)
        if scipy_result is not None:
            labels, component_count = scipy_result
            return labels, int(component_count), CONNECTED_COMPONENT_BACKEND_SCIPY

    labels, component_count = _label_components_with_python(mask)
    return labels, int(component_count), CONNECTED_COMPONENT_BACKEND_PYTHON


def label_connected_components(
    mask: np.ndarray,
    *,
    backend: str = CONNECTED_COMPONENT_BACKEND_CYTHON,
) -> tuple[np.ndarray, list[ComponentSummary], str]:
    """Return labels, component summaries, and the backend that was used."""
    active_mask = np.asarray(mask, dtype=bool)
    if active_mask.ndim != 2:
        raise ValueError("connected component labeling expects a two-dimensional mask")
    if active_mask.size == 0 or not bool(np.any(active_mask)):
        return np.zeros_like(active_mask, dtype=np.int32), [], str(backend)

    labels, component_count, backend_used = _label_with_requested_backend(active_mask, str(backend))

    summaries: list[ComponentSummary] = []
    try:
        from scipy import ndimage
    except ImportError:
        component_slices = [None] * int(component_count)
    else:
        component_slices = list(ndimage.find_objects(labels))

    for component_label in range(1, int(component_count) + 1):
        component_slice = component_slices[component_label - 1] if component_label - 1 < len(component_slices) else None
        if component_slice is None:
            component_rows, component_columns = np.nonzero(labels == component_label)
            row_offset = 0
            column_offset = 0
        else:
            row_slice, column_slice = component_slice
            local_component_mask = labels[component_slice] == component_label
            component_rows, component_columns = np.nonzero(local_component_mask)
            row_offset = int(row_slice.start or 0)
            column_offset = int(column_slice.start or 0)

        if component_rows.size == 0:
            continue

        absolute_rows = component_rows + row_offset
        absolute_columns = component_columns + column_offset
        summaries.append(
            ComponentSummary(
                label=int(component_label),
                cell_count=int(component_rows.size),
                row_count=int(np.unique(absolute_rows).size),
                column_count=int(np.unique(absolute_columns).size),
                y_minimum=int(absolute_rows.min()),
                y_maximum=int(absolute_rows.max()),
                x_minimum=int(absolute_columns.min()),
                x_maximum=int(absolute_columns.max()),
            )
        )

    return labels, summaries, backend_used


def dilate_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    """Expand active cells by a square radius while preserving matrix shape."""
    active_mask = np.asarray(mask, dtype=bool)
    dilation_radius = int(radius)
    if dilation_radius <= 0 or active_mask.size == 0 or not bool(np.any(active_mask)):
        return active_mask.copy()

    try:
        from scipy import ndimage
    except ImportError:
        row_count, column_count = active_mask.shape
        dilated_mask = np.zeros(active_mask.shape, dtype=bool)
        active_rows, active_columns = np.nonzero(active_mask)
        for row_index, column_index in zip(active_rows, active_columns):
            row_start = max(0, int(row_index) - dilation_radius)
            row_end = min(row_count, int(row_index) + dilation_radius + 1)
            column_start = max(0, int(column_index) - dilation_radius)
            column_end = min(column_count, int(column_index) + dilation_radius + 1)
            dilated_mask[row_start:row_end, column_start:column_end] = True
        return dilated_mask

    structure_size = int(dilation_radius) * 2 + 1
    structure = np.ones((structure_size, structure_size), dtype=bool)
    return np.asarray(ndimage.binary_dilation(active_mask, structure=structure), dtype=bool)


def active_mask_geometry(mask: np.ndarray) -> dict[str, int | float]:
    """Describe how much of a matrix remains active after preprocessing."""
    active_mask = np.asarray(mask, dtype=bool)
    active_rows, active_columns = np.nonzero(active_mask)
    active_cell_count = int(active_rows.size)
    if active_cell_count == 0:
        return {
            "active_cell_count": 0,
            "active_fraction": 0.0,
            "active_row_count": 0,
            "active_column_count": 0,
            "x_span": 0,
            "y_span": 0,
        }

    return {
        "active_cell_count": int(active_cell_count),
        "active_fraction": float(active_cell_count / active_mask.size) if active_mask.size else 0.0,
        "active_row_count": int(np.unique(active_rows).size),
        "active_column_count": int(np.unique(active_columns).size),
        "x_span": int(active_columns.max() - active_columns.min() + 1),
        "y_span": int(active_rows.max() - active_rows.min() + 1),
    }
