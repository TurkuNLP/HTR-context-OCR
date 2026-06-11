from __future__ import annotations

"""Render one document as the simple tuner's 2x3 diagnostic panel."""

from pathlib import Path
from typing import Any, Iterable, Sequence
import math

import numpy as np

# Compute or store RAW_HOUGH_LINE_COLOR so later code can reuse this named value clearly.
RAW_HOUGH_LINE_COLOR = "#E03131"
# Compute or store FINAL_LINE_COLOR so later code can reuse this named value clearly.
FINAL_LINE_COLOR = "#1971C2"
# Compute or store RAW_LINE_BOX_HALF_WIDTH_CELLS so later code can reuse this named value clearly.
RAW_LINE_BOX_HALF_WIDTH_CELLS = 1.7
# Compute or store FINAL_LINE_BOX_HALF_WIDTH_CELLS so later code can reuse this named value clearly.
FINAL_LINE_BOX_HALF_WIDTH_CELLS = 2.0


# Define the safe_path_component function; its body below performs one named step of the pipeline.
def safe_path_component(value: str) -> str:
    """Return a file-system friendly version of a language or document name."""
    # Compute or store cleaned so later code can reuse this named value clearly.
    cleaned = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in str(value))
    # Return this computed value to the caller so the next pipeline stage can use it.
    return cleaned.strip("._") or "unknown"


# Define the endpoint_tuple_from_raw_hough_segment function; its body below performs one named step of the pipeline.
def endpoint_tuple_from_raw_hough_segment(raw_segment: Any) -> tuple[float, float, float, float] | None:
    """Convert a raw Hough line ``((x0, y0), (x1, y1))`` into numeric endpoints."""
    # Define the try field so this data object records that value explicitly.
    try:
        # Execute this statement as the next small step in the surrounding pipeline logic.
        (x0, y0), (x1, y1) = raw_segment
        # Return this computed value to the caller so the next pipeline stage can use it.
        return float(x0), float(y0), float(x1), float(y1)
    # Catch the matching failure type and turn it into explicit handling instead of crashing silently.
    except (TypeError, ValueError):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return None


# Define the endpoint_tuple_from_filtered_line_record function; its body below performs one named step of the pipeline.
def endpoint_tuple_from_filtered_line_record(line_record: Any) -> tuple[float, float, float, float] | None:
    """Convert one filtered line dictionary into numeric endpoints."""
    # Check whether not isinstance(line_record, dict); the indented block handles that specific case.
    if not isinstance(line_record, dict):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return None
    # Check whether any(line_record.get(key) is None for key in ("x0", "y0", "x1", "y1")); the indented block handles that specific case.
    if any(line_record.get(key) is None for key in ("x0", "y0", "x1", "y1")):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return None
    # Return this computed value to the caller so the next pipeline stage can use it.
    return float(line_record["x0"]), float(line_record["y0"]), float(line_record["x1"]), float(line_record["y1"])


# Define the raw_source_line_ids_from_filtered_line_record function; its body below performs one named step of the pipeline.
def raw_source_line_ids_from_filtered_line_record(line_record: Any) -> list[int]:
    """Return raw Hough line identifiers stored on a surviving filtered line."""
    # Check whether not isinstance(line_record, dict); the indented block handles that specific case.
    if not isinstance(line_record, dict):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return []
    # Compute or store source_raw_line_ids so later code can reuse this named value clearly.
    source_raw_line_ids = line_record.get("source_raw_line_ids")
    # Check whether source_raw_line_ids is None; the indented block handles that specific case.
    if source_raw_line_ids is None:
        # Compute or store source_raw_line_ids so later code can reuse this named value clearly.
        source_raw_line_ids = line_record.get("raw_line_id")
    # Check whether source_raw_line_ids is None; the indented block handles that specific case.
    if source_raw_line_ids is None:
        # Return this computed value to the caller so the next pipeline stage can use it.
        return []
    # Check whether isinstance(source_raw_line_ids, (list, tuple, set)); the indented block handles that specific case.
    if isinstance(source_raw_line_ids, (list, tuple, set)):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return [int(raw_line_id) for raw_line_id in source_raw_line_ids]
    # Return this computed value to the caller so the next pipeline stage can use it.
    return [int(source_raw_line_ids)]


# Define the compact_identifier_sequence function; its body below performs one named step of the pipeline.
def compact_identifier_sequence(values: Sequence[int]) -> str:
    """Return a short readable label for a sorted list of integer identifiers."""
    # Compute or store identifiers so later code can reuse this named value clearly.
    identifiers = sorted({int(value) for value in values})
    # Check whether not identifiers; the indented block handles that specific case.
    if not identifiers:
        # Return this computed value to the caller so the next pipeline stage can use it.
        return "unknown"
    # Check whether len(identifiers) > 2 and identifiers == list(range(identifiers[0], identifiers[-1] + 1)); the indented block handles that specific case.
    if len(identifiers) > 2 and identifiers == list(range(identifiers[0], identifiers[-1] + 1)):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return f"{identifiers[0]}..{identifiers[-1]}"
    # Return this computed value to the caller so the next pipeline stage can use it.
    return ",".join(str(identifier) for identifier in identifiers)


# Define the segment_unit_vectors function; its body below performs one named step of the pipeline.
def segment_unit_vectors(*, x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return direction and perpendicular unit vectors for one line segment."""
    # Compute or store delta_x so later code can reuse this named value clearly.
    delta_x = float(x1) - float(x0)
    # Compute or store delta_y so later code can reuse this named value clearly.
    delta_y = float(y1) - float(y0)
    # Compute or store segment_length so later code can reuse this named value clearly.
    segment_length = math.hypot(delta_x, delta_y)
    # Check whether segment_length <= 0.0 or not math.isfinite(segment_length); the indented block handles that specific case.
    if segment_length <= 0.0 or not math.isfinite(segment_length):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return (1.0, 0.0), (0.0, 1.0)
    # Compute or store direction_x so later code can reuse this named value clearly.
    direction_x = delta_x / segment_length
    # Compute or store direction_y so later code can reuse this named value clearly.
    direction_y = delta_y / segment_length
    # Return this computed value to the caller so the next pipeline stage can use it.
    return (direction_x, direction_y), (-direction_y, direction_x)


# Define the oriented_segment_box_points function; its body below performs one named step of the pipeline.
def oriented_segment_box_points(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the x0 field so this data object records that value explicitly.
    x0: float,
    # Define the y0 field so this data object records that value explicitly.
    y0: float,
    # Define the x1 field so this data object records that value explicitly.
    x1: float,
    # Define the y1 field so this data object records that value explicitly.
    y1: float,
    # Define the half_width_cells field so this data object records that value explicitly.
    half_width_cells: float,
    # Define the end_padding_cells field so this data object records that value explicitly.
    end_padding_cells: float,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> list[tuple[float, float]]:
    """Return four points for a hollow box around a Hough line segment."""
    # Execute this statement as the next small step in the surrounding pipeline logic.
    (direction_x, direction_y), (normal_x, normal_y) = segment_unit_vectors(x0=x0, y0=y0, x1=x1, y1=y1)
    # Compute or store half_width so later code can reuse this named value clearly.
    half_width = max(0.5, float(half_width_cells))
    # Compute or store end_padding so later code can reuse this named value clearly.
    end_padding = max(0.0, float(end_padding_cells))
    # Compute or store start_x so later code can reuse this named value clearly.
    start_x = float(x0) - direction_x * end_padding
    # Compute or store start_y so later code can reuse this named value clearly.
    start_y = float(y0) - direction_y * end_padding
    # Compute or store end_x so later code can reuse this named value clearly.
    end_x = float(x1) + direction_x * end_padding
    # Compute or store end_y so later code can reuse this named value clearly.
    end_y = float(y1) + direction_y * end_padding
    # Return this computed value to the caller so the next pipeline stage can use it.
    return [
        # Pass this value into the surrounding multi-line call or collection.
        (start_x + normal_x * half_width, start_y + normal_y * half_width),
        # Pass this value into the surrounding multi-line call or collection.
        (end_x + normal_x * half_width, end_y + normal_y * half_width),
        # Pass this value into the surrounding multi-line call or collection.
        (end_x - normal_x * half_width, end_y - normal_y * half_width),
        # Pass this value into the surrounding multi-line call or collection.
        (start_x - normal_x * half_width, start_y - normal_y * half_width),
    ]


# Define the draw_score_matrix_heatmap function; its body below performs one named step of the pipeline.
def draw_score_matrix_heatmap(axis: Any, score_matrix: np.ndarray | None, title: str) -> Any | None:
    """Draw one score matrix with the shared 0-100 color scale."""
    # Check whether score_matrix is None; the indented block handles that specific case.
    if score_matrix is None:
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axis.text(0.5, 0.5, "Matrix missing", ha="center", va="center", transform=axis.transAxes)
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axis.set_title(title)
        # Return this computed value to the caller so the next pipeline stage can use it.
        return None
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    matrix = np.asarray(score_matrix, dtype=float)
    # Check whether matrix.ndim != 2 or matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0; the indented block handles that specific case.
    if matrix.ndim != 2 or matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axis.text(0.5, 0.5, f"Empty matrix\nshape={matrix.shape}", ha="center", va="center", transform=axis.transAxes)
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axis.set_title(title)
        # Return this computed value to the caller so the next pipeline stage can use it.
        return None
    # Compute or store image so later code can reuse this named value clearly.
    image = axis.imshow(matrix, origin="upper", aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_title(title)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_xlabel("Prediction/self window index")
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_ylabel("Reference window index")
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_xlim(-0.5, matrix.shape[1] - 0.5)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_ylim(matrix.shape[0] - 0.5, -0.5)
    # Return this computed value to the caller so the next pipeline stage can use it.
    return image


# Define the draw_binary_mask_panel function; its body below performs one named step of the pipeline.
def draw_binary_mask_panel(axis: Any, mask: np.ndarray | None, title: str, active_label: str) -> None:
    """Draw one black-and-white binary preprocessing mask."""
    # Check whether mask is None; the indented block handles that specific case.
    if mask is None:
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axis.text(0.5, 0.5, "Mask missing", ha="center", va="center", transform=axis.transAxes)
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axis.set_title(title)
        # Exit the function here without returning a separate data value.
        return
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    mask_array = np.asarray(mask, dtype=bool)
    # Check whether mask_array.ndim != 2 or mask_array.size == 0 or mask_array.shape[0] == 0 or mask_array.shape[1] == 0; the indented block handles that specific case.
    if mask_array.ndim != 2 or mask_array.size == 0 or mask_array.shape[0] == 0 or mask_array.shape[1] == 0:
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axis.text(0.5, 0.5, f"Empty mask\nshape={mask_array.shape}", ha="center", va="center", transform=axis.transAxes)
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axis.set_title(title)
        # Exit the function here without returning a separate data value.
        return
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    axis.imshow(mask_array.astype(np.uint8), origin="upper", aspect="auto", cmap="gray_r", vmin=0, vmax=1)
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    active_cells = int(np.count_nonzero(mask_array))
    # Compute or store total_cells so later code can reuse this named value clearly.
    total_cells = int(mask_array.size)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_title(f"{title} ({active_label}: {active_cells}/{total_cells})")
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_xlabel("Prediction window index")
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_ylabel("Reference window index")
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_xlim(-0.5, mask_array.shape[1] - 0.5)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    axis.set_ylim(mask_array.shape[0] - 0.5, -0.5)


# Define the draw_segment_box function; its body below performs one named step of the pipeline.
def draw_segment_box(
    # Define the axis field so this data object records that value explicitly.
    axis: Any,
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the x0 field so this data object records that value explicitly.
    x0: float,
    # Define the y0 field so this data object records that value explicitly.
    y0: float,
    # Define the x1 field so this data object records that value explicitly.
    x1: float,
    # Define the y1 field so this data object records that value explicitly.
    y1: float,
    # Define the color field so this data object records that value explicitly.
    color: str,
    # Define the label field so this data object records that value explicitly.
    label: str | None,
    # Define the linewidth field so this data object records that value explicitly.
    linewidth: float,
    # Define the alpha field so this data object records that value explicitly.
    alpha: float,
    # Define the linestyle field so this data object records that value explicitly.
    linestyle: str,
    # Define the half_width_cells field so this data object records that value explicitly.
    half_width_cells: float,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> None:
    """Draw a hollow box around a detected line so the score ridge stays visible."""
    from matplotlib.patches import Polygon

    # Compute or store box_points so later code can reuse this named value clearly.
    box_points = oriented_segment_box_points(
        # Pass the x0 argument into the surrounding call so the callee receives that setting explicitly.
        x0=x0,
        # Pass the y0 argument into the surrounding call so the callee receives that setting explicitly.
        y0=y0,
        # Pass the x1 argument into the surrounding call so the callee receives that setting explicitly.
        x1=x1,
        # Pass the y1 argument into the surrounding call so the callee receives that setting explicitly.
        y1=y1,
        # Pass the half_width_cells argument into the surrounding call so the callee receives that setting explicitly.
        half_width_cells=half_width_cells,
        # Pass the end_padding_cells argument into the surrounding call so the callee receives that setting explicitly.
        end_padding_cells=0.75,
    )
    # Start a multi-line call or data structure so related arguments stay readable.
    axis.add_patch(
        # Start a multi-line call or data structure so related arguments stay readable.
        Polygon(
            # Pass this value into the surrounding multi-line call or collection.
            box_points,
            # Pass the closed argument into the surrounding call so the callee receives that setting explicitly.
            closed=True,
            # Pass the fill argument into the surrounding call so the callee receives that setting explicitly.
            fill=False,
            # Pass the edgecolor argument into the surrounding call so the callee receives that setting explicitly.
            edgecolor=color,
            # Pass the linewidth argument into the surrounding call so the callee receives that setting explicitly.
            linewidth=linewidth,
            # Pass the alpha argument into the surrounding call so the callee receives that setting explicitly.
            alpha=alpha,
            # Pass the linestyle argument into the surrounding call so the callee receives that setting explicitly.
            linestyle=linestyle,
            # Pass the label argument into the surrounding call so the callee receives that setting explicitly.
            label=label,
            # Pass the joinstyle argument into the surrounding call so the callee receives that setting explicitly.
            joinstyle="miter",
            # Pass the zorder argument into the surrounding call so the callee receives that setting explicitly.
            zorder=6,
        )
    )


# Define the draw_line_label_near_segment function; its body below performs one named step of the pipeline.
def draw_line_label_near_segment(axis: Any, *, x0: float, y0: float, x1: float, y1: float, label_text: str) -> None:
    """Draw a compact line label next to the detected segment."""
    # Execute this statement as the next small step in the surrounding pipeline logic.
    _, (normal_x, normal_y) = segment_unit_vectors(x0=x0, y0=y0, x1=x1, y1=y1)
    # Compute or store midpoint_x so later code can reuse this named value clearly.
    midpoint_x = (float(x0) + float(x1)) / 2.0 + normal_x * 4.0
    # Compute or store midpoint_y so later code can reuse this named value clearly.
    midpoint_y = (float(y0) + float(y1)) / 2.0 + normal_y * 4.0
    # Start a multi-line call or data structure so related arguments stay readable.
    axis.text(
        # Pass this value into the surrounding multi-line call or collection.
        midpoint_x,
        # Pass this value into the surrounding multi-line call or collection.
        midpoint_y,
        # Pass this value into the surrounding multi-line call or collection.
        str(label_text),
        # Pass the color argument into the surrounding call so the callee receives that setting explicitly.
        color="#FFE066",
        # Pass the fontsize argument into the surrounding call so the callee receives that setting explicitly.
        fontsize=8,
        # Pass the fontweight argument into the surrounding call so the callee receives that setting explicitly.
        fontweight="bold",
        # Pass the ha argument into the surrounding call so the callee receives that setting explicitly.
        ha="center",
        # Pass the va argument into the surrounding call so the callee receives that setting explicitly.
        va="center",
        # Pass the bbox argument into the surrounding call so the callee receives that setting explicitly.
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "edgecolor": "#FFE066", "linewidth": 0.45, "alpha": 0.72},
        # Pass the zorder argument into the surrounding call so the callee receives that setting explicitly.
        zorder=8,
    )


# Define the draw_endpoint_segments function; its body below performs one named step of the pipeline.
def draw_endpoint_segments(
    # Define the axis field so this data object records that value explicitly.
    axis: Any,
    # Define the endpoint_segments field so this data object records that value explicitly.
    endpoint_segments: Iterable[tuple[float, float, float, float]],
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the color field so this data object records that value explicitly.
    color: str,
    # Define the legend_label field so this data object records that value explicitly.
    legend_label: str,
    # Define the linewidth field so this data object records that value explicitly.
    linewidth: float,
    # Define the alpha field so this data object records that value explicitly.
    alpha: float,
    # Define the linestyle field so this data object records that value explicitly.
    linestyle: str,
    # Define the segment_labels field so this data object records that value explicitly.
    segment_labels: Sequence[str] | None,
    # Define the show_line_ids field; it stores whether raw and final line labels are printed on plot overlays.
    show_line_ids: bool,
    # Define the half_width_cells field so this data object records that value explicitly.
    half_width_cells: float,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> int:
    """Draw all detected segments as boxes and return how many were drawn."""
    # Compute or store drawn_count so later code can reuse this named value clearly.
    drawn_count = 0
    # Compute or store labels so later code can reuse this named value clearly.
    labels = list(segment_labels or [])
    # Iterate over segment_index, (x0, y0, x1, y1) in enumerate(endpoint_segments) so each item is processed with the same logic.
    for segment_index, (x0, y0, x1, y1) in enumerate(endpoint_segments):
        # Start a multi-line call or data structure so related arguments stay readable.
        draw_segment_box(
            # Pass this value into the surrounding multi-line call or collection.
            axis,
            # Pass the x0 argument into the surrounding call so the callee receives that setting explicitly.
            x0=x0,
            # Pass the y0 argument into the surrounding call so the callee receives that setting explicitly.
            y0=y0,
            # Pass the x1 argument into the surrounding call so the callee receives that setting explicitly.
            x1=x1,
            # Pass the y1 argument into the surrounding call so the callee receives that setting explicitly.
            y1=y1,
            # Pass the color argument into the surrounding call so the callee receives that setting explicitly.
            color=color,
            # Pass the label argument into the surrounding call so the callee receives that setting explicitly.
            label=legend_label if drawn_count == 0 else None,
            # Pass the linewidth argument into the surrounding call so the callee receives that setting explicitly.
            linewidth=linewidth,
            # Pass the alpha argument into the surrounding call so the callee receives that setting explicitly.
            alpha=alpha,
            # Pass the linestyle argument into the surrounding call so the callee receives that setting explicitly.
            linestyle=linestyle,
            # Pass the half_width_cells argument into the surrounding call so the callee receives that setting explicitly.
            half_width_cells=half_width_cells,
        )
        # Check whether show_line_ids and segment_index < len(labels); the indented block handles that specific case.
        if show_line_ids and segment_index < len(labels):
            # Execute this statement as the next small step in the surrounding pipeline logic.
            draw_line_label_near_segment(axis, x0=x0, y0=y0, x1=x1, y1=y1, label_text=labels[segment_index])
        # Compute or store drawn_count + so later code can reuse this named value clearly.
        drawn_count += 1
    # Return this computed value to the caller so the next pipeline stage can use it.
    return drawn_count


# Define the draw_raw_hough_overlay function; its body below performs one named step of the pipeline.
def draw_raw_hough_overlay(axis: Any, raw_lines: Sequence[Any], *, show_line_ids: bool) -> int:
    """Draw raw Hough line candidates on an existing score-matrix panel."""
    # Compute or store endpoint_segments: list[tuple[float, float, float, float]] so later code can reuse this named value clearly.
    endpoint_segments: list[tuple[float, float, float, float]] = []
    # Compute or store segment_labels: list[str] so later code can reuse this named value clearly.
    segment_labels: list[str] = []
    # Iterate over raw_line_index, raw_line in enumerate(raw_lines or []) so each item is processed with the same logic.
    for raw_line_index, raw_line in enumerate(raw_lines or []):
        # Compute or store endpoint_tuple so later code can reuse this named value clearly.
        endpoint_tuple = endpoint_tuple_from_raw_hough_segment(raw_line)
        # Check whether endpoint_tuple is None; the indented block handles that specific case.
        if endpoint_tuple is None:
            # Skip the rest of this loop iteration and move to the next item.
            continue
        # Add this item to the list that is accumulating results for later output.
        endpoint_segments.append(endpoint_tuple)
        # Add this item to the list that is accumulating results for later output.
        segment_labels.append(str(int(raw_line_index)))
    # Return this computed value to the caller so the next pipeline stage can use it.
    return draw_endpoint_segments(
        # Pass this value into the surrounding multi-line call or collection.
        axis,
        # Pass this value into the surrounding multi-line call or collection.
        endpoint_segments,
        # Pass the color argument into the surrounding call so the callee receives that setting explicitly.
        color=RAW_HOUGH_LINE_COLOR,
        # Pass the legend_label argument into the surrounding call so the callee receives that setting explicitly.
        legend_label="Raw Hough segment",
        # Pass the linewidth argument into the surrounding call so the callee receives that setting explicitly.
        linewidth=1.8,
        # Pass the alpha argument into the surrounding call so the callee receives that setting explicitly.
        alpha=0.85,
        # Pass the linestyle argument into the surrounding call so the callee receives that setting explicitly.
        linestyle="-",
        # Pass the segment_labels argument into the surrounding call so the callee receives that setting explicitly.
        segment_labels=segment_labels,
        # Pass show_line_ids into the surrounding call; this supplies whether raw and final line labels are printed on plot overlays.
        show_line_ids=show_line_ids,
        # Pass the half_width_cells argument into the surrounding call so the callee receives that setting explicitly.
        half_width_cells=RAW_LINE_BOX_HALF_WIDTH_CELLS,
    )


# Define the draw_final_line_overlay function; its body below performs one named step of the pipeline.
def draw_final_line_overlay(axis: Any, final_lines: Sequence[Any], *, show_line_ids: bool) -> int:
    """Draw final surviving lines on an existing score-matrix panel."""
    # Compute or store endpoint_segments: list[tuple[float, float, float, float]] so later code can reuse this named value clearly.
    endpoint_segments: list[tuple[float, float, float, float]] = []
    # Compute or store segment_labels: list[str] so later code can reuse this named value clearly.
    segment_labels: list[str] = []
    # Iterate over final_line_index, line_record in enumerate(final_lines or []) so each item is processed with the same logic.
    for final_line_index, line_record in enumerate(final_lines or []):
        # Compute or store endpoint_tuple so later code can reuse this named value clearly.
        endpoint_tuple = endpoint_tuple_from_filtered_line_record(line_record)
        # Check whether endpoint_tuple is None; the indented block handles that specific case.
        if endpoint_tuple is None:
            # Skip the rest of this loop iteration and move to the next item.
            continue
        # Compute or store raw_line_ids so later code can reuse this named value clearly.
        raw_line_ids = raw_source_line_ids_from_filtered_line_record(line_record)
        # Add this item to the list that is accumulating results for later output.
        endpoint_segments.append(endpoint_tuple)
        # Add this item to the list that is accumulating results for later output.
        segment_labels.append(f"F{final_line_index} <- {compact_identifier_sequence(raw_line_ids)}")
    # Return this computed value to the caller so the next pipeline stage can use it.
    return draw_endpoint_segments(
        # Pass this value into the surrounding multi-line call or collection.
        axis,
        # Pass this value into the surrounding multi-line call or collection.
        endpoint_segments,
        # Pass the color argument into the surrounding call so the callee receives that setting explicitly.
        color=FINAL_LINE_COLOR,
        # Pass the legend_label argument into the surrounding call so the callee receives that setting explicitly.
        legend_label="Surviving line after filtering",
        # Pass the linewidth argument into the surrounding call so the callee receives that setting explicitly.
        linewidth=2.0,
        # Pass the alpha argument into the surrounding call so the callee receives that setting explicitly.
        alpha=0.95,
        # Pass the linestyle argument into the surrounding call so the callee receives that setting explicitly.
        linestyle="-",
        # Pass the segment_labels argument into the surrounding call so the callee receives that setting explicitly.
        segment_labels=segment_labels,
        # Pass show_line_ids into the surrounding call; this supplies whether raw and final line labels are printed on plot overlays.
        show_line_ids=show_line_ids,
        # Pass the half_width_cells argument into the surrounding call so the callee receives that setting explicitly.
        half_width_cells=FINAL_LINE_BOX_HALF_WIDTH_CELLS,
    )


# Define the format_metric_line function; its body below performs one named step of the pipeline.
def format_metric_line(label: str, value: Any) -> str:
    """Format one metric value for the bottom text band."""
    # Check whether isinstance(value, float); the indented block handles that specific case.
    if isinstance(value, float):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return f"{label}: {value:.6f}"
    # Return this computed value to the caller so the next pipeline stage can use it.
    return f"{label}: {value}"


# Define the build_metrics_text function; its body below performs one named step of the pipeline.
def build_metrics_text(result_row: dict[str, Any]) -> str:
    """Build the compact metrics band shown below each document panel."""
    # Compute or store lines so later code can reuse this named value clearly.
    lines = [
        # Pass this value into the surrounding multi-line call or collection.
        format_metric_line("document_normalised_levenshtein", result_row.get("document_normalised_levenshtein")),
        # Pass this value into the surrounding multi-line call or collection.
        format_metric_line("weighted_along_lines_normalised_levenshtein", result_row.get("weighted_along_lines_normalised_levenshtein")),
        # Pass this value into the surrounding multi-line call or collection.
        format_metric_line("correct_ref_coverage", result_row.get("correct_ref_coverage")),
        # Pass this value into the surrounding multi-line call or collection.
        format_metric_line("missing_ref_coverage", result_row.get("missing_ref_coverage")),
        # Pass this value into the surrounding multi-line call or collection.
        format_metric_line("repetition_on_reference", result_row.get("repetition_on_reference")),
        # Pass this value into the surrounding multi-line call or collection.
        format_metric_line("hallucination", result_row.get("hallucination")),
        # Provide this literal text value to the surrounding path, message, or argument definition.
        "",
        # Compute or store f"score_floor_alpha so later code can reuse this named value clearly.
        f"score_floor_alpha={result_row.get('score_floor_alpha')} | score_floor_ref_to_pred={result_row.get('score_floor_ref_to_pred')}",
        # Compute or store f"hough_threshold so later code can reuse this named value clearly.
        f"hough_threshold={result_row.get('hough_threshold')} | line_length={result_row.get('hough_line_length')} | line_gap={result_row.get('hough_line_gap')} | seed={result_row.get('hough_seed')}",
        # Compute or store f"raw_lines so later code can reuse this named value clearly.
        f"raw_lines={result_row.get('raw_line_count')} | candidates={result_row.get('candidate_line_count')} | final_lines={result_row.get('used_line_count')}",
    ]
    # Return this computed value to the caller so the next pipeline stage can use it.
    return "\n".join(lines)


# Define the render_document_panel function; its body below performs one named step of the pipeline.
def render_document_panel(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the plot_payload field so this data object records that value explicitly.
    plot_payload: dict[str, Any],
    # Define the output_path field so this data object records that value explicitly.
    output_path: Path,
    # Define the saved_figure_dpi field; it stores the resolution used when saving plot images.
    saved_figure_dpi: int,
    # Define the show_line_ids field; it stores whether raw and final line labels are printed on plot overlays.
    show_line_ids: bool,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> Path:
    """Render one document panel and save it as a temporary PNG."""
    import matplotlib

    # Execute this statement as the next small step in the surrounding pipeline logic.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Compute or store output_path so later code can reuse this named value clearly.
    output_path = Path(output_path)
    # Ensure the target directory exists before later code tries to write files into it.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Compute or store document so later code can reuse this named value clearly.
    document = plot_payload["document"]
    # Compute or store result_row so later code can reuse this named value clearly.
    result_row = plot_payload["result_row"]

    # Compute or store fig so later code can reuse this named value clearly.
    fig = plt.figure(figsize=(20, 24.8), constrained_layout=False)
    # Compute or store grid_specification so later code can reuse this named value clearly.
    grid_specification = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 0.42], hspace=0.30, wspace=0.18)
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    axes = np.array(
        # Start a multi-line collection so related values can be listed clearly.
        [
            # Pass this value into the surrounding multi-line call or collection.
            [fig.add_subplot(grid_specification[0, 0]), fig.add_subplot(grid_specification[0, 1])],
            # Pass this value into the surrounding multi-line call or collection.
            [fig.add_subplot(grid_specification[1, 0]), fig.add_subplot(grid_specification[1, 1])],
        ],
        # Pass the dtype argument into the surrounding call so the callee receives that setting explicitly.
        dtype=object,
    )
    # Compute or store hough_input_axis so later code can reuse this named value clearly.
    hough_input_axis = fig.add_subplot(grid_specification[2, :])
    # Compute or store metrics_axis so later code can reuse this named value clearly.
    metrics_axis = fig.add_subplot(grid_specification[3, :])

    # Start a multi-line call or data structure so related arguments stay readable.
    fig.suptitle(
        # Execute this statement as the next small step in the surrounding pipeline logic.
        f"{document.main_language} / {document.document_type} | {document.fname}\n"
        # Provide this literal text value to the surrounding path, message, or argument definition.
        "Score matrices, ref-to-pred Hough overlays, and final Hough input",
        # Pass the fontsize argument into the surrounding call so the callee receives that setting explicitly.
        fontsize=15,
        # Pass the y argument into the surrounding call so the callee receives that setting explicitly.
        y=0.995,
    )

    # Compute or store image_0 so later code can reuse this named value clearly.
    image_0 = draw_score_matrix_heatmap(axes[0, 0], plot_payload.get("ref_to_pred_score_matrix"), "ref_to_pred score matrix without Hough")
    # Compute or store image_1 so later code can reuse this named value clearly.
    image_1 = draw_score_matrix_heatmap(axes[0, 1], plot_payload.get("ref_to_ref_score_matrix"), "ref_to_ref score matrix without Hough")
    # Compute or store image_2 so later code can reuse this named value clearly.
    image_2 = draw_score_matrix_heatmap(axes[1, 0], plot_payload.get("ref_to_pred_score_matrix"), "Raw Hough lines on ref_to_pred")
    # Compute or store raw_count so later code can reuse this named value clearly.
    raw_count = draw_raw_hough_overlay(axes[1, 0], plot_payload.get("raw_ref_to_pred_hough_lines") or [], show_line_ids=show_line_ids)
    # Check whether raw_count > 0; the indented block handles that specific case.
    if raw_count > 0:
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axes[1, 0].legend(loc="upper right")

    # Compute or store image_3 so later code can reuse this named value clearly.
    image_3 = draw_score_matrix_heatmap(axes[1, 1], plot_payload.get("ref_to_pred_score_matrix"), "Surviving lines after filtering on ref_to_pred")
    # Compute or store final_count so later code can reuse this named value clearly.
    final_count = draw_final_line_overlay(axes[1, 1], plot_payload.get("final_surviving_ref_to_pred_lines") or [], show_line_ids=show_line_ids)
    # Check whether final_count > 0; the indented block handles that specific case.
    if final_count > 0:
        # Execute this statement as the next small step in the surrounding pipeline logic.
        axes[1, 1].legend(loc="upper right")

    # The score-floor mask is passed directly to Hough and shown here as the final binary input.
    draw_binary_mask_panel(hough_input_axis, plot_payload.get("ref_to_pred_hough_input_mask"), "ref_to_pred final Hough input", "voters")

    # Iterate over axis, image in zip(axes.ravel(), [image_0, image_1, image_2, image_3]) so each item is processed with the same logic.
    for axis, image in zip(axes.ravel(), [image_0, image_1, image_2, image_3]):
        # Check whether image is not None; the indented block handles that specific case.
        if image is not None:
            # Execute this statement as the next small step in the surrounding pipeline logic.
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Score")

    # Execute this statement as the next small step in the surrounding pipeline logic.
    metrics_axis.set_axis_off()
    # Start a multi-line call or data structure so related arguments stay readable.
    metrics_axis.text(
        # Pass this value into the surrounding multi-line call or collection.
        0.01,
        # Pass this value into the surrounding multi-line call or collection.
        0.98,
        # Pass this value into the surrounding multi-line call or collection.
        build_metrics_text(result_row),
        # Pass the transform argument into the surrounding call so the callee receives that setting explicitly.
        transform=metrics_axis.transAxes,
        # Pass the ha argument into the surrounding call so the callee receives that setting explicitly.
        ha="left",
        # Pass the va argument into the surrounding call so the callee receives that setting explicitly.
        va="top",
        # Pass the fontsize argument into the surrounding call so the callee receives that setting explicitly.
        fontsize=10,
        # Pass the family argument into the surrounding call so the callee receives that setting explicitly.
        family="monospace",
        # Pass the linespacing argument into the surrounding call so the callee receives that setting explicitly.
        linespacing=1.25,
    )

    # Execute this statement as the next small step in the surrounding pipeline logic.
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.035, top=0.93, hspace=0.34, wspace=0.20)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    fig.savefig(output_path, dpi=saved_figure_dpi, bbox_inches="tight")
    # Execute this statement as the next small step in the surrounding pipeline logic.
    plt.close(fig)
    # Return this computed value to the caller so the next pipeline stage can use it.
    return output_path


__all__ = ["render_document_panel", "safe_path_component"]
