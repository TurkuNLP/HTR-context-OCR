from __future__ import annotations

"""Window-count and matrix-shape helpers."""


# Define the count_sliding_windows function; its body below performs one named step of the pipeline.
def count_sliding_windows(text: str, *, window_size: int, window_stride: int) -> int:
    """Return how many fixed-size windows fit in one text string."""
    # Compute or store text_length so later code can reuse this named value clearly.
    text_length = len(str(text))
    # Check whether text_length < int(window_size); the indented block handles that specific case.
    if text_length < int(window_size):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return 0
    # Return this computed value to the caller so the next pipeline stage can use it.
    return ((text_length - int(window_size)) // int(window_stride)) + 1


# Define the sliding_text_windows function; its body below performs one named step of the pipeline.
def sliding_text_windows(text: str, *, window_size: int, window_stride: int) -> list[str]:
    """Return the exact text windows represented by matrix rows or columns."""
    # Compute or store source_text so later code can reuse this named value clearly.
    source_text = str(text)
    # Check whether len(source_text) < int(window_size); the indented block handles that specific case.
    if len(source_text) < int(window_size):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return []
    # Return this computed value to the caller so the next pipeline stage can use it.
    return [
        # Execute this statement as the next small step in the surrounding pipeline logic.
        source_text[start_index : start_index + int(window_size)]
        # Iterate over start_index in range(0, len(source_text) - int(window_size) + 1, int(window_stride)) so each item is processed with the same logic.
        for start_index in range(0, len(source_text) - int(window_size) + 1, int(window_stride))
    ]


# Define the matrix_is_large_enough function; its body below performs one named step of the pipeline.
def matrix_is_large_enough(matrix_shape: tuple[int, int], *, minimum_rows: int, minimum_columns: int) -> bool:
    """Return True when a matrix can reasonably be passed to Hough detection."""
    # Compute or store row_count, column_count so later code can reuse this named value clearly.
    row_count, column_count = int(matrix_shape[0]), int(matrix_shape[1])
    # Return this computed value to the caller so the next pipeline stage can use it.
    return row_count >= int(minimum_rows) and column_count >= int(minimum_columns)


# Define the matrix_size_skip_reason function; its body below performs one named step of the pipeline.
def matrix_size_skip_reason(matrix_shape: tuple[int, int], *, minimum_rows: int, minimum_columns: int) -> str | None:
    """Return a human-readable skip reason for a too-small matrix."""
    # Compute or store row_count, column_count so later code can reuse this named value clearly.
    row_count, column_count = int(matrix_shape[0]), int(matrix_shape[1])
    # Check whether row_count < int(minimum_rows); the indented block handles that specific case.
    if row_count < int(minimum_rows):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return f"matrix_rows_below_minimum:rows={row_count}:minimum={int(minimum_rows)}"
    # Check whether column_count < int(minimum_columns); the indented block handles that specific case.
    if column_count < int(minimum_columns):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return f"matrix_columns_below_minimum:columns={column_count}:minimum={int(minimum_columns)}"
    # Return this computed value to the caller so the next pipeline stage can use it.
    return None


__all__ = [
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "count_sliding_windows",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "matrix_is_large_enough",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "matrix_size_skip_reason",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "sliding_text_windows",
]
