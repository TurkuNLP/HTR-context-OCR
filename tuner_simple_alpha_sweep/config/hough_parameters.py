from __future__ import annotations

"""Validated probabilistic Hough parameter values."""

from dataclasses import dataclass


# Ask Python to generate common data-container methods for the class defined next.
@dataclass(frozen=True)
# Define the ProbabilisticHoughParameters class, which groups related state and behavior for this part of the pipeline.
class ProbabilisticHoughParameters:
    """One fixed Hough parameter set applied to every selected document."""

    # Define the hough_threshold field so this data object records that value explicitly.
    hough_threshold: int
    # Define the hough_line_length field so this data object records that value explicitly.
    hough_line_length: int
    # Define the hough_line_gap field so this data object records that value explicitly.
    hough_line_gap: int
    # Define the hough_seed field so this data object records that value explicitly.
    hough_seed: int

    # Define the validate function; its body below performs one named step of the pipeline.
    def validate(self) -> "ProbabilisticHoughParameters":
        """Return this object after checking that every value is usable."""
        # Check whether int(self.hough_threshold) < 0; the indented block handles that specific case.
        if int(self.hough_threshold) < 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--hough-threshold must be zero or positive")
        # Check whether int(self.hough_line_length) < 0; the indented block handles that specific case.
        if int(self.hough_line_length) < 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--hough-line-length must be zero or positive")
        # Check whether int(self.hough_line_gap) < 0; the indented block handles that specific case.
        if int(self.hough_line_gap) < 0:
            # Stop execution for this invalid state by raising an explicit exception.
            raise ValueError("--hough-line-gap must be zero or positive")
        # Return this computed value to the caller so the next pipeline stage can use it.
        return self


__all__ = ["ProbabilisticHoughParameters"]
