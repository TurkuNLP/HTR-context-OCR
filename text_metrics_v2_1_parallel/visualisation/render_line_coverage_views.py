"""Render line-coverage diagnostics on reference and prediction axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_y_axis_count_visualisation(
    *,
    refref_y: np.ndarray,
    other_y: np.ndarray,
    y_diff: np.ndarray,
    out_path: Path,
    file_name: str,
) -> str:
    """Render Y-axis count profiles and subtraction diagnostics."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    refref_y = np.asarray(refref_y, dtype=np.int32)
    other_y = np.asarray(other_y, dtype=np.int32)
    y_diff = np.asarray(y_diff, dtype=np.int32)

    xs = np.arange(refref_y.shape[0], dtype=np.int32)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)

    axes[0].plot(xs, refref_y, color="#1f77b4", linewidth=1.2, label="ref_to_ref_y")
    axes[0].plot(xs, other_y, color="#2ca02c", linewidth=1.0, alpha=0.85, label="ref_to_other_y")
    axes[0].set_ylabel("coverage count")
    axes[0].set_title(f"{file_name} | count_line_coverage (y axis)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(xs, y_diff, color="#d62728", linewidth=1.0, label="y_diff = other_y - refref_y")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axes[1].set_xlabel("character index on reference axis")
    axes[1].set_ylabel("y_diff")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_path)


def save_x_axis_count_visualisation(
    *,
    other_x: np.ndarray,
    out_path: Path,
    file_name: str,
) -> str:
    """Render X-axis coverage profile used for hallucination diagnostics."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    other_x = np.asarray(other_x, dtype=np.int32)
    xs = np.arange(other_x.shape[0], dtype=np.int32)

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.8))
    ax.plot(xs, other_x, color="#9467bd", linewidth=1.0, label="ref_to_other_x")

    zero_mask = other_x == 0
    if np.any(zero_mask):
        ax.scatter(
            xs[zero_mask],
            np.zeros(int(np.count_nonzero(zero_mask))),
            s=4,
            color="#ff7f0e",
            alpha=0.8,
            label="x==0",
        )

    ax.set_xlabel("character index on other/prediction axis")
    ax.set_ylabel("coverage count")
    ax.set_title(f"{file_name} | count_line_coverage (x axis)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_path)


def save_count_line_coverage_visualisations(
    *,
    coverage_refref_y: np.ndarray,
    coverage_other_y: np.ndarray,
    coverage_other_x: np.ndarray,
    coverage_y_diff: np.ndarray,
    case_prefix: str,
    file_name: str,
    output_dir: Path,
) -> dict:
    """Save Y and X count-line-coverage visualisations for one document."""
    output_dir = Path(output_dir)
    vis_y_dir = output_dir / "visualise_count_line_coverage_y"
    vis_x_dir = output_dir / "visualise_count_line_coverage_x"

    y_path = save_y_axis_count_visualisation(
        refref_y=np.asarray(coverage_refref_y, dtype=np.int32),
        other_y=np.asarray(coverage_other_y, dtype=np.int32),
        y_diff=np.asarray(coverage_y_diff, dtype=np.int32),
        out_path=vis_y_dir / f"{case_prefix}_count_line_coverage_y.png",
        file_name=file_name,
    )
    x_path = save_x_axis_count_visualisation(
        other_x=np.asarray(coverage_other_x, dtype=np.int32),
        out_path=vis_x_dir / f"{case_prefix}_count_line_coverage_x.png",
        file_name=file_name,
    )

    return {
        "visualise_count_line_coverage_y_path": y_path,
        "visualise_count_line_coverage_x_path": x_path,
    }
