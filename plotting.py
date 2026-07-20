"""Plots for the earthquake-response course project."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ode_solver import HorizontalResponse, PeakResponse


def plot_response(
    response: HorizontalResponse,
    peak: PeakResponse,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(response.time, response.acceleration_east, label="HNE", linewidth=0.8)
    axes[0].plot(response.time, response.acceleration_north, label="HNN", linewidth=0.8)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title("Ground acceleration")
    axes[0].legend()

    axes[1].plot(response.time, response.displacement_east, label="East", linewidth=0.9)
    axes[1].plot(response.time, response.displacement_north, label="North", linewidth=0.9)
    axes[1].axvline(peak.time, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Relative displacement (m)")
    axes[1].set_title(f"Building displacement ({response.method.upper()})")
    axes[1].legend()

    axes[2].plot(response.time, response.total_displacement, color="purple", linewidth=1.0)
    axes[2].scatter(
        peak.time, peak.maximum_displacement, color="red", zorder=3,
        label=f"max = {peak.maximum_displacement:.4g} m at {peak.time:.3f} s",
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("r(t) (m)")
    axes[2].set_title("Total horizontal displacement")
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig
