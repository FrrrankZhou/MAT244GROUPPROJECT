"""Plots for the earthquake-response course project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ode_solver import (
    DampingStiffnessStudy,
    HorizontalResponse,
    ParameterStudy,
    PeakResponse,
)


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


def plot_parameter_study(
    study: ParameterStudy,
    method: str,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot how r_max changes when m, d, and k vary one at a time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = (
        (study.mass_values, study.mass_max_displacements,
         "Mass, m (kg)", "Mass sensitivity"),
        (study.damping_values, study.damping_max_displacements,
         "Damping, d (N·s/m)", "Damping sensitivity"),
        (study.stiffness_values, study.stiffness_max_displacements,
         "Stiffness, k (N/m)", "Stiffness sensitivity"),
    )
    for axis, (values, maxima, xlabel, title) in zip(axes, panels):
        axis.plot(values, maxima, marker="o", linewidth=1.2)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Maximum horizontal displacement (m)")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))

    fig.suptitle(
        f"One-at-a-time parameter study - {method.upper()} "
        "(other parameters held constant)"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_method_displacement_comparison(
    euler: HorizontalResponse,
    rk4: HorizontalResponse,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Compare Euler and RK4 total horizontal displacement on one plot."""
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(euler.time, euler.total_displacement, label="Euler", linewidth=0.9)
    axis.plot(rk4.time, rk4.total_displacement, label="RK4", linewidth=0.9)
    axis.set_title("Total horizontal displacement: Euler vs RK4")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("r(t) (m)")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_method_velocity_comparison(
    euler: HorizontalResponse,
    rk4: HorizontalResponse,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Compare Euler and RK4 total horizontal speed on one plot."""
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(euler.time, euler.total_velocity, label="Euler", linewidth=0.9)
    axis.plot(rk4.time, rk4.total_velocity, label="RK4", linewidth=0.9)
    axis.set_title("Total horizontal velocity magnitude: Euler vs RK4")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Horizontal speed (m/s)")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_simple_harmonic_validation(
    time: NDArray[np.float64],
    exact_east: NDArray[np.float64],
    exact_north: NDArray[np.float64],
    euler: HorizontalResponse,
    rk4: HorizontalResponse,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Compare exact, Euler, and RK4 solutions for 2D simple harmonic motion."""
    exact_total = np.hypot(exact_east, exact_north)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    panels = (
        (exact_east, euler.displacement_east, rk4.displacement_east,
         "East displacement", "x_E(t) (m)"),
        (exact_north, euler.displacement_north, rk4.displacement_north,
         "North displacement", "x_N(t) (m)"),
        (exact_total, euler.total_displacement, rk4.total_displacement,
         "Total horizontal displacement", "r(t) (m)"),
    )
    for axis, (exact, euler_values, rk4_values, title, ylabel) in zip(axes, panels):
        axis.plot(time, exact, label="Exact", color="black", linewidth=2.0)
        axis.plot(time, euler_values, label="Euler", linestyle="--", linewidth=1.0)
        axis.plot(time, rk4_values, label="RK4", linestyle=":", linewidth=1.5)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("2D simple harmonic motion: Exact vs Euler vs RK4")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_damping_stiffness_heatmap(
    study: DampingStiffnessStudy,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot r_max against damping and stiffness for one fixed mass."""
    fig, axis = plt.subplots(figsize=(9, 7))
    heatmap = axis.pcolormesh(
        study.damping_values,
        study.stiffness_values,
        study.maximum_displacements,
        cmap="Reds",
        shading="nearest",
    )
    axis.set_xlabel("Damping, d (N·s/m)")
    axis.set_ylabel("Stiffness, k (N/m)")
    axis.set_title(f"Maximum displacement at fixed mass m = {study.mass:.3g} kg")
    axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4))
    colorbar = fig.colorbar(heatmap, ax=axis)
    colorbar.set_label("Maximum horizontal displacement, r_max (m)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig
