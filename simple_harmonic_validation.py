"""Validate the 2D ODE solver against an exact simple harmonic solution."""

import csv
from pathlib import Path

import numpy as np

from data_reader import GroundMotion2D
from ode_solver import BuildingParameters, simulate_2d
from plotting import plot_simple_harmonic_validation


RESULTS_DIR = Path("results")
MASS = 1.0
NATURAL_PERIOD = 1.0
DURATION = 5.0
DT = 0.01


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    omega = 2.0 * np.pi / NATURAL_PERIOD
    time = np.arange(0.0, DURATION + 0.5 * DT, DT)
    zeros = np.zeros_like(time)
    motion = GroundMotion2D(time, zeros, zeros, DT, "zero input East", "zero input North")
    parameters = BuildingParameters(mass=MASS, stiffness=MASS * omega**2, damping=0.0)

    # Exact circular 2D harmonic motion:
    # x_E(t) = cos(omega*t), x_N(t) = sin(omega*t), r(t) = 1.
    initial_conditions = dict(
        initial_displacement_east=1.0,
        initial_velocity_east=0.0,
        initial_displacement_north=0.0,
        initial_velocity_north=omega,
    )
    euler = simulate_2d(motion, parameters, method="euler", **initial_conditions)
    rk4 = simulate_2d(motion, parameters, method="rk4", **initial_conditions)
    exact_east = np.cos(omega * time)
    exact_north = np.sin(omega * time)
    exact_total = np.ones_like(time)

    plot_simple_harmonic_validation(
        time,
        exact_east,
        exact_north,
        euler,
        rk4,
        save_path=RESULTS_DIR / "simple_harmonic_validation.png",
        show=False,
    )

    rows = []
    for method, result in (("Euler", euler), ("RK4", rk4)):
        east_error = float(np.max(np.abs(result.displacement_east - exact_east)))
        north_error = float(np.max(np.abs(result.displacement_north - exact_north)))
        total_error = float(np.max(np.abs(result.total_displacement - exact_total)))
        rows.append((method, east_error, north_error, total_error))

    csv_path = RESULTS_DIR / "simple_harmonic_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "method",
            "max_abs_error_east_m",
            "max_abs_error_north_m",
            "max_abs_error_total_m",
        ])
        writer.writerows(rows)

    print("2D simple harmonic motion validation")
    print(f"omega = {omega:.6f} rad/s, dt = {DT:.4f} s, duration = {DURATION:.1f} s")
    for method, east_error, north_error, total_error in rows:
        print(
            f"{method}: max errors East={east_error:.6e}, "
            f"North={north_error:.6e}, total={total_error:.6e}"
        )
    print(f"Saved figure: {RESULTS_DIR / 'simple_harmonic_validation.png'}")
    print(f"Saved error table: {csv_path}")


if __name__ == "__main__":
    main()
