"""Run the three-week ODE course project example."""

import numpy as np

from data_reader import load_horizontal_motion
from ode_solver import (
    BuildingParameters,
    find_peak,
    simulate_2d,
    study_damping_stiffness,
    study_parameters,
)
from plotting import plot_damping_stiffness_heatmap, plot_parameter_study, plot_response


# Change these values to perform different experiments.
DATA_FILE = "Calama.zip"
MASS_KG = 1_000_000.0
NATURAL_PERIOD_S = 1.0
DAMPING_RATIO = 0.05
METHOD = "rk4"  # "euler" or "rk4"

# Use more samples near the current building parameters (factor = 1).  This
# gives better resolution where the natural frequency is close to 1 Hz and a
# resonance-related peak may occur.
PARAMETER_FACTORS = np.array([
    0.50, 0.625, 0.75, 0.80, 0.85, 0.90,
    0.925, 0.95, 0.975, 1.00, 1.025, 1.05, 1.075, 1.10,
    1.15, 1.25, 1.375, 1.50,
])
GENERATE_HEATMAP = False


def main() -> None:
    motion = load_horizontal_motion(DATA_FILE)
    parameters = BuildingParameters.from_period(
        mass=MASS_KG,
        natural_period=NATURAL_PERIOD_S,
        damping_ratio=DAMPING_RATIO,
    )
    response = simulate_2d(motion, parameters, method=METHOD)
    peak = find_peak(response)

    print(f"Method: {response.method.upper()}")
    print(f"Samples: {len(response.time)}")
    print(f"Time step: {motion.dt:.6f} s")
    print(f"Natural period: {parameters.natural_period:.3f} s")
    print(f"Damping ratio: {parameters.damping_ratio:.3f}")
    print(f"Normalized damping d/m: {parameters.damping_per_mass:.6f} 1/s")
    print(f"Normalized stiffness k/m: {parameters.stiffness_per_mass:.6f} 1/s^2")
    print()
    print(f"Maximum horizontal displacement: {peak.maximum_displacement:.6f} m")
    print(f"Time of maximum: {peak.time:.3f} s")
    print(f"East displacement at maximum: {peak.east_displacement:.6f} m")
    print(f"North displacement at maximum: {peak.north_displacement:.6f} m")
    print(f"Simplified building gap (2*r_max): {peak.simplified_gap:.6f} m")

    plot_response(response, peak, save_path="earthquake_response.png", show=False)
    study = study_parameters(
        motion, parameters, method=METHOD, factors=PARAMETER_FACTORS
    )
    plot_parameter_study(
        study, save_path="parameter_study.png", show=False
    )
    print("Saved parameter study: parameter_study.png")

    if GENERATE_HEATMAP:
        heatmap_study = study_damping_stiffness(motion, parameters, method=METHOD)
        plot_damping_stiffness_heatmap(
            heatmap_study, save_path="parameter_heatmap_2d.png", show=False
        )
        print("Saved 2D parameter heat map: parameter_heatmap_2d.png")


if __name__ == "__main__":
    main()
