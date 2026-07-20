"""Run the three-week ODE course project example."""

from data_reader import load_horizontal_motion
from ode_solver import BuildingParameters, find_peak, simulate_2d
from plotting import plot_response


# Change these values to perform different experiments.
DATA_FILE = "Calama.zip"
MASS_KG = 1_000_000.0
NATURAL_PERIOD_S = 1.0
DAMPING_RATIO = 0.05
METHOD = "rk4"  # "euler" or "rk4"


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
    print()
    print(f"Maximum horizontal displacement: {peak.maximum_displacement:.6f} m")
    print(f"Time of maximum: {peak.time:.3f} s")
    print(f"East displacement at maximum: {peak.east_displacement:.6f} m")
    print(f"North displacement at maximum: {peak.north_displacement:.6f} m")
    print(f"Simplified building gap (2*r_max): {peak.simplified_gap:.6f} m")

    plot_response(response, peak, save_path="earthquake_response.png", show=False)


if __name__ == "__main__":
    main()
