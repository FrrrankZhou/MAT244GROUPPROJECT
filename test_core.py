"""Small tests covering the mathematically important parts of the project."""

import numpy as np

from data_reader import GroundMotion2D, load_horizontal_motion
from ode_solver import BuildingParameters, find_peak, simulate_2d


def test_parameter_conversion() -> None:
    params = BuildingParameters.from_period(10.0, 2.0, 0.05)
    assert np.isclose(params.stiffness, 10.0 * np.pi**2)
    assert np.isclose(params.damping, np.pi)


def test_zero_input_gives_zero_response() -> None:
    zeros = np.zeros(11)
    motion = GroundMotion2D(np.arange(11) * 0.1, zeros, zeros, 0.1, "HNE", "HNN")
    response = simulate_2d(motion, BuildingParameters.from_period(1.0, 1.0, 0.05))
    assert np.allclose(response.total_displacement, 0.0)


def test_resultant_is_combined_at_each_time() -> None:
    from ode_solver import HorizontalResponse

    response = HorizontalResponse(
        time=np.array([0.0, 1.0]),
        displacement_east=np.array([3.0, 0.0]),
        displacement_north=np.array([4.0, 10.0]),
        total_displacement=np.array([5.0, 10.0]),
        acceleration_east=np.zeros(2),
        acceleration_north=np.zeros(2),
        method="rk4",
    )
    peak = find_peak(response)
    assert peak.maximum_displacement == 10.0
    assert peak.east_displacement == 0.0
    assert peak.north_displacement == 10.0
    assert peak.simplified_gap == 20.0


def test_real_nested_zip() -> None:
    motion = load_horizontal_motion("Calama.zip")
    assert len(motion.time) == 20100
    assert np.isclose(motion.dt, 0.01)
