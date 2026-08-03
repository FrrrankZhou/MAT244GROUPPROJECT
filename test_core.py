"""Small tests covering the mathematically important parts of the project."""

import numpy as np

from data_reader import GroundMotion2D, load_horizontal_motion
from ode_solver import (
    BuildingParameters,
    find_peak,
    simulate_2d,
    study_damping_stiffness,
    study_parameters,
)


def test_parameter_conversion() -> None:
    params = BuildingParameters.from_period(10.0, 2.0, 0.05)
    assert np.isclose(params.stiffness, 10.0 * np.pi**2)
    assert np.isclose(params.damping, np.pi)
    assert np.isclose(params.damping_per_mass, np.pi / 10.0)
    assert np.isclose(params.stiffness_per_mass, np.pi**2)


def test_zero_input_gives_zero_response() -> None:
    zeros = np.zeros(11)
    motion = GroundMotion2D(np.arange(11) * 0.1, zeros, zeros, 0.1, "HNE", "HNN")
    response = simulate_2d(motion, BuildingParameters.from_period(1.0, 1.0, 0.05))
    assert np.allclose(response.total_displacement, 0.0)


def test_initial_conditions_are_applied_to_both_directions() -> None:
    zeros = np.zeros(3)
    motion = GroundMotion2D(np.arange(3) * 0.1, zeros, zeros, 0.1, "HNE", "HNN")
    response = simulate_2d(
        motion,
        BuildingParameters(1.0, 1.0, 0.0),
        initial_displacement_east=2.0,
        initial_velocity_east=3.0,
        initial_displacement_north=4.0,
        initial_velocity_north=5.0,
    )
    assert response.displacement_east[0] == 2.0
    assert response.velocity_east[0] == 3.0
    assert response.displacement_north[0] == 4.0
    assert response.velocity_north[0] == 5.0
    assert response.total_displacement[0] == np.hypot(2.0, 4.0)


def test_resultant_is_combined_at_each_time() -> None:
    from ode_solver import HorizontalResponse

    response = HorizontalResponse(
        time=np.array([0.0, 1.0]),
        displacement_east=np.array([3.0, 0.0]),
        displacement_north=np.array([4.0, 10.0]),
        total_displacement=np.array([5.0, 10.0]),
        velocity_east=np.zeros(2),
        velocity_north=np.zeros(2),
        total_velocity=np.zeros(2),
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


def test_parameter_study_varies_one_parameter_at_a_time() -> None:
    zeros = np.zeros(5)
    motion = GroundMotion2D(np.arange(5) * 0.1, zeros, zeros, 0.1, "HNE", "HNN")
    params = BuildingParameters(2.0, 8.0, 0.4)
    study = study_parameters(motion, params, factors=np.array([0.5, 1.0, 1.5]))
    assert np.allclose(study.mass_values, [1.0, 2.0, 3.0])
    assert np.allclose(study.damping_values, [0.2, 0.4, 0.6])
    assert np.allclose(study.stiffness_values, [4.0, 8.0, 12.0])
    assert np.allclose(study.mass_max_displacements, 0.0)


def test_2d_parameter_grid_contains_all_combinations() -> None:
    zeros = np.zeros(3)
    motion = GroundMotion2D(np.arange(3) * 0.1, zeros, zeros, 0.1, "HNE", "HNN")
    study = study_damping_stiffness(
        motion,
        BuildingParameters(2.0, 8.0, 0.4),
        factors=np.array([0.5, 1.5]),
    )
    assert study.maximum_displacements.shape == (2, 2)
    assert study.mass == 2.0
    assert np.allclose(study.maximum_displacements, 0.0)
