"""Unit tests for CESMD parsing, ODE solving, and response analysis."""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from data_reader import GroundMotion2D, _parse_component, load_horizontal_motion
from ode_solver import (
    BuildingParameters,
    HorizontalResponse,
    find_peak,
    simulate_2d,
    study_damping_stiffness,
    study_parameters,
)


def make_motion(
    east: np.ndarray | None = None,
    north: np.ndarray | None = None,
    dt: float = 0.1,
) -> GroundMotion2D:
    if east is None:
        east = np.zeros(5)
    if north is None:
        north = np.zeros_like(east)
    time = np.arange(len(east), dtype=float) * dt
    return GroundMotion2D(time, east, north, dt, "test.HNE.acc.V2c", "test.HNN.acc.V2c")


def make_v2c(component: str, values: list[float], rate: float = 100.0) -> bytes:
    data = " ".join(str(value) for value in values)
    text = (
        f"|<DECIMATE> Data decimated to {rate:.2f} samples/sec\n"
        f"{len(values)} acceleration pts, approx {len(values) / rate:g} secs, "
        "units=cm/sec2(04),Format=(1E15.6)\n"
        f"{data}\n"
    )
    return text.encode("utf-8")


class BuildingParameterTests(unittest.TestCase):
    def test_period_and_damping_ratio_conversion(self) -> None:
        params = BuildingParameters.from_period(10.0, 2.0, 0.05)
        self.assertAlmostEqual(params.stiffness, 10.0 * np.pi**2)
        self.assertAlmostEqual(params.damping, np.pi)
        self.assertAlmostEqual(params.natural_period, 2.0)
        self.assertAlmostEqual(params.damping_ratio, 0.05)
        self.assertAlmostEqual(params.damping_per_mass, np.pi / 10.0)
        self.assertAlmostEqual(params.stiffness_per_mass, np.pi**2)

    def test_invalid_direct_parameters_are_rejected(self) -> None:
        invalid = (
            BuildingParameters(0.0, 1.0, 0.1),
            BuildingParameters(1.0, 0.0, 0.1),
            BuildingParameters(1.0, 1.0, -0.1),
            BuildingParameters(np.nan, 1.0, 0.1),
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    parameters.validated()

    def test_invalid_period_parameters_are_rejected(self) -> None:
        for period, ratio in ((0.0, 0.05), (-1.0, 0.05), (1.0, -0.01)):
            with self.subTest(period=period, ratio=ratio):
                with self.assertRaises(ValueError):
                    BuildingParameters.from_period(1.0, period, ratio)


class CesmdReaderTests(unittest.TestCase):
    def test_component_parser_converts_cm_per_second_squared(self) -> None:
        acceleration, dt = _parse_component(
            "station.HNE.acc.V2c", make_v2c("HNE", [100.0, -50.0], 200.0)
        )
        np.testing.assert_allclose(acceleration, [1.0, -0.5])
        self.assertAlmostEqual(dt, 0.005)

    def test_component_parser_uses_duration_fallback(self) -> None:
        raw = (
            "4 acceleration pts, approx 2 secs, "
            "units=cm/sec2(04),Format=(1E15.6)\n0 0 0 0\n"
        ).encode()
        acceleration, dt = _parse_component("station.HNE.acc.V2c", raw)
        np.testing.assert_array_equal(acceleration, np.zeros(4))
        self.assertAlmostEqual(dt, 0.5)

    def test_component_parser_rejects_wrong_point_count(self) -> None:
        raw = (
            "3 acceleration pts, approx 1 secs, "
            "units=cm/sec2(04),Format=(1E15.6)\n1 2\n"
        ).encode()
        with self.assertRaisesRegex(ValueError, "declares 3 points"):
            _parse_component("station.HNE.acc.V2c", raw)

    def test_plain_sibling_files_are_loaded_as_a_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            east = folder / "station.event.HNE.--.acc.V2c"
            north = folder / "station.event.HNN.--.acc.V2c"
            east.write_bytes(make_v2c("HNE", [100.0, 200.0]))
            north.write_bytes(make_v2c("HNN", [300.0, 400.0]))
            motion = load_horizontal_motion(east)
        np.testing.assert_allclose(motion.acceleration_east, [1.0, 2.0])
        np.testing.assert_allclose(motion.acceleration_north, [3.0, 4.0])

    def test_nested_zip_is_loaded(self) -> None:
        nested_bytes = io.BytesIO()
        with zipfile.ZipFile(nested_bytes, "w") as nested:
            nested.writestr("V2C/station.event.HNE.--.acc.V2c", make_v2c("HNE", [1, 2]))
            nested.writestr("V2C/station.event.HNN.--.acc.V2c", make_v2c("HNN", [3, 4]))
        with tempfile.TemporaryDirectory() as directory:
            outer_path = Path(directory) / "outer.zip"
            with zipfile.ZipFile(outer_path, "w") as outer:
                outer.writestr("nested.zip", nested_bytes.getvalue())
            motion = load_horizontal_motion(outer_path)
        self.assertEqual(len(motion.time), 2)
        self.assertAlmostEqual(motion.dt, 0.01)

    def test_missing_horizontal_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "only-east.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("station.event.HNE.--.acc.V2c", make_v2c("HNE", [1, 2]))
            with self.assertRaisesRegex(ValueError, "No matching HNE and HNN"):
                load_horizontal_motion(archive)

    def test_real_calama_nested_zip_metadata(self) -> None:
        motion = load_horizontal_motion("Calama.zip")
        self.assertEqual(len(motion.time), 20100)
        self.assertAlmostEqual(motion.dt, 0.01)
        self.assertTrue(np.isfinite(motion.acceleration_east).all())
        self.assertTrue(np.isfinite(motion.acceleration_north).all())


class OdeSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = BuildingParameters.from_period(1.0, 1.0, 0.05)

    def test_zero_input_gives_zero_displacement_and_velocity(self) -> None:
        response = simulate_2d(make_motion(), self.parameters, method="rk4")
        np.testing.assert_allclose(response.total_displacement, 0.0)
        np.testing.assert_allclose(response.total_velocity, 0.0)

    def test_both_methods_return_finite_results(self) -> None:
        east = np.array([0.0, 1.0, -1.0, 0.5, 0.0])
        north = np.array([0.0, -0.5, 0.25, 0.0, 0.0])
        for method in ("euler", "rk4"):
            with self.subTest(method=method):
                response = simulate_2d(make_motion(east, north, 0.01), self.parameters, method)
                self.assertTrue(np.isfinite(response.total_displacement).all())
                self.assertTrue(np.isfinite(response.total_velocity).all())
                np.testing.assert_allclose(
                    response.total_displacement,
                    np.hypot(response.displacement_east, response.displacement_north),
                )

    def test_euler_first_step_matches_update_formula(self) -> None:
        east = np.array([2.0, 2.0])
        response = simulate_2d(make_motion(east, np.zeros(2), 0.1), self.parameters, "euler")
        self.assertAlmostEqual(response.displacement_east[1], 0.0)
        self.assertAlmostEqual(response.velocity_east[1], -0.2)

    def test_rk4_constant_forcing_matches_exact_short_step(self) -> None:
        # x'' + x = -1, x(0)=x'(0)=0 gives x(t)=cos(t)-1.
        parameters = BuildingParameters(1.0, 1.0, 0.0)
        response = simulate_2d(
            make_motion(np.ones(2), np.zeros(2), 0.1), parameters, "rk4"
        )
        self.assertAlmostEqual(response.displacement_east[1], np.cos(0.1) - 1.0, places=7)

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "method must be"):
            simulate_2d(make_motion(), self.parameters, method="bad")  # type: ignore[arg-type]

    def test_invalid_building_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate_2d(make_motion(), BuildingParameters(-1.0, 1.0, 0.1))


class AnalysisAndStudyTests(unittest.TestCase):
    def test_peak_uses_east_and_north_from_the_same_time(self) -> None:
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
        self.assertEqual(peak.index, 1)
        self.assertEqual(peak.time, 1.0)
        self.assertEqual(peak.maximum_displacement, 10.0)
        self.assertEqual(peak.east_displacement, 0.0)
        self.assertEqual(peak.north_displacement, 10.0)
        self.assertEqual(peak.simplified_gap, 20.0)

    def test_one_at_a_time_parameter_study_values(self) -> None:
        parameters = BuildingParameters(2.0, 8.0, 0.4)
        study = study_parameters(
            make_motion(), parameters, factors=np.array([0.5, 1.0, 1.5])
        )
        np.testing.assert_allclose(study.mass_values, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(study.damping_values, [0.2, 0.4, 0.6])
        np.testing.assert_allclose(study.stiffness_values, [4.0, 8.0, 12.0])
        np.testing.assert_allclose(study.mass_max_displacements, 0.0)

    def test_parameter_study_rejects_invalid_factors(self) -> None:
        parameters = BuildingParameters(2.0, 8.0, 0.4)
        invalid_factors = (np.array([]), np.array([0.0, 1.0]), np.array([np.nan]))
        for factors in invalid_factors:
            with self.subTest(factors=factors):
                with self.assertRaises(ValueError):
                    study_parameters(make_motion(), parameters, factors=factors)

    def test_damping_stiffness_grid_shape(self) -> None:
        study = study_damping_stiffness(
            make_motion(),
            BuildingParameters(2.0, 8.0, 0.4),
            factors=np.array([0.5, 1.5]),
        )
        self.assertEqual(study.maximum_displacements.shape, (2, 2))
        self.assertEqual(study.mass, 2.0)
        np.testing.assert_allclose(study.maximum_displacements, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
