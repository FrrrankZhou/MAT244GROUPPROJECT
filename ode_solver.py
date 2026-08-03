"""ODE models and numerical methods for a two-dimensional SDOF building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from data_reader import GroundMotion2D


FloatArray = NDArray[np.float64]
Method = Literal["euler", "rk4"]


@dataclass(frozen=True)
class BuildingParameters:
    mass: float
    stiffness: float
    damping: float

    @classmethod
    def from_period(
        cls, mass: float, natural_period: float, damping_ratio: float
    ) -> "BuildingParameters":
        if natural_period <= 0 or damping_ratio < 0:
            raise ValueError("natural_period must be positive and damping_ratio nonnegative.")
        omega_n = 2.0 * np.pi / natural_period
        stiffness = mass * omega_n**2
        damping = 2.0 * damping_ratio * mass * omega_n
        return cls(mass, stiffness, damping).validated()

    def validated(self) -> "BuildingParameters":
        values = np.array([self.mass, self.stiffness, self.damping])
        if not np.isfinite(values).all():
            raise ValueError("Building parameters must be finite.")
        if self.mass <= 0 or self.stiffness <= 0 or self.damping < 0:
            raise ValueError("mass and stiffness must be positive; damping cannot be negative.")
        return self

    @property
    def natural_period(self) -> float:
        return 2.0 * np.pi * np.sqrt(self.mass / self.stiffness)

    @property
    def damping_ratio(self) -> float:
        return self.damping / (2.0 * np.sqrt(self.mass * self.stiffness))

    @property
    def damping_per_mass(self) -> float:
        """Normalized damping coefficient d/m, with units 1/s."""
        return self.damping / self.mass

    @property
    def stiffness_per_mass(self) -> float:
        """Normalized stiffness coefficient k/m, with units 1/s²."""
        return self.stiffness / self.mass


@dataclass(frozen=True)
class HorizontalResponse:
    time: FloatArray
    displacement_east: FloatArray
    displacement_north: FloatArray
    total_displacement: FloatArray
    velocity_east: FloatArray
    velocity_north: FloatArray
    total_velocity: FloatArray
    acceleration_east: FloatArray
    acceleration_north: FloatArray
    method: Method


@dataclass(frozen=True)
class PeakResponse:
    maximum_displacement: float
    time: float
    east_displacement: float
    north_displacement: float
    index: int
    simplified_gap: float


@dataclass(frozen=True)
class ParameterStudy:
    """Maximum displacement obtained by varying one parameter at a time."""

    mass_values: FloatArray
    mass_max_displacements: FloatArray
    damping_values: FloatArray
    damping_max_displacements: FloatArray
    stiffness_values: FloatArray
    stiffness_max_displacements: FloatArray


@dataclass(frozen=True)
class DampingStiffnessStudy:
    """Results for combinations of damping and stiffness at a fixed mass."""

    mass: float
    damping_values: FloatArray
    stiffness_values: FloatArray
    maximum_displacements: FloatArray


def _derivative(
    state: NDArray[np.float64],
    ground_acceleration: float,
    damping_per_mass: float,
    stiffness_per_mass: float,
) -> NDArray[np.float64]:
    """Return [x', v'] for m*x'' + c*x' + k*x = -m*a_g(t)."""
    displacement, velocity = state
    acceleration = (
        -damping_per_mass * velocity
        -stiffness_per_mass * displacement
        - ground_acceleration
    )
    return np.array([velocity, acceleration], dtype=np.float64)


def _solve_component(
    ground_acceleration: FloatArray,
    dt: float,
    params: BuildingParameters,
    method: Method,
) -> tuple[FloatArray, FloatArray]:
    state = np.zeros((len(ground_acceleration), 2), dtype=np.float64)
    damping_per_mass = params.damping_per_mass
    stiffness_per_mass = params.stiffness_per_mass

    def derivative(current: NDArray[np.float64], acceleration: float) -> FloatArray:
        return _derivative(
            current, acceleration, damping_per_mass, stiffness_per_mass
        )

    for i in range(len(ground_acceleration) - 1):
        current = state[i]
        a0 = ground_acceleration[i]
        if method == "euler":
            state[i + 1] = current + dt * derivative(current, a0)
        elif method == "rk4":
            a1 = ground_acceleration[i + 1]
            amid = 0.5 * (a0 + a1)  # linear interpolation of the earthquake input
            k1 = derivative(current, a0)
            k2 = derivative(current + 0.5 * dt * k1, amid)
            k3 = derivative(current + 0.5 * dt * k2, amid)
            k4 = derivative(current + dt * k3, a1)
            state[i + 1] = current + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        else:
            raise ValueError("method must be 'euler' or 'rk4'.")
    return state[:, 0], state[:, 1]


def simulate_2d(
    motion: GroundMotion2D,
    parameters: BuildingParameters,
    method: Method = "rk4",
) -> HorizontalResponse:
    """Solve HNE/HNN separately and combine them at every time sample."""
    parameters.validated()
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'.")
    east, velocity_east = _solve_component(
        motion.acceleration_east, motion.dt, parameters, method
    )
    north, velocity_north = _solve_component(
        motion.acceleration_north, motion.dt, parameters, method
    )
    total = np.hypot(east, north)
    total_velocity = np.hypot(velocity_east, velocity_north)
    return HorizontalResponse(
        motion.time.copy(), east, north, total,
        velocity_east, velocity_north, total_velocity,
        motion.acceleration_east.copy(), motion.acceleration_north.copy(), method,
    )


def find_peak(response: HorizontalResponse) -> PeakResponse:
    index = int(np.argmax(response.total_displacement))
    maximum = float(response.total_displacement[index])
    return PeakResponse(
        maximum_displacement=maximum,
        time=float(response.time[index]),
        east_displacement=float(response.displacement_east[index]),
        north_displacement=float(response.displacement_north[index]),
        index=index,
        simplified_gap=2.0 * maximum,
    )


def study_parameters(
    motion: GroundMotion2D,
    baseline: BuildingParameters,
    method: Method = "rk4",
    factors: FloatArray | None = None,
) -> ParameterStudy:
    """Vary m, d, and k separately and calculate r_max for each value.

    This is a one-at-a-time sensitivity study: while one parameter changes,
    the other two remain at their baseline values.
    """
    baseline.validated()
    if factors is None:
        factors = np.linspace(0.5, 1.5, 9, dtype=np.float64)
    else:
        factors = np.asarray(factors, dtype=np.float64)
    if factors.ndim != 1 or len(factors) == 0:
        raise ValueError("factors must be a nonempty one-dimensional array.")
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("All parameter factors must be finite and positive.")

    def maximum(params: BuildingParameters) -> float:
        return find_peak(simulate_2d(motion, params, method)).maximum_displacement

    mass_values = baseline.mass * factors
    damping_values = baseline.damping * factors
    stiffness_values = baseline.stiffness * factors
    mass_max = np.array([
        maximum(BuildingParameters(value, baseline.stiffness, baseline.damping))
        for value in mass_values
    ])
    damping_max = np.array([
        maximum(BuildingParameters(baseline.mass, baseline.stiffness, value))
        for value in damping_values
    ])
    stiffness_max = np.array([
        maximum(BuildingParameters(baseline.mass, value, baseline.damping))
        for value in stiffness_values
    ])
    return ParameterStudy(
        mass_values, mass_max,
        damping_values, damping_max,
        stiffness_values, stiffness_max,
    )


def study_damping_stiffness(
    motion: GroundMotion2D,
    baseline: BuildingParameters,
    method: Method = "rk4",
    factors: FloatArray | None = None,
) -> DampingStiffnessStudy:
    """Evaluate r_max over a d-k grid while holding mass constant."""
    baseline.validated()
    if factors is None:
        factors = np.linspace(0.5, 1.5, 7, dtype=np.float64)
    else:
        factors = np.asarray(factors, dtype=np.float64)
    if factors.ndim != 1 or len(factors) == 0:
        raise ValueError("factors must be a nonempty one-dimensional array.")
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("All parameter factors must be finite and positive.")

    dampings, stiffnesses = np.meshgrid(
        baseline.damping * factors,
        baseline.stiffness * factors,
        indexing="xy",
    )
    maxima = np.empty(dampings.size, dtype=np.float64)
    for index, (damping, stiffness) in enumerate(zip(
        dampings.ravel(), stiffnesses.ravel()
    )):
        parameters = BuildingParameters(baseline.mass, stiffness, damping)
        maxima[index] = find_peak(
            simulate_2d(motion, parameters, method)
        ).maximum_displacement

    return DampingStiffnessStudy(
        baseline.mass,
        dampings[0, :].copy(),
        stiffnesses[:, 0].copy(),
        maxima.reshape(stiffnesses.shape),
    )
