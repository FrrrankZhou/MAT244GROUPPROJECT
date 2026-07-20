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


@dataclass(frozen=True)
class HorizontalResponse:
    time: FloatArray
    displacement_east: FloatArray
    displacement_north: FloatArray
    total_displacement: FloatArray
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


def _derivative(
    state: NDArray[np.float64], ground_acceleration: float, params: BuildingParameters
) -> NDArray[np.float64]:
    """Return [x', v'] for m*x'' + c*x' + k*x = -m*a_g(t)."""
    displacement, velocity = state
    acceleration = (
        -(params.damping / params.mass) * velocity
        - (params.stiffness / params.mass) * displacement
        - ground_acceleration
    )
    return np.array([velocity, acceleration], dtype=np.float64)


def _solve_component(
    ground_acceleration: FloatArray,
    dt: float,
    params: BuildingParameters,
    method: Method,
) -> FloatArray:
    state = np.zeros((len(ground_acceleration), 2), dtype=np.float64)
    for i in range(len(ground_acceleration) - 1):
        current = state[i]
        a0 = ground_acceleration[i]
        if method == "euler":
            state[i + 1] = current + dt * _derivative(current, a0, params)
        elif method == "rk4":
            a1 = ground_acceleration[i + 1]
            amid = 0.5 * (a0 + a1)  # linear interpolation of the earthquake input
            k1 = _derivative(current, a0, params)
            k2 = _derivative(current + 0.5 * dt * k1, amid, params)
            k3 = _derivative(current + 0.5 * dt * k2, amid, params)
            k4 = _derivative(current + dt * k3, a1, params)
            state[i + 1] = current + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        else:
            raise ValueError("method must be 'euler' or 'rk4'.")
    return state[:, 0]


def simulate_2d(
    motion: GroundMotion2D,
    parameters: BuildingParameters,
    method: Method = "rk4",
) -> HorizontalResponse:
    """Solve HNE/HNN separately and combine them at every time sample."""
    parameters.validated()
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'.")
    east = _solve_component(motion.acceleration_east, motion.dt, parameters, method)
    north = _solve_component(motion.acceleration_north, motion.dt, parameters, method)
    total = np.hypot(east, north)
    return HorizontalResponse(
        motion.time.copy(), east, north, total,
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
