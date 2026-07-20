from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Strategy = Literal["euler", "rk4"]


@dataclass(frozen=True)
class GroundMotionData:
  """A uniformly sampled ground-acceleration record in SI units."""

  time: FloatArray
  acceleration: FloatArray  # m/s^2
  dt: float  # seconds
  source_name: str
  component: str
  original_unit: str
  metadata: dict[str, str | float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
  """Numerical solution of the building's relative motion."""

  time: FloatArray
  displacement: FloatArray  # m
  velocity: FloatArray  # m/s
  ground_acceleration: FloatArray  # m/s^2
  mass: float
  damping: float
  stiffness: float
  strategy: Strategy


@dataclass(frozen=True)
class MaximumDisplacementResult:
  """Information about max_t |x(t)|."""

  absolute_maximum: float
  signed_displacement: float
  time: float
  index: int
  recommended_gap: float


# regex expression for data fetching
_POINT_HEADER_RE = re.compile(
  r"^\s*(\d+)\s+acceleration\s+pts.*?units=([^,\s]+).*?$",
  re.IGNORECASE,
)
_DT_TEXT_RE = re.compile(r"(\d+(?:\.\d+)?)\s+samples/sec", re.IGNORECASE)
_COMPONENT_RE = re.compile(r"\.(HN[ENZ])\.", re.IGNORECASE)


def _read_v2c_text(file_name: str | Path, component: str) -> tuple[str, str]:
  """
    Return (text, logical_source_name).

    Supports:
      1. a plain .V2c file;
      2. a ZIP containing .V2c files;
      3. the supplied CESMD outer ZIP, which contains another ZIP.
    """
  path = Path(file_name)
  if not path.exists():
    raise FileNotFoundError(f"File not found: {path}")

  component = component.upper()
  if component not in {"HNE", "HNN", "HNZ"}:
    raise ValueError("component must be 'HNE', 'HNN', or 'HNZ'")

  if not zipfile.is_zipfile(path):
    return path.read_text(encoding="utf-8", errors="replace"), path.name

  def search_zip(zipfilename: zipfile.ZipFile, parent_name: str) -> tuple[str, str] | None:
    names = zipfilename.namelist()

    # Prefer a corrected acceleration V2c file with the requested component.
    candidates = [
      name for name in names
      if (name.lower().endswith(".acc.v2c") and
          (f".{component.lower()}." in name.lower())
          )
    ]
    if candidates:
      selected = sorted(candidates)[0]
      raw = zipfilename.read(selected)
      return raw.decode("utf-8", errors="replace"), f"{parent_name}!{selected}"

    # Recursively inspect nested ZIP files.
    for name in names:
      if name.lower().endswith(".zip"):
        nested_bytes = zipfilename.read(name)
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
          founded = search_zip(nested, f"{parent_name}!{name}")
          if founded is not None:
            return founded
    return None

  with zipfile.ZipFile(path) as zf:
    found = search_zip(zf, path.name)

  if found is None:
    raise ValueError(
      f"No corrected acceleration .acc.V2c record for {component} "
      f"was found inside {path.name}."
    )
  return found


def fetch_data(
    file_name: str | Path,
    component: str = "HNE",
) -> GroundMotionData:
  """
    Read a CESMD corrected-acceleration V2c record.

    Returned acceleration is always converted from cm/s^2 to m/s^2.
    """
  text, source_name = _read_v2c_text(file_name, component)
  lines = text.splitlines()

  data_start: int | None = None
  expected_points: int | None = None
  original_unit: str | None = None

  for i, line in enumerate(lines):
    match = _POINT_HEADER_RE.match(line)
    if match:
      expected_points = int(match.group(1))
      original_unit = match.group(2)
      data_start = i + 1
      break

  if data_start is None or expected_points is None or original_unit is None:
    raise ValueError(
      "Could not find the CESMD acceleration-data header "
      "('<N> acceleration pts ... units=...')."
    )

  values: list[float] = []
  for line in lines[data_start:]:
    stripped = line.strip()
    if not stripped:
      continue
    try:
      values.append(float(stripped))
    except ValueError:
      # Stop if a nonnumeric footer ever appears.
      break

  if len(values) != expected_points:
    raise ValueError(
      f"Header says {expected_points} acceleration points, "
      f"but {len(values)} numeric values were read."
    )

  # Prefer the explicit sampling-rate comment.
  sample_rate: float | None = None
  for line in lines[:data_start]:
    if "DECIMATE" in line.upper() or "RESAMPLE" in line.upper():
      match = _DT_TEXT_RE.search(line)
      if match:
        candidate = float(match.group(1))
        # The final DECIMATE line appears after RESAMPLE in this dataset.
        sample_rate = candidate

  # Fallback: use duration from the point-header line.
  if sample_rate is None:
    duration_match = re.search(
      r"approx\s+([0-9.]+)\s+secs", lines[data_start - 1], re.IGNORECASE
    )
    if duration_match:
      duration = float(duration_match.group(1))
      sample_rate = expected_points / duration

  if sample_rate is None or sample_rate <= 0:
    raise ValueError("Could not determine a positive sampling rate.")

  dt = 1.0 / sample_rate
  acceleration_cm_s2 = np.asarray(values, dtype=np.float64)
  acceleration_m_s2 = acceleration_cm_s2 / 100.0
  time = np.arange(expected_points, dtype=np.float64) * dt

  component_match = _COMPONENT_RE.search(source_name)
  detected_component = (
    component_match.group(1).upper() if component_match else component.upper()
  )

  data = GroundMotionData(
    time=time,
    acceleration=acceleration_m_s2,
    dt=dt,
    source_name=source_name,
    component=detected_component,
    original_unit=original_unit,
    metadata={
      "number_of_points": expected_points,
      "sample_rate_hz": sample_rate,
      "duration_seconds": expected_points * dt,
    },
  )
  validate_ground_motion(data)
  return data


def validate_ground_motion(data: GroundMotionData) -> None:
  if data.time.ndim != 1 or data.acceleration.ndim != 1:
    raise ValueError("time and acceleration must be one-dimensional arrays.")
  if len(data.time) == 0:
    raise ValueError("Ground-motion record is empty.")
  if len(data.time) != len(data.acceleration):
    raise ValueError("time and acceleration must have the same length.")
  if not np.isfinite(data.time).all() or not np.isfinite(data.acceleration).all():
    raise ValueError("Ground-motion record contains NaN or infinite values.")
  if data.dt <= 0:
    raise ValueError("dt must be positive.")
  if len(data.time) > 1 and not np.allclose(
      np.diff(data.time), data.dt, rtol=1e-9, atol=1e-12
  ):
    raise ValueError("time must be uniformly sampled.")


def validate_model_parameters(
    mass: float,
    damping: float,
    stiffness: float,
) -> None:
  for name, value in {
    "mass": mass,
    "damping": damping,
    "stiffness": stiffness,
  }.items():
    if not np.isfinite(value):
      raise ValueError(f"{name} must be finite.")

  if mass <= 0:
    raise ValueError("mass must be positive.")
  if damping < 0:
    raise ValueError("damping must be nonnegative.")
  if stiffness <= 0:
    raise ValueError("stiffness must be positive.")


def evaluate_derivative(
    displacement: float,
    velocity: float,
    ground_acceleration: float,
    mass: float,
    damping: float,
    stiffness: float,
) -> tuple[float, float]:
  """
    First-order form of:
        m*x'' + d*x' + k*x = -m*a_g(t)

    Returns:
        (x', v')
    """
  dx_dt = velocity
  dv_dt = (
      -(damping / mass) * velocity
      - (stiffness / mass) * displacement
      - ground_acceleration
  )
  return dx_dt, dv_dt


def simulate_euler(
    mass: float,
    damping: float,
    stiffness: float,
    ground_motion: GroundMotionData,
    initial_displacement: float = 0.0,
    initial_velocity: float = 0.0,
) -> SimulationResult:
  n = len(ground_motion.time)
  x = np.zeros(n, dtype=np.float64)
  v = np.zeros(n, dtype=np.float64)
  x[0] = initial_displacement
  v[0] = initial_velocity
  dt = ground_motion.dt
  a_g = ground_motion.acceleration

  for i in range(n - 1):
    dx_dt, dv_dt = evaluate_derivative(
      x[i], v[i], a_g[i], mass, damping, stiffness
    )
    x[i + 1] = x[i] + dt * dx_dt
    v[i + 1] = v[i] + dt * dv_dt

  return SimulationResult(
    time=ground_motion.time.copy(),
    displacement=x,
    velocity=v,
    ground_acceleration=a_g.copy(),
    mass=mass,
    damping=damping,
    stiffness=stiffness,
    strategy="euler",
  )


def simulate_rk4(
    mass: float,
    damping: float,
    stiffness: float,
    ground_motion: GroundMotionData,
    initial_displacement: float = 0.0,
    initial_velocity: float = 0.0,
) -> SimulationResult:
  n = len(ground_motion.time)
  x = np.zeros(n, dtype=np.float64)
  v = np.zeros(n, dtype=np.float64)
  x[0] = initial_displacement
  v[0] = initial_velocity
  dt = ground_motion.dt
  a_g = ground_motion.acceleration

  for i in range(n - 1):
    a_start = a_g[i]
    a_end = a_g[i + 1]
    a_mid = 0.5 * (a_start + a_end)

    k1_x, k1_v = evaluate_derivative(
      x[i], v[i], a_start, mass, damping, stiffness
    )
    k2_x, k2_v = evaluate_derivative(
      x[i] + 0.5 * dt * k1_x,
      v[i] + 0.5 * dt * k1_v,
      a_mid,
      mass,
      damping,
      stiffness,
    )
    k3_x, k3_v = evaluate_derivative(
      x[i] + 0.5 * dt * k2_x,
      v[i] + 0.5 * dt * k2_v,
      a_mid,
      mass,
      damping,
      stiffness,
    )
    k4_x, k4_v = evaluate_derivative(
      x[i] + dt * k3_x,
      v[i] + dt * k3_v,
      a_end,
      mass,
      damping,
      stiffness,
    )

    x[i + 1] = x[i] + (dt / 6.0) * (
        k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x
    )
    v[i + 1] = v[i] + (dt / 6.0) * (
        k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
    )

  return SimulationResult(
    time=ground_motion.time.copy(),
    displacement=x,
    velocity=v,
    ground_acceleration=a_g.copy(),
    mass=mass,
    damping=damping,
    stiffness=stiffness,
    strategy="rk4",
  )


def apply_simulation(
    mass: float,
    damping: float,
    stiffness: float,
    ground_motion: GroundMotionData,
    strategy: Strategy = "rk4",
    initial_displacement: float = 0.0,
    initial_velocity: float = 0.0,
) -> SimulationResult:
  """Validate inputs and dispatch to the selected numerical method."""
  validate_ground_motion(ground_motion)
  validate_model_parameters(mass, damping, stiffness)

  if not np.isfinite(initial_displacement) or not np.isfinite(initial_velocity):
    raise ValueError("Initial displacement and velocity must be finite.")

  normalized_strategy = strategy.lower()
  if normalized_strategy == "euler":
    return simulate_euler(
      mass,
      damping,
      stiffness,
      ground_motion,
      initial_displacement,
      initial_velocity,
    )
  if normalized_strategy == "rk4":
    return simulate_rk4(
      mass,
      damping,
      stiffness,
      ground_motion,
      initial_displacement,
      initial_velocity,
    )

  raise ValueError("strategy must be either 'euler' or 'rk4'.")


def find_maximum_displacement(
    result: SimulationResult,
    safety_factor: float = 1.0,
) -> MaximumDisplacementResult:
  """
    Find max_t |x(t)| and compute the simplified two-building gap:
        gap = 2 * safety_factor * max_t |x(t)|
    """
  if not np.isfinite(safety_factor) or safety_factor < 1.0:
    raise ValueError("safety_factor must be finite and at least 1.0.")

  index = int(np.argmax(np.abs(result.displacement)))
  signed = float(result.displacement[index])
  absolute = abs(signed)

  return MaximumDisplacementResult(
    absolute_maximum=absolute,
    signed_displacement=signed,
    time=float(result.time[index]),
    index=index,
    recommended_gap=2.0 * safety_factor * absolute,
  )


def plot_ground_acceleration(
    data: GroundMotionData,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
  fig, ax = plt.subplots(figsize=(10, 4.5))
  ax.plot(data.time, data.acceleration, linewidth=0.8)
  ax.set_title(f"Ground acceleration ({data.component})")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Ground acceleration (m/s²)")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  if save_path is not None:
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
  if show:
    plt.show()
  else:
    plt.close(fig)


def plot_displacement_time_series(
    result: SimulationResult,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
  maximum = find_maximum_displacement(result)

  fig, ax = plt.subplots(figsize=(10, 4.5))
  ax.plot(result.time, result.displacement, linewidth=0.9)
  ax.scatter(
    [maximum.time],
    [maximum.signed_displacement],
    zorder=3,
    label=(
      f"max |x| = {maximum.absolute_maximum:.6g} m "
      f"at t = {maximum.time:.3f} s"
    ),
  )
  ax.axhline(0.0, linewidth=0.8)
  ax.set_title(f"Building displacement ({result.strategy.upper()})")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Relative displacement x(t) (m)")
  ax.grid(True, alpha=0.3)
  ax.legend()
  fig.tight_layout()

  if save_path is not None:
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
  if show:
    plt.show()
  else:
    plt.close(fig)


def main(filename: str) -> None:
  """
    Example first-round workflow.

    Replace FILE_NAME and model parameters with your chosen values.
    The example parameters correspond to:
      natural period T_n = 1.0 s
      damping ratio zeta = 0.05
      mass = 1.0e6 kg
    """

  mass = 1.0e6
  natural_period = 1.0
  damping_ratio = 0.05

  stiffness = mass * (2.0 * np.pi / natural_period) ** 2
  damping = 2.0 * damping_ratio * np.sqrt(mass * stiffness)

  data = fetch_data(filename, component="HNE")
  result = apply_simulation(
    mass=mass,
    damping=damping,
    stiffness=stiffness,
    ground_motion=data,
    strategy="rk4",
  )
  maximum = find_maximum_displacement(result)

  print(f"Source: {data.source_name}")
  print(f"Samples: {len(data.time)}")
  print(f"dt: {data.dt:.6f} s")
  print(f"Maximum displacement: {maximum.absolute_maximum:.6f} m")
  print(f"Signed displacement: {maximum.signed_displacement:.6f} m")
  print(f"Time of maximum: {maximum.time:.3f} s")
  print(f"Recommended gap (2*x_max): {maximum.recommended_gap:.6f} m")
  print(np.max(np.abs(data.acceleration)))
  plot_ground_acceleration(data)
  plot_displacement_time_series(result)


if __name__ == "__main__":
  main('Honaunau-Napoopoo.zip')
