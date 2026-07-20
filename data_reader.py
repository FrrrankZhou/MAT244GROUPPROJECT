"""Read the two horizontal components of a CESMD strong-motion record."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GroundMotion2D:
    """Uniformly sampled east/north ground accelerations in SI units."""

    time: FloatArray
    acceleration_east: FloatArray
    acceleration_north: FloatArray
    dt: float
    source_east: str
    source_north: str


_COMPONENT_RE = re.compile(r"\.(HNE|HNN)\.", re.IGNORECASE)
_DATA_HEADER_RE = re.compile(
    r"^\s*(\d+)\s+acceleration\s+pts.*?units=([^,\s]+)", re.IGNORECASE
)
_RATE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+samples/sec", re.IGNORECASE)
_DURATION_RE = re.compile(r"approx\s+([0-9]+(?:\.[0-9]+)?)\s+secs", re.IGNORECASE)


def _collect_v2c_files(path: Path) -> list[tuple[str, bytes]]:
    """Recursively collect .acc.V2c members from a file or nested ZIP."""
    if not path.exists():
        raise FileNotFoundError(path)

    if not zipfile.is_zipfile(path):
        if not path.name.lower().endswith(".acc.v2c"):
            raise ValueError("Input must be a CESMD .acc.V2c file or ZIP archive.")
        return [(path.name, path.read_bytes())]

    records: list[tuple[str, bytes]] = []

    def visit(zf: zipfile.ZipFile, prefix: str) -> None:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            logical_name = f"{prefix}!{name}"
            if name.lower().endswith(".acc.v2c"):
                records.append((logical_name, zf.read(name)))
            elif name.lower().endswith(".zip"):
                nested_data = zf.read(name)
                try:
                    with zipfile.ZipFile(io.BytesIO(nested_data)) as nested:
                        visit(nested, logical_name)
                except zipfile.BadZipFile as exc:
                    raise ValueError(f"Invalid nested ZIP: {logical_name}") from exc

    with zipfile.ZipFile(path) as outer:
        visit(outer, path.name)
    return records


def _record_key(name: str) -> str:
    """Return a name that is identical for a matching HNE/HNN pair."""
    return _COMPONENT_RE.sub(".HN?.", name.lower())


def _parse_component(name: str, raw: bytes) -> tuple[FloatArray, float]:
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    header_index = None
    point_count = None
    unit = None
    for index, line in enumerate(lines):
        match = _DATA_HEADER_RE.match(line)
        if match:
            header_index = index
            point_count = int(match.group(1))
            unit = match.group(2).lower()
            break

    if header_index is None or point_count is None or unit is None:
        raise ValueError(f"Cannot find acceleration header in {name}.")

    values: list[float] = []
    for line in lines[header_index + 1 :]:
        # CESMD files may contain one or several fixed-width values per line.
        try:
            values.extend(float(token) for token in line.split())
        except ValueError:
            break
        if len(values) >= point_count:
            break

    if len(values) != point_count:
        raise ValueError(
            f"{name}: header declares {point_count} points, read {len(values)}."
        )

    rates = [float(match.group(1)) for line in lines[: header_index + 1]
             if (match := _RATE_RE.search(line))]
    if rates:
        sample_rate = rates[-1]
    else:
        duration_match = _DURATION_RE.search(lines[header_index])
        if duration_match is None:
            raise ValueError(f"Cannot determine the sampling interval in {name}.")
        duration = float(duration_match.group(1))
        sample_rate = point_count / duration

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in {name}.")

    acceleration = np.asarray(values, dtype=np.float64)
    normalized_unit = unit.replace("²", "2").replace("^", "")
    if normalized_unit.startswith("cm/"):
        acceleration = acceleration / 100.0
    elif not normalized_unit.startswith("m/"):
        raise ValueError(f"Unsupported acceleration unit {unit!r} in {name}.")
    return acceleration, 1.0 / sample_rate


def load_horizontal_motion(file_name: str | Path) -> GroundMotion2D:
    """Load a matching HNE/HNN pair from a CESMD file or (nested) ZIP."""
    path = Path(file_name)
    records = _collect_v2c_files(path)
    # A plain V2c contains one component. Automatically look for its sibling
    # file, e.g. station.HNE.--.acc.V2c -> station.HNN.--.acc.V2c.
    if not zipfile.is_zipfile(path) and _COMPONENT_RE.search(path.name):
        component = _COMPONENT_RE.search(path.name).group(1).upper()
        other = "HNN" if component == "HNE" else "HNE"
        sibling_name = _COMPONENT_RE.sub(f".{other}.", path.name)
        sibling = path.with_name(sibling_name)
        if sibling.exists():
            records.extend(_collect_v2c_files(sibling))
    pairs: dict[str, dict[str, tuple[str, bytes]]] = {}
    for name, raw in records:
        match = _COMPONENT_RE.search(name)
        if match:
            pairs.setdefault(_record_key(name), {})[match.group(1).upper()] = (name, raw)

    complete_pairs = [pair for pair in pairs.values() if {"HNE", "HNN"} <= pair.keys()]
    if not complete_pairs:
        raise ValueError("No matching HNE and HNN acceleration records were found.")
    if len(complete_pairs) > 1:
        raise ValueError("More than one HNE/HNN pair was found; choose a single station record.")

    pair = complete_pairs[0]
    east_name, east_raw = pair["HNE"]
    north_name, north_raw = pair["HNN"]
    east, dt_east = _parse_component(east_name, east_raw)
    north, dt_north = _parse_component(north_name, north_raw)

    if len(east) != len(north):
        raise ValueError("HNE and HNN have different numbers of samples.")
    if not np.isclose(dt_east, dt_north, rtol=1e-9, atol=1e-12):
        raise ValueError("HNE and HNN have different sampling intervals.")

    time = np.arange(len(east), dtype=np.float64) * dt_east
    return GroundMotion2D(time, east, north, dt_east, east_name, north_name)
