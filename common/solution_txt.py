"""Read, validate, and write GIS Cup solution TXT blocks."""

from __future__ import annotations

import math
import numbers
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
POINT_RE = re.compile(rf"\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)")


@dataclass(frozen=True)
class SolutionBlock:
    tau: float
    k: int
    coordinates: np.ndarray
    building_ids: tuple[int, ...]


def _geometry_is_used_by_visibility_engine(geometry) -> bool:
    """Mirror the geometry filtering in build_points_and_offsets()."""
    from shapely.geometry import MultiPolygon, Polygon

    if geometry is None or geometry.is_empty:
        return False

    if isinstance(geometry, Polygon):
        coordinates = list(geometry.exterior.coords)
    elif isinstance(geometry, MultiPolygon):
        coordinates = list(geometry.envelope.exterior.coords)
    else:
        return False

    if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
        coordinates = coordinates[:-1]
    return len(coordinates) >= 3


def load_official_building_ids(
    geojson_path: Path,
    building_id_column: str = "id",
) -> np.ndarray:
    """Return official IDs indexed by the scorer's dense internal building ID."""
    import geopandas as gpd

    buildings = gpd.read_file(geojson_path)
    if building_id_column not in buildings.columns:
        raise ValueError(
            f"{geojson_path} has no building ID column {building_id_column!r}"
        )

    source_indices = [
        source_index
        for source_index, geometry in enumerate(buildings.geometry)
        if _geometry_is_used_by_visibility_engine(geometry)
    ]
    if not source_indices:
        raise ValueError(f"No usable building geometries were found in {geojson_path}")

    official_ids: list[int] = []
    for source_index, value in zip(
        source_indices,
        buildings.iloc[source_indices][building_id_column],
    ):
        if isinstance(value, numbers.Integral) and not isinstance(
            value,
            (bool, np.bool_),
        ):
            official_id = int(value)
        elif isinstance(value, numbers.Real):
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or not numeric_value.is_integer():
                raise ValueError(
                    f"Building row {source_index} has non-integer official ID {value!r}"
                )
            official_id = int(numeric_value)
        elif isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
            official_id = int(value)
        else:
            raise ValueError(
                f"Building row {source_index} has non-integer official ID {value!r}"
            )
        official_ids.append(official_id)

    if len(set(official_ids)) != len(official_ids):
        raise ValueError(
            f"Column {building_id_column!r} in {geojson_path} contains duplicate IDs"
        )
    return np.ascontiguousarray(official_ids, dtype=np.int64)


def map_internal_building_ids(
    internal_ids: Iterable[int],
    official_ids_by_internal: np.ndarray,
) -> list[int]:
    internal = np.asarray(list(internal_ids), dtype=np.int64)
    if internal.size == 0:
        return []
    if np.any(internal < 0) or np.any(internal >= len(official_ids_by_internal)):
        bad = internal[
            (internal < 0) | (internal >= len(official_ids_by_internal))
        ]
        raise ValueError(
            "Internal building IDs are outside the official-ID mapping: "
            + ",".join(str(int(value)) for value in bad[:10])
        )

    official = [
        int(value) for value in official_ids_by_internal[internal].tolist()
    ]
    if len(set(official)) != len(official):
        raise ValueError("Mapped official building IDs are not unique")
    return sorted(official)


def format_solution_block(
    tau: float,
    coordinates: Iterable[tuple[float, float]] | np.ndarray,
    building_ids: Iterable[int],
) -> str:
    points = np.asarray(list(coordinates), dtype=np.float64)
    if points.size == 0:
        points = points.reshape((0, 2))
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coordinates must have shape (k, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("coordinates contain a non-finite value")
    if not math.isfinite(float(tau)):
        raise ValueError("tau must be finite")

    ids = [int(value) for value in building_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("serviced building IDs must be unique")

    point_line = ",".join(
        f"({float(x):.17g},{float(y):.17g})" for x, y in points
    )
    building_line = ",".join(str(value) for value in ids)
    return f"({float(tau):g}, {len(points)})\n{point_line}\n{building_line}\n"


def write_solution_block(
    output_path: Path,
    tau: float,
    coordinates: Iterable[tuple[float, float]] | np.ndarray,
    building_ids: Iterable[int],
) -> None:
    text = format_solution_block(tau, coordinates, building_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output_path)


def parse_solution_blocks(path: Path) -> list[SolutionBlock]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) % 3 != 0 or not lines:
        raise ValueError(f"{path} must contain a nonzero multiple of three lines")

    blocks: list[SolutionBlock] = []
    for offset in range(0, len(lines), 3):
        header_values = re.findall(NUMBER, lines[offset])
        if len(header_values) != 2:
            raise ValueError(
                f"{path}:{offset + 1} must contain exactly the pair (tau, k)"
            )
        tau = float(header_values[0])
        k_value = float(header_values[1])
        if not math.isfinite(tau):
            raise ValueError(f"{path}:{offset + 1} has a non-finite tau")
        if not math.isfinite(k_value) or not k_value.is_integer() or k_value < 0:
            raise ValueError(f"{path}:{offset + 1} has an invalid antenna count")
        k = int(k_value)
        coordinates = np.ascontiguousarray(
            [
                (float(x), float(y))
                for x, y in POINT_RE.findall(lines[offset + 1])
            ],
            dtype=np.float64,
        ).reshape((-1, 2))
        if len(coordinates) != k:
            raise ValueError(
                f"{path}:{offset + 2} has {len(coordinates)} coordinates, expected {k}"
            )

        id_text = lines[offset + 2].strip()
        id_tokens = re.split(r"[,\s]+", id_text) if id_text else []
        if any(re.fullmatch(r"[-+]?\d+", token) is None for token in id_tokens):
            raise ValueError(f"{path}:{offset + 3} has an invalid building ID list")
        building_ids = tuple(int(value) for value in id_tokens)
        if len(building_ids) != len(set(building_ids)):
            raise ValueError(f"{path}:{offset + 3} contains duplicate building IDs")
        blocks.append(SolutionBlock(tau, k, coordinates, building_ids))
    return blocks


def is_canonical_single_block(path: Path) -> bool:
    """Return whether *path* is one complete block in the required format."""
    try:
        blocks = parse_solution_blocks(path)
    except (OSError, UnicodeError, ValueError):
        return False
    if len(blocks) != 1:
        return False

    block = blocks[0]
    expected = format_solution_block(
        block.tau,
        block.coordinates,
        block.building_ids,
    )
    try:
        return path.read_text(encoding="utf-8") == expected
    except (OSError, UnicodeError):
        return False
