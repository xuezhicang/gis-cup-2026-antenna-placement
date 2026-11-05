"""Helpers shared by the point-swap and simulated-annealing steps."""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.solution_txt import (
    map_internal_building_ids,
    parse_solution_blocks,
    write_solution_block,
)


@dataclass(frozen=True)
class InputSolution:
    tau: float
    coordinates: np.ndarray


class StopFlag:
    """Becomes true after SIGINT/SIGTERM, or once an optional deadline passes.

    ``deadline`` is a ``time.monotonic()`` value; ``None`` means no time limit.
    """

    def __init__(self, deadline: float | None = None) -> None:
        self.signal_received = False
        self.deadline = deadline
        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)

    @property
    def time_up(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    @property
    def requested(self) -> bool:
        return self.signal_received or self.time_up

    def _stop(self, signum, _frame) -> None:
        print(f"Received signal {signum}; stopping after the current trial.", flush=True)
        self.signal_received = True


def read_solution(path: Path) -> InputSolution:
    blocks = parse_solution_blocks(path)
    if len(blocks) != 1:
        raise ValueError(f"{path} must contain exactly one three-line solution block")
    block = blocks[0]
    return InputSolution(block.tau, block.coordinates)


def read_point_coordinates(segments_csv: Path, chunksize: int) -> dict[int, tuple[float, float]]:
    coordinates: dict[int, tuple[float, float]] = {}
    for chunk in pd.read_csv(
        segments_csv,
        usecols=["point_index", "point_x", "point_y"],
        dtype={"point_index": np.int64, "point_x": np.float64, "point_y": np.float64},
        chunksize=chunksize,
    ):
        first_rows = chunk.drop_duplicates("point_index", keep="first")
        for row in first_rows.itertuples(index=False):
            coordinates.setdefault(
                int(row.point_index), (float(row.point_x), float(row.point_y))
            )
    if not coordinates:
        raise ValueError(f"No candidate coordinates found in {segments_csv}")
    return coordinates


def match_solution_points(
    solution_coordinates: np.ndarray,
    point_coordinates: dict[int, tuple[float, float]],
    tolerance: float,
) -> list[int]:
    ids = np.fromiter(point_coordinates, dtype=np.int64)
    xy = np.asarray([point_coordinates[int(point_id)] for point_id in ids])
    matched: list[int] = []
    for x, y in solution_coordinates:
        distances = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        positions = np.flatnonzero(distances <= tolerance)
        if len(positions) != 1:
            raise ValueError(
                f"Coordinate ({x:.17g}, {y:.17g}) matched {len(positions)} candidates "
                f"within tolerance {tolerance:g}"
            )
        matched.append(int(ids[int(positions[0])]))
    if len(set(matched)) != len(matched):
        raise ValueError("The input solution maps multiple antennas to one point_index")
    return matched


def write_best_solution(
    scorer,
    point_coordinates: dict[int, tuple[float, float]],
    official_building_ids: np.ndarray,
    output: Path,
) -> None:
    selected = [int(value) for value in scorer.selected_point_ids]
    coordinates = [point_coordinates[point_id] for point_id in selected]
    visible = scorer.current_visible_segments()
    building_ids = np.unique(np.asarray(visible["building_ids"], dtype=np.int64))
    scored_internal = [
        int(building_id)
        for building_id in building_ids
        if scorer.building_coverage(int(building_id)) >= scorer.tau
    ]
    if len(scored_internal) != int(scorer.current_score):
        raise RuntimeError("Scorer score and qualifying-building count disagree")
    scored_official = map_internal_building_ids(
        scored_internal,
        official_building_ids,
    )
    write_solution_block(output, scorer.tau, coordinates, scored_official)
