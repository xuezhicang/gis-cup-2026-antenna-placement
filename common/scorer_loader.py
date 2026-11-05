"""Load a visible-segment table into the C++ incremental coverage scorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import coverage_scorer_cpp


SEGMENT_COLUMNS = [
    "point_index",
    "building_id",
    "building_perimeter",
    "edge_index",
    "edge_length",
    "s1",
    "s2",
]


def load_incremental_scorer(
    segments_csv: Path,
    tau: float,
    chunksize: int,
) -> coverage_scorer_cpp.IncrementalCoverageScorer:
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")

    point_ids: list[int] = []
    point_offset_starts: list[int] = []
    building_id_chunks: list[np.ndarray] = []
    edge_index_chunks: list[np.ndarray] = []
    edge_length_chunks: list[np.ndarray] = []
    segment_range_chunks: list[np.ndarray] = []
    perimeter_by_building: dict[int, float] = {}
    total_rows = 0

    try:
        reader = pd.read_csv(
            segments_csv,
            usecols=SEGMENT_COLUMNS,
            dtype={
                "point_index": np.int64,
                "building_id": np.int32,
                "building_perimeter": np.float64,
                "edge_index": np.int32,
                "edge_length": np.float64,
                "s1": np.float64,
                "s2": np.float64,
            },
            chunksize=chunksize,
        )
        for chunk in reader:
            if chunk.empty:
                continue

            chunk_point_ids = chunk["point_index"].to_numpy(dtype=np.int64)
            if np.any(chunk_point_ids[1:] < chunk_point_ids[:-1]):
                raise ValueError("segment CSV must be sorted by point_index")
            if point_ids and int(chunk_point_ids[0]) < point_ids[-1]:
                raise ValueError("segment CSV must be sorted by point_index")

            run_starts = np.flatnonzero(
                np.r_[True, chunk_point_ids[1:] != chunk_point_ids[:-1]]
            )
            for run_start in run_starts:
                point_id = int(chunk_point_ids[run_start])
                if point_ids and point_id == point_ids[-1]:
                    continue
                point_ids.append(point_id)
                point_offset_starts.append(total_rows + int(run_start))

            building_ids = np.ascontiguousarray(
                chunk["building_id"].to_numpy(dtype=np.int32)
            )
            building_perimeters = chunk["building_perimeter"].to_numpy(
                dtype=np.float64
            )
            unique_building_ids, first_indices = np.unique(
                building_ids,
                return_index=True,
            )
            for building_id, first_index in zip(unique_building_ids, first_indices):
                building_id_int = int(building_id)
                perimeter = float(building_perimeters[first_index])
                previous = perimeter_by_building.get(building_id_int)
                if previous is not None and not np.isclose(
                    previous,
                    perimeter,
                    rtol=1e-10,
                    atol=1e-10,
                ):
                    raise ValueError(
                        f"building {building_id_int} has inconsistent perimeter values"
                    )
                perimeter_by_building[building_id_int] = perimeter

            building_id_chunks.append(building_ids)
            edge_index_chunks.append(
                np.ascontiguousarray(chunk["edge_index"].to_numpy(dtype=np.int32))
            )
            edge_length_chunks.append(
                np.ascontiguousarray(chunk["edge_length"].to_numpy(dtype=np.float64))
            )
            segment_range_chunks.append(
                np.ascontiguousarray(
                    chunk[["s1", "s2"]].to_numpy(dtype=np.float64)
                )
            )
            total_rows += len(chunk)
    except ValueError as exc:
        missing_columns = ", ".join(SEGMENT_COLUMNS)
        raise ValueError(
            f"Could not load {segments_csv}. Required columns: {missing_columns}. "
            f"Original error: {exc}"
        ) from exc

    if total_rows == 0:
        raise ValueError(f"No visible-segment rows were found in {segments_csv}")

    point_offsets = np.ascontiguousarray(
        np.asarray(point_offset_starts + [total_rows], dtype=np.int64)
    )
    point_ids_array = np.ascontiguousarray(np.asarray(point_ids, dtype=np.int64))
    segment_building_ids = np.ascontiguousarray(np.concatenate(building_id_chunks))
    segment_edge_indices = np.ascontiguousarray(np.concatenate(edge_index_chunks))
    segment_edge_lengths = np.ascontiguousarray(np.concatenate(edge_length_chunks))
    segment_ranges = np.ascontiguousarray(np.concatenate(segment_range_chunks, axis=0))

    max_building_id = max(perimeter_by_building)
    building_perimeters = np.full(max_building_id + 1, np.nan, dtype=np.float64)
    for building_id, perimeter in perimeter_by_building.items():
        building_perimeters[building_id] = perimeter

    return coverage_scorer_cpp.IncrementalCoverageScorer(
        point_ids_array,
        point_offsets,
        segment_building_ids,
        segment_edge_indices,
        segment_ranges,
        segment_edge_lengths,
        building_perimeters,
        tau,
    )
