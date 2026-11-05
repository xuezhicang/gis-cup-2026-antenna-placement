#!/usr/bin/env python3
"""Score a hand-picked set of candidate points and export the viewed segments.

Debugging/visualization helper; not part of the main pipeline.  Example:

    python tools/export_coverage.py --segments-csv outputs/cache/<name>/visible_segments \
        --tau 0.5 --point-ids 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

import coverage_scorer_cpp
from common.scorer_loader import load_incremental_scorer


EXPORT_COLUMNS = [
    "point_index",
    "point_x",
    "point_y",
    "building_id",
    "edge_index",
    "s1",
    "s2",
    "x1",
    "y1",
    "x2",
    "y2",
]


def export_current_coverage_gpkg(
    scorer: coverage_scorer_cpp.IncrementalCoverageScorer,
    segments_csv: Path,
    output_gpkg: Path,
    crs: str,
    chunksize: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    selected_point_ids = [int(value) for value in scorer.selected_point_ids]
    if not selected_point_ids:
        raise ValueError("No selected points are available to export.")

    current = scorer.current_visible_segments()
    building_ids = np.asarray(current["building_ids"], dtype=np.int32)
    edge_indices = np.asarray(current["edge_indices"], dtype=np.int32)
    segment_ranges = np.asarray(current["segment_ranges"], dtype=np.float64)
    edge_lengths = np.asarray(current["edge_lengths"], dtype=np.float64)
    target_edges = set(
        zip(building_ids.astype(int).tolist(), edge_indices.astype(int).tolist())
    )

    selected_id_set = set(selected_point_ids)
    point_coordinates: dict[int, tuple[float, float]] = {}
    edge_endpoints: dict[
        tuple[int, int],
        tuple[float, float, float, float],
    ] = {}

    reader = pd.read_csv(
        segments_csv,
        usecols=EXPORT_COLUMNS,
        dtype={
            "point_index": np.int64,
            "point_x": np.float64,
            "point_y": np.float64,
            "building_id": np.int32,
            "edge_index": np.int32,
            "s1": np.float64,
            "s2": np.float64,
            "x1": np.float64,
            "y1": np.float64,
            "x2": np.float64,
            "y2": np.float64,
        },
        chunksize=chunksize,
    )
    for chunk in reader:
        selected_rows = chunk[chunk["point_index"].isin(selected_id_set)]
        for row in selected_rows.itertuples(index=False):
            point_id = int(row.point_index)
            point_coordinates.setdefault(
                point_id,
                (float(row.point_x), float(row.point_y)),
            )

            edge_key = (int(row.building_id), int(row.edge_index))
            if edge_key not in target_edges or edge_key in edge_endpoints:
                continue

            parameter_length = float(row.s2 - row.s1)
            if abs(parameter_length) <= 1e-12:
                continue
            edge_dx = float(row.x2 - row.x1) / parameter_length
            edge_dy = float(row.y2 - row.y1) / parameter_length
            edge_start_x = float(row.x1) - float(row.s1) * edge_dx
            edge_start_y = float(row.y1) - float(row.s1) * edge_dy
            edge_endpoints[edge_key] = (
                edge_start_x,
                edge_start_y,
                edge_start_x + edge_dx,
                edge_start_y + edge_dy,
            )

    missing_points = [
        point_id for point_id in selected_point_ids
        if point_id not in point_coordinates
    ]
    if missing_points:
        raise ValueError(
            f"Could not recover coordinates for {len(missing_points)} selected points."
        )
    missing_edges = target_edges.difference(edge_endpoints)
    if missing_edges:
        raise ValueError(
            f"Could not recover geometry for {len(missing_edges)} viewed edges."
        )

    points_df = pd.DataFrame(
        {
            "selection_order": np.arange(1, len(selected_point_ids) + 1),
            "point_index": selected_point_ids,
        }
    )
    points_geometry = [
        Point(point_coordinates[point_id])
        for point_id in selected_point_ids
    ]
    points_gdf = gpd.GeoDataFrame(points_df, geometry=points_geometry, crs=crs)

    segment_rows: list[dict] = []
    segment_geometry: list[LineString] = []
    for row_index in range(len(building_ids)):
        building_id = int(building_ids[row_index])
        edge_index = int(edge_indices[row_index])
        s1 = float(segment_ranges[row_index, 0])
        s2 = float(segment_ranges[row_index, 1])
        edge_length = float(edge_lengths[row_index])
        ax, ay, bx, by = edge_endpoints[(building_id, edge_index)]
        dx = bx - ax
        dy = by - ay
        segment_rows.append(
            {
                "building_id": building_id,
                "edge_index": edge_index,
                "s1": s1,
                "s2": s2,
                "edge_length": edge_length,
                "visible_length": (s2 - s1) * edge_length,
            }
        )
        segment_geometry.append(
            LineString(
                [
                    (ax + s1 * dx, ay + s1 * dy),
                    (ax + s2 * dx, ay + s2 * dy),
                ]
            )
        )

    segments_gdf = gpd.GeoDataFrame(
        segment_rows,
        geometry=segment_geometry,
        crs=crs,
    )

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()
    points_gdf.to_file(
        output_gpkg,
        layer="selected_points",
        driver="GPKG",
        index=False,
    )
    segments_gdf.to_file(
        output_gpkg,
        layer="viewed_segments",
        driver="GPKG",
        mode="a",
        index=False,
    )
    return points_gdf, segments_gdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate independent hard/soft incremental coverage scoring "
            "from a self-contained visible-segment CSV."
        )
    )
    parser.add_argument("--segments-csv", type=Path, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument(
        "--point-ids",
        type=int,
        nargs="+",
        required=True,
        help="Candidate point IDs to evaluate against the current selected state.",
    )
    parser.add_argument(
        "--initial-point-ids",
        type=int,
        nargs="*",
        default=[],
        help="Point IDs to add before evaluating the candidates.",
    )

    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--output-gpkg",
        type=Path,
        default=Path("outputs/incremental_coverage_selection.gpkg"),
    )
    parser.add_argument("--crs", default="EPSG:32611")
    args = parser.parse_args()

    duplicate_ids = set(args.point_ids).intersection(args.initial_point_ids)
    if duplicate_ids:
        duplicate_text = ", ".join(str(value) for value in sorted(duplicate_ids))
        raise ValueError(
            f"Candidate point IDs are already in the initial selection: {duplicate_text}"
        )

    start = time.perf_counter()
    scorer = load_incremental_scorer(args.segments_csv, args.tau, args.chunksize)
    load_seconds = time.perf_counter() - start

    print("load csv (s):", round(load_seconds, 3))


    start = time.perf_counter()
    if args.initial_point_ids:
        scorer.add_points(np.asarray(args.initial_point_ids, dtype=np.int64))
    initial_add_seconds = time.perf_counter() - start
    print("initial add_points (s):", round(initial_add_seconds, 3))

    print("segments csv:", args.segments_csv)
    print("tau:", scorer.tau)
    print("points loaded:", scorer.num_points)
    print("input segment rows:", scorer.input_segment_count)
    print("stored merged intervals:", scorer.stored_interval_count)
    print("initial selected points:", scorer.selected_count)
    print("initial current score:", scorer.current_score)

    start = time.perf_counter()
    print("\nIndependent point scoring:")
    for point_id in args.point_ids:
        hard = scorer.hard_score(point_id)
        soft = scorer.soft_score(point_id)
        combined = scorer.evaluate_point(point_id)
        print(
            f"point {point_id}: hard={hard}, soft={soft:.12g}, "
            f"unique_length={combined['unique_length']:.12g}"
        )
    score_seconds = time.perf_counter() - start
    print("evaluation time (ms):", round(score_seconds*1000, 3))

    start = time.perf_counter()
    candidate_ids = np.asarray(args.point_ids, dtype=np.int64)
    batch = scorer.evaluate_all(candidate_ids)

    best_position = max(
        range(len(candidate_ids)),
        key=lambda index: (
            int(batch["hard_scores"][index]),
            float(batch["soft_scores"][index]),
            float(batch["unique_lengths"][index]),
            -int(batch["point_indices"][index]),
        ),
    )
    best_point_id = int(batch["point_indices"][best_position])
    print("\nbest point:", best_point_id)
    print("best hard score:", int(batch["hard_scores"][best_position]))
    print("best soft score:", float(batch["soft_scores"][best_position]))
    print("best unique length:", float(batch["unique_lengths"][best_position]))
    judge_seconds = time.perf_counter() - start
    print("judge time (ms):", round(judge_seconds*1000, 3))
    print()

    start = time.perf_counter()
    scorer.add_point(best_point_id)
    print("current score after add_point:", scorer.current_score)
    print("selected points after add_point:", scorer.selected_count)
    add_seconds = time.perf_counter() - start
    print("add_point time (ms):", round(add_seconds*1000, 3))

    export_start = time.perf_counter()
    points_gdf, segments_gdf = export_current_coverage_gpkg(
        scorer,
        args.segments_csv,
        args.output_gpkg,
        args.crs,
        args.chunksize,
    )
    print("output gpkg:", args.output_gpkg)
    print("selected point features:", len(points_gdf))
    print("viewed union segment features:", len(segments_gdf))
    print("export time (s):", round(time.perf_counter() - export_start, 3))


if __name__ == "__main__":
    main()
