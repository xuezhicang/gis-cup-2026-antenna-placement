from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import geopandas as gpd
import numpy as np
import pandas as pd

import visibility_tree_cpp
from common.buildings import build_points_and_offsets
from common.mpi_compat import MPI


DEFAULT_GEOJSON = PROJECT_ROOT / "data" / "GIS-cup-sample-dataset.geojson"
SEGMENT_CSV_COLUMNS = [
    "point_index",
    "point_x",
    "point_y",
    "building_id",
    "building_perimeter",
    "edge_index",
    "edge_length",
    "s1",
    "s2",
    "angle1",
    "angle2",
    "x1",
    "y1",
    "x2",
    "y2",
]
NECESSARY_SEGMENT_COLUMNS = [
    "point_index",
    "building_id",
    "building_perimeter",
    "edge_index",
    "edge_length",
    "s1",
    "s2",
]
SEGMENT_COLUMN_DTYPES = {
    "point_index": np.int64,
    "point_x": np.float64,
    "point_y": np.float64,
    "building_id": np.int32,
    "building_perimeter": np.float64,
    "edge_index": np.int32,
    "edge_length": np.float64,
    "s1": np.float64,
    "s2": np.float64,
    "angle1": np.float64,
    "angle2": np.float64,
    "x1": np.float64,
    "y1": np.float64,
    "x2": np.float64,
    "y2": np.float64,
}


def read_candidate_points(path: Path, max_points: int | None) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError("candidate CSV must contain x and y columns")
        for row in reader:
            rows.append((float(row["x"]), float(row["y"])))
            if max_points is not None and len(rows) >= max_points:
                break
    return np.asarray(rows, dtype=np.float64).reshape((-1, 2))


def candidate_points_from_vertices(
    points: np.ndarray,
    max_points: int | None,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    _, unique_indices = np.unique(points, axis=0, return_index=True)
    unique_indices.sort()
    candidate_points = points[unique_indices]
    if max_points is not None:
        candidate_points = candidate_points[:max_points]
    return np.ascontiguousarray(candidate_points, dtype=np.float64)


def candidate_points_from_edges(
    points: np.ndarray,
    offsets: np.ndarray,
    spacing: float,
    max_points: int | None,
) -> np.ndarray:
    """
    Add interior points at exact spacing intervals and deduplicate only vertices.
    """
    if spacing <= 0.0 or not math.isfinite(spacing):
        raise ValueError("--edge-spacing must be a finite number greater than zero")

    candidate_points: list[tuple[float, float]] = []
    seen_vertices: set[tuple[float, float]] = set()

    def reached_max_points() -> bool:
        return max_points is not None and len(candidate_points) >= max_points

    for building_index in range(len(offsets) - 1):
        start = int(offsets[building_index])
        end = int(offsets[building_index + 1])
        vertices = points[start:end]

        for edge_index in range(len(vertices)):
            a = vertices[edge_index]
            b = vertices[(edge_index + 1) % len(vertices)]
            edge_length = math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))

            vertex = (float(a[0]), float(a[1]))
            if vertex not in seen_vertices:
                seen_vertices.add(vertex)
                candidate_points.append(vertex)
                if reached_max_points():
                    return np.ascontiguousarray(candidate_points, dtype=np.float64)

            endpoint_eps = 64.0 * np.finfo(np.float64).eps * max(
                1.0,
                edge_length,
                spacing,
            )
            interior_count = max(
                0,
                int(math.floor((edge_length - endpoint_eps) / spacing)),
            )
            for sample_index in range(1, interior_count + 1):
                distance = sample_index * spacing
                t = distance / edge_length
                x = float(a[0] + (b[0] - a[0]) * t)
                y = float(a[1] + (b[1] - a[1]) * t)
                candidate_points.append((x, y))
                if reached_max_points():
                    return np.ascontiguousarray(candidate_points, dtype=np.float64)

    return np.asarray(candidate_points, dtype=np.float64).reshape((-1, 2))


def point_slice_for_rank(num_points: int, rank: int, size: int) -> tuple[int, int]:
    base = num_points // size
    extra = num_points % size
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end


def output_path_for_rank(path: Path, rank: int, size: int) -> Path:
    if size == 1:
        return path
    return path.with_name(f"{path.stem}_rank_{rank:04d}{path.suffix}")


def candidate_points_gpkg_path(
    output_table_file: Path | None,
    requested_path: Path | None,
) -> Path:
    if requested_path is not None:
        return requested_path
    if output_table_file is not None:
        return output_table_file.with_name(
            f"{output_table_file.stem}_candidate_points.gpkg"
        )
    return Path("outputs/candidate_points.gpkg")


def abort_mpi_with_message(comm, is_root: bool, message: str) -> None:
    if is_root:
        print(f"error: {message}", flush=True)
    comm.Abort(2)


def validate_input_geojson(path: Path, comm, is_root: bool) -> None:
    if path == Path("."):
        abort_mpi_with_message(
            comm,
            is_root,
            (
                "--geojson resolved to '.', which usually means the shell "
                "variable passed to --geojson was empty. Set GEOJSON to the "
                "GeoJSON file path before running mpirun."
            ),
        )
    if not path.exists():
        abort_mpi_with_message(comm, is_root, f"GeoJSON file not found: {path}")
    if not path.is_file():
        abort_mpi_with_message(comm, is_root, f"GeoJSON path is not a file: {path}")


def validate_output_path_arg(
    path: Path | None,
    option_name: str,
    comm,
    is_root: bool,
) -> None:
    if path is None:
        return
    if path == Path("."):
        abort_mpi_with_message(
            comm,
            is_root,
            (
                f"{option_name} resolved to '.', which usually means the shell "
                f"variable passed to {option_name} was empty."
            ),
        )
    if path.exists() and path.is_dir():
        abort_mpi_with_message(
            comm,
            is_root,
            f"{option_name} must be a file path, not a directory: {path}",
        )


def write_candidate_points_gpkg(
    path: Path,
    candidate_points: np.ndarray,
    crs,
    chunk_size: int = 250_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(candidate_points), chunk_size):
        end = min(start + chunk_size, len(candidate_points))
        point_chunk = candidate_points[start:end]
        point_indices = np.arange(start, end, dtype=np.int64)
        points_gdf = gpd.GeoDataFrame(
            {
                "point_index": point_indices,
                "x": point_chunk[:, 0],
                "y": point_chunk[:, 1],
            },
            geometry=gpd.points_from_xy(point_chunk[:, 0], point_chunk[:, 1]),
            crs=crs,
        )
        points_gdf.to_file(
            path,
            layer="candidate_points",
            driver="GPKG",
            mode="w" if start == 0 else "a",
            index=False,
        )


def write_segments_csv(
    path: Path,
    result: dict | None,
    candidate_points: np.ndarray,
    point_start: int = 0,
    append: bool = False,
) -> None:
    """Write (or append) visible segments; ``result=None`` writes only the header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if result is None:
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(SEGMENT_CSV_COLUMNS)
        return

    offsets = result["point_offsets"]
    building_ids = result["building_ids"]
    edge_indices = result["edge_indices"]
    segment_data = result["segment_data"]
    building_perimeters = result["building_perimeters"]
    edge_offsets = result["edge_offsets"]
    edge_lengths = result["edge_lengths"]

    with path.open("a" if append else "w", newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(SEGMENT_CSV_COLUMNS)
        for point_index in range(len(candidate_points)):
            start = int(offsets[point_index])
            end = int(offsets[point_index + 1])
            px, py = candidate_points[point_index]
            for row_index in range(start, end):
                s1, s2, angle1, angle2, x1, y1, x2, y2 = segment_data[row_index]
                building_id = int(building_ids[row_index])
                edge_index = int(edge_indices[row_index])
                global_edge_index = int(edge_offsets[building_id]) + edge_index
                writer.writerow([
                    point_start + point_index,
                    px,
                    py,
                    building_id,
                    building_perimeters[building_id],
                    edge_index,
                    edge_lengths[global_edge_index],
                    s1,
                    s2,
                    angle1,
                    angle2,
                    x1,
                    y1,
                    x2,
                    y2,
                ])


def merge_rank_csvs(rank_paths: list[Path], output_path: Path) -> int:
    rows_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as out_f:
        writer = csv.writer(out_f)
        wrote_header = False
        for rank_path in rank_paths:
            with rank_path.open(newline="") as in_f:
                reader = csv.reader(in_f)
                header = next(reader, None)
                if header is None:
                    continue
                if not wrote_header:
                    writer.writerow(header)
                    wrote_header = True
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
    return rows_written


def empty_segment_dataframe(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(dtype=SEGMENT_COLUMN_DTYPES[column])
            for column in columns
        },
        columns=columns,
    )


def parquet_paths_for_table_file(output_table_file: Path) -> tuple[Path, Path]:
    necessary_path = output_table_file.with_name(
        f"{output_table_file.stem}_necessary.parquet"
    )
    full_path = output_table_file.with_name(f"{output_table_file.stem}_full.parquet")
    return necessary_path, full_path


def write_parquet_tables_from_csv(
    csv_path: Path,
    necessary_path: Path,
    full_path: Path,
    chunksize: int = 500_000,
) -> tuple[int, int]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Writing Parquet outputs requires pyarrow. Install project "
            "dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    necessary_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    necessary_writer = None
    full_writer = None
    rows_written = 0

    try:
        reader = pd.read_csv(
            csv_path,
            dtype=SEGMENT_COLUMN_DTYPES,
            chunksize=chunksize,
        )
        for chunk in reader:
            chunk = chunk[SEGMENT_CSV_COLUMNS]
            necessary_chunk = chunk[NECESSARY_SEGMENT_COLUMNS]

            full_table = pa.Table.from_pandas(chunk, preserve_index=False)
            necessary_table = pa.Table.from_pandas(
                necessary_chunk,
                preserve_index=False,
            )

            if full_writer is None:
                full_writer = pq.ParquetWriter(full_path, full_table.schema)
            if necessary_writer is None:
                necessary_writer = pq.ParquetWriter(
                    necessary_path,
                    necessary_table.schema,
                )

            full_writer.write_table(full_table)
            necessary_writer.write_table(necessary_table)
            rows_written += len(chunk)
    finally:
        if full_writer is not None:
            full_writer.close()
        if necessary_writer is not None:
            necessary_writer.close()

    if rows_written == 0:
        full_table = pa.Table.from_pandas(
            empty_segment_dataframe(SEGMENT_CSV_COLUMNS),
            preserve_index=False,
        )
        necessary_table = pa.Table.from_pandas(
            empty_segment_dataframe(NECESSARY_SEGMENT_COLUMNS),
            preserve_index=False,
        )
        pq.write_table(full_table, full_path)
        pq.write_table(necessary_table, necessary_path)

    return rows_written, rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate visible building-edge segments for antenna candidate points."
    )
    
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--max-leaf-size", type=int, default=64)
    parser.add_argument("--ring-step", type=float, default=19.0)
    parser.add_argument("--max-rings", type=int, default=1000000000)
    parser.add_argument(
        "--point",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Run one point. If omitted, use the first building representative point.",
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        help="Optional batch input CSV with x,y columns.",
    )
    parser.add_argument(
        "--points-from-vertices",
        action="store_true",
        help="Use building vertices from the GeoJSON as candidate antenna points.",
    )
    parser.add_argument(
        "--points-from-edges",
        action="store_true",
        help=(
            "Densify all building edges and use the generated boundary points as "
            "candidate antenna points. Original edge vertices are always included."
        ),
    )
    parser.add_argument(
        "--edge-spacing",
        type=float,
        default=1.0,
        help=(
            "Maximum distance between candidates generated along an edge when "
            "--points-from-edges is used. Default: 1.0."
        ),
    )
    parser.add_argument("--max-points", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Candidate points processed (and written) per batch. Default: 2000.",
    )
    parser.add_argument(
        "--output-table-file",
        dest="output_table_file",
        type=Path,
        help=(
            "Optional output CSV for visible segments. With --write-parquet, two "
            "Parquet siblings are also written: *_necessary.parquet with the "
            "scorer columns and *_full.parquet with all CSV columns."
        ),
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Also write Parquet copies of the segment table (not needed by later steps).",
    )
    parser.add_argument(
        "--output-csv",
        dest="output_table_file",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--candidate-points-gpkg",
        type=Path,
        help=(
            "Output GeoPackage for all candidate points. If omitted, derive the "
            "path from --output-table-file."
        ),
    )
    parser.add_argument(
        "--benchmark-single-loop",
        action="store_true",
        help="Also time a Python loop that calls visible_from_point once per point.",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_root = rank == 0

    validate_input_geojson(args.geojson, comm, is_root)
    validate_output_path_arg(
        args.output_table_file,
        "--output-table-file",
        comm,
        is_root,
    )
    validate_output_path_arg(
        args.candidate_points_gpkg,
        "--candidate-points-gpkg",
        comm,
        is_root,
    )

    # read the GeoJSON file and print basic info
    t0 = time.perf_counter()
    gdf = gpd.read_file(args.geojson)
    if is_root:
        print("mpi ranks:", size)
        print("geojson:", args.geojson)
        print("rows:", len(gdf))
        print("crs:", gdf.crs)
        print("read time (s):", round(time.perf_counter() - t0, 3))

    # convert the GeoDataFrame to points and offsets for the visibility engine
    t1 = time.perf_counter()
    points, offsets, source_indices = build_points_and_offsets(gdf)
    if is_root:
        print("valid buildings:", len(source_indices))
        print("points shape:", points.shape)
        print("offsets shape:", offsets.shape)
        print("convert time (s):", round(time.perf_counter() - t1, 3))

    # build the visibility engine
    t2 = time.perf_counter()
    engine = visibility_tree_cpp.VisibilityEngine(
        points,
        offsets,
        args.max_leaf_size,
        args.ring_step,
    )
    if is_root:
        print("engine build time (s):", round(time.perf_counter() - t2, 3))
        print("tree buildings:", engine.num_buildings)
        print("tree nodes:", engine.num_nodes)
        print("tree leaves:", engine.num_leaves)

    batch_source_count = sum(
        (
            bool(args.points_csv),
            args.points_from_vertices,
            args.points_from_edges,
        )
    )
    if batch_source_count > 1:
        raise ValueError(
            "Use only one of --points-csv, --points-from-vertices, or "
            "--points-from-edges."
        )

    # Handle batch visibility queries from a CSV, vertices, or densified edges.
    if args.points_csv or args.points_from_vertices or args.points_from_edges:
        if args.points_csv:
            candidate_points = read_candidate_points(args.points_csv, args.max_points)
            if is_root:
                print("candidate source:", args.points_csv)
        elif args.points_from_vertices:
            candidate_points = candidate_points_from_vertices(points, args.max_points)
            if is_root:
                print("candidate source: building vertices")
        else:
            candidate_points = candidate_points_from_edges(
                points,
                offsets,
                args.edge_spacing,
                args.max_points,
            )

            if is_root:
                print("candidate source: densified building edges")
                print("edge spacing:", args.edge_spacing)
        if len(candidate_points) == 0:
            raise ValueError("No candidate points were generated.")
        if is_root:
            print("candidate points:", candidate_points.shape)
            candidate_gpkg = candidate_points_gpkg_path(
                args.output_table_file,
                args.candidate_points_gpkg,
            )
            write_candidate_points_gpkg(
                candidate_gpkg,
                candidate_points,
                gdf.crs,
            )
            print("candidate points gpkg:", candidate_gpkg)

        point_start, point_end = point_slice_for_rank(len(candidate_points), rank, size)
        local_candidate_points = np.ascontiguousarray(
            candidate_points[point_start:point_end],
            dtype=np.float64,
        )
        print(
            f"rank {rank}/{size}: point slice [{point_start}, {point_end}) "
            f"shape {local_candidate_points.shape}",
            flush=True,
        )

        comm.Barrier()
        t3 = MPI.Wtime()
        rank_output_csv = (
            output_path_for_rank(args.output_table_file, rank, size)
            if args.output_table_file
            else None
        )
        local_count = len(local_candidate_points)
        batch_size = max(1, args.batch_size)
        processed_local = 0
        if rank_output_csv is not None and local_count == 0:
            write_segments_csv(rank_output_csv, None, local_candidate_points, point_start)

        # Points are independent, so processing them in batches gives the same
        # table while keeping memory flat and allowing progress reports.
        for batch_start in range(0, local_count, batch_size):
            batch = local_candidate_points[batch_start:batch_start + batch_size]
            result = engine.visible_from_points(batch, args.max_rings)
            processed_local += int(np.sum(result["processed_node_counts"]))
            if rank_output_csv is not None:
                write_segments_csv(
                    rank_output_csv,
                    result,
                    batch,
                    point_start + batch_start,
                    append=batch_start > 0,
                )
            del result
            if is_root:
                done = batch_start + len(batch)
                elapsed = MPI.Wtime() - t3
                remaining = elapsed / done * (local_count - done)
                print(
                    f"rank 0 progress: {done}/{local_count} points "
                    f"({100.0 * done / local_count:.0f}%), "
                    f"{elapsed:.0f}s elapsed, about {remaining:.0f}s left",
                    flush=True,
                )

        dt = MPI.Wtime() - t3
        print(
            "rank", rank, "visibility time (s):", round(dt, 3),
            "average (ms/point):",
            round(dt / local_count * 1000.0, 4) if local_count else 0.0,
            flush=True,
        )

        processed_sum = comm.reduce(processed_local, op=MPI.SUM, root=0)
        if is_root:
            processed_mean = processed_sum / len(candidate_points) if len(candidate_points) else 0.0
            print("processed nodes mean:", float(processed_mean))

        if args.output_table_file:
            print("rank", rank, "wrote:", rank_output_csv, flush=True)
            if size > 1:
                comm.Barrier()
                if is_root:
                    rank_paths = [
                        output_path_for_rank(args.output_table_file, rank_id, size)
                        for rank_id in range(size)
                    ]
                    rows_written = merge_rank_csvs(rank_paths, args.output_table_file)
                    for rank_path in rank_paths:
                        rank_path.unlink()
                    print(
                        "merged",
                        size,
                        "rank CSVs into",
                        args.output_table_file,
                        "with",
                        rows_written,
                        "data rows",
                    )
            if is_root and args.write_parquet:
                necessary_path, full_path = parquet_paths_for_table_file(
                    args.output_table_file
                )
                necessary_rows, full_rows = write_parquet_tables_from_csv(
                    args.output_table_file,
                    necessary_path,
                    full_path,
                )
                print(
                    "wrote necessary parquet:",
                    necessary_path,
                    "rows:",
                    necessary_rows,
                )
                print("wrote full parquet:", full_path, "rows:", full_rows)
        return

    if size > 1 and not is_root:
        return

    if args.point:
        print("single point source:", args.point)
        x, y = args.point
    else:
        sample_geom = gdf.geometry.iloc[int(source_indices[0])]
        sample_point = sample_geom.representative_point()
        x, y = float(sample_point.x), float(sample_point.y)
    
    # run a single visibility query for the specified point
    print("single point:", (x, y))

    t4 = time.perf_counter()
    result = engine.visible_from_point(x, y, args.max_rings)
    print("single visibility time (s):", round(time.perf_counter() - t4, 6))
    print("visible segment count:", len(result["building_ids"]))
    print("visited nodes:", result["visited_node_count"])
    print("processed nodes:", result["processed_node_count"])
    print("rings:", result["ring_count"])
    print("remaining angles shape:", result["remaining_angles"].shape)

    # rows = min(10, len(result["building_ids"]))
    # for i in range(rows):
    #     s1, s2, angle1, angle2, x1, y1, x2, y2 = result["segment_data"][i]
    #     print(
    #         "segment",
    #         i,
    #         "building",
    #         int(result["building_ids"][i]),
    #         "edge",
    #         int(result["edge_indices"][i]),
    #         "s",
    #         (round(float(s1), 6), round(float(s2), 6)),
    #         "angle",
    #         (round(float(angle1), 6), round(float(angle2), 6)),
    #         "from/to",
    #         (round(float(x1), 3), round(float(y1), 3)),
    #         (round(float(x2), 3), round(float(y2), 3)),
    #     )


if __name__ == "__main__":
    main()
