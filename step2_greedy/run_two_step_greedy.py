from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import geopandas as gpd
import numpy as np

from common.mpi_compat import MPI
from common.scorer_loader import load_incremental_scorer
from step2_greedy.two_step_greedy import run_two_step_greedy
from common.solution_txt import (
    load_official_building_ids,
    map_internal_building_ids,
    write_solution_block,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all alpha x lookahead-width 2-step greedy parameter "
            "combinations across MPI ranks."
        )
    )
    parser.add_argument("--segments-csv", type=Path, required=True)
    parser.add_argument(
        "--geojson",
        type=Path,
        required=True,
        help="Building GeoJSON used to map internal scorer IDs to official IDs.",
    )
    parser.add_argument(
        "--building-id-column",
        default="id",
        help="Official building ID column in --geojson. Default: id.",
    )
    parser.add_argument("--tau", type=float, nargs="+", default=[0.25])
    parser.add_argument("--antenna-count", type=int, nargs="+", default=[50])
    parser.add_argument(
        "--lookahead-widths",
        "--lookahead-width",
        dest="lookahead_widths",
        type=int,
        nargs="+",
        default=[100],
        help="One or more lookahead widths. Both flag spellings are accepted.",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[1.0],
        help="One or more alpha values in [0, 1].",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--candidate-points-gpkg",
        type=Path,
        required=True,
        help="GeoPackage containing point_index, x, and y for candidate points.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help=(
            "Base output directory. A tau_<tau>_k_<antenna-count> directory "
            "is created inside it. Default: outputs."
        ),
    )
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(count <= 0 for count in args.antenna_count):
        raise ValueError("--antenna-count values must all be greater than zero")
    if any(not 0.0 <= tau <= 1.0 for tau in args.tau):
        raise ValueError("--tau values must all be in [0, 1]")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be greater than zero")
    if any(width <= 0 for width in args.lookahead_widths):
        raise ValueError("--lookahead-widths must all be greater than zero")
    if any(not 0.0 <= alpha <= 1.0 for alpha in args.alpha_values):
        raise ValueError("--alpha-values must all be in [0, 1]")


def assign_tasks(
    tau_values: list[float],
    antenna_counts: list[int],
    alpha_values: list[float],
    lookahead_widths: list[int],
    world_size: int,
) -> list[list[tuple[float, int, float, int]]]:
    """Balance all parameter combinations using k and lookahead as cost estimates."""
    tasks = [
        (tau, antenna_count, alpha, width)
        for tau in tau_values
        for antenna_count in antenna_counts
        for width in lookahead_widths
        for alpha in alpha_values
    ]
    assignments: list[list[tuple[float, int, float, int]]] = [
        [] for _ in range(world_size)
    ]
    estimated_costs = [0] * world_size
    for task in sorted(tasks, key=lambda item: item[1] * item[3], reverse=True):
        target_rank = min(range(world_size), key=estimated_costs.__getitem__)
        assignments[target_rank].append(task)
        estimated_costs[target_rank] += task[1] * task[3]
    return assignments


def filename_number(value: float) -> str:
    """Format a number as a filesystem-friendly token, such as 0p75."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def write_solution_txt(
    scorer,
    candidate_points: gpd.GeoDataFrame,
    official_building_ids: np.ndarray,
    output_dir: Path,
    tau: float,
    antenna_count: int,
    alpha: float,
    lookahead_width: int,
) -> Path:
    selected_point_ids = [int(value) for value in scorer.selected_point_ids]
    if len(selected_point_ids) != antenna_count:
        raise RuntimeError(
            f"Expected k={antenna_count} selected points, found "
            f"{len(selected_point_ids)}"
        )
    selected_id_set = set(selected_point_ids)
    selected_rows = candidate_points[
        candidate_points["point_index"].isin(selected_id_set)
    ].set_index("point_index")

    missing_ids = [
        point_id for point_id in selected_point_ids
        if point_id not in selected_rows.index
    ]
    if missing_ids:
        raise ValueError(
            f"Candidate GeoPackage is missing {len(missing_ids)} selected points."
        )

    coordinates = [
        (float(selected_rows.at[point_id, "x"]), float(selected_rows.at[point_id, "y"]))
        for point_id in selected_point_ids
    ]

    current_segments = scorer.current_visible_segments()
    visible_building_ids = np.unique(
        np.asarray(current_segments["building_ids"], dtype=np.int64)
    )
    scored_internal_building_ids = [
        int(building_id)
        for building_id in visible_building_ids
        if scorer.building_coverage(int(building_id)) >= tau
    ]

    score = int(scorer.current_score)
    if len(scored_internal_building_ids) != score:
        raise RuntimeError(
            f"Scorer reported score={score}, but found "
            f"{len(scored_internal_building_ids)} qualifying buildings."
        )
    scored_building_ids = map_internal_building_ids(
        scored_internal_building_ids,
        official_building_ids,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"lookahead_{lookahead_width}_"
        f"alpha_{filename_number(alpha)}_"
        f"score_{score}.txt"
    )
    write_solution_block(
        output_path,
        tau,
        coordinates,
        scored_building_ids,
    )
    return output_path


def main() -> None:
    args = parse_args()
    validate_args(args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    if rank == 0:
        for tau in args.tau:
            for antenna_count in args.antenna_count:
                (args.output_root / (
                    f"tau_{filename_number(tau)}_k_{antenna_count}"
                )).mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    assignments = assign_tasks(
        args.tau,
        args.antenna_count,
        args.alpha_values,
        args.lookahead_widths,
        world_size,
    )
    local_tasks = assignments[rank]

    if rank == 0:
        official_building_ids = load_official_building_ids(
            args.geojson,
            args.building_id_column,
        )
    else:
        official_building_ids = None
    official_building_ids = np.ascontiguousarray(
        comm.bcast(official_building_ids, root=0),
        dtype=np.int64,
    )

    if local_tasks:
        candidate_points = gpd.read_file(
            args.candidate_points_gpkg,
            layer="candidate_points",
        )[["point_index", "x", "y"]]

        for tau, antenna_count, alpha, lookahead_width in local_tasks:
            task_start = time.perf_counter()
            scorer = load_incremental_scorer(
                args.segments_csv,
                tau,
                args.chunksize,
            )
            rows = run_two_step_greedy(
                scorer,
                alpha=alpha,
                antenna_count=antenna_count,
                lookahead_width=lookahead_width,
                print_every=args.print_every,
            )
            del rows
            output_dir = args.output_root / (
                f"tau_{filename_number(tau)}_k_{antenna_count}"
            )
            output_path = write_solution_txt(
                scorer,
                candidate_points,
                official_building_ids,
                output_dir,
                tau,
                antenna_count,
                alpha,
                lookahead_width,
            )
            task_seconds = time.perf_counter() - task_start
            summary = {
                "rank": rank,
                "tau": tau,
                "antenna_number": antenna_count,
                "alpha": alpha,
                "lookahead_width": lookahead_width,
                "score": int(scorer.current_score),
                "selected_points": int(scorer.selected_count),
                "task_seconds": task_seconds,
            }
            print(
                f"final: rank={summary['rank']} "
                f"tau={summary['tau']} "
                f"antenna_number={summary['antenna_number']} "
                f"alpha={summary['alpha']} "
                f"lookahead_width={summary['lookahead_width']} "
                f"score={summary['score']} "
                f"selected_points={summary['selected_points']} "
                f"task_time={summary['task_seconds']:.3f}s "
                f"txt={output_path}",
                flush=True,
            )
            print(flush=True)
    else:
        print(
            f"rank={rank}/{world_size} has no assigned parameter runs",
            flush=True,
        )

    comm.Barrier()


if __name__ == "__main__":
    main()
