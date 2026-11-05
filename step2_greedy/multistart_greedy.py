#!/usr/bin/env python3
"""MPI multi-start greedy antenna selection.

For each tau, every scoreable point is evaluated as a one-antenna solution.
Starts are then selected from three deterministic sources:

* the best ``grid_top_k`` points in each fixed CRS-origin grid cell;
* the global top fraction;
* a reproducible uniform random sample from the remaining points.

Every selected start is forced into a solution and ordinary greedy selection is
used to add the remaining ``k - 1`` antennas.  Rank 0 owns all files and uses
the other MPI ranks as dynamically scheduled workers.  The output journal and
best solution are durable while the job is running, and ``--resume`` can
continue an interrupted output directory.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

import coverage_scorer_cpp
from common.mpi_compat import MPI

from common.scorer_loader import load_incremental_scorer
from common.solution_txt import (
    load_official_building_ids,
    map_internal_building_ids,
    write_solution_block,
)


SCHEMA_VERSION = "1"
TASK_TAG = 101
STOP_TAG = 102
RESULT_TAG = 103
ERROR_TAG = 104

FIRST_SCORE_FIELDS = (
    "tau",
    "point_id",
    "x",
    "y",
    "cell_x",
    "cell_y",
    "first_hard",
    "first_soft_raw",
    "first_unique_length",
    "first_combined",
    "global_rank",
    "rank_in_cell",
    "selected_start",
    "source",
)

FIRST_POINT_FIELDS = (
    "case_id",
    "task_index",
    "tau",
    "start_point_id",
    "source",
    "x",
    "y",
    "cell_x",
    "cell_y",
    "global_rank",
    "rank_in_cell",
    "first_hard",
    "first_soft_raw",
    "first_unique_length",
    "first_combined",
)

RESULT_FIELDS = (
    "case_id",
    "task_index",
    "tau",
    "antenna_count",
    "alpha",
    "start_point_id",
    "source",
    "cell_x",
    "cell_y",
    "global_rank",
    "rank_in_cell",
    "hard_score",
    "soft_score",
    "selected_point_ids",
    "elapsed_seconds",
    "worker_rank",
)

HISTORY_FIELDS = (
    "timestamp_utc",
    "completed_results",
    "case_id",
    "hard_score",
    "soft_score",
    "start_point_id",
    "selected_point_ids",
    "best_solution_path",
)


@dataclass(frozen=True)
class CandidateCoordinates:
    point_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    by_id: dict[int, tuple[float, float]]
    crs: str


@dataclass(frozen=True)
class SolutionDetails:
    coordinates: np.ndarray
    building_ids: tuple[int, ...]
    hard_score: int
    soft_score: float


@dataclass(frozen=True)
class TopSolution:
    top_rank: int
    result: dict[str, Any]
    details: SolutionDetails
    path: Path
    quality: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tau_token(tau: float) -> str:
    # repr(float) is the shortest round-trip spelling, so distinct float tau
    # values cannot collide through the six-significant-digit default of :g.
    return (
        repr(float(tau))
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def float_text(value: float) -> str:
    return f"{float(value):.17g}"


def point_ids_text(point_ids: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in point_ids)


def parse_point_ids(text: str) -> tuple[int, ...]:
    if not text:
        return ()
    return tuple(int(value) for value in text.split(","))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_tsv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fields),
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_tsv_row(
    stream,
    writer: csv.DictWriter,
    row: dict[str, Any],
    *,
    do_fsync: bool,
) -> None:
    writer.writerow(row)
    stream.flush()
    if do_fsync:
        os.fsync(stream.fileno())


def input_signature(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": str(stat.st_size),
        "mtime_ns": str(stat.st_mtime_ns),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_config(args: argparse.Namespace) -> dict[str, str]:
    config: dict[str, str] = {
        "schema_version": SCHEMA_VERSION,
        "program_sha256": sha256_file(Path(__file__).resolve()),
        "coverage_scorer_cpp_sha256": sha256_file(
            Path(coverage_scorer_cpp.__file__).resolve()
        ),
        "scorer_loader_sha256": sha256_file(
            Path(load_incremental_scorer.__code__.co_filename).resolve()
        ),
        "solution_txt_sha256": sha256_file(
            Path(write_solution_block.__code__.co_filename).resolve()
        ),
        "antenna_count": str(args.antenna_count),
        "tau_values": ":".join(float_text(value) for value in args.tau),
        "alpha": float_text(args.alpha),
        "grid_size_meters": float_text(args.grid_size_meters),
        "grid_top_k": str(args.grid_top_k),
        "global_top_fraction": float_text(args.global_top_fraction),
        "random_count": str(args.random_count),
        "random_seed": str(args.random_seed),
        "keep_top": str(args.keep_top),
        "point_swap_input_count": str(args.point_swap_input_count),
        "elite_count": str(args.elite_count),
        "chunksize": str(args.chunksize),
        "candidate_layer": args.candidate_layer,
        "building_id_column": args.building_id_column,
    }
    for prefix, path in (
        ("segments_csv", args.segments_csv),
        ("candidate_points_gpkg", args.candidate_points_gpkg),
        ("geojson", args.geojson),
    ):
        for key, value in input_signature(path).items():
            config[f"{prefix}_{key}"] = value
    return config


def read_two_column_tsv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or set(rows[0]) != {"key", "value"}:
        raise ValueError(f"Invalid run configuration file: {path}")
    output: dict[str, str] = {}
    for row in rows:
        key = row["key"]
        if key in output:
            raise ValueError(f"Duplicate configuration key {key!r} in {path}")
        output[key] = row["value"]
    return output


def prepare_output_directory(
    output_dir: Path,
    config: dict[str, str],
    *,
    resume: bool,
    mpi_size: int,
) -> Any:
    if resume and not output_dir.is_dir():
        raise ValueError(f"--resume output directory does not exist: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_output_lock(output_dir)
    try:
        config_path = output_dir / "run_config.tsv"
        if resume:
            if not config_path.is_file():
                raise ValueError(
                    f"--resume requires an existing {config_path}"
                )
            existing = read_two_column_tsv(config_path)
            if existing != config:
                changed = sorted(
                    key
                    for key in set(existing).union(config)
                    if existing.get(key) != config.get(key)
                )
                raise ValueError(
                    "Resume configuration does not match; changed keys: "
                    + ", ".join(changed)
                )
        else:
            existing_entries = [
                path for path in output_dir.iterdir() if path.name != ".run.lock"
            ]
            if existing_entries:
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. "
                    "Use a new directory or pass --resume with identical parameters."
                )
            atomic_write_tsv(
                config_path,
                ("key", "value"),
                ({"key": key, "value": value} for key, value in sorted(config.items())),
            )

        invocation_path = output_dir / "invocations.tsv"
        invocation_exists = invocation_path.exists()
        with invocation_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("timestamp_utc", "mpi_ranks", "resume", "pid"),
                delimiter="\t",
                lineterminator="\n",
            )
            if not invocation_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": utc_now(),
                    "mpi_ranks": mpi_size,
                    "resume": int(resume),
                    "pid": os.getpid(),
                }
            )
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        run_lock.close()
        raise
    return run_lock


def acquire_output_lock(output_dir: Path):
    """Hold a nonblocking advisory lock for this output directory."""
    lock_path = output_dir / ".run.lock"
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(
            f"Another process is already writing output directory {output_dir}"
        ) from exc
    stream.seek(0)
    stream.truncate()
    stream.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": os.uname().nodename,
                "acquired_utc": utc_now(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def load_candidate_coordinates(path: Path, layer: str) -> CandidateCoordinates:
    import geopandas as gpd
    import pandas as pd

    frame = gpd.read_file(path, layer=layer)
    if "point_index" not in frame.columns:
        raise ValueError(f"{path}:{layer} has no point_index column")
    if frame.empty:
        raise ValueError(f"{path}:{layer} contains no candidate points")
    if frame["point_index"].isna().any():
        raise ValueError(f"{path}:{layer} contains a null point_index")
    if frame["point_index"].duplicated().any():
        duplicates = frame.loc[
            frame["point_index"].duplicated(), "point_index"
        ].head(10)
        raise ValueError(
            "Candidate point IDs are not unique: "
            + ",".join(str(value) for value in duplicates)
        )

    try:
        numeric_point_ids = pd.to_numeric(frame["point_index"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{layer} contains a non-numeric point_index") from exc
    if pd.api.types.is_integer_dtype(numeric_point_ids.dtype):
        exact_ids = numeric_point_ids.to_numpy()
        if (
            np.min(exact_ids) < np.iinfo(np.int64).min
            or np.max(exact_ids) > np.iinfo(np.int64).max
        ):
            raise ValueError(f"{path}:{layer} has point_index outside int64 range")
        point_ids = exact_ids.astype(np.int64)
    else:
        point_id_numbers = numeric_point_ids.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(point_id_numbers)):
            raise ValueError(f"{path}:{layer} contains a non-finite point_index")
        point_ids = point_id_numbers.astype(np.int64)
        if not np.array_equal(point_id_numbers, point_ids.astype(np.float64)):
            raise ValueError(f"{path}:{layer} contains a non-integer point_index")
    if len(np.unique(point_ids)) != len(point_ids):
        raise ValueError(f"{path}:{layer} contains duplicate numeric point IDs")
    if "x" in frame.columns and "y" in frame.columns:
        x = frame["x"].to_numpy(dtype=np.float64)
        y = frame["y"].to_numpy(dtype=np.float64)
    else:
        if frame.geometry.isna().any():
            raise ValueError(f"{path}:{layer} contains null point geometry")
        x = frame.geometry.x.to_numpy(dtype=np.float64)
        y = frame.geometry.y.to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Candidate coordinates contain non-finite values")
    if frame.crs is None:
        raise ValueError("Candidate points have no CRS; 100 m cells are ambiguous")
    if not bool(frame.crs.is_projected):
        raise ValueError(
            f"Candidate CRS {frame.crs} is not projected; grid size is in metres"
        )
    axis_info = frame.crs.axis_info
    if axis_info:
        unit_names = {str(axis.unit_name).lower() for axis in axis_info[:2]}
        metre_names = {"metre", "meter", "metres", "meters"}
        if not unit_names.issubset(metre_names):
            raise ValueError(
                f"Candidate CRS {frame.crs} axes are not measured in metres: "
                + ", ".join(sorted(unit_names))
            )

    order = np.argsort(point_ids, kind="stable")
    point_ids = np.ascontiguousarray(point_ids[order], dtype=np.int64)
    x = np.ascontiguousarray(x[order], dtype=np.float64)
    y = np.ascontiguousarray(y[order], dtype=np.float64)
    by_id = {
        int(point_id): (float(px), float(py))
        for point_id, px, py in zip(point_ids, x, y)
    }
    return CandidateCoordinates(point_ids, x, y, by_id, str(frame.crs))


def align_coordinates(
    coordinates: CandidateCoordinates,
    scorer_point_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scorer_ids = np.asarray(scorer_point_ids, dtype=np.int64)
    if scorer_ids.ndim != 1 or scorer_ids.size == 0:
        raise ValueError("The scorer has no point IDs")
    if np.any(scorer_ids[1:] <= scorer_ids[:-1]):
        raise ValueError("scorer.point_ids must be strictly increasing")
    positions = np.searchsorted(coordinates.point_ids, scorer_ids)
    found = positions < len(coordinates.point_ids)
    if np.any(found):
        found[found] &= coordinates.point_ids[positions[found]] == scorer_ids[found]
    if not np.all(found):
        missing = scorer_ids[~found][:10]
        raise ValueError(
            "Scorer point IDs are missing from the candidate GeoPackage: "
            + ",".join(str(int(value)) for value in missing)
        )
    return coordinates.x[positions], coordinates.y[positions]


def score_candidates(scorer: Any, candidate_ids: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    candidate_ids = np.ascontiguousarray(candidate_ids, dtype=np.int64)
    if candidate_ids.size == 0:
        raise ValueError("candidate_ids is empty")
    batch = scorer.evaluate_all(candidate_ids)
    point_ids = np.asarray(batch["point_indices"], dtype=np.int64)
    hard = np.asarray(batch["hard_scores"], dtype=np.float64)
    soft = np.asarray(batch["soft_scores"], dtype=np.float64)
    unique = np.asarray(batch["unique_lengths"], dtype=np.float64)
    expected_shape = candidate_ids.shape
    for name, values in (
        ("point_indices", point_ids),
        ("hard_scores", hard),
        ("soft_scores", soft),
        ("unique_lengths", unique),
    ):
        if values.shape != expected_shape:
            raise RuntimeError(
                f"evaluate_all returned {name} shape {values.shape}, expected {expected_shape}"
            )
    if not np.array_equal(point_ids, candidate_ids):
        raise RuntimeError("evaluate_all changed the candidate point order")
    if not all(np.all(np.isfinite(values)) for values in (hard, soft, unique)):
        raise RuntimeError("evaluate_all returned a non-finite score")

    max_hard = float(np.max(hard))
    max_soft = float(np.max(soft))
    hard_norm = hard / max_hard if max_hard > 0.0 else np.zeros_like(hard)
    soft_norm = soft / max_soft if max_soft > 0.0 else np.zeros_like(soft)
    combined = alpha * hard_norm + (1.0 - alpha) * soft_norm
    return {
        "point_ids": point_ids,
        "hard": hard,
        "soft": soft,
        "unique": unique,
        "combined": combined,
    }


def deterministic_order(scores: dict[str, np.ndarray]) -> np.ndarray:
    """combined, hard, soft, unique descending; point ID ascending."""
    return np.lexsort(
        (
            scores["point_ids"],
            -scores["unique"],
            -scores["soft"],
            -scores["hard"],
            -scores["combined"],
        )
    )


def best_position(scores: dict[str, np.ndarray]) -> int:
    candidates = np.arange(len(scores["point_ids"]), dtype=np.int64)
    for field in ("combined", "hard", "soft", "unique"):
        values = scores[field]
        maximum = np.max(values[candidates])
        candidates = candidates[values[candidates] == maximum]
        if len(candidates) == 1:
            return int(candidates[0])
    point_ids = scores["point_ids"][candidates]
    return int(candidates[int(np.argmin(point_ids))])


def select_start_tasks(
    *,
    tau: float,
    scorer: Any,
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    grid_size_meters: float,
    grid_top_k: int,
    global_top_fraction: float,
    random_count: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    scorer.reset()
    point_ids = np.asarray(scorer.point_ids, dtype=np.int64)
    scores = score_candidates(scorer, point_ids, alpha)
    order = deterministic_order(scores)
    count = len(point_ids)

    global_rank = np.empty(count, dtype=np.int64)
    global_rank[order] = np.arange(1, count + 1, dtype=np.int64)
    cell_x = np.floor(x / grid_size_meters).astype(np.int64)
    cell_y = np.floor(y / grid_size_meters).astype(np.int64)
    rank_in_cell = np.empty(count, dtype=np.int64)
    cell_counts: dict[tuple[int, int], int] = {}
    grid_selected = np.zeros(count, dtype=bool)
    for position in order:
        index = int(position)
        key = (int(cell_x[index]), int(cell_y[index]))
        cell_rank = cell_counts.get(key, 0) + 1
        cell_counts[key] = cell_rank
        rank_in_cell[index] = cell_rank
        if cell_rank <= grid_top_k:
            grid_selected[index] = True

    global_selected = np.zeros(count, dtype=bool)
    global_count = int(math.floor(global_top_fraction * count))
    if global_top_fraction > 0.0:
        global_count = max(1, global_count)
    global_count = min(count, global_count)
    if global_count:
        global_selected[order[:global_count]] = True

    selected = grid_selected | global_selected
    random_selected = np.zeros(count, dtype=bool)
    remaining = np.flatnonzero(~selected)
    actual_random_count = min(random_count, len(remaining))
    if actual_random_count:
        seed_sequence = np.random.SeedSequence(
            [int(random_seed), int(round(float(tau) * 1_000_000_000))]
        )
        rng = np.random.default_rng(seed_sequence)
        chosen = rng.choice(remaining, size=actual_random_count, replace=False)
        random_selected[chosen] = True
        selected[chosen] = True

    source = np.full(count, "", dtype=object)
    for index in np.flatnonzero(selected):
        parts: list[str] = []
        if grid_selected[index]:
            parts.append("grid")
        if global_selected[index]:
            parts.append("global")
        if random_selected[index]:
            parts.append("random")
        source[index] = "+".join(parts)

    selected_positions = [int(index) for index in order if selected[index]]
    tasks: list[dict[str, Any]] = []
    for task_index, index in enumerate(selected_positions, start=1):
        tasks.append(
            {
                "case_id": f"case_{task_index:06d}",
                "task_index": task_index,
                "tau": float(tau),
                "start_point_id": int(point_ids[index]),
                "source": str(source[index]),
                "x": float(x[index]),
                "y": float(y[index]),
                "cell_x": int(cell_x[index]),
                "cell_y": int(cell_y[index]),
                "global_rank": int(global_rank[index]),
                "rank_in_cell": int(rank_in_cell[index]),
                "first_hard": int(scores["hard"][index]),
                "first_soft_raw": float(scores["soft"][index]),
                "first_unique_length": float(scores["unique"][index]),
                "first_combined": float(scores["combined"][index]),
            }
        )

    metadata = {
        **scores,
        "x": x,
        "y": y,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "global_rank": global_rank,
        "rank_in_cell": rank_in_cell,
        "selected": selected,
        "source": source,
        "order": order,
        "occupied_cell_count": np.asarray([len(cell_counts)], dtype=np.int64),
        "grid_selected_count": np.asarray(
            [int(np.count_nonzero(grid_selected))], dtype=np.int64
        ),
        "global_selected_count": np.asarray([global_count], dtype=np.int64),
        "actual_random_count": np.asarray([actual_random_count], dtype=np.int64),
    }
    return tasks, metadata


def first_score_rows(tau: float, metadata: dict[str, np.ndarray]) -> Iterator[dict[str, Any]]:
    for index in range(len(metadata["point_ids"])):
        yield {
            "tau": float_text(tau),
            "point_id": int(metadata["point_ids"][index]),
            "x": float_text(metadata["x"][index]),
            "y": float_text(metadata["y"][index]),
            "cell_x": int(metadata["cell_x"][index]),
            "cell_y": int(metadata["cell_y"][index]),
            "first_hard": int(metadata["hard"][index]),
            "first_soft_raw": float_text(metadata["soft"][index]),
            "first_unique_length": float_text(metadata["unique"][index]),
            "first_combined": float_text(metadata["combined"][index]),
            "global_rank": int(metadata["global_rank"][index]),
            "rank_in_cell": int(metadata["rank_in_cell"][index]),
            "selected_start": int(bool(metadata["selected"][index])),
            "source": str(metadata["source"][index]),
        }


def task_tsv_row(task: dict[str, Any]) -> dict[str, Any]:
    row = dict(task)
    for key in (
        "tau",
        "x",
        "y",
        "first_soft_raw",
        "first_unique_length",
        "first_combined",
    ):
        row[key] = float_text(row[key])
    return row


def parse_task_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "task_index": int(row["task_index"]),
        "tau": float(row["tau"]),
        "start_point_id": int(row["start_point_id"]),
        "source": row["source"],
        "x": float(row["x"]),
        "y": float(row["y"]),
        "cell_x": int(row["cell_x"]),
        "cell_y": int(row["cell_y"]),
        "global_rank": int(row["global_rank"]),
        "rank_in_cell": int(row["rank_in_cell"]),
        "first_hard": int(row["first_hard"]),
        "first_soft_raw": float(row["first_soft_raw"]),
        "first_unique_length": float(row["first_unique_length"]),
        "first_combined": float(row["first_combined"]),
    }


def load_tasks(path: Path, tau: float) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIRST_POINT_FIELDS:
            raise ValueError(f"Unexpected first-point schema in {path}")
        tasks = [parse_task_row(row) for row in reader]
    if not tasks:
        raise ValueError(f"No start tasks found in {path}")
    expected_ids = [f"case_{index:06d}" for index in range(1, len(tasks) + 1)]
    if [task["case_id"] for task in tasks] != expected_ids:
        raise ValueError(f"Case IDs are not contiguous in {path}")
    if any(not math.isclose(task["tau"], tau) for task in tasks):
        raise ValueError(f"Tau mismatch in {path}")
    return tasks


def run_greedy_task(
    scorer: Any,
    task: dict[str, Any],
    *,
    antenna_count: int,
    alpha: float,
    worker_rank: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    scorer.reset()
    all_point_ids = np.asarray(scorer.point_ids, dtype=np.int64)
    if antenna_count > len(all_point_ids):
        raise ValueError(
            f"antenna_count={antenna_count} exceeds {len(all_point_ids)} scoreable points"
        )
    available = np.ones(len(all_point_ids), dtype=bool)
    start_point_id = int(task["start_point_id"])
    start_position = int(np.searchsorted(all_point_ids, start_point_id))
    if start_position >= len(all_point_ids) or int(all_point_ids[start_position]) != start_point_id:
        raise ValueError(f"Start point {start_point_id} is absent from scorer.point_ids")

    initial_scores = score_candidates(
        scorer,
        np.asarray([start_point_id], dtype=np.int64),
        alpha,
    )
    raw_soft = float(initial_scores["soft"][0])
    scorer.add_point(start_point_id)
    available[start_position] = False
    selected_order = [start_point_id]

    for _ in range(1, antenna_count):
        candidate_ids = np.ascontiguousarray(all_point_ids[available], dtype=np.int64)
        scores = score_candidates(scorer, candidate_ids, alpha)
        position = best_position(scores)
        chosen = int(scores["point_ids"][position])
        raw_soft += float(scores["soft"][position])
        scorer.add_point(chosen)
        chosen_position = int(np.searchsorted(all_point_ids, chosen))
        if chosen_position >= len(all_point_ids) or not available[chosen_position]:
            raise RuntimeError(f"Greedy selected unavailable point {chosen}")
        available[chosen_position] = False
        selected_order.append(chosen)

    if int(scorer.selected_count) != antenna_count:
        raise RuntimeError("Scorer selected_count does not equal antenna_count")
    if start_point_id not in selected_order:
        raise RuntimeError("Forced start disappeared from the greedy solution")
    selected_ids = tuple(sorted(selected_order))
    if len(selected_ids) != antenna_count or len(set(selected_ids)) != antenna_count:
        raise RuntimeError("Greedy solution has duplicate or missing point IDs")
    return {
        "case_id": task["case_id"],
        "task_index": int(task["task_index"]),
        "tau": float(task["tau"]),
        "antenna_count": antenna_count,
        "alpha": float(alpha),
        "start_point_id": start_point_id,
        "source": task["source"],
        "cell_x": int(task["cell_x"]),
        "cell_y": int(task["cell_y"]),
        "global_rank": int(task["global_rank"]),
        "rank_in_cell": int(task["rank_in_cell"]),
        "hard_score": int(scorer.current_score),
        "soft_score": raw_soft / float(scorer.tau),
        "selected_point_ids": selected_ids,
        "elapsed_seconds": time.perf_counter() - started,
        "worker_rank": worker_rank,
    }


def result_tsv_row(result: dict[str, Any]) -> dict[str, Any]:
    row = dict(result)
    row["tau"] = float_text(row["tau"])
    row["alpha"] = float_text(row["alpha"])
    row["soft_score"] = float_text(row["soft_score"])
    row["elapsed_seconds"] = f"{float(row['elapsed_seconds']):.6f}"
    row["selected_point_ids"] = point_ids_text(row["selected_point_ids"])
    return row


def parse_result_row(row: dict[str, str], path: Path, line_number: int) -> dict[str, Any]:
    try:
        result = {
            "case_id": row["case_id"],
            "task_index": int(row["task_index"]),
            "tau": float(row["tau"]),
            "antenna_count": int(row["antenna_count"]),
            "alpha": float(row["alpha"]),
            "start_point_id": int(row["start_point_id"]),
            "source": row["source"],
            "cell_x": int(row["cell_x"]),
            "cell_y": int(row["cell_y"]),
            "global_rank": int(row["global_rank"]),
            "rank_in_cell": int(row["rank_in_cell"]),
            "hard_score": int(row["hard_score"]),
            "soft_score": float(row["soft_score"]),
            "selected_point_ids": parse_point_ids(row["selected_point_ids"]),
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "worker_rank": int(row["worker_rank"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid result at {path}:{line_number}: {exc}") from exc
    if len(result["selected_point_ids"]) != result["antenna_count"]:
        raise ValueError(f"Wrong point count at {path}:{line_number}")
    if tuple(sorted(set(result["selected_point_ids"]))) != result["selected_point_ids"]:
        raise ValueError(f"Non-canonical point IDs at {path}:{line_number}")
    if result["start_point_id"] not in result["selected_point_ids"]:
        raise ValueError(f"Forced start is absent at {path}:{line_number}")
    if result["hard_score"] < 0:
        raise ValueError(f"Negative hard score at {path}:{line_number}")
    if not math.isfinite(result["soft_score"]):
        raise ValueError(f"Non-finite soft score at {path}:{line_number}")
    if not math.isfinite(result["elapsed_seconds"]) or result["elapsed_seconds"] < 0.0:
        raise ValueError(f"Invalid elapsed time at {path}:{line_number}")
    return result


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise ValueError(f"Unexpected result schema in {path}")
        rows = list(reader)

    results: list[dict[str, Any]] = []
    for offset, row in enumerate(rows, start=2):
        try:
            results.append(parse_result_row(row, path, offset))
        except ValueError:
            if offset != len(rows) + 1:
                raise
            print(
                f"WARNING: ignoring incomplete final journal row in {path}",
                file=sys.stderr,
                flush=True,
            )
    by_case: dict[str, dict[str, Any]] = {}
    for result in results:
        prior = by_case.get(result["case_id"])
        if prior is not None and result_tsv_row(prior) != result_tsv_row(result):
            raise ValueError(f"Conflicting duplicate result for {result['case_id']} in {path}")
        by_case[result["case_id"]] = result
    return sorted(by_case.values(), key=lambda value: int(value["task_index"]))


def result_is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    if int(candidate["hard_score"]) != int(incumbent["hard_score"]):
        return int(candidate["hard_score"]) > int(incumbent["hard_score"])
    if float(candidate["soft_score"]) != float(incumbent["soft_score"]):
        return float(candidate["soft_score"]) > float(incumbent["soft_score"])
    candidate_ids = tuple(int(value) for value in candidate["selected_point_ids"])
    incumbent_ids = tuple(int(value) for value in incumbent["selected_point_ids"])
    if candidate_ids != incumbent_ids:
        return candidate_ids < incumbent_ids
    return int(candidate["task_index"]) < int(incumbent["task_index"])


def result_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(result["hard_score"]),
        -float(result["soft_score"]),
        tuple(int(value) for value in result["selected_point_ids"]),
        int(result["task_index"]),
    )


def rebuild_solution(
    scorer: Any,
    selected_point_ids: Sequence[int],
    coordinate_by_id: dict[int, tuple[float, float]],
    official_building_ids: np.ndarray,
) -> SolutionDetails:
    canonical = tuple(sorted(int(value) for value in selected_point_ids))
    scorer.reset()
    raw_soft = 0.0
    for point_id in canonical:
        raw_soft += float(scorer.soft_score(point_id))
        scorer.add_point(point_id)
    visible = scorer.current_visible_segments()
    internal_ids = np.unique(np.asarray(visible["building_ids"], dtype=np.int64))
    qualifying_internal = [
        int(building_id)
        for building_id in internal_ids
        if scorer.building_coverage(int(building_id)) >= float(scorer.tau)
    ]
    hard_score = int(scorer.current_score)
    if len(qualifying_internal) != hard_score:
        raise RuntimeError(
            "Scorer current_score disagrees with qualifying building count"
        )
    building_ids = tuple(
        map_internal_building_ids(qualifying_internal, official_building_ids)
    )
    try:
        coordinates = np.asarray(
            [coordinate_by_id[point_id] for point_id in canonical],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise ValueError(f"No coordinate for selected point ID {exc.args[0]}") from exc
    return SolutionDetails(
        coordinates=coordinates,
        building_ids=building_ids,
        hard_score=hard_score,
        soft_score=raw_soft / float(scorer.tau),
    )


class ResultCollector:
    def __init__(
        self,
        *,
        setting_dir: Path,
        tau: float,
        antenna_count: int,
        scorer: Any,
        coordinate_by_id: dict[int, tuple[float, float]],
        official_building_ids: np.ndarray,
        expected_results: int,
        checkpoint_every: int,
        print_every: int,
        existing_results: list[dict[str, Any]],
    ) -> None:
        self.setting_dir = setting_dir
        self.tau = tau
        self.antenna_count = antenna_count
        self.scorer = scorer
        self.coordinate_by_id = coordinate_by_id
        self.official_building_ids = official_building_ids
        self.expected_results = expected_results
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.results = list(existing_results)
        self.initial_completed = len(self.results)
        self.completed_case_ids = {result["case_id"] for result in self.results}
        self.best = min(self.results, key=result_sort_key) if self.results else None
        self.started = time.perf_counter()

        self.result_path = setting_dir / "all_results.tsv"
        atomic_write_tsv(
            self.result_path,
            RESULT_FIELDS,
            (result_tsv_row(result) for result in self.results),
        )
        self.result_stream = self.result_path.open(
            "a", encoding="utf-8", newline=""
        )
        self.result_writer = csv.DictWriter(
            self.result_stream,
            fieldnames=list(RESULT_FIELDS),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        self.history_path = setting_dir / "best_history.tsv"
        history_exists = self.history_path.exists() and self.history_path.stat().st_size > 0
        self.history_stream = self.history_path.open(
            "a", encoding="utf-8", newline=""
        )
        self.history_writer = csv.DictWriter(
            self.history_stream,
            fieldnames=list(HISTORY_FIELDS),
            delimiter="\t",
            lineterminator="\n",
        )
        if not history_exists:
            self.history_writer.writeheader()
            self.history_stream.flush()
            os.fsync(self.history_stream.fileno())

        if self.best is not None:
            self._write_best(self.best, append_history=False)

    def close(self) -> None:
        if not self.result_stream.closed:
            self.result_stream.flush()
            os.fsync(self.result_stream.fileno())
            self.result_stream.close()
        if not self.history_stream.closed:
            self.history_stream.flush()
            os.fsync(self.history_stream.fileno())
            self.history_stream.close()

    def _write_best(self, result: dict[str, Any], *, append_history: bool) -> None:
        details = rebuild_solution(
            self.scorer,
            result["selected_point_ids"],
            self.coordinate_by_id,
            self.official_building_ids,
        )
        if details.hard_score != int(result["hard_score"]):
            raise RuntimeError(
                f"Rebuilt hard score {details.hard_score} differs from result "
                f"{result['hard_score']} for {result['case_id']}"
            )
        if not math.isclose(
            details.soft_score,
            float(result["soft_score"]),
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                f"Rebuilt soft score {details.soft_score} differs from result "
                f"{result['soft_score']} for {result['case_id']}"
            )
        solution_path = self.setting_dir / "best_solution.txt"
        write_solution_block(
            solution_path,
            self.tau,
            details.coordinates,
            details.building_ids,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "updated_utc": utc_now(),
            "tau": self.tau,
            "antenna_count": self.antenna_count,
            "completed_results": len(self.results),
            "expected_results": self.expected_results,
            "case_id": result["case_id"],
            "task_index": int(result["task_index"]),
            "start_point_id": int(result["start_point_id"]),
            "source": result["source"],
            "hard_score": int(result["hard_score"]),
            "soft_score": float(result["soft_score"]),
            "selected_point_ids": [int(value) for value in result["selected_point_ids"]],
            "building_count": len(details.building_ids),
            "solution_path": str(solution_path.resolve()),
        }
        atomic_write_json(self.setting_dir / "best.json", payload)
        if append_history:
            append_tsv_row(
                self.history_stream,
                self.history_writer,
                {
                    "timestamp_utc": payload["updated_utc"],
                    "completed_results": len(self.results),
                    "case_id": result["case_id"],
                    "hard_score": int(result["hard_score"]),
                    "soft_score": float_text(result["soft_score"]),
                    "start_point_id": int(result["start_point_id"]),
                    "selected_point_ids": point_ids_text(result["selected_point_ids"]),
                    "best_solution_path": str(solution_path.resolve()),
                },
                do_fsync=True,
            )

    def add(self, result: dict[str, Any]) -> None:
        case_id = result["case_id"]
        if case_id in self.completed_case_ids:
            raise RuntimeError(f"Duplicate completed case received: {case_id}")
        if (
            int(result["antenna_count"]) != self.antenna_count
            or not math.isclose(float(result["tau"]), self.tau)
            or len(result["selected_point_ids"]) != self.antenna_count
            or int(result["start_point_id"]) not in result["selected_point_ids"]
        ):
            raise RuntimeError(f"Invalid result payload for {case_id}")
        self.results.append(result)
        self.completed_case_ids.add(case_id)
        completed = len(self.results)
        append_tsv_row(
            self.result_stream,
            self.result_writer,
            result_tsv_row(result),
            do_fsync=(completed % self.checkpoint_every == 0),
        )
        if result_is_better(result, self.best):
            self.best = result
            self._write_best(result, append_history=True)
            print(
                f"[{utc_now()}] tau={self.tau:g} NEW_BEST "
                f"hard={result['hard_score']} soft={result['soft_score']:.9f} "
                f"case={case_id} completed={completed}/{self.expected_results}",
                flush=True,
            )
        if completed % self.print_every == 0 or completed == self.expected_results:
            elapsed = time.perf_counter() - self.started
            new_completed = completed - self.initial_completed
            rate = new_completed / elapsed if elapsed > 0.0 else 0.0
            print(
                f"[{utc_now()}] tau={self.tau:g} progress "
                f"{completed}/{self.expected_results}, {rate:.3f} new results/s",
                flush=True,
            )

    def refresh_best_metadata(self) -> None:
        if self.best is not None:
            self._write_best(self.best, append_history=False)


def execute_tasks(
    *,
    comm: Any,
    rank: int,
    size: int,
    scorer: Any,
    pending_tasks: list[dict[str, Any]] | None,
    antenna_count: int,
    alpha: float,
    collector: ResultCollector | None,
) -> None:
    if size == 1:
        assert pending_tasks is not None and collector is not None
        for task in pending_tasks:
            collector.add(
                run_greedy_task(
                    scorer,
                    task,
                    antenna_count=antenna_count,
                    alpha=alpha,
                    worker_rank=0,
                )
            )
        return

    if rank == 0:
        assert pending_tasks is not None and collector is not None
        next_task = 0
        active_workers = 0
        for worker in range(1, size):
            if next_task < len(pending_tasks):
                comm.send(pending_tasks[next_task], dest=worker, tag=TASK_TAG)
                next_task += 1
                active_workers += 1
            else:
                comm.send(None, dest=worker, tag=STOP_TAG)

        status = MPI.Status()
        while active_workers:
            payload = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.Get_source()
            tag = status.Get_tag()
            if tag == ERROR_TAG:
                print(
                    f"Worker rank {worker} failed:\n{payload}",
                    file=sys.stderr,
                    flush=True,
                )
                collector.close()
                comm.Abort(2)
                raise RuntimeError(f"Worker rank {worker} failed")
            if tag != RESULT_TAG:
                collector.close()
                comm.Abort(2)
                raise RuntimeError(f"Unexpected MPI tag {tag} from rank {worker}")
            try:
                collector.add(payload)
            except Exception:
                print(traceback.format_exc(), file=sys.stderr, flush=True)
                collector.close()
                comm.Abort(2)
                raise
            active_workers -= 1
            if next_task < len(pending_tasks):
                comm.send(pending_tasks[next_task], dest=worker, tag=TASK_TAG)
                next_task += 1
                active_workers += 1
            else:
                comm.send(None, dest=worker, tag=STOP_TAG)
    else:
        while True:
            status = MPI.Status()
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == STOP_TAG:
                break
            if status.Get_tag() != TASK_TAG:
                comm.send(
                    f"Unexpected MPI tag {status.Get_tag()} from rank 0",
                    dest=0,
                    tag=ERROR_TAG,
                )
                break
            try:
                result = run_greedy_task(
                    scorer,
                    task,
                    antenna_count=antenna_count,
                    alpha=alpha,
                    worker_rank=rank,
                )
            except Exception:
                comm.send(traceback.format_exc(), dest=0, tag=ERROR_TAG)
                break
            comm.send(result, dest=0, tag=RESULT_TAG)


def deduplicate_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_points: dict[tuple[int, ...], dict[str, Any]] = {}
    for result in results:
        key = tuple(int(value) for value in result["selected_point_ids"])
        incumbent = by_points.get(key)
        if incumbent is None or result_is_better(result, incumbent):
            by_points[key] = result
    return sorted(by_points.values(), key=result_sort_key)


def select_point_swap_inputs(
    top_solutions: Sequence[TopSolution],
    *,
    requested_count: int,
    elite_count: int,
) -> list[dict[str, Any]]:
    target = min(requested_count, len(top_solutions))
    elite = min(elite_count, target)
    selected: list[TopSolution] = []
    rows: list[dict[str, Any]] = []
    point_sets = {
        item.top_rank: frozenset(int(value) for value in item.result["selected_point_ids"])
        for item in top_solutions
    }
    building_sets = {
        item.top_rank: frozenset(int(value) for value in item.details.building_ids)
        for item in top_solutions
    }
    similarity_cache: dict[tuple[int, int], tuple[float, float]] = {}

    def pair_similarity(left: TopSolution, right: TopSolution) -> tuple[float, float]:
        key = tuple(sorted((left.top_rank, right.top_rank)))
        cached = similarity_cache.get(key)
        if cached is not None:
            return cached
        left_points = point_sets[left.top_rank]
        right_points = point_sets[right.top_rank]
        point_denominator = max(len(left_points), len(right_points), 1)
        point_similarity = len(left_points.intersection(right_points)) / point_denominator
        left_buildings = building_sets[left.top_rank]
        right_buildings = building_sets[right.top_rank]
        union = left_buildings.union(right_buildings)
        building_similarity = (
            len(left_buildings.intersection(right_buildings)) / len(union)
            if union
            else 1.0
        )
        similarity_cache[key] = (point_similarity, building_similarity)
        return point_similarity, building_similarity

    def diversity(candidate: TopSolution) -> tuple[float, float]:
        if not selected:
            return 1.0, 1.0
        similarities = [pair_similarity(candidate, item) for item in selected]
        return (
            1.0 - max(value[0] for value in similarities),
            1.0 - max(value[1] for value in similarities),
        )

    for candidate in top_solutions[:elite]:
        point_diversity, building_diversity = diversity(candidate)
        selected.append(candidate)
        rows.append(
            {
                "selection_rank": len(selected),
                "stage": "elite",
                "top_rank": candidate.top_rank,
                "quality": candidate.quality,
                "point_diversity": point_diversity,
                "building_diversity": building_diversity,
                "utility": candidate.quality,
                "hard_score": candidate.result["hard_score"],
                "soft_score": candidate.result["soft_score"],
                "case_id": candidate.result["case_id"],
                "solution_path": str(candidate.path.resolve()),
                "_solution": candidate,
            }
        )

    remaining = list(top_solutions[elite:])
    while len(selected) < target:
        scored: list[tuple[tuple[float, ...], TopSolution, float, float, float]] = []
        for candidate in remaining:
            point_diversity, building_diversity = diversity(candidate)
            utility = (
                0.60 * candidate.quality
                + 0.20 * point_diversity
                + 0.20 * building_diversity
            )
            key = (
                utility,
                candidate.quality,
                point_diversity,
                building_diversity,
                -float(candidate.top_rank),
            )
            scored.append(
                (key, candidate, point_diversity, building_diversity, utility)
            )
        _, chosen, point_diversity, building_diversity, utility = max(
            scored, key=lambda item: item[0]
        )
        selected.append(chosen)
        remaining = [candidate for candidate in remaining if candidate is not chosen]
        rows.append(
            {
                "selection_rank": len(selected),
                "stage": "diverse",
                "top_rank": chosen.top_rank,
                "quality": chosen.quality,
                "point_diversity": point_diversity,
                "building_diversity": building_diversity,
                "utility": utility,
                "hard_score": chosen.result["hard_score"],
                "soft_score": chosen.result["soft_score"],
                "case_id": chosen.result["case_id"],
                "solution_path": str(chosen.path.resolve()),
                "_solution": chosen,
            }
        )
    return rows


def finalize_setting(
    *,
    collector: ResultCollector,
    keep_top: int,
    point_swap_input_count: int,
    elite_count: int,
) -> dict[str, Any]:
    if len(collector.results) != collector.expected_results:
        raise RuntimeError(
            f"Cannot finalize {len(collector.results)} results; "
            f"expected {collector.expected_results}"
        )
    collector.refresh_best_metadata()
    unique_results = deduplicate_results(collector.results)
    kept_results = unique_results[:keep_top]
    solution_dir = collector.setting_dir / "top_solutions"
    solution_dir.mkdir(parents=True, exist_ok=True)
    top_solutions: list[TopSolution] = []
    top_rows: list[dict[str, Any]] = []
    denominator = max(1, len(kept_results) - 1)

    for index, result in enumerate(kept_results, start=1):
        details = rebuild_solution(
            collector.scorer,
            result["selected_point_ids"],
            collector.coordinate_by_id,
            collector.official_building_ids,
        )
        if details.hard_score != int(result["hard_score"]):
            raise RuntimeError(f"Top solution {result['case_id']} failed hard-score validation")
        if not math.isclose(
            details.soft_score,
            float(result["soft_score"]),
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            raise RuntimeError(f"Top solution {result['case_id']} failed soft-score validation")
        # Stable rank-only names let a resumed finalization overwrite old ranks
        # without leaving stale case/score-named TXT files behind.
        filename = f"rank_{index:03d}.txt"
        path = solution_dir / filename
        write_solution_block(
            path,
            collector.tau,
            details.coordinates,
            details.building_ids,
        )
        quality = 1.0 - (index - 1) / denominator if len(kept_results) > 1 else 1.0
        top = TopSolution(index, result, details, path, quality)
        top_solutions.append(top)
        top_rows.append(
            {
                "top_rank": index,
                "case_id": result["case_id"],
                "hard_score": int(result["hard_score"]),
                "soft_score": float_text(result["soft_score"]),
                "quality": float_text(quality),
                "start_point_id": int(result["start_point_id"]),
                "source": result["source"],
                "selected_point_ids": point_ids_text(result["selected_point_ids"]),
                "building_count": len(details.building_ids),
                "solution_path": str(path.resolve()),
            }
        )

    atomic_write_tsv(
        collector.setting_dir / "top_solutions.tsv",
        (
            "top_rank",
            "case_id",
            "hard_score",
            "soft_score",
            "quality",
            "start_point_id",
            "source",
            "selected_point_ids",
            "building_count",
            "solution_path",
        ),
        top_rows,
    )

    selection_rows = select_point_swap_inputs(
        top_solutions,
        requested_count=point_swap_input_count,
        elite_count=elite_count,
    )
    atomic_write_text(
        collector.setting_dir / "point_swap_inputs.txt",
        "".join(f"{row['solution_path']}\n" for row in selection_rows),
    )
    atomic_write_tsv(
        collector.setting_dir / "point_swap_selection.tsv",
        (
            "selection_rank",
            "stage",
            "top_rank",
            "quality",
            "point_diversity",
            "building_diversity",
            "utility",
            "hard_score",
            "soft_score",
            "case_id",
            "solution_path",
        ),
        (
            {
                key: float_text(value) if key in {
                    "quality",
                    "point_diversity",
                    "building_diversity",
                    "utility",
                    "soft_score",
                } else value
                for key, value in row.items()
                if key != "_solution"
            }
            for row in selection_rows
        ),
    )
    return {
        "tau": collector.tau,
        "results": len(collector.results),
        "unique_solutions": len(unique_results),
        "top_solutions": len(top_solutions),
        "point_swap_inputs": len(selection_rows),
        "best_hard": int(collector.best["hard_score"]) if collector.best else None,
        "best_soft": float(collector.best["soft_score"]) if collector.best else None,
    }


def validate_args(args: argparse.Namespace) -> None:
    for path_name in ("segments_csv", "candidate_points_gpkg", "geojson"):
        path = getattr(args, path_name)
        if not path.is_file():
            raise FileNotFoundError(f"--{path_name.replace('_', '-')} not found: {path}")
    if not args.tau:
        raise ValueError("At least one --tau value is required")
    if len(set(args.tau)) != len(args.tau):
        raise ValueError("--tau values must be unique")
    if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in args.tau):
        raise ValueError("Every --tau value must be finite and in (0, 1]")
    if args.antenna_count < 1:
        raise ValueError("--antenna-count must be at least 1")
    if not math.isfinite(args.alpha) or not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be finite and in [0, 1]")
    if not math.isfinite(args.grid_size_meters) or args.grid_size_meters <= 0.0:
        raise ValueError("--grid-size-meters must be finite and positive")
    if args.grid_top_k < 0:
        raise ValueError("--grid-top-k must be nonnegative")
    if not 0.0 <= args.global_top_fraction <= 1.0:
        raise ValueError("--global-top-fraction must be in [0, 1]")
    if args.random_count < 0:
        raise ValueError("--random-count must be nonnegative")
    if args.random_seed < 0:
        raise ValueError("--random-seed must be nonnegative")
    if args.grid_top_k == 0 and args.global_top_fraction == 0.0 and args.random_count == 0:
        raise ValueError("The start-selection parameters would select no starts")
    if args.keep_top < 1:
        raise ValueError("--keep-top must be at least 1")
    if args.point_swap_input_count < 0:
        raise ValueError("--point-swap-input-count must be nonnegative")
    if args.point_swap_input_count > args.keep_top:
        raise ValueError("--point-swap-input-count cannot exceed --keep-top")
    if args.elite_count < 0:
        raise ValueError("--elite-count must be nonnegative")
    if args.chunksize < 1 or args.checkpoint_every < 1 or args.print_every < 1:
        raise ValueError("chunksize/checkpoint-every/print-every must be positive")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grid/global/random multi-start greedy antenna selection with MPI."
    )
    parser.add_argument("--segments-csv", type=Path, required=True)
    parser.add_argument("--candidate-points-gpkg", type=Path, required=True)
    parser.add_argument("--candidate-layer", default="candidate_points")
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--building-id-column", default="id")
    parser.add_argument("--tau", type=float, nargs="+", required=True)
    parser.add_argument("--antenna-count", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--grid-size-meters", type=float, default=100.0)
    parser.add_argument("--grid-top-k", type=int, default=10)
    parser.add_argument("--global-top-fraction", type=float, default=0.01)
    parser.add_argument("--random-count", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260815)
    parser.add_argument("--keep-top", type=int, default=200)
    parser.add_argument("--point-swap-input-count", type=int, default=50)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing output directory after strict config validation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    root_payload: dict[str, Any] | None = None
    coordinates: CandidateCoordinates | None = None
    official_building_ids: np.ndarray | None = None
    run_lock = None
    if rank == 0:
        try:
            validate_args(args)
            config = build_run_config(args)
            run_lock = prepare_output_directory(
                args.output_dir,
                config,
                resume=args.resume,
                mpi_size=size,
            )
            print(f"[{utc_now()}] loading candidate coordinates", flush=True)
            coordinates = load_candidate_coordinates(
                args.candidate_points_gpkg,
                args.candidate_layer,
            )
            print(f"[{utc_now()}] loading official building IDs", flush=True)
            official_building_ids = load_official_building_ids(
                args.geojson,
                args.building_id_column,
            )
            root_payload = {"ok": True}
        except Exception:
            root_payload = {"ok": False, "error": traceback.format_exc()}
    root_payload = comm.bcast(root_payload, root=0)
    if not root_payload["ok"]:
        if rank == 0:
            print(root_payload["error"], file=sys.stderr)
        return 2

    summaries: list[dict[str, Any]] = []
    for tau in args.tau:
        if rank == 0:
            print(
                f"[{utc_now()}] loading scorer for tau={tau:g} on {size} MPI ranks",
                flush=True,
            )
        scorer: Any | None = None
        load_error: str | None = None
        try:
            scorer = load_incremental_scorer(
                args.segments_csv,
                float(tau),
                args.chunksize,
            )
        except Exception:
            load_error = traceback.format_exc()
        load_errors = comm.allgather(load_error)
        if any(error is not None for error in load_errors):
            if rank == 0:
                for failed_rank, error in enumerate(load_errors):
                    if error is not None:
                        print(f"Rank {failed_rank} scorer load failed:\n{error}", file=sys.stderr)
            return 2
        assert scorer is not None

        setting_dir = args.output_dir / f"tau_{tau_token(tau)}_k_{args.antenna_count}"
        tasks: list[dict[str, Any]] | None = None
        prep_payload: dict[str, Any] | None = None
        if rank == 0:
            try:
                assert coordinates is not None
                scorer_ids = np.asarray(scorer.point_ids, dtype=np.int64)
                if args.antenna_count > len(scorer_ids):
                    raise ValueError(
                        f"antenna_count={args.antenna_count} exceeds "
                        f"{len(scorer_ids)} scoreable points"
                    )
                aligned_x, aligned_y = align_coordinates(coordinates, scorer_ids)
                first_points_path = setting_dir / "first_points.tsv"
                if args.resume and first_points_path.is_file():
                    tasks = load_tasks(first_points_path, float(tau))
                    print(
                        f"[{utc_now()}] tau={tau:g} loaded {len(tasks)} immutable starts",
                        flush=True,
                    )
                else:
                    setting_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"[{utc_now()}] tau={tau:g} scoring all {len(scorer_ids)} first points",
                        flush=True,
                    )
                    tasks, metadata = select_start_tasks(
                        tau=float(tau),
                        scorer=scorer,
                        x=aligned_x,
                        y=aligned_y,
                        alpha=args.alpha,
                        grid_size_meters=args.grid_size_meters,
                        grid_top_k=args.grid_top_k,
                        global_top_fraction=args.global_top_fraction,
                        random_count=args.random_count,
                        random_seed=args.random_seed,
                    )
                    atomic_write_tsv(
                        setting_dir / "first_point_scores.tsv",
                        FIRST_SCORE_FIELDS,
                        first_score_rows(float(tau), metadata),
                    )
                    atomic_write_json(
                        setting_dir / "start_selection_summary.json",
                        {
                            "tau": float(tau),
                            "scoreable_points": len(scorer_ids),
                            "occupied_cells": int(metadata["occupied_cell_count"][0]),
                            "selected_starts": len(tasks),
                            "grid_selected_starts": int(metadata["grid_selected_count"][0]),
                            "global_selected_starts": int(metadata["global_selected_count"][0]),
                            "actual_random_starts": int(metadata["actual_random_count"][0]),
                            "grid_size_meters": args.grid_size_meters,
                            "grid_top_k": args.grid_top_k,
                            "global_top_fraction": args.global_top_fraction,
                            "requested_random_starts": args.random_count,
                            "random_seed": args.random_seed,
                        },
                    )
                    # Write the immutable task manifest last.  On resume its
                    # presence means both companion first-point files are done.
                    atomic_write_tsv(
                        first_points_path,
                        FIRST_POINT_FIELDS,
                        (task_tsv_row(task) for task in tasks),
                    )
                    print(
                        f"[{utc_now()}] tau={tau:g} selected {len(tasks)} starts "
                        f"from {int(metadata['occupied_cell_count'][0])} cells",
                        flush=True,
                    )
                existing_results = load_existing_results(setting_dir / "all_results.tsv")
                task_by_case = {task["case_id"]: task for task in tasks}
                if len(task_by_case) != len(tasks):
                    raise ValueError("Duplicate case IDs in first_points.tsv")
                for result in existing_results:
                    task = task_by_case.get(result["case_id"])
                    if task is None:
                        raise ValueError(
                            f"Journal case {result['case_id']} is absent from first_points.tsv"
                        )
                    if (
                        int(result["antenna_count"]) != args.antenna_count
                        or not math.isclose(float(result["tau"]), float(tau))
                        or not math.isclose(float(result["alpha"]), args.alpha)
                    ):
                        raise ValueError(
                            f"Journal result {result['case_id']} does not match this setting"
                        )
                    for field in (
                        "task_index",
                        "start_point_id",
                        "source",
                        "cell_x",
                        "cell_y",
                        "global_rank",
                        "rank_in_cell",
                    ):
                        if result[field] != task[field]:
                            raise ValueError(
                                f"Journal result {result['case_id']} has a mismatched {field}"
                            )
                completed = {result["case_id"] for result in existing_results}
                pending_tasks = [task for task in tasks if task["case_id"] not in completed]
                prep_payload = {
                    "ok": True,
                    "task_count": len(tasks),
                    "completed_count": len(existing_results),
                    "pending_count": len(pending_tasks),
                }
            except Exception:
                prep_payload = {"ok": False, "error": traceback.format_exc()}
        prep_payload = comm.bcast(prep_payload, root=0)
        if not prep_payload["ok"]:
            if rank == 0:
                print(prep_payload["error"], file=sys.stderr)
            return 2

        collector: ResultCollector | None = None
        pending_tasks_for_execution: list[dict[str, Any]] | None = None
        collector_payload: dict[str, Any] | None = None
        if rank == 0:
            try:
                assert tasks is not None
                assert coordinates is not None
                assert official_building_ids is not None
                existing_results = load_existing_results(setting_dir / "all_results.tsv")
                completed = {result["case_id"] for result in existing_results}
                pending_tasks_for_execution = [
                    task for task in tasks if task["case_id"] not in completed
                ]
                collector = ResultCollector(
                    setting_dir=setting_dir,
                    tau=float(tau),
                    antenna_count=args.antenna_count,
                    scorer=scorer,
                    coordinate_by_id=coordinates.by_id,
                    official_building_ids=official_building_ids,
                    expected_results=len(tasks),
                    checkpoint_every=args.checkpoint_every,
                    print_every=args.print_every,
                    existing_results=existing_results,
                )
                collector_payload = {"ok": True}
                print(
                    f"[{utc_now()}] tau={tau:g} running "
                    f"{len(pending_tasks_for_execution)} pending / {len(tasks)} total starts "
                    f"with {size} MPI ranks",
                    flush=True,
                )
            except Exception:
                if collector is not None:
                    collector.close()
                collector_payload = {"ok": False, "error": traceback.format_exc()}
        collector_payload = comm.bcast(collector_payload, root=0)
        if not collector_payload["ok"]:
            if rank == 0:
                print(collector_payload["error"], file=sys.stderr)
            return 2

        execute_tasks(
            comm=comm,
            rank=rank,
            size=size,
            scorer=scorer,
            pending_tasks=pending_tasks_for_execution,
            antenna_count=args.antenna_count,
            alpha=args.alpha,
            collector=collector,
        )

        final_payload: dict[str, Any] | None = None
        if rank == 0:
            assert collector is not None
            try:
                summary = finalize_setting(
                    collector=collector,
                    keep_top=args.keep_top,
                    point_swap_input_count=args.point_swap_input_count,
                    elite_count=args.elite_count,
                )
                collector.close()
                summaries.append(summary)
                final_payload = {"ok": True, "summary": summary}
                print(
                    f"[{utc_now()}] tau={tau:g} complete: "
                    f"best_hard={summary['best_hard']} unique={summary['unique_solutions']}",
                    flush=True,
                )
            except Exception:
                collector.close()
                final_payload = {"ok": False, "error": traceback.format_exc()}
        final_payload = comm.bcast(final_payload, root=0)
        if not final_payload["ok"]:
            if rank == 0:
                print(final_payload["error"], file=sys.stderr)
            return 2
        collector = None
        del scorer
        gc.collect()
        comm.Barrier()

    if rank == 0:
        atomic_write_json(
            args.output_dir / "run_summary.json",
            {
                "completed_utc": utc_now(),
                "mpi_ranks": size,
                "settings": summaries,
            },
        )
        print(f"[{utc_now()}] all settings complete: {args.output_dir.resolve()}", flush=True)
        if run_lock is not None:
            run_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
