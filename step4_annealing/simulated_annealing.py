#!/usr/bin/env python3
"""MPI island simulated annealing for fixed-size antenna solutions.

Every MPI rank runs an independent annealing island.  The official objective
is the hard score: the number of buildings whose union boundary coverage is at
least ``tau``.  A capped soft potential is used only to guide moves across a
hard-score plateau::

    soft(S) = sum_b min(coverage_b(S) / tau, 1)

The implementation deliberately keeps ``current`` and ``best`` states
separate.  The current state may move sideways or temporarily lose hard
score, while the best state is never replaced by a worse solution.

The scorer already exposes exact marginal soft gains.  If points R are
removed from a state, adding R back to a clone telescopes to the exact soft
loss.  This lets the program maintain the state soft potential without
rescanning every building and without changing the C++ extension.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.scorer_loader import load_incremental_scorer
from common.search_common import (
    StopFlag,
    match_solution_points,
    read_point_coordinates,
    read_solution,
    write_best_solution,
)
from common.solution_txt import (
    load_official_building_ids,
    map_internal_building_ids,
    parse_solution_blocks,
)


CHECKPOINT_VERSION = 2
MOVE_ONE = "one_swap"
MOVE_TWO = "two_swap"
MOVE_RUIN = "ruin_recreate"
MOVE_NAMES = (MOVE_ONE, MOVE_TWO, MOVE_RUIN)


@dataclass
class SearchState:
    """A mutable scorer paired with its two optimization metrics."""

    scorer: Any
    hard_score: int
    soft_score: float

    @property
    def selected_ids(self) -> tuple[int, ...]:
        return canonical_ids(self.scorer.selected_point_ids)


@dataclass(frozen=True)
class Proposal:
    """One complete fixed-k neighbor generated from the current state."""

    state: SearchState
    move_name: str
    removed_ids: tuple[int, ...]
    added_ids: tuple[int, ...]
    evaluated_candidates: int


@dataclass(frozen=True)
class IslandSeedAssignment:
    """The immutable input seed and temperature assignment for one MPI rank."""

    seed_index: int
    temperature_profile_index: int
    replica_index: int
    selected_point_ids: tuple[int, ...]
    expected_building_ids: tuple[int, ...]
    seed_path: str


@dataclass
class RunStatistics:
    """Counters written to logs and checkpoints."""

    proposed: int = 0
    accepted: int = 0
    accepted_hard_down: int = 0
    accepted_plateau: int = 0
    hard_improvements: int = 0
    best_updates: int = 0
    one_swap_proposals: int = 0
    two_swap_proposals: int = 0
    ruin_proposals: int = 0

    def record_proposal(self, move_name: str) -> None:
        self.proposed += 1
        if move_name == MOVE_ONE:
            self.one_swap_proposals += 1
        elif move_name == MOVE_TWO:
            self.two_swap_proposals += 1
        elif move_name == MOVE_RUIN:
            self.ruin_proposals += 1
        else:
            raise ValueError(f"Unknown move name: {move_name}")

    def to_dict(self) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in vars(self).items()
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "RunStatistics":
        fields = cls().__dict__
        return cls(
            **{
                key: int(values.get(key, default))
                for key, default in fields.items()
            }
        )


@dataclass
class ResumeState:
    current: SearchState
    best: SearchState
    iteration: int
    cycle_iteration: int
    reheat_count: int
    last_best_iteration: int
    checkpoint_serial: int
    statistics: RunStatistics
    rng_state: dict[str, object]
    hard_temperature_start: float
    hard_temperature_end: float
    soft_temperature_start: float
    soft_temperature_end: float


def canonical_ids(values: Iterable[int]) -> tuple[int, ...]:
    """Return a sorted point-set identity and reject duplicates."""

    result = tuple(sorted(int(value) for value in values))
    if len(result) != len(set(result)):
        raise ValueError("A solution contains duplicate point IDs")
    return result


def hard_temperature_for_loss_acceptance(probability: float) -> float:
    """Temperature that accepts a one-hard-point loss with ``probability``."""

    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("loss acceptance probability must be in (0, 1)")
    return -1.0 / math.log(probability)


def geometric_temperature(
    start: float,
    end: float,
    progress: float,
) -> float:
    """Geometrically interpolate a positive temperature over [0, 1]."""

    if not math.isfinite(start) or start <= 0.0:
        raise ValueError("start temperature must be finite and positive")
    if not math.isfinite(end) or end <= 0.0:
        raise ValueError("end temperature must be finite and positive")
    if not math.isfinite(progress):
        raise ValueError("temperature progress must be finite")
    fraction = min(1.0, max(0.0, progress))
    return start * math.pow(end / start, fraction)


def island_temperature_multiplier(
    rank: int,
    levels: int,
    log_spread: float,
) -> float:
    """Assign repeating cold-to-hot temperature profiles across MPI ranks."""

    if rank < 0:
        raise ValueError("rank cannot be negative")
    if levels <= 0:
        raise ValueError("temperature profile levels must be positive")
    if not math.isfinite(log_spread) or log_spread < 0.0:
        raise ValueError("temperature log spread must be finite and non-negative")
    if levels == 1:
        return 1.0
    position = (rank % levels) / float(levels - 1)
    return math.exp(log_spread * (2.0 * position - 1.0))


def balanced_island_assignment(
    rank: int,
    world_size: int,
    seed_count: int,
    temperature_levels: int,
) -> tuple[int, int, int]:
    """Assign every seed/temperature combination the same number of islands.

    Ranks are arranged as ``replica -> temperature -> seed``.  This avoids
    coupling seed index and temperature when their counts happen to match.
    The strict divisibility check makes the allocation easy to audit.
    """

    if world_size <= 0:
        raise ValueError("MPI world size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be within the MPI world")
    if seed_count <= 0:
        raise ValueError("seed count must be positive")
    if temperature_levels <= 0:
        raise ValueError("temperature profile levels must be positive")

    combinations = seed_count * temperature_levels
    if world_size < combinations:
        raise ValueError(
            f"MPI world size {world_size} is too small for {seed_count} seeds x "
            f"{temperature_levels} temperature profiles; need at least {combinations}"
        )
    if world_size % combinations != 0:
        raise ValueError(
            f"MPI world size {world_size} must be divisible by {seed_count} seeds x "
            f"{temperature_levels} temperature profiles ({combinations})"
        )

    seed_index = rank % seed_count
    temperature_index = (rank // seed_count) % temperature_levels
    replica_index = rank // combinations
    return seed_index, temperature_index, replica_index


def assign_island_seed(
    *,
    rank: int,
    world_size: int,
    seed_ids: Sequence[tuple[int, ...]],
    seed_expected_building_ids: Sequence[tuple[int, ...]],
    seed_paths: Sequence[str],
    temperature_levels: int,
) -> IslandSeedAssignment:
    """Freeze the complete per-rank seed assignment before input validation.

    Keeping the selected seed in an immutable object prevents a later loop
    variable from accidentally changing which seed this rank starts from.
    """

    if not (
        len(seed_ids) == len(seed_expected_building_ids) == len(seed_paths)
    ):
        raise ValueError(
            "seed IDs, expected building IDs, and seed paths must have equal lengths"
        )
    seed_index, temperature_index, replica_index = balanced_island_assignment(
        rank,
        world_size,
        len(seed_ids),
        temperature_levels,
    )
    return IslandSeedAssignment(
        seed_index=seed_index,
        temperature_profile_index=temperature_index,
        replica_index=replica_index,
        selected_point_ids=seed_ids[seed_index],
        expected_building_ids=seed_expected_building_ids[seed_index],
        seed_path=seed_paths[seed_index],
    )


def is_better(
    hard_score: int,
    soft_score: float,
    reference_hard: int,
    reference_soft: float,
    tolerance: float = 1e-12,
) -> bool:
    """Lexicographic best-state comparison with hard score first."""

    if hard_score != reference_hard:
        return hard_score > reference_hard
    return soft_score > reference_soft + tolerance


def accept_move(
    *,
    delta_hard: int,
    delta_soft: float,
    hard_temperature: float,
    soft_temperature: float,
    rng: np.random.Generator,
) -> bool:
    """Apply hard-primary, soft-on-plateau Metropolis acceptance.

    A hard improvement is always accepted.  A hard loss is controlled only by
    the hard temperature, preventing a large soft change from overruling the
    official objective.  On a hard-score plateau, the soft potential supplies
    the annealing gradient.
    """

    if not math.isfinite(delta_soft):
        raise ValueError("delta_soft must be finite")
    if not math.isfinite(hard_temperature) or hard_temperature < 0.0:
        raise ValueError("hard_temperature must be finite and non-negative")
    if not math.isfinite(soft_temperature) or soft_temperature < 0.0:
        raise ValueError("soft_temperature must be finite and non-negative")

    if delta_hard > 0:
        return True
    if delta_hard == 0 and delta_soft >= 0.0:
        return True

    if delta_hard < 0:
        if hard_temperature == 0.0:
            return False
        log_acceptance = float(delta_hard) / hard_temperature
    else:
        if soft_temperature == 0.0:
            return False
        log_acceptance = delta_soft / soft_temperature

    # Comparing in log space avoids overflow and remains stable for tiny
    # probabilities.  Generator.random() is in [0, 1), so guard the zero case.
    draw = max(float(rng.random()), np.finfo(np.float64).tiny)
    return math.log(draw) < min(0.0, log_acceptance)


def sample_without_replacement(
    rng: np.random.Generator,
    values: np.ndarray,
    size: int,
) -> np.ndarray:
    """Return a shuffled full set or a uniform subset without replacement."""

    if size <= 0:
        raise ValueError("sample size must be positive")
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("sample values must be one-dimensional")
    if values.size == 0:
        raise ValueError("cannot sample an empty candidate set")
    if size >= values.size:
        result = values.copy()
        rng.shuffle(result)
        return result
    return np.asarray(
        rng.choice(values, size=size, replace=False),
        dtype=np.int64,
    )


def choose_ranked_candidate(
    *,
    point_ids: np.ndarray,
    hard_gains: np.ndarray,
    soft_gains: np.ndarray,
    unique_lengths: np.ndarray,
    top_count: int,
    rank_temperature: float,
    random_candidate_probability: float,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Choose a candidate from a hard/soft/unique lexicographic ranking.

    Most proposals sample from the top part of the evaluated pool using an
    exponential distribution over rank.  A configurable fraction chooses a
    completely random evaluated candidate to preserve broad exploration.
    Returns ``(point_id, raw_soft_gain)``.
    """

    arrays = [
        np.asarray(point_ids, dtype=np.int64),
        np.asarray(hard_gains, dtype=np.int64),
        np.asarray(soft_gains, dtype=np.float64),
        np.asarray(unique_lengths, dtype=np.float64),
    ]
    length = arrays[0].size
    if length == 0:
        raise ValueError("candidate score arrays cannot be empty")
    if any(array.ndim != 1 or array.size != length for array in arrays):
        raise ValueError("candidate score arrays must be aligned 1-D arrays")
    if top_count <= 0:
        raise ValueError("top_count must be positive")
    if not math.isfinite(rank_temperature) or rank_temperature <= 0.0:
        raise ValueError("rank_temperature must be finite and positive")
    if not 0.0 <= random_candidate_probability <= 1.0:
        raise ValueError("random candidate probability must be in [0, 1]")
    if not np.all(np.isfinite(arrays[2])) or not np.all(np.isfinite(arrays[3])):
        raise ValueError("candidate soft scores and lengths must be finite")

    if float(rng.random()) < random_candidate_probability:
        chosen_position = int(rng.integers(length))
    else:
        # np.lexsort uses its last key as primary.  The desired ordering is:
        # hard descending, soft descending, unique length descending, ID ascending.
        order = np.lexsort(
            (
                arrays[0],
                -arrays[3],
                -arrays[2],
                -arrays[1],
            )
        )
        ranked = order[: min(top_count, length)]
        rank_numbers = np.arange(ranked.size, dtype=np.float64)
        weights = np.exp(-rank_numbers / rank_temperature)
        weights /= weights.sum()
        chosen_rank = int(rng.choice(ranked.size, p=weights))
        chosen_position = int(ranked[chosen_rank])

    return int(arrays[0][chosen_position]), float(arrays[2][chosen_position])


def available_ids(
    all_point_ids: np.ndarray,
    selected_ids: Iterable[int],
    blocked_ids: Iterable[int],
) -> np.ndarray:
    """Return legal replacements, excluding selected and deliberately removed IDs."""

    excluded = tuple(
        sorted(
            set(int(value) for value in selected_ids).union(
                int(value) for value in blocked_ids
            )
        )
    )
    if not excluded:
        return np.asarray(all_point_ids, dtype=np.int64).copy()
    excluded_array = np.asarray(excluded, dtype=np.int64)
    mask = ~np.isin(
        np.asarray(all_point_ids, dtype=np.int64),
        excluded_array,
        assume_unique=True,
    )
    return np.asarray(all_point_ids[mask], dtype=np.int64)


def build_state(empty_scorer: Any, selected_ids: Sequence[int]) -> SearchState:
    """Rebuild a state and telescope the exact normalized soft potential."""

    canonical = canonical_ids(selected_ids)
    scorer = empty_scorer.clone()
    scorer.reset()
    raw_soft = 0.0
    for point_id in canonical:
        raw_soft += float(scorer.soft_score(point_id))
        scorer.add_point(point_id)
    if int(scorer.selected_count) != len(canonical):
        raise RuntimeError("Rebuilt scorer has the wrong selected count")
    return SearchState(
        scorer=scorer,
        hard_score=int(scorer.current_score),
        soft_score=raw_soft / float(scorer.tau),
    )


def exact_removed_soft_loss(
    scorer_after_removal: Any,
    removed_ids: Sequence[int],
) -> float:
    """Return normalized soft lost by removing several points.

    Adding all removed points back to a clone restores the original set.  The
    sum of their sequential marginal gains is therefore the exact state-soft
    difference, even when their visibility overlaps.
    """

    restoration = scorer_after_removal.clone()
    raw_loss = 0.0
    for point_id in removed_ids:
        raw_loss += float(restoration.soft_score(int(point_id)))
        restoration.add_point(int(point_id))
    return raw_loss / float(scorer_after_removal.tau)


def choose_move_size(
    *,
    selected_count: int,
    one_probability: float,
    two_probability: float,
    ruin_probability: float,
    ruin_size: int,
    rng: np.random.Generator,
) -> tuple[str, int]:
    """Sample one of the configured fixed-k neighborhood sizes."""

    if selected_count <= 0:
        raise ValueError("selected_count must be positive")
    probabilities = np.asarray(
        [one_probability, two_probability, ruin_probability],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("move probabilities must be finite and non-negative")
    if probabilities.sum() <= 0.0:
        raise ValueError("at least one move probability must be positive")
    if ruin_size <= 0:
        raise ValueError("ruin_size must be positive")

    sizes = np.asarray(
        [1, min(2, selected_count), min(ruin_size, selected_count)],
        dtype=np.int64,
    )
    # Merge move mass into one-swap when k is too small to create the requested
    # larger neighborhood.  This keeps every sampled move a real fixed-k swap.
    if selected_count < 2:
        probabilities[0] += probabilities[1] + probabilities[2]
        probabilities[1:] = 0.0
    elif sizes[2] < 3:
        probabilities[1] += probabilities[2]
        probabilities[2] = 0.0
    probabilities /= probabilities.sum()
    move_index = int(rng.choice(3, p=probabilities))
    return MOVE_NAMES[move_index], int(sizes[move_index])


def propose_neighbor(
    *,
    current: SearchState,
    all_point_ids: np.ndarray,
    move_name: str,
    move_size: int,
    candidate_pool_size: int,
    top_candidates: int,
    rank_temperature: float,
    random_candidate_probability: float,
    rng: np.random.Generator,
) -> Proposal:
    """Generate one guided stochastic r-remove/r-add proposal."""

    selected = np.asarray(current.selected_ids, dtype=np.int64)
    if not 1 <= move_size <= selected.size:
        raise ValueError("move_size must be in [1, selected_count]")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")

    removed = canonical_ids(
        rng.choice(selected, size=move_size, replace=False)
    )
    trial = current.scorer.clone()
    for point_id in removed:
        trial.remove_point(point_id)

    removed_soft_loss = exact_removed_soft_loss(trial, removed)
    proposed_soft = current.soft_score - removed_soft_loss
    added: list[int] = []
    total_evaluated = 0

    for _ in range(move_size):
        legal = available_ids(
            all_point_ids,
            trial.selected_point_ids,
            tuple(removed) + tuple(added),
        )
        if legal.size == 0:
            raise RuntimeError("No legal replacement candidates remain")
        pool = sample_without_replacement(
            rng,
            legal,
            min(candidate_pool_size, int(legal.size)),
        )
        batch = trial.evaluate_all(pool)
        point_ids = np.asarray(batch["point_indices"], dtype=np.int64)
        hard_gains = np.asarray(batch["hard_scores"], dtype=np.int64)
        soft_gains = np.asarray(batch["soft_scores"], dtype=np.float64)
        unique_lengths = np.asarray(batch["unique_lengths"], dtype=np.float64)
        replacement, raw_soft_gain = choose_ranked_candidate(
            point_ids=point_ids,
            hard_gains=hard_gains,
            soft_gains=soft_gains,
            unique_lengths=unique_lengths,
            top_count=top_candidates,
            rank_temperature=rank_temperature,
            random_candidate_probability=random_candidate_probability,
            rng=rng,
        )
        proposed_soft += raw_soft_gain / float(trial.tau)
        trial.add_point(replacement)
        added.append(replacement)
        total_evaluated += int(pool.size)

    if int(trial.selected_count) != int(current.scorer.selected_count):
        raise RuntimeError("A proposal changed the antenna count")
    if set(removed).intersection(added):
        raise RuntimeError("A proposal re-added a deliberately removed point")

    state = SearchState(
        scorer=trial,
        hard_score=int(trial.current_score),
        soft_score=float(proposed_soft),
    )
    return Proposal(
        state=state,
        move_name=move_name,
        removed_ids=removed,
        added_ids=tuple(added),
        evaluated_candidates=total_evaluated,
    )


def atomic_write_json(path: Path, value: object) -> None:
    """Atomically replace one JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_compatible(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_compatible(value: object) -> object:
    """Convert NumPy-rich RNG metadata into standard JSON values."""

    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def file_fingerprint(path: Path) -> dict[str, object]:
    """Fingerprint mutable input data strongly enough to catch path overwrites."""

    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def search_configuration(args: argparse.Namespace) -> dict[str, object]:
    """Return proposal settings that must remain unchanged on exact resume."""

    names = (
        "candidate_pool_size",
        "top_candidates",
        "proposal_rank_temperature",
        "random_candidate_probability",
        "one_swap_probability",
        "two_swap_probability",
        "ruin_probability",
        "ruin_size",
        "temperature_profile_levels",
        "temperature_log_spread",
        "cycle_iterations",
        "stagnation_iterations",
    )
    return {name: getattr(args, name) for name in names}


def qualifying_official_ids(
    scorer: Any,
    official_building_ids: np.ndarray,
) -> tuple[int, ...]:
    """Recompute the exact official building-ID set for one scorer state."""

    visible = scorer.current_visible_segments()
    visible_buildings = np.unique(
        np.asarray(visible["building_ids"], dtype=np.int64)
    )
    qualifying_internal = [
        int(building_id)
        for building_id in visible_buildings
        if scorer.building_coverage(int(building_id)) >= scorer.tau
    ]
    if len(qualifying_internal) != int(scorer.current_score):
        raise RuntimeError("Scorer score and qualifying-building count disagree")
    return tuple(
        map_internal_building_ids(
            qualifying_internal,
            official_building_ids,
        )
    )


def next_checkpoint_serial(rank_folder: Path, requested: int) -> int:
    """Skip incomplete orphan checkpoint directories left by an interruption."""

    if requested < 0:
        raise ValueError("checkpoint serial cannot be negative")
    serial = requested
    while (rank_folder / "checkpoints" / f"checkpoint_{serial:06d}").exists():
        serial += 1
    return serial


def write_rank_best(
    *,
    rank_folder: Path,
    state: SearchState,
    point_coordinates: dict[int, tuple[float, float]],
    official_building_ids: np.ndarray,
    iteration: int,
) -> None:
    """Atomically refresh the convenient per-rank best solution and metadata."""

    write_best_solution(
        state.scorer,
        point_coordinates,
        official_building_ids,
        rank_folder / "best_solution.txt",
    )
    atomic_write_json(
        rank_folder / "best.json",
        {
            "iteration": iteration,
            "hard_score": state.hard_score,
            "soft_score": state.soft_score,
            "selected_count": int(state.scorer.selected_count),
            "solution": "best_solution.txt",
        },
    )


def write_checkpoint(
    *,
    rank_folder: Path,
    serial: int,
    rank: int,
    world_size: int,
    current: SearchState,
    best: SearchState,
    point_coordinates: dict[int, tuple[float, float]],
    official_building_ids: np.ndarray,
    iteration: int,
    cycle_iteration: int,
    reheat_count: int,
    last_best_iteration: int,
    statistics: RunStatistics,
    rng: np.random.Generator,
    hard_temperature: float,
    soft_temperature: float,
    hard_temperature_start: float,
    hard_temperature_end: float,
    soft_temperature_start: float,
    soft_temperature_end: float,
    segments_fingerprint: dict[str, object],
    geojson_fingerprint: dict[str, object],
    building_id_column: str,
    proposal_configuration: dict[str, object],
) -> Path:
    """Write an immutable, resumable checkpoint and update latest.json last."""

    checkpoint_name = f"checkpoint_{serial:06d}"
    checkpoint_folder = rank_folder / "checkpoints" / checkpoint_name
    if checkpoint_folder.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_folder}")
    checkpoint_folder.mkdir(parents=True)

    write_best_solution(
        current.scorer,
        point_coordinates,
        official_building_ids,
        checkpoint_folder / "current_solution.txt",
    )
    write_best_solution(
        best.scorer,
        point_coordinates,
        official_building_ids,
        checkpoint_folder / "best_solution.txt",
    )
    metadata = {
        "version": CHECKPOINT_VERSION,
        "rank": rank,
        "world_size": world_size,
        "iteration": iteration,
        "cycle_iteration": cycle_iteration,
        "reheat_count": reheat_count,
        "last_best_iteration": last_best_iteration,
        "checkpoint_serial": serial,
        "tau": float(current.scorer.tau),
        "k": int(current.scorer.selected_count),
        "current": {
            "hard_score": current.hard_score,
            "soft_score": current.soft_score,
            "solution": "current_solution.txt",
        },
        "best": {
            "hard_score": best.hard_score,
            "soft_score": best.soft_score,
            "solution": "best_solution.txt",
        },
        "temperatures": {
            "current_hard": hard_temperature,
            "current_soft": soft_temperature,
            "hard_start": hard_temperature_start,
            "hard_end": hard_temperature_end,
            "soft_start": soft_temperature_start,
            "soft_end": soft_temperature_end,
        },
        "statistics": statistics.to_dict(),
        "rng_state": rng.bit_generator.state,
        "segments_fingerprint": segments_fingerprint,
        "geojson_fingerprint": geojson_fingerprint,
        "building_id_column": building_id_column,
        "proposal_configuration": proposal_configuration,
        "written_unix_time": time.time(),
    }
    atomic_write_json(checkpoint_folder / "metadata.json", metadata)
    atomic_write_json(
        rank_folder / "latest.json",
        {
            "checkpoint": f"checkpoints/{checkpoint_name}",
            "metadata": f"checkpoints/{checkpoint_name}/metadata.json",
            "iteration": iteration,
            "best_hard_score": best.hard_score,
        },
    )
    return checkpoint_folder


def load_checkpoint(
    *,
    rank_folder: Path,
    rank: int,
    world_size: int,
    empty_scorer: Any,
    point_coordinates: dict[int, tuple[float, float]],
    coordinate_tolerance: float,
    expected_fingerprint: dict[str, object],
    expected_geojson_fingerprint: dict[str, object],
    expected_building_id_column: str,
    expected_proposal_configuration: dict[str, object],
    expected_k: int,
) -> ResumeState:
    """Load and verify the latest complete checkpoint for one island."""

    latest_path = rank_folder / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Missing resume pointer: {latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    metadata_path = rank_folder / str(latest["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    if int(metadata["rank"]) != rank or int(metadata["world_size"]) != world_size:
        raise ValueError("Checkpoint rank/world_size does not match this MPI run")
    if metadata["segments_fingerprint"] != expected_fingerprint:
        raise ValueError(
            "Segments file changed since the checkpoint was written; refusing resume"
        )
    if metadata.get("geojson_fingerprint") != expected_geojson_fingerprint:
        raise ValueError(
            "Building GeoJSON changed since the checkpoint was written; "
            "refusing resume"
        )
    if metadata.get("building_id_column") != expected_building_id_column:
        raise ValueError("Building ID column does not match the checkpoint")
    if metadata.get("proposal_configuration") != expected_proposal_configuration:
        raise ValueError(
            "SA proposal parameters do not match the checkpoint; exact resume "
            "requires the same search settings"
        )
    if int(metadata["k"]) != expected_k:
        raise ValueError("Checkpoint k does not match the input solution k")
    if not math.isclose(float(metadata["tau"]), float(empty_scorer.tau)):
        raise ValueError("Checkpoint tau does not match the loaded scorer")

    checkpoint_folder = metadata_path.parent

    def load_named_state(section: str) -> SearchState:
        section_data = metadata[section]
        solution_path = checkpoint_folder / str(section_data["solution"])
        solution = read_solution(solution_path)
        selected = canonical_ids(
            match_solution_points(
                solution.coordinates,
                point_coordinates,
                coordinate_tolerance,
            )
        )
        state = build_state(empty_scorer, selected)
        if state.hard_score != int(section_data["hard_score"]):
            raise ValueError(f"Checkpoint {section} hard score failed verification")
        if not math.isclose(
            state.soft_score,
            float(section_data["soft_score"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"Checkpoint {section} soft score failed verification")
        return state

    current = load_named_state("current")
    best = load_named_state("best")
    if (
        int(current.scorer.selected_count) != expected_k
        or int(best.scorer.selected_count) != expected_k
    ):
        raise ValueError("Checkpoint antenna count failed verification")
    return ResumeState(
        current=current,
        best=best,
        iteration=int(metadata["iteration"]),
        cycle_iteration=int(metadata["cycle_iteration"]),
        reheat_count=int(metadata["reheat_count"]),
        last_best_iteration=int(metadata["last_best_iteration"]),
        checkpoint_serial=int(metadata["checkpoint_serial"]),
        statistics=RunStatistics.from_dict(metadata["statistics"]),
        rng_state=metadata["rng_state"],
        hard_temperature_start=float(metadata["temperatures"]["hard_start"]),
        hard_temperature_end=float(metadata["temperatures"]["hard_end"]),
        soft_temperature_start=float(metadata["temperatures"]["soft_start"]),
        soft_temperature_end=float(metadata["temperatures"]["soft_end"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent MPI simulated-annealing islands using guided "
            "one-swap, two-swap, and ruin/recreate proposals."
        )
    )
    parser.add_argument(
        "--input-result",
        type=Path,
        action="append",
        required=True,
        help=(
            "Initial three-line solution. Repeat the option to distribute "
            "several diverse seeds round-robin across ranks."
        ),
    )
    parser.add_argument("--segments-csv", type=Path, required=True)
    parser.add_argument(
        "--geojson",
        type=Path,
        required=True,
    )
    parser.add_argument("--building-id-column", default="id")
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-6)
    parser.add_argument("--random-seed-base", type=int, default=42)

    parser.add_argument("--candidate-pool-size", type=int, default=1_000)
    parser.add_argument("--top-candidates", type=int, default=32)
    parser.add_argument("--proposal-rank-temperature", type=float, default=6.0)
    parser.add_argument(
        "--random-candidate-probability",
        type=float,
        default=0.10,
    )
    parser.add_argument("--one-swap-probability", type=float, default=0.75)
    parser.add_argument("--two-swap-probability", type=float, default=0.20)
    parser.add_argument("--ruin-probability", type=float, default=0.05)
    parser.add_argument("--ruin-size", type=int, default=5)

    parser.add_argument(
        "--hard-loss-acceptance-start",
        type=float,
        default=0.10,
        help="Initial probability of accepting a one-building hard-score loss.",
    )
    parser.add_argument(
        "--hard-loss-acceptance-end",
        type=float,
        default=1e-4,
        help="Final probability of accepting a one-building hard-score loss.",
    )
    parser.add_argument(
        "--soft-worse-acceptance-start",
        type=float,
        default=0.50,
        help="Target initial acceptance for a typical soft-worsening plateau move.",
    )
    parser.add_argument(
        "--soft-worse-acceptance-end",
        type=float,
        default=0.01,
        help="Target final acceptance for a typical soft-worsening plateau move.",
    )
    parser.add_argument(
        "--soft-temperature-start",
        type=float,
        default=0.0,
        help="Override auto-calibration when positive.",
    )
    parser.add_argument(
        "--soft-temperature-end",
        type=float,
        default=0.0,
        help="Override auto-calibration when positive.",
    )
    parser.add_argument("--calibration-trials", type=int, default=12)
    parser.add_argument("--temperature-profile-levels", type=int, default=5)
    parser.add_argument("--temperature-log-spread", type=float, default=0.70)
    parser.add_argument("--cycle-iterations", type=int, default=20_000)
    parser.add_argument("--stagnation-iterations", type=int, default=20_000)

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Zero means no iteration limit.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Zero means run until an iteration limit or signal.",
    )
    parser.add_argument("--checkpoint-seconds", type=float, default=900.0)
    parser.add_argument("--print-seconds", type=float, default=60.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.segments_csv, "segments CSV"),
        (args.geojson, "building GeoJSON"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    for path in args.input_result:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input solution: {path}")
    if args.chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if args.random_seed_base < 0:
        raise ValueError("random seed base must be non-negative")
    if args.candidate_pool_size <= 0 or args.top_candidates <= 0:
        raise ValueError("candidate pool and top-candidate counts must be positive")
    if (
        not math.isfinite(args.proposal_rank_temperature)
        or args.proposal_rank_temperature <= 0.0
    ):
        raise ValueError("proposal rank temperature must be finite and positive")
    if args.ruin_size <= 0:
        raise ValueError("ruin size must be positive")
    if args.cycle_iterations <= 0 or args.stagnation_iterations <= 0:
        raise ValueError("cycle and stagnation iterations must be positive")
    if (
        args.max_iterations < 0
        or not math.isfinite(args.max_seconds)
        or args.max_seconds < 0.0
    ):
        raise ValueError("search limits cannot be negative")
    if (
        not math.isfinite(args.checkpoint_seconds)
        or not math.isfinite(args.print_seconds)
        or args.checkpoint_seconds <= 0.0
        or args.print_seconds <= 0.0
    ):
        raise ValueError("checkpoint and print intervals must be positive")
    if args.calibration_trials < 0:
        raise ValueError("calibration trials cannot be negative")
    if args.temperature_profile_levels <= 0:
        raise ValueError("temperature profile levels must be positive")
    if args.temperature_log_spread < 0.0:
        raise ValueError("temperature log spread cannot be negative")
    if (
        not math.isfinite(args.soft_temperature_start)
        or not math.isfinite(args.soft_temperature_end)
        or args.soft_temperature_start < 0.0
        or args.soft_temperature_end < 0.0
    ):
        raise ValueError("soft temperature overrides must be finite and non-negative")
    if (args.soft_temperature_start == 0.0) != (
        args.soft_temperature_end == 0.0
    ):
        raise ValueError(
            "set both soft temperature overrides or leave both at zero"
        )
    for value, label in (
        (args.random_candidate_probability, "random candidate probability"),
        (args.hard_loss_acceptance_start, "hard start acceptance"),
        (args.hard_loss_acceptance_end, "hard end acceptance"),
        (args.soft_worse_acceptance_start, "soft start acceptance"),
        (args.soft_worse_acceptance_end, "soft end acceptance"),
    ):
        if not 0.0 < value < 1.0:
            if label == "random candidate probability" and value == 0.0:
                continue
            raise ValueError(f"{label} must be in (0, 1)")
    move_total = (
        args.one_swap_probability
        + args.two_swap_probability
        + args.ruin_probability
    )
    move_values = np.asarray(
        [
            args.one_swap_probability,
            args.two_swap_probability,
            args.ruin_probability,
        ],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(move_values))
        or move_total <= 0.0
        or np.any(move_values < 0.0)
    ):
        raise ValueError("move probabilities must be non-negative with positive sum")
    if not math.isfinite(args.coordinate_tolerance) or args.coordinate_tolerance < 0:
        raise ValueError("coordinate tolerance must be finite and non-negative")


def prepare_root_payload(args: argparse.Namespace) -> dict[str, object]:
    """Read coordinates once, validate every seed, and prepare MPI broadcast data."""

    point_coordinates = read_point_coordinates(args.segments_csv, args.chunksize)
    seed_ids: list[tuple[int, ...]] = []
    seed_expected_building_ids: list[tuple[int, ...]] = []
    seed_tau: float | None = None
    seed_k: int | None = None
    for path in args.input_result:
        solution = read_solution(path)
        blocks = parse_solution_blocks(path)
        if len(blocks) != 1:
            raise ValueError(f"Seed must contain exactly one solution block: {path}")
        selected = canonical_ids(
            match_solution_points(
                solution.coordinates,
                point_coordinates,
                args.coordinate_tolerance,
            )
        )
        if seed_tau is None:
            seed_tau = float(solution.tau)
            seed_k = len(selected)
            if seed_k <= 0:
                raise ValueError("Input solutions must contain at least one antenna")
        elif not math.isclose(float(solution.tau), seed_tau):
            raise ValueError("All input solutions must use the same tau")
        elif len(selected) != seed_k:
            raise ValueError("All input solutions must use the same k")
        seed_ids.append(selected)
        seed_expected_building_ids.append(
            tuple(sorted(int(value) for value in blocks[0].building_ids))
        )

    if seed_tau is None or seed_k is None:
        raise RuntimeError("No input seeds were loaded")
    official_ids = load_official_building_ids(
        args.geojson,
        args.building_id_column,
    )
    return {
        "error": None,
        "tau": seed_tau,
        "k": seed_k,
        "seed_ids": seed_ids,
        "seed_expected_building_ids": seed_expected_building_ids,
        "seed_paths": [str(path.resolve()) for path in args.input_result],
        "point_coordinates": point_coordinates,
        "official_building_ids": official_ids,
        "segments_fingerprint": file_fingerprint(args.segments_csv),
        "geojson_fingerprint": file_fingerprint(args.geojson),
        "building_id_column": args.building_id_column,
    }


def calibrate_soft_temperatures(
    *,
    current: SearchState,
    all_point_ids: np.ndarray,
    args: argparse.Namespace,
    island_multiplier: float,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Estimate the typical negative soft change from discarded pilot proposals."""

    if args.soft_temperature_start > 0.0:
        return (
            args.soft_temperature_start * island_multiplier,
            args.soft_temperature_end * island_multiplier,
            0,
        )
    losses: list[float] = []
    for _ in range(args.calibration_trials):
        proposal = propose_neighbor(
            current=current,
            all_point_ids=all_point_ids,
            move_name=MOVE_ONE,
            move_size=1,
            candidate_pool_size=args.candidate_pool_size,
            top_candidates=args.top_candidates,
            rank_temperature=args.proposal_rank_temperature * island_multiplier,
            random_candidate_probability=args.random_candidate_probability,
            rng=rng,
        )
        delta_soft = proposal.state.soft_score - current.soft_score
        if delta_soft < -1e-12:
            losses.append(-delta_soft)
    typical_loss = float(np.median(losses)) if losses else 1.0
    start = typical_loss / -math.log(args.soft_worse_acceptance_start)
    end = typical_loss / -math.log(args.soft_worse_acceptance_end)
    return start * island_multiplier, end * island_multiplier, len(losses)


def run(args: argparse.Namespace, MPI: Any) -> None:
    """Collective startup followed by independent annealing islands."""

    validate_args(args)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    stop = StopFlag()

    root_payload: dict[str, object] | None = None
    if rank == 0:
        try:
            root_payload = prepare_root_payload(args)
        except Exception as error:
            root_payload = {
                "error": f"{type(error).__name__}: {error}",
            }
    payload = comm.bcast(root_payload, root=0)
    if payload["error"] is not None:
        raise RuntimeError(f"Rank 0 input validation failed: {payload['error']}")

    tau = float(payload["tau"])
    k = int(payload["k"])
    seed_ids = [canonical_ids(values) for values in payload["seed_ids"]]
    seed_expected_building_ids = [
        tuple(int(value) for value in values)
        for values in payload["seed_expected_building_ids"]
    ]
    seed_paths = [str(value) for value in payload["seed_paths"]]
    island_assignment = assign_island_seed(
        rank=rank,
        world_size=world_size,
        seed_ids=seed_ids,
        seed_expected_building_ids=seed_expected_building_ids,
        seed_paths=seed_paths,
        temperature_levels=args.temperature_profile_levels,
    )
    point_coordinates = {
        int(point_id): (float(xy[0]), float(xy[1]))
        for point_id, xy in payload["point_coordinates"].items()
    }
    official_building_ids = np.ascontiguousarray(
        payload["official_building_ids"],
        dtype=np.int64,
    )
    segments_fingerprint = dict(payload["segments_fingerprint"])
    geojson_fingerprint = dict(payload["geojson_fingerprint"])
    building_id_column = str(payload["building_id_column"])
    proposal_configuration = search_configuration(args)
    if len(seed_ids) > 1:
        # These per-rank values make a multi-seed checkpoint reject a resume
        # whose seed count, ordering, temperature layout, or world size would
        # silently assign that rank to a different island.
        proposal_configuration.update(
            {
                "island_assignment_scheme": "seed_fastest_temperature_next_v1",
                "island_seed_count": len(seed_ids),
                "island_seed_index": island_assignment.seed_index,
                "island_temperature_profile_index": (
                    island_assignment.temperature_profile_index
                ),
                "island_replica_index": island_assignment.replica_index,
            }
        )

    load_started = time.monotonic()
    empty_scorer = load_incremental_scorer(
        args.segments_csv,
        tau,
        args.chunksize,
    )
    empty_scorer.reset()
    all_point_ids = np.unique(
        np.asarray(empty_scorer.point_ids, dtype=np.int64)
    )
    available_set = set(int(value) for value in all_point_ids)
    for checked_seed_index, selected in enumerate(seed_ids):
        missing = set(selected).difference(available_set)
        if missing:
            raise ValueError(
                f"Seed {seed_paths[checked_seed_index]} contains point IDs absent from "
                f"the scorer: {sorted(missing)[:10]}"
            )
    if all_point_ids.size <= k:
        raise ValueError("The scorer has no unselected replacement candidates")

    rank_folder = args.output_folder / f"rank_{rank:04d}"
    if rank_folder.exists() and any(rank_folder.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Rank output is non-empty: {rank_folder}; use --resume or a new folder"
        )
    rank_folder.mkdir(parents=True, exist_ok=True)

    rank_seed = args.random_seed_base + rank
    rng = np.random.default_rng(rank_seed)
    temperature_multiplier = island_temperature_multiplier(
        island_assignment.temperature_profile_index,
        args.temperature_profile_levels,
        args.temperature_log_spread,
    )
    hard_start = (
        hard_temperature_for_loss_acceptance(args.hard_loss_acceptance_start)
        * temperature_multiplier
    )
    hard_end = (
        hard_temperature_for_loss_acceptance(args.hard_loss_acceptance_end)
        * temperature_multiplier
    )

    if args.resume:
        resumed = load_checkpoint(
            rank_folder=rank_folder,
            rank=rank,
            world_size=world_size,
            empty_scorer=empty_scorer,
            point_coordinates=point_coordinates,
            coordinate_tolerance=args.coordinate_tolerance,
            expected_fingerprint=segments_fingerprint,
            expected_geojson_fingerprint=geojson_fingerprint,
            expected_building_id_column=building_id_column,
            expected_proposal_configuration=proposal_configuration,
            expected_k=k,
        )
        current = resumed.current
        best = resumed.best
        iteration = resumed.iteration
        cycle_iteration = resumed.cycle_iteration
        reheat_count = resumed.reheat_count
        last_best_iteration = resumed.last_best_iteration
        checkpoint_serial = resumed.checkpoint_serial
        statistics = resumed.statistics
        rng.bit_generator.state = resumed.rng_state
        calibration_count = 0
        hard_start = resumed.hard_temperature_start
        hard_end = resumed.hard_temperature_end
        soft_start = resumed.soft_temperature_start
        soft_end = resumed.soft_temperature_end
    else:
        current = build_state(empty_scorer, island_assignment.selected_point_ids)
        if int(current.scorer.selected_count) != k:
            raise RuntimeError("Initial state has the wrong antenna count")
        expected_ids = island_assignment.expected_building_ids
        actual_ids = qualifying_official_ids(current.scorer, official_building_ids)
        if actual_ids != expected_ids:
            missing_ids = sorted(set(expected_ids).difference(actual_ids))[:10]
            extra_ids = sorted(set(actual_ids).difference(expected_ids))[:10]
            raise ValueError(
                f"Seed building-ID verification failed for "
                f"{island_assignment.seed_path}: TXT has {len(expected_ids)} IDs, "
                f"the scorer has {len(actual_ids)}; "
                f"missing_from_scorer={missing_ids}, extra_from_scorer={extra_ids}. "
                "The seed, segments, or building-ID mapping do not match."
            )
        best = SearchState(
            scorer=current.scorer.clone(),
            hard_score=current.hard_score,
            soft_score=current.soft_score,
        )
        iteration = 0
        cycle_iteration = 0
        reheat_count = 0
        last_best_iteration = 0
        checkpoint_serial = 0
        statistics = RunStatistics()
        soft_start, soft_end, calibration_count = calibrate_soft_temperatures(
            current=current,
            all_point_ids=all_point_ids,
            args=args,
            island_multiplier=temperature_multiplier,
            rng=rng,
        )

    hard_temperature = geometric_temperature(hard_start, hard_end, 0.0)
    soft_temperature = geometric_temperature(soft_start, soft_end, 0.0)

    if not args.resume:
        checkpoint_serial = next_checkpoint_serial(rank_folder, checkpoint_serial)
        write_rank_best(
            rank_folder=rank_folder,
            state=best,
            point_coordinates=point_coordinates,
            official_building_ids=official_building_ids,
            iteration=iteration,
        )
        write_checkpoint(
            rank_folder=rank_folder,
            serial=checkpoint_serial,
            rank=rank,
            world_size=world_size,
            current=current,
            best=best,
            point_coordinates=point_coordinates,
            official_building_ids=official_building_ids,
            iteration=iteration,
            cycle_iteration=cycle_iteration,
            reheat_count=reheat_count,
            last_best_iteration=last_best_iteration,
            statistics=statistics,
            rng=rng,
            hard_temperature=hard_temperature,
            soft_temperature=soft_temperature,
            hard_temperature_start=hard_start,
            hard_temperature_end=hard_end,
            soft_temperature_start=soft_start,
            soft_temperature_end=soft_end,
            segments_fingerprint=segments_fingerprint,
            geojson_fingerprint=geojson_fingerprint,
            building_id_column=building_id_column,
            proposal_configuration=proposal_configuration,
        )

    print(
        f"[rank {rank}/{world_size}] seed={rank_seed} "
        f"seed_index={island_assignment.seed_index} "
        f"seed_file={island_assignment.seed_path} "
        f"temperature_profile={island_assignment.temperature_profile_index} "
        f"replica={island_assignment.replica_index} tau={tau:g} k={k} "
        f"initial_or_resumed_hard={current.hard_score} "
        f"soft={current.soft_score:.6f} candidates={len(all_point_ids)} "
        f"temperature_multiplier={temperature_multiplier:.4f} "
        f"hard_T={hard_start:.6g}->{hard_end:.6g} "
        f"soft_T={soft_start:.6g}->{soft_end:.6g} "
        f"calibration_negative_samples={calibration_count} "
        f"load_seconds={time.monotonic()-load_started:.1f} "
        f"output={rank_folder}",
        flush=True,
    )

    search_started = time.monotonic()
    last_print = search_started
    last_checkpoint = search_started
    interval_proposed = statistics.proposed
    interval_accepted = statistics.accepted

    while not stop.requested:
        now = time.monotonic()
        if args.max_iterations and iteration >= args.max_iterations:
            break
        if args.max_seconds and now - search_started >= args.max_seconds:
            break

        progress = min(1.0, cycle_iteration / float(args.cycle_iterations))
        hard_temperature = geometric_temperature(hard_start, hard_end, progress)
        soft_temperature = geometric_temperature(soft_start, soft_end, progress)
        proposal_rank_temperature = max(
            0.25,
            args.proposal_rank_temperature
            * temperature_multiplier
            * (1.0 - 0.85 * progress),
        )
        move_name, move_size = choose_move_size(
            selected_count=k,
            one_probability=args.one_swap_probability,
            two_probability=args.two_swap_probability,
            ruin_probability=args.ruin_probability,
            ruin_size=args.ruin_size,
            rng=rng,
        )
        proposal = propose_neighbor(
            current=current,
            all_point_ids=all_point_ids,
            move_name=move_name,
            move_size=move_size,
            candidate_pool_size=args.candidate_pool_size,
            top_candidates=args.top_candidates,
            rank_temperature=proposal_rank_temperature,
            random_candidate_probability=args.random_candidate_probability,
            rng=rng,
        )
        statistics.record_proposal(move_name)
        iteration += 1
        cycle_iteration += 1
        delta_hard = proposal.state.hard_score - current.hard_score
        delta_soft = proposal.state.soft_score - current.soft_score
        accepted = accept_move(
            delta_hard=delta_hard,
            delta_soft=delta_soft,
            hard_temperature=hard_temperature,
            soft_temperature=soft_temperature,
            rng=rng,
        )
        if accepted:
            if delta_hard < 0:
                statistics.accepted_hard_down += 1
            elif delta_hard == 0:
                statistics.accepted_plateau += 1
            else:
                statistics.hard_improvements += 1
            statistics.accepted += 1
            current = proposal.state

            if is_better(
                current.hard_score,
                current.soft_score,
                best.hard_score,
                best.soft_score,
            ):
                hard_best_improved = current.hard_score > best.hard_score
                best = SearchState(
                    scorer=current.scorer.clone(),
                    hard_score=current.hard_score,
                    soft_score=current.soft_score,
                )
                last_best_iteration = iteration
                statistics.best_updates += 1
                if hard_best_improved:
                    write_rank_best(
                        rank_folder=rank_folder,
                        state=best,
                        point_coordinates=point_coordinates,
                        official_building_ids=official_building_ids,
                        iteration=iteration,
                    )
                    print(
                        f"[rank {rank}] NEW_BEST iteration={iteration} "
                        f"hard={best.hard_score} soft={best.soft_score:.6f} "
                        f"move={proposal.move_name} "
                        f"removed={proposal.removed_ids} added={proposal.added_ids} "
                        f"elapsed_seconds={time.monotonic()-search_started:.1f}",
                        flush=True,
                    )

        should_reheat = (
            cycle_iteration >= args.cycle_iterations
            or iteration - last_best_iteration >= args.stagnation_iterations
        )
        if should_reheat:
            current = SearchState(
                scorer=best.scorer.clone(),
                hard_score=best.hard_score,
                soft_score=best.soft_score,
            )
            cycle_iteration = 0
            last_best_iteration = iteration
            reheat_count += 1
            print(
                f"[rank {rank}] REHEAT iteration={iteration} "
                f"restart_hard={best.hard_score} restart_soft={best.soft_score:.6f} "
                f"reheat_count={reheat_count}",
                flush=True,
            )

        now = time.monotonic()
        if now - last_print >= args.print_seconds:
            proposed_delta = statistics.proposed - interval_proposed
            accepted_delta = statistics.accepted - interval_accepted
            acceptance_rate = (
                accepted_delta / proposed_delta if proposed_delta else 0.0
            )
            print(
                f"[rank {rank}] HEARTBEAT iteration={iteration} "
                f"current_hard={current.hard_score} best_hard={best.hard_score} "
                f"current_soft={current.soft_score:.6f} "
                f"best_soft={best.soft_score:.6f} hard_T={hard_temperature:.6g} "
                f"soft_T={soft_temperature:.6g} "
                f"interval_acceptance={acceptance_rate:.4f} "
                f"accepted_down={statistics.accepted_hard_down} "
                f"accepted_plateau={statistics.accepted_plateau} "
                f"reheats={reheat_count} "
                f"elapsed_seconds={now-search_started:.1f}",
                flush=True,
            )
            last_print = now
            interval_proposed = statistics.proposed
            interval_accepted = statistics.accepted

        if now - last_checkpoint >= args.checkpoint_seconds:
            checkpoint_serial = next_checkpoint_serial(
                rank_folder,
                checkpoint_serial + 1,
            )
            checkpoint_folder = write_checkpoint(
                rank_folder=rank_folder,
                serial=checkpoint_serial,
                rank=rank,
                world_size=world_size,
                current=current,
                best=best,
                point_coordinates=point_coordinates,
                official_building_ids=official_building_ids,
                iteration=iteration,
                cycle_iteration=cycle_iteration,
                reheat_count=reheat_count,
                last_best_iteration=last_best_iteration,
                statistics=statistics,
                rng=rng,
                hard_temperature=hard_temperature,
                soft_temperature=soft_temperature,
                hard_temperature_start=hard_start,
                hard_temperature_end=hard_end,
                soft_temperature_start=soft_start,
                soft_temperature_end=soft_end,
                segments_fingerprint=segments_fingerprint,
                geojson_fingerprint=geojson_fingerprint,
                building_id_column=building_id_column,
                proposal_configuration=proposal_configuration,
            )
            # Refresh same-hard soft improvements at the checkpoint cadence.
            write_rank_best(
                rank_folder=rank_folder,
                state=best,
                point_coordinates=point_coordinates,
                official_building_ids=official_building_ids,
                iteration=iteration,
            )
            print(
                f"[rank {rank}] CHECKPOINT iteration={iteration} "
                f"path={checkpoint_folder}",
                flush=True,
            )
            last_checkpoint = now

    # Finish with a complete resumable checkpoint, whether stopped by a time
    # limit or by SIGTERM/SIGINT.
    checkpoint_serial = next_checkpoint_serial(
        rank_folder,
        checkpoint_serial + 1,
    )
    write_rank_best(
        rank_folder=rank_folder,
        state=best,
        point_coordinates=point_coordinates,
        official_building_ids=official_building_ids,
        iteration=iteration,
    )
    final_checkpoint = write_checkpoint(
        rank_folder=rank_folder,
        serial=checkpoint_serial,
        rank=rank,
        world_size=world_size,
        current=current,
        best=best,
        point_coordinates=point_coordinates,
        official_building_ids=official_building_ids,
        iteration=iteration,
        cycle_iteration=cycle_iteration,
        reheat_count=reheat_count,
        last_best_iteration=last_best_iteration,
        statistics=statistics,
        rng=rng,
        hard_temperature=hard_temperature,
        soft_temperature=soft_temperature,
        hard_temperature_start=hard_start,
        hard_temperature_end=hard_end,
        soft_temperature_start=soft_start,
        soft_temperature_end=soft_end,
        segments_fingerprint=segments_fingerprint,
        geojson_fingerprint=geojson_fingerprint,
        building_id_column=building_id_column,
        proposal_configuration=proposal_configuration,
    )
    print(
        f"[rank {rank}] FINISHED iteration={iteration} "
        f"current_hard={current.hard_score} best_hard={best.hard_score} "
        f"best_soft={best.soft_score:.6f} proposed={statistics.proposed} "
        f"accepted={statistics.accepted} reheats={reheat_count} "
        f"elapsed_seconds={time.monotonic()-search_started:.1f} "
        f"best_solution={rank_folder / 'best_solution.txt'} "
        f"checkpoint={final_checkpoint}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    from common.mpi_compat import MPI

    comm = MPI.COMM_WORLD
    try:
        run(args, MPI)
    except BaseException as error:
        print(
            f"[rank {comm.Get_rank()}] fatal: {type(error).__name__}: {error}",
            flush=True,
        )
        comm.Abort(1)
        raise


if __name__ == "__main__":
    main()
