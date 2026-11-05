#!/usr/bin/env python3
"""MPI beam search using exact one-point swaps.

Each beam state contains exactly ``k`` selected antenna points.  Expanding one
state means trying every selected point as the point to remove.  For each
removal, the program scans every legal, currently unselected candidate and
keeps the best replacement.  MPI distributes the removal points across ranks.

At the end of a generation, rank 0 keeps at most ``--beam-width`` distinct,
strictly improving children.  Those children form the next generation.  With
``--beam-width 1`` this is deterministic best-improvement one-point search;
larger values retain several promising search branches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

from common.scorer_loader import load_incremental_scorer
from common.search_common import (
    StopFlag,
    match_solution_points,
    read_point_coordinates,
    read_solution,
    write_best_solution,
)
from common.solution_txt import load_official_building_ids


@dataclass(frozen=True)
class BeamState:
    """A scored solution represented by canonical, sorted point IDs."""

    selected_ids: tuple[int, ...]
    score: int


@dataclass(frozen=True)
class SwapCandidate:
    """One strictly improving child reported by an MPI rank."""

    parent_index: int
    removed_id: int
    replacement_id: int
    score: int


@dataclass(frozen=True)
class SelectedChild:
    """A child retained by rank 0 for the next beam."""

    state: BeamState
    swap: SwapCandidate


@dataclass(frozen=True)
class SelectionStatistics:
    """Counts useful for progress reporting."""

    reported_candidates: int
    distinct_unseen_children: int


def canonical_point_ids(values: Iterable[int]) -> tuple[int, ...]:
    """Return a sorted solution identity and reject duplicate point IDs."""

    point_ids = tuple(sorted(int(value) for value in values))
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("A solution contains duplicate point IDs")
    return point_ids


def swapped_point_ids(
    selected_ids: Sequence[int],
    removed_id: int,
    replacement_id: int,
) -> tuple[int, ...]:
    """Return the canonical child produced by one real point swap."""

    selected = set(int(value) for value in selected_ids)
    removed_id = int(removed_id)
    replacement_id = int(replacement_id)

    if removed_id not in selected:
        raise ValueError(f"Removed point {removed_id} is not selected")
    if replacement_id in selected:
        raise ValueError(f"Replacement point {replacement_id} is already selected")
    if replacement_id == removed_id:
        raise ValueError("A swap must use a different replacement point")

    selected.remove(removed_id)
    selected.add(replacement_id)
    child = canonical_point_ids(selected)
    if len(child) != len(selected_ids):
        raise RuntimeError("A one-point swap changed the antenna count")
    return child


def select_next_beam(
    current_beam: Sequence[BeamState],
    candidates: Iterable[SwapCandidate],
    beam_width: int,
    seen_states: set[tuple[int, ...]],
) -> tuple[list[SelectedChild], SelectionStatistics]:
    """Select the highest-scoring distinct, unseen children.

    A child only has to improve its own parent.  Requiring it to beat the
    global best would collapse the search toward beam width one and discard
    useful alternative branches.
    """

    if beam_width <= 0:
        raise ValueError("beam_width must be greater than zero")

    reported = 0
    by_identity: dict[tuple[int, ...], SelectedChild] = {}

    for candidate in candidates:
        reported += 1
        if not 0 <= candidate.parent_index < len(current_beam):
            raise ValueError(
                f"Invalid parent index {candidate.parent_index} for "
                f"beam size {len(current_beam)}"
            )

        parent = current_beam[candidate.parent_index]
        if candidate.score <= parent.score:
            continue

        child_ids = swapped_point_ids(
            parent.selected_ids,
            candidate.removed_id,
            candidate.replacement_id,
        )
        if child_ids in seen_states:
            continue

        child = SelectedChild(
            state=BeamState(child_ids, int(candidate.score)),
            swap=candidate,
        )
        previous = by_identity.get(child_ids)
        if previous is not None and previous.state.score != child.state.score:
            raise RuntimeError("The same point set was reported with different scores")

        # Multiple swaps can lead to the same set.  Keep a deterministic
        # provenance record so gathering order cannot change the result.
        if previous is None or swap_order_key(child.swap) < swap_order_key(
            previous.swap
        ):
            by_identity[child_ids] = child

    distinct_count = len(by_identity)
    ranked = sorted(
        by_identity.values(),
        key=lambda child: (
            -child.state.score,
            child.state.selected_ids,
            swap_order_key(child.swap),
        ),
    )
    selected = ranked[:beam_width]
    return selected, SelectionStatistics(reported, distinct_count)


def swap_order_key(candidate: SwapCandidate) -> tuple[int, int, int]:
    """Stable tie-breaking for equivalent child descriptions."""

    return (
        int(candidate.parent_index),
        int(candidate.removed_id),
        int(candidate.replacement_id),
    )


def build_scorer(empty_scorer: Any, state: BeamState) -> Any:
    """Reconstruct one beam state while sharing the scorer's immutable data."""

    scorer = empty_scorer.clone()
    scorer.reset()
    scorer.add_points(np.asarray(state.selected_ids, dtype=np.int64))
    if int(scorer.selected_count) != len(state.selected_ids):
        raise RuntimeError("Reconstructed scorer has the wrong selected-point count")
    if int(scorer.current_score) != state.score:
        raise RuntimeError(
            f"Reconstructed score {scorer.current_score} does not match "
            f"recorded score {state.score}"
        )
    return scorer


def best_replacement(
    scorer_after_removal: Any,
    candidate_ids: np.ndarray,
) -> tuple[int, int] | None:
    """Return ``(point_id, hard_gain)`` using deterministic tie-breaking."""

    if candidate_ids.size == 0:
        return None

    gains = np.asarray(
        scorer_after_removal.evaluate_all_hard(candidate_ids),
        dtype=np.int64,
    )
    if gains.shape != candidate_ids.shape:
        raise RuntimeError("Scorer returned an unexpected candidate-score shape")

    maximum_gain = int(gains.max())
    best_ids = candidate_ids[gains == maximum_gain]
    return int(best_ids.min()), maximum_gain


def expand_local_beam(
    empty_scorer: Any,
    all_point_ids: np.ndarray,
    beam: Sequence[BeamState],
    rank: int,
    world_size: int,
    stop: StopFlag,
) -> tuple[list[SwapCandidate], int]:
    """Expand this rank's share of all ``(parent, removed point)`` tasks."""

    local_children: list[SwapCandidate] = []
    completed_tasks = 0

    for parent_index, state in enumerate(beam):
        if stop.requested:
            break

        parent_scorer = build_scorer(empty_scorer, state)
        selected = np.asarray(state.selected_ids, dtype=np.int64)

        # The removed point is deliberately excluded too, which guarantees a
        # real swap instead of immediately putting the same point back.
        replacement_candidates = np.setdiff1d(
            all_point_ids,
            selected,
            assume_unique=True,
        )

        for removed_id in state.selected_ids[rank::world_size]:
            if stop.requested:
                break

            completed_tasks += 1
            trial = parent_scorer.clone()
            trial.remove_point(int(removed_id))

            replacement = best_replacement(trial, replacement_candidates)
            if replacement is None:
                continue
            replacement_id, hard_gain = replacement
            expected_score = int(trial.current_score) + hard_gain
            if expected_score <= state.score:
                continue

            trial.add_point(replacement_id)
            actual_score = int(trial.current_score)
            if actual_score != expected_score:
                raise RuntimeError(
                    f"Predicted score {expected_score}, obtained {actual_score} "
                    f"for swap {removed_id}->{replacement_id}"
                )
            if int(trial.selected_count) != len(state.selected_ids):
                raise RuntimeError("A trial swap changed the selected-point count")

            local_children.append(
                SwapCandidate(
                    parent_index=parent_index,
                    removed_id=int(removed_id),
                    replacement_id=replacement_id,
                    score=actual_score,
                )
            )

    return local_children, completed_tasks


def atomic_write_json(path: Path, value: object) -> None:
    """Atomically replace a small JSON metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def best_state(states: Sequence[BeamState]) -> BeamState:
    """Return the best state with deterministic score-tie behavior."""

    if not states:
        raise ValueError("Cannot choose the best state from an empty sequence")
    return min(states, key=lambda state: (-state.score, state.selected_ids))


def write_best_checkpoint(
    *,
    output_folder: Path,
    generation: int,
    best_ever: BeamState,
    empty_scorer: Any,
    point_coordinates: dict[int, tuple[float, float]],
    official_building_ids: np.ndarray,
) -> None:
    """Atomically replace each file in the small rolling best checkpoint."""

    scorer = build_scorer(empty_scorer, best_ever)
    write_best_solution(
        scorer,
        point_coordinates,
        official_building_ids,
        output_folder / "best_solution.txt",
    )
    atomic_write_json(
        output_folder / "best_solution.json",
        {
            "generation": generation,
            "score": best_ever.score,
            "selected_count": len(best_ever.selected_ids),
        },
    )


def write_generation_checkpoint(
    *,
    output_folder: Path,
    generation: int,
    beam: Sequence[BeamState],
    transitions: Sequence[SwapCandidate | None],
    empty_scorer: Any,
    point_coordinates: dict[int, tuple[float, float]],
    official_building_ids: np.ndarray,
    best_ever: BeamState,
    write_best_ever: bool,
) -> None:
    """Write one immutable, complete generation and update ``latest.json``."""

    if len(beam) != len(transitions):
        raise ValueError("Every beam state needs one transition record")

    generation_name = f"generation_{generation:06d}"
    generation_folder = output_folder / generation_name
    if generation_folder.exists():
        raise FileExistsError(f"Checkpoint already exists: {generation_folder}")
    generation_folder.mkdir(parents=True)

    members: list[dict[str, object]] = []
    for beam_index, (state, transition) in enumerate(zip(beam, transitions)):
        filename = f"beam_{beam_index:03d}_score_{state.score}.txt"
        scorer = build_scorer(empty_scorer, state)
        write_best_solution(
            scorer,
            point_coordinates,
            official_building_ids,
            generation_folder / filename,
        )
        members.append(
            {
                "beam_index": beam_index,
                "score": state.score,
                "selected_count": len(state.selected_ids),
                "file": filename,
                "parent_index": (
                    None if transition is None else transition.parent_index
                ),
                "removed_id": None if transition is None else transition.removed_id,
                "replacement_id": (
                    None if transition is None else transition.replacement_id
                ),
            }
        )

    metadata = {
        "generation": generation,
        "beam_size": len(beam),
        "best_in_generation": best_state(beam).score,
        "best_ever": best_ever.score,
        "members": members,
    }
    atomic_write_json(generation_folder / "metadata.json", metadata)

    # Update best_solution only when its score/identity changes.
    if write_best_ever:
        write_best_checkpoint(
            output_folder=output_folder,
            generation=generation,
            best_ever=best_ever,
            empty_scorer=empty_scorer,
            point_coordinates=point_coordinates,
            official_building_ids=official_building_ids,
        )

    # This pointer is replaced only after every solution and the generation
    # metadata have been written.  An interrupted generation therefore leaves
    # the preceding checkpoint as the latest complete one.
    atomic_write_json(
        output_folder / "latest.json",
        {
            "generation": generation,
            "checkpoint": generation_name,
            "metadata": f"{generation_name}/metadata.json",
            "best_ever": best_ever.score,
        },
    )


def write_search_checkpoint(
    *,
    checkpoint_mode: str,
    output_folder: Path,
    generation: int,
    beam: Sequence[BeamState],
    transitions: Sequence[SwapCandidate | None],
    empty_scorer: Any,
    point_coordinates: dict[int, tuple[float, float]],
    official_building_ids: np.ndarray,
    best_ever: BeamState,
    write_best_ever: bool,
) -> None:
    """Write either full generation history or only the rolling best state."""

    if checkpoint_mode == "full":
        write_generation_checkpoint(
            output_folder=output_folder,
            generation=generation,
            beam=beam,
            transitions=transitions,
            empty_scorer=empty_scorer,
            point_coordinates=point_coordinates,
            official_building_ids=official_building_ids,
            best_ever=best_ever,
            write_best_ever=write_best_ever,
        )
        return
    if checkpoint_mode == "best-only":
        if write_best_ever:
            write_best_checkpoint(
                output_folder=output_folder,
                generation=generation,
                best_ever=best_ever,
                empty_scorer=empty_scorer,
                point_coordinates=point_coordinates,
                official_building_ids=official_building_ids,
            )
        return
    raise ValueError(f"Unsupported checkpoint mode: {checkpoint_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use MPI to expand every one-point removal and retain a "
            "configurable beam of strictly improving solutions."
        )
    )
    parser.add_argument(
        "--input-result",
        type=Path,
        required=True,
        help=(
            "Initial three-line solution TXT. Its coordinates are rematched "
            "to candidate point IDs and its score is recomputed."
        ),
    )
    parser.add_argument(
        "--segments-csv",
        type=Path,
        required=True,
        help="Visible-segment table containing candidate point IDs and coordinates.",
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        required=True,
        help="Building GeoJSON used to write official building IDs.",
    )
    parser.add_argument("--building-id-column", default="id")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Number of distinct improving states retained per generation.",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=0,
        help="Maximum improving generations; zero means run until convergence.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Wall-clock limit for the search (after loading); zero means no limit.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("full", "best-only"),
        default="best-only",
        help=(
            "'best-only' (default) atomically keeps only best_solution.txt/json "
            "and writes no generation folders; 'full' writes every beam generation."
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="A new or empty directory for checkpoints and the best solution.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be greater than zero")
    if args.max_generations < 0:
        raise ValueError("--max-generations cannot be negative")
    if not np.isfinite(args.max_seconds) or args.max_seconds < 0.0:
        raise ValueError("--max-seconds must be finite and non-negative")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be greater than zero")
    if (
        not np.isfinite(args.coordinate_tolerance)
        or args.coordinate_tolerance < 0.0
    ):
        raise ValueError("--coordinate-tolerance must be finite and non-negative")


def prepare_root_inputs(args: argparse.Namespace) -> tuple[
    float,
    tuple[int, ...],
    dict[int, tuple[float, float]],
    np.ndarray,
]:
    """Read root-only inputs and match solution coordinates to point IDs."""

    for path, description in (
        (args.input_result, "input solution"),
        (args.segments_csv, "segment table"),
        (args.geojson, "building GeoJSON"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    if args.output_folder.exists() and any(args.output_folder.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {args.output_folder}. "
            "Use a new directory to protect existing checkpoints."
        )
    args.output_folder.mkdir(parents=True, exist_ok=True)

    solution = read_solution(args.input_result)
    point_coordinates = read_point_coordinates(args.segments_csv, args.chunksize)
    selected_ids = canonical_point_ids(
        match_solution_points(
            solution.coordinates,
            point_coordinates,
            args.coordinate_tolerance,
        )
    )
    if not selected_ids:
        raise ValueError("The initial solution must contain at least one point")
    official_building_ids = load_official_building_ids(
        args.geojson,
        args.building_id_column,
    )
    return (
        float(solution.tau),
        selected_ids,
        point_coordinates,
        official_building_ids,
    )


def run(args: argparse.Namespace, MPI: Any) -> None:
    """Collective MPI driver."""

    validate_args(args)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    stop = StopFlag()

    root_coordinates: dict[int, tuple[float, float]] | None = None
    root_official_ids: np.ndarray | None = None
    root_payload: dict[str, object] | None = None

    if rank == 0:
        try:
            (
                tau,
                initial_ids,
                root_coordinates,
                root_official_ids,
            ) = prepare_root_inputs(args)
            root_payload = {
                "error": None,
                "tau": tau,
                "initial_ids": initial_ids,
            }
        except Exception as error:  # distributed startup must fail collectively
            root_payload = {
                "error": f"{type(error).__name__}: {error}",
            }

    root_payload = comm.bcast(root_payload, root=0)
    if root_payload["error"] is not None:
        raise RuntimeError(f"Rank 0 input setup failed: {root_payload['error']}")

    tau = float(root_payload["tau"])
    initial_ids = canonical_point_ids(root_payload["initial_ids"])

    load_started = time.monotonic()
    empty_scorer = load_incremental_scorer(args.segments_csv, tau, args.chunksize)
    empty_scorer.reset()
    all_point_ids = np.unique(
        np.asarray(empty_scorer.point_ids, dtype=np.int64)
    )
    missing = set(initial_ids).difference(int(value) for value in all_point_ids)
    if missing:
        raise ValueError(
            "Initial solution contains point IDs absent from the scorer: "
            + ", ".join(str(value) for value in sorted(missing)[:10])
        )

    initial_scorer = empty_scorer.clone()
    initial_scorer.add_points(np.asarray(initial_ids, dtype=np.int64))
    initial_score = int(initial_scorer.current_score)
    del initial_scorer

    rank_scores = comm.allgather(initial_score)
    if len(set(rank_scores)) != 1:
        raise RuntimeError(f"MPI ranks disagree on initial score: {rank_scores}")

    beam = [BeamState(initial_ids, initial_score)]
    seen_states = {initial_ids}
    best_ever = beam[0]
    generation = 0
    search_started = time.monotonic()
    if args.max_seconds > 0:
        stop.deadline = search_started + args.max_seconds

    if rank == 0:
        assert root_coordinates is not None
        assert root_official_ids is not None
        write_search_checkpoint(
            checkpoint_mode=args.checkpoint_mode,
            output_folder=args.output_folder,
            generation=0,
            beam=beam,
            transitions=[None],
            empty_scorer=empty_scorer,
            point_coordinates=root_coordinates,
            official_building_ids=root_official_ids,
            best_ever=best_ever,
            write_best_ever=True,
        )
        print(
            f"MPI ranks: {world_size}\n"
            f"tau={tau:g} k={len(initial_ids)} initial_score={initial_score}\n"
            f"candidate_points={len(all_point_ids)} beam_width={args.beam_width} "
            f"max_generations={args.max_generations} "
            f"checkpoint_mode={args.checkpoint_mode}\n"
            f"load_seconds={time.monotonic()-load_started:.1f}\n"
            f"output_folder={args.output_folder}",
            flush=True,
        )

    comm.Barrier()

    while args.max_generations == 0 or generation < args.max_generations:
        generation_started = time.monotonic()
        parent_count = len(beam)
        expected_tasks = sum(len(state.selected_ids) for state in beam)
        local_candidates, local_completed = expand_local_beam(
            empty_scorer,
            all_point_ids,
            beam,
            rank,
            world_size,
            stop,
        )

        gathered_candidates = comm.gather(local_candidates, root=0)
        completed_tasks = comm.reduce(local_completed, op=MPI.SUM, root=0)
        signal_anywhere = comm.allreduce(bool(stop.signal_received), op=MPI.LOR)
        time_up_anywhere = comm.allreduce(bool(stop.time_up), op=MPI.LOR)
        if signal_anywhere:
            if rank == 0:
                preserved_output = (
                    "The rolling best solution is unchanged."
                    if args.checkpoint_mode == "best-only"
                    else "The latest completed checkpoint is unchanged."
                )
                print(
                    "Stop requested; discarding the incomplete generation. "
                    + preserved_output,
                    flush=True,
                )
            break
        if time_up_anywhere:
            # Every reported swap is an exact, strictly improving child, so a
            # partially expanded generation can still be used before stopping.
            if rank == 0:
                print(
                    f"Time limit of {args.max_seconds:g}s reached; keeping the "
                    "improving swaps found so far in this generation.",
                    flush=True,
                )
        elif rank == 0 and completed_tasks != expected_tasks:
            raise RuntimeError(
                f"MPI completed {completed_tasks} removal tasks, expected "
                f"{expected_tasks}"
            )

        selection_payload: dict[str, object] | None = None
        selected_children: list[SelectedChild] = []
        if rank == 0:
            flat_candidates = [
                candidate
                for rank_candidates in gathered_candidates
                for candidate in rank_candidates
            ]
            selected_children, statistics = select_next_beam(
                beam,
                flat_candidates,
                args.beam_width,
                seen_states,
            )
            selection_payload = {
                "states": [child.state for child in selected_children],
                "reported_candidates": statistics.reported_candidates,
                "distinct_unseen_children": statistics.distinct_unseen_children,
            }

        selection_payload = comm.bcast(selection_payload, root=0)
        next_beam = list(selection_payload["states"])
        if not next_beam:
            if rank == 0:
                reason = "time limit reached" if time_up_anywhere else "converged"
                print(
                    f"generation={generation + 1} completed_tasks={completed_tasks} "
                    f"improving_children=0; {reason}",
                    flush=True,
                )
            break

        beam = next_beam
        generation += 1

        if rank == 0:
            for state in beam:
                seen_states.add(state.selected_ids)

            previous_best = best_ever
            generation_best = best_state(beam)
            if generation_best.score > best_ever.score:
                best_ever = generation_best
            best_changed = best_ever != previous_best

            assert root_coordinates is not None
            assert root_official_ids is not None
            write_search_checkpoint(
                checkpoint_mode=args.checkpoint_mode,
                output_folder=args.output_folder,
                generation=generation,
                beam=beam,
                transitions=[child.swap for child in selected_children],
                empty_scorer=empty_scorer,
                point_coordinates=root_coordinates,
                official_building_ids=root_official_ids,
                best_ever=best_ever,
                write_best_ever=best_changed,
            )
            print(
                f"generation={generation} parents_expanded={parent_count} "
                f"completed_tasks={completed_tasks} "
                f"improving_swaps={selection_payload['reported_candidates']} "
                f"distinct_unseen={selection_payload['distinct_unseen_children']} "
                f"next_beam={len(beam)} best_in_beam={generation_best.score} "
                f"best_ever={best_ever.score} "
                f"generation_seconds={time.monotonic()-generation_started:.1f}",
                flush=True,
            )

        # Rank 0 owns the visited-state archive; other ranks only need the
        # current beam, which arrives in the next collective expansion.
        comm.Barrier()
        if time_up_anywhere:
            break

    if rank == 0:
        print(
            f"Finished: generations={generation} best_score={best_ever.score} "
            f"elapsed_seconds={time.monotonic()-search_started:.1f} "
            f"best_solution={args.output_folder / 'best_solution.txt'}",
            flush=True,
        )


def main() -> None:
    args = parse_args()

    # Importing MPI here keeps the pure selection helpers easy to unit test.
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
