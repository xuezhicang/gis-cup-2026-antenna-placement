from __future__ import annotations

import argparse
import heapq
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd

import coverage_scorer_cpp
from common.scorer_loader import load_incremental_scorer


DEFAULT_OUTPUT_CSV = Path("outputs/two_step_greedy_selection.csv")


@dataclass(frozen=True)
class CandidateScore:
    point_id: int
    combined_score: float
    soft_score: float
    hard_score: int
    unique_length: float


@dataclass(frozen=True)
class TwoStepChoice:
    first: CandidateScore
    second: CandidateScore | None
    pair_combined_score: float
    pair_soft_score: float
    pair_hard_score: int
    pair_unique_length: float


def candidate_key(score: CandidateScore) -> tuple[float, float, int, float, int]:
    return (
        score.combined_score,
        score.soft_score,
        score.hard_score,
        score.unique_length,
        -score.point_id,
    )


def two_step_key(choice: TwoStepChoice) -> tuple[float, float, int, float, tuple]:
    return (
        choice.pair_combined_score,
        choice.pair_soft_score,
        choice.pair_hard_score,
        choice.pair_unique_length,
        candidate_key(choice.first),
    )


def score_candidates(
    scorer: coverage_scorer_cpp.IncrementalCoverageScorer,
    candidate_ids: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    if candidate_ids.size == 0:
        raise ValueError("candidate_ids is empty")

    batch = scorer.evaluate_all(candidate_ids)
    hard_scores = np.asarray(batch["hard_scores"], dtype=np.float64)
    soft_scores = np.asarray(batch["soft_scores"], dtype=np.float64)
    unique_lengths = np.asarray(batch["unique_lengths"], dtype=np.float64)
    point_indices = np.asarray(batch["point_indices"], dtype=np.int64)

    max_hard = float(np.max(hard_scores))
    max_soft = float(np.max(soft_scores))
    normalized_hard = (
        hard_scores / max_hard
        if max_hard > 0.0
        else np.zeros_like(hard_scores)
    )
    normalized_soft = (
        soft_scores / max_soft
        if max_soft > 0.0
        else np.zeros_like(soft_scores)
    )
    combined_scores = alpha * normalized_hard + (1.0 - alpha) * normalized_soft

    return {
        "point_indices": point_indices,
        "combined_scores": combined_scores,
        "soft_scores": soft_scores,
        "hard_scores": hard_scores,
        "unique_lengths": unique_lengths,
    }


def candidate_from_batch(
    batch: dict[str, np.ndarray],
    index: int,
) -> CandidateScore:
    return CandidateScore(
        point_id=int(batch["point_indices"][index]),
        combined_score=float(batch["combined_scores"][index]),
        soft_score=float(batch["soft_scores"][index]),
        hard_score=int(batch["hard_scores"][index]),
        unique_length=float(batch["unique_lengths"][index]),
    )


def top_candidate_scores(
    scorer: coverage_scorer_cpp.IncrementalCoverageScorer,
    candidate_ids: np.ndarray,
    alpha: float,
    limit: int,
) -> list[CandidateScore]:
    '''
    Returns the top candidate scores for the given set of candidate IDs.
    '''
    batch = score_candidates(scorer, candidate_ids, alpha)
    top_count = min(limit, candidate_ids.size)

    def batch_key(index: int) -> tuple[float, float, int, float, int]:
        return (
            float(batch["combined_scores"][index]),
            float(batch["soft_scores"][index]),
            int(batch["hard_scores"][index]),
            float(batch["unique_lengths"][index]),
            -int(batch["point_indices"][index]),
        )

    top_positions = heapq.nlargest(
        top_count,
        range(candidate_ids.size),
        key=batch_key,
    )
    return [candidate_from_batch(batch, index) for index in top_positions]

def choose_with_two_step_lookahead(
    scorer: coverage_scorer_cpp.IncrementalCoverageScorer,
    candidate_ids: np.ndarray,
    alpha: float,
    lookahead_width: int,
) -> TwoStepChoice:
    '''
    Chooses the best candidate point using a two-step lookahead approach.
    '''
    # Get the top candidate scores for the first step
    first_candidates = top_candidate_scores(
        scorer,
        candidate_ids,
        alpha,
        lookahead_width,
    )

    choices: list[TwoStepChoice] = []
    for first in first_candidates:
        trial = scorer.clone()
        trial.add_point(first.point_id)

        second_candidate_ids = candidate_ids[candidate_ids != first.point_id]
        second_candidates = (
            top_candidate_scores(trial, second_candidate_ids, alpha, lookahead_width)
            if second_candidate_ids.size > 0
            else []
        )
        # second = second_candidates[0] if second_candidates else None

        if not second_candidates:
            choices.append(
                TwoStepChoice(
                    first=first,
                    second=None,
                    pair_combined_score=0.0,
                    pair_soft_score=first.soft_score,
                    pair_hard_score=first.hard_score,
                    pair_unique_length=first.unique_length,
                )
            )
            continue

        for second in second_candidates:
            choice = TwoStepChoice(
                first=first,
                second=second,
                pair_combined_score=0.0,
                pair_soft_score=first.soft_score
                + (second.soft_score if second else 0.0),
                pair_hard_score=first.hard_score
                + (second.hard_score if second else 0),
                pair_unique_length=first.unique_length
                + (second.unique_length if second else 0.0),
            )
            choices.append(choice)

    if not choices:
        raise RuntimeError("No candidate point could be selected.")

    max_pair_soft = max(choice.pair_soft_score for choice in choices)
    max_pair_hard = max(choice.pair_hard_score for choice in choices)

    def get_pair_combined_score(choice: TwoStepChoice) -> float:
        normalized_soft = (
            choice.pair_soft_score / max_pair_soft
            if max_pair_soft > 0.0
            else 0.0
        )
        normalized_hard = (
            choice.pair_hard_score / max_pair_hard
            if max_pair_hard > 0
            else 0.0
        )
        return alpha * normalized_hard + (1.0 - alpha) * normalized_soft

    best_choice = max(
        choices,
        key=lambda choice: (
            get_pair_combined_score(choice),
            choice.pair_soft_score,
            choice.pair_hard_score,
            choice.pair_unique_length,
            candidate_key(choice.first),
        ),
    )

    return TwoStepChoice(
        first=best_choice.first,
        second=best_choice.second,
        pair_combined_score=get_pair_combined_score(best_choice),
        pair_soft_score=best_choice.pair_soft_score,
        pair_hard_score=best_choice.pair_hard_score,
        pair_unique_length=best_choice.pair_unique_length,
    )


def sorted_remaining_ids(remaining_point_ids: set[int]) -> np.ndarray:
    return np.asarray(sorted(remaining_point_ids), dtype=np.int64)


def run_two_step_greedy(
    scorer: coverage_scorer_cpp.IncrementalCoverageScorer,
    *,
    alpha: float,
    antenna_count: int,
    lookahead_width: int,
    print_every: int,
) -> list[dict]:
    scorer.reset()
    remaining_point_ids = set(int(point_id) for point_id in scorer.point_ids)
    rows: list[dict] = []

    for iteration in range(1, antenna_count + 1):
        if not remaining_point_ids:
            print("No candidate points remain.")
            break

        iteration_start = time.perf_counter()
        candidate_ids = sorted_remaining_ids(remaining_point_ids)
        choice = choose_with_two_step_lookahead(
            scorer,
            candidate_ids,
            alpha,
            lookahead_width,
        )

        scorer.add_point(choice.first.point_id)
        remaining_point_ids.remove(choice.first.point_id)
        iteration_seconds = time.perf_counter() - iteration_start

        second_point_id = choice.second.point_id if choice.second else None
        rows.append(
            {
                "alpha": alpha,
                "iteration": iteration,
                "selected_point_id": choice.first.point_id,
                "lookahead_second_point_id": second_point_id,
                "first_combined_score": choice.first.combined_score,
                "first_soft_score": choice.first.soft_score,
                "first_hard_score": choice.first.hard_score,
                "first_unique_length": choice.first.unique_length,
                "pair_combined_score": choice.pair_combined_score,
                "pair_soft_score": choice.pair_soft_score,
                "pair_hard_score": choice.pair_hard_score,
                "pair_unique_length": choice.pair_unique_length,
                "current_score": int(scorer.current_score),
                "selected_count": int(scorer.selected_count),
                "candidate_count": int(candidate_ids.size),
                "lookahead_width": min(lookahead_width, int(candidate_ids.size)),
                "iteration_seconds": iteration_seconds,
            }
        )

        if print_every > 0 and iteration % print_every == 0:
            print(
                f"tau={scorer.tau:.3f} "
                f"k={antenna_count} "
                f"alpha={alpha:.3f} "
                f"lookahead={lookahead_width} "
                f"iteration={iteration} "
                f"selected={choice.first.point_id} "
                f"next={second_point_id} "
                f"score={scorer.current_score} "
                f"time={iteration_seconds:.3f}s",
                flush=True,
            )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a readable 2-step greedy antenna selection. Each iteration "
            "tests the current top-N first points, looks one point ahead from "
            "each clone, and commits only the best first point."
        )
    )
    parser.add_argument("--segments-csv", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--antenna-count", type=int, default=50)
    parser.add_argument(
        "--lookahead-width",
        type=int,
        default=100,
        help="How many current top first-points to test with clone lookahead.",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[1.0],
        help="One or more alpha values in [0, 1]. Default: 1.0.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Print progress every N selected points. Default: 100.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.antenna_count <= 0:
        raise ValueError("--antenna-count must be greater than zero")
    if args.lookahead_width <= 0:
        raise ValueError("--lookahead-width must be greater than zero")
    for alpha in args.alpha_values:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("--alpha-values must all be in [0, 1]")


def main() -> None:
    args = parse_args()
    validate_args(args)

    start = time.perf_counter()
    scorer = load_incremental_scorer(args.segments_csv, args.tau, args.chunksize)
    # print("load csv (s):", round(time.perf_counter() - start, 3))
    # print("candidate points:", scorer.num_points)
    # print("tau:", scorer.tau)
    # print("antenna count:", args.antenna_count)
    # print("lookahead width:", args.lookahead_width)

    all_rows: list[dict] = []
    final_summaries: list[str] = []
    for alpha in args.alpha_values:
        rows = run_two_step_greedy(
            scorer,
            alpha=alpha,
            antenna_count=args.antenna_count,
            lookahead_width=args.lookahead_width,
            print_every=args.print_every,
        )
        all_rows.extend(rows)
        final_summaries.append(
            f"final: tau={args.tau} "
            f"antenna_number={args.antenna_count} "
            f"alpha={alpha} "
            f"lookahead_width={args.lookahead_width} "
            f"score={scorer.current_score} "
            f"selected_points={scorer.selected_count}"
        )

    result_df = pd.DataFrame(all_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_csv, index=False)
    print("selection log csv:", args.output_csv)
    for summary in final_summaries:
        print(summary)
        print()


if __name__ == "__main__":
    main()
