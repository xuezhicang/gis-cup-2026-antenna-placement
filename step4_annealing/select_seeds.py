#!/usr/bin/env python3
"""Choose high-quality, building-diverse solutions as simulated-annealing seeds.

The selection rule is the one used for the competition submission
(``SA_method/select_point_swap_seeds.py``), without its HPC batch checks:

* solutions with identical antenna sets are counted once;
* the best-scoring solution is always selected first;
* each later seed maximises

      utility = 0.6 * normalized_score_quality + 0.4 * building_diversity

  where ``building_diversity`` is one minus the largest Jaccard similarity
  between its serviced-building set and that of any seed already selected;
* only solutions within ``tolerance`` of the best score are considered
  (score >= ceil(best * (1 - tolerance))).  If that leaves too few, the score
  floor is lowered just enough to reach the requested number of seeds;
* point-set diversity is used only to break ties.

Standalone use:

    python step4_annealing/select_seeds.py --seeds 3 solution_1.txt solution_2.txt ...
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.solution_txt import parse_solution_blocks  # noqa: E402


QUALITY_WEIGHT = Fraction(3, 5)
DIVERSITY_WEIGHT = Fraction(2, 5)


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    """One solution file that may become a seed."""

    name: str
    path: Path
    score: int
    building_ids: frozenset[int]
    point_signature: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SeedChoice:
    """A selected seed and the metrics used when it was chosen."""

    candidate: Candidate
    rank: int
    stage: str
    score_quality: Fraction
    building_diversity: Fraction
    point_diversity: Fraction
    utility: Fraction


@dataclass
class SelectionResult:
    seeds: list[SeedChoice]
    distinct_candidates: int
    best_score: int
    requested_floor: int
    effective_floor: int


def load_candidate(path: Path, name: str | None = None) -> Candidate:
    blocks = parse_solution_blocks(path)
    if len(blocks) != 1:
        raise SelectionError(f"Expected one solution block in {path}")
    block = blocks[0]
    signature = tuple(sorted(
        (format(float(x), ".17g"), format(float(y), ".17g")) for x, y in block.coordinates
    ))
    return Candidate(
        name=name or str(path),
        path=path,
        score=len(block.building_ids),
        building_ids=frozenset(int(value) for value in block.building_ids),
        point_signature=signature,
    )


def jaccard(left: frozenset[int], right: frozenset[int]) -> Fraction:
    """Building-set Jaccard similarity, treating two empty sets as equal."""

    union_size = len(left | right)
    if union_size == 0:
        return Fraction(1, 1)
    return Fraction(len(left & right), union_size)


def point_overlap(left: Candidate, right: Candidate) -> Fraction:
    """Fraction of antenna coordinates shared by two solutions."""

    left_points = set(left.point_signature)
    right_points = set(right.point_signature)
    denominator = min(len(left_points), len(right_points))
    if denominator == 0:
        return Fraction(1 if left_points == right_points else 0, 1)
    return Fraction(len(left_points & right_points), denominator)


def deduplicate(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Keep one solution per antenna set, best score first."""

    by_points: dict[tuple[tuple[str, str], ...], list[Candidate]] = {}
    for candidate in candidates:
        by_points.setdefault(candidate.point_signature, []).append(candidate)
    representatives = []
    for same_points in by_points.values():
        ordered = sorted(same_points, key=lambda item: item.name)
        first = ordered[0]
        for duplicate in ordered[1:]:
            if duplicate.score != first.score or duplicate.building_ids != first.building_ids:
                raise SelectionError(
                    "identical antenna sets have different serviced buildings: "
                    f"{first.path} and {duplicate.path}"
                )
        representatives.append(first)
    representatives.sort(key=lambda item: (-item.score, item.name))
    return representatives


def score_floor(best_score: int, tolerance: Decimal) -> int:
    """ceil(best * (1 - tolerance)), computed exactly."""

    floor = Decimal(best_score) * (Decimal(1) - tolerance)
    return int(floor.to_integral_value(rounding=ROUND_CEILING))


def score_quality(score: int, best_score: int, effective_floor: int) -> Fraction:
    """0 at the effective floor, 1 at the best score."""

    denominator = best_score - effective_floor
    if denominator == 0:
        return Fraction(1, 1)
    return Fraction(score - effective_floor, denominator)


def diversity_from_selected(
    candidate: Candidate, selected: Sequence[SeedChoice]
) -> tuple[Fraction, Fraction]:
    """Building and point diversity relative to the closest selected seed."""

    max_building_similarity = max(
        jaccard(candidate.building_ids, step.candidate.building_ids) for step in selected
    )
    max_point_similarity = max(point_overlap(candidate, step.candidate) for step in selected)
    return 1 - max_building_similarity, 1 - max_point_similarity


def select_seeds(
    candidates: Sequence[Candidate],
    num_seeds: int,
    tolerance: float | Decimal = Decimal("0.05"),
    auto_relax: bool = True,
) -> SelectionResult:
    if num_seeds < 1:
        raise SelectionError("num_seeds must be at least 1")
    representatives = deduplicate(candidates)
    if not representatives:
        raise SelectionError("cannot select seeds from an empty list")
    tolerance = Decimal(str(tolerance))
    best_score = representatives[0].score
    requested_floor = score_floor(best_score, tolerance)
    target_count = min(num_seeds, len(representatives))

    effective_floor = requested_floor
    requested_pool = [c for c in representatives if c.score >= requested_floor]
    if auto_relax and len(requested_pool) < target_count:
        effective_floor = min(requested_floor, representatives[target_count - 1].score)
    pool = [c for c in representatives if c.score >= effective_floor]
    wanted = min(target_count, len(pool))

    first = min((c for c in pool if c.score == best_score), key=lambda item: item.name)
    selected = [SeedChoice(first, 1, "best", Fraction(1), Fraction(1), Fraction(1), Fraction(1))]
    remaining = [c for c in pool if c != first]

    while remaining and len(selected) < wanted:
        ranked = []
        for candidate in remaining:
            quality = score_quality(candidate.score, best_score, effective_floor)
            building_diversity, point_diversity = diversity_from_selected(candidate, selected)
            utility = QUALITY_WEIGHT * quality + DIVERSITY_WEIGHT * building_diversity
            ranked.append((candidate, quality, building_diversity, point_diversity, utility))
        # Point diversity is only a tie-break after utility, score and building diversity.
        best_key = max((item[4], item[0].score, item[2], item[3]) for item in ranked)
        tied = [item for item in ranked if (item[4], item[0].score, item[2], item[3]) == best_key]
        candidate, quality, building_diversity, point_diversity, utility = min(
            tied, key=lambda item: item[0].name
        )
        selected.append(SeedChoice(
            candidate, len(selected) + 1, "utility",
            quality, building_diversity, point_diversity, utility,
        ))
        remaining.remove(candidate)

    return SelectionResult(
        seeds=selected,
        distinct_candidates=len(representatives),
        best_score=best_score,
        requested_floor=requested_floor,
        effective_floor=effective_floor,
    )


def report_rows(result: SelectionResult) -> list[str]:
    rows = ["rank\tstage\tscore\tquality\tbuilding_diversity\tutility\tname\tpath"]
    for seed in result.seeds:
        rows.append(
            f"{seed.rank}\t{seed.stage}\t{seed.candidate.score}\t"
            f"{float(seed.score_quality):.4f}\t{float(seed.building_diversity):.4f}\t"
            f"{float(seed.utility):.4f}\t{seed.candidate.name}\t{seed.candidate.path}"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("solutions", type=Path, nargs="+", help="Three-line solution files")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--output-tsv", type=Path, help="Optional report file")
    args = parser.parse_args(argv)

    result = select_seeds([load_candidate(path) for path in args.solutions], args.seeds, args.tolerance)
    rows = report_rows(result)
    print("\n".join(rows))
    print(f"\n{len(result.seeds)} seed(s) from {result.distinct_candidates} distinct solution(s); "
          f"score floor {result.effective_floor} (best {result.best_score})")
    if args.output_tsv:
        args.output_tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
