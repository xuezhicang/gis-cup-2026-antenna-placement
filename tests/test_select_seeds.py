from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.solution_txt import write_solution_block  # noqa: E402
from step4_annealing.select_seeds import load_candidate, select_seeds  # noqa: E402


class SelectSeedsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self) -> None:
        self.folder.cleanup()

    def solution(self, name: str, points: list[tuple[float, float]], buildings: list[int]):
        path = self.root / f"{name}.txt"
        write_solution_block(path, 0.5, points, buildings)
        return load_candidate(path, name)

    def test_best_first_then_diverse_over_similar(self) -> None:
        best = self.solution("best", [(0, 0), (1, 1)], list(range(1, 11)))
        similar = self.solution("similar", [(0, 0), (2, 2)], list(range(1, 11)))      # same buildings
        diverse = self.solution("diverse", [(5, 5), (6, 6)], list(range(11, 21)))     # other buildings
        result = select_seeds([similar, diverse, best], num_seeds=2, tolerance=0.05)
        self.assertEqual([seed.candidate.name for seed in result.seeds], ["best", "diverse"])
        self.assertEqual(result.seeds[1].building_diversity, 1)

    def test_identical_antenna_sets_count_once(self) -> None:
        first = self.solution("a", [(0, 0), (1, 1)], [1, 2, 3])
        copy = self.solution("b", [(1, 1), (0, 0)], [1, 2, 3])
        result = select_seeds([first, copy], num_seeds=2)
        self.assertEqual(result.distinct_candidates, 1)
        self.assertEqual(len(result.seeds), 1)

    def test_tolerance_is_relaxed_only_when_needed(self) -> None:
        best = self.solution("best", [(0, 0)], list(range(100)))
        weaker = self.solution("weaker", [(9, 9)], list(range(200, 280)))   # 20% below best
        strict = select_seeds([best, weaker], num_seeds=1, tolerance=0.05)
        self.assertEqual(strict.effective_floor, 95)
        relaxed = select_seeds([best, weaker], num_seeds=2, tolerance=0.05)
        self.assertEqual(relaxed.effective_floor, 80)
        self.assertEqual(len(relaxed.seeds), 2)


if __name__ == "__main__":
    unittest.main()
