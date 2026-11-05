from __future__ import annotations

import json
import math
import tempfile
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

import coverage_scorer_cpp
from step4_annealing import simulated_annealing as sa


class FixedDrawRng:
    """Small RNG stand-in for exact Metropolis boundary tests."""

    def __init__(self, draw: float) -> None:
        self.draw = draw

    def random(self) -> float:
        return self.draw


def make_tiny_scorer() -> coverage_scorer_cpp.IncrementalCoverageScorer:
    """Build a six-point, three-building scorer with overlapping visibility."""

    return coverage_scorer_cpp.IncrementalCoverageScorer(
        np.ascontiguousarray([10, 20, 30, 40, 50, 60], dtype=np.int64),
        np.ascontiguousarray([0, 1, 2, 3, 4, 6, 7], dtype=np.int64),
        np.ascontiguousarray([0, 1, 2, 2, 0, 1, 2], dtype=np.int32),
        np.ascontiguousarray([0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
        np.ascontiguousarray(
            [
                [0.00, 0.50],
                [0.00, 0.50],
                [0.00, 0.49],
                [0.49, 1.00],
                [0.00, 0.25],
                [0.50, 0.75],
                [0.00, 0.50],
            ],
            dtype=np.float64,
        ),
        np.ascontiguousarray([1.0] * 7, dtype=np.float64),
        np.ascontiguousarray([1.0, 1.0, 1.0], dtype=np.float64),
        0.50,
    )


class AcceptanceTests(unittest.TestCase):
    def test_hard_improvements_and_nonworsening_plateaus_are_accepted(self) -> None:
        rng = FixedDrawRng(0.999999)
        self.assertTrue(
            sa.accept_move(
                delta_hard=1,
                delta_soft=-1_000.0,
                hard_temperature=0.0,
                soft_temperature=0.0,
                rng=rng,
            )
        )
        self.assertTrue(
            sa.accept_move(
                delta_hard=0,
                delta_soft=0.0,
                hard_temperature=0.0,
                soft_temperature=0.0,
                rng=rng,
            )
        )
        self.assertTrue(
            sa.accept_move(
                delta_hard=0,
                delta_soft=0.1,
                hard_temperature=0.0,
                soft_temperature=0.0,
                rng=rng,
            )
        )

    def test_hard_loss_uses_only_hard_temperature(self) -> None:
        temperature = sa.hard_temperature_for_loss_acceptance(0.20)
        common = {
            "delta_hard": -1,
            "delta_soft": 1_000_000.0,
            "hard_temperature": temperature,
            "soft_temperature": 1e30,
        }
        self.assertTrue(sa.accept_move(**common, rng=FixedDrawRng(0.10)))
        self.assertFalse(sa.accept_move(**common, rng=FixedDrawRng(0.30)))
        self.assertFalse(
            sa.accept_move(
                delta_hard=-1,
                delta_soft=1_000_000.0,
                hard_temperature=0.0,
                soft_temperature=1e30,
                rng=FixedDrawRng(0.0),
            )
        )

    def test_soft_loss_on_plateau_uses_soft_temperature(self) -> None:
        # exp(-2 / T) = 0.25
        temperature = -2.0 / math.log(0.25)
        common = {
            "delta_hard": 0,
            "delta_soft": -2.0,
            "hard_temperature": 1e30,
            "soft_temperature": temperature,
        }
        self.assertTrue(sa.accept_move(**common, rng=FixedDrawRng(0.10)))
        self.assertFalse(sa.accept_move(**common, rng=FixedDrawRng(0.40)))
        self.assertFalse(
            sa.accept_move(
                delta_hard=0,
                delta_soft=-1.0,
                hard_temperature=1.0,
                soft_temperature=0.0,
                rng=FixedDrawRng(0.0),
            )
        )


class TemperatureTests(unittest.TestCase):
    def test_probability_to_temperature_round_trip(self) -> None:
        for probability in (0.001, 0.10, 0.75):
            temperature = sa.hard_temperature_for_loss_acceptance(probability)
            self.assertAlmostEqual(math.exp(-1.0 / temperature), probability)

    def test_geometric_temperature_and_clamping(self) -> None:
        self.assertAlmostEqual(sa.geometric_temperature(8.0, 0.5, 0.0), 8.0)
        self.assertAlmostEqual(sa.geometric_temperature(8.0, 0.5, 0.5), 2.0)
        self.assertAlmostEqual(sa.geometric_temperature(8.0, 0.5, 1.0), 0.5)
        self.assertAlmostEqual(sa.geometric_temperature(8.0, 0.5, -10.0), 8.0)
        self.assertAlmostEqual(sa.geometric_temperature(8.0, 0.5, 10.0), 0.5)

    def test_island_profiles_repeat_and_span_log_range(self) -> None:
        spread = 0.70
        self.assertAlmostEqual(
            sa.island_temperature_multiplier(0, 5, spread),
            math.exp(-spread),
        )
        self.assertAlmostEqual(sa.island_temperature_multiplier(2, 5, spread), 1.0)
        self.assertAlmostEqual(
            sa.island_temperature_multiplier(4, 5, spread),
            math.exp(spread),
        )
        self.assertAlmostEqual(
            sa.island_temperature_multiplier(7, 5, spread),
            sa.island_temperature_multiplier(2, 5, spread),
        )

    def test_balanced_seed_temperature_assignment(self) -> None:
        assignments = [
            sa.balanced_island_assignment(rank, 100, 5, 5)
            for rank in range(100)
        ]
        pair_counts = {
            (seed_index, temperature_index): 0
            for seed_index in range(5)
            for temperature_index in range(5)
        }
        for seed_index, temperature_index, replica_index in assignments:
            pair_counts[(seed_index, temperature_index)] += 1
            self.assertIn(replica_index, range(4))

        self.assertEqual(set(pair_counts.values()), {4})
        self.assertEqual(assignments[0], (0, 0, 0))
        self.assertEqual(assignments[4], (4, 0, 0))
        self.assertEqual(assignments[5], (0, 1, 0))
        self.assertEqual(assignments[25], (0, 0, 1))

    def test_complete_island_assignment_keeps_each_selected_seed(self) -> None:
        seeds = [(100 + index,) for index in range(5)]
        expected_buildings = [(1000 + index,) for index in range(5)]
        paths = [f"seed_{index + 1:02d}.txt" for index in range(5)]

        assignments = [
            sa.assign_island_seed(
                rank=rank,
                world_size=100,
                seed_ids=seeds,
                seed_expected_building_ids=expected_buildings,
                seed_paths=paths,
                temperature_levels=5,
            )
            for rank in range(5)
        ]

        self.assertEqual(
            [assignment.selected_point_ids for assignment in assignments],
            seeds,
        )
        self.assertEqual(
            [assignment.expected_building_ids for assignment in assignments],
            expected_buildings,
        )
        self.assertEqual(
            [assignment.seed_path for assignment in assignments],
            paths,
        )
        self.assertEqual(
            [assignment.seed_index for assignment in assignments],
            list(range(5)),
        )

    def test_complete_island_assignment_rejects_misaligned_seed_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have equal lengths"):
            sa.assign_island_seed(
                rank=0,
                world_size=25,
                seed_ids=[(1,), (2,)],
                seed_expected_building_ids=[(10,)],
                seed_paths=["seed_01.txt", "seed_02.txt"],
                temperature_levels=5,
            )

    def test_assignment_period_and_single_axis_cases(self) -> None:
        first_period = [
            sa.balanced_island_assignment(rank, 50, 5, 5)
            for rank in range(25)
        ]
        second_period = [
            sa.balanced_island_assignment(rank, 50, 5, 5)
            for rank in range(25, 50)
        ]
        self.assertEqual(
            [(seed, temperature) for seed, temperature, _ in first_period],
            [(seed, temperature) for seed, temperature, _ in second_period],
        )
        self.assertEqual({replica for _, _, replica in first_period}, {0})
        self.assertEqual({replica for _, _, replica in second_period}, {1})

        one_seed = [
            sa.balanced_island_assignment(rank, 10, 1, 5)
            for rank in range(10)
        ]
        self.assertEqual(
            [temperature for _, temperature, _ in one_seed],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        )
        one_temperature = [
            sa.balanced_island_assignment(rank, 10, 5, 1)
            for rank in range(10)
        ]
        self.assertEqual(
            [seed for seed, _, _ in one_temperature],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        )
        self.assertEqual({temperature for _, temperature, _ in one_temperature}, {0})

    def test_balanced_assignment_supports_dynamic_seed_counts(self) -> None:
        assignments = [
            sa.balanced_island_assignment(rank, 140, 7, 5)
            for rank in range(140)
        ]
        self.assertEqual(
            {
                (seed_index, temperature_index): sum(
                    1
                    for assigned_seed, assigned_temperature, _ in assignments
                    if (assigned_seed, assigned_temperature)
                    == (seed_index, temperature_index)
                )
                for seed_index in range(7)
                for temperature_index in range(5)
            },
            {
                (seed_index, temperature_index): 4
                for seed_index in range(7)
                for temperature_index in range(5)
            },
        )

    def test_unbalanced_island_layout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "need at least 25"):
            sa.balanced_island_assignment(0, 20, 5, 5)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            sa.balanced_island_assignment(0, 101, 5, 5)
        with self.assertRaisesRegex(ValueError, "within the MPI world"):
            sa.balanced_island_assignment(100, 100, 5, 5)
        with self.assertRaisesRegex(ValueError, "world size must be positive"):
            sa.balanced_island_assignment(0, 0, 5, 5)
        with self.assertRaisesRegex(ValueError, "seed count must be positive"):
            sa.balanced_island_assignment(0, 25, 0, 5)
        with self.assertRaisesRegex(ValueError, "profile levels must be positive"):
            sa.balanced_island_assignment(0, 25, 5, 0)
        with self.assertRaisesRegex(ValueError, "within the MPI world"):
            sa.balanced_island_assignment(-1, 25, 5, 5)


class CandidateAndStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.empty = make_tiny_scorer()
        self.all_ids = np.asarray(self.empty.point_ids, dtype=np.int64)

    def test_set_identity_and_available_candidate_exclusions(self) -> None:
        self.assertEqual(sa.canonical_ids([20, 10]), (10, 20))
        with self.assertRaises(ValueError):
            sa.canonical_ids([10, 10])

        legal = sa.available_ids(
            self.all_ids,
            selected_ids=[10, 20],
            blocked_ids=[20, 30, 30],
        )
        np.testing.assert_array_equal(legal, [40, 50, 60])

    def test_exact_soft_state_and_multiremove_loss(self) -> None:
        simple = sa.build_state(self.empty, [10, 20])
        self.assertEqual(simple.hard_score, 2)
        self.assertAlmostEqual(simple.soft_score, 2.0)

        # Point 30 overlaps point 40 on building 2, while point 50 supplies
        # partial coverage on two other buildings.  Sequential restoration
        # must still telescope to the exact whole-state soft difference.
        original = sa.build_state(self.empty, [30, 40, 50])
        after_removal = original.scorer.clone()
        for point_id in (30, 50):
            after_removal.remove_point(point_id)
        loss = sa.exact_removed_soft_loss(after_removal, [30, 50])
        remaining = sa.build_state(self.empty, [40])

        self.assertEqual(original.hard_score, 1)
        self.assertAlmostEqual(original.soft_score, 2.0)
        self.assertAlmostEqual(remaining.soft_score, 1.0)
        self.assertAlmostEqual(loss, 1.0)
        self.assertAlmostEqual(original.soft_score - loss, remaining.soft_score)

    def test_proposals_preserve_k_uniqueness_and_exact_scores(self) -> None:
        current = sa.build_state(self.empty, [10, 20])
        original_ids = set(current.selected_ids)

        for move_name, move_size in ((sa.MOVE_ONE, 1), (sa.MOVE_TWO, 2)):
            for seed in range(10):
                proposal = sa.propose_neighbor(
                    current=current,
                    all_point_ids=self.all_ids,
                    move_name=move_name,
                    move_size=move_size,
                    candidate_pool_size=100,
                    top_candidates=10,
                    rank_temperature=2.0,
                    random_candidate_probability=0.20,
                    rng=np.random.default_rng(seed),
                )
                proposed_ids = set(proposal.state.selected_ids)
                rebuilt = sa.build_state(self.empty, proposal.state.selected_ids)

                self.assertEqual(len(proposed_ids), len(original_ids))
                self.assertTrue(proposed_ids.issubset(set(self.all_ids.tolist())))
                self.assertTrue(set(proposal.removed_ids).isdisjoint(proposal.added_ids))
                self.assertEqual(
                    proposed_ids,
                    original_ids.difference(proposal.removed_ids).union(
                        proposal.added_ids
                    ),
                )
                self.assertEqual(proposal.state.hard_score, rebuilt.hard_score)
                self.assertAlmostEqual(
                    proposal.state.soft_score,
                    rebuilt.soft_score,
                    places=12,
                )


class CheckpointUtilityTests(unittest.TestCase):
    def test_numpy_metadata_is_json_safe_and_atomically_written(self) -> None:
        value = {
            "integer": np.int64(7),
            "floating": np.float64(0.25),
            "array": np.asarray([1, 2, 3], dtype=np.uint64),
            "nested": (np.int32(4), {"x": np.float32(1.5)}),
        }
        converted = sa.json_compatible(value)
        self.assertEqual(converted["integer"], 7)
        self.assertEqual(converted["array"], [1, 2, 3])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "metadata.json"
            sa.atomic_write_json(output, value)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded, converted)
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_checkpoint_serial_skips_existing_orphan_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rank_folder = Path(temporary) / "rank_0000"
            checkpoints = rank_folder / "checkpoints"
            (checkpoints / "checkpoint_000001").mkdir(parents=True)
            (checkpoints / "checkpoint_000002").mkdir()

            self.assertEqual(sa.next_checkpoint_serial(rank_folder, 0), 0)
            self.assertEqual(sa.next_checkpoint_serial(rank_folder, 1), 3)
            with self.assertRaises(ValueError):
                sa.next_checkpoint_serial(rank_folder, -1)


if __name__ == "__main__":
    unittest.main()
