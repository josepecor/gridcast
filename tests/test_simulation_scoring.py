"""
Tests de simulation/scoring.py

Sin llamadas a red.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridcast.simulation.scoring import (
    _inverse_permutation_batch,
    build_race_points_table,
    build_sprint_points_table,
    points_for_position,
    score_finishing_order,
    score_finishing_orders_batch,
)


# ---------------------------------------------------------------------------
# build_race_points_table / build_sprint_points_table
# ---------------------------------------------------------------------------

class TestBuildPointsTables:
    def test_race_winner_gets_25(self):
        table = build_race_points_table(20)
        assert table[0] == 25.0

    def test_race_second_gets_18(self):
        table = build_race_points_table(20)
        assert table[1] == 18.0

    def test_race_tenth_gets_1(self):
        table = build_race_points_table(20)
        assert table[9] == 1.0

    def test_race_eleventh_gets_0(self):
        table = build_race_points_table(20)
        assert table[10] == 0.0

    def test_race_scored_positions(self):
        table = build_race_points_table(20)
        unique_nonzero = sorted(set(table[table > 0].tolist()), reverse=True)
        assert unique_nonzero == [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

    def test_race_table_length(self):
        for n in [10, 20, 22]:
            assert len(build_race_points_table(n)) == n

    def test_race_monotone_decreasing_in_scored_zone(self):
        table = build_race_points_table(20)
        nonzero = table[table > 0]
        assert all(nonzero[i] > nonzero[i + 1] for i in range(len(nonzero) - 1))

    def test_sprint_winner_gets_8(self):
        table = build_sprint_points_table(20)
        assert table[0] == 8.0

    def test_sprint_eighth_gets_1(self):
        table = build_sprint_points_table(20)
        assert table[7] == 1.0

    def test_sprint_ninth_gets_0(self):
        table = build_sprint_points_table(20)
        assert table[8] == 0.0

    def test_sprint_no_race_points(self):
        table = build_sprint_points_table(20)
        assert 25.0 not in table
        assert 18.0 not in table


# ---------------------------------------------------------------------------
# points_for_position
# ---------------------------------------------------------------------------

class TestPointsForPosition:
    def test_race_first(self):
        assert points_for_position(1) == 25.0

    def test_race_tenth(self):
        assert points_for_position(10) == 1.0

    def test_race_eleventh_zero(self):
        assert points_for_position(11) == 0.0

    def test_sprint_first(self):
        assert points_for_position(1, is_sprint=True) == 8.0

    def test_sprint_eighth(self):
        assert points_for_position(8, is_sprint=True) == 1.0

    def test_sprint_ninth_zero(self):
        assert points_for_position(9, is_sprint=True) == 0.0


# ---------------------------------------------------------------------------
# _inverse_permutation_batch
# ---------------------------------------------------------------------------

class TestInversePermutationBatch:
    def test_identity_permutation(self):
        perm = np.arange(5).reshape(1, 5)
        rank = _inverse_permutation_batch(perm)
        np.testing.assert_array_equal(rank[0], np.arange(5))

    def test_reversal(self):
        perm = np.array([[4, 3, 2, 1, 0]])
        rank = _inverse_permutation_batch(perm)
        assert rank[0, 0] == 4
        assert rank[0, 4] == 0

    def test_batch_correctness(self):
        perm = np.array([[2, 0, 1], [1, 2, 0]])
        rank = _inverse_permutation_batch(perm)
        # Sim 0: driver 0→pos1, driver 1→pos2, driver 2→pos0
        np.testing.assert_array_equal(rank[0], [1, 2, 0])
        # Sim 1: driver 0→pos2, driver 1→pos0, driver 2→pos1
        np.testing.assert_array_equal(rank[1], [2, 0, 1])

    def test_double_inverse_is_identity(self):
        rng = np.random.default_rng(0)
        perm = np.argsort(rng.random((50, 10)), axis=1)
        rank = _inverse_permutation_batch(perm)
        recovered = _inverse_permutation_batch(rank)
        np.testing.assert_array_equal(recovered, perm)

    def test_apply_to_score(self):
        """Usar rank para asignar puntos es equivalente a score_finishing_order."""
        perm = np.array([[2, 0, 1, 3, 4]])
        table = build_race_points_table(5)
        rank = _inverse_permutation_batch(perm)
        pts_via_rank = table[rank[0]]         # puntos indexados por posición del driver
        pts_direct   = score_finishing_order(perm[0], table)
        np.testing.assert_array_almost_equal(pts_via_rank, pts_direct)


# ---------------------------------------------------------------------------
# score_finishing_order
# ---------------------------------------------------------------------------

class TestScoreFinishingOrder:
    def _race_table(self, n: int = 5) -> np.ndarray:
        return build_race_points_table(n)

    def test_winner_gets_25(self):
        order = np.array([2, 0, 1, 3, 4])
        pts = score_finishing_order(order, self._race_table())
        assert pts[2] == 25.0

    def test_second_gets_18(self):
        order = np.array([2, 0, 1, 3, 4])
        pts = score_finishing_order(order, self._race_table())
        assert pts[0] == 18.0

    def test_all_non_negative(self):
        order = np.arange(5)
        assert (score_finishing_order(order, self._race_table()) >= 0).all()

    def test_sum_of_points(self):
        # 5 pilotos, posiciones 1-5: 25+18+15+12+10 = 80
        order = np.arange(5)
        total = score_finishing_order(order, self._race_table()).sum()
        assert total == pytest.approx(80.0)

    def test_dnf_driver_gets_zero(self):
        order = np.array([2, 0, 1, 3, 4])
        pts = score_finishing_order(order, self._race_table(), n_finishers=3)
        assert pts[3] == 0.0  # driver at pos 3 → DNF
        assert pts[4] == 0.0  # driver at pos 4 → DNF

    def test_finishers_still_score(self):
        order = np.array([0, 1, 2, 3, 4])
        pts = score_finishing_order(order, self._race_table(), n_finishers=2)
        assert pts[0] == 25.0
        assert pts[1] == 18.0


# ---------------------------------------------------------------------------
# score_finishing_orders_batch
# ---------------------------------------------------------------------------

class TestScoreFinishingOrdersBatch:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        n_sims, n_d = 100, 20
        orders = np.argsort(rng.random((n_sims, n_d)), axis=1)
        pts = score_finishing_orders_batch(orders, build_race_points_table(n_d))
        assert pts.shape == (n_sims, n_d)

    def test_matches_scalar(self):
        order = np.array([[2, 0, 1, 3, 4]])
        table = build_race_points_table(5)
        batch  = score_finishing_orders_batch(order, table)
        scalar = score_finishing_order(order[0], table)
        np.testing.assert_array_almost_equal(batch[0], scalar)

    def test_winner_25_in_each_sim(self):
        n_d = 10
        orders = np.array([
            [3] + list(range(10)) if 3 not in list(range(10)) else [3, 0, 1, 2, 4, 5, 6, 7, 8, 9],
            [7, 0, 1, 2, 3, 4, 5, 6, 8, 9],
        ])
        orders = np.array([
            [3, 0, 1, 2, 4, 5, 6, 7, 8, 9],
            [7, 0, 1, 2, 3, 4, 5, 6, 8, 9],
        ])
        table = build_race_points_table(n_d)
        pts = score_finishing_orders_batch(orders, table)
        assert pts[0, 3] == 25.0
        assert pts[1, 7] == 25.0

    def test_dnf_drivers_get_zero(self):
        n_d = 5
        orders = np.tile(np.arange(n_d), (3, 1))
        table = build_race_points_table(n_d)
        n_fin = np.array([2, 3, 5])
        pts = score_finishing_orders_batch(orders, table, n_finishers=n_fin)
        # Sim 0: solo 2 finishers
        assert pts[0, 0] == 25.0
        assert pts[0, 1] == 18.0
        assert pts[0, 2] == 0.0
        # Sim 2: todos finalizan
        assert pts[2, 4] == 10.0  # 5th = 10 pts

    def test_all_non_negative(self):
        rng = np.random.default_rng(42)
        orders = np.argsort(rng.random((200, 20)), axis=1)
        pts = score_finishing_orders_batch(orders, build_race_points_table(20))
        assert (pts >= 0).all()

    def test_sprint_table(self):
        order = np.array([[0, 1, 2, 3, 4]])
        table = build_sprint_points_table(5)
        pts = score_finishing_orders_batch(order, table)
        assert pts[0, 0] == 8.0
        assert pts[0, 1] == 7.0

    def test_total_points_sum_without_dnf(self):
        n_d = 20
        rng = np.random.default_rng(7)
        orders = np.argsort(rng.random((5, n_d)), axis=1)
        table = build_race_points_table(n_d)
        pts = score_finishing_orders_batch(orders, table)
        expected_total = table.sum()
        for s in range(5):
            assert pts[s].sum() == pytest.approx(expected_total)
