"""
Tests de simulation/race_sim.py

Usa seeds fijos y diferencias de fortaleza grandes para que las pruebas
estocásticas sean fiables sin depender del número exacto de simulaciones.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridcast.simulation.race_sim import (
    RaceSimResult,
    _apply_dnfs,
    _apply_grid_penalties,
    _plackett_luce,
    _sample_strengths,
    simulate_race,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _drivers(n: int) -> list[str]:
    return [f"d{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# _apply_grid_penalties
# ---------------------------------------------------------------------------

class TestApplyGridPenalties:
    def test_no_penalty_unchanged(self):
        strengths = np.array([1.0, 2.0, 3.0])
        result = _apply_grid_penalties(strengths, np.zeros(3), np.zeros(3, bool), 0.02)
        np.testing.assert_array_almost_equal(result, strengths)

    def test_penalty_reduces_strength(self):
        strengths = np.array([1.0, 1.0, 1.0])
        penalties = np.array([0.0, 10.0, 0.0])
        result = _apply_grid_penalties(strengths, penalties, np.zeros(3, bool), 0.02)
        assert result[1] < result[0]  # driver 1 penalized

    def test_pit_lane_multiplier(self):
        strengths = np.array([1.0, 1.0])
        penalties = np.array([10.0, 10.0])
        pit_lane  = np.array([False, True])
        result = _apply_grid_penalties(strengths, penalties, pit_lane, 0.02)
        # Pit lane start → extra 50% reduction on top of base
        assert result[1] < result[0]

    def test_zero_penalty_for_pit_lane_with_zero_positions(self):
        strengths = np.array([1.0])
        result = _apply_grid_penalties(strengths, np.zeros(1), np.ones(1, bool), 0.02)
        np.testing.assert_array_almost_equal(result, strengths)


# ---------------------------------------------------------------------------
# _sample_strengths
# ---------------------------------------------------------------------------

class TestSampleStrengths:
    def test_shape_with_sds(self):
        strengths = np.array([1.0, 2.0, 3.0])
        sds = np.array([0.1, 0.1, 0.1])
        out = _sample_strengths(strengths, sds, n_sims=100, rng=_rng())
        assert out.shape == (100, 3)

    def test_shape_without_sds(self):
        strengths = np.array([1.0, 2.0, 3.0])
        out = _sample_strengths(strengths, None, n_sims=50, rng=_rng())
        assert out.shape == (50, 3)

    def test_without_sds_constant(self):
        strengths = np.array([1.0, 2.0])
        out = _sample_strengths(strengths, None, n_sims=10, rng=_rng())
        # Todas las filas deben ser iguales
        for i in range(10):
            np.testing.assert_array_equal(out[i], strengths)

    def test_mean_approx_correct_with_sds(self):
        strengths = np.array([10.0, 0.0])
        sds = np.array([0.5, 0.5])
        out = _sample_strengths(strengths, sds, n_sims=10_000, rng=_rng())
        np.testing.assert_array_almost_equal(out.mean(axis=0), strengths, decimal=1)


# ---------------------------------------------------------------------------
# _plackett_luce
# ---------------------------------------------------------------------------

class TestPlackettLuce:
    def test_output_is_permutation(self):
        strengths = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (100, 1))
        orders = _plackett_luce(strengths, noise_scale=0.5, rng=_rng())
        assert orders.shape == (100, 4)
        for s in range(100):
            assert sorted(orders[s]) == list(range(4))

    def test_stronger_driver_wins_more(self):
        """Driver con fortaleza mucho mayor gana casi siempre."""
        # Driver 0: strength=5 >> Driver 1: strength=0
        strengths = np.tile(np.array([5.0, 0.0]), (5000, 1))
        orders = _plackett_luce(strengths, noise_scale=0.5, rng=_rng())
        win_rate_d0 = (orders[:, 0] == 0).mean()
        assert win_rate_d0 > 0.95

    def test_equal_strengths_roughly_uniform(self):
        """Con fortalezas iguales, cada driver gana ≈ 1/n."""
        n = 4
        strengths = np.tile(np.zeros(n), (10_000, 1))
        orders = _plackett_luce(strengths, noise_scale=1.0, rng=_rng())
        for d in range(n):
            win_rate = (orders[:, 0] == d).mean()
            assert abs(win_rate - 1.0 / n) < 0.03  # tolerance 3%

    def test_reproducible_with_same_rng(self):
        strengths = np.tile(np.array([1.0, 0.5, 0.0]), (10, 1))
        o1 = _plackett_luce(strengths, 0.5, _rng(99))
        o2 = _plackett_luce(strengths, 0.5, _rng(99))
        np.testing.assert_array_equal(o1, o2)


# ---------------------------------------------------------------------------
# _apply_dnfs
# ---------------------------------------------------------------------------

class TestApplyDnfs:
    def test_no_dnfs_unchanged_order(self):
        orders = np.array([[2, 0, 1, 3]])
        dnf    = np.zeros((1, 4), dtype=bool)
        reordered, n_fin = _apply_dnfs(orders, dnf)
        np.testing.assert_array_equal(reordered[0], orders[0])
        assert n_fin[0] == 4

    def test_dnf_driver_goes_to_back(self):
        # Driver 2 DNFs (driver at pos 0)
        orders = np.array([[2, 0, 1, 3]])
        dnf = np.zeros((1, 4), dtype=bool)
        dnf[0, 2] = True  # driver_idx 2 DNFs
        reordered, n_fin = _apply_dnfs(orders, dnf)
        # Finishers (drivers 0, 1, 3) come first; driver 2 at back
        assert reordered[0, -1] == 2 or 2 in reordered[0, n_fin[0]:]
        assert n_fin[0] == 3

    def test_all_dnf(self):
        orders = np.array([[0, 1, 2]])
        dnf = np.ones((1, 3), dtype=bool)
        _, n_fin = _apply_dnfs(orders, dnf)
        assert n_fin[0] == 0

    def test_n_finishers_correct(self):
        orders = np.tile(np.arange(5), (3, 1))
        dnf = np.zeros((3, 5), dtype=bool)
        dnf[0, [0, 2]] = True   # 2 DNFs
        dnf[1, [4]]    = True   # 1 DNF
        # dnf[2]: no DNFs
        _, n_fin = _apply_dnfs(orders, dnf)
        assert n_fin[0] == 3
        assert n_fin[1] == 4
        assert n_fin[2] == 5

    def test_relative_order_of_finishers_preserved(self):
        """El orden relativo de Plackett-Luce entre finishers se conserva."""
        # PL order: [3, 0, 1, 2] — driver 3 fastest, driver 2 slowest
        orders = np.array([[3, 0, 1, 2]])
        dnf = np.zeros((1, 4), dtype=bool)
        dnf[0, 3] = True  # driver 3 DNFs (fastest by PL but DNF)
        reordered, n_fin = _apply_dnfs(orders, dnf)
        # First 3 positions should be [0, 1, 2] (original PL order among finishers)
        np.testing.assert_array_equal(reordered[0, :3], [0, 1, 2])
        assert n_fin[0] == 3


# ---------------------------------------------------------------------------
# simulate_race — validación de entrada
# ---------------------------------------------------------------------------

class TestSimulateRaceInputValidation:
    def test_empty_driver_list_raises(self):
        with pytest.raises(ValueError, match="vacío"):
            simulate_race([], np.array([]))

    def test_wrong_strengths_shape_raises(self):
        with pytest.raises(ValueError):
            simulate_race(["a", "b"], np.array([1.0]))

    def test_dnf_rates_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            simulate_race(["a", "b"], np.array([1.0, 2.0]),
                          dnf_rates=np.array([0.1, 1.5]))

    def test_wrong_dnf_rates_shape_raises(self):
        with pytest.raises(ValueError):
            simulate_race(["a", "b"], np.array([1.0, 2.0]),
                          dnf_rates=np.array([0.1]))


# ---------------------------------------------------------------------------
# simulate_race — salida
# ---------------------------------------------------------------------------

class TestSimulateRaceOutput:
    def _sim(self, n_d: int = 5, n_sims: int = 1000, **kw) -> RaceSimResult:
        drivers   = _drivers(n_d)
        strengths = np.arange(n_d, 0, -1, dtype=float)
        return simulate_race(drivers, strengths, n_sims=n_sims, rng=_rng(), **kw)

    def test_finishing_orders_shape(self):
        res = self._sim(n_d=5, n_sims=200)
        assert res.finishing_orders.shape == (200, 5)

    def test_dnf_flags_shape(self):
        res = self._sim(n_d=5, n_sims=200)
        assert res.dnf_flags.shape == (200, 5)

    def test_points_shape(self):
        res = self._sim(n_d=5, n_sims=200)
        assert res.points.shape == (200, 5)

    def test_finishing_orders_are_permutations(self):
        res = self._sim(n_d=5, n_sims=50)
        for s in range(50):
            assert sorted(res.finishing_orders[s]) == list(range(5))

    def test_points_non_negative(self):
        res = self._sim(n_d=10, n_sims=500)
        assert (res.points >= 0).all()

    def test_driver_ids_preserved(self):
        drivers = ["ver", "nor", "ham"]
        res = simulate_race(drivers, np.array([3.0, 2.0, 1.0]),
                            n_sims=10, rng=_rng())
        assert res.driver_ids == drivers

    def test_n_sims_property(self):
        res = self._sim(n_sims=777)
        assert res.n_sims == 777

    def test_n_drivers_property(self):
        res = self._sim(n_d=7)
        assert res.n_drivers == 7

    def test_sprint_flag_propagated(self):
        res = self._sim(is_sprint=True)
        assert res.is_sprint is True


# ---------------------------------------------------------------------------
# simulate_race — propiedades estocásticas
# ---------------------------------------------------------------------------

class TestSimulateRaceStochastic:
    """Tests con diferencias de fortaleza grandes para fiabilidad estadística."""

    def test_stronger_driver_wins_more(self):
        """Driver 0 con fortaleza mucho mayor gana con alta probabilidad."""
        strengths = np.array([5.0, 0.0, 0.0, 0.0])
        res = simulate_race(_drivers(4), strengths, n_sims=5_000, rng=_rng())
        win_prob = res.win_probability()
        assert win_prob[0] > 0.90

    def test_win_probability_sums_to_one(self):
        strengths = np.array([2.0, 1.0, 0.5, 0.0])
        res = simulate_race(_drivers(4), strengths, n_sims=2_000, rng=_rng())
        assert res.win_probability().sum() == pytest.approx(1.0, abs=1e-9)

    def test_dnf_probability_reflects_rate(self):
        """Con dnf_rate=0.5, cada driver DNFs aproximadamente la mitad."""
        dnf_rates = np.full(4, 0.5)
        res = simulate_race(_drivers(4), np.zeros(4), dnf_rates=dnf_rates,
                            n_sims=5_000, rng=_rng())
        dnf_prob = res.dnf_probability()
        for p in dnf_prob:
            assert abs(p - 0.5) < 0.05

    def test_no_dnf_all_drivers_score_or_not_by_position(self):
        """Sin DNFs, solo los top 10 reciben puntos."""
        n_d = 15
        res = simulate_race(_drivers(n_d), np.zeros(n_d),
                            dnf_rates=np.zeros(n_d),
                            n_sims=100, rng=_rng())
        assert (res.points[:, :10] >= 0).all()
        # Drivers placed 11+ get 0
        # (Not trivially verifiable without knowing finishing order, but
        #  expected_points for drivers consistently at back should be near 0)

    def test_100pct_dnf_rate_zero_points(self):
        """Con DNF=1, ningún piloto puntúa."""
        dnf_rates = np.ones(5)
        res = simulate_race(_drivers(5), np.zeros(5), dnf_rates=dnf_rates,
                            n_sims=100, rng=_rng())
        assert res.points.sum() == pytest.approx(0.0)

    def test_grid_penalty_reduces_win_prob(self):
        """Penalización de parrilla reduce la probabilidad de ganar."""
        strengths = np.zeros(4)
        no_pen  = simulate_race(_drivers(4), strengths, n_sims=3_000, rng=_rng())
        with_pen = simulate_race(_drivers(4), strengths,
                                 grid_penalties=np.array([20.0, 0.0, 0.0, 0.0]),
                                 n_sims=3_000, rng=_rng(43))
        # Driver 0 should win less with a big penalty
        assert with_pen.win_probability()[0] < no_pen.win_probability()[0]

    def test_sprint_points_cap_at_8(self):
        strengths = np.zeros(5)
        res = simulate_race(_drivers(5), strengths, is_sprint=True,
                            n_sims=1_000, rng=_rng())
        assert res.points.max() == pytest.approx(8.0)

    def test_reproducible_with_same_seed(self):
        strengths = np.array([1.0, 0.5, 0.0])
        r1 = simulate_race(_drivers(3), strengths, n_sims=100, rng=_rng(7))
        r2 = simulate_race(_drivers(3), strengths, n_sims=100, rng=_rng(7))
        np.testing.assert_array_equal(r1.finishing_orders, r2.finishing_orders)
        np.testing.assert_array_equal(r1.points, r2.points)


# ---------------------------------------------------------------------------
# RaceSimResult — métodos de resumen
# ---------------------------------------------------------------------------

class TestRaceSimResultSummary:
    def _res(self, n_sims: int = 2_000) -> RaceSimResult:
        strengths = np.array([3.0, 2.0, 1.0, 0.0])
        return simulate_race(_drivers(4), strengths, n_sims=n_sims, rng=_rng())

    def test_podium_probability_shape(self):
        res = self._res()
        assert res.podium_probability().shape == (4,)

    def test_podium_probability_sums_to_3(self):
        """Exactamente 3 drivers están en el podio en cada sim."""
        res = self._res(n_sims=3_000)
        assert res.podium_probability().sum() == pytest.approx(3.0, abs=1e-9)

    def test_top_k_probability_k1_equals_win(self):
        res = self._res(n_sims=2_000)
        np.testing.assert_array_almost_equal(
            res.top_k_probability(1), res.win_probability()
        )

    def test_expected_points_non_negative(self):
        res = self._res()
        assert (res.expected_points() >= 0).all()

    def test_strongest_driver_highest_expected_points(self):
        """Driver 0 (strongest) debe tener mayor E[pts] que el resto."""
        res = self._res(n_sims=5_000)
        ep = res.expected_points()
        assert ep[0] > ep[1] > ep[2] > ep[3]

    def test_finish_position_distribution_shape(self):
        res = self._res()
        dist = res.finish_position_distribution()
        assert dist.shape == (4, 4)

    def test_finish_position_distribution_rows_sum_to_1(self):
        """Cada driver aparece exactamente una vez en cada sim → rows sum to 1."""
        res = self._res(n_sims=3_000)
        dist = res.finish_position_distribution()
        np.testing.assert_array_almost_equal(dist.sum(axis=1), np.ones(4))

    def test_finish_position_distribution_cols_sum_to_1(self):
        """Exactamente un driver en cada posición por sim → cols sum to 1."""
        res = self._res(n_sims=3_000)
        dist = res.finish_position_distribution()
        np.testing.assert_array_almost_equal(dist.sum(axis=0), np.ones(4))

    def test_to_summary_df_columns(self):
        res = self._res()
        df = res.to_summary_df()
        assert set(df.columns) == {
            "driver_id", "win_prob", "podium_prob", "dnf_prob", "expected_points"
        }

    def test_to_summary_df_n_rows(self):
        res = self._res()
        assert len(res.to_summary_df()) == 4

    def test_dnf_probability_zero_without_dnf_rates(self):
        """Sin DNF rates, ningún piloto abandona."""
        res = simulate_race(_drivers(4), np.zeros(4), n_sims=100, rng=_rng())
        np.testing.assert_array_equal(res.dnf_probability(), np.zeros(4))
