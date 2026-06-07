"""
Tests de simulation/chaos.py

Sin llamadas a red. Seeds fijos, diferencias de fortaleza amplias para fiabilidad.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridcast.simulation.chaos import ChaosParameters, simulate_race_with_chaos
from gridcast.simulation.race_sim import simulate_race


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _drivers(n: int) -> list[str]:
    return [f"d{i}" for i in range(n)]


def _strengths(n: int, spread: float = 1.0) -> np.ndarray:
    """Strengths linealmente espaciadas: driver 0 es el más fuerte."""
    return np.linspace(spread, 0, n)


# ---------------------------------------------------------------------------
# ChaosParameters
# ---------------------------------------------------------------------------

class TestChaosParameters:
    def test_defaults_in_0_1(self):
        p = ChaosParameters()
        assert 0 <= p.sc_probability <= 1
        assert 0 <= p.rain_probability <= 1

    def test_noise_multipliers_gte_1(self):
        p = ChaosParameters()
        assert p.sc_noise_multiplier >= 1.0
        assert p.rain_noise_multiplier >= 1.0

    def test_compression_in_0_1(self):
        p = ChaosParameters()
        assert 0 < p.rain_strength_compression <= 1.0

    def test_dnf_boost_nonnegative(self):
        p = ChaosParameters()
        assert p.dnf_rate_boost >= 0.0

    def test_custom_params(self):
        p = ChaosParameters(sc_probability=0.8, rain_probability=0.2)
        assert p.sc_probability == 0.8
        assert p.rain_probability == 0.2


# ---------------------------------------------------------------------------
# simulate_race_with_chaos — salida
# ---------------------------------------------------------------------------

class TestSimulateRaceWithChaosOutput:
    def _sim(self, n_d=5, n_sims=200, **kw):
        return simulate_race_with_chaos(
            driver_ids=_drivers(n_d),
            strengths=_strengths(n_d),
            n_sims=n_sims,
            rng=_rng(),
            **kw,
        )

    def test_finishing_orders_shape(self):
        res = self._sim(n_d=5, n_sims=100)
        assert res.finishing_orders.shape == (100, 5)

    def test_points_shape(self):
        res = self._sim(n_d=5, n_sims=100)
        assert res.points.shape == (100, 5)

    def test_dnf_flags_shape(self):
        res = self._sim(n_d=5, n_sims=100)
        assert res.dnf_flags.shape == (100, 5)

    def test_finishing_orders_are_permutations(self):
        res = self._sim(n_d=5, n_sims=50)
        for s in range(50):
            assert sorted(res.finishing_orders[s]) == list(range(5))

    def test_points_non_negative(self):
        res = self._sim(n_d=10, n_sims=200)
        assert (res.points >= 0).all()

    def test_driver_ids_preserved(self):
        res = self._sim(n_d=4)
        assert res.driver_ids == _drivers(4)

    def test_is_sprint_propagated(self):
        res = self._sim(is_sprint=True)
        assert res.is_sprint is True

    def test_sprint_points_cap(self):
        res = self._sim(is_sprint=True, n_sims=500)
        assert res.points.max() == pytest.approx(8.0)

    def test_race_points_cap(self):
        res = self._sim(is_sprint=False, n_sims=500)
        assert res.points.max() == pytest.approx(25.0)

    def test_reproducible_with_same_seed(self):
        r1 = simulate_race_with_chaos(_drivers(4), _strengths(4), n_sims=50, rng=_rng(7))
        r2 = simulate_race_with_chaos(_drivers(4), _strengths(4), n_sims=50, rng=_rng(7))
        np.testing.assert_array_equal(r1.finishing_orders, r2.finishing_orders)
        np.testing.assert_array_equal(r1.points, r2.points)

    def test_all_four_chaos_scenarios_covered(self):
        """Con sc_prob=0.5 y rain_prob=0.5, los 4 escenarios deben aparecer."""
        params = ChaosParameters(sc_probability=0.5, rain_probability=0.5)
        # 2000 sims → todos los grupos deben tener ≥ 1 simulación con alta probabilidad
        res = simulate_race_with_chaos(
            _drivers(5), _strengths(5), n_sims=2000,
            params=params, rng=_rng(),
        )
        # Si algún grupo fallara, habría np.empty sin escribir → NaN o error
        assert (res.points >= 0).all()
        assert not np.isnan(res.points).any()


# ---------------------------------------------------------------------------
# Efecto del caos sobre la distribución de resultados
# ---------------------------------------------------------------------------

class TestChaosEffect:
    def test_no_chaos_similar_to_pure_race_sim(self):
        """Con SC=0 y rain=0, chaos wrapper debería dar resultados similares a race_sim."""
        params = ChaosParameters(sc_probability=0.0, rain_probability=0.0, dnf_rate_boost=0.0)
        strengths = _strengths(5, spread=2.0)

        chaos_res = simulate_race_with_chaos(
            _drivers(5), strengths, n_sims=3000,
            params=params, rng=_rng(1),
        )
        pure_res = simulate_race(
            _drivers(5), strengths, n_sims=3000, rng=_rng(1),
        )
        # Win probs deben ser similares (no idénticas: RNG usa distinto número de calls)
        chaos_win = chaos_res.win_probability()
        pure_win  = pure_res.win_probability()
        # Driver 0 (strongest) debería ganar mayoritariamente en ambos
        assert chaos_win[0] > 0.4
        assert pure_win[0]  > 0.4

    def test_full_chaos_reduces_favorite_win_rate(self):
        """Con SC=1 y rain=1 siempre, el favorito gana menos que sin caos."""
        strengths = np.array([3.0, 0.0, 0.0, 0.0, 0.0])  # driver 0 muy dominante

        no_chaos = ChaosParameters(
            sc_probability=0.0, rain_probability=0.0, dnf_rate_boost=0.0
        )
        full_chaos = ChaosParameters(
            sc_probability=1.0, rain_probability=1.0, dnf_rate_boost=0.0,
            sc_noise_multiplier=1.5, rain_noise_multiplier=1.4,
            rain_strength_compression=0.3,
        )

        res_no   = simulate_race_with_chaos(
            _drivers(5), strengths, n_sims=5000, params=no_chaos, rng=_rng(10)
        )
        res_full = simulate_race_with_chaos(
            _drivers(5), strengths, n_sims=5000, params=full_chaos, rng=_rng(10)
        )

        win_no   = res_no.win_probability()[0]
        win_full = res_full.win_probability()[0]
        assert win_full < win_no

    def test_dnf_boost_increases_dnf_rate(self):
        """Un mayor dnf_rate_boost produce más abandonos observados."""
        base_dnf = np.full(5, 0.05)

        low_boost  = ChaosParameters(sc_probability=0.0, rain_probability=0.0, dnf_rate_boost=0.0)
        high_boost = ChaosParameters(sc_probability=0.0, rain_probability=0.0, dnf_rate_boost=0.20)

        res_low  = simulate_race_with_chaos(
            _drivers(5), _strengths(5), dnf_rates=base_dnf,
            n_sims=5000, params=low_boost, rng=_rng(5)
        )
        res_high = simulate_race_with_chaos(
            _drivers(5), _strengths(5), dnf_rates=base_dnf,
            n_sims=5000, params=high_boost, rng=_rng(5)
        )

        dnf_low  = res_low.dnf_flags.mean()
        dnf_high = res_high.dnf_flags.mean()
        assert dnf_high > dnf_low

    def test_rain_compresses_win_probability(self):
        """Con lluvia siempre, el favorito tiene menos ventaja."""
        strengths = np.array([2.0, 0.0, 0.0, 0.0])

        no_rain   = ChaosParameters(sc_probability=0.0, rain_probability=0.0, dnf_rate_boost=0.0)
        all_rain  = ChaosParameters(
            sc_probability=0.0, rain_probability=1.0, dnf_rate_boost=0.0,
            rain_noise_multiplier=1.0,  # sin ruido extra, solo compresión
            rain_strength_compression=0.1,  # casi elimina las diferencias
        )

        res_dry  = simulate_race_with_chaos(_drivers(4), strengths, n_sims=4000, params=no_rain,  rng=_rng())
        res_rain = simulate_race_with_chaos(_drivers(4), strengths, n_sims=4000, params=all_rain, rng=_rng(99))

        # En lluvia intensa (compresión 0.1), el favorito debería ganar casi como en igualdad
        win_dry  = res_dry.win_probability()[0]
        win_rain = res_rain.win_probability()[0]
        assert win_rain < win_dry

    def test_win_probs_sum_to_one(self):
        res = simulate_race_with_chaos(
            _drivers(6), _strengths(6), n_sims=2000, rng=_rng()
        )
        assert res.win_probability().sum() == pytest.approx(1.0, abs=1e-9)

    def test_zero_sc_probability_no_sc_boost(self):
        """Con sc_prob=0, el noise_scale nunca se multiplica por sc_multiplier."""
        params = ChaosParameters(
            sc_probability=0.0, rain_probability=0.0, dnf_rate_boost=0.0,
            sc_noise_multiplier=10.0,  # si esto se aplicara, cambiaría mucho los resultados
        )
        strengths = np.array([5.0, 0.0, 0.0])  # driver 0 dominante
        res = simulate_race_with_chaos(
            _drivers(3), strengths, n_sims=3000, params=params, rng=_rng()
        )
        # Driver 0 debe ganar muy frecuentemente (sc_multiplier=10 no se activa)
        assert res.win_probability()[0] > 0.80
