"""
Tests de simulation/season_sim.py

Sin llamadas a red. Fixtures sintéticos con seeds fijos.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gridcast.ingest.jolpica_client import (
    ConstructorStanding,
    DriverStanding,
    RoundInfo,
)
from gridcast.simulation.season_sim import (
    SeasonSimResult,
    _build_constructor_arrays,
    _build_dnf_array,
    _build_driver_arrays,
    _DEFAULT_DNF_RATE,
    _UNKNOWN_STRENGTH_SD,
    simulate_championship,
)


# ---------------------------------------------------------------------------
# Factories de fixtures
# ---------------------------------------------------------------------------

def _round(round_num: int, has_sprint: bool = False) -> RoundInfo:
    return RoundInfo(
        season=2026,
        round=round_num,
        race_name=f"GP {round_num}",
        jolpica_circuit_id=f"circuit_{round_num}",
        circuit_name=f"Circuit {round_num}",
        date=date(2026, 3 + round_num, 1),
        has_sprint=has_sprint,
    )


def _driver_standing(driver_id: str, position: int, points: float) -> DriverStanding:
    return DriverStanding(
        position=position,
        driver_id=driver_id,
        code=driver_id[:3].upper(),
        points=points,
        wins=0,
    )


def _constr_standing(constructor_id: str, position: int, points: float) -> ConstructorStanding:
    return ConstructorStanding(
        position=position,
        constructor_id=constructor_id,
        name=constructor_id.title(),
        points=points,
        wins=0,
    )


def _strength_df(driver_ids: list[str], means: list[float], sds: list[float] | None = None) -> pd.DataFrame:
    data: dict = {"driver_id": driver_ids, "mean": means}
    if sds is not None:
        data["sd"] = sds
    return pd.DataFrame(data)


def _season_result(
    driver_ids: list[str],
    pts_start: list[float],
    pts_sims: np.ndarray,
    constructor_ids: list[str] | None = None,
    constr_pts_start: list[float] | None = None,
    constr_pts_sims: np.ndarray | None = None,
) -> SeasonSimResult:
    n_sims, n_d = pts_sims.shape
    if constructor_ids is None:
        constructor_ids = []
    if constr_pts_start is None:
        constr_pts_start_arr = np.array([])
    else:
        constr_pts_start_arr = np.array(constr_pts_start)
    if constr_pts_sims is None:
        constr_pts_sims = np.zeros((n_sims, max(len(constructor_ids), 1)))
    return SeasonSimResult(
        driver_ids=driver_ids,
        constructor_ids=constructor_ids,
        driver_points_sims=pts_sims,
        constructor_points_sims=constr_pts_sims,
        driver_points_start=np.array(pts_start),
        constructor_points_start=constr_pts_start_arr,
        n_rounds_simulated=3,
    )


# ---------------------------------------------------------------------------
# SeasonSimResult — propiedades básicas
# ---------------------------------------------------------------------------

class TestSeasonSimResultProperties:
    def _basic(self, n_sims: int = 100, n_d: int = 3) -> SeasonSimResult:
        rng = np.random.default_rng(0)
        pts = list(range(100, 100 - n_d * 20, -20))  # [100, 80, 60, ...] para cualquier n_d
        return _season_result(
            driver_ids=[f"d{i}" for i in range(n_d)],
            pts_start=pts[:n_d],
            pts_sims=rng.uniform(0, 50, (n_sims, n_d)),
        )

    def test_n_sims(self):
        r = self._basic(n_sims=200)
        assert r.n_sims == 200

    def test_n_drivers(self):
        r = self._basic(n_d=3)
        assert r.n_drivers == 3

    def test_driver_total_shape(self):
        r = self._basic(n_sims=50, n_d=4)
        assert r.driver_total.shape == (50, 4)

    def test_driver_total_equals_start_plus_sims(self):
        r = self._basic(n_sims=10, n_d=3)
        expected = r.driver_points_start[np.newaxis, :] + r.driver_points_sims
        np.testing.assert_array_almost_equal(r.driver_total, expected)

    def test_constructor_total_shape(self):
        rng = np.random.default_rng(1)
        r = _season_result(
            driver_ids=["d0", "d1"],
            pts_start=[100.0, 80.0],
            pts_sims=rng.uniform(0, 50, (20, 2)),
            constructor_ids=["teamA"],
            constr_pts_start=[180.0],
            constr_pts_sims=rng.uniform(0, 100, (20, 1)),
        )
        assert r.constructor_total.shape == (20, 1)


# ---------------------------------------------------------------------------
# SeasonSimResult — driver_title_probability
# ---------------------------------------------------------------------------

class TestDriverTitleProbability:
    def test_sums_to_one(self):
        rng = np.random.default_rng(42)
        r = _season_result(
            driver_ids=["d0", "d1", "d2"],
            pts_start=[100.0, 90.0, 80.0],
            pts_sims=rng.uniform(0, 100, (500, 3)),
        )
        probs = r.driver_title_probability()
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)

    def test_all_probs_in_0_1(self):
        rng = np.random.default_rng(7)
        r = _season_result(
            driver_ids=["a", "b", "c"],
            pts_start=[50.0, 50.0, 50.0],
            pts_sims=rng.uniform(0, 100, (200, 3)),
        )
        for prob in r.driver_title_probability().values():
            assert 0.0 <= prob <= 1.0

    def test_leader_wins_most_simulations(self):
        """Piloto con muchos más puntos de inicio debería ganar la mayoría."""
        rng = np.random.default_rng(5)
        r = _season_result(
            driver_ids=["leader", "other"],
            pts_start=[500.0, 10.0],   # líder inalcanzable
            pts_sims=rng.uniform(0, 10, (1000, 2)),
        )
        probs = r.driver_title_probability()
        assert probs["leader"] > 0.95

    def test_driver_ids_in_output(self):
        rng = np.random.default_rng(1)
        r = _season_result(
            driver_ids=["verstappen", "norris"],
            pts_start=[200.0, 180.0],
            pts_sims=rng.uniform(0, 50, (100, 2)),
        )
        assert set(r.driver_title_probability().keys()) == {"verstappen", "norris"}


# ---------------------------------------------------------------------------
# SeasonSimResult — constructor_title_probability
# ---------------------------------------------------------------------------

class TestConstructorTitleProbability:
    def test_no_constructors_returns_empty(self):
        rng = np.random.default_rng(0)
        r = _season_result(
            driver_ids=["d0", "d1"],
            pts_start=[100.0, 80.0],
            pts_sims=rng.uniform(0, 50, (100, 2)),
        )
        assert r.constructor_title_probability() == {}

    def test_sums_to_one(self):
        rng = np.random.default_rng(3)
        r = _season_result(
            driver_ids=["d0", "d1", "d2", "d3"],
            pts_start=[50.0, 50.0, 50.0, 50.0],
            pts_sims=rng.uniform(0, 50, (200, 4)),
            constructor_ids=["teamA", "teamB"],
            constr_pts_start=[100.0, 100.0],
            constr_pts_sims=rng.uniform(0, 100, (200, 2)),
        )
        probs = r.constructor_title_probability()
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# SeasonSimResult — expected_points
# ---------------------------------------------------------------------------

class TestExpectedPoints:
    def test_expected_total_includes_start(self):
        pts_start = np.array([100.0, 50.0])
        pts_sims  = np.full((10, 2), 10.0)
        r = _season_result(
            driver_ids=["a", "b"],
            pts_start=pts_start.tolist(),
            pts_sims=pts_sims,
        )
        exp = r.driver_expected_total_points()
        assert exp["a"] == pytest.approx(110.0)
        assert exp["b"] == pytest.approx(60.0)

    def test_expected_additional_excludes_start(self):
        pts_sims = np.full((5, 2), 20.0)
        r = _season_result(
            driver_ids=["a", "b"],
            pts_start=[100.0, 100.0],
            pts_sims=pts_sims,
        )
        exp = r.driver_expected_additional_points()
        assert exp["a"] == pytest.approx(20.0)
        assert exp["b"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# SeasonSimResult — magic_number
# ---------------------------------------------------------------------------

class TestMagicNumber:
    def _result_for_magic(self, leader_pts: float, second_pts: float, n_rounds: int) -> SeasonSimResult:
        rng = np.random.default_rng(0)
        r = _season_result(
            driver_ids=["leader", "second"],
            pts_start=[leader_pts, second_pts],
            pts_sims=rng.uniform(0, 5, (10, 2)),
        )
        r.n_rounds_simulated = n_rounds
        return r

    def test_unknown_driver_returns_none(self):
        r = self._result_for_magic(300.0, 200.0, 5)
        assert r.magic_number("unknown") is None

    def test_clinched_leader_positive(self):
        """Líder con 400, segundo con 200, 0 rondas → magic_number = 200."""
        r = self._result_for_magic(400.0, 200.0, 0)
        mn = r.magic_number("leader", max_pts_per_round=33.0)
        assert mn == pytest.approx(200.0)

    def test_not_clinched_negative(self):
        """Líder con 200, segundo con 190, 10 rondas → second puede alcanzar 190+330."""
        r = self._result_for_magic(200.0, 190.0, 10)
        mn = r.magic_number("leader", max_pts_per_round=33.0)
        assert mn < 0

    def test_zero_rounds_max_is_just_second_current(self):
        """Sin rondas restantes, magic_number = líder_pts - segundo_pts."""
        r = self._result_for_magic(300.0, 250.0, 0)
        mn = r.magic_number("leader")
        assert mn == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# SeasonSimResult — to_driver_summary_df / to_constructor_summary_df
# ---------------------------------------------------------------------------

class TestSummaryDataFrames:
    def _result(self):
        rng = np.random.default_rng(99)
        return _season_result(
            driver_ids=["d0", "d1", "d2"],
            pts_start=[300.0, 200.0, 100.0],
            pts_sims=rng.uniform(0, 50, (500, 3)),
        )

    def test_driver_summary_columns(self):
        df = self._result().to_driver_summary_df()
        assert {"driver_id", "current_points", "title_prob", "expected_total"}.issubset(df.columns)

    def test_driver_summary_sorted_by_prob(self):
        df = self._result().to_driver_summary_df()
        probs = df["title_prob"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_driver_summary_n_rows(self):
        df = self._result().to_driver_summary_df()
        assert len(df) == 3

    def test_constructor_summary_empty_without_map(self):
        df = self._result().to_constructor_summary_df()
        assert len(df) == 0
        assert "constructor_id" in df.columns

    def test_constructor_summary_with_constructors(self):
        rng = np.random.default_rng(1)
        r = _season_result(
            driver_ids=["d0", "d1"],
            pts_start=[100.0, 80.0],
            pts_sims=rng.uniform(0, 50, (100, 2)),
            constructor_ids=["teamA", "teamB"],
            constr_pts_start=[100.0, 80.0],
            constr_pts_sims=rng.uniform(0, 100, (100, 2)),
        )
        df = r.to_constructor_summary_df()
        assert len(df) == 2
        assert "title_prob" in df.columns


# ---------------------------------------------------------------------------
# _build_driver_arrays
# ---------------------------------------------------------------------------

class TestBuildDriverArrays:
    def test_alignment_with_standings(self):
        standings = [
            _driver_standing("d0", 1, 100.0),
            _driver_standing("d1", 2, 80.0),
        ]
        df = _strength_df(["d0", "d1"], means=[2.0, 1.0], sds=[0.3, 0.5])
        ids, strengths, sds, pts = _build_driver_arrays(df, standings)
        assert ids == ["d0", "d1"]
        assert strengths[0] == pytest.approx(2.0)
        assert strengths[1] == pytest.approx(1.0)
        assert pts[0] == pytest.approx(100.0)
        assert pts[1] == pytest.approx(80.0)

    def test_missing_driver_gets_zero_mean(self):
        standings = [_driver_standing("d0", 1, 50.0), _driver_standing("d_new", 2, 0.0)]
        df = _strength_df(["d0"], means=[1.5], sds=[0.4])
        ids, strengths, sds, pts = _build_driver_arrays(df, standings)
        d_new_idx = ids.index("d_new")
        assert strengths[d_new_idx] == pytest.approx(0.0)

    def test_missing_driver_gets_unknown_sd(self):
        standings = [_driver_standing("d0", 1, 50.0), _driver_standing("d_new", 2, 0.0)]
        df = _strength_df(["d0"], means=[1.5], sds=[0.4])
        _, _, sds, _ = _build_driver_arrays(df, standings)
        # sds != None because d0 has a known sd
        assert sds is not None
        d_new_idx = standings.index(next(s for s in standings if s.driver_id == "d_new"))
        assert sds[d_new_idx] == pytest.approx(_UNKNOWN_STRENGTH_SD)

    def test_all_unknown_sds_returns_none(self):
        """Si NO se pasa sd en el DataFrame, todos quedan como _UNKNOWN_STRENGTH_SD → sds=None."""
        standings = [_driver_standing("d0", 1, 50.0), _driver_standing("d1", 2, 40.0)]
        df = _strength_df(["d0", "d1"], means=[1.0, 0.5])  # sin sd
        _, _, sds, _ = _build_driver_arrays(df, standings)
        assert sds is None

    def test_empty_strength_df_all_unknown(self):
        standings = [_driver_standing("d0", 1, 100.0)]
        df = pd.DataFrame(columns=["driver_id", "mean"])
        _, strengths, sds, pts = _build_driver_arrays(df, standings)
        assert sds is None
        assert strengths[0] == pytest.approx(0.0)

    def test_points_preserved(self):
        standings = [_driver_standing("d0", 1, 123.0)]
        df = _strength_df(["d0"], means=[0.5], sds=[0.2])
        _, _, _, pts = _build_driver_arrays(df, standings)
        assert pts[0] == pytest.approx(123.0)


# ---------------------------------------------------------------------------
# _build_constructor_arrays
# ---------------------------------------------------------------------------

class TestBuildConstructorArrays:
    def test_no_standings_returns_empty(self):
        ids, idx, pts = _build_constructor_arrays([], None, ["d0"])
        assert ids == []
        assert idx is None
        assert len(pts) == 0

    def test_no_driver_map_returns_none_idx(self):
        standings = [_constr_standing("teamA", 1, 100.0)]
        ids, idx, pts = _build_constructor_arrays(standings, None, ["d0"])
        assert ids == ["teamA"]
        assert idx is None

    def test_correct_idx_mapping(self):
        standings = [
            _constr_standing("teamA", 1, 100.0),
            _constr_standing("teamB", 2, 80.0),
        ]
        driver_map = {"d0": "teamA", "d1": "teamB", "d2": "teamA"}
        ids, idx, pts = _build_constructor_arrays(standings, driver_map, ["d0", "d1", "d2"])
        assert ids == ["teamA", "teamB"]
        assert idx[0] == 0   # d0 → teamA → index 0
        assert idx[1] == 1   # d1 → teamB → index 1
        assert idx[2] == 0   # d2 → teamA → index 0

    def test_unmapped_driver_gets_minus1(self):
        standings = [_constr_standing("teamA", 1, 100.0)]
        driver_map = {}  # d0 no está en el mapa
        _, idx, _ = _build_constructor_arrays(standings, driver_map, ["d0"])
        assert idx is not None
        assert idx[0] == -1

    def test_constructor_points_preserved(self):
        standings = [_constr_standing("teamA", 1, 250.0)]
        _, _, pts = _build_constructor_arrays(standings, None, ["d0"])
        assert pts[0] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# _build_dnf_array
# ---------------------------------------------------------------------------

class TestBuildDnfArray:
    def test_none_returns_all_default(self):
        arr = _build_dnf_array(None, ["d0", "d1", "d2"])
        np.testing.assert_array_almost_equal(arr, [_DEFAULT_DNF_RATE] * 3)

    def test_known_rates_used(self):
        arr = _build_dnf_array({"d0": 0.05, "d1": 0.15}, ["d0", "d1"])
        assert arr[0] == pytest.approx(0.05)
        assert arr[1] == pytest.approx(0.15)

    def test_missing_driver_uses_default(self):
        arr = _build_dnf_array({"d0": 0.05}, ["d0", "d_new"])
        assert arr[1] == pytest.approx(_DEFAULT_DNF_RATE)

    def test_empty_dict_all_default(self):
        arr = _build_dnf_array({}, ["d0", "d1"])
        np.testing.assert_array_almost_equal(arr, [_DEFAULT_DNF_RATE, _DEFAULT_DNF_RATE])


# ---------------------------------------------------------------------------
# simulate_championship — interfaz y casos extremos
# ---------------------------------------------------------------------------

class TestSimulateChampionship:
    def _standings(self, n: int = 3) -> list[DriverStanding]:
        pts = [100.0, 80.0, 60.0, 40.0, 20.0][:n]
        return [_driver_standing(f"d{i}", i + 1, pts[i]) for i in range(n)]

    def _constr_standings(self, n: int = 2) -> list[ConstructorStanding]:
        pts = [200.0, 150.0][:n]
        return [_constr_standing(f"team{i}", i + 1, pts[i]) for i in range(n)]

    def _strength(self, driver_ids: list[str]) -> pd.DataFrame:
        means = np.linspace(1.5, 0.0, len(driver_ids))
        sds   = np.full(len(driver_ids), 0.3)
        return _strength_df(driver_ids, means.tolist(), sds.tolist())

    def test_returns_required_keys(self):
        standings = self._standings(3)
        result = simulate_championship(
            strength=self._strength(["d0", "d1", "d2"]),
            remaining_rounds=[_round(1)],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=50,
            seed=0,
        )
        assert "driver_title_probs" in result
        assert "constructor_title_probs" in result
        assert "driver_expected_points" in result
        assert "driver_points_dist" in result

    def test_zero_rounds_returns_current_standings(self):
        """Sin rondas, la distribución de puntos es determinista (solo los actuales)."""
        standings = self._standings(3)
        result = simulate_championship(
            strength=self._strength(["d0", "d1", "d2"]),
            remaining_rounds=[],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=100,
            seed=1,
        )
        # Con 0 rondas y d0 con 100 pts, d1 con 80, d0 debe ganar todas las sims
        probs = result["driver_title_probs"]
        assert probs["d0"] == pytest.approx(1.0)
        assert probs["d1"] == pytest.approx(0.0)

    def test_driver_title_probs_sum_to_one(self):
        standings = self._standings(4)
        result = simulate_championship(
            strength=self._strength(["d0", "d1", "d2", "d3"]),
            remaining_rounds=[_round(1), _round(2)],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=200,
            seed=2,
        )
        total = sum(result["driver_title_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_driver_points_dist_shape(self):
        n_sims = 150
        standings = self._standings(3)
        result = simulate_championship(
            strength=self._strength(["d0", "d1", "d2"]),
            remaining_rounds=[_round(1)],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=n_sims,
            seed=3,
        )
        dist = result["driver_points_dist"]
        assert dist.shape == (n_sims, 3)

    def test_driver_points_dist_columns_are_driver_ids(self):
        standings = self._standings(3)
        result = simulate_championship(
            strength=self._strength(["d0", "d1", "d2"]),
            remaining_rounds=[_round(1)],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=50,
            seed=4,
        )
        assert set(result["driver_points_dist"].columns) == {"d0", "d1", "d2"}

    def test_expected_points_keys_match_drivers(self):
        standings = self._standings(3)
        result = simulate_championship(
            strength=self._strength(["d0", "d1", "d2"]),
            remaining_rounds=[_round(1)],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=50,
            seed=5,
        )
        assert set(result["driver_expected_points"].keys()) == {"d0", "d1", "d2"}

    def test_constructor_probs_empty_without_map(self):
        standings = self._standings(3)
        result = simulate_championship(
            strength=self._strength(["d0", "d1", "d2"]),
            remaining_rounds=[_round(1)],
            driver_standings=standings,
            constructor_standings=self._constr_standings(),
            driver_constructor_map=None,
            n_sims=50,
            seed=6,
        )
        assert result["constructor_title_probs"] == {}

    def test_constructor_probs_with_map(self):
        standings = self._standings(4)
        constr = self._constr_standings(2)
        driver_map = {"d0": "team0", "d1": "team0", "d2": "team1", "d3": "team1"}
        result = simulate_championship(
            strength=self._strength(["d0", "d1", "d2", "d3"]),
            remaining_rounds=[_round(1)],
            driver_standings=standings,
            constructor_standings=constr,
            driver_constructor_map=driver_map,
            n_sims=200,
            seed=7,
        )
        probs = result["constructor_title_probs"]
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_sprint_round_adds_more_points(self):
        """Un fin de semana sprint produce más puntos esperados que uno normal."""
        standings = self._standings(3)
        strength = self._strength(["d0", "d1", "d2"])

        res_normal = simulate_championship(
            strength=strength,
            remaining_rounds=[_round(1, has_sprint=False)],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=500,
            seed=8,
        )
        res_sprint = simulate_championship(
            strength=strength,
            remaining_rounds=[_round(1, has_sprint=True)],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=500,
            seed=8,
        )
        # Suma total de puntos en la distribución debe ser mayor en sprint
        total_normal = res_normal["driver_points_dist"].values.sum()
        total_sprint = res_sprint["driver_points_dist"].values.sum()
        # Los puntos de partida son iguales; el sprint añade los adicionales de la sprint race
        assert total_sprint > total_normal

    def test_reproducible_with_same_seed(self):
        standings = self._standings(3)
        strength = self._strength(["d0", "d1", "d2"])

        r1 = simulate_championship(
            strength=strength, remaining_rounds=[_round(1)],
            driver_standings=standings, constructor_standings=[],
            n_sims=100, seed=42,
        )
        r2 = simulate_championship(
            strength=strength, remaining_rounds=[_round(1)],
            driver_standings=standings, constructor_standings=[],
            n_sims=100, seed=42,
        )
        pd.testing.assert_frame_equal(r1["driver_points_dist"], r2["driver_points_dist"])

    def test_dominant_driver_wins_most_titles(self):
        """Piloto con fortaleza mucho mayor debe tener probabilidad de título alta."""
        standings = [
            _driver_standing("star",  1, 200.0),
            _driver_standing("other", 2, 100.0),
        ]
        strength = _strength_df(
            driver_ids=["star", "other"],
            means=[5.0, 0.0],
            sds=[0.2, 0.2],
        )
        result = simulate_championship(
            strength=strength,
            remaining_rounds=[_round(i) for i in range(1, 6)],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=1000,
            seed=99,
        )
        assert result["driver_title_probs"]["star"] > 0.85

    def test_no_strength_data_simulation_runs(self):
        """Con DataFrame vacío (sin modelo aún), la simulación debe completarse."""
        standings = self._standings(3)
        result = simulate_championship(
            strength=pd.DataFrame(columns=["driver_id", "mean"]),
            remaining_rounds=[_round(1)],
            driver_standings=standings,
            constructor_standings=[],
            n_sims=50,
            seed=10,
        )
        assert "driver_title_probs" in result
        assert sum(result["driver_title_probs"].values()) == pytest.approx(1.0, abs=1e-9)
