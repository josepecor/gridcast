"""
Tests de models/strength.py

Sin llamadas a red. DataFrames sintéticos que imitan la salida de pace.compute_pace_features().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.models.strength import (
    _WIDE_PRIOR_SD,
    _MIN_SD,
    fit_all,
    fit_driver_strengths,
    fit_engine_strengths,
    fit_team_strengths,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _pace_row(
    abbreviation: str,
    team: str,
    round_: int,
    quali_pace: float | None = None,
    long_run_pace: float | None = None,
    season: int = 2026,
) -> dict:
    return {
        "season":         season,
        "round":          round_,
        "abbreviation":   abbreviation,
        "team_name":      team,
        "quali_pace_s":   quali_pace,
        "long_run_pace_s": long_run_pace,
        "session_type":   "FP2",
        "long_run_n":     5,
        "fp_best_lap_s":  (quali_pace or 90.0) + 0.3,
    }


def _make_pace(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _simple_pace(fast_gap: float = 0.3) -> pd.DataFrame:
    """
    2 equipos, 2 pilotos cada uno, 3 rondas.
    Equipo A: VER es más rápido que PER por fast_gap segundos.
    Equipo B: NOR y PIA son iguales.
    """
    rows = []
    for r in range(1, 4):
        base_a = 90.0
        base_b = 91.0  # equipo B es más lento (chasis diferente)
        rows += [
            _pace_row("VER", "teamA", r, quali_pace=base_a,              long_run_pace=base_a),
            _pace_row("PER", "teamA", r, quali_pace=base_a + fast_gap,   long_run_pace=base_a + fast_gap),
            _pace_row("NOR", "teamB", r, quali_pace=base_b,              long_run_pace=base_b),
            _pace_row("PIA", "teamB", r, quali_pace=base_b,              long_run_pace=base_b),
        ]
    return _make_pace(rows)


# ---------------------------------------------------------------------------
# fit_driver_strengths — validación de inputs
# ---------------------------------------------------------------------------

class TestFitDriverStrengthsValidation:
    def test_missing_abbreviation_raises(self):
        df = _simple_pace().drop(columns=["abbreviation"])
        with pytest.raises(ValueError, match="abbreviation"):
            fit_driver_strengths(df)

    def test_missing_team_name_raises(self):
        df = _simple_pace().drop(columns=["team_name"])
        with pytest.raises(ValueError, match="team_name"):
            fit_driver_strengths(df)

    def test_missing_signal_col_raises(self):
        df = _simple_pace().drop(columns=["quali_pace_s"])
        with pytest.raises(ValueError, match="quali_pace_s"):
            fit_driver_strengths(df, primary_signal="quali")

    def test_invalid_primary_signal_raises(self):
        with pytest.raises(ValueError, match="primary_signal"):
            fit_driver_strengths(_simple_pace(), primary_signal="invalid")

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["season", "round", "abbreviation", "team_name", "quali_pace_s"])
        result = fit_driver_strengths(df)
        assert result.empty
        assert set(result.columns) >= {"driver_id", "mean", "sd", "n_obs"}

    def test_all_nan_signal_returns_empty(self):
        df = _simple_pace()
        df["quali_pace_s"] = float("nan")
        result = fit_driver_strengths(df)
        assert result.empty


# ---------------------------------------------------------------------------
# fit_driver_strengths — salida correcta
# ---------------------------------------------------------------------------

class TestFitDriverStrengthsOutput:
    def test_output_columns(self):
        result = fit_driver_strengths(_simple_pace())
        assert set(result.columns) == {"driver_id", "mean", "sd", "n_obs"}

    def test_n_drivers(self):
        result = fit_driver_strengths(_simple_pace())
        assert len(result) == 4   # VER, PER, NOR, PIA

    def test_mean_centered_at_zero(self):
        result = fit_driver_strengths(_simple_pace())
        assert result["mean"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_sd_positive(self):
        result = fit_driver_strengths(_simple_pace())
        assert (result["sd"] > 0).all()

    def test_sd_ge_min_sd(self):
        result = fit_driver_strengths(_simple_pace())
        assert (result["sd"] >= _MIN_SD).all()

    def test_n_obs_correct(self):
        result = fit_driver_strengths(_simple_pace())
        # 3 rondas por piloto
        assert (result["n_obs"] == 3).all()

    def test_faster_driver_higher_mean(self):
        """VER más rápido que PER en todas las rondas → VER debe tener mean > PER."""
        result = fit_driver_strengths(_simple_pace(fast_gap=0.5))
        r = result.set_index("driver_id")
        assert r.loc["VER", "mean"] > r.loc["PER", "mean"]

    def test_gap_magnitude(self):
        """Con gap de 0.4 s, la diferencia VER–PER en strength debe ser ~0.4."""
        gap = 0.4
        result = fit_driver_strengths(_simple_pace(fast_gap=gap))
        r = result.set_index("driver_id")
        diff = r.loc["VER", "mean"] - r.loc["PER", "mean"]
        assert diff == pytest.approx(gap, abs=1e-9)

    def test_equal_teammates_have_same_mean(self):
        """NOR y PIA con mismos tiempos → same strength."""
        result = fit_driver_strengths(_simple_pace())
        r = result.set_index("driver_id")
        assert r.loc["NOR", "mean"] == pytest.approx(r.loc["PIA", "mean"], abs=1e-9)

    def test_works_without_season_col(self):
        """Debe funcionar aunque no haya columna 'season'."""
        df = _simple_pace().drop(columns=["season"])
        result = fit_driver_strengths(df)
        assert len(result) == 4

    def test_race_signal(self):
        result = fit_driver_strengths(_simple_pace(), primary_signal="race")
        assert len(result) == 4
        r = result.set_index("driver_id")
        assert r.loc["VER", "mean"] > r.loc["PER", "mean"]


# ---------------------------------------------------------------------------
# fit_driver_strengths — min_rounds y sd
# ---------------------------------------------------------------------------

class TestFitDriverStrengthsMinRounds:
    def test_driver_with_one_round_included_by_default(self):
        """min_rounds=1 (default) → piloto con solo 1 ronda se incluye."""
        rows = [
            _pace_row("VER", "teamA", 1, quali_pace=90.0),
            _pace_row("PER", "teamA", 1, quali_pace=90.5),
            # Ronda 2 solo VER (PER no tiene datos)
            _pace_row("VER", "teamA", 2, quali_pace=90.1),
            _pace_row("NOR", "teamB", 1, quali_pace=91.0),
            _pace_row("PIA", "teamB", 1, quali_pace=91.0),
            _pace_row("NOR", "teamB", 2, quali_pace=91.0),
            _pace_row("PIA", "teamB", 2, quali_pace=91.0),
        ]
        result = fit_driver_strengths(_make_pace(rows))
        # PER solo tiene ronda 1 (en ronda 2 no hay compañero válido) → n_obs=1
        per = result[result["driver_id"] == "PER"]
        assert len(per) == 1
        assert per.iloc[0]["n_obs"] == 1

    def test_min_rounds_filters_drivers(self):
        """Con min_rounds=2, PER con 1 ronda no aparece."""
        rows = [
            _pace_row("VER", "teamA", 1, quali_pace=90.0),
            _pace_row("PER", "teamA", 1, quali_pace=90.5),
            _pace_row("NOR", "teamB", 1, quali_pace=91.0),
            _pace_row("PIA", "teamB", 1, quali_pace=91.0),
        ]
        result = fit_driver_strengths(_make_pace(rows), min_rounds=2)
        assert result.empty  # nadie tiene 2 rondas

    def test_single_driver_per_team_excluded(self):
        """Una ronda donde el equipo tiene solo 1 piloto no contribuye."""
        rows = [
            _pace_row("VER", "teamA", 1, quali_pace=90.0),   # solo VER en r1
            _pace_row("VER", "teamA", 2, quali_pace=90.0),
            _pace_row("PER", "teamA", 2, quali_pace=90.3),
            _pace_row("NOR", "teamB", 1, quali_pace=91.0),
            _pace_row("PIA", "teamB", 1, quali_pace=91.0),
            _pace_row("NOR", "teamB", 2, quali_pace=91.0),
            _pace_row("PIA", "teamB", 2, quali_pace=91.0),
        ]
        result = fit_driver_strengths(_make_pace(rows))
        ver = result[result["driver_id"] == "VER"]
        # VER solo contribuye en ronda 2 (ronda 1 no hay compañero)
        assert ver.iloc[0]["n_obs"] == 1

    def test_sd_wide_prior_for_single_obs(self):
        """Piloto con n_obs=1 no puede calcular std → sd = _WIDE_PRIOR_SD."""
        rows = [
            _pace_row("VER", "teamA", 1, quali_pace=90.0),
            _pace_row("PER", "teamA", 1, quali_pace=90.5),
            _pace_row("NOR", "teamB", 1, quali_pace=91.0),
            _pace_row("PIA", "teamB", 1, quali_pace=91.0),
        ]
        result = fit_driver_strengths(_make_pace(rows))
        for _, row in result.iterrows():
            assert row["sd"] == pytest.approx(_WIDE_PRIOR_SD, abs=1e-9)

    def test_sd_decreases_with_more_rounds(self):
        """Más rondas → error estándar más pequeño (se usa std/sqrt(n)).

        Datos deterministas: VER siempre 0.1 s más rápido que PER, con variación
        constante ronda a ronda (alternando +0.1 y -0.1 alrededor del gap medio).
        Así std es fijo, y sd = std/sqrt(n) decrece estrictamente con más n.
        """
        gaps = [0.3, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1]  # alterna

        def _make_multi_round(n: int) -> pd.DataFrame:
            rows = []
            for r in range(1, n + 1):
                g = gaps[(r - 1) % len(gaps)]
                rows += [
                    _pace_row("VER", "teamA", r, quali_pace=90.0),
                    _pace_row("PER", "teamA", r, quali_pace=90.0 + g),
                ]
            return _make_pace(rows)

        res_3  = fit_driver_strengths(_make_multi_round(4))   # 4 obs (parejo: 2×0.3 + 2×0.1)
        res_10 = fit_driver_strengths(_make_multi_round(10))  # 10 obs
        sd_4  = res_3[res_3["driver_id"] == "VER"]["sd"].values[0]
        sd_10 = res_10[res_10["driver_id"] == "VER"]["sd"].values[0]
        assert sd_10 < sd_4


# ---------------------------------------------------------------------------
# fit_driver_strengths — múltiples temporadas
# ---------------------------------------------------------------------------

class TestMultiSeason:
    def test_two_seasons_stacked(self):
        rows_2025 = [_pace_row("VER", "teamA", r, quali_pace=90.0, season=2025)
                     for r in range(1, 4)]
        rows_2025 += [_pace_row("PER", "teamA", r, quali_pace=90.5, season=2025)
                      for r in range(1, 4)]
        rows_2026 = [_pace_row("VER", "teamA", r, quali_pace=90.0, season=2026)
                     for r in range(1, 3)]
        rows_2026 += [_pace_row("NOR", "teamA", r, quali_pace=90.8, season=2026)
                      for r in range(1, 3)]
        df = _make_pace(rows_2025 + rows_2026)
        result = fit_driver_strengths(df)
        assert "VER" in result["driver_id"].values
        assert result["mean"].mean() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# fit_team_strengths
# ---------------------------------------------------------------------------

class TestFitTeamStrengths:
    def test_output_columns(self):
        pace = _simple_pace()
        driver = fit_driver_strengths(pace)
        result = fit_team_strengths(pace, driver)
        assert set(result.columns) == {"team_id", "mean", "sd", "n_obs", "engine_id"}

    def test_n_teams(self):
        pace = _simple_pace()
        driver = fit_driver_strengths(pace)
        result = fit_team_strengths(pace, driver)
        assert len(result) == 2   # teamA, teamB

    def test_faster_team_higher_mean(self):
        """teamA (pace 90s) es más rápido que teamB (pace 91s)."""
        pace = _simple_pace()
        driver = fit_driver_strengths(pace)
        result = fit_team_strengths(pace, driver).set_index("team_id")
        assert result.loc["teamA", "mean"] > result.loc["teamB", "mean"]

    def test_engine_map_applied(self):
        pace = _simple_pace()
        driver = fit_driver_strengths(pace)
        engine_map = {"teamA": "engine_X", "teamB": "engine_Y"}
        result = fit_team_strengths(pace, driver, engine_map=engine_map).set_index("team_id")
        assert result.loc["teamA", "engine_id"] == "engine_X"
        assert result.loc["teamB", "engine_id"] == "engine_Y"

    def test_no_engine_map_nan(self):
        pace = _simple_pace()
        driver = fit_driver_strengths(pace)
        result = fit_team_strengths(pace, driver, engine_map=None)
        assert result["engine_id"].isna().all()

    def test_sd_positive(self):
        pace = _simple_pace()
        driver = fit_driver_strengths(pace)
        result = fit_team_strengths(pace, driver)
        assert (result["sd"] > 0).all()

    def test_missing_team_name_raises(self):
        pace = _simple_pace().drop(columns=["team_name"])
        driver = fit_driver_strengths(_simple_pace())
        with pytest.raises(ValueError, match="team_name"):
            fit_team_strengths(pace, driver)

    def test_race_signal(self):
        pace = _simple_pace()
        driver = fit_driver_strengths(pace, primary_signal="race")
        result = fit_team_strengths(pace, driver, primary_signal="race")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# fit_engine_strengths
# ---------------------------------------------------------------------------

class TestFitEngineStrengths:
    def _team_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team_id": ["mercedes", "aston_martin", "redbull"],
            "mean":    [0.5, 0.3, 0.2],
            "sd":      [0.1, 0.2, 0.15],
            "n_obs":   [10, 8, 12],
        })

    def test_output_columns(self):
        engine_map = {"mercedes": "mercedes", "aston_martin": "mercedes", "redbull": "honda"}
        result = fit_engine_strengths(self._team_df(), engine_map)
        assert set(result.columns) == {"engine_id", "mean", "sd", "n_teams"}

    def test_n_engines(self):
        engine_map = {"mercedes": "mercedes", "aston_martin": "mercedes", "redbull": "honda"}
        result = fit_engine_strengths(self._team_df(), engine_map)
        assert len(result) == 2   # mercedes, honda

    def test_engine_with_two_teams_has_pooled_mean(self):
        """Mercedes tiene 2 equipos; su mean debe ser entre los dos."""
        engine_map = {"mercedes": "mercedes", "aston_martin": "mercedes", "redbull": "honda"}
        result = fit_engine_strengths(self._team_df(), engine_map)
        r = result.set_index("engine_id")
        # mercedes teams: 0.5 y 0.3 (ponderados por n_obs 10 y 8)
        # wmean = (0.5*10 + 0.3*8) / 18 ≈ 0.411
        assert 0.3 < r.loc["mercedes", "mean"] < 0.5

    def test_n_teams_correct(self):
        engine_map = {"mercedes": "mercedes", "aston_martin": "mercedes", "redbull": "honda"}
        result = fit_engine_strengths(self._team_df(), engine_map).set_index("engine_id")
        assert result.loc["mercedes", "n_teams"] == 2
        assert result.loc["honda",    "n_teams"] == 1

    def test_single_team_per_engine_gets_wide_prior_sd(self):
        """Un único equipo cliente → sd amplio (_WIDE_PRIOR_SD)."""
        engine_map = {"redbull": "honda"}
        result = fit_engine_strengths(
            self._team_df()[self._team_df()["team_id"] == "redbull"],
            engine_map,
        )
        assert result.iloc[0]["sd"] == pytest.approx(_WIDE_PRIOR_SD)

    def test_unmapped_teams_excluded(self):
        """Equipos no en engine_map no aparecen en el resultado."""
        engine_map = {"mercedes": "mercedes"}  # redbull y aston_martin sin mapeo
        result = fit_engine_strengths(self._team_df(), engine_map)
        assert len(result) == 1
        assert result.iloc[0]["engine_id"] == "mercedes"

    def test_empty_team_df_returns_empty(self):
        result = fit_engine_strengths(pd.DataFrame(columns=["team_id", "mean", "sd", "n_obs"]), {})
        assert result.empty

    def test_no_matching_teams_returns_empty(self):
        engine_map = {"no_team": "some_engine"}
        result = fit_engine_strengths(self._team_df(), engine_map)
        assert result.empty

    def test_missing_required_cols_raises(self):
        bad = self._team_df().drop(columns=["mean"])
        with pytest.raises(ValueError, match="mean"):
            fit_engine_strengths(bad, {"mercedes": "mercedes"})


# ---------------------------------------------------------------------------
# fit_all
# ---------------------------------------------------------------------------

class TestFitAll:
    def test_returns_three_keys(self):
        result = fit_all(_simple_pace())
        assert set(result.keys()) == {"driver", "team", "engine"}

    def test_driver_output_correct(self):
        result = fit_all(_simple_pace())
        assert len(result["driver"]) == 4

    def test_engine_empty_without_map(self):
        result = fit_all(_simple_pace(), engine_map=None)
        assert result["engine"].empty

    def test_engine_filled_with_map(self):
        engine_map = {"teamA": "engine_alpha", "teamB": "engine_beta"}
        result = fit_all(_simple_pace(), engine_map=engine_map)
        assert len(result["engine"]) == 2

    def test_consistent_driver_team_directions(self):
        """El equipo más rápido en team strength debe coincidir con el equipo
        cuyos pilotos tienen mayor mean en driver strength."""
        pace = _simple_pace(fast_gap=0.3)
        result = fit_all(pace)

        driver = result["driver"].set_index("driver_id")
        team   = result["team"].set_index("team_id")

        # teamA tiene VER (rápido) y PER (más lento); teamB tiene NOR y PIA (iguales)
        # teamA media pace = 90 + (90+0.3)/2 = 90.15 → car_speed = -90.15
        # teamB media pace = 91 → car_speed = -91
        # teamA es más rápido → team mean más alto
        assert team.loc["teamA", "mean"] > team.loc["teamB", "mean"]

        # VER tiene el mean más alto entre todos
        assert driver.loc["VER", "mean"] == result["driver"]["mean"].max()
