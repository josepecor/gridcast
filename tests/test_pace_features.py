"""
Tests de features/pace.py y features/teammate_gaps.py.

Sin llamadas a FastF1 — todo con DataFrames sintéticos que imitan el formato real.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from gridcast.features.pace import (
    _add_fuel_correction,
    _extract_long_run_pace,
    _extract_session_best,
    _filter_long_stints,
    _lap_times_to_seconds,
    _remove_stint_outliers,
    _to_seconds,
    compute_pace_features,
)
from gridcast.features.teammate_gaps import compute_teammate_gaps


# ---------------------------------------------------------------------------
# Helpers para construir DataFrames sintéticos
# ---------------------------------------------------------------------------

def _make_laps(
    driver: str,
    team: str,
    lap_times_s: list[float],
    stint: int = 1,
    tyre_life_start: int = 1,
    session: str = "FP2",
    season: int = 2026,
    round_: int = 3,
    track_status: str = "1",
) -> pd.DataFrame:
    """Crea un DataFrame de vueltas que imita la salida de fastf1_client.get_laps()."""
    n = len(lap_times_s)
    return pd.DataFrame({
        "Driver":       [driver] * n,
        "Team":         [team] * n,
        "LapNumber":    list(range(1, n + 1)),
        "LapTime":      pd.to_timedelta([f"{t:.3f}s" for t in lap_times_s]),
        "Stint":        [stint] * n,
        "TyreLife":     list(range(tyre_life_start, tyre_life_start + n)),
        "TrackStatus":  [track_status] * n,
        "IsAccurate":   [True] * n,
        "season":       [season] * n,
        "round":        [round_] * n,
        "session_type": [session] * n,
    })


def _concat(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# _to_seconds
# ---------------------------------------------------------------------------

class TestToSeconds:
    def test_timedelta(self):
        assert _to_seconds(timedelta(seconds=90.5)) == pytest.approx(90.5)

    def test_pd_timedelta(self):
        assert _to_seconds(pd.Timedelta("1m30.5s")) == pytest.approx(90.5)

    def test_nan_returns_nan(self):
        assert math.isnan(_to_seconds(float("nan")))

    def test_nat_returns_nan(self):
        assert math.isnan(_to_seconds(pd.NaT))


# ---------------------------------------------------------------------------
# _lap_times_to_seconds
# ---------------------------------------------------------------------------

class TestLapTimesToSeconds:
    def test_timedelta64_dtype(self):
        df = _make_laps("VER", "RB", [90.1, 90.2, 90.3])
        out = _lap_times_to_seconds(df)
        assert out["lap_time_s"].tolist() == pytest.approx([90.1, 90.2, 90.3])

    def test_original_column_preserved(self):
        df = _make_laps("VER", "RB", [90.0])
        out = _lap_times_to_seconds(df)
        assert "LapTime" in out.columns


# ---------------------------------------------------------------------------
# _add_fuel_correction
# ---------------------------------------------------------------------------

class TestFuelCorrection:
    def test_last_lap_has_zero_correction(self):
        """La última vuelta del stint tiene remaining_laps=0 → corrección=0."""
        df = _make_laps("VER", "RB", [90.0, 89.8, 89.6])
        df = _lap_times_to_seconds(df)
        out = _add_fuel_correction(df, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        # Última vuelta (TyreLife=3, max=3): fuel_remaining=0
        assert out["fuel_correction_s"].iloc[-1] == pytest.approx(0.0)

    def test_first_lap_has_max_correction(self):
        """La primera vuelta tiene la mayor corrección (más combustible)."""
        df = _make_laps("VER", "RB", [92.0, 91.8, 91.5])
        df = _lap_times_to_seconds(df)
        out = _add_fuel_correction(df, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        # Stint de 3 vueltas: primer vuelta tiene 2 vueltas de combustible por delante
        # correction = 2 × 1.6 × 0.035 = 0.112
        assert out["fuel_correction_s"].iloc[0] == pytest.approx(2 * 1.6 * 0.035)

    def test_corrected_times_more_uniform(self):
        """Los tiempos corregidos deben tener menos varianza que los crudos."""
        # Simular mejora natural por quema de combustible: 0.056 s/vuelta
        raw = [90.0 + 9 * 0.056, 90.0 + 8 * 0.056,
               90.0 + 7 * 0.056, 90.0 + 6 * 0.056,
               90.0 + 5 * 0.056, 90.0 + 4 * 0.056]
        df = _make_laps("VER", "RB", raw)
        df = _lap_times_to_seconds(df)
        out = _add_fuel_correction(df, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        raw_var = df["lap_time_s"].var()
        cor_var = out["corrected_lap_time_s"].var()
        assert cor_var < raw_var

    def test_fallback_without_tyre_life(self):
        """Sin TyreLife usa rank de LapNumber como posición en stint."""
        df = _make_laps("VER", "RB", [90.0, 89.8, 89.6])
        df = _lap_times_to_seconds(df)
        df = df.drop(columns=["TyreLife"])
        out = _add_fuel_correction(df, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        assert "corrected_lap_time_s" in out.columns
        assert out["fuel_correction_s"].iloc[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _filter_long_stints
# ---------------------------------------------------------------------------

class TestFilterLongStints:
    def test_short_stint_excluded(self):
        df = _make_laps("VER", "RB", [90.0, 90.1, 90.2])  # 3 vueltas
        df = _lap_times_to_seconds(df)
        out = _filter_long_stints(df, min_laps=5)
        assert out.empty

    def test_long_stint_kept(self):
        df = _make_laps("VER", "RB", [90.0] * 7)
        df = _lap_times_to_seconds(df)
        out = _filter_long_stints(df, min_laps=5)
        assert len(out) == 7

    def test_exactly_min_laps_kept(self):
        df = _make_laps("VER", "RB", [90.0] * 5)
        df = _lap_times_to_seconds(df)
        out = _filter_long_stints(df, min_laps=5)
        assert len(out) == 5


# ---------------------------------------------------------------------------
# _remove_stint_outliers
# ---------------------------------------------------------------------------

class TestRemoveStintOutliers:
    def test_removes_very_slow_lap(self):
        """Una vuelta anormalmente lenta (pit stop, neutralización) se elimina."""
        times = [90.0] * 8 + [120.0]  # último es outlier claro
        df = _make_laps("VER", "RB", times)
        df = _lap_times_to_seconds(df)
        df["corrected_lap_time_s"] = df["lap_time_s"]
        out = _remove_stint_outliers(df)
        assert len(out) < len(df)
        assert 120.0 not in out["lap_time_s"].tolist()

    def test_uniform_laps_all_kept(self):
        df = _make_laps("VER", "RB", [90.0 + i * 0.05 for i in range(8)])
        df = _lap_times_to_seconds(df)
        df["corrected_lap_time_s"] = df["lap_time_s"]
        out = _remove_stint_outliers(df)
        assert len(out) == len(df)


# ---------------------------------------------------------------------------
# _extract_long_run_pace
# ---------------------------------------------------------------------------

class TestExtractLongRunPace:
    def test_empty_input(self):
        cols = ["Driver", "Team", "LapNumber", "LapTime", "Stint", "TyreLife",
                "TrackStatus", "IsAccurate", "season", "round", "session_type"]
        df = pd.DataFrame(columns=cols)
        df["LapTime"] = pd.to_timedelta(df["LapTime"])
        out = _extract_long_run_pace(df, min_laps=5, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        assert out.empty

    def test_too_few_laps_returns_empty(self):
        df = _make_laps("VER", "RB", [90.0, 90.1])  # 2 vueltas < min_laps=5
        out = _extract_long_run_pace(df, min_laps=5, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        assert out.empty

    def test_long_run_pace_computed(self):
        df = _make_laps("VER", "RB", [90.0 + i * 0.056 for i in range(8)])
        out = _extract_long_run_pace(df, min_laps=5, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        assert len(out) == 1
        assert out.iloc[0]["abbreviation"] == "VER"
        assert pd.notna(out.iloc[0]["long_run_pace_s"])

    def test_two_drivers(self):
        ver = _make_laps("VER", "RB", [89.0 + i * 0.056 for i in range(8)])
        nor = _make_laps("NOR", "MCL", [90.0 + i * 0.056 for i in range(8)])
        df = _concat(ver, nor)
        out = _extract_long_run_pace(df, min_laps=5, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        assert set(out["abbreviation"]) == {"VER", "NOR"}

    def test_faster_driver_has_lower_pace(self):
        fast = _make_laps("VER", "RB",  [89.0 + i * 0.056 for i in range(8)])
        slow = _make_laps("NOR", "MCL", [91.0 + i * 0.056 for i in range(8)])
        df = _concat(fast, slow)
        out = _extract_long_run_pace(df, min_laps=5, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        ver_pace = out[out["abbreviation"] == "VER"]["long_run_pace_s"].iloc[0]
        nor_pace = out[out["abbreviation"] == "NOR"]["long_run_pace_s"].iloc[0]
        assert ver_pace < nor_pace

    def test_sc_laps_excluded(self):
        """Vueltas bajo safety car (TrackStatus != '1') se excluyen."""
        green = _make_laps("VER", "RB", [90.0] * 5)
        sc = _make_laps("VER", "RB", [95.0] * 3, stint=2, tyre_life_start=6,
                        track_status="4")  # SC
        df = _concat(green, sc)
        out = _extract_long_run_pace(df, min_laps=5, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        # Solo el stint verde tiene suficientes vueltas
        assert len(out) == 1
        # El pace no debe estar influenciado por las vueltas de SC
        assert out.iloc[0]["long_run_pace_s"] < 91.0

    def test_invalid_lap_times_filtered(self):
        """Tiempos < 60s (errores de telemetría) se descartan."""
        times = [90.0] * 6 + [5.0]  # 5s = claramente inválido
        df = _make_laps("VER", "RB", times)
        out = _extract_long_run_pace(df, min_laps=5, fuel_kg_per_lap=1.6, fuel_s_per_kg=0.035)
        # 5s se descarta, quedan 6 vueltas válidas → sí supera min_laps=5
        assert len(out) == 1


# ---------------------------------------------------------------------------
# _extract_session_best
# ---------------------------------------------------------------------------

class TestExtractSessionBest:
    def test_returns_minimum_lap_time(self):
        df = _make_laps("VER", "RB", [90.5, 89.8, 91.0])
        out = _extract_session_best(df, out_col="fp_best_lap_s")
        assert out.iloc[0]["fp_best_lap_s"] == pytest.approx(89.8)

    def test_two_drivers(self):
        ver = _make_laps("VER", "RB",  [89.8, 90.0])
        nor = _make_laps("NOR", "MCL", [90.5, 90.2])
        df = _concat(ver, nor)
        out = _extract_session_best(df, out_col="fp_best_lap_s")
        assert set(out["abbreviation"]) == {"VER", "NOR"}
        ver_best = out[out["abbreviation"] == "VER"]["fp_best_lap_s"].iloc[0]
        assert ver_best == pytest.approx(89.8)

    def test_empty_returns_empty(self):
        cols = ["Driver", "Team", "LapNumber", "LapTime", "Stint"]
        df = pd.DataFrame(columns=cols)
        df["LapTime"] = pd.to_timedelta(df["LapTime"])
        out = _extract_session_best(df, out_col="fp_best_lap_s")
        assert out.empty


# ---------------------------------------------------------------------------
# compute_pace_features — función pública
# ---------------------------------------------------------------------------

class TestComputePaceFeatures:
    def _practice(self, drivers_times: dict[tuple[str, str], list[float]]) -> pd.DataFrame:
        """drivers_times: {(abbreviation, team): [lap_times_s]}"""
        parts = []
        for (drv, team), times in drivers_times.items():
            parts.append(_make_laps(drv, team, times))
        return _concat(*parts)

    def test_output_columns_present(self):
        df = self._practice({("VER", "RB"): [90.0] * 8})
        out = compute_pace_features(df)
        expected = {"season", "round", "session_type", "abbreviation", "team_name",
                    "long_run_pace_s", "long_run_n", "fp_best_lap_s", "quali_pace_s"}
        assert expected <= set(out.columns)

    def test_quali_pace_nan_without_q_laps(self):
        df = self._practice({("VER", "RB"): [90.0] * 8})
        out = compute_pace_features(df, quali_laps=None)
        assert out["quali_pace_s"].isna().all()

    def test_quali_pace_from_q_laps(self):
        practice = self._practice({("VER", "RB"): [90.0] * 8})
        q_laps = _make_laps("VER", "RB", [88.5, 88.2], session="Q")
        out = compute_pace_features(practice, quali_laps=q_laps)
        assert out["quali_pace_s"].notna().any()
        assert out["quali_pace_s"].iloc[0] == pytest.approx(88.2)

    def test_season_round_propagated(self):
        df = self._practice({("VER", "RB"): [90.0] * 8})
        out = compute_pace_features(df)
        assert out["season"].iloc[0] == 2026
        assert out["round"].iloc[0] == 3

    def test_missing_required_col_raises(self):
        df = self._practice({("VER", "RB"): [90.0] * 8})
        df = df.drop(columns=["LapTime"])
        with pytest.raises(ValueError, match="LapTime"):
            compute_pace_features(df)

    def test_fp_best_lap_present(self):
        df = self._practice({("VER", "RB"): [90.5, 89.8, 91.0, 90.2]})
        out = compute_pace_features(df)
        assert out["fp_best_lap_s"].iloc[0] == pytest.approx(89.8)

    def test_long_run_nan_when_no_long_stints(self):
        """Sin stints suficientemente largos, long_run_pace_s = NaN."""
        df = self._practice({("VER", "RB"): [90.0, 90.1]})  # 2 < min_laps=5
        out = compute_pace_features(df)
        assert pd.isna(out["long_run_pace_s"].iloc[0])

    def test_two_drivers_one_row_each(self):
        df = self._practice({
            ("VER", "RB"):  [89.0] * 8,
            ("PER", "RB"):  [90.0] * 8,
        })
        out = compute_pace_features(df)
        assert len(out) == 2
        assert set(out["abbreviation"]) == {"VER", "PER"}


# ---------------------------------------------------------------------------
# compute_teammate_gaps — función pública
# ---------------------------------------------------------------------------

def _pace_df(
    entries: list[tuple[str, str, float | None, float | None]],
    round_: int = 3,
) -> pd.DataFrame:
    """entries: [(abbreviation, team, long_run_pace_s, quali_pace_s)]"""
    return pd.DataFrame([
        {
            "abbreviation":    abbr,
            "team_name":       team,
            "round":           round_,
            "long_run_pace_s": lrp,
            "quali_pace_s":    qp,
        }
        for abbr, team, lrp, qp in entries
    ])


class TestComputeTeammateGaps:
    def test_output_columns_present(self):
        df = _pace_df([("VER", "RB", 89.0, 88.0), ("PER", "RB", 90.0, 89.0)])
        out = compute_teammate_gaps(df)
        assert {"abbreviation", "round", "gap_to_teammate_race",
                "gap_to_teammate_quali"} <= set(out.columns)

    def test_gaps_are_symmetric(self):
        """gap_A + gap_B = 0 cuando ambos tienen datos válidos."""
        df = _pace_df([("VER", "RB", 89.0, 88.0), ("PER", "RB", 90.0, 89.0)])
        out = compute_teammate_gaps(df)
        ver = out[out["abbreviation"] == "VER"].iloc[0]
        per = out[out["abbreviation"] == "PER"].iloc[0]
        assert ver["gap_to_teammate_race"] + per["gap_to_teammate_race"] == pytest.approx(0.0)
        assert ver["gap_to_teammate_quali"] + per["gap_to_teammate_quali"] == pytest.approx(0.0)

    def test_faster_driver_has_negative_gap(self):
        """Driver más rápido → gap negativo (su tiempo < compañero)."""
        df = _pace_df([("VER", "RB", 89.0, 88.0), ("PER", "RB", 90.0, 89.0)])
        out = compute_teammate_gaps(df)
        ver_gap = out[out["abbreviation"] == "VER"]["gap_to_teammate_race"].iloc[0]
        assert ver_gap < 0

    def test_single_driver_gets_nan_gaps(self):
        """Solo un piloto en el equipo → gaps = NaN."""
        df = _pace_df([("VER", "RB", 89.0, 88.0)])
        out = compute_teammate_gaps(df)
        assert out["gap_to_teammate_race"].isna().all()
        assert out["gap_to_teammate_quali"].isna().all()

    def test_missing_pace_gives_nan_gap(self):
        """Si un piloto no tiene long_run_pace_s, su gap_race = NaN."""
        df = _pace_df([("VER", "RB", None, 88.0), ("PER", "RB", 90.0, 89.0)])
        out = compute_teammate_gaps(df)
        ver_race = out[out["abbreviation"] == "VER"]["gap_to_teammate_race"].iloc[0]
        assert pd.isna(ver_race)

    def test_quali_gap_nan_when_both_missing(self):
        df = _pace_df([("VER", "RB", 89.0, None), ("PER", "RB", 90.0, None)])
        out = compute_teammate_gaps(df)
        assert out["gap_to_teammate_quali"].isna().all()

    def test_quali_gap_nan_when_one_missing(self):
        df = _pace_df([("VER", "RB", 89.0, 88.0), ("PER", "RB", 90.0, None)])
        out = compute_teammate_gaps(df)
        ver_q = out[out["abbreviation"] == "VER"]["gap_to_teammate_quali"].iloc[0]
        per_q = out[out["abbreviation"] == "PER"]["gap_to_teammate_quali"].iloc[0]
        assert pd.isna(ver_q)
        assert pd.isna(per_q)

    def test_two_teams_independent(self):
        """Cada equipo calcula sus propios gaps."""
        df = _pace_df([
            ("VER", "RB",  89.0, 88.0),
            ("PER", "RB",  90.0, 89.0),
            ("NOR", "MCL", 90.2, 89.2),
            ("PIA", "MCL", 90.5, 89.5),
        ])
        out = compute_teammate_gaps(df)
        assert len(out) == 4
        # Verificar que los gaps de RB y MCL son independientes
        rb_ver = out[out["abbreviation"] == "VER"]["gap_to_teammate_race"].iloc[0]
        mcl_nor = out[out["abbreviation"] == "NOR"]["gap_to_teammate_race"].iloc[0]
        assert rb_ver == pytest.approx(89.0 - 90.0)
        assert mcl_nor == pytest.approx(90.2 - 90.5)

    def test_multiple_rounds(self):
        """Cada ronda tiene sus propios gaps, sin mezclar rondas distintas."""
        df = pd.concat([
            _pace_df([("VER", "RB", 89.0, 88.0), ("PER", "RB", 90.0, 89.0)], round_=3),
            _pace_df([("VER", "RB", 88.5, 87.8), ("PER", "RB", 89.8, 89.0)], round_=4),
        ], ignore_index=True)
        out = compute_teammate_gaps(df)
        assert set(out["round"]) == {3, 4}
        assert len(out) == 4

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=["abbreviation", "team_name", "round",
                                   "long_run_pace_s", "quali_pace_s"])
        out = compute_teammate_gaps(df)
        assert out.empty

    def test_missing_required_col_raises(self):
        df = _pace_df([("VER", "RB", 89.0, 88.0)])
        df = df.drop(columns=["team_name"])
        with pytest.raises(ValueError, match="team_name"):
            compute_teammate_gaps(df)

    def test_gap_magnitude_correct(self):
        """gap = driver_pace - teammate_pace, en segundos."""
        df = _pace_df([("VER", "RB", 89.0, None), ("PER", "RB", 90.5, None)])
        out = compute_teammate_gaps(df)
        ver = out[out["abbreviation"] == "VER"]["gap_to_teammate_race"].iloc[0]
        assert ver == pytest.approx(89.0 - 90.5)
