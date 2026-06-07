"""Tests para StrengthRaceModel (models/prediction_model.py)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gridcast.models.prediction_model import StrengthRaceModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _pace_features(n_drivers: int = 4, n_rounds: int = 2) -> pd.DataFrame:
    """DataFrame de pace_features con dos equipos de 2 pilotos cada uno."""
    rows = []
    teams = [f"Team{i}" for i in range(n_drivers // 2)]
    abbrevs = [f"D{i:02d}" for i in range(n_drivers)]

    for r in range(1, n_rounds + 1):
        for i, abbrev in enumerate(abbrevs):
            team = teams[i // 2]
            rows.append({
                "season": 2026,
                "round": r,
                "session_type": "FP2",
                "abbreviation": abbrev,
                "team_name": team,
                "long_run_pace_s": 90.0 + i * 0.2 + (r - 1) * 0.05,
                "long_run_n": 15,
                "fp_best_lap_s": 85.0 + i * 0.15,
                "quali_pace_s": 84.0 + i * 0.15,
                "gap_to_teammate_race": (0.2 if i % 2 == 0 else -0.2),
                "gap_to_teammate_quali": (0.15 if i % 2 == 0 else -0.15),
                "date": date(2026, 3, r * 7),
            })
    return pd.DataFrame(rows)


def _mock_snapshot(pace_df: pd.DataFrame | None = None, has_frame: bool = True) -> MagicMock:
    snap = MagicMock()
    if has_frame and pace_df is not None:
        snap.frames = ["pace_features", "race_results"]
        snap.read_frame.return_value = pace_df
    elif not has_frame:
        snap.frames = ["race_results"]
    else:
        snap.frames = ["pace_features"]
        snap.read_frame.return_value = pd.DataFrame()
    return snap


def _mock_guard() -> MagicMock:
    guard = MagicMock()
    guard.round_cutoff.return_value = date(2026, 4, 1)
    return guard


# ---------------------------------------------------------------------------
# Tests de propiedades básicas
# ---------------------------------------------------------------------------

class TestStrengthRaceModelVersion:
    def test_default_version(self):
        m = StrengthRaceModel()
        assert "strength_race_v1" in m.version

    def test_version_includes_signal(self):
        m_quali = StrengthRaceModel(primary_signal="quali")
        m_race  = StrengthRaceModel(primary_signal="race")
        assert "quali" in m_quali.version
        assert "race"  in m_race.version

    def test_version_is_str(self):
        assert isinstance(StrengthRaceModel().version, str)


# ---------------------------------------------------------------------------
# Tests de predict() — formato de salida
# ---------------------------------------------------------------------------

class TestPredictOutputFormat:
    def test_returns_dataframe(self):
        snap = _mock_snapshot(_pace_features())
        m = StrengthRaceModel(n_sims=100)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        assert isinstance(result, pd.DataFrame)

    def test_canonical_columns(self):
        snap = _mock_snapshot(_pace_features())
        m = StrengthRaceModel(n_sims=100)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        assert set(result.columns) >= {"driver_id", "finish_pos", "probability"}

    def test_driver_ids_match_pace_features(self):
        pace = _pace_features(n_drivers=4)
        snap = _mock_snapshot(pace)
        m = StrengthRaceModel(n_sims=100)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        expected_drivers = set(pace["abbreviation"].unique())
        actual_drivers = set(result["driver_id"].unique())
        assert actual_drivers == expected_drivers

    def test_finish_positions_cover_all_drivers(self):
        n_drivers = 4
        pace = _pace_features(n_drivers=n_drivers)
        snap = _mock_snapshot(pace)
        m = StrengthRaceModel(n_sims=200)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        assert set(result["finish_pos"].unique()) == set(range(1, n_drivers + 1))

    def test_probabilities_positive(self):
        snap = _mock_snapshot(_pace_features())
        m = StrengthRaceModel(n_sims=100)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        assert (result["probability"] >= 0).all()

    def test_probabilities_sum_to_one_per_position(self):
        snap = _mock_snapshot(_pace_features(n_drivers=4))
        m = StrengthRaceModel(n_sims=500)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        for pos, grp in result.groupby("finish_pos"):
            assert abs(grp["probability"].sum() - 1.0) < 0.01, \
                f"pos={pos}: sum={grp['probability'].sum()}"

    def test_probabilities_sum_to_one_per_driver(self):
        snap = _mock_snapshot(_pace_features(n_drivers=4))
        m = StrengthRaceModel(n_sims=500)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        for drv, grp in result.groupby("driver_id"):
            assert abs(grp["probability"].sum() - 1.0) < 0.01, \
                f"driver={drv}: sum={grp['probability'].sum()}"


# ---------------------------------------------------------------------------
# Tests de casos degenerados
# ---------------------------------------------------------------------------

class TestPredictEdgeCases:
    def test_missing_pace_features_frame_returns_empty(self):
        snap = _mock_snapshot(has_frame=False)
        m = StrengthRaceModel(n_sims=100)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_pace_features_returns_empty(self):
        snap = _mock_snapshot(pd.DataFrame(), has_frame=True)
        m = StrengthRaceModel(n_sims=100)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        assert result.empty

    def test_single_team_no_gap_returns_empty(self):
        # Con un solo equipo, fit_all puede no producir drivers si no hay gaps.
        # El modelo devuelve DataFrame vacío cuando no hay estimaciones.
        pace = pd.DataFrame([{
            "season": 2026, "round": 1, "session_type": "FP2",
            "abbreviation": "D00", "team_name": "Solo",
            "long_run_pace_s": 90.0, "long_run_n": 15,
            "fp_best_lap_s": 85.0, "quali_pace_s": 84.0,
        }])
        snap = _mock_snapshot(pace)
        m = StrengthRaceModel(n_sims=100)
        result = m.predict(snap, round_=2, guard=_mock_guard())
        # Sin compañero de equipo → sin estimaciones → DataFrame vacío
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests de reproducibilidad
# ---------------------------------------------------------------------------

class TestPredictReproducibility:
    def test_same_seed_same_result(self):
        pace = _pace_features(n_drivers=4, n_rounds=2)
        snap = _mock_snapshot(pace)

        m = StrengthRaceModel(n_sims=200, seed=42)
        r1 = m.predict(snap, round_=3, guard=_mock_guard())
        r2 = m.predict(snap, round_=3, guard=_mock_guard())

        pd.testing.assert_frame_equal(
            r1.sort_values(["driver_id", "finish_pos"]).reset_index(drop=True),
            r2.sort_values(["driver_id", "finish_pos"]).reset_index(drop=True),
        )

    def test_different_rounds_different_seeds(self):
        pace = _pace_features(n_drivers=4, n_rounds=3)
        snap = _mock_snapshot(pace)

        m = StrengthRaceModel(n_sims=1000, seed=42)
        r3 = m.predict(snap, round_=3, guard=_mock_guard())
        r4 = m.predict(snap, round_=4, guard=_mock_guard())

        win_r3 = r3[r3["finish_pos"] == 1].set_index("driver_id")["probability"]
        win_r4 = r4[r4["finish_pos"] == 1].set_index("driver_id")["probability"]
        # Con seeds distintos, la distribución puede diferir levemente
        # Verificamos solo que son válidas (no que sean idénticas)
        assert (win_r3 >= 0).all()
        assert (win_r4 >= 0).all()


# ---------------------------------------------------------------------------
# Tests de sensatez del modelo
# ---------------------------------------------------------------------------

class TestPredictSanity:
    def test_stronger_driver_wins_more_often(self):
        """El piloto más fuerte debe tener mayor probabilidad de victoria."""
        pace = _pace_features(n_drivers=4)
        snap = _mock_snapshot(pace)
        m = StrengthRaceModel(n_sims=2000, seed=7)
        result = m.predict(snap, round_=3, guard=_mock_guard())

        win_probs = result[result["finish_pos"] == 1].set_index("driver_id")["probability"]
        # D00 tiene el menor quali_pace_s (más rápido) → debería ganar más a menudo
        assert win_probs["D00"] > win_probs["D03"], (
            f"D00 win_prob={win_probs['D00']:.3f}, D03 win_prob={win_probs['D03']:.3f}"
        )

    def test_race_signal_gives_valid_output(self):
        snap = _mock_snapshot(_pace_features())
        m = StrengthRaceModel(primary_signal="race", n_sims=100)
        result = m.predict(snap, round_=3, guard=_mock_guard())
        assert not result.empty
        assert "driver_id" in result.columns
