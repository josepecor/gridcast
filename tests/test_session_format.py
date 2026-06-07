"""
Tests de la lógica de formato de fin de semana y reglas del driver_session_map.
No llaman a FastF1 ni a Jolpica — testean solo las invariantes de dominio.
"""

import pandas as pd
import pytest

from gridcast.ingest.fastf1_client import (
    PRACTICE_SESSIONS,
    PACE_SESSION,
    rookie_check_applies,
)
from gridcast.data.driver_session_map import (
    _compute_es_titular,
    build_session_map,
)


# ---------------------------------------------------------------------------
# Formato de fin de semana
# ---------------------------------------------------------------------------

class TestPracticeSessions:
    def test_normal_has_three_practice_sessions(self):
        assert PRACTICE_SESSIONS["conventional"] == ["FP1", "FP2", "FP3"]

    def test_sprint_has_only_fp1(self):
        assert PRACTICE_SESSIONS["sprint_qualifying"] == ["FP1"]

    def test_fp2_not_in_sprint(self):
        assert "FP2" not in PRACTICE_SESSIONS["sprint_qualifying"]

    def test_fp3_not_in_sprint(self):
        assert "FP3" not in PRACTICE_SESSIONS["sprint_qualifying"]


class TestPaceSession:
    def test_normal_pace_session_is_fp2(self):
        assert PACE_SESSION["conventional"] == "FP2"

    def test_sprint_pace_session_is_fp1(self):
        assert PACE_SESSION["sprint_qualifying"] == "FP1"


# ---------------------------------------------------------------------------
# Regla de rookies: solo FP1 convencional
# ---------------------------------------------------------------------------

class TestRookieCheckApplies:
    @pytest.mark.parametrize("session", ["FP2", "FP3", "Q", "R", "SQ", "S"])
    def test_no_rookie_check_outside_fp1(self, session):
        assert not rookie_check_applies("conventional", session)

    def test_no_rookie_check_sprint_fp1(self):
        # Sprint FP1: único libre → titulares siempre
        assert not rookie_check_applies("sprint_qualifying", "FP1")

    def test_rookie_check_only_conventional_fp1(self):
        assert rookie_check_applies("conventional", "FP1")


# ---------------------------------------------------------------------------
# Lógica de es_titular
# ---------------------------------------------------------------------------

class TestComputeEsTitular:
    def _make_df(self, entries: list[tuple[str, str]]) -> pd.DataFrame:
        """Construye un DataFrame mínimo de pilotos para testear _compute_es_titular."""
        rows = [{"car_number": cn, "abbreviation": ab} for cn, ab in entries]
        return pd.DataFrame(rows)

    def test_same_driver_is_titular(self):
        df = self._make_df([("63", "RUS"), ("12", "ANT")])
        ref = {"63": "RUS", "12": "ANT"}
        result = _compute_es_titular(df, ref)
        assert result.tolist() == [True, True]

    def test_different_driver_not_titular(self):
        # Coche 63 conducido por un rookie en FP1, Russell conduce en la carrera
        df = self._make_df([("63", "ROO"), ("12", "ANT")])
        ref = {"63": "RUS", "12": "ANT"}
        result = _compute_es_titular(df, ref)
        assert result.tolist() == [False, True]

    def test_unknown_car_number_assumed_titular(self):
        # Coche no visto antes (equipo nuevo o asignación especial)
        df = self._make_df([("99", "NEW")])
        ref = {"63": "RUS"}
        result = _compute_es_titular(df, ref)
        assert result.tolist() == [True]

    def test_all_rookies_returns_all_false(self):
        df = self._make_df([("63", "ROO1"), ("12", "ROO2")])
        ref = {"63": "RUS", "12": "ANT"}
        result = _compute_es_titular(df, ref)
        assert result.tolist() == [False, False]

    def test_rookie_with_new_car_number_is_known_false_negative(self):
        # Crawford #34 sustituye a Alonso #14 con número de coche nuevo.
        # FALSA NEGATIVA CONOCIDA: no podemos distinguir esto de equipo nuevo.
        # El módulo emite warning pero marca True (conservador).
        # Estos casos requieren anotación manual posterior.
        df = self._make_df([("18", "STR"), ("34", "CRA")])
        ref = {"14": "ALO", "18": "STR"}
        result = _compute_es_titular(df, ref)
        titulares = dict(zip(["18", "34"], result.tolist()))
        assert titulares["18"] is True   # Stroll en su propio coche
        assert titulares["34"] is True   # falsa negativa: Crawford marcado como titular


# ---------------------------------------------------------------------------
# build_session_map: invariantes sin API (mocking de las funciones de ingesta)
# ---------------------------------------------------------------------------

class TestBuildSessionMapDomainRules:
    """
    Verifica las reglas de dominio de build_session_map mockeando FastF1.
    No toca la red.
    """

    def _fake_drivers(self, *_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "DriverNumber": [63, 12],
            "Abbreviation": ["RUS", "ANT"],
            "FullName":     ["George Russell", "Andrea Kimi Antonelli"],
            "TeamName":     ["Mercedes", "Mercedes"],
        })

    def test_fp2_conventional_all_titular(self, monkeypatch):
        monkeypatch.setattr(
            "gridcast.data.driver_session_map.get_drivers_in_session",
            self._fake_drivers,
        )
        df = build_session_map(2026, 1, "FP2", "conventional")
        assert df["es_titular"].all()

    def test_fp3_conventional_all_titular(self, monkeypatch):
        monkeypatch.setattr(
            "gridcast.data.driver_session_map.get_drivers_in_session",
            self._fake_drivers,
        )
        df = build_session_map(2026, 1, "FP3", "conventional")
        assert df["es_titular"].all()

    def test_race_always_titular(self, monkeypatch):
        monkeypatch.setattr(
            "gridcast.data.driver_session_map.get_drivers_in_session",
            self._fake_drivers,
        )
        df = build_session_map(2026, 1, "R", "conventional")
        assert df["es_titular"].all()

    def test_sprint_fp1_all_titular(self, monkeypatch):
        monkeypatch.setattr(
            "gridcast.data.driver_session_map.get_drivers_in_session",
            self._fake_drivers,
        )
        df = build_session_map(2026, 2, "FP1", "sprint_qualifying")
        assert df["es_titular"].all()

    def test_conventional_fp1_detects_rookie(self, monkeypatch):
        def fake_drivers(*_a, **_kw):
            # El coche 63 lleva un rookie en FP1
            return pd.DataFrame({
                "DriverNumber": [63, 12],
                "Abbreviation": ["ROO", "ANT"],
                "FullName":     ["Rookie Piloto", "Andrea Kimi Antonelli"],
                "TeamName":     ["Mercedes", "Mercedes"],
            })

        def fake_race_results(*_a, **_kw):
            # Carrera previa: coche 63 = Russell
            return pd.DataFrame({
                "DriverNumber": [63, 12],
                "Abbreviation": ["RUS", "ANT"],
            })

        monkeypatch.setattr(
            "gridcast.data.driver_session_map.get_drivers_in_session", fake_drivers
        )
        monkeypatch.setattr(
            "gridcast.data.driver_session_map.get_results", fake_race_results
        )

        df = build_session_map(2026, 2, "FP1", "conventional")
        titulares = dict(zip(df["car_number"], df["es_titular"]))
        assert titulares["63"] is False   # rookie en FP1
        assert titulares["12"] is True    # Antonelli en su propio coche

    def test_output_columns_present(self, monkeypatch):
        monkeypatch.setattr(
            "gridcast.data.driver_session_map.get_drivers_in_session",
            self._fake_drivers,
        )
        df = build_session_map(2026, 1, "FP2", "conventional")
        expected_cols = {
            "season", "round", "session_type", "event_format",
            "car_number", "abbreviation", "full_name", "team_name", "es_titular",
        }
        assert expected_cols <= set(df.columns)

    def test_metadata_values_propagated(self, monkeypatch):
        monkeypatch.setattr(
            "gridcast.data.driver_session_map.get_drivers_in_session",
            self._fake_drivers,
        )
        df = build_session_map(2026, 7, "FP2", "conventional")
        assert (df["season"] == 2026).all()
        assert (df["round"] == 7).all()
        assert (df["session_type"] == "FP2").all()
        assert (df["event_format"] == "conventional").all()
