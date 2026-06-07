"""
Tests de features/reliability.py

Sin llamadas a red. Los RaceResult se construyen directamente.
"""

from __future__ import annotations

import pytest

from gridcast.features.reliability import (
    _PRIOR_INCIDENT,
    _PRIOR_MECHANICAL,
    _PRIOR_TOTAL,
    _bayesian_rate,
    _results_to_df,
    classify_status,
    compute_driver_reliability,
    compute_engine_reliability,
    compute_reliability_features,
    compute_team_reliability,
)
from gridcast.ingest.jolpica_client import RaceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(
    driver_id: str = "max_verstappen",
    constructor_id: str = "red_bull",
    status: str = "Finished",
    season: int = 2026,
    round_: int = 1,
    position: int | None = 1,
) -> RaceResult:
    return RaceResult(
        season=season,
        round=round_,
        driver_id=driver_id,
        constructor_id=constructor_id,
        grid=1,
        position=position,
        position_text=str(position) if position else "R",
        status=status,
        points=25.0,
        laps_completed=58,
    )


def _many(driver_id: str, constructor_id: str, statuses: list[str]) -> list[RaceResult]:
    return [
        _result(driver_id=driver_id, constructor_id=constructor_id,
                status=s, round_=i + 1)
        for i, s in enumerate(statuses)
    ]


# ---------------------------------------------------------------------------
# classify_status
# ---------------------------------------------------------------------------

class TestClassifyStatus:
    def test_finished(self):
        assert classify_status("Finished") == "finished"

    def test_lapped_is_finished(self):
        assert classify_status("+1 Lap") == "finished"
        assert classify_status("+3 Laps") == "finished"

    def test_engine(self):
        assert classify_status("Engine") == "mechanical"

    def test_gearbox(self):
        assert classify_status("Gearbox") == "mechanical"

    def test_power_unit(self):
        assert classify_status("Power Unit") == "mechanical"

    def test_brakes(self):
        assert classify_status("Brakes") == "mechanical"

    def test_hydraulics(self):
        assert classify_status("Hydraulics") == "mechanical"

    def test_suspension(self):
        assert classify_status("Suspension") == "mechanical"

    def test_collision(self):
        assert classify_status("Collision") == "incident"

    def test_accident(self):
        assert classify_status("Accident") == "incident"

    def test_spun_off(self):
        assert classify_status("Spun off") == "incident"

    def test_collision_damage(self):
        assert classify_status("Collision damage") == "incident"

    def test_retired_is_other(self):
        assert classify_status("Retired") == "other"

    def test_disqualified_is_other(self):
        assert classify_status("Disqualified") == "other"

    def test_case_insensitive(self):
        assert classify_status("ENGINE") == "mechanical"
        assert classify_status("COLLISION") == "incident"
        assert classify_status("FINISHED") == "finished"

    def test_turbo(self):
        assert classify_status("Turbocharger") == "mechanical"


# ---------------------------------------------------------------------------
# _bayesian_rate
# ---------------------------------------------------------------------------

class TestBayesianRate:
    def test_prior_mean_with_zero_data(self):
        # Con 0 observaciones, rate = prior mean = alpha / (alpha + beta)
        alpha, beta = _PRIOR_MECHANICAL
        rate = _bayesian_rate(0, 0, _PRIOR_MECHANICAL)
        assert rate == pytest.approx(alpha / (alpha + beta))

    def test_converges_with_many_observations(self):
        # Con muchas observaciones, el prior pesa poco
        rate_high = _bayesian_rate(80, 100, _PRIOR_MECHANICAL)  # 80% observado
        assert rate_high > 0.70

        rate_low = _bayesian_rate(2, 100, _PRIOR_MECHANICAL)    # 2% observado
        assert rate_low < 0.05

    def test_never_zero_or_one(self):
        alpha, beta = _PRIOR_TOTAL
        assert _bayesian_rate(0, 50, _PRIOR_TOTAL) > 0.0
        assert _bayesian_rate(50, 50, _PRIOR_TOTAL) < 1.0

    def test_monotone_in_successes(self):
        r1 = _bayesian_rate(2, 20, _PRIOR_TOTAL)
        r2 = _bayesian_rate(5, 20, _PRIOR_TOTAL)
        r3 = _bayesian_rate(10, 20, _PRIOR_TOTAL)
        assert r1 < r2 < r3


# ---------------------------------------------------------------------------
# _results_to_df
# ---------------------------------------------------------------------------

class TestResultsToDf:
    def test_empty_list(self):
        df = _results_to_df([])
        assert df.empty

    def test_columns_present(self):
        df = _results_to_df([_result()])
        assert {"driver_id", "constructor_id", "status", "dnf_category"} <= set(df.columns)

    def test_dnf_category_assigned(self):
        df = _results_to_df([_result(status="Engine")])
        assert df.iloc[0]["dnf_category"] == "mechanical"


# ---------------------------------------------------------------------------
# compute_driver_reliability
# ---------------------------------------------------------------------------

class TestComputeDriverReliability:
    def test_empty_results(self):
        out = compute_driver_reliability([])
        assert out.empty

    def test_output_columns(self):
        out = compute_driver_reliability([_result()])
        expected = {
            "driver_id", "starts",
            "mechanical_dnfs", "incident_dnfs", "other_dnfs", "total_dnfs",
            "mechanical_dnf_rate", "incident_dnf_rate", "total_dnf_rate",
        }
        assert expected <= set(out.columns)

    def test_one_driver_all_finished(self):
        results = _many("verstappen", "red_bull", ["Finished"] * 10)
        out = compute_driver_reliability(results)
        row = out[out["driver_id"] == "verstappen"].iloc[0]
        assert row["starts"] == 10
        assert row["mechanical_dnfs"] == 0
        assert row["total_dnfs"] == 0

    def test_rate_reflects_dnfs(self):
        """Driver con 5 abandonos mecánicos en 10 carreras debe tener rate > prior."""
        alpha, beta = _PRIOR_MECHANICAL
        prior_mean = alpha / (alpha + beta)
        statuses = ["Engine"] * 5 + ["Finished"] * 5
        results = _many("ver", "rb", statuses)
        out = compute_driver_reliability(results)
        rate = out[out["driver_id"] == "ver"]["mechanical_dnf_rate"].iloc[0]
        assert rate > prior_mean

    def test_two_drivers_independent(self):
        r1 = _many("ver", "rb",  ["Finished"] * 10)
        r2 = _many("nor", "mcl", ["Engine"]   * 3 + ["Finished"] * 7)
        out = compute_driver_reliability(r1 + r2)
        ver_rate = out[out["driver_id"] == "ver"]["mechanical_dnf_rate"].iloc[0]
        nor_rate = out[out["driver_id"] == "nor"]["mechanical_dnf_rate"].iloc[0]
        assert nor_rate > ver_rate

    def test_incident_dnf_classified_correctly(self):
        results = _many("ver", "rb", ["Collision", "Finished", "Accident"])
        out = compute_driver_reliability(results)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert row["incident_dnfs"] == 2
        assert row["mechanical_dnfs"] == 0

    def test_other_dnf_counts_in_total(self):
        results = _many("ver", "rb", ["Retired", "Finished", "Finished"])
        out = compute_driver_reliability(results)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert row["other_dnfs"] == 1
        assert row["total_dnfs"] == 1
        assert row["mechanical_dnfs"] == 0

    def test_rates_between_0_and_1(self):
        statuses = ["Engine", "Collision", "Retired", "Finished"] * 3
        results = _many("ver", "rb", statuses)
        out = compute_driver_reliability(results)
        row = out.iloc[0]
        for col in ["mechanical_dnf_rate", "incident_dnf_rate", "total_dnf_rate"]:
            assert 0.0 < row[col] < 1.0

    def test_total_rate_geq_mechanical_and_incident(self):
        statuses = ["Engine", "Collision", "Finished"] * 3
        results = _many("ver", "rb", statuses)
        out = compute_driver_reliability(results)
        row = out.iloc[0]
        assert row["total_dnf_rate"] >= row["mechanical_dnf_rate"]
        assert row["total_dnf_rate"] >= row["incident_dnf_rate"]


# ---------------------------------------------------------------------------
# compute_team_reliability
# ---------------------------------------------------------------------------

class TestComputeTeamReliability:
    def test_empty_results(self):
        out = compute_team_reliability([])
        assert out.empty

    def test_output_columns(self):
        out = compute_team_reliability([_result()])
        assert {"constructor_id", "starts", "mechanical_dnfs", "total_dnfs",
                "mechanical_dnf_rate", "total_dnf_rate"} <= set(out.columns)

    def test_both_drivers_count(self):
        """Ambos pilotos del equipo cuentan como arranques independientes."""
        r1 = _many("ver", "red_bull", ["Finished"] * 5)
        r2 = _many("per", "red_bull", ["Engine", "Finished", "Finished", "Finished", "Finished"])
        out = compute_team_reliability(r1 + r2)
        row = out[out["constructor_id"] == "red_bull"].iloc[0]
        assert row["starts"] == 10
        assert row["mechanical_dnfs"] == 1

    def test_two_teams_independent(self):
        rb  = _many("ver", "red_bull", ["Engine"] * 3 + ["Finished"] * 7)
        mcl = _many("nor", "mclaren",  ["Finished"] * 10)
        out = compute_team_reliability(rb + mcl)
        rb_rate  = out[out["constructor_id"] == "red_bull"]["mechanical_dnf_rate"].iloc[0]
        mcl_rate = out[out["constructor_id"] == "mclaren"]["mechanical_dnf_rate"].iloc[0]
        assert rb_rate > mcl_rate

    def test_rate_bayesian_smoothed(self):
        alpha, beta = _PRIOR_MECHANICAL
        prior_mean = alpha / (alpha + beta)
        # Equipo con 0 fallos mecánicos → rate debe ser el prior (solo prior influye)
        results = _many("ver", "rb", ["Finished"] * 5)
        out = compute_team_reliability(results)
        rate = out.iloc[0]["mechanical_dnf_rate"]
        # Rate < prior_mean porque tenemos 5 "éxitos" que tiran hacia abajo
        assert rate < prior_mean


# ---------------------------------------------------------------------------
# compute_engine_reliability
# ---------------------------------------------------------------------------

class TestComputeEngineReliability:
    ENGINE_MAP = {
        "red_bull": "honda_rbpt",
        "racing_bulls": "honda_rbpt",
        "mclaren": "mercedes",
        "mercedes": "mercedes",
    }

    def test_empty_results(self):
        out = compute_engine_reliability([], self.ENGINE_MAP)
        assert out.empty

    def test_empty_engine_map(self):
        results = _many("ver", "red_bull", ["Finished"] * 5)
        out = compute_engine_reliability(results, {})
        assert out.empty

    def test_output_columns(self):
        results = _many("ver", "red_bull", ["Finished"] * 3)
        out = compute_engine_reliability(results, self.ENGINE_MAP)
        assert {"engine_supplier", "starts", "mechanical_dnfs", "total_dnfs",
                "mechanical_dnf_rate", "total_dnf_rate"} <= set(out.columns)

    def test_client_teams_pooled(self):
        """RB y Racing Bulls comparten Honda RBPT → sus resultados se agrupan."""
        rb   = _many("ver", "red_bull",    ["Engine", "Finished"] * 5)
        atb  = _many("tsu", "racing_bulls", ["Finished"] * 10)
        out  = compute_engine_reliability(rb + atb, self.ENGINE_MAP)
        honda = out[out["engine_supplier"] == "honda_rbpt"].iloc[0]
        assert honda["starts"] == 20  # 10 de RB + 10 de ATB

    def test_unknown_constructor_excluded(self):
        """Equipos no en engine_map no contribuyen al agregado."""
        results = _many("ver", "unknown_team", ["Engine"] * 5)
        out = compute_engine_reliability(results, self.ENGINE_MAP)
        # unknown_team no está en el map → DataFrame vacío o sin esa entrada
        assert out.empty or "unknown_team" not in out["engine_supplier"].values

    def test_two_engines_independent(self):
        honda_r = _many("ver", "red_bull", ["Engine"] * 3 + ["Finished"] * 7)
        merc_r  = _many("ham", "mercedes", ["Finished"] * 10)
        out = compute_engine_reliability(honda_r + merc_r, self.ENGINE_MAP)
        honda_rate = out[out["engine_supplier"] == "honda_rbpt"]["mechanical_dnf_rate"].iloc[0]
        merc_rate  = out[out["engine_supplier"] == "mercedes"]["mechanical_dnf_rate"].iloc[0]
        assert honda_rate > merc_rate


# ---------------------------------------------------------------------------
# compute_reliability_features (wrapper)
# ---------------------------------------------------------------------------

class TestComputeReliabilityFeatures:
    def test_returns_driver_and_team(self):
        results = _many("ver", "red_bull", ["Finished"] * 5)
        out = compute_reliability_features(results)
        assert "driver" in out
        assert "team"   in out

    def test_engine_none_without_map(self):
        results = _many("ver", "red_bull", ["Finished"] * 5)
        out = compute_reliability_features(results, engine_map=None)
        assert out["engine"] is None

    def test_engine_present_with_map(self):
        results = _many("ver", "red_bull", ["Finished"] * 5)
        engine_map = {"red_bull": "honda_rbpt"}
        out = compute_reliability_features(results, engine_map=engine_map)
        assert out["engine"] is not None
        assert not out["engine"].empty

    def test_empty_results(self):
        out = compute_reliability_features([])
        assert out["driver"].empty
        assert out["team"].empty
