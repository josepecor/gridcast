"""
Tests de features/penalties.py

Sin llamadas a red. Los DataFrames de uso se construyen directamente.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridcast.features.penalties import (
    COMPONENT_TYPES,
    _poisson_cdf,
    compute_driver_penalties,
    compute_penalty_positions,
    estimate_penalty_risk,
    get_2026_allocations,
    starts_from_pit_lane,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _usage_row(
    driver_id: str,
    component_type: str,
    units_used: int,
    units_allowed: int,
    season: int = 2026,
) -> dict:
    return {
        "driver_id":      driver_id,
        "season":         season,
        "component_type": component_type,
        "units_used":     units_used,
        "units_allowed":  units_allowed,
    }


def _usage_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _driver_within_limits(driver_id: str = "ver", season: int = 2026) -> pd.DataFrame:
    """Driver con todos los componentes dentro del límite."""
    alloc = get_2026_allocations()
    rows = [_usage_row(driver_id, comp, 1, alloc[comp], season) for comp in COMPONENT_TYPES]
    return _usage_df(*rows)


# ---------------------------------------------------------------------------
# compute_penalty_positions
# ---------------------------------------------------------------------------

class TestComputePenaltyPositions:
    def test_zero_over_limit(self):
        assert compute_penalty_positions(0) == 0

    def test_negative_over_limit(self):
        assert compute_penalty_positions(-1) == 0

    def test_first_extra_element(self):
        # 1 elemento extra → 10 puestos (según config)
        assert compute_penalty_positions(1) == 10

    def test_second_extra_element(self):
        # 2 elementos extra → 10 + 5 = 15 puestos
        assert compute_penalty_positions(2) == 15

    def test_third_extra_element(self):
        # 3 elementos extra → 10 + 5 + 5 = 20 puestos
        assert compute_penalty_positions(3) == 20

    def test_monotone_increasing(self):
        p1 = compute_penalty_positions(1)
        p2 = compute_penalty_positions(2)
        p3 = compute_penalty_positions(3)
        assert p1 < p2 < p3


# ---------------------------------------------------------------------------
# starts_from_pit_lane
# ---------------------------------------------------------------------------

class TestStartsFromPitLane:
    def test_below_threshold_no_pit_lane(self):
        # ≤ 15 puestos → no pit lane
        assert not starts_from_pit_lane(15)

    def test_above_threshold_pit_lane(self):
        # > 15 puestos → pit lane
        assert starts_from_pit_lane(16)
        assert starts_from_pit_lane(30)

    def test_zero_penalty_no_pit_lane(self):
        assert not starts_from_pit_lane(0)

    def test_at_boundary_no_pit_lane(self):
        # Exactamente 15: "supera 15" → no (solo si > 15)
        assert not starts_from_pit_lane(15)


# ---------------------------------------------------------------------------
# _poisson_cdf
# ---------------------------------------------------------------------------

class TestPoissonCdf:
    def test_zero_lambda_returns_one(self):
        assert _poisson_cdf(5, 0.0) == pytest.approx(1.0)

    def test_negative_k_returns_zero(self):
        assert _poisson_cdf(-1, 1.0) == 0.0

    def test_large_k_approaches_one(self):
        # P(X ≤ 100) para Poisson(1) ≈ 1.0
        assert _poisson_cdf(100, 1.0) == pytest.approx(1.0, abs=1e-6)

    def test_k_zero_poisson1(self):
        # P(X ≤ 0) para Poisson(1) = e^-1 ≈ 0.368
        import math
        expected = math.exp(-1.0)
        assert _poisson_cdf(0, 1.0) == pytest.approx(expected, rel=1e-6)

    def test_monotone_in_k(self):
        lam = 2.5
        assert _poisson_cdf(0, lam) < _poisson_cdf(1, lam) < _poisson_cdf(5, lam)

    def test_monotone_in_lambda(self):
        # Mayor lambda → menos probabilidad de quedarse en ≤ k
        assert _poisson_cdf(2, 1.0) > _poisson_cdf(2, 5.0)


# ---------------------------------------------------------------------------
# get_2026_allocations
# ---------------------------------------------------------------------------

class TestGet2026Allocations:
    def test_returns_all_component_types(self):
        alloc = get_2026_allocations()
        for comp in COMPONENT_TYPES:
            assert comp in alloc

    def test_ice_allocation(self):
        alloc = get_2026_allocations()
        assert alloc["ICE"] == 4  # 3 base + 1 bonus año nuevo motor

    def test_mgu_k_allocation(self):
        alloc = get_2026_allocations()
        assert alloc["MGU_K"] == 3  # 2 base + 1 bonus

    def test_gb_allocation(self):
        alloc = get_2026_allocations()
        assert alloc["GB"] == 4  # 4 por temporada según config

    def test_all_values_positive(self):
        alloc = get_2026_allocations()
        for comp, v in alloc.items():
            assert v > 0, f"{comp} debe ser > 0"


# ---------------------------------------------------------------------------
# compute_driver_penalties
# ---------------------------------------------------------------------------

class TestComputeDriverPenalties:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"driver_id": ["ver"], "season": [2026]})
        with pytest.raises(ValueError, match="faltan columnas"):
            compute_driver_penalties(df, 2026)

    def test_empty_df(self):
        alloc = get_2026_allocations()
        rows = [_usage_row("ver", comp, 1, alloc[comp]) for comp in COMPONENT_TYPES]
        df = _usage_df(*rows)
        # Filtrar temporada inexistente → vacío
        out = compute_driver_penalties(df, 2025)
        assert out.empty

    def test_within_limits_no_penalty(self):
        df = _driver_within_limits("ver")
        out = compute_driver_penalties(df, 2026)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert row["penalty_positions_total"] == 0
        assert not row["starts_from_pit_lane"]

    def test_one_ice_over_limit(self):
        alloc = get_2026_allocations()
        rows = [_usage_row("ver", comp, 1, alloc[comp]) for comp in COMPONENT_TYPES]
        # 5 ICEs usados, límite = 4 → 1 extra → 10 puestos
        rows_ver = [r for r in rows if r["component_type"] != "ICE"]
        rows_ver.append(_usage_row("ver", "ICE", 5, alloc["ICE"]))
        df = _usage_df(*rows_ver)
        out = compute_driver_penalties(df, 2026)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert row["penalty_positions_total"] == 10
        assert not row["starts_from_pit_lane"]

    def test_two_extras_same_component(self):
        alloc = get_2026_allocations()
        rows = [_usage_row("ver", comp, 1, alloc[comp]) for comp in COMPONENT_TYPES]
        rows = [r for r in rows if r["component_type"] != "ICE"]
        rows.append(_usage_row("ver", "ICE", 6, alloc["ICE"]))  # 2 extras
        df = _usage_df(*rows)
        out = compute_driver_penalties(df, 2026)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert row["penalty_positions_total"] == 15

    def test_pit_lane_when_over_threshold(self):
        """ICE + TC extra → 10 + 10 = 20 > 15 → pit lane."""
        alloc = get_2026_allocations()
        rows = [
            _usage_row("ver", "ICE",   alloc["ICE"]  + 1, alloc["ICE"]),
            _usage_row("ver", "TC",    alloc["TC"]   + 1, alloc["TC"]),
            _usage_row("ver", "EX",    1, alloc["EX"]),
            _usage_row("ver", "MGU_K", 1, alloc["MGU_K"]),
            _usage_row("ver", "ES",    1, alloc["ES"]),
            _usage_row("ver", "CE",    1, alloc["CE"]),
            _usage_row("ver", "GB",    1, alloc["GB"]),
        ]
        df = _usage_df(*rows)
        out = compute_driver_penalties(df, 2026)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert row["penalty_positions_total"] == 20
        assert row["starts_from_pit_lane"]

    def test_output_columns_present(self):
        df = _driver_within_limits("ver")
        out = compute_driver_penalties(df, 2026)
        for comp in COMPONENT_TYPES:
            assert f"{comp}_used" in out.columns
            assert f"{comp}_remaining" in out.columns
        assert "penalty_positions_total" in out.columns
        assert "starts_from_pit_lane"    in out.columns

    def test_remaining_computed_correctly(self):
        alloc = get_2026_allocations()
        df = _driver_within_limits("ver")
        out = compute_driver_penalties(df, 2026)
        row = out[out["driver_id"] == "ver"].iloc[0]
        # Usó 1 ICE, permitidos 4 → remaining = 3
        assert row["ICE_remaining"] == alloc["ICE"] - 1

    def test_two_drivers_independent(self):
        alloc = get_2026_allocations()
        ver_rows = [_usage_row("ver", comp, 1, alloc[comp]) for comp in COMPONENT_TYPES]
        nor_rows = [_usage_row("nor", comp, 1, alloc[comp]) for comp in COMPONENT_TYPES]
        # NOR toma 1 ICE extra
        nor_rows = [r for r in nor_rows if r["component_type"] != "ICE"]
        nor_rows.append(_usage_row("nor", "ICE", 5, alloc["ICE"]))
        df = _usage_df(*ver_rows, *nor_rows)
        out = compute_driver_penalties(df, 2026)
        ver_row = out[out["driver_id"] == "ver"].iloc[0]
        nor_row = out[out["driver_id"] == "nor"].iloc[0]
        assert ver_row["penalty_positions_total"] == 0
        assert nor_row["penalty_positions_total"] == 10

    def test_season_filter(self):
        """Solo devuelve datos de la temporada solicitada."""
        alloc = get_2026_allocations()
        rows_2026 = [_usage_row("ver", comp, 1, alloc[comp], season=2026) for comp in COMPONENT_TYPES]
        rows_2025 = [_usage_row("ver", comp, 2, alloc[comp], season=2025) for comp in COMPONENT_TYPES]
        df = _usage_df(*rows_2026, *rows_2025)
        out = compute_driver_penalties(df, 2026)
        row = out[out["driver_id"] == "ver"].iloc[0]
        # En 2026 usó 1 ICE, no extra → sin penalización
        assert row["penalty_positions_total"] == 0


# ---------------------------------------------------------------------------
# estimate_penalty_risk
# ---------------------------------------------------------------------------

class TestEstimatePenaltyRisk:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"driver_id": ["ver"], "season": [2026]})
        with pytest.raises(ValueError, match="faltan columnas"):
            estimate_penalty_risk(df, 2026, races_remaining=5)

    def test_negative_races_remaining_raises(self):
        df = _driver_within_limits("ver")
        with pytest.raises(ValueError, match="races_remaining"):
            estimate_penalty_risk(df, 2026, races_remaining=-1)

    def test_zero_races_remaining_empty(self):
        df = _driver_within_limits("ver")
        out = estimate_penalty_risk(df, 2026, races_remaining=0)
        assert out.empty

    def test_output_columns_present(self):
        df = _driver_within_limits("ver")
        out = estimate_penalty_risk(df, 2026, races_remaining=5)
        assert {"driver_id", "penalty_risk_season",
                "penalty_risk_next_race", "expected_penalty_pos"} <= set(out.columns)

    def test_full_inventory_low_risk(self):
        """Driver con inventario completo y pocas carreras → riesgo bajo."""
        df = _driver_within_limits("ver")  # 1 ICE used, 3 remaining
        out = estimate_penalty_risk(df, 2026, races_remaining=3)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert row["penalty_risk_season"] < 0.2

    def test_zero_inventory_high_risk(self):
        """Driver sin componentes restantes y muchas carreras → riesgo alto."""
        alloc = get_2026_allocations()
        rows = []
        for comp in COMPONENT_TYPES:
            # Usó toda la asignación
            rows.append(_usage_row("ver", comp, alloc[comp], alloc[comp]))
        df = _usage_df(*rows)
        out = estimate_penalty_risk(df, 2026, races_remaining=10, avg_laps_per_race=60.0)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert row["penalty_risk_season"] > 0.5

    def test_risk_increases_with_more_races(self):
        """Más carreras restantes → mayor riesgo de penalización."""
        alloc = get_2026_allocations()
        # 1 componente restante de cada tipo (ajustado)
        rows = [_usage_row("ver", comp, alloc[comp] - 1, alloc[comp]) for comp in COMPONENT_TYPES]
        df = _usage_df(*rows)
        risk_low  = estimate_penalty_risk(df, 2026, races_remaining=2)
        risk_high = estimate_penalty_risk(df, 2026, races_remaining=10)
        r_low  = risk_low[risk_low["driver_id"] == "ver"]["penalty_risk_season"].iloc[0]
        r_high = risk_high[risk_high["driver_id"] == "ver"]["penalty_risk_season"].iloc[0]
        assert r_low < r_high

    def test_penalty_risk_in_0_1(self):
        df = _driver_within_limits("ver")
        out = estimate_penalty_risk(df, 2026, races_remaining=8)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert 0.0 <= row["penalty_risk_season"] <= 1.0
        assert 0.0 <= row["penalty_risk_next_race"] <= 1.0

    def test_expected_penalty_pos_minimum_10(self):
        """Si hay riesgo, la penalización esperada es al menos 10 puestos."""
        alloc = get_2026_allocations()
        rows = [_usage_row("ver", comp, alloc[comp], alloc[comp]) for comp in COMPONENT_TYPES]
        df = _usage_df(*rows)
        out = estimate_penalty_risk(df, 2026, races_remaining=10)
        row = out[out["driver_id"] == "ver"].iloc[0]
        if row["penalty_risk_season"] > 0:
            assert row["expected_penalty_pos"] >= 10

    def test_next_race_risk_leq_season_risk(self):
        """El riesgo en la próxima carrera no puede superar el riesgo total de temporada."""
        alloc = get_2026_allocations()
        rows = [_usage_row("ver", comp, alloc[comp] - 1, alloc[comp]) for comp in COMPONENT_TYPES]
        df = _usage_df(*rows)
        out = estimate_penalty_risk(df, 2026, races_remaining=6)
        row = out[out["driver_id"] == "ver"].iloc[0]
        assert row["penalty_risk_next_race"] <= row["penalty_risk_season"] + 1e-9

    def test_custom_avg_laps_override(self):
        """Vida media corta → riesgo más alto."""
        alloc = get_2026_allocations()
        rows = [_usage_row("ver", comp, alloc[comp] - 1, alloc[comp]) for comp in COMPONENT_TYPES]
        df = _usage_df(*rows)
        long_life  = {"ICE": 2000.0, "TC": 2000.0, "EX": 2000.0,
                      "MGU_K": 2000.0, "ES": 2000.0, "CE": 2000.0, "GB": 2000.0}
        short_life = {"ICE": 200.0,  "TC": 200.0,  "EX": 200.0,
                      "MGU_K": 200.0, "ES": 200.0, "CE": 200.0, "GB": 200.0}
        r_long  = estimate_penalty_risk(df, 2026, 8, avg_laps_per_component=long_life)
        r_short = estimate_penalty_risk(df, 2026, 8, avg_laps_per_component=short_life)
        risk_long  = r_long[r_long["driver_id"]  == "ver"]["penalty_risk_season"].iloc[0]
        risk_short = r_short[r_short["driver_id"] == "ver"]["penalty_risk_season"].iloc[0]
        assert risk_short > risk_long

    def test_two_drivers_independent(self):
        """Riesgo de cada driver es independiente."""
        alloc = get_2026_allocations()
        # ver: inventario lleno (bajo riesgo)
        ver_rows = [_usage_row("ver", comp, 1, alloc[comp]) for comp in COMPONENT_TYPES]
        # nor: sin inventario (alto riesgo)
        nor_rows = [_usage_row("nor", comp, alloc[comp], alloc[comp]) for comp in COMPONENT_TYPES]
        df = _usage_df(*ver_rows, *nor_rows)
        out = estimate_penalty_risk(df, 2026, races_remaining=8)
        ver_risk = out[out["driver_id"] == "ver"]["penalty_risk_season"].iloc[0]
        nor_risk = out[out["driver_id"] == "nor"]["penalty_risk_season"].iloc[0]
        assert ver_risk < nor_risk
