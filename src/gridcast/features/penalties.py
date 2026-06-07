"""
Features de penalización de parrilla por uso de componentes de Unidad de Potencia.

Dos responsabilidades:
  1. Cálculo determinista — dado el uso actual de componentes, cuántas posiciones
     de penalización acumula un piloto y si sale desde el pit lane.
  2. Estimación de riesgo futuro — dado el inventario actual, probabilidad de
     necesitar tomar más componentes en las rondas restantes (modelo hazard simple).

Reglas de penalización (§3.4 CLAUDE.md, fuente: reglamento FIA 2026):
  - 1er elemento extra de un tipo en la temporada → 10 puestos de retroceso.
  - Cada elemento extra adicional del mismo tipo →  5 puestos.
  - Si la penalización total acumulada supera 15 puestos → salida desde pit lane.

Fuente de datos de uso:
  El uso real de componentes NO está disponible en FastF1 ni en Jolpica. Los datos
  de uso se alimentan desde la tabla component_usage (mantenerla manualmente o
  desde documentos técnicos FIA). Ver §3.4 CLAUDE.md.

Formato esperado del DataFrame de uso (tabla component_usage):
  driver_id        — identificador de piloto (jolpica_id)
  season           — temporada
  component_type   — "ICE" | "TC" | "EX" | "MGU_K" | "ES" | "CE" | "GB"
  units_used       — número de unidades tomadas hasta ahora (incluyendo la inicial)
  units_allowed    — asignación permitida (de config/components.yaml)

Modelo de riesgo futuro (hazard):
  Usa una aproximación Poisson para estimar P(necesitar más unidades en las
  rondas restantes). Los parámetros de vida media por componente son estimaciones
  del dominio; en 2026 (reglamento nuevo) la incertidumbre es alta.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import pandas as pd

from gridcast.config import get_components

log = logging.getLogger(__name__)

# Tipos de componente de PU válidos en 2026 (MGU-H eliminado)
ComponentType = Literal["ICE", "TC", "EX", "MGU_K", "ES", "CE", "GB"]

COMPONENT_TYPES: tuple[ComponentType, ...] = ("ICE", "TC", "EX", "MGU_K", "ES", "CE", "GB")

# Columnas mínimas del DataFrame de uso de componentes
_REQUIRED_USAGE_COLS: frozenset[str] = frozenset({
    "driver_id", "season", "component_type", "units_used", "units_allowed",
})

# Vida media estimada por componente en vueltas de carrera (2026, alta incertidumbre)
# Derivado de: temporada 22 carreras × ~60 vueltas = ~1320 total / asignación
#   ICE: 1320/4 = 330 vueltas esperadas por unidad → equipos las estiran a ~450
# Estos defaults se actualizarán empíricamente tras las primeras carreras de 2026.
_DEFAULT_AVG_LAPS_PER_COMPONENT: dict[str, float] = {
    "ICE":   450.0,   # ~7-8 fines de semana por unidad
    "TC":    450.0,
    "EX":    450.0,
    "MGU_K": 500.0,   # ligeramente más duradero que el ICE
    "ES":    600.0,   # suele durar más
    "CE":    700.0,   # más duradero todavía
    "GB":    400.0,   # caja de cambios: ~6-7 carreras
}


# ---------------------------------------------------------------------------
# Cálculo determinista de penalizaciones
# ---------------------------------------------------------------------------

def compute_penalty_positions(units_over_limit: int) -> int:
    """
    Posiciones de penalización por tomar `units_over_limit` elementos por encima
    del límite en un MISMO tipo de componente y temporada.

    Ejemplo: 2 ICEs extra → 10 + 5 = 15 puestos.
    """
    if units_over_limit <= 0:
        return 0
    cfg = get_components()
    rules = cfg.penalties
    return rules.first_extra_element + (units_over_limit - 1) * rules.subsequent_elements


def starts_from_pit_lane(total_penalty_positions: int) -> bool:
    """
    True si la penalización total acumulada supera el umbral para salir desde pit lane.
    El umbral es estrictamente mayor que (no >=) según el reglamento FIA.
    """
    cfg = get_components()
    return total_penalty_positions > cfg.penalties.pit_lane_threshold


# ---------------------------------------------------------------------------
# Features por piloto (deterministas)
# ---------------------------------------------------------------------------

def compute_driver_penalties(usage_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Calcula penalizaciones de parrilla ya acumuladas por cada piloto.

    La penalización es el resultado de haber tomado más componentes de los
    permitidos. Esta función asume que `usage_df` refleja el estado actual
    (point-in-time del llamador).

    Args:
        usage_df: DataFrame con columnas: driver_id, season, component_type,
                  units_used, units_allowed.
        season:   Temporada a filtrar.

    Returns:
        DataFrame con una fila por driver_id, columnas:
          driver_id,
          {comp}_used, {comp}_remaining   — para cada ComponentType,
          penalty_positions_total,        — suma de posiciones de penalización
          starts_from_pit_lane            — bool

    Raises:
        ValueError: si usage_df no tiene las columnas mínimas.
    """
    missing = _REQUIRED_USAGE_COLS - set(usage_df.columns)
    if missing:
        raise ValueError(
            f"usage_df le faltan columnas: {missing}. "
            "Formato esperado: driver_id, season, component_type, units_used, units_allowed."
        )

    season_df = usage_df[usage_df["season"] == season].copy()
    if season_df.empty:
        log.debug("compute_driver_penalties: no hay datos para temporada %d", season)
        return _empty_penalty_df()

    rows = []
    for driver_id, group in season_df.groupby("driver_id"):
        row: dict = {"driver_id": driver_id}
        total_penalty = 0

        for comp in COMPONENT_TYPES:
            comp_rows = group[group["component_type"] == comp]
            if comp_rows.empty:
                row[f"{comp}_used"]      = 0
                row[f"{comp}_remaining"] = 0
                continue

            # Si hay múltiples entradas del mismo tipo, tomar la más reciente (max units_used)
            units_used    = int(comp_rows["units_used"].max())
            units_allowed = int(comp_rows["units_allowed"].iloc[-1])
            units_over    = max(0, units_used - units_allowed)

            row[f"{comp}_used"]      = units_used
            row[f"{comp}_remaining"] = max(0, units_allowed - units_used)

            comp_penalty = compute_penalty_positions(units_over)
            total_penalty += comp_penalty

            if comp_penalty > 0:
                log.debug(
                    "driver=%s comp=%s: %d sobre límite → %d puestos",
                    driver_id, comp, units_over, comp_penalty,
                )

        row["penalty_positions_total"] = total_penalty
        row["starts_from_pit_lane"]    = starts_from_pit_lane(total_penalty)
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def _empty_penalty_df() -> pd.DataFrame:
    cols = ["driver_id"]
    for comp in COMPONENT_TYPES:
        cols += [f"{comp}_used", f"{comp}_remaining"]
    cols += ["penalty_positions_total", "starts_from_pit_lane"]
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Estimación de riesgo futuro (modelo hazard)
# ---------------------------------------------------------------------------

def _poisson_cdf(k: int, lam: float) -> float:
    """P(X ≤ k) para X ~ Poisson(λ). Implementado sin scipy."""
    if lam <= 0.0:
        return 1.0
    if k < 0:
        return 0.0
    cumulative = 0.0
    term = math.exp(-lam)
    cumulative += term
    for i in range(1, k + 1):
        term *= lam / i
        cumulative += term
    return min(cumulative, 1.0)


def estimate_penalty_risk(
    usage_df: pd.DataFrame,
    season: int,
    races_remaining: int,
    avg_laps_per_race: float = 58.0,
    avg_laps_per_component: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Estima la probabilidad de incurrir en penalización de parrilla durante
    las rondas restantes de la temporada.

    Modelo: Poisson con λ = laps_remaining / avg_life_component.
    P(necesitar > units_remaining) = 1 - Poisson_CDF(units_remaining, λ).
    P(penalización en algún tipo) = 1 - producto(1 - P(tipo)) sobre todos los tipos.

    La probabilidad en la PRÓXIMA carrera se aproxima dividiendo la probabilidad
    de temporada entre el número de carreras restantes. Esta aproximación ignora
    que los equipos eligen CUÁNDO tomar la penalización, pero da un gradiente útil
    para la simulación Monte Carlo.

    Args:
        usage_df:               DataFrame de uso de componentes (mismo formato que
                                compute_driver_penalties).
        season:                 Temporada a filtrar.
        races_remaining:        Rondas que quedan en la temporada.
        avg_laps_per_race:      Vueltas de carrera medias (no incluye práctica).
        avg_laps_per_component: Override de vida media en vueltas por tipo.
                                Si None, usa los defaults de 2026.

    Returns:
        DataFrame con una fila por driver_id, columnas:
          driver_id,
          penalty_risk_season    — P(penalización alguna vez en resto de temporada)
          penalty_risk_next_race — P(penalización en la próxima carrera; aproximación)
          expected_penalty_pos   — posiciones esperadas de penalización si ocurre
                                   (siempre ≥ 10, en función de cuántas unidades extra)
    """
    missing = _REQUIRED_USAGE_COLS - set(usage_df.columns)
    if missing:
        raise ValueError(
            f"usage_df le faltan columnas: {missing}. "
            "Formato esperado: driver_id, season, component_type, units_used, units_allowed."
        )

    if races_remaining < 0:
        raise ValueError(f"races_remaining debe ser ≥ 0, recibido: {races_remaining}")

    avg_life = {**_DEFAULT_AVG_LAPS_PER_COMPONENT, **(avg_laps_per_component or {})}
    season_df = usage_df[usage_df["season"] == season].copy()

    if season_df.empty or races_remaining == 0:
        log.debug("estimate_penalty_risk: sin datos o races_remaining=0")
        return pd.DataFrame(columns=[
            "driver_id", "penalty_risk_season", "penalty_risk_next_race",
            "expected_penalty_pos",
        ])

    remaining_laps = races_remaining * avg_laps_per_race
    rows = []

    for driver_id, group in season_df.groupby("driver_id"):
        # P(sin penalización en ningún tipo) = producto de P(tipo OK)
        p_no_penalty = 1.0
        expected_extra_units_total = 0.0

        for comp in COMPONENT_TYPES:
            comp_rows = group[group["component_type"] == comp]
            if comp_rows.empty:
                continue

            units_used    = int(comp_rows["units_used"].max())
            units_allowed = int(comp_rows["units_allowed"].iloc[-1])
            units_remaining = max(0, units_allowed - units_used)
            life = avg_life.get(comp, 450.0)

            # λ = unidades esperadas necesarias para el resto de temporada
            lam = remaining_laps / life
            # P(necesitar exactamente k o menos unidades adicionales)
            p_within_budget = _poisson_cdf(units_remaining, lam)
            p_over = 1.0 - p_within_budget

            p_no_penalty *= (1.0 - p_over)

            # Unidades extra esperadas dado que se supera el límite
            # E[X - units_remaining | X > units_remaining] ≈ max(0, λ - units_remaining)
            expected_extra = max(0.0, lam - units_remaining)
            expected_extra_units_total += expected_extra * p_over

        p_season = 1.0 - p_no_penalty
        p_next   = p_season / races_remaining if races_remaining > 0 else 0.0

        # Posiciones esperadas si se toma penalización (redondeo a entero)
        # = compute_penalty_positions(expected_extra_units)
        cfg        = get_components()
        rules      = cfg.penalties
        exp_extra  = round(expected_extra_units_total)
        exp_pos    = (
            rules.first_extra_element
            + max(0, exp_extra - 1) * rules.subsequent_elements
        ) if exp_extra >= 1 else rules.first_extra_element

        rows.append({
            "driver_id":           driver_id,
            "penalty_risk_season": round(p_season, 6),
            "penalty_risk_next_race": round(min(p_next, 1.0), 6),
            "expected_penalty_pos":   int(exp_pos),
        })

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Función de conveniencia: asignaciones del reglamento para la temporada activa
# ---------------------------------------------------------------------------

def get_2026_allocations() -> dict[str, int]:
    """
    Devuelve el mapa component_type → units_allowed para 2026, desde config.

    Útil para rellenar la columna units_allowed del DataFrame de uso cuando
    se construye manualmente la tabla component_usage.
    """
    cfg = get_components()
    alloc = cfg.allocations_2026
    return {
        "ICE":   alloc.ICE,
        "TC":    alloc.TC,
        "EX":    alloc.EX,
        "MGU_K": alloc.MGU_K,
        "ES":    alloc.ES,
        "CE":    alloc.CE,
        "GB":    cfg.gearbox.allowed_per_season,
    }
