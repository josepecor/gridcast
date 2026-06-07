"""
Features de fiabilidad por piloto, equipo y motorista.

Qué computa:
  mechanical_dnf_rate — abandonos mecánicos / arranques (Bayesian-smoothed)
  incident_dnf_rate   — abandonos por incidente / arranques (Bayesian-smoothed)
  total_dnf_rate      — todos los abandonos / arranques (Bayesian-smoothed)

Los rates se suavizan con un prior Beta para evitar tasas 0% o 100% con pocos datos.
El prior equivale a ~15 arranques de historia previa con tasas F1 típicas.

Separación mecánico / incidente:
  - Mecánico: fallo del coche (motor, caja, frenos, etc.)
  - Incidente: piloto o contacto (colisión, vuelta)
  - Otro: "Retired" sin razón clara → cuenta en total_dnf_rate pero no en los sub-rates

Input: list[RaceResult] ya filtrada point-in-time (el llamador garantiza leakage=0).
Output: DataFrames listos para usar en race_sim / season_sim.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from gridcast.ingest.jolpica_client import RaceResult

log = logging.getLogger(__name__)

# Palabras clave en status de Jolpica que indican fallo mecánico
_MECHANICAL_KEYWORDS: frozenset[str] = frozenset({
    "engine", "gearbox", "hydraulics", "power unit", "turbo", "turbocharger",
    "brakes", "suspension", "wheel", "driveshaft", "differential", "electrical",
    "electronics", "cooling", "oil leak", "water leak", "fuel system",
    "clutch", "transmission", "exhaust", "overheating", "mechanical",
    "power loss", "fire", "oil pressure", "water pressure", "fuel pressure",
    "battery", "energy store", "vibrations", "technical",
})

# Palabras clave en status que indican abandono por incidente de carrera
_INCIDENT_KEYWORDS: frozenset[str] = frozenset({
    "collision", "accident", "spun off", "spin", "crash", "debris",
    "collision damage", "damage", "track limits", "safety concerns",
    "contact", "out of fuel",  # "out of fuel" = driver error estratégico
})

# Priors Beta(α, β) basados en tasas históricas de F1 (~2022-2025)
# Equivale a observar α DNFs en (α+β) arranques como creencia previa
_PRIOR_MECHANICAL: tuple[float, float] = (1.2, 13.8)   # media prior ≈ 8%
_PRIOR_INCIDENT:   tuple[float, float] = (0.8, 14.2)   # media prior ≈ 5%
_PRIOR_TOTAL:      tuple[float, float] = (2.0, 13.0)   # media prior ≈ 13%

DnfCategory = Literal["finished", "mechanical", "incident", "other"]


# ---------------------------------------------------------------------------
# Clasificación de status
# ---------------------------------------------------------------------------

def classify_status(status: str) -> DnfCategory:
    """
    Clasifica el status de Jolpica en una categoría de abandono.

    Returns:
      "finished"   — terminó (incluyendo +N laps clasificados)
      "mechanical" — fallo mecánico confirmado
      "incident"   — accidente/colisión
      "other"      — "Retired" genérico u otro motivo no clasificado
    """
    s = status.lower().strip()

    if s == "finished" or s.startswith("+"):
        return "finished"

    for kw in _MECHANICAL_KEYWORDS:
        if kw in s:
            return "mechanical"

    for kw in _INCIDENT_KEYWORDS:
        if kw in s:
            return "incident"

    return "other"  # "Retired", "Disqualified", "Did not finish", etc.


# ---------------------------------------------------------------------------
# Smoothing bayesiano
# ---------------------------------------------------------------------------

def _bayesian_rate(successes: int, trials: int, prior: tuple[float, float]) -> float:
    """Posterior mean de Beta(α+successes, β+trials-successes)."""
    alpha, beta = prior
    return (successes + alpha) / (trials + alpha + beta)


# ---------------------------------------------------------------------------
# Conversión list[RaceResult] → DataFrame interno
# ---------------------------------------------------------------------------

def _results_to_df(results: list[RaceResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=[
            "season", "round", "driver_id", "constructor_id", "status",
            "laps_completed", "dnf_category",
        ])
    rows = []
    for r in results:
        rows.append({
            "season":         r.season,
            "round":          r.round,
            "driver_id":      r.driver_id,
            "constructor_id": r.constructor_id,
            "status":         r.status,
            "laps_completed": r.laps_completed,
            "dnf_category":   classify_status(r.status),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compute_driver_reliability(
    results: list[RaceResult],
    min_starts: int = 1,
) -> pd.DataFrame:
    """
    Tasas de abandono Bayesian-smoothed por piloto.

    Args:
        results:    Resultados de carrera ya filtrados point-in-time.
        min_starts: Pilotos con menos de este número de arranques se incluyen
                    pero sus rates estarán muy dominados por el prior.

    Returns:
        DataFrame con una fila por driver_id, columnas:
          driver_id, starts,
          mechanical_dnfs, incident_dnfs, other_dnfs, total_dnfs,
          mechanical_dnf_rate, incident_dnf_rate, total_dnf_rate
          (rates = posterior mean de Beta)
    """
    df = _results_to_df(results)
    if df.empty:
        return pd.DataFrame(columns=[
            "driver_id", "starts",
            "mechanical_dnfs", "incident_dnfs", "other_dnfs", "total_dnfs",
            "mechanical_dnf_rate", "incident_dnf_rate", "total_dnf_rate",
        ])

    rows = []
    for driver_id, group in df.groupby("driver_id"):
        starts    = len(group)
        mech_dnfs = (group["dnf_category"] == "mechanical").sum()
        inc_dnfs  = (group["dnf_category"] == "incident").sum()
        other_dnf = (group["dnf_category"] == "other").sum()
        total_dnf = mech_dnfs + inc_dnfs + other_dnf

        if starts < min_starts:
            log.debug("driver %s: solo %d arranques — rate muy influido por prior", driver_id, starts)

        rows.append({
            "driver_id":           driver_id,
            "starts":              starts,
            "mechanical_dnfs":     int(mech_dnfs),
            "incident_dnfs":       int(inc_dnfs),
            "other_dnfs":          int(other_dnf),
            "total_dnfs":          int(total_dnf),
            "mechanical_dnf_rate": _bayesian_rate(mech_dnfs, starts, _PRIOR_MECHANICAL),
            "incident_dnf_rate":   _bayesian_rate(inc_dnfs,  starts, _PRIOR_INCIDENT),
            "total_dnf_rate":      _bayesian_rate(total_dnf, starts, _PRIOR_TOTAL),
        })

    return pd.DataFrame(rows).reset_index(drop=True)


def compute_team_reliability(
    results: list[RaceResult],
) -> pd.DataFrame:
    """
    Tasas de abandono mecánico Bayesian-smoothed por equipo (constructor_id).

    Un fallo mecánico que afecte a ambos pilotos del equipo cuenta dos veces:
    la unidad de medida es el arranque de piloto, no el fin de semana.

    Returns:
        DataFrame con una fila por constructor_id, columnas:
          constructor_id, starts, mechanical_dnfs, total_dnfs,
          mechanical_dnf_rate, total_dnf_rate
    """
    df = _results_to_df(results)
    if df.empty:
        return pd.DataFrame(columns=[
            "constructor_id", "starts", "mechanical_dnfs", "total_dnfs",
            "mechanical_dnf_rate", "total_dnf_rate",
        ])

    rows = []
    for constructor_id, group in df.groupby("constructor_id"):
        starts    = len(group)
        mech_dnfs = (group["dnf_category"] == "mechanical").sum()
        total_dnf = (group["dnf_category"] != "finished").sum()
        rows.append({
            "constructor_id":      constructor_id,
            "starts":              starts,
            "mechanical_dnfs":     int(mech_dnfs),
            "total_dnfs":          int(total_dnf),
            "mechanical_dnf_rate": _bayesian_rate(mech_dnfs, starts, _PRIOR_MECHANICAL),
            "total_dnf_rate":      _bayesian_rate(total_dnf, starts, _PRIOR_TOTAL),
        })

    return pd.DataFrame(rows).reset_index(drop=True)


def compute_engine_reliability(
    results: list[RaceResult],
    engine_map: dict[str, str],
) -> pd.DataFrame:
    """
    Tasas de fallo de motor Bayesian-smoothed por motorista.

    El efecto del motorista es compartido entre todos sus equipos cliente
    (ver §4.1 CLAUDE.md: efecto_motorista compartido). Esta función agrega
    todos los arranques de pilotos con el mismo motor.

    Args:
        results:    Resultados ya filtrados point-in-time.
        engine_map: constructor_id → engine_supplier (e.g. {"red_bull": "honda_rbpt"}).
                    Equipos no presentes se excluyen del agregado.

    Returns:
        DataFrame con una fila por engine_supplier, columnas:
          engine_supplier, starts, mechanical_dnfs, total_dnfs,
          mechanical_dnf_rate, total_dnf_rate
    """
    df = _results_to_df(results)
    if df.empty or not engine_map:
        return pd.DataFrame(columns=[
            "engine_supplier", "starts", "mechanical_dnfs", "total_dnfs",
            "mechanical_dnf_rate", "total_dnf_rate",
        ])

    df["engine_supplier"] = df["constructor_id"].map(engine_map)
    df = df.dropna(subset=["engine_supplier"])

    if df.empty:
        log.warning("engine_map no mapeó ningún constructor_id — DataFrame vacío")
        return pd.DataFrame(columns=[
            "engine_supplier", "starts", "mechanical_dnfs", "total_dnfs",
            "mechanical_dnf_rate", "total_dnf_rate",
        ])

    rows = []
    for engine, group in df.groupby("engine_supplier"):
        starts    = len(group)
        mech_dnfs = (group["dnf_category"] == "mechanical").sum()
        total_dnf = (group["dnf_category"] != "finished").sum()
        rows.append({
            "engine_supplier":     engine,
            "starts":              starts,
            "mechanical_dnfs":     int(mech_dnfs),
            "total_dnfs":          int(total_dnf),
            "mechanical_dnf_rate": _bayesian_rate(mech_dnfs, starts, _PRIOR_MECHANICAL),
            "total_dnf_rate":      _bayesian_rate(total_dnf, starts, _PRIOR_TOTAL),
        })

    return pd.DataFrame(rows).reset_index(drop=True)


def compute_reliability_features(
    results: list[RaceResult],
    engine_map: dict[str, str] | None = None,
    min_starts: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Calcula todas las features de fiabilidad en una sola llamada.

    Returns:
        {
          "driver": compute_driver_reliability(...),
          "team":   compute_team_reliability(...),
          "engine": compute_engine_reliability(...) | None si no se provee engine_map,
        }
    """
    driver_df = compute_driver_reliability(results, min_starts=min_starts)
    team_df   = compute_team_reliability(results)

    if engine_map:
        engine_df: pd.DataFrame | None = compute_engine_reliability(results, engine_map)
    else:
        log.debug("engine_map no provisto — engine_df omitido")
        engine_df = None

    return {
        "driver": driver_df,
        "team":   team_df,
        "engine": engine_df,
    }
