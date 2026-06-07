"""
Estimación de fortaleza de piloto, equipo y motor.

Jerarquía del modelo (§4.1 CLAUDE.md):
  rendimiento = efecto_equipo + efecto_piloto + ruido
  efecto_equipo = efecto_chasis + efecto_motorista

Método (baseline):
  Within-team normalization — backbone del sistema.
  Para cada ronda, la diferencia de ritmo entre compañeros de equipo aísla el efecto
  de piloto (el coche es constante para ambos). Encadenando todas las rondas con datos
  de ambos pilotos válidos se obtiene una estimación de piloto robusta.

  car_speed_{team, r} = -mean_pace_{team, r}   (para 2 pilotos, el driver effect cancela)
  within_team_speed_i  = team_mean_pace_r - pace_i   (higher = faster)
  driver_strength_i    = mean_r(within_team_speed_i)   (centrado en 0)

Incertidumbre:
  sd_i = std(within_team_speed_i_across_rounds) / sqrt(n_obs_i)
  Esta es la incertidumbre del *estimador* (error estándar de la media), no la
  variabilidad de rendimiento vuelta a vuelta (el ruido Gumbel en race_sim lo captura).

Futura extensión (roadmap §10):
  Modelo jerárquico bayesiano (numpyro/pymc) para partial pooling cross-team,
  incertidumbre correlacionada, y prior informativo 2022-2025 para pilotos.
  Esta baseline ya produce una salida con el mismo schema (driver_id, mean, sd).

Convención de escala (ver race_sim.py):
  higher mean → more likely to win   (negado respecto a tiempo en pista)
  sd en la misma unidad (segundos)

Responsabilidad del caller:
  La selección de datos (qué rondas incluir) es responsabilidad del caller.
  Este módulo NO filtra por fecha. El leakage_guard es externo (§9 CLAUDE.md).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_WIDE_PRIOR_SD: float = 2.0
"""sd asignada a pilotos/equipos con < 2 observaciones (incertidumbre amplia)."""

_MIN_SD: float = 0.01
"""Cota inferior de sd — evita sd=0 que bloquea la simulación bayesiana."""

_SIGNAL_COLS: dict[str, str] = {
    "quali": "quali_pace_s",
    "race":  "long_run_pace_s",
}

# Gap máximo aceptable respecto al mínimo del campo por ronda.
# Valores reales F1: spread P1→P22 ~2-3s en quali, ~3-5s en long run.
# Cualquier valor por encima implica que el piloto no estableció tiempo válido.
_MAX_GAP_FROM_FIELD_MIN: dict[str, float] = {
    "quali_pace_s":      5.0,   # segundos; 5s ya es imposible en condiciones normales
    "long_run_pace_s":  10.0,   # long runs tienen más varianza; margen más generoso
}

_REQUIRED_DRIVER_COLS = frozenset({"abbreviation", "team_name", "round"})
_REQUIRED_TEAM_COLS   = frozenset({"team_name", "round"})

# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def fit_driver_strengths(
    pace_features: pd.DataFrame,
    primary_signal: Literal["quali", "race"] = "quali",
    min_rounds: int = 1,
) -> pd.DataFrame:
    """
    Estima la fortaleza relativa de cada piloto a partir de pace features.

    Usa within-team normalization: para cada ronda, computa la velocidad del
    piloto respecto a la media de su equipo. Solo contribuyen las rondas donde
    ambos compañeros tienen datos válidos.

    Args:
        pace_features:   Salida de pace.compute_pace_features(), puede ser multi-ronda.
                         Columnas requeridas: abbreviation, team_name, round.
                         Columnas de señal: quali_pace_s y/o long_run_pace_s.
        primary_signal:  "quali" (usa quali_pace_s) | "race" (usa long_run_pace_s).
                         §4.1 CLAUDE.md: preferir "quali" por menor ruido.
        min_rounds:      Número mínimo de rondas con datos para incluir al piloto.

    Returns:
        DataFrame con columnas:
          driver_id — abreviatura del piloto (de FastF1, e.g. "VER", "NOR")
          mean      — fortaleza estimada en segundos (higher = faster; centrado en 0)
          sd        — incertidumbre del estimador (error estándar de la media)
          n_obs     — número de rondas usadas

    Raises:
        ValueError: si faltan columnas requeridas o primary_signal es inválido.
    """
    signal_col = _get_signal_col(primary_signal)
    _check_cols(pace_features, _REQUIRED_DRIVER_COLS | {signal_col})
    return _fit_driver_within_team(pace_features, signal_col, min_rounds)


def fit_team_strengths(
    pace_features: pd.DataFrame,
    driver_strengths: pd.DataFrame,
    engine_map: dict[str, str] | None = None,
    primary_signal: Literal["quali", "race"] = "quali",
) -> pd.DataFrame:
    """
    Estima la fortaleza del chasis de cada equipo.

    Método: car_speed_{team, r} = -mean_pace_{team, r}, centrado globalmente.
    Para 2 pilotos los efectos de piloto se cancelan en la media del equipo.

    Args:
        pace_features:    Salida de pace.compute_pace_features().
        driver_strengths: Salida de fit_driver_strengths() (no se usa en el cálculo,
                          solo se acepta para documentar la dependencia lógica).
        engine_map:       team_name → engine_id. Si None, engine_id = NaN.
        primary_signal:   Columna de pace a usar.

    Returns:
        DataFrame: team_id, mean, sd, n_obs, engine_id.
    """
    signal_col = _get_signal_col(primary_signal)
    _check_cols(pace_features, _REQUIRED_TEAM_COLS | {signal_col})
    return _fit_team_from_pace(pace_features, signal_col, engine_map)


def fit_engine_strengths(
    team_strengths: pd.DataFrame,
    engine_map: dict[str, str],
) -> pd.DataFrame:
    """
    Estima la fortaleza del motor agrupando los equipos cliente.

    En 2026 el efecto motorista es compartido (§4.1): si un motor es exitoso,
    sube a todos sus clientes simultáneamente.

    Args:
        team_strengths: Salida de fit_team_strengths() con columnas team_id, mean, sd, n_obs.
        engine_map:     team_name → engine_id (ej. {"aston_martin": "mercedes"}).

    Returns:
        DataFrame: engine_id, mean, sd, n_teams.
    """
    if team_strengths.empty:
        return pd.DataFrame(columns=["engine_id", "mean", "sd", "n_teams"])

    _check_cols(team_strengths, {"team_id", "mean", "sd", "n_obs"})

    ts = team_strengths.copy()
    ts["engine_id"] = ts["team_id"].map(engine_map)
    ts = ts.dropna(subset=["engine_id"])

    if ts.empty:
        log.warning("engine_map no coincide con ningún team_id — resultado vacío")
        return pd.DataFrame(columns=["engine_id", "mean", "sd", "n_teams"])

    return _aggregate_engine_strengths(ts)


def fit_all(
    pace_features: pd.DataFrame,
    engine_map: dict[str, str] | None = None,
    primary_signal: Literal["quali", "race"] = "quali",
    min_rounds: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Estima fortalezas de piloto, equipo y motor en un solo paso.

    Returns:
        {
          "driver": DataFrame (driver_id, mean, sd, n_obs),
          "team":   DataFrame (team_id, mean, sd, n_obs, engine_id),
          "engine": DataFrame (engine_id, mean, sd, n_teams) — vacío si engine_map=None,
        }
    """
    driver = fit_driver_strengths(pace_features, primary_signal, min_rounds)
    team   = fit_team_strengths(pace_features, driver, engine_map, primary_signal)
    engine = (
        fit_engine_strengths(team, engine_map)
        if engine_map
        else pd.DataFrame(columns=["engine_id", "mean", "sd", "n_teams"])
    )
    return {"driver": driver, "team": team, "engine": engine}


def fit_strength_model(
    pace_features: pd.DataFrame,
    as_of_round: int,
    engine_map: dict[str, str] | None = None,
    primary_signal: Literal["quali", "race"] = "quali",
    min_rounds: int = 1,
) -> pd.DataFrame:
    """
    Ajusta el modelo de fuerza y devuelve un DataFrame en formato largo.

    Interfaz para pipeline 04. Combina estimaciones de piloto, equipo y motor
    en una tabla única con columna entity_type para facilitar el almacenamiento.

    Args:
        pace_features:  Salida de pipeline 02, ya filtrada al corte point-in-time.
        as_of_round:    Número de la última ronda incluida (se añade como columna).
        engine_map:     team_name → engine_id. None = no agregar por motor.
        primary_signal: "quali" | "race" — señal primaria de pace.
        min_rounds:     Número mínimo de rondas para incluir un piloto.

    Returns:
        DataFrame con columnas:
          as_of_round  — ronda de corte
          entity_type  — "driver" | "team" | "engine"
          entity_id    — abreviatura del piloto, nombre de equipo, o id de motor
          mean         — fortaleza estimada (higher = faster, centrado en 0)
          sd           — incertidumbre del estimador (error estándar)
    """
    all_est = fit_all(pace_features, engine_map, primary_signal, min_rounds)

    parts: list[pd.DataFrame] = []

    def _long(df: pd.DataFrame, id_col: str, etype: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["as_of_round", "entity_type", "entity_id", "mean", "sd"])
        out = df.rename(columns={id_col: "entity_id"})[["entity_id", "mean", "sd"]].copy()
        out["entity_type"] = etype
        out["as_of_round"] = as_of_round
        return out[["as_of_round", "entity_type", "entity_id", "mean", "sd"]]

    parts.append(_long(all_est["driver"], "driver_id", "driver"))
    parts.append(_long(all_est["team"],   "team_id",   "team"))
    if not all_est["engine"].empty:
        parts.append(_long(all_est["engine"], "engine_id", "engine"))

    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        return pd.DataFrame(columns=["as_of_round", "entity_type", "entity_id", "mean", "sd"])

    return pd.concat(non_empty, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers de validación y limpieza
# ---------------------------------------------------------------------------

def _drop_pace_outliers(df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    """
    NaN-ea tiempos de pace claramente inválidos (piloto sin tiempo puesto).

    Regla: para cada ronda, si el valor de un piloto es > mínimo_del_campo + umbral,
    se sustituye por NaN. El umbral refleja el spread máximo realista en F1
    (casi nunca > 3s en quali, > 5s en long run).

    No toca filas sin 'round' ni columnas con otro nombre.
    """
    if "round" not in df.columns or signal_col not in df.columns:
        return df
    max_gap = _MAX_GAP_FROM_FIELD_MIN.get(signal_col)
    if max_gap is None:
        return df

    df = df.copy()
    field_min = df.groupby("round")[signal_col].transform("min")
    bad_mask  = df[signal_col].notna() & (df[signal_col] > field_min + max_gap)
    if bad_mask.any():
        log.warning(
            "%s: %d valor(es) outlier descartados (> campo_min + %.1fs): %s",
            signal_col,
            bad_mask.sum(),
            max_gap,
            df.loc[bad_mask, "abbreviation"].tolist()
            if "abbreviation" in df.columns else "(sin columna abbreviation)",
        )
        df.loc[bad_mask, signal_col] = float("nan")
    return df


def _get_signal_col(primary_signal: str) -> str:
    col = _SIGNAL_COLS.get(primary_signal)
    if col is None:
        raise ValueError(
            f"primary_signal debe ser 'quali' o 'race', no {primary_signal!r}"
        )
    return col


def _check_cols(df: pd.DataFrame, required: frozenset[str] | set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame le faltan columnas: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Implementación: driver strength
# ---------------------------------------------------------------------------

def _fit_driver_within_team(
    pace_features: pd.DataFrame,
    signal_col: str,
    min_rounds: int,
) -> pd.DataFrame:
    """
    Estima fortaleza de piloto via within-team normalization.

    Solo contribuyen las rondas donde el equipo tiene ≥ 2 pilotos con datos
    válidos en la sesión de referencia (para que el gap sea calculable).
    """
    group_cols = [c for c in ("season", "round") if c in pace_features.columns]
    if "round" not in group_cols:
        raise ValueError("pace_features debe tener columna 'round'")
    group_cols.append("team_name")

    df = pace_features[group_cols + ["abbreviation", signal_col]].copy()
    df = _drop_pace_outliers(df, signal_col)
    df = df.dropna(subset=[signal_col, "team_name", "abbreviation"])
    df = df[df["team_name"].astype(str).str.strip() != ""]

    if df.empty:
        log.warning("No hay datos válidos de pace (%s) — DataFrame vacío", signal_col)
        return pd.DataFrame(columns=["driver_id", "mean", "sd", "n_obs"])

    # Solo rondas con ≥ 2 pilotos del equipo con datos
    count_per_group = df.groupby(group_cols)["abbreviation"].transform("count")
    df = df[count_per_group >= 2].copy()

    if df.empty:
        log.warning(
            "Ninguna ronda tiene ≥ 2 pilotos válidos del mismo equipo en %s", signal_col
        )
        return pd.DataFrame(columns=["driver_id", "mean", "sd", "n_obs"])

    # Velocidad relativa a la media del equipo en esa ronda (higher = faster)
    team_mean_pace = df.groupby(group_cols)[signal_col].transform("mean")
    df["within_team_speed"] = team_mean_pace - df[signal_col]

    # Agregar por piloto
    grouped = df.groupby("abbreviation")["within_team_speed"]
    result = pd.DataFrame({
        "driver_id": grouped.mean().index,
        "mean":      grouped.mean().values,
        "_std":      grouped.std(ddof=1).values,
        "n_obs":     grouped.count().values.astype(int),
    }).reset_index(drop=True)

    # Error estándar de la media como medida de incertidumbre del estimador
    result["sd"] = np.where(
        result["n_obs"] >= 2,
        np.maximum(result["_std"] / np.sqrt(result["n_obs"]), _MIN_SD),
        _WIDE_PRIOR_SD,
    )
    result = result.drop(columns=["_std"])

    result = result[result["n_obs"] >= min_rounds].copy().reset_index(drop=True)

    if result.empty:
        return result[["driver_id", "mean", "sd", "n_obs"]].reset_index(drop=True)

    # Centrar en 0 — escala relativa (identifiabilidad)
    result["mean"] = result["mean"] - result["mean"].mean()

    log.info(
        "Driver strengths: %d pilotos, %.1f rondas/piloto promedio, "
        "spread ±%.3f s (1-sigma)",
        len(result),
        result["n_obs"].mean(),
        result["mean"].std(),
    )

    return result[["driver_id", "mean", "sd", "n_obs"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Implementación: team strength
# ---------------------------------------------------------------------------

def _fit_team_from_pace(
    pace_features: pd.DataFrame,
    signal_col: str,
    engine_map: dict[str, str] | None,
) -> pd.DataFrame:
    """
    Estima fortaleza del chasis. car_speed_{team, r} = -mean_pace_{team, r}.

    El driver effect se cancela en la media para equipos de 2 pilotos
    (sum of within-team deviations = 0 por construcción).
    """
    group_cols = [c for c in ("season", "round") if c in pace_features.columns]
    group_cols.append("team_name")

    df = pace_features[group_cols + [signal_col]].copy()
    df = _drop_pace_outliers(df, signal_col)
    df = df.dropna(subset=[signal_col, "team_name"])
    df = df[df["team_name"].astype(str).str.strip() != ""]

    if df.empty:
        return pd.DataFrame(columns=["team_id", "mean", "sd", "n_obs", "engine_id"])

    # Pace medio por equipo y ronda
    team_round = df.groupby(group_cols)[signal_col].mean().reset_index()
    team_round.columns = group_cols + ["mean_pace"]

    # Car speed = negated pace; centrar POR RONDA (relativo al campo en ese circuito)
    # Centrado global mezclaría circuitos con tiempos absolutos distintos (Monza vs Monaco)
    team_round["car_speed"] = -team_round["mean_pace"]
    round_means = team_round.groupby("round")["car_speed"].transform("mean")
    team_round["car_speed"] -= round_means

    # Agregar por equipo
    grouped = team_round.groupby("team_name")["car_speed"]
    result = pd.DataFrame({
        "team_id": grouped.mean().index,
        "mean":    grouped.mean().values,
        "_std":    grouped.std(ddof=1).values,
        "n_obs":   grouped.count().values.astype(int),
    }).reset_index(drop=True)

    result["sd"] = np.where(
        result["n_obs"] >= 2,
        np.maximum(result["_std"] / np.sqrt(result["n_obs"]), _MIN_SD),
        _WIDE_PRIOR_SD,
    )
    result = result.drop(columns=["_std"])
    result["engine_id"] = result["team_id"].map(engine_map) if engine_map else float("nan")

    log.info("Team strengths: %d equipos", len(result))

    return result[["team_id", "mean", "sd", "n_obs", "engine_id"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Implementación: engine strength
# ---------------------------------------------------------------------------

def _aggregate_engine_strengths(team_strengths_with_engine: pd.DataFrame) -> pd.DataFrame:
    """
    Media ponderada de fortalezas de equipo por fabricante de motor.

    Ponderación por n_obs (más datos = más peso).
    sd del motor: desviación ponderada entre equipos cliente / sqrt(n_equipos).
    """
    def _engine_stats(grp: pd.DataFrame) -> pd.Series:
        weights = grp["n_obs"].values.astype(float)
        means   = grp["mean"].values
        n_teams = len(grp)

        if weights.sum() == 0:
            return pd.Series({"mean": 0.0, "sd": _WIDE_PRIOR_SD, "n_teams": n_teams})

        wmean = float(np.average(means, weights=weights))

        if n_teams >= 2:
            wstd  = float(np.sqrt(np.average((means - wmean) ** 2, weights=weights)))
            n_eff = float(weights.sum() / weights.max())
            sd    = max(wstd / np.sqrt(n_eff), _MIN_SD)
        else:
            sd = _WIDE_PRIOR_SD  # un único equipo → incertidumbre amplia

        return pd.Series({"mean": wmean, "sd": sd, "n_teams": float(n_teams)})

    agg = (
        team_strengths_with_engine
        .groupby("engine_id", group_keys=False)
        .apply(_engine_stats, include_groups=False)
        .reset_index()
    )
    agg["n_teams"] = agg["n_teams"].astype(int)

    return agg[["engine_id", "mean", "sd", "n_teams"]].reset_index(drop=True)
