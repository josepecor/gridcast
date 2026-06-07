"""
Features de pace por piloto a partir de datos de práctica y clasificación.

Dos features principales (§4.1, §7 CLAUDE.md):
  long_run_pace_s  — mediana del tiempo de vuelta fuel-corrected en stints largos de
                     FP2 (normal) o FP1 (sprint). Captura el ritmo de carrera.
  quali_pace_s     — mejor vuelta en clasificación (Q). Señal más limpia que la carrera
                     porque elimina ruido de estrategia; se usa como backbone del modelo.

Corrección de combustible:
  En FP2 los equipos hacen simulaciones de carrera con carga de combustible real.
  Sin corrección, las vueltas tempranas (más pesadas) parecen más lentas que las tardías.
  Se normaliza todas las vueltas al nivel de combustible de la última vuelta del stint:
    corrected = raw - (max_tyre_age - tyre_age) × fuel_kg_per_lap × fuel_s_per_kg
  Config: model.fuel_correction_kg_per_lap=1.6, model.fuel_lap_time_delta_s=0.035

Inputs esperados (columnas de fastf1_client.get_pace_laps / get_laps):
  Driver, Team, LapNumber, LapTime (Timedelta), Stint, TyreLife,
  TrackStatus, season, round, session_type
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from gridcast.config import get_settings

log = logging.getLogger(__name__)

# Columnas mínimas requeridas en practice_laps
_REQUIRED_PRACTICE_COLS = {"Driver", "Team", "LapNumber", "LapTime", "Stint"}


# ---------------------------------------------------------------------------
# Conversión de tiempos
# ---------------------------------------------------------------------------

def _to_seconds(td) -> float:
    if pd.isna(td):
        return float("nan")
    if isinstance(td, (pd.Timedelta, timedelta)):
        return td.total_seconds()
    return float("nan")


def _lap_times_to_seconds(laps: pd.DataFrame) -> pd.DataFrame:
    """Añade columna lap_time_s. Preserva el resto del DataFrame."""
    laps = laps.copy()
    if pd.api.types.is_timedelta64_dtype(laps["LapTime"]):
        laps["lap_time_s"] = laps["LapTime"].dt.total_seconds()
    else:
        laps["lap_time_s"] = laps["LapTime"].apply(_to_seconds)
    return laps


# ---------------------------------------------------------------------------
# Corrección de combustible
# ---------------------------------------------------------------------------

def _add_fuel_correction(
    laps: pd.DataFrame,
    fuel_kg_per_lap: float,
    fuel_s_per_kg: float,
) -> pd.DataFrame:
    """
    Añade corrected_lap_time_s: raw_time normalizado al nivel de combustible
    de la última vuelta del stint (referencia de mínimo combustible).

    Fórmula:
      remaining_laps = max_tyre_age_in_stint − tyre_age
      correction_s   = remaining_laps × fuel_kg_per_lap × fuel_s_per_kg
      corrected      = raw − correction_s

    TyreLife preferido (posición real en el set). Si no disponible, usa
    rank de LapNumber dentro del stint como aproximación.
    """
    laps = laps.copy()

    if "TyreLife" in laps.columns and laps["TyreLife"].notna().any():
        tyre_age     = laps["TyreLife"]
        max_tyre_age = laps.groupby(["Driver", "Stint"])["TyreLife"].transform("max")
    else:
        tyre_age = laps.groupby(["Driver", "Stint"])["LapNumber"].rank(method="first")
        max_tyre_age = laps.groupby(["Driver", "Stint"])["LapNumber"].transform("count")

    fuel_remaining_laps = (max_tyre_age - tyre_age).clip(lower=0)
    laps["fuel_correction_s"]    = fuel_remaining_laps * fuel_kg_per_lap * fuel_s_per_kg
    laps["corrected_lap_time_s"] = laps["lap_time_s"] - laps["fuel_correction_s"]
    return laps


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

def _filter_green_flag(laps: pd.DataFrame) -> pd.DataFrame:
    """Elimina vueltas bajo safety car, VSC o bandera amarilla (TrackStatus != '1')."""
    if "TrackStatus" not in laps.columns:
        return laps
    return laps[laps["TrackStatus"].astype(str) == "1"]


def _filter_long_stints(laps: pd.DataFrame, min_laps: int) -> pd.DataFrame:
    """Deja solo stints con >= min_laps vueltas con lap_time_s válido."""
    counts = (
        laps.dropna(subset=["lap_time_s"])
        .groupby(["Driver", "Stint"])["LapNumber"]
        .count()
    )
    long = counts[counts >= min_laps].reset_index()[["Driver", "Stint"]]
    return laps.merge(long, on=["Driver", "Stint"])


def _remove_stint_outliers(laps: pd.DataFrame) -> pd.DataFrame:
    """Elimina vueltas > Q3 + 3×IQR por driver-stint (vueltas de entrada/salida lentas)."""
    parts: list[pd.DataFrame] = []
    for _, group in laps.groupby(["Driver", "Stint"]):
        q25 = group["corrected_lap_time_s"].quantile(0.25)
        q75 = group["corrected_lap_time_s"].quantile(0.75)
        upper = q75 + 3.0 * (q75 - q25)
        parts.append(group[group["corrected_lap_time_s"] <= upper])
    return pd.concat(parts, ignore_index=True) if parts else laps.iloc[:0].copy()


# ---------------------------------------------------------------------------
# Extracción de features
# ---------------------------------------------------------------------------

def _extract_long_run_pace(
    laps: pd.DataFrame,
    min_laps: int,
    fuel_kg_per_lap: float,
    fuel_s_per_kg: float,
) -> pd.DataFrame:
    """
    Pace de tanda larga por piloto.

    Returns DataFrame: abbreviation, team_name, long_run_pace_s, long_run_n
    """
    empty = pd.DataFrame(
        columns=["abbreviation", "team_name", "long_run_pace_s", "long_run_n"]
    )
    if laps.empty:
        return empty

    laps = _lap_times_to_seconds(laps)
    laps = laps.dropna(subset=["lap_time_s"])
    laps = laps[laps["lap_time_s"] > 60.0]       # < 60s = telemetría errónea
    laps = _filter_green_flag(laps)
    laps = _filter_long_stints(laps, min_laps)
    if laps.empty:
        log.debug("No hay stints con >= %d vueltas válidas en bandera verde", min_laps)
        return empty

    laps = _add_fuel_correction(laps, fuel_kg_per_lap, fuel_s_per_kg)
    laps = laps.dropna(subset=["corrected_lap_time_s"])
    laps = _remove_stint_outliers(laps)
    if laps.empty:
        return empty

    result = (
        laps.groupby("Driver")
        .agg(
            long_run_pace_s=("corrected_lap_time_s", "median"),
            long_run_n=("corrected_lap_time_s", "count"),
            team_name=("Team", "first"),
        )
        .reset_index()
        .rename(columns={"Driver": "abbreviation"})
    )
    result["long_run_n"] = result["long_run_n"].astype(int)
    return result


def _extract_session_best(
    laps: pd.DataFrame,
    out_col: str,
) -> pd.DataFrame:
    """
    Mejor vuelta por piloto en la sesión (sin corrección de combustible).
    Válido para práctica (proxy de quali) y para Q (tiempo real de clasificación).

    Returns DataFrame: abbreviation, {out_col}
    """
    empty = pd.DataFrame(columns=["abbreviation", out_col])
    if laps.empty:
        return empty

    laps = _lap_times_to_seconds(laps)
    laps = laps.dropna(subset=["lap_time_s"])
    laps = laps[laps["lap_time_s"] > 60.0]

    result = (
        laps.groupby("Driver")["lap_time_s"]
        .min()
        .reset_index()
        .rename(columns={"Driver": "abbreviation", "lap_time_s": out_col})
    )
    return result


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compute_pace_features(
    practice_laps: pd.DataFrame,
    quali_laps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calcula features de pace por piloto para una ronda.

    Args:
        practice_laps: Vueltas de FP2 (convencional) o FP1 (sprint).
                       Salida de fastf1_client.get_pace_laps(). Necesita:
                       Driver, Team, LapNumber, LapTime, Stint
                       + season, round, session_type (añadidos por el cliente).
        quali_laps:    Vueltas de Q (fastf1_client.get_laps(..., "Q")).
                       Si None, quali_pace_s = NaN en todas las filas.

    Returns:
        DataFrame con una fila por piloto, columnas:
          season          — temporada
          round           — número de ronda
          session_type    — sesión de práctica usada (FP2 / FP1)
          abbreviation    — código 3 letras del piloto (de FastF1)
          team_name       — nombre del equipo
          long_run_pace_s — mediana fuel-corrected de vueltas en tanda larga (s)
          long_run_n      — número de vueltas usadas en long_run
          fp_best_lap_s   — mejor vuelta en práctica (proxy de pace de clasificación)
          quali_pace_s    — mejor vuelta en Q (NaN si quali_laps no se provee)

    Raises:
        ValueError: si practice_laps no tiene las columnas mínimas.
    """
    missing = _REQUIRED_PRACTICE_COLS - set(practice_laps.columns)
    if missing:
        raise ValueError(
            f"practice_laps le faltan columnas: {missing}. "
            f"Columnas presentes: {list(practice_laps.columns)}"
        )

    cfg = get_settings()

    # Metadata de sesión
    season       = int(practice_laps["season"].iloc[0])       if "season"       in practice_laps.columns and not practice_laps.empty else None
    round_       = int(practice_laps["round"].iloc[0])        if "round"        in practice_laps.columns and not practice_laps.empty else None
    session_type = str(practice_laps["session_type"].iloc[0]) if "session_type" in practice_laps.columns and not practice_laps.empty else "unknown"

    # Feature 1: long run pace (fuel-corrected, solo stints largos en verde)
    long_run = _extract_long_run_pace(
        practice_laps,
        min_laps=cfg.model.min_laps_long_run,
        fuel_kg_per_lap=cfg.model.fuel_correction_kg_per_lap,
        fuel_s_per_kg=cfg.model.fuel_lap_time_delta_s,
    )

    # Feature 2: mejor vuelta en práctica (sin corrección, proxy de pace de Q)
    best_fp = _extract_session_best(practice_laps, out_col="fp_best_lap_s")

    # Merge en outer para incluir pilotos que solo tienen una de las dos métricas
    result = long_run.merge(best_fp, on="abbreviation", how="outer")

    # Feature 3: mejor vuelta en clasificación real
    if quali_laps is not None and not quali_laps.empty:
        q_missing = {"Driver", "LapTime"} - set(quali_laps.columns)
        if q_missing:
            log.warning(
                "quali_laps le faltan columnas %s — quali_pace_s = NaN", q_missing
            )
            result["quali_pace_s"] = float("nan")
        else:
            q_best = _extract_session_best(quali_laps, out_col="quali_pace_s")
            result = result.merge(q_best, on="abbreviation", how="left")
    else:
        result["quali_pace_s"] = float("nan")

    # Rellenar team_name para pilotos que solo tienen best_fp (sin long run)
    if "team_name" not in result.columns:
        result["team_name"] = float("nan")

    # team_name puede quedar vacío para pilotos sin long run; recuperar de práctica
    if practice_laps.empty is False:
        team_lookup = (
            practice_laps[["Driver", "Team"]]
            .drop_duplicates()
            .rename(columns={"Driver": "abbreviation", "Team": "_team_fallback"})
        )
        result = result.merge(team_lookup, on="abbreviation", how="left")
        result["team_name"] = result["team_name"].fillna(result["_team_fallback"])
        result = result.drop(columns=["_team_fallback"])

    result["season"]       = season
    result["round"]        = round_
    result["session_type"] = session_type

    # Garantizar columnas de salida aunque alguna no se haya podido computar
    for col in ["long_run_pace_s", "long_run_n", "fp_best_lap_s", "quali_pace_s"]:
        if col not in result.columns:
            result[col] = float("nan")

    out_cols = [
        "season", "round", "session_type", "abbreviation", "team_name",
        "long_run_pace_s", "long_run_n", "fp_best_lap_s", "quali_pace_s",
    ]
    return result[out_cols].reset_index(drop=True)
