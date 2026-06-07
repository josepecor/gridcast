"""
Tabla driver_session_map: quién condujo realmente en cada sesión y flag es_titular.

Reglas de dominio (§3.3 CLAUDE.md):
  - es_titular = False  SOLO puede ocurrir en FP1 de fin de semana CONVENCIONAL.
  - Sprint FP1           → siempre es_titular = True. Los equipos necesitan al
                           titular en la única práctica disponible; no hay obligación
                           de rookie cuando solo hay una sesión de libres.
  - FP2, FP3, Q, SQ, S, R → siempre es_titular = True.

Columnas de salida:
  season        int    temporada
  round         int    número de ronda (de Jolpica, no hardcodeado)
  session_type  str    'FP1' | 'FP2' | 'FP3' | 'SQ' | 'S' | 'Q' | 'R'
  event_format  str    'conventional' | 'sprint_qualifying'
  car_number    str    número de coche ('63', '44', …)
  abbreviation  str    código 3 letras ('RUS', 'LEC', …)
  full_name     str    nombre completo del piloto
  team_name     str    nombre del equipo
  es_titular    bool   False solo en FP1 conventional si el piloto ≠ al de la carrera anterior

Estrategia para determinar el piloto "titular":
  - Ronda N (N > 1): comparar vs. los pilotos de la carrera de la ronda N-1 (FastF1).
  - Ronda 1:         usar los propios resultados de la carrera de la ronda 1 como
                     referencia (metadata post-hoc, no un feature predictivo).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gridcast.ingest.fastf1_client import (
    EventFormat,
    SessionType,
    PRACTICE_SESSIONS,
    rookie_check_applies,
    get_results,
    get_drivers_in_session,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_INTERIM_DIR  = _PROJECT_ROOT / "data" / "interim"

# Sesiones completas por formato que el pipeline procesa
_ALL_SESSIONS: dict[EventFormat, list[SessionType]] = {
    "conventional":      ["FP1", "FP2", "FP3", "Q", "R"],
    "sprint_qualifying": ["FP1", "SQ", "S", "Q", "R"],
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _race_driver_map(season: int, round_: int) -> dict[str, str]:
    """
    Devuelve {car_number: abbreviation} de los pilotos que corrieron la carrera.
    Fuente: FastF1 session.results para 'R'.
    """
    try:
        results = get_results(season, round_, "R")
        return {
            str(int(row["DriverNumber"])): row["Abbreviation"]
            for _, row in results.iterrows()
            if pd.notna(row.get("DriverNumber")) and pd.notna(row.get("Abbreviation"))
        }
    except Exception as exc:
        log.warning("No se pudo obtener carrera R%d/%d: %s", season, round_, exc)
        return {}


def _compute_es_titular(
    session_drivers: pd.DataFrame,
    reference: dict[str, str],
) -> pd.Series:
    """
    Para cada fila de session_drivers, devuelve True si el piloto es el titular.

    Detección fiable:
      Mismo car_number, distinta abbreviation → es_titular = False ✓
      (rookie que usa el número habitual del titular, caso más frecuente)

    Falsa negativa conocida:
      Rookie que recibe un número de coche NUEVO (ej. Crawford #34 en lugar de Alonso #14).
      No podemos distinguir esto de un equipo nuevo → marcamos True y emitimos warning.
      Anotación manual necesaria para estos casos.
    """
    present_abbrevs = set(session_drivers["abbreviation"])
    absent_titulares = {
        abbrev for abbrev in reference.values()
        if abbrev not in present_abbrevs
    }
    if absent_titulares:
        log.warning(
            "FP1: titulares ausentes %s — posible rookie con número nuevo (falsa negativa en es_titular)",
            sorted(absent_titulares),
        )

    def _check(row: pd.Series) -> bool:
        car = str(int(row["car_number"]))
        abbrev = row["abbreviation"]
        expected = reference.get(car)
        if expected is None:
            # Número de coche no visto antes: no podemos determinar si es titular.
            # Asumimos True (conservador) para evitar falsos positivos.
            return True
        return abbrev == expected

    return session_drivers.apply(_check, axis=1)


def _drivers_df(
    season: int,
    round_: int,
    session_type: SessionType,
) -> pd.DataFrame:
    """
    Carga los pilotos de una sesión y devuelve un DataFrame normalizado con
    columnas: car_number, abbreviation, full_name, team_name.
    """
    raw = get_drivers_in_session(season, round_, session_type)

    df = pd.DataFrame()
    df["car_number"]  = raw["DriverNumber"].astype(str).str.strip().str.lstrip("0")
    df["abbreviation"] = raw["Abbreviation"].str.upper()
    df["full_name"]   = raw.get("FullName", pd.Series([""] * len(raw)))
    df["team_name"]   = raw.get("TeamName", pd.Series([""] * len(raw)))
    return df


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def build_session_map(
    season: int,
    round_: int,
    session_type: SessionType,
    event_format: EventFormat,
) -> pd.DataFrame:
    """
    Construye el driver_session_map para UNA sesión.

    Para FP1 convencional calcula es_titular comparando contra la carrera
    de la ronda anterior (o la propia R1 si no hay rondas previas).
    Para el resto de sesiones es_titular = True siempre.
    """
    log.info("driver_session_map: season=%d round=%d session=%s format=%s",
             season, round_, session_type, event_format)

    df = _drivers_df(season, round_, session_type)

    if rookie_check_applies(event_format, session_type):
        reference_round = round_ if round_ == 1 else round_ - 1
        reference = _race_driver_map(season, reference_round)
        if not reference:
            log.warning(
                "Sin referencia de carrera para R%d — marcando todos como titulares", reference_round
            )
            df["es_titular"] = True
        else:
            df["es_titular"] = _compute_es_titular(df, reference)
        non_titular = (~df["es_titular"]).sum()
        if non_titular:
            log.info("FP1 R%d: %d piloto(s) no titular(es) detectados", round_, non_titular)
    else:
        df["es_titular"] = True

    df.insert(0, "season",       season)
    df.insert(1, "round",        round_)
    df.insert(2, "session_type", session_type)
    df.insert(3, "event_format", event_format)

    return df.reset_index(drop=True)


def build_round_map(
    season: int,
    round_: int,
    event_format: EventFormat,
) -> pd.DataFrame:
    """
    Construye el driver_session_map para TODAS las sesiones de una ronda.
    Las sesiones se seleccionan según el formato del fin de semana.
    """
    sessions = _ALL_SESSIONS[event_format]
    frames: list[pd.DataFrame] = []

    for stype in sessions:
        try:
            frames.append(build_session_map(season, round_, stype, event_format))
        except Exception as exc:
            log.warning("Sesión %s R%d no disponible: %s", stype, round_, exc)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_season_map(
    season: int,
    rounds: list[tuple[int, EventFormat]],
) -> pd.DataFrame:
    """
    Construye el driver_session_map para TODA la temporada.

    Args:
        rounds: lista de (round_number, event_format) — viene de jolpica_client.get_schedule().
                Usar has_sprint para derivar el format:
                    event_format = "sprint_qualifying" if round_info.has_sprint else "conventional"
    """
    frames: list[pd.DataFrame] = []
    for round_, fmt in rounds:
        log.info("Construyendo mapa ronda %d/%d", round_, len(rounds))
        frame = build_round_map(season, round_, fmt)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_season_map(df: pd.DataFrame, season: int) -> Path:
    """Guarda el mapa de la temporada en data/interim/."""
    out = _INTERIM_DIR / f"driver_session_map_{season}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("driver_session_map guardado: %s (%d filas)", out, len(df))
    return out


def load_season_map(season: int) -> pd.DataFrame | None:
    """Carga el mapa desde parquet si existe. Devuelve None si no hay fichero."""
    path = _INTERIM_DIR / f"driver_session_map_{season}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Nota de implementación: identificadores estables entre temporadas
# ---------------------------------------------------------------------------
# El car_number NO es estable entre temporadas:
#   Verstappen: #33 (2015-2022) → #1 (2023-2025) → #3 (2026)
#   Norris:     #4  (2019-2025) → #1 (2026, tras ganar campeonato 2025)
#
# Para joins entre temporadas (priors 2022-2025 → modelo 2026) usar SIEMPRE
# el driver_id de Jolpica ('max_verstappen', 'norris', ...) que es permanente.
# El car_number solo se usa para la comparación INTRA-sesión de este módulo.
