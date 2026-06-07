"""
FastF1 client — fuente primaria de todos los datos de sesión.

FastF1 lee directamente livetiming.formula1.com (API oficial de F1),
disponible minutos/horas después de cada sesión. Independiente de Jolpica.

Flujo correcto de round numbers:
  1. jolpica_client.get_schedule(season)  → round numbers actuales
  2. fastf1_client.get_session(season, round, session_type)  → datos de sesión

Sesiones por formato de fin de semana:
  Normal:  FP1  FP2  FP3  Q   R       (FP2 = sesión de pace primaria)
  Sprint:  FP1  SQ   S    Q   R       (FP1 = ÚNICA práctica; NO hay FP2/FP3)

Rookies en FP1: SOLO en formato normal. En sprint FP1 todos son titulares.
Usar get_pace_laps() en lugar de get_laps('FP2') directamente: selecciona
la sesión correcta según el formato del fin de semana.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import fastf1
import pandas as pd

from gridcast.config import get_settings

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

SessionType  = Literal["FP1", "FP2", "FP3", "SQ", "S", "Q", "R"]
EventFormat  = Literal["conventional", "sprint_qualifying"]

# Sesiones de práctica disponibles por formato. Pedir FP2/FP3 en sprint lanza error en FastF1.
PRACTICE_SESSIONS: dict[EventFormat, list[SessionType]] = {
    "conventional":      ["FP1", "FP2", "FP3"],
    "sprint_qualifying": ["FP1"],
}

# Sesión de pace primaria por formato (ver §3.3 CLAUDE.md)
PACE_SESSION: dict[EventFormat, SessionType] = {
    "conventional":      "FP2",   # sim quali + sim carrera; predictor más fuerte
    "sprint_qualifying": "FP1",   # única práctica disponible
}

# En sprint FP1 no hay rookies: los equipos necesitan al titular para preparar el coche
def rookie_check_applies(event_format: EventFormat, session_type: SessionType) -> bool:
    """True si la sesión puede contener pilotos no titulares (rookies FP1 en semanas normales)."""
    return event_format == "conventional" and session_type == "FP1"

# FastF1 acepta strings cortos pero internamente los mapea a nombres largos.
# Mantenemos un mapa explícito para evitar sorpresas entre versiones.
_SESSION_IDENTIFIER: dict[SessionType, str] = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "SQ":  "Sprint Qualifying",
    "S":   "Sprint",
    "Q":   "Qualifying",
    "R":   "Race",
}

# Columnas que siempre seleccionamos de session.laps para mantener DataFrames manejables
_LAP_COLUMNS = [
    "Driver", "DriverNumber", "Team",
    "LapNumber", "LapTime", "Stint",
    "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "Compound", "TyreLife",
    "PitInTime", "PitOutTime",
    "TrackStatus", "IsAccurate",
    "Position",
]

# Columnas que siempre seleccionamos de session.results
_RESULT_COLUMNS = [
    "DriverNumber", "Abbreviation", "FullName", "TeamName",
    "GridPosition", "Position", "ClassifiedPosition",
    "Status", "Points", "Time",
]


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------

class FastF1Client:
    _cache_enabled: bool = False

    def __init__(self) -> None:
        cfg = get_settings()
        self._cache_path = _PROJECT_ROOT / cfg.paths.fastf1_cache
        self._load_telemetry_default = cfg.fastf1.load_telemetry
        self._log_level = cfg.fastf1.log_level
        self._ensure_cache()

    def _ensure_cache(self) -> None:
        if FastF1Client._cache_enabled:
            return
        self._cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self._cache_path))
        # Silencia el logger interno de FastF1 según la config
        logging.getLogger("fastf1").setLevel(self._log_level)
        FastF1Client._cache_enabled = True
        log.info("fastf1 cache habilitado en %s", self._cache_path)

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def get_session(
        self,
        season: int,
        round_: int,
        session_type: SessionType,
        *,
        load_laps: bool = True,
        load_telemetry: bool | None = None,
        load_weather: bool = False,
        load_messages: bool = False,
    ) -> fastf1.core.Session:
        """
        Carga y devuelve una sesión de FastF1.

        `round_` debe venir de jolpica_client.get_schedule() — no usar valores
        cacheados de circuits.yaml porque pueden desactualizarse si cambia el calendario.

        Defaults: laps=True, telemetry=False (pesado en disco), weather=False.
        """
        identifier = _SESSION_IDENTIFIER[session_type]
        use_telemetry = self._load_telemetry_default if load_telemetry is None else load_telemetry

        log.info("fastf1 load: season=%d round=%d session=%s", season, round_, session_type)
        session = fastf1.get_session(season, round_, identifier)
        session.load(
            laps=load_laps,
            telemetry=use_telemetry,
            weather=load_weather,
            messages=load_messages,
        )
        return session

    def get_laps(
        self,
        season: int,
        round_: int,
        session_type: SessionType,
        *,
        accurate_only: bool = True,
    ) -> pd.DataFrame:
        """
        Devuelve el DataFrame de vueltas de una sesión, con columnas relevantes.

        accurate_only=True filtra vueltas marcadas como no representativas por FastF1
        (entrada/salida de pit, safety car, etc.).
        """
        session = self.get_session(season, round_, session_type, load_laps=True)
        laps = session.laps

        if accurate_only:
            laps = laps[laps["IsAccurate"]]

        available = [c for c in _LAP_COLUMNS if c in laps.columns]
        laps = laps[available].copy()

        # Añadimos metadatos de sesión para facilitar joins downstream
        laps["season"]       = season
        laps["round"]        = round_
        laps["session_type"] = session_type

        return laps.reset_index(drop=True)

    def get_results(
        self,
        season: int,
        round_: int,
        session_type: SessionType,
    ) -> pd.DataFrame:
        """
        Devuelve el DataFrame de resultados de sesión (posiciones, status, puntos).
        Para clasificación incluye tiempos Q1/Q2/Q3.
        """
        session = self.get_session(season, round_, session_type, load_laps=False)
        results = session.results

        available = [c for c in _RESULT_COLUMNS if c in results.columns]
        results = results[available].copy()

        results["season"]       = season
        results["round"]        = round_
        results["session_type"] = session_type

        return results.reset_index(drop=True)

    def get_event_schedule(self, season: int) -> pd.DataFrame:
        """
        Devuelve el EventSchedule de FastF1 para la temporada.
        Útil para verificar qué sesiones existen antes de intentar cargarlas,
        y para confirmar que los round numbers de FastF1 coinciden con Jolpica.
        """
        log.info("fastf1 event schedule: season=%d", season)
        return fastf1.get_event_schedule(season, include_testing=False)

    def get_testing_sessions_info(self, season: int) -> list[dict]:
        """
        Devuelve información sobre los tests de pretemporada disponibles.
        Prueba test_number 1..4 y day 1..3; para cuando FastF1 deje de
        encontrar un test (excepción) corta la búsqueda.
        """
        self._ensure_cache()
        results = []
        for test_num in range(1, 5):
            found_test = False
            for day in range(1, 4):
                try:
                    s = fastf1.get_testing_session(season, test_num, day)
                    results.append({
                        "test_number": test_num,
                        "day": day,
                        "location": s.event.get("Location", ""),
                        "date": s.date.date(),
                    })
                    found_test = True
                except Exception:
                    break   # no más días en este test
            if not found_test:
                break       # no más tests
        return results

    def get_testing_laps(
        self,
        season: int,
        test_number: int,
        day: int,
        *,
        accurate_only: bool = True,
    ) -> pd.DataFrame:
        """
        Devuelve vueltas de una sesión de test de pretemporada.

        accurate_only=True filtra vueltas no representativas.
        El DataFrame incluye columnas season/test_number/day para joins downstream.
        """
        self._ensure_cache()
        log.info("fastf1 testing load: season=%d test=%d day=%d", season, test_number, day)
        session = fastf1.get_testing_session(season, test_number, day)
        session.load(laps=True, telemetry=False, weather=False, messages=False)

        laps = session.laps
        if accurate_only:
            laps = laps[laps["IsAccurate"]]

        available = [c for c in _LAP_COLUMNS if c in laps.columns]
        laps = laps[available].copy()

        laps["season"]      = season
        laps["test_number"] = test_number
        laps["day"]         = day
        laps["location"]    = session.event.get("Location", "")

        return laps.reset_index(drop=True)

    def get_pace_laps(
        self,
        season: int,
        round_: int,
        event_format: EventFormat,
        *,
        accurate_only: bool = True,
    ) -> pd.DataFrame:
        """
        Devuelve vueltas de la sesión de práctica primaria para estimar pace.

        Selecciona automáticamente FP2 (normal) o FP1 (sprint), evitando pedir
        una sesión que no existe en el formato del fin de semana.
        El campo 'session_type' en el DataFrame indica qué sesión se cargó.
        """
        session_type = PACE_SESSION[event_format]
        return self.get_laps(season, round_, session_type, accurate_only=accurate_only)

    def get_drivers_in_session(
        self,
        season: int,
        round_: int,
        session_type: SessionType,
    ) -> pd.DataFrame:
        """
        Devuelve los pilotos que condujeron realmente en la sesión.
        Usado por driver_session_map.py para detectar rookies en FP1.

        Columnas: DriverNumber, Abbreviation, FullName, TeamName
        """
        session = self.get_session(season, round_, session_type, load_laps=False)
        results = session.results
        cols = [c for c in ["DriverNumber", "Abbreviation", "FullName", "TeamName"]
                if c in results.columns]
        df = results[cols].copy()
        df["season"]       = season
        df["round"]        = round_
        df["session_type"] = session_type
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Singleton + funciones de módulo (interfaz pública para pipelines)
# ---------------------------------------------------------------------------

_client: FastF1Client | None = None


def _get_client() -> FastF1Client:
    global _client
    if _client is None:
        _client = FastF1Client()
    return _client


def get_session(
    season: int,
    round_: int,
    session_type: SessionType,
    *,
    load_laps: bool = True,
    load_telemetry: bool | None = None,
    load_weather: bool = False,
) -> fastf1.core.Session:
    return _get_client().get_session(
        season, round_, session_type,
        load_laps=load_laps,
        load_telemetry=load_telemetry,
        load_weather=load_weather,
    )


def get_laps(
    season: int,
    round_: int,
    session_type: SessionType,
    *,
    accurate_only: bool = True,
) -> pd.DataFrame:
    return _get_client().get_laps(season, round_, session_type, accurate_only=accurate_only)


def get_results(
    season: int,
    round_: int,
    session_type: SessionType,
) -> pd.DataFrame:
    return _get_client().get_results(season, round_, session_type)


def get_event_schedule(season: int) -> pd.DataFrame:
    return _get_client().get_event_schedule(season)


def get_pace_laps(
    season: int,
    round_: int,
    event_format: EventFormat,
    *,
    accurate_only: bool = True,
) -> pd.DataFrame:
    """FP2 en normal, FP1 en sprint. Nunca pide una sesión inexistente."""
    return _get_client().get_pace_laps(season, round_, event_format, accurate_only=accurate_only)


def get_drivers_in_session(
    season: int,
    round_: int,
    session_type: SessionType,
) -> pd.DataFrame:
    return _get_client().get_drivers_in_session(season, round_, session_type)


def get_qualifying_results(season: int, round_: int) -> pd.DataFrame:
    """
    Retorna los resultados de clasificación con tiempos y posición.

    Columnas de salida:
        abbreviation, team_name, quali_pos (1=pole), best_lap_s (NaN si no completó)
        round, season

    Usa Q3 → Q2 → Q1 para determinar el mejor tiempo de cada piloto.
    """
    client = _get_client()
    session = client.get_session(season, round_, "Q", load_laps=False)
    results = session.results.copy()

    def _best_lap_s(row: pd.Series) -> float | None:
        for q in ("Q3", "Q2", "Q1"):
            v = row.get(q)
            if v is not None and hasattr(v, "total_seconds"):
                secs = v.total_seconds()
                if secs > 0:
                    return round(secs, 4)
        return None

    results["best_lap_s"] = results.apply(_best_lap_s, axis=1)
    results["quali_pos"]  = pd.to_numeric(results["Position"], errors="coerce")
    results["round"]      = round_
    results["season"]     = season

    keep = ["Abbreviation", "TeamName", "quali_pos", "best_lap_s", "round", "season"]
    out = results[[c for c in keep if c in results.columns]].rename(
        columns={"Abbreviation": "abbreviation", "TeamName": "team_name"}
    )
    return out.reset_index(drop=True)


def get_testing_sessions_info(season: int) -> list[dict]:
    return _get_client().get_testing_sessions_info(season)


def get_testing_laps(
    season: int,
    test_number: int,
    day: int,
    *,
    accurate_only: bool = True,
) -> pd.DataFrame:
    return _get_client().get_testing_laps(season, test_number, day, accurate_only=accurate_only)
