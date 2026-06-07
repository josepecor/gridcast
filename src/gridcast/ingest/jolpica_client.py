"""
Jolpica-F1 API client.

Responsabilidades: calendario/schedule y standings históricos ÚNICAMENTE.
Frecuencia de actualización: ~una vez por fin de semana (lunes post-carrera).
NO usar para datos de sesión en caliente — usar fastf1_client.py para eso.

El campo jolpica_circuit_id es la clave estable para joins aunque cambien los
round numbers (ej. si Bahrain/Saudi se reinsertan al calendario).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import requests

from gridcast.config import get_settings

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Tipos de retorno
# ---------------------------------------------------------------------------

@dataclass
class RoundInfo:
    season: int
    round: int
    race_name: str
    jolpica_circuit_id: str
    circuit_name: str
    date: date
    # True si la API devuelve clave "Sprint" en el objeto Race
    has_sprint: bool
    # Fechas de cada sesión disponibles en la respuesta (pueden estar vacías)
    session_dates: dict[str, date] = field(default_factory=dict)


@dataclass
class DriverStanding:
    position: int
    driver_id: str
    code: str
    points: float
    wins: int


@dataclass
class ConstructorStanding:
    position: int
    constructor_id: str
    name: str
    points: float
    wins: int


@dataclass
class RaceResult:
    season: int
    round: int
    driver_id: str
    constructor_id: str
    grid: int
    position: int | None        # None si no clasificó (R, W, D...)
    position_text: str          # "1", "R", "W", "D", "N", "F"
    status: str                 # "Finished", "Retired", "+1 Lap", etc.
    points: float
    laps_completed: int


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------

class JolpicaClient:
    # Tiempo máximo de caché por tipo de dato (segundos)
    _SCHEDULE_TTL  = 6 * 3600       # 6h — el calendario puede cambiar
    _STANDINGS_TTL = 2 * 3600       # 2h — actualizan ~lunes post-carrera
    _RESULTS_TTL   = 7 * 24 * 3600  # 7d — resultados de ronda ya disputada son inmutables

    def __init__(self) -> None:
        cfg = get_settings()
        self._base_url   = cfg.jolpica.base_url.rstrip("/")
        self._timeout    = cfg.jolpica.timeout_s
        self._max_retries = cfg.jolpica.max_retries
        self._cache_dir  = _PROJECT_ROOT / cfg.paths.jolpica_cache
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internals: HTTP + caché local JSON
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _is_stale(self, path: Path, ttl_s: float) -> bool:
        if not path.exists():
            return True
        return (time.time() - path.stat().st_mtime) > ttl_s

    def _get(self, endpoint: str, cache_key: str, ttl_s: float) -> dict:
        path = self._cache_path(cache_key)
        if not self._is_stale(path, ttl_s):
            log.debug("jolpica cache hit: %s", cache_key)
            return json.loads(path.read_text())

        url = f"{self._base_url}/{endpoint}"
        log.info("jolpica fetch: %s", url)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                resp = requests.get(url, timeout=self._timeout, params={"limit": 100})
                resp.raise_for_status()
                data = resp.json()
                path.write_text(json.dumps(data, ensure_ascii=False))
                return data
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    log.warning("jolpica retry %d/%d en %ds: %s", attempt + 1, self._max_retries, wait, exc)
                    time.sleep(wait)

        raise RuntimeError(f"Jolpica: {self._max_retries} intentos fallidos para {url}") from last_exc

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def get_schedule(self, season: int) -> list[RoundInfo]:
        """
        Devuelve el calendario actual de la temporada con round numbers de Jolpica.

        SIEMPRE llamar a esto antes de usar round numbers en FastF1 — los rounds
        pueden cambiar si se añaden o cancelan GPs (ej. Bahrain/Saudi 2026).
        """
        data = self._get(f"{season}.json", f"schedule_{season}", self._SCHEDULE_TTL)
        races = data["MRData"]["RaceTable"]["Races"]
        result: list[RoundInfo] = []

        for race in races:
            session_dates: dict[str, date] = {}
            for key in ("FirstPractice", "SecondPractice", "ThirdPractice",
                        "SprintQualifying", "Sprint", "Qualifying"):
                if key in race:
                    session_dates[key] = date.fromisoformat(race[key]["date"])

            result.append(RoundInfo(
                season=season,
                round=int(race["round"]),
                race_name=race["raceName"],
                jolpica_circuit_id=race["Circuit"]["circuitId"],
                circuit_name=race["Circuit"]["circuitName"],
                date=date.fromisoformat(race["date"]),
                has_sprint="Sprint" in race,
                session_dates=session_dates,
            ))

        log.info("jolpica: %d rondas en temporada %d", len(result), season)
        return result

    def get_race_results(self, season: int, round_: int) -> list[RaceResult]:
        """
        Resultados de carrera de una ronda ya disputada.
        Uso: verificación / fallback cuando FastF1 no está disponible.
        Los resultados de rondas pasadas son inmutables → TTL largo.
        """
        data = self._get(
            f"{season}/{round_}/results.json",
            f"race_results_{season}_r{round_}",
            self._RESULTS_TTL,
        )
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return []

        race = races[0]
        result: list[RaceResult] = []
        for entry in race["Results"]:
            pos_text = entry["positionText"]
            position = int(pos_text) if pos_text.isdigit() else None
            result.append(RaceResult(
                season=season,
                round=round_,
                driver_id=entry["Driver"]["driverId"],
                constructor_id=entry["Constructor"]["constructorId"],
                grid=int(entry["grid"]),
                position=position,
                position_text=pos_text,
                status=entry["status"],
                points=float(entry["points"]),
                laps_completed=int(entry["laps"]),
            ))
        return result

    def get_driver_standings(
        self, season: int, after_round: int | None = None
    ) -> list[DriverStanding]:
        """
        Standings de pilotos al final de `after_round` (o al día de hoy si None).
        Uso principal: verificación post-facto de que nuestros puntos calculados coinciden.
        """
        if after_round is not None:
            endpoint  = f"{season}/{after_round}/driverStandings.json"
            cache_key = f"driver_standings_{season}_r{after_round}"
            ttl       = self._RESULTS_TTL
        else:
            endpoint  = f"{season}/driverStandings.json"
            cache_key = f"driver_standings_{season}_latest"
            ttl       = self._STANDINGS_TTL

        data = self._get(endpoint, cache_key, ttl)
        lists_ = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists_:
            return []

        return [
            DriverStanding(
                position=int(e["position"]),
                driver_id=e["Driver"]["driverId"],
                code=e["Driver"].get("code", ""),
                points=float(e["points"]),
                wins=int(e["wins"]),
            )
            for e in lists_[0]["DriverStandings"]
        ]

    def get_constructor_standings(
        self, season: int, after_round: int | None = None
    ) -> list[ConstructorStanding]:
        if after_round is not None:
            endpoint  = f"{season}/{after_round}/constructorStandings.json"
            cache_key = f"constructor_standings_{season}_r{after_round}"
            ttl       = self._RESULTS_TTL
        else:
            endpoint  = f"{season}/constructorStandings.json"
            cache_key = f"constructor_standings_{season}_latest"
            ttl       = self._STANDINGS_TTL

        data = self._get(endpoint, cache_key, ttl)
        lists_ = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists_:
            return []

        return [
            ConstructorStanding(
                position=int(e["position"]),
                constructor_id=e["Constructor"]["constructorId"],
                name=e["Constructor"]["name"],
                points=float(e["points"]),
                wins=int(e["wins"]),
            )
            for e in lists_[0]["ConstructorStandings"]
        ]


# ---------------------------------------------------------------------------
# Singleton + funciones de módulo (interfaz pública para pipelines)
# ---------------------------------------------------------------------------

_client: JolpicaClient | None = None


def _get_client() -> JolpicaClient:
    global _client
    if _client is None:
        _client = JolpicaClient()
    return _client


def get_schedule(season: int) -> list[RoundInfo]:
    return _get_client().get_schedule(season)


def get_race_results(season: int, round_: int) -> list[RaceResult]:
    return _get_client().get_race_results(season, round_)


def get_driver_standings(season: int, after_round: int | None = None) -> list[DriverStanding]:
    return _get_client().get_driver_standings(season, after_round)


def get_constructor_standings(season: int, after_round: int | None = None) -> list[ConstructorStanding]:
    return _get_client().get_constructor_standings(season, after_round)
