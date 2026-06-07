"""Carga y valida los YAMLs de config/ usando Pydantic. Punto de entrada único para settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


# ---------------------------------------------------------------------------
# settings.yaml
# ---------------------------------------------------------------------------

class SeasonSettings(BaseModel):
    active: int
    rounds: int
    first_round_date: str


class SimulationSettings(BaseModel):
    n_sims: int = Field(ge=1000)
    seed: int
    chunk_size: int = Field(ge=1)
    car_development_sigma: float = Field(default=0.0, ge=0.0)


class PathSettings(BaseModel):
    data_raw: str
    data_interim: str
    data_processed: str
    snapshots: str
    outputs: str
    fastf1_cache: str
    jolpica_cache: str


class FastF1Settings(BaseModel):
    load_laps: bool
    load_telemetry: bool
    log_level: str


class JolpicaSettings(BaseModel):
    base_url: str
    timeout_s: int
    max_retries: int


class ModelSettings(BaseModel):
    driver_shrinkage: bool
    min_laps_long_run: int
    fuel_correction_kg_per_lap: float
    fuel_lap_time_delta_s: float


class BacktestSettings(BaseModel):
    freeze_on_write: bool
    snapshot_version: str


class Settings(BaseModel):
    season: SeasonSettings
    simulation: SimulationSettings
    paths: PathSettings
    fastf1: FastF1Settings
    jolpica: JolpicaSettings
    model: ModelSettings
    backtest: BacktestSettings


# ---------------------------------------------------------------------------
# scoring.yaml
# ---------------------------------------------------------------------------

class RaceScoring(BaseModel):
    positions: dict[int, int]
    fastest_lap_point: bool

    @model_validator(mode="after")
    def no_fastest_lap(self) -> "RaceScoring":
        if self.fastest_lap_point:
            raise ValueError("fastest_lap_point must be false — eliminated from 2025 onwards")
        return self


class SprintScoring(BaseModel):
    positions: dict[int, int]

    @model_validator(mode="after")
    def max_eight_positions(self) -> "SprintScoring":
        if max(self.positions.keys()) != 8:
            raise ValueError("Sprint scoring must cover exactly positions 1-8")
        return self


class Scoring(BaseModel):
    race: RaceScoring
    sprint: SprintScoring


# ---------------------------------------------------------------------------
# components.yaml
# ---------------------------------------------------------------------------

class ComponentAllocations(BaseModel):
    ICE: int
    TC: int
    EX: int
    MGU_K: int
    ES: int
    CE: int
    MGU_H: int


class GearboxSettings(BaseModel):
    allowed_per_season: int


class PenaltyRules(BaseModel):
    first_extra_element: int
    subsequent_elements: int
    pit_lane_threshold: int


class Components(BaseModel):
    allocations_2026: ComponentAllocations
    allocations_2027: ComponentAllocations
    gearbox: GearboxSettings
    new_supplier_bonus: int
    penalties: PenaltyRules


# ---------------------------------------------------------------------------
# circuits.yaml
# ---------------------------------------------------------------------------

class CircuitMeta(BaseModel):
    name: str
    city: str
    country: str
    # jolpica_id: identificador estable del circuito en Jolpica/Ergast.
    # Usar este campo para joins contra la API; NO usar round_2026 como clave.
    jolpica_id: str
    # round_2026 es un valor cacheado. Si Bahrain/Saudi se reinsertan, Jolpica
    # renumera y este campo queda obsoleto. schedule.py debe derivarlo del API live.
    round_2026: int
    date_2026: str
    weekend_format: Literal["normal", "sprint"]
    power_sensitive: bool
    downforce_level: Literal["low", "medium", "high"]
    street_circuit: bool
    overtaking: Literal["easy", "medium", "hard"]
    tyre_deg: Literal["low", "medium", "high"]


class Circuits(BaseModel):
    circuits: dict[str, CircuitMeta]


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------

def _load_yaml(filename: str) -> dict:
    with open(_CONFIG_DIR / filename) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(**_load_yaml("settings.yaml"))


@lru_cache(maxsize=1)
def get_scoring() -> Scoring:
    return Scoring(**_load_yaml("scoring.yaml"))


@lru_cache(maxsize=1)
def get_components() -> Components:
    return Components(**_load_yaml("components.yaml"))


@lru_cache(maxsize=1)
def get_circuits() -> dict[str, CircuitMeta]:
    raw = _load_yaml("circuits.yaml")
    return {k: CircuitMeta(**v) for k, v in raw["circuits"].items()}
