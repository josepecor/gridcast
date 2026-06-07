"""
Replay — walk-forward sobre rondas ya disputadas para backtesting point-in-time.

Para cada ronda N disputada:
  1. Lee el snapshot congelado (todos los datos disponibles antes de la FP1 de N).
  2. Pasa el snapshot al modelo de predicción.
  3. Guarda la predicción como artefacto inmutable en outputs/predictions/.

Separación de responsabilidades:
  - Este módulo NO construye snapshots — eso es pipelines/03_freeze_history.py.
  - Este módulo NO evalúa predicciones — eso es evaluate/.
  - El modelo NO toca datos fuera del snapshot — el leakage ya fue validado.

Reglas (§2.2 CLAUDE.md):
  - Las predicciones congeladas son inmutables: representan lo que el modelo
    habría producido con la información disponible en el corte de cada ronda.
  - Nunca reescribir una predicción congelada. Si cambia el modelo → nueva
    versión en settings.backtest.snapshot_version y re-generar todo.

Uso típico (pipeline 04/05):

    schedule = jolpica_client.get_schedule(2026)
    guard = LeakageGuard(schedule)
    model = MyRaceModel()

    replay = WalkForwardReplay(2026, guard, model)
    completed = [r.round for r in schedule if r.date < date.today()]

    predictions = replay.run_all(completed)
    # → lista de RoundPrediction congeladas; rondas ya congeladas se cargan de disco.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from gridcast.backtest.leakage_guard import LeakageGuard
from gridcast.backtest.snapshot import Snapshot, get_snapshot
from gridcast.config import get_settings

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_PREDICTIONS_SUBDIR = "predictions"

REQUIRED_FINISH_PROBS_COLS: frozenset[str] = frozenset({"driver_id"})


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------

class FrozenPredictionError(RuntimeError):
    """Intento de sobrescribir una predicción ya congelada."""


class SnapshotNotFrozenError(RuntimeError):
    """El snapshot de la ronda requerida no está congelado todavía."""


# ---------------------------------------------------------------------------
# Estado del walk-forward por ronda
# ---------------------------------------------------------------------------

class ReplayStatus(enum.Enum):
    SNAPSHOT_MISSING   = "snapshot_missing"    # sin snapshot congelado
    PREDICTION_MISSING = "prediction_missing"  # snapshot OK, predicción pendiente
    PREDICTION_FROZEN  = "prediction_frozen"   # predicción generada y congelada


# ---------------------------------------------------------------------------
# Protocolo del modelo de predicción
# ---------------------------------------------------------------------------

@runtime_checkable
class PredictionModel(Protocol):
    """
    Interfaz que todo modelo de predicción debe implementar.

    El modelo recibe el snapshot congelado de los datos disponibles antes
    de la ronda N y devuelve una distribución de probabilidad sobre el orden.

    Contrato mínimo de la salida (finish_probs):
      - Columna obligatoria: 'driver_id' (str, identificador Jolpica estable).
      - Las columnas adicionales dependen del modelo.

    Formato canónico recomendado (Plackett-Luce, §4.3 CLAUDE.md):
        driver_id  finish_pos  probability
        VER        1           0.35
        VER        2           0.20
        ...
      (N_pilotos × N_posiciones filas; probability suma ≈ 1 por driver y por posición)
    """

    @property
    def version(self) -> str:
        """Identificador de versión del modelo (p.ej. 'baseline_mixedlm_v1')."""
        ...

    def predict(
        self,
        snapshot: Snapshot,
        round_: int,
        guard: LeakageGuard,
    ) -> pd.DataFrame:
        """
        Genera predicciones para la ronda `round_`.

        Args:
            snapshot: Estado congelado de los datos hasta el corte de round_.
            round_:   Número de ronda a predecir.
            guard:    LeakageGuard por si el modelo necesita verificar operaciones
                      adicionales (los datos del snapshot ya están validados).

        Returns:
            DataFrame con al menos la columna 'driver_id'.
        """
        ...


# ---------------------------------------------------------------------------
# RoundPrediction — predicción inmutable para una ronda
# ---------------------------------------------------------------------------

@dataclass
class RoundPrediction:
    """
    Predicción congelada para una ronda concreta.

    Attributes:
        season:        Temporada.
        round_:        Número de ronda predicha.
        cutoff_date:   Fecha de corte (FP1 de la ronda — solo datos anteriores).
        version:       Versión del sistema (snapshots + modelo).
        finish_probs:  DataFrame con la distribución de probabilidad.
        model_version: Versión del modelo concreto que generó esta predicción.
        generated_at:  ISO timestamp UTC de cuándo se generó.
    """

    season:        int
    round_:        int
    cutoff_date:   date
    version:       str
    finish_probs:  pd.DataFrame
    cutoff_name:   str = "pre-fp1"
    model_version: str = "unknown"
    generated_at:  str = field(default_factory=lambda: _now_iso())

    def validate(self) -> None:
        """Verifica invariantes mínimas. Llamado automáticamente por save_prediction()."""
        missing = REQUIRED_FINISH_PROBS_COLS - set(self.finish_probs.columns)
        if missing:
            raise ValueError(
                f"finish_probs le faltan columnas requeridas: {missing}. "
                f"Columnas presentes: {list(self.finish_probs.columns)}"
            )
        if self.finish_probs.empty:
            raise ValueError(
                f"finish_probs para R{self.round_:02d} está vacío. "
                "El modelo debe devolver al menos un piloto."
            )


# ---------------------------------------------------------------------------
# Almacenamiento de predicciones en disco
# ---------------------------------------------------------------------------

def _cutoff_to_suffix(cutoff: str) -> str:
    """Mapea nombre de cutoff a sufijo de directorio. 'pre-fp1' → 'pre' (backwards compat)."""
    return "pre" if cutoff == "pre-fp1" else cutoff


def _prediction_path(
    season: int, round_: int, version: str, cutoff: str = "pre-fp1"
) -> Path:
    cfg = get_settings()
    return (
        _PROJECT_ROOT
        / cfg.paths.outputs
        / _PREDICTIONS_SUBDIR
        / str(season)
        / version
        / f"round_{round_:02d}_{_cutoff_to_suffix(cutoff)}"
    )


def prediction_exists(
    season: int, round_: int, version: str, cutoff: str = "pre-fp1"
) -> bool:
    """True si la predicción para esta ronda y cutoff ya está guardada en disco."""
    return (_prediction_path(season, round_, version, cutoff) / "metadata.json").exists()


def save_prediction(pred: RoundPrediction, *, overwrite: bool = False) -> Path:
    """
    Guarda una predicción en disco como artefacto inmutable.

    Args:
        pred:      Predicción a guardar.
        overwrite: Si True, permite sobrescribir (solo para re-generación con nueva versión).

    Returns:
        Path del directorio donde se guardó.

    Raises:
        FrozenPredictionError: si la predicción ya existe y overwrite=False.
        ValueError:            si la predicción no pasa la validación estructural.
    """
    pred.validate()

    cutoff = getattr(pred, "cutoff_name", "pre-fp1")
    path = _prediction_path(pred.season, pred.round_, pred.version, cutoff)
    meta_path = path / "metadata.json"

    if meta_path.exists() and not overwrite:
        raise FrozenPredictionError(
            f"Predicción para R{pred.round_:02d}/{cutoff} (versión '{pred.version}') ya existe en {path}. "
            "Para regenerar: usa overwrite=True o una nueva versión del sistema "
            "(settings.backtest.snapshot_version)."
        )

    path.mkdir(parents=True, exist_ok=True)
    pred.finish_probs.to_parquet(path / "finish_probs.parquet", index=False)

    meta_path.write_text(
        json.dumps(
            {
                "season":        pred.season,
                "round":         pred.round_,
                "cutoff_name":   cutoff,
                "cutoff_date":   pred.cutoff_date.isoformat(),
                "version":       pred.version,
                "model_version": pred.model_version,
                "generated_at":  pred.generated_at,
                "frozen":        True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    log.info(
        "Predicción R%02d congelada → %s (%d pilotos en finish_probs)",
        pred.round_,
        path,
        pred.finish_probs["driver_id"].nunique(),
    )
    return path


def load_prediction(
    season: int, round_: int, version: str, cutoff: str = "pre-fp1"
) -> RoundPrediction | None:
    """
    Carga una predicción congelada desde disco.

    Returns:
        RoundPrediction si existe y es legible, None en caso contrario.
    """
    path = _prediction_path(season, round_, version, cutoff)
    meta_path = path / "metadata.json"

    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("No se pudo leer metadata de R%02d: %s", round_, exc)
        return None

    fp_path = path / "finish_probs.parquet"
    if not fp_path.exists():
        log.warning(
            "metadata.json existe pero no finish_probs.parquet en %s — predicción incompleta",
            path,
        )
        return None

    try:
        finish_probs = pd.read_parquet(fp_path)
    except Exception as exc:
        log.warning("No se pudo leer finish_probs.parquet de R%02d: %s", round_, exc)
        return None

    return RoundPrediction(
        season=meta["season"],
        round_=meta["round"],
        cutoff_date=date.fromisoformat(meta["cutoff_date"]),
        version=meta["version"],
        finish_probs=finish_probs,
        cutoff_name=meta.get("cutoff_name", "pre-fp1"),
        model_version=meta.get("model_version", "unknown"),
        generated_at=meta.get("generated_at", ""),
    )


def list_predictions(
    season: int, version: str, cutoff: str = "pre-fp1"
) -> list[int]:
    """Devuelve los números de ronda con predicción disponible para (season, version, cutoff)."""
    cfg = get_settings()
    root = (
        _PROJECT_ROOT
        / cfg.paths.outputs
        / _PREDICTIONS_SUBDIR
        / str(season)
        / version
    )
    if not root.exists():
        return []
    suffix = f"_{_cutoff_to_suffix(cutoff)}"
    rounds = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("round_") and d.name.endswith(suffix):
            try:
                rounds.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return rounds


# ---------------------------------------------------------------------------
# WalkForwardReplay — orquestador del backtest walk-forward
# ---------------------------------------------------------------------------

class WalkForwardReplay:
    """
    Orquesta el backtest walk-forward: para cada ronda, lee el snapshot
    congelado, pasa los datos al modelo y guarda la predicción.

    Requisito previo: los snapshots de las rondas a procesar deben estar
    congelados (pipelines/03_freeze_history.py).

    Args:
        season:    Temporada.
        guard:     LeakageGuard construido con el calendario de la temporada.
        model:     Modelo de predicción (implementa PredictionModel Protocol).
        version:   Versión del sistema. Default: settings.backtest.snapshot_version.
        fail_fast: Si True (default), cualquier error en una ronda aborta el replay.
    """

    def __init__(
        self,
        season: int,
        guard: LeakageGuard,
        model: PredictionModel,
        *,
        version: str | None = None,
        fail_fast: bool = True,
    ) -> None:
        self.season    = season
        self.guard     = guard
        self.model     = model
        self.version   = version or get_settings().backtest.snapshot_version
        self.fail_fast = fail_fast

    # ------------------------------------------------------------------
    # Consulta de estado
    # ------------------------------------------------------------------

    def status(self, round_: int) -> ReplayStatus:
        """Estado actual del walk-forward para una ronda concreta."""
        snap = get_snapshot(self.season, round_, self.version)
        if not snap.is_frozen:
            return ReplayStatus.SNAPSHOT_MISSING
        if prediction_exists(self.season, round_, self.version):
            return ReplayStatus.PREDICTION_FROZEN
        return ReplayStatus.PREDICTION_MISSING

    def status_all(self, rounds: list[int]) -> dict[int, ReplayStatus]:
        """Estado de todas las rondas especificadas, ordenadas."""
        return {r: self.status(r) for r in sorted(rounds)}

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def run_round(self, round_: int, *, force: bool = False) -> RoundPrediction:
        """
        Genera y congela la predicción para una sola ronda.

        Si la predicción ya existe y force=False, la carga desde disco sin
        re-ejecutar el modelo.

        Args:
            round_: Número de ronda a predecir.
            force:  Si True, regenera aunque la predicción ya exista.

        Returns:
            RoundPrediction congelada.

        Raises:
            SnapshotNotFrozenError: si el snapshot de la ronda no está listo.
        """
        snap = get_snapshot(self.season, round_, self.version)

        if not snap.is_frozen:
            raise SnapshotNotFrozenError(
                f"Snapshot para R{round_:02d} (temporada {self.season}, versión '{self.version}') "
                "no está congelado. Congela el snapshot antes de llamar al replay."
            )

        if not force and prediction_exists(self.season, round_, self.version):
            pred = load_prediction(self.season, round_, self.version)
            if pred is not None:
                log.info("R%02d: predicción ya congelada — cargando desde disco", round_)
                return pred
            log.warning(
                "R%02d: metadata existe pero predicción corrupta — regenerando", round_
            )

        cutoff = self.guard.round_cutoff(self.season, round_)
        log.info(
            "R%02d: generando predicción (corte: %s, modelo: %s)",
            round_,
            cutoff,
            self.model.version,
        )

        finish_probs = self.model.predict(snap, round_, self.guard)

        pred = RoundPrediction(
            season=self.season,
            round_=round_,
            cutoff_date=cutoff,
            version=self.version,
            finish_probs=finish_probs,
            model_version=self.model.version,
        )
        save_prediction(pred, overwrite=force)
        return pred

    def run_all(
        self,
        rounds: list[int],
        *,
        skip_frozen: bool = True,
        force: bool = False,
    ) -> list[RoundPrediction]:
        """
        Walk-forward sobre un conjunto de rondas, en orden creciente.

        Args:
            rounds:      Lista de rondas a procesar.
            skip_frozen: Si True (default), carga predicciones ya congeladas
                         sin re-ejecutar el modelo.
            force:       Si True, regenera todas aunque existan.

        Returns:
            Lista de RoundPrediction en orden de ronda.
            Con fail_fast=False, las rondas con error se omiten del resultado.
        """
        results:  list[RoundPrediction]  = []
        errors:   dict[int, Exception]   = {}

        for r in sorted(rounds):
            if skip_frozen and not force and prediction_exists(self.season, r, self.version):
                log.info("R%02d: predicción congelada — skip", r)
                pred = load_prediction(self.season, r, self.version)
                if pred is not None:
                    results.append(pred)
                continue

            try:
                results.append(self.run_round(r, force=force))
            except Exception as exc:
                log.error("Error en R%02d: %s", r, exc)
                if self.fail_fast:
                    raise
                errors[r] = exc

        if errors:
            log.warning(
                "Walk-forward terminado con %d error(es) en rondas: %s",
                len(errors),
                sorted(errors.keys()),
            )

        return results

    # ------------------------------------------------------------------
    # Acceso a predicciones guardadas
    # ------------------------------------------------------------------

    def load_prediction(self, round_: int) -> RoundPrediction | None:
        """Carga la predicción congelada de una ronda, o None si no existe."""
        return load_prediction(self.season, round_, self.version)

    def list_predictions(self) -> list[int]:
        """Rondas con predicción disponible para esta temporada y versión."""
        return list_predictions(self.season, self.version)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
