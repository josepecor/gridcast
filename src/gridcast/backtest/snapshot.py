"""
Snapshot — estados point-in-time congelados para el backtest walk-forward.

Un snapshot para la ronda N contiene todos los datos disponibles ANTES del corte
de esa ronda. Una vez llamado freeze(), el snapshot es inmutable: ninguna escritura
posterior es posible.

Estructura en disco:
    data/snapshots/{season}/{version}/round_{N:02d}_pre/
        metadata.json       ← versión, cutoff, frames, frozen flag
        race_results.parquet
        driver_session_map.parquet
        pace_laps.parquet
        ...

Reglas (§2.2 CLAUDE.md):
  - Nunca reescribir un snapshot congelado.
  - Si cambia el modelo → nueva version en settings + regenerar todo el histórico.
  - Siempre validar leakage antes de escribir cualquier frame.

Uso típico:

    snap = get_snapshot(2026, round_=5)
    if snap.is_frozen:
        return snap          # ya está completo, no hacer nada

    with guard.for_round(5) as ctx:
        results_df['date'] = cutoff_of_r4
        ctx.check(results_df)
        snap.write_frame("race_results", results_df)

    snap.freeze()
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gridcast.backtest.leakage_guard import LeakageError, LeakageGuard
from gridcast.config import get_settings

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------

class FrozenSnapshotError(RuntimeError):
    """Intento de modificar un snapshot ya congelado."""


class SnapshotLeakageError(LeakageError):
    """Leakage detectado al intentar escribir en un snapshot."""


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class Snapshot:
    """
    Snapshot point-in-time para una ronda concreta.

    No instanciar directamente: usar get_snapshot() o build_snapshot().
    """

    _METADATA_FILE = "metadata.json"

    def __init__(self, path: Path) -> None:
        self._path = path

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def exists(self) -> bool:
        return self._path.exists() and (self._path / self._METADATA_FILE).exists()

    @property
    def is_frozen(self) -> bool:
        if not self.exists:
            return False
        return self._read_metadata().get("frozen", False)

    @property
    def metadata(self) -> dict[str, Any]:
        if not self.exists:
            return {}
        return self._read_metadata()

    @property
    def frames(self) -> list[str]:
        """Nombres de los DataFrames guardados en este snapshot."""
        return self._read_metadata().get("frames", []) if self.exists else []

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------

    def write_frame(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        date_col: str = "date",
        cutoff: date | None = None,
        skip_leakage_check: bool = False,
    ) -> None:
        """
        Escribe un DataFrame en el snapshot como {name}.parquet.

        Args:
            name:               Identificador del frame ('race_results', 'pace_laps', …).
            df:                 DataFrame a guardar.
            date_col:           Columna de fecha para la validación de leakage.
            cutoff:             Fecha de corte. Si None, la infiere del metadata.
            skip_leakage_check: True solo si el DataFrame ya fue validado externamente
                                (debe documentarse el motivo en el código llamador).
        """
        self._assert_not_frozen(name)
        self._path.mkdir(parents=True, exist_ok=True)

        if not skip_leakage_check:
            effective_cutoff = cutoff or self._cutoff_from_metadata()
            if effective_cutoff is None:
                raise ValueError(
                    f"No se puede validar leakage para '{name}': "
                    "pasa cutoff= o llama a init_metadata() primero."
                )
            if date_col not in df.columns:
                raise ValueError(
                    f"Columna de fecha '{date_col}' no encontrada en el frame '{name}'. "
                    f"Columnas disponibles: {list(df.columns)}. "
                    "Añade la columna o usa skip_leakage_check=True con justificación."
                )
            _validate_leakage(df, effective_cutoff, date_col, frame_name=name)

        out = self._path / f"{name}.parquet"
        df.to_parquet(out, index=False)
        log.info("snapshot: frame '%s' escrito (%d filas) → %s", name, len(df), out)

        self._update_metadata({"frames": sorted(set(self.frames + [name]))})

    def init_metadata(
        self,
        season: int,
        round_: int,
        cutoff: date,
        version: str,
    ) -> None:
        """
        Inicializa el metadata del snapshot. Llamar antes de write_frame()
        si no se pasa cutoff= explícito en cada llamada.
        """
        self._assert_not_frozen("metadata")
        self._path.mkdir(parents=True, exist_ok=True)

        existing = self._read_metadata() if self.exists else {}
        meta = {
            **existing,
            "version":      version,
            "season":       season,
            "round":        round_,
            "cutoff_date":  cutoff.isoformat(),
            "frozen":       False,
            "frames":       existing.get("frames", []),
            "created_at":   existing.get("created_at", _now_iso()),
        }
        self._write_metadata(meta)

    # ------------------------------------------------------------------
    # Congelado
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """
        Marca el snapshot como inmutable. Después de esto, write_frame() lanzará
        FrozenSnapshotError. Esta operación es irreversible.
        """
        if not self.exists:
            raise RuntimeError("No se puede congelar un snapshot vacío (sin metadata).")
        if self.is_frozen:
            log.warning("snapshot ya estaba congelado: %s", self._path)
            return

        self._update_metadata({"frozen": True, "frozen_at": _now_iso()})
        log.info("snapshot CONGELADO: %s", self._path)

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def read_frame(self, name: str) -> pd.DataFrame:
        """Lee un DataFrame desde el snapshot."""
        if not self.exists:
            raise FileNotFoundError(f"Snapshot no existe: {self._path}")
        path = self._path / f"{name}.parquet"
        if not path.exists():
            available = self.frames
            raise FileNotFoundError(
                f"Frame '{name}' no encontrado en snapshot {self._path.name}. "
                f"Disponibles: {available}"
            )
        return pd.read_parquet(path)

    def read_all_frames(self) -> dict[str, pd.DataFrame]:
        """Lee todos los DataFrames del snapshot."""
        return {name: self.read_frame(name) for name in self.frames}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_not_frozen(self, operation: str) -> None:
        if self.is_frozen:
            raise FrozenSnapshotError(
                f"Snapshot {self._path.name} está congelado. "
                f"Operación '{operation}' rechazada. "
                "Para regenerar: usa una nueva versión en config/settings.yaml "
                "(backtest.snapshot_version) y vuelve a ejecutar pipelines/03_freeze_history.py."
            )

    def _metadata_path(self) -> Path:
        return self._path / self._METADATA_FILE

    def _read_metadata(self) -> dict[str, Any]:
        try:
            return json.loads(self._metadata_path().read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_metadata(self, meta: dict[str, Any]) -> None:
        self._metadata_path().write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    def _update_metadata(self, updates: dict[str, Any]) -> None:
        meta = self._read_metadata()
        meta.update(updates)
        self._write_metadata(meta)

    def _cutoff_from_metadata(self) -> date | None:
        raw = self._read_metadata().get("cutoff_date")
        if raw:
            return date.fromisoformat(raw)
        return None

    def __repr__(self) -> str:
        status = "frozen" if self.is_frozen else "open"
        return f"Snapshot({self._path.name}, {status}, frames={self.frames})"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _snapshot_root(season: int, version: str) -> Path:
    cfg = get_settings()
    return _PROJECT_ROOT / cfg.paths.snapshots / str(season) / version


def _snapshot_dir(season: int, round_: int, version: str) -> Path:
    return _snapshot_root(season, version) / f"round_{round_:02d}_pre"


def get_snapshot(
    season: int,
    round_: int,
    version: str | None = None,
) -> Snapshot:
    """
    Devuelve el objeto Snapshot para (season, round_).
    Si no existe en disco, devuelve un Snapshot vacío (sin crear nada todavía).
    """
    v = version or get_settings().backtest.snapshot_version
    return Snapshot(_snapshot_dir(season, round_, v))


def build_snapshot(
    season: int,
    round_: int,
    frames: dict[str, pd.DataFrame],
    guard: LeakageGuard,
    date_col: str = "date",
    version: str | None = None,
) -> Snapshot:
    """
    Crea, valida y congela un snapshot en una sola operación.

    Args:
        season:   Temporada.
        round_:   Número de ronda a predecir (el snapshot contiene datos de rounds < N).
        frames:   Dict {nombre: DataFrame}. Cada DataFrame debe tener date_col.
        guard:    LeakageGuard construido con el calendario de la temporada.
        date_col: Columna de fecha usada para la validación de leakage.
        version:  Versión del snapshot (default: settings.backtest.snapshot_version).

    Returns:
        Snapshot congelado.

    Raises:
        FrozenSnapshotError: si el snapshot ya existe y está congelado.
        SnapshotLeakageError: si algún frame contiene datos posteriores al corte.
    """
    snap = get_snapshot(season, round_, version)

    if snap.is_frozen:
        log.info(
            "snapshot R%02d ya congelado (%s) — saltando",
            round_,
            snap.path.name,
        )
        return snap

    cutoff = guard.round_cutoff(season, round_)
    snap.init_metadata(
        season=season,
        round_=round_,
        cutoff=cutoff,
        version=version or get_settings().backtest.snapshot_version,
    )

    for name, df in frames.items():
        snap.write_frame(name, df, date_col=date_col, cutoff=cutoff)

    snap.freeze()
    return snap


def list_snapshots(
    season: int,
    version: str | None = None,
) -> list[int]:
    """Devuelve los números de ronda con snapshot disponible (congelado o no)."""
    v = version or get_settings().backtest.snapshot_version
    root = _snapshot_root(season, v)
    if not root.exists():
        return []
    rounds = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("round_") and d.name.endswith("_pre"):
            try:
                rounds.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return rounds


def list_frozen_snapshots(season: int, version: str | None = None) -> list[int]:
    """Devuelve solo los snapshots ya congelados."""
    v = version or get_settings().backtest.snapshot_version
    return [
        r for r in list_snapshots(season, v)
        if get_snapshot(season, r, v).is_frozen
    ]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _validate_leakage(
    df: pd.DataFrame,
    cutoff: date,
    date_col: str,
    frame_name: str,
) -> None:
    """Lanza SnapshotLeakageError si el df contiene datos >= cutoff."""
    dates = df[date_col]

    def to_date(v: Any) -> date | None:
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        if isinstance(v, pd.Timestamp):
            return v.date()
        if isinstance(v, str):
            try:
                return date.fromisoformat(v[:10])
            except ValueError:
                return None
        return None

    if pd.api.types.is_datetime64_any_dtype(dates):
        date_series = dates.dt.date
    else:
        date_series = dates.apply(to_date)

    mask = date_series.apply(lambda d: d is not None and d >= cutoff)
    if mask.any():
        bad_dates = date_series[mask].dropna().unique()
        msg = (
            f"SnapshotLeakageError en frame '{frame_name}': "
            f"{mask.sum()} fila(s) con fecha >= corte ({cutoff}). "
            f"Fechas violadoras: {sorted(str(d) for d in bad_dates)}"
        )
        log.error(msg)
        raise SnapshotLeakageError(msg)
