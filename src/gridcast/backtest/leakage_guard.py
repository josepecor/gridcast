"""
LeakageGuard — point-in-time correctness para el backtest walk-forward.

Regla absoluta (§2.2 CLAUDE.md):
  Para predecir la ronda N, solo se pueden usar datos con fecha < corte(N).
  El corte es el inicio del primer día del fin de semana de la ronda N (FP1).

Lanza LeakageError (subclase de AssertionError) si se detecta violación.
Nunca silencia: un falso positivo es mejor que leakage silencioso.

Uso típico en un pipeline:

    from gridcast.backtest.leakage_guard import LeakageGuard

    schedule = get_schedule(2026)
    guard = LeakageGuard(schedule)

    cutoff = guard.round_cutoff(season=2026, round_=5)

    laps = get_laps(2026, 3, 'FP2')          # datos de R3
    laps['date'] = date(2026, 3, 29)           # fecha de Suzuka
    guard.check_frame(laps, cutoff)            # OK — R3 < R5

    laps_bad = get_laps(2026, 5, 'FP2')       # ← leakage!
    laps_bad['date'] = date(2026, 5, 23)
    guard.check_frame(laps_bad, cutoff)        # LeakageError ✗
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Generator

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Excepción
# ---------------------------------------------------------------------------

class LeakageError(AssertionError):
    """
    Datos con fecha >= corte detectados en el conjunto de features.
    Subclase de AssertionError para que sea capturable con `assert` semántico.
    """


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _to_date(value: object) -> date | None:
    """Normaliza un valor a datetime.date. Devuelve None si no es convertible."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _series_to_dates(series: pd.Series) -> pd.Series:
    """Convierte una serie a objetos date, NaT → None."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.date
    return series.apply(_to_date)


# ---------------------------------------------------------------------------
# LeakageGuard
# ---------------------------------------------------------------------------

class LeakageGuard:
    """
    Valida point-in-time correctness usando el calendario de Jolpica.

    Args:
        schedule: lista de RoundInfo devuelta por jolpica_client.get_schedule().
    """

    def __init__(self, schedule: list) -> None:
        # Indexado por round para O(1) lookup
        self._by_round: dict[int, object] = {r.round: r for r in schedule}
        self._season: int = schedule[0].season if schedule else 0

    # ------------------------------------------------------------------
    # Cálculo del corte
    # ------------------------------------------------------------------

    def round_cutoff(self, season: int, round_: int) -> date:
        """
        Fecha de corte para la ronda N: primer día del fin de semana (FP1).
        Ningún dato con fecha >= este valor puede usarse para predecir la ronda N.

        Fallback: si FP1 no está en session_dates, usa la fecha de carrera - 2 días
        (mínimo fin de semana normal: viernes práctica, sábado quali, domingo carrera).
        """
        round_info = self._by_round.get(round_)
        if round_info is None:
            raise ValueError(f"Ronda {round_} no encontrada en el calendario {season}")

        fp1_date = round_info.session_dates.get("FirstPractice")
        if fp1_date is not None:
            return fp1_date

        # Fallback conservador
        log.warning(
            "FP1 date no disponible para R%d — usando race_date - 2 días como corte",
            round_,
        )
        return round_info.date - timedelta(days=2)

    def cutoffs_for_season(self) -> dict[int, date]:
        """Devuelve {round: cutoff_date} para todas las rondas del calendario."""
        return {r: self.round_cutoff(self._season, r) for r in self._by_round}

    # ------------------------------------------------------------------
    # Verificación de DataFrames
    # ------------------------------------------------------------------

    def check_frame(
        self,
        df: pd.DataFrame,
        cutoff: date,
        date_col: str = "date",
        context: str = "",
    ) -> None:
        """
        Verifica que todas las filas de df tienen date_col < cutoff.
        Lanza LeakageError describiendo las filas violadoras.

        Args:
            df:       DataFrame con columna de fecha.
            cutoff:   fecha de corte (exclusiva — los datos deben ser estrictamente anteriores).
            date_col: nombre de la columna de fecha.
            context:  string opcional para mensajes de error más descriptivos.
        """
        if df.empty:
            return

        if date_col not in df.columns:
            raise ValueError(
                f"Columna '{date_col}' no encontrada. Columnas disponibles: {list(df.columns)}"
            )

        dates = _series_to_dates(df[date_col])
        mask = dates.apply(lambda d: d is not None and d >= cutoff)

        if not mask.any():
            return

        violators = df[mask][[date_col]].copy()
        violators["_date_parsed"] = dates[mask]

        prefix = f"[{context}] " if context else ""
        msg = (
            f"{prefix}LeakageError: {mask.sum()} fila(s) con fecha >= corte "
            f"({cutoff}). "
            f"Rango violador: {violators['_date_parsed'].min()} – "
            f"{violators['_date_parsed'].max()}"
        )
        log.error(msg)
        raise LeakageError(msg)

    def check_round(
        self,
        data_round: int,
        predict_round: int,
        context: str = "",
    ) -> None:
        """
        Verifica que los datos de `data_round` son anteriores al corte de `predict_round`.
        Compara fechas de carrera completas (nivel de ronda, sin DataFrame).

        Útil para verificar antes de cargar una sesión completa de FastF1.
        """
        cutoff = self.round_cutoff(self._season, predict_round)
        data_info = self._by_round.get(data_round)

        if data_info is None:
            raise ValueError(f"Ronda de datos {data_round} no encontrada en el calendario")

        # Usamos la fecha de FP1 de los datos (la sesión más temprana del finde)
        data_date = data_info.session_dates.get("FirstPractice", data_info.date)

        if data_date >= cutoff:
            prefix = f"[{context}] " if context else ""
            msg = (
                f"{prefix}LeakageError: datos de R{data_round} "
                f"(inicio {data_date}) >= corte de R{predict_round} ({cutoff})"
            )
            log.error(msg)
            raise LeakageError(msg)

    # ------------------------------------------------------------------
    # Interfaz de contexto
    # ------------------------------------------------------------------

    @contextmanager
    def for_round(
        self, predict_round: int, date_col: str = "date"
    ) -> Generator["_RoundContext", None, None]:
        """
        Context manager que expone un objeto para verificar múltiples DataFrames
        contra el corte de una ronda concreta.

        Uso:
            with guard.for_round(5) as ctx:
                ctx.check(laps_r3)
                ctx.check(quali_r4)
        """
        cutoff = self.round_cutoff(self._season, predict_round)
        yield _RoundContext(self, cutoff, date_col, predict_round)


class _RoundContext:
    """Objeto devuelto por LeakageGuard.for_round()."""

    def __init__(
        self, guard: LeakageGuard, cutoff: date, date_col: str, predict_round: int
    ) -> None:
        self._guard = guard
        self.cutoff = cutoff
        self._date_col = date_col
        self._predict_round = predict_round

    def check(self, df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
        """Verifica df y lo devuelve para encadenamiento."""
        col = date_col or self._date_col
        self._guard.check_frame(
            df, self.cutoff, col, context=f"R{self._predict_round}"
        )
        return df

    def check_round(self, data_round: int) -> None:
        """Verifica que data_round es anterior al corte de predict_round."""
        self._guard.check_round(
            data_round, self._predict_round, context=f"R{self._predict_round}"
        )
