"""
Tests del LeakageGuard — marcado como crítico en CLAUDE.md.
Deben pasar antes de cualquier commit que toque código de backtest o features.

Estos tests NO tocan FastF1 ni Jolpica: usan un schedule sintético.
"""

from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from gridcast.backtest.leakage_guard import LeakageError, LeakageGuard


# ---------------------------------------------------------------------------
# Fixtures: calendario sintético sin dependencias externas
# ---------------------------------------------------------------------------

def _make_round(
    round_: int,
    fp1_date: date,
    race_date: date | None = None,
    has_sprint: bool = False,
) -> Any:
    """Construye un RoundInfo mínimo compatible con LeakageGuard."""

    class _RoundInfo:
        season = 2026
        round = round_
        date = race_date or (fp1_date + timedelta(days=2))
        session_dates = {"FirstPractice": fp1_date}

    return _RoundInfo()


@pytest.fixture
def schedule():
    """
    Calendario sintético de 5 rondas.
    Los fines de semana empiezan los viernes, con 2 semanas entre rondas.
    """
    base = date(2026, 3, 6)   # viernes R1
    return [
        _make_round(r, base + timedelta(weeks=(r - 1) * 2))
        for r in range(1, 6)
    ]


@pytest.fixture
def guard(schedule):
    return LeakageGuard(schedule)


def _df_with_date(d: date, n_rows: int = 3, date_col: str = "date") -> pd.DataFrame:
    return pd.DataFrame({date_col: [d] * n_rows, "value": range(n_rows)})


# ---------------------------------------------------------------------------
# round_cutoff
# ---------------------------------------------------------------------------

class TestRoundCutoff:
    def test_cutoff_is_fp1_date(self, guard, schedule):
        for r in schedule:
            assert guard.round_cutoff(2026, r.round) == r.session_dates["FirstPractice"]

    def test_cutoff_increases_with_round(self, guard, schedule):
        cutoffs = [guard.round_cutoff(2026, r.round) for r in schedule]
        assert cutoffs == sorted(cutoffs)

    def test_unknown_round_raises(self, guard):
        with pytest.raises(ValueError, match="no encontrada"):
            guard.round_cutoff(2026, 99)

    def test_fallback_when_no_fp1_date(self):
        class _NoFP1:
            season = 2026
            round = 1
            date = date(2026, 3, 8)
            session_dates: dict = {}  # sin FirstPractice

        guard = LeakageGuard([_NoFP1()])
        cutoff = guard.round_cutoff(2026, 1)
        assert cutoff == date(2026, 3, 6)   # race_date - 2 días


# ---------------------------------------------------------------------------
# check_frame — casos básicos
# ---------------------------------------------------------------------------

class TestCheckFrame:
    def test_empty_df_passes(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        guard.check_frame(pd.DataFrame(columns=["date", "value"]), cutoff)

    def test_data_strictly_before_cutoff_passes(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = _df_with_date(cutoff - timedelta(days=1))
        guard.check_frame(df, cutoff)  # no debe lanzar

    def test_data_on_cutoff_date_fails(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = _df_with_date(cutoff)    # exactamente el día del corte
        with pytest.raises(LeakageError):
            guard.check_frame(df, cutoff)

    def test_data_after_cutoff_fails(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = _df_with_date(cutoff + timedelta(days=1))
        with pytest.raises(LeakageError):
            guard.check_frame(df, cutoff)

    def test_mixed_dates_fails_if_any_future(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = pd.DataFrame({
            "date": [
                cutoff - timedelta(days=7),   # OK
                cutoff - timedelta(days=1),   # OK
                cutoff,                        # ← leakage
            ],
            "value": [1, 2, 3],
        })
        with pytest.raises(LeakageError):
            guard.check_frame(df, cutoff)

    def test_error_message_contains_cutoff_date(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = _df_with_date(cutoff + timedelta(days=5))
        with pytest.raises(LeakageError, match=str(cutoff)):
            guard.check_frame(df, cutoff)

    def test_missing_date_column_raises_valueerror(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = pd.DataFrame({"value": [1, 2]})
        with pytest.raises(ValueError, match="no encontrada"):
            guard.check_frame(df, cutoff)

    def test_custom_date_col(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = _df_with_date(cutoff - timedelta(days=1), date_col="session_date")
        guard.check_frame(df, cutoff, date_col="session_date")


# ---------------------------------------------------------------------------
# check_frame — formatos de fecha soportados
# ---------------------------------------------------------------------------

class TestDateFormats:
    def test_date_object(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = _df_with_date(cutoff - timedelta(days=1))
        guard.check_frame(df, cutoff)

    def test_pandas_timestamp(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = pd.DataFrame({
            "date": pd.to_datetime([cutoff - timedelta(days=1)]),
            "value": [1],
        })
        guard.check_frame(df, cutoff)

    def test_pandas_timestamp_leaks(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = pd.DataFrame({
            "date": pd.to_datetime([cutoff + timedelta(days=1)]),
            "value": [1],
        })
        with pytest.raises(LeakageError):
            guard.check_frame(df, cutoff)

    def test_string_isoformat(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = pd.DataFrame({
            "date": [(cutoff - timedelta(days=1)).isoformat()],
            "value": [1],
        })
        guard.check_frame(df, cutoff)

    def test_string_isoformat_leaks(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 3)
        df = pd.DataFrame({
            "date": [(cutoff + timedelta(days=3)).isoformat()],
            "value": [1],
        })
        with pytest.raises(LeakageError):
            guard.check_frame(df, cutoff)


# ---------------------------------------------------------------------------
# check_round
# ---------------------------------------------------------------------------

class TestCheckRound:
    def test_prior_round_passes(self, guard):
        guard.check_round(data_round=1, predict_round=3)
        guard.check_round(data_round=2, predict_round=3)

    def test_same_round_fails(self, guard):
        with pytest.raises(LeakageError):
            guard.check_round(data_round=3, predict_round=3)

    def test_future_round_fails(self, guard):
        with pytest.raises(LeakageError):
            guard.check_round(data_round=4, predict_round=3)

    def test_unknown_data_round_raises(self, guard):
        with pytest.raises(ValueError):
            guard.check_round(data_round=99, predict_round=3)

    def test_round_1_has_no_prior_rounds(self, guard):
        # No hay rondas anteriores a R1; cualquier intento de usar R1 datos para R1 falla
        with pytest.raises(LeakageError):
            guard.check_round(data_round=1, predict_round=1)


# ---------------------------------------------------------------------------
# context manager for_round
# ---------------------------------------------------------------------------

class TestForRound:
    def test_context_passes_for_prior_data(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 4)
        df = _df_with_date(cutoff - timedelta(days=3))
        with guard.for_round(4) as ctx:
            result = ctx.check(df)
        assert result is df  # devuelve el df para encadenamiento

    def test_context_fails_for_future_data(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 4)
        df = _df_with_date(cutoff + timedelta(days=1))
        with pytest.raises(LeakageError):
            with guard.for_round(4) as ctx:
                ctx.check(df)

    def test_context_check_round_passes(self, guard):
        with guard.for_round(4) as ctx:
            ctx.check_round(1)
            ctx.check_round(2)
            ctx.check_round(3)

    def test_context_check_round_fails_for_current(self, guard):
        with pytest.raises(LeakageError):
            with guard.for_round(4) as ctx:
                ctx.check_round(4)   # misma ronda → leakage

    def test_context_exposes_cutoff(self, guard, schedule):
        with guard.for_round(3) as ctx:
            assert ctx.cutoff == guard.round_cutoff(2026, 3)

    def test_error_message_includes_predict_round(self, guard, schedule):
        cutoff = guard.round_cutoff(2026, 4)
        df = _df_with_date(cutoff + timedelta(days=1))
        with pytest.raises(LeakageError, match="R4"):
            with guard.for_round(4) as ctx:
                ctx.check(df)


# ---------------------------------------------------------------------------
# cutoffs_for_season
# ---------------------------------------------------------------------------

class TestCutoffsForSeason:
    def test_returns_all_rounds(self, guard, schedule):
        cutoffs = guard.cutoffs_for_season()
        assert set(cutoffs.keys()) == {1, 2, 3, 4, 5}

    def test_values_match_round_cutoff(self, guard, schedule):
        cutoffs = guard.cutoffs_for_season()
        for r in range(1, 6):
            assert cutoffs[r] == guard.round_cutoff(2026, r)
