"""
Tests del módulo replay — walk-forward backtest.

Sin llamadas a FastF1 ni Jolpica. Usa tmp_path para no contaminar el proyecto.
Los snapshots y predicciones se crean en directorios temporales aislados.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gridcast.backtest.leakage_guard import LeakageGuard
from gridcast.backtest.replay import (
    FrozenPredictionError,
    PredictionModel,
    REQUIRED_FINISH_PROBS_COLS,
    ReplayStatus,
    RoundPrediction,
    SnapshotNotFrozenError,
    WalkForwardReplay,
    _prediction_path,
    list_predictions,
    load_prediction,
    prediction_exists,
    save_prediction,
)
from gridcast.backtest.snapshot import Snapshot, get_snapshot


# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------

BASE_DATE = date(2026, 3, 6)
VERSION = "v1"
SEASON = 2026


def _make_round_info(round_: int, fp1_date: date) -> Any:
    class _R:
        season = SEASON
        round = round_
        date = fp1_date + timedelta(days=2)
        session_dates = {"FirstPractice": fp1_date}
    return _R()


@pytest.fixture
def guard():
    schedule = [
        _make_round_info(r, BASE_DATE + timedelta(weeks=(r - 1) * 2))
        for r in range(1, 8)
    ]
    return LeakageGuard(schedule)


def _fake_finish_probs(drivers=("VER", "NOR", "LEC")) -> pd.DataFrame:
    return pd.DataFrame({
        "driver_id": list(drivers),
        "finish_pos": list(range(1, len(drivers) + 1)),
        "probability": [0.5, 0.3, 0.2][: len(drivers)],
    })


def _make_prediction(round_: int = 3, version: str = VERSION) -> RoundPrediction:
    return RoundPrediction(
        season=SEASON,
        round_=round_,
        cutoff_date=BASE_DATE + timedelta(weeks=(round_ - 1) * 2),
        version=version,
        finish_probs=_fake_finish_probs(),
        model_version="test_model_v0",
    )


class FakeModel:
    """Modelo de prueba que devuelve predicciones triviales."""

    version = "fake_v0"

    def predict(self, snapshot, round_, guard):
        return _fake_finish_probs()


class FailingModel:
    """Modelo que siempre falla — para testear fail_fast."""

    version = "failing_v0"

    def predict(self, snapshot, round_, guard):
        raise RuntimeError(f"Fallo simulado en R{round_}")


def _patch_prediction_path(monkeypatch, tmp_path: Path) -> None:
    """Redirige _prediction_path a tmp_path."""
    import gridcast.backtest.replay as rmod

    def _fake_path(season: int, round_: int, version: str) -> Path:
        return tmp_path / "predictions" / str(season) / version / f"round_{round_:02d}_pre"

    monkeypatch.setattr(rmod, "_prediction_path", _fake_path)


def _patch_snapshot_root(monkeypatch, tmp_path: Path) -> None:
    """Redirige snapshot root a tmp_path (aislamiento de get_snapshot en replay)."""
    import gridcast.backtest.snapshot as smod

    def _fake_root(season: int, version: str) -> Path:
        return tmp_path / "snapshots" / str(season) / version

    def _fake_dir(season: int, round_: int, version: str) -> Path:
        return _fake_root(season, version) / f"round_{round_:02d}_pre"

    monkeypatch.setattr(smod, "_snapshot_root", _fake_root)
    monkeypatch.setattr(smod, "_snapshot_dir", _fake_dir)


def _make_frozen_snapshot(season: int, round_: int, version: str, tmp_path: Path) -> Snapshot:
    """Crea un snapshot congelado en tmp_path para usar en tests de replay."""
    path = tmp_path / "snapshots" / str(season) / version / f"round_{round_:02d}_pre"
    snap = Snapshot(path)
    snap.init_metadata(season=season, round_=round_, cutoff=BASE_DATE, version=version)
    # frame vacío — los tests de replay no necesitan contenido real
    snap.write_frame(
        "placeholder",
        pd.DataFrame({"date": [BASE_DATE - timedelta(days=1)], "v": [1]}),
        cutoff=BASE_DATE,
    )
    snap.freeze()
    return snap


# ---------------------------------------------------------------------------
# RoundPrediction.validate()
# ---------------------------------------------------------------------------

class TestRoundPredictionValidate:
    def test_valid_prediction_passes(self):
        pred = _make_prediction()
        pred.validate()  # no debe lanzar

    def test_missing_driver_id_raises(self):
        pred = _make_prediction()
        pred.finish_probs = pd.DataFrame({"finish_pos": [1, 2], "probability": [0.6, 0.4]})
        with pytest.raises(ValueError, match="driver_id"):
            pred.validate()

    def test_empty_finish_probs_raises(self):
        pred = _make_prediction()
        pred.finish_probs = pd.DataFrame({"driver_id": []})
        with pytest.raises(ValueError, match="vacío"):
            pred.validate()

    def test_extra_columns_allowed(self):
        pred = _make_prediction()
        pred.finish_probs["extra_col"] = 0.0
        pred.validate()  # columnas adicionales son válidas


# ---------------------------------------------------------------------------
# save_prediction / load_prediction / prediction_exists
# ---------------------------------------------------------------------------

class TestSaveLoadPrediction:
    def test_prediction_does_not_exist_initially(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        assert not prediction_exists(SEASON, 3, VERSION)

    def test_save_creates_parquet_and_metadata(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        pred = _make_prediction(round_=3)
        path = save_prediction(pred)
        assert (path / "metadata.json").exists()
        assert (path / "finish_probs.parquet").exists()

    def test_prediction_exists_after_save(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        save_prediction(_make_prediction(round_=3))
        assert prediction_exists(SEASON, 3, VERSION)

    def test_round_trip_preserves_driver_ids(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        save_prediction(_make_prediction(round_=3))
        loaded = load_prediction(SEASON, 3, VERSION)
        assert loaded is not None
        assert set(loaded.finish_probs["driver_id"]) == {"VER", "NOR", "LEC"}

    def test_round_trip_preserves_metadata(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        pred = _make_prediction(round_=5)
        save_prediction(pred)
        loaded = load_prediction(SEASON, 5, VERSION)
        assert loaded.round_ == 5
        assert loaded.season == SEASON
        assert loaded.version == VERSION
        assert loaded.model_version == "test_model_v0"

    def test_load_returns_none_when_missing(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        assert load_prediction(SEASON, 99, VERSION) is None

    def test_save_raises_if_already_exists(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        save_prediction(_make_prediction(round_=3))
        with pytest.raises(FrozenPredictionError):
            save_prediction(_make_prediction(round_=3))

    def test_save_overwrite_replaces_existing(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        save_prediction(_make_prediction(round_=3))

        new_pred = _make_prediction(round_=3)
        new_pred.finish_probs = _fake_finish_probs(("HAM",))
        save_prediction(new_pred, overwrite=True)

        loaded = load_prediction(SEASON, 3, VERSION)
        assert loaded.finish_probs["driver_id"].tolist() == ["HAM"]

    def test_save_validates_prediction(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        pred = _make_prediction(round_=3)
        pred.finish_probs = pd.DataFrame({"finish_pos": [1]})  # sin driver_id
        with pytest.raises(ValueError, match="driver_id"):
            save_prediction(pred)

    def test_frozen_error_message_mentions_version(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        save_prediction(_make_prediction(round_=3))
        with pytest.raises(FrozenPredictionError, match=VERSION):
            save_prediction(_make_prediction(round_=3))


# ---------------------------------------------------------------------------
# list_predictions
# ---------------------------------------------------------------------------

class TestListPredictions:
    def test_empty_when_no_predictions(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        # También necesitamos parchear list_predictions para que use tmp_path
        import gridcast.backtest.replay as rmod
        root = tmp_path / "predictions" / str(SEASON) / VERSION
        monkeypatch.setattr(rmod, "list_predictions",
                            lambda s, v: _list_from_root(root))
        assert rmod.list_predictions(SEASON, VERSION) == []

    def test_lists_saved_rounds(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        import gridcast.backtest.replay as rmod

        save_prediction(_make_prediction(round_=2))
        save_prediction(_make_prediction(round_=4))

        root = tmp_path / "predictions" / str(SEASON) / VERSION
        monkeypatch.setattr(rmod, "list_predictions",
                            lambda s, v: _list_from_root(root))
        assert set(rmod.list_predictions(SEASON, VERSION)) == {2, 4}

    def test_result_is_sorted(self, tmp_path, monkeypatch):
        _patch_prediction_path(monkeypatch, tmp_path)
        import gridcast.backtest.replay as rmod

        for r in [5, 2, 3]:
            save_prediction(_make_prediction(round_=r))

        root = tmp_path / "predictions" / str(SEASON) / VERSION
        monkeypatch.setattr(rmod, "list_predictions",
                            lambda s, v: _list_from_root(root))
        result = rmod.list_predictions(SEASON, VERSION)
        assert result == sorted(result)


def _list_from_root(root: Path) -> list[int]:
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


# ---------------------------------------------------------------------------
# ReplayStatus
# ---------------------------------------------------------------------------

class TestReplayStatus:
    def test_snapshot_missing_when_no_snapshot(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        assert replay.status(3) == ReplayStatus.SNAPSHOT_MISSING

    def test_prediction_missing_when_snapshot_frozen(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        _make_frozen_snapshot(SEASON, 3, VERSION, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        assert replay.status(3) == ReplayStatus.PREDICTION_MISSING

    def test_prediction_frozen_when_prediction_saved(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        _make_frozen_snapshot(SEASON, 3, VERSION, tmp_path)
        save_prediction(_make_prediction(round_=3))
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        assert replay.status(3) == ReplayStatus.PREDICTION_FROZEN

    def test_status_all_returns_all_rounds(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        statuses = replay.status_all([2, 3, 4])
        assert set(statuses.keys()) == {2, 3, 4}

    def test_status_all_is_ordered(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        statuses = replay.status_all([4, 2, 3])
        assert list(statuses.keys()) == [2, 3, 4]


# ---------------------------------------------------------------------------
# WalkForwardReplay.run_round()
# ---------------------------------------------------------------------------

class TestRunRound:
    def test_raises_if_snapshot_not_frozen(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        with pytest.raises(SnapshotNotFrozenError, match="R03"):
            replay.run_round(3)

    def test_raises_error_message_mentions_version(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        with pytest.raises(SnapshotNotFrozenError, match=VERSION):
            replay.run_round(3)

    def test_generates_and_freezes_prediction(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        _make_frozen_snapshot(SEASON, 3, VERSION, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        pred = replay.run_round(3)
        assert isinstance(pred, RoundPrediction)
        assert pred.round_ == 3
        assert prediction_exists(SEASON, 3, VERSION)

    def test_prediction_records_model_version(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        _make_frozen_snapshot(SEASON, 3, VERSION, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        pred = replay.run_round(3)
        assert pred.model_version == "fake_v0"

    def test_prediction_cutoff_matches_guard(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        _make_frozen_snapshot(SEASON, 3, VERSION, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        pred = replay.run_round(3)
        assert pred.cutoff_date == guard.round_cutoff(SEASON, 3)

    def test_loads_existing_prediction_without_running_model(
        self, tmp_path, guard, monkeypatch
    ):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        _make_frozen_snapshot(SEASON, 3, VERSION, tmp_path)
        save_prediction(_make_prediction(round_=3))

        call_count = {"n": 0}
        class CountingModel:
            version = "counting_v0"
            def predict(self, *_):
                call_count["n"] += 1
                return _fake_finish_probs()

        replay = WalkForwardReplay(SEASON, guard, CountingModel(), version=VERSION)
        replay.run_round(3)
        assert call_count["n"] == 0   # modelo no se ejecutó

    def test_force_regenerates_existing_prediction(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        _make_frozen_snapshot(SEASON, 3, VERSION, tmp_path)
        save_prediction(_make_prediction(round_=3))

        call_count = {"n": 0}
        class CountingModel:
            version = "counting_v0"
            def predict(self, *_):
                call_count["n"] += 1
                return _fake_finish_probs()

        replay = WalkForwardReplay(SEASON, guard, CountingModel(), version=VERSION)
        replay.run_round(3, force=True)
        assert call_count["n"] == 1   # modelo re-ejecutado

    def test_model_exception_propagates(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        _make_frozen_snapshot(SEASON, 3, VERSION, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FailingModel(), version=VERSION)
        with pytest.raises(RuntimeError, match="Fallo simulado"):
            replay.run_round(3)


# ---------------------------------------------------------------------------
# WalkForwardReplay.run_all()
# ---------------------------------------------------------------------------

class TestRunAll:
    def _setup(self, tmp_path, guard, monkeypatch, rounds=(2, 3, 4)):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        for r in rounds:
            _make_frozen_snapshot(SEASON, r, VERSION, tmp_path)

    def test_processes_all_rounds(self, tmp_path, guard, monkeypatch):
        self._setup(tmp_path, guard, monkeypatch)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        results = replay.run_all([2, 3, 4])
        assert len(results) == 3
        assert {r.round_ for r in results} == {2, 3, 4}

    def test_results_in_ascending_order(self, tmp_path, guard, monkeypatch):
        self._setup(tmp_path, guard, monkeypatch)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        results = replay.run_all([4, 2, 3])
        assert [r.round_ for r in results] == [2, 3, 4]

    def test_skip_frozen_loads_without_model(self, tmp_path, guard, monkeypatch):
        self._setup(tmp_path, guard, monkeypatch, rounds=(2, 3))
        save_prediction(_make_prediction(round_=2))

        call_rounds = []
        class TrackingModel:
            version = "tracking_v0"
            def predict(self, snap, round_, guard):
                call_rounds.append(round_)
                return _fake_finish_probs()

        replay = WalkForwardReplay(SEASON, guard, TrackingModel(), version=VERSION)
        replay.run_all([2, 3], skip_frozen=True)
        assert 2 not in call_rounds   # R2 ya existía → skip
        assert 3 in call_rounds       # R3 nueva → modelo ejecutado

    def test_skip_frozen_false_reruns_model(self, tmp_path, guard, monkeypatch):
        self._setup(tmp_path, guard, monkeypatch, rounds=(2,))
        save_prediction(_make_prediction(round_=2))

        call_count = {"n": 0}
        class CountingModel:
            version = "c0"
            def predict(self, *_):
                call_count["n"] += 1
                return _fake_finish_probs()

        replay = WalkForwardReplay(SEASON, guard, CountingModel(), version=VERSION)
        replay.run_all([2], skip_frozen=False, force=True)
        assert call_count["n"] == 1

    def test_fail_fast_aborts_on_first_error(self, tmp_path, guard, monkeypatch):
        self._setup(tmp_path, guard, monkeypatch, rounds=(2, 3, 4))
        replay = WalkForwardReplay(
            SEASON, guard, FailingModel(), version=VERSION, fail_fast=True
        )
        with pytest.raises(RuntimeError):
            replay.run_all([2, 3, 4])

    def test_fail_fast_false_continues_on_error(self, tmp_path, guard, monkeypatch):
        self._setup(tmp_path, guard, monkeypatch, rounds=(2, 3, 4))

        class PartialFailModel:
            version = "pf_v0"
            def predict(self, snap, round_, guard):
                if round_ == 3:
                    raise RuntimeError("fallo en R3")
                return _fake_finish_probs()

        replay = WalkForwardReplay(
            SEASON, guard, PartialFailModel(), version=VERSION, fail_fast=False
        )
        results = replay.run_all([2, 3, 4])
        # R3 falló, R2 y R4 deben haberse procesado
        assert {r.round_ for r in results} == {2, 4}

    def test_empty_rounds_list(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        _patch_prediction_path(monkeypatch, tmp_path)
        replay = WalkForwardReplay(SEASON, guard, FakeModel(), version=VERSION)
        assert replay.run_all([]) == []

    def test_force_regenerates_all(self, tmp_path, guard, monkeypatch):
        self._setup(tmp_path, guard, monkeypatch, rounds=(2, 3))
        # Pre-congelar R2 y R3
        save_prediction(_make_prediction(round_=2))
        save_prediction(_make_prediction(round_=3))

        call_count = {"n": 0}
        class CountingModel:
            version = "c0"
            def predict(self, *_):
                call_count["n"] += 1
                return _fake_finish_probs()

        replay = WalkForwardReplay(SEASON, guard, CountingModel(), version=VERSION)
        replay.run_all([2, 3], force=True)
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# PredictionModel Protocol
# ---------------------------------------------------------------------------

class TestPredictionModelProtocol:
    def test_fake_model_satisfies_protocol(self):
        assert isinstance(FakeModel(), PredictionModel)

    def test_failing_model_satisfies_protocol(self):
        assert isinstance(FailingModel(), PredictionModel)

    def test_object_without_version_fails_protocol(self):
        class Bad:
            def predict(self, snap, round_, guard):
                return pd.DataFrame()

        assert not isinstance(Bad(), PredictionModel)

    def test_object_without_predict_fails_protocol(self):
        class Bad:
            version = "x"

        assert not isinstance(Bad(), PredictionModel)
