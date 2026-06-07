"""
Tests del módulo snapshot — estados point-in-time congelados para el backtest.

Sin llamadas a FastF1 ni Jolpica. Se usa tmp_path para no contaminar data/.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gridcast.backtest.leakage_guard import LeakageGuard
from gridcast.backtest.snapshot import (
    FrozenSnapshotError,
    Snapshot,
    SnapshotLeakageError,
    _validate_leakage,
    build_snapshot,
    get_snapshot,
    list_frozen_snapshots,
    list_snapshots,
)


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------

CUTOFF = date(2026, 4, 17)     # viernes ficticio


def _snap(tmp_path: Path) -> Snapshot:
    return Snapshot(tmp_path / "round_05_pre")


def _df(d: date, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"date": [d] * n, "value": range(n)})


def _make_round(round_: int, fp1_date: date) -> Any:
    class _R:
        season = 2026
        round = round_
        date = fp1_date + timedelta(days=2)
        session_dates = {"FirstPractice": fp1_date}

    return _R()


@pytest.fixture
def snap(tmp_path):
    return _snap(tmp_path)


@pytest.fixture
def guard():
    base = date(2026, 3, 6)
    schedule = [
        _make_round(r, base + timedelta(weeks=(r - 1) * 2))
        for r in range(1, 8)
    ]
    return LeakageGuard(schedule)


@pytest.fixture
def snap_with_meta(tmp_path):
    s = _snap(tmp_path)
    s.init_metadata(season=2026, round_=5, cutoff=CUTOFF, version="v1")
    return s


# ---------------------------------------------------------------------------
# Propiedades en snapshot vacío
# ---------------------------------------------------------------------------

class TestEmptySnapshot:
    def test_not_exists_initially(self, snap):
        assert not snap.exists

    def test_not_frozen_initially(self, snap):
        assert not snap.is_frozen

    def test_metadata_empty_dict(self, snap):
        assert snap.metadata == {}

    def test_frames_empty_list(self, snap):
        assert snap.frames == []

    def test_repr_open(self, snap):
        assert "open" in repr(snap) or "frozen" not in repr(snap)


# ---------------------------------------------------------------------------
# init_metadata
# ---------------------------------------------------------------------------

class TestInitMetadata:
    def test_creates_metadata_file(self, snap):
        snap.init_metadata(2026, 5, CUTOFF, "v1")
        assert (snap.path / "metadata.json").exists()

    def test_metadata_content(self, snap):
        snap.init_metadata(2026, 5, CUTOFF, "v1")
        meta = snap.metadata
        assert meta["season"] == 2026
        assert meta["round"] == 5
        assert meta["cutoff_date"] == CUTOFF.isoformat()
        assert meta["version"] == "v1"
        assert meta["frozen"] is False
        assert meta["frames"] == []

    def test_exists_after_init(self, snap):
        snap.init_metadata(2026, 5, CUTOFF, "v1")
        assert snap.exists

    def test_repeated_init_preserves_frames(self, snap):
        """Re-init no borra frames ya escritos (reintentos de pipeline)."""
        snap.init_metadata(2026, 5, CUTOFF, "v1")
        df = _df(CUTOFF - timedelta(days=1))
        snap.write_frame("results", df, cutoff=CUTOFF)
        # Segunda llamada: no debe borrar frames
        snap.init_metadata(2026, 5, CUTOFF, "v1")
        assert "results" in snap.frames

    def test_init_after_freeze_raises(self, snap):
        snap.init_metadata(2026, 5, CUTOFF, "v1")
        snap.freeze()
        with pytest.raises(FrozenSnapshotError):
            snap.init_metadata(2026, 5, CUTOFF, "v1")


# ---------------------------------------------------------------------------
# write_frame — casos válidos
# ---------------------------------------------------------------------------

class TestWriteFrame:
    def test_write_creates_parquet(self, snap_with_meta):
        df = _df(CUTOFF - timedelta(days=1))
        snap_with_meta.write_frame("results", df, cutoff=CUTOFF)
        assert (snap_with_meta.path / "results.parquet").exists()

    def test_write_registers_frame_name(self, snap_with_meta):
        df = _df(CUTOFF - timedelta(days=1))
        snap_with_meta.write_frame("results", df, cutoff=CUTOFF)
        assert "results" in snap_with_meta.frames

    def test_multiple_frames_all_registered(self, snap_with_meta):
        df = _df(CUTOFF - timedelta(days=1))
        snap_with_meta.write_frame("results", df, cutoff=CUTOFF)
        snap_with_meta.write_frame("driver_map", df, cutoff=CUTOFF)
        assert set(snap_with_meta.frames) == {"results", "driver_map"}

    def test_frames_list_is_sorted(self, snap_with_meta):
        df = _df(CUTOFF - timedelta(days=1))
        snap_with_meta.write_frame("zzz", df, cutoff=CUTOFF)
        snap_with_meta.write_frame("aaa", df, cutoff=CUTOFF)
        assert snap_with_meta.frames == sorted(snap_with_meta.frames)

    def test_cutoff_inferred_from_metadata(self, snap_with_meta):
        """write_frame infiere el cutoff de init_metadata si no se pasa explícito."""
        df = _df(CUTOFF - timedelta(days=3))
        snap_with_meta.write_frame("results", df)   # sin cutoff=

    def test_no_metadata_no_cutoff_raises(self, snap):
        """Sin metadata y sin cutoff= explícito, no se puede validar leakage."""
        df = _df(CUTOFF - timedelta(days=1))
        with pytest.raises(ValueError, match="cutoff"):
            snap.write_frame("results", df)

    def test_custom_date_col(self, snap_with_meta):
        df = pd.DataFrame({"session_date": [CUTOFF - timedelta(days=1)], "v": [1]})
        snap_with_meta.write_frame("pace", df, date_col="session_date", cutoff=CUTOFF)
        assert "pace" in snap_with_meta.frames


# ---------------------------------------------------------------------------
# write_frame — leakage
# ---------------------------------------------------------------------------

class TestWriteFrameLeakage:
    def test_data_on_cutoff_raises(self, snap_with_meta):
        df = _df(CUTOFF)    # fecha == corte → leakage
        with pytest.raises(SnapshotLeakageError):
            snap_with_meta.write_frame("results", df, cutoff=CUTOFF)

    def test_data_after_cutoff_raises(self, snap_with_meta):
        df = _df(CUTOFF + timedelta(days=1))
        with pytest.raises(SnapshotLeakageError):
            snap_with_meta.write_frame("results", df, cutoff=CUTOFF)

    def test_missing_date_col_raises_value_error(self, snap_with_meta):
        df = pd.DataFrame({"value": [1, 2]})
        with pytest.raises(ValueError, match="date"):
            snap_with_meta.write_frame("results", df, cutoff=CUTOFF)

    def test_skip_leakage_check_bypasses_validation(self, snap):
        """Con skip_leakage_check=True no se comprueba nada — el caller es responsable."""
        snap.path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"value": [1, 2, 3]})    # sin columna de fecha
        snap.write_frame("model_est", df, skip_leakage_check=True)
        assert "model_est" in snap.frames

    def test_error_message_contains_cutoff(self, snap_with_meta):
        df = _df(CUTOFF + timedelta(days=5))
        with pytest.raises(SnapshotLeakageError, match=str(CUTOFF)):
            snap_with_meta.write_frame("results", df, cutoff=CUTOFF)

    def test_error_message_contains_frame_name(self, snap_with_meta):
        df = _df(CUTOFF)
        with pytest.raises(SnapshotLeakageError, match="race_results"):
            snap_with_meta.write_frame("race_results", df, cutoff=CUTOFF)


# ---------------------------------------------------------------------------
# write_frame — formatos de fecha
# ---------------------------------------------------------------------------

class TestWriteFrameDateFormats:
    def test_datetime_date_object(self, snap_with_meta):
        df = _df(CUTOFF - timedelta(days=1))
        snap_with_meta.write_frame("f", df, cutoff=CUTOFF)

    def test_pandas_timestamp_before(self, snap_with_meta):
        df = pd.DataFrame({
            "date": pd.to_datetime([CUTOFF - timedelta(days=1)]),
            "v": [1],
        })
        snap_with_meta.write_frame("f", df, cutoff=CUTOFF)

    def test_pandas_timestamp_on_cutoff_raises(self, snap_with_meta):
        df = pd.DataFrame({"date": pd.to_datetime([CUTOFF]), "v": [1]})
        with pytest.raises(SnapshotLeakageError):
            snap_with_meta.write_frame("f", df, cutoff=CUTOFF)

    def test_iso_string_before(self, snap_with_meta):
        df = pd.DataFrame({
            "date": [(CUTOFF - timedelta(days=1)).isoformat()],
            "v": [1],
        })
        snap_with_meta.write_frame("f", df, cutoff=CUTOFF)

    def test_iso_string_on_cutoff_raises(self, snap_with_meta):
        df = pd.DataFrame({"date": [CUTOFF.isoformat()], "v": [1]})
        with pytest.raises(SnapshotLeakageError):
            snap_with_meta.write_frame("f", df, cutoff=CUTOFF)


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------

class TestFreeze:
    def test_freeze_marks_frozen(self, snap_with_meta):
        snap_with_meta.freeze()
        assert snap_with_meta.is_frozen

    def test_freeze_writes_frozen_at(self, snap_with_meta):
        snap_with_meta.freeze()
        assert "frozen_at" in snap_with_meta.metadata

    def test_freeze_without_metadata_raises(self, snap):
        with pytest.raises(RuntimeError):
            snap.freeze()

    def test_double_freeze_no_error(self, snap_with_meta):
        snap_with_meta.freeze()
        snap_with_meta.freeze()    # idempotente — no debe lanzar
        assert snap_with_meta.is_frozen

    def test_write_after_freeze_raises(self, snap_with_meta):
        snap_with_meta.freeze()
        df = _df(CUTOFF - timedelta(days=1))
        with pytest.raises(FrozenSnapshotError):
            snap_with_meta.write_frame("results", df, cutoff=CUTOFF)

    def test_frozen_error_message_mentions_version(self, snap_with_meta):
        snap_with_meta.freeze()
        df = _df(CUTOFF - timedelta(days=1))
        with pytest.raises(FrozenSnapshotError, match="snapshot_version"):
            snap_with_meta.write_frame("results", df, cutoff=CUTOFF)

    def test_repr_shows_frozen(self, snap_with_meta):
        snap_with_meta.freeze()
        assert "frozen" in repr(snap_with_meta)


# ---------------------------------------------------------------------------
# read_frame / read_all_frames
# ---------------------------------------------------------------------------

class TestReadFrame:
    def test_read_back_same_data(self, snap_with_meta):
        original = _df(CUTOFF - timedelta(days=3))
        snap_with_meta.write_frame("results", original, cutoff=CUTOFF)
        snap_with_meta.freeze()
        loaded = snap_with_meta.read_frame("results")
        assert loaded["value"].tolist() == original["value"].tolist()

    def test_read_missing_frame_raises(self, snap_with_meta):
        with pytest.raises(FileNotFoundError, match="pace"):
            snap_with_meta.read_frame("pace")

    def test_read_error_lists_available_frames(self, snap_with_meta):
        df = _df(CUTOFF - timedelta(days=1))
        snap_with_meta.write_frame("results", df, cutoff=CUTOFF)
        with pytest.raises(FileNotFoundError, match="results"):
            snap_with_meta.read_frame("nonexistent")

    def test_read_all_frames(self, snap_with_meta):
        df = _df(CUTOFF - timedelta(days=1))
        snap_with_meta.write_frame("r1", df, cutoff=CUTOFF)
        snap_with_meta.write_frame("r2", df, cutoff=CUTOFF)
        snap_with_meta.freeze()
        all_frames = snap_with_meta.read_all_frames()
        assert set(all_frames.keys()) == {"r1", "r2"}

    def test_read_from_nonexistent_snapshot_raises(self, snap):
        with pytest.raises(FileNotFoundError):
            snap.read_frame("results")


# ---------------------------------------------------------------------------
# build_snapshot — función de alto nivel
# ---------------------------------------------------------------------------

class TestBuildSnapshot:
    def test_build_creates_frozen_snapshot(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        frames = {"results": _df(date(2026, 3, 7))}   # antes del corte de R2
        snap = build_snapshot(2026, 2, frames, guard)
        assert snap.is_frozen

    def test_build_idempotent_if_already_frozen(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        frames = {"results": _df(date(2026, 3, 7))}
        snap1 = build_snapshot(2026, 2, frames, guard)
        snap2 = build_snapshot(2026, 2, frames, guard)    # segunda llamada
        assert snap2.is_frozen
        assert snap1.path == snap2.path

    def test_build_raises_on_leakage(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        cutoff_r2 = guard.round_cutoff(2026, 2)
        frames = {"results": _df(cutoff_r2 + timedelta(days=1))}  # leakage
        with pytest.raises(SnapshotLeakageError):
            build_snapshot(2026, 2, frames, guard)

    def test_build_snapshot_writes_all_frames(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        ok_date = date(2026, 3, 7)
        frames = {
            "results": _df(ok_date),
            "driver_map": _df(ok_date),
        }
        snap = build_snapshot(2026, 2, frames, guard)
        assert set(snap.frames) == {"driver_map", "results"}

    def test_build_snapshot_r1_has_no_prior_race_data(self, tmp_path, guard, monkeypatch):
        """R1 no tiene datos previos — snapshot se crea vacío o con solo prior estático."""
        _patch_snapshot_root(monkeypatch, tmp_path)
        # No hay carreras anteriores, frames vacío es válido
        snap = build_snapshot(2026, 1, {}, guard)
        assert snap.is_frozen


# ---------------------------------------------------------------------------
# list_snapshots / list_frozen_snapshots
# ---------------------------------------------------------------------------

class TestListSnapshots:
    def test_empty_when_no_snapshots(self, tmp_path, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        assert list_snapshots(2026) == []

    def test_lists_created_snapshots(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        build_snapshot(2026, 2, {"r": _df(date(2026, 3, 7))}, guard)
        build_snapshot(2026, 3, {"r": _df(date(2026, 3, 7))}, guard)
        assert set(list_snapshots(2026)) == {2, 3}

    def test_list_frozen_only_frozen(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        # Crear un snapshot frozen y uno abierto
        build_snapshot(2026, 2, {"r": _df(date(2026, 3, 7))}, guard)

        # Snapshot abierto (sin freeze())
        snap3_path = tmp_path / "2026" / "v1" / "round_03_pre"
        s = Snapshot(snap3_path)
        s.init_metadata(2026, 3, date(2026, 3, 20), "v1")
        # No freeze

        frozen = list_frozen_snapshots(2026)
        assert 2 in frozen
        assert 3 not in frozen

    def test_list_sorted_by_round(self, tmp_path, guard, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        build_snapshot(2026, 4, {"r": _df(date(2026, 3, 7))}, guard)
        build_snapshot(2026, 2, {"r": _df(date(2026, 3, 7))}, guard)
        build_snapshot(2026, 3, {"r": _df(date(2026, 3, 7))}, guard)
        result = list_snapshots(2026)
        assert result == sorted(result)

    def test_ignores_non_round_directories(self, tmp_path, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        # Directorio que no sigue el patrón
        (tmp_path / "2026" / "v1" / "misc_dir").mkdir(parents=True)
        assert list_snapshots(2026) == []


# ---------------------------------------------------------------------------
# _validate_leakage (función interna — testeable directamente)
# ---------------------------------------------------------------------------

class TestValidateLeakage:
    def test_all_before_cutoff_passes(self):
        df = _df(CUTOFF - timedelta(days=5))
        _validate_leakage(df, CUTOFF, "date", "test_frame")

    def test_on_cutoff_raises(self):
        df = _df(CUTOFF)
        with pytest.raises(SnapshotLeakageError):
            _validate_leakage(df, CUTOFF, "date", "test_frame")

    def test_after_cutoff_raises(self):
        df = _df(CUTOFF + timedelta(days=2))
        with pytest.raises(SnapshotLeakageError):
            _validate_leakage(df, CUTOFF, "date", "test_frame")

    def test_empty_df_passes(self):
        df = pd.DataFrame({"date": pd.Series([], dtype="object"), "v": []})
        _validate_leakage(df, CUTOFF, "date", "empty")

    def test_mixed_valid_invalid_raises(self):
        df = pd.DataFrame({
            "date": [CUTOFF - timedelta(days=1), CUTOFF],
            "v": [1, 2],
        })
        with pytest.raises(SnapshotLeakageError):
            _validate_leakage(df, CUTOFF, "date", "mixed")

    def test_error_count_in_message(self):
        df = _df(CUTOFF + timedelta(days=1), n=7)
        with pytest.raises(SnapshotLeakageError, match="7"):
            _validate_leakage(df, CUTOFF, "date", "test_frame")

    def test_datetime64_dtype(self):
        df = pd.DataFrame({
            "date": pd.to_datetime([CUTOFF + timedelta(days=1)]),
            "v": [1],
        })
        with pytest.raises(SnapshotLeakageError):
            _validate_leakage(df, CUTOFF, "date", "ts_frame")


# ---------------------------------------------------------------------------
# get_snapshot — factory
# ---------------------------------------------------------------------------

class TestGetSnapshot:
    def test_returns_snapshot_object(self, tmp_path, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        s = get_snapshot(2026, 5)
        assert isinstance(s, Snapshot)

    def test_does_not_create_on_disk(self, tmp_path, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        s = get_snapshot(2026, 5)
        assert not s.exists

    def test_path_includes_round_number(self, tmp_path, monkeypatch):
        _patch_snapshot_root(monkeypatch, tmp_path)
        s = get_snapshot(2026, 5)
        assert "round_05_pre" in str(s.path)


# ---------------------------------------------------------------------------
# Helpers para aislar tests de la config real
# ---------------------------------------------------------------------------

def _patch_snapshot_root(monkeypatch, tmp_path: Path) -> None:
    """Redirige _snapshot_root a tmp_path para que los tests no contaminen data/."""
    import gridcast.backtest.snapshot as snap_mod

    def _fake_root(season: int, version: str) -> Path:
        return tmp_path / str(season) / version

    monkeypatch.setattr(snap_mod, "_snapshot_root", _fake_root)
    monkeypatch.setattr(snap_mod, "_snapshot_dir",
                        lambda s, r, v: _fake_root(s, v) / f"round_{r:02d}_pre")
