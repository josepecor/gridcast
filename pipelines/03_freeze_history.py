#!/usr/bin/env python3
"""
Backfill point-in-time: construye y congela snapshots históricos + predicciones.

Qué hace por cada ronda ya disputada (ronda N):
  1. Recopila todos los datos disponibles ANTES de la FP1 de N:
       - Resultados de carrera de las rondas 1..N-1 (Jolpica)
       - driver_session_map de las rondas 1..N-1
       - Standings de pilotos y constructores tras la ronda N-1
       - pace_features de las rondas 1..N-1 (si pipeline 02 se ejecutó)
  2. Construye el snapshot congelado en data/snapshots/{season}/v1/round_NN_pre/.
  3. Genera y congela la predicción para la ronda N usando WalkForwardReplay
     con StrengthRaceModel (usa pace_features del snapshot).

Prerequisitos: 01_ingest.py (driver_session_map), 02_build_features.py (pace_features).
  Si 02 no se ejecutó, el snapshot se construye sin pace_features y el replay
  no genera predicción para la ronda (warning, no error).

Invariante (§2.2 CLAUDE.md): un snapshot congelado nunca se reescribe.
  Para regenerar con nueva versión: cambiar backtest.snapshot_version en
  config/settings.yaml y volver a ejecutar.

Uso:
    python pipelines/03_freeze_history.py
    python pipelines/03_freeze_history.py --season 2026 --round 8
    python pipelines/03_freeze_history.py --season 2026 --force
    python pipelines/03_freeze_history.py --no-predict   # solo snapshots, sin predicciones
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from gridcast.backtest.leakage_guard import LeakageGuard
from gridcast.backtest.replay import WalkForwardReplay
from gridcast.backtest.snapshot import get_snapshot
from gridcast.config import get_settings
from gridcast.data.driver_session_map import load_season_map
from gridcast.ingest.jolpica_client import (
    get_constructor_standings,
    get_driver_standings,
    get_race_results,
    get_schedule,
)
from gridcast.models.prediction_model import StrengthRaceModel

log = logging.getLogger("gridcast.freeze")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Backfill snapshots point-in-time — pipeline 03")
    p.add_argument("--season", type=int, default=cfg.season.active)
    p.add_argument("--round", type=int, dest="round_", default=None, metavar="N",
                   help="Congelar solo esta ronda (default: todas las completadas)")
    p.add_argument("--force", action="store_true",
                   help="Regenerar aunque el snapshot ya exista (solo si la versión cambió)")
    p.add_argument("--no-predict", action="store_true",
                   help="Solo construir snapshots, sin generar predicciones walk-forward")
    p.add_argument("--force-predict", action="store_true",
                   help="Regenerar predicciones aunque ya existan (modelo cambió)")
    p.add_argument("--n-sims", type=int, default=None,
                   help="Simulaciones por carrera en el replay (default: settings)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _race_date_map(schedule) -> dict[int, date]:
    return {r.round: r.date for r in schedule}


def _load_pace_features(season: int) -> pd.DataFrame | None:
    """Carga pace_features del parquet de pipeline 02. None si no existe."""
    path = (
        _PROJECT_ROOT
        / get_settings().paths.data_processed
        / f"pace_features_{season}.parquet"
    )
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _build_pace_features_frame(
    pace_all: pd.DataFrame | None,
    prior_rounds: list,
    date_map: dict[int, date],
) -> pd.DataFrame | None:
    """
    Filtra pace_features a las rondas previas y añade columna 'date'.
    Devuelve None si no hay datos disponibles.
    """
    if pace_all is None or pace_all.empty:
        return None

    prior_round_nums = {r.round for r in prior_rounds}
    pace_prior = pace_all[pace_all["round"].isin(prior_round_nums)].copy()
    if pace_prior.empty:
        return None

    pace_prior["date"] = pace_prior["round"].map(date_map)
    return pace_prior


def _build_race_results_frame(
    season: int,
    prior_rounds: list,
    date_map: dict[int, date],
) -> pd.DataFrame | None:
    rows: list[dict] = []
    for r in prior_rounds:
        try:
            results = get_race_results(season, r.round)
            for rr in results:
                row = vars(rr).copy()
                row["date"] = date_map[r.round]
                rows.append(row)
        except Exception as exc:
            log.warning("  resultados R%02d no disponibles en Jolpica: %s", r.round, exc)

    return pd.DataFrame(rows) if rows else None


def _build_dsm_frame(
    season: int,
    prior_rounds: list,
    date_map: dict[int, date],
) -> pd.DataFrame | None:
    dsm = load_season_map(season)
    if dsm is None or dsm.empty:
        log.warning(
            "driver_session_map no encontrado en data/interim/ — "
            "ejecuta primero: python pipelines/01_ingest.py --season %d",
            season,
        )
        return None

    prior_round_nums = {r.round for r in prior_rounds}
    dsm_prior = dsm[dsm["round"].isin(prior_round_nums)].copy()
    if dsm_prior.empty:
        return None

    dsm_prior["date"] = dsm_prior["round"].map(date_map)
    return dsm_prior


def _build_standings_frames(
    season: int,
    last_round: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    d_df = c_df = None
    try:
        d_st = get_driver_standings(season, after_round=last_round)
        if d_st:
            d_df = pd.DataFrame([vars(s) for s in d_st])
    except Exception as exc:
        log.warning("  standings pilotos R%02d no disponibles: %s", last_round, exc)

    try:
        c_st = get_constructor_standings(season, after_round=last_round)
        if c_st:
            c_df = pd.DataFrame([vars(s) for s in c_st])
    except Exception as exc:
        log.warning("  standings constructores R%02d no disponibles: %s", last_round, exc)

    return d_df, c_df


def freeze_round(
    season: int,
    round_info,
    schedule: list,
    guard: LeakageGuard,
    version: str,
    pace_all: pd.DataFrame | None = None,
    *,
    force: bool = False,
) -> bool:
    """
    Construye y congela el snapshot para una ronda.
    Devuelve True si el snapshot fue creado/actualizado, False si ya existía.
    """
    round_ = round_info.round
    snap = get_snapshot(season, round_, version)

    if snap.is_frozen and not force:
        log.info("R%02d: snapshot ya congelado — saltando", round_)
        return False

    if snap.is_frozen and force:
        log.warning(
            "R%02d: snapshot congelado + --force: para regenerar cambia "
            "backtest.snapshot_version en config/settings.yaml",
            round_,
        )
        return False

    cutoff = guard.round_cutoff(season, round_)
    log.info("R%02d: construyendo snapshot (corte: %s)", round_, cutoff)

    prior_rounds = [r for r in schedule if r.round < round_]
    date_map = _race_date_map(schedule)

    snap.init_metadata(season=season, round_=round_, cutoff=cutoff, version=version)

    # Frame: resultados de carrera de rondas anteriores
    results_df = _build_race_results_frame(season, prior_rounds, date_map)
    if results_df is not None:
        snap.write_frame("race_results", results_df, date_col="date", cutoff=cutoff)
        log.info("  race_results: %d filas", len(results_df))
    else:
        log.info("  race_results: sin datos previos (ronda 1)")

    # Frame: driver_session_map de rondas anteriores
    dsm_df = _build_dsm_frame(season, prior_rounds, date_map)
    if dsm_df is not None:
        snap.write_frame("driver_session_map", dsm_df, date_col="date", cutoff=cutoff)
        log.info("  driver_session_map: %d filas", len(dsm_df))

    # Frames: standings tras la última ronda anterior
    if prior_rounds:
        last_round = max(r.round for r in prior_rounds)
        d_df, c_df = _build_standings_frames(season, last_round)
        if d_df is not None:
            # point-in-time por after_round= → sin columna de fecha necesaria
            snap.write_frame("driver_standings", d_df, skip_leakage_check=True)
            log.info("  driver_standings: %d pilotos", len(d_df))
        if c_df is not None:
            snap.write_frame("constructor_standings", c_df, skip_leakage_check=True)
            log.info("  constructor_standings: %d equipos", len(c_df))

    # Frame: pace_features de rondas anteriores (requiere pipeline 02)
    pace_df = _build_pace_features_frame(pace_all, prior_rounds, date_map)
    if pace_df is not None:
        snap.write_frame("pace_features", pace_df, date_col="date", cutoff=cutoff)
        log.info("  pace_features: %d filas (%d rondas)", len(pace_df), pace_df["round"].nunique())
    else:
        log.info(
            "  pace_features: sin datos (ejecuta 02_build_features.py para habilitar predicciones)"
        )

    snap.freeze()
    log.info("R%02d: snapshot congelado ✓", round_)
    return True


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = get_settings()
    season = args.season
    version = cfg.backtest.snapshot_version
    log.info("=== Freeze history temporada %d (versión %s) ===", season, version)

    schedule = get_schedule(season)
    guard = LeakageGuard(schedule)

    today = date.today()
    to_process = [r for r in schedule if r.date <= today]
    if args.round_ is not None:
        to_process = [r for r in to_process if r.round == args.round_]
        if not to_process:
            log.error("Ronda %d no encontrada o no completada todavía", args.round_)
            sys.exit(1)

    log.info("Rondas a congelar: %s", [r.round for r in to_process])

    # Cargar pace_features una vez (puede ser None si pipeline 02 no se ejecutó)
    pace_all = _load_pace_features(season)
    if pace_all is None:
        log.warning(
            "pace_features no encontradas — snapshots sin ese frame; "
            "las predicciones walk-forward no estarán disponibles. "
            "Ejecuta: python pipelines/02_build_features.py --season %d",
            season,
        )
    else:
        log.info("pace_features cargadas: %d filas (%d rondas)", len(pace_all), pace_all["round"].nunique())

    created = skipped = 0
    for r in to_process:
        try:
            was_created = freeze_round(
                season, r, schedule, guard, version, pace_all, force=args.force
            )
            if was_created:
                created += 1
            else:
                skipped += 1
        except Exception as exc:
            log.error("R%02d: error inesperado: %s", r.round, exc)
            sys.exit(1)

    log.info(
        "=== Freeze completado: %d creados, %d ya existían ===",
        created,
        skipped,
    )

    # --- Walk-forward replay (predicciones por ronda) ---
    if args.no_predict:
        log.info("--no-predict: saltando generación de predicciones")
        return

    rounds_with_snapshots = [r.round for r in to_process]
    if not rounds_with_snapshots:
        return

    model = StrengthRaceModel(n_sims=args.n_sims)
    replay = WalkForwardReplay(season, guard, model, version=version, fail_fast=False)

    log.info(
        "=== Walk-forward replay: %d rondas (modelo: %s) ===",
        len(rounds_with_snapshots),
        model.version,
    )
    force_predict = getattr(args, "force_predict", False)
    predictions = replay.run_all(
        rounds_with_snapshots,
        skip_frozen=not force_predict,
        force=force_predict,
    )
    log.info(
        "=== Replay completado: %d predicciones generadas/cargadas ===",
        len(predictions),
    )


if __name__ == "__main__":
    main()
