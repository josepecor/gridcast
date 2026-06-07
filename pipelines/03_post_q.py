#!/usr/bin/env python3
"""
Predicciones post-clasificación (post-q) para el backtest walk-forward.

Qué hace:
  Para cada ronda completada:
  1. Obtiene los resultados de clasificación (tiempos Q1/Q2/Q3 + posición).
  2. Construye extra_pace con los tiempos de clasificación de esa ronda.
  3. Lee el snapshot pre-fp1 congelado (datos históricos hasta el corte).
  4. Genera una predicción con la señal 'quali' (mucho más informativa que pre-fp1).
  5. Aplica modificador de posición de parrilla.
  6. Guarda en outputs/predictions/{season}/{version}/round_{N:02d}_post-q/.

Point-in-time correctness:
  Los datos de clasificación se usan SOLO para la ronda que predicen.
  No contienen información de la carrera (que no ha ocurrido todavía).

Uso:
  python pipelines/03_post_q.py --season 2026
  python pipelines/03_post_q.py --season 2026 --round 6
  python pipelines/03_post_q.py --season 2026 --force-predict
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
from gridcast.backtest.replay import (
    RoundPrediction,
    prediction_exists,
    save_prediction,
)
from gridcast.backtest.snapshot import get_snapshot
from gridcast.config import get_settings
from gridcast.ingest.fastf1_client import get_qualifying_results
from gridcast.ingest.jolpica_client import get_schedule
from gridcast.models.prediction_model import StrengthRaceModel

log = logging.getLogger("gridcast.post_q")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Predicciones post-q — pipeline 03_post_q")
    p.add_argument("--season",      type=int, default=cfg.season.active)
    p.add_argument("--round",       type=int, dest="round_", default=None, metavar="N",
                   help="Procesar solo esta ronda (default: todas las completadas)")
    p.add_argument("--force-predict", action="store_true",
                   help="Regenerar aunque la predicción post-q ya exista")
    p.add_argument("--n-sims",      type=int, default=None)
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Qualifying features builder
# ---------------------------------------------------------------------------

def _build_qualifying_extra_pace(
    season: int,
    round_: int,
) -> tuple[pd.DataFrame | None, dict[str, int] | None]:
    """
    Obtiene los tiempos de clasificación de la ronda y construye:
      - extra_pace: DataFrame con esquema de pace_features (quali_pace_s rellena)
      - grid_positions: {abbreviation: qualifying_position}

    Retorna (None, None) si no hay datos de clasificación disponibles.
    """
    try:
        quali = get_qualifying_results(season, round_)
    except Exception as exc:
        log.warning("R%02d: no se pudo obtener clasificación: %s", round_, exc)
        return None, None

    if quali.empty:
        log.warning("R%02d: clasificación vacía", round_)
        return None, None

    # Grid positions (qualifying classification order)
    grid_positions: dict[str, int] = {}
    if "quali_pos" in quali.columns:
        for _, row in quali.iterrows():
            qp = row.get("quali_pos")
            if pd.notna(qp):
                grid_positions[row["abbreviation"]] = int(qp)

    # Extra pace: quali_pace_s = best lap time
    extra_pace = quali[["abbreviation", "team_name", "round", "season", "best_lap_s"]].copy()
    extra_pace = extra_pace.rename(columns={"best_lap_s": "quali_pace_s"})
    extra_pace["long_run_pace_s"] = float("nan")
    extra_pace["fp_best_lap_s"]   = float("nan")
    extra_pace["long_run_n"]      = 0
    extra_pace = extra_pace.dropna(subset=["quali_pace_s"])

    if extra_pace.empty:
        log.warning("R%02d: ningún piloto tiene tiempo de clasificación", round_)
        return None, grid_positions or None

    log.info(
        "R%02d: clasificación disponible — %d pilotos, mejor %.3f s",
        round_,
        len(extra_pace),
        extra_pace["quali_pace_s"].min(),
    )
    return extra_pace, grid_positions or None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg     = get_settings()
    season  = args.season
    version = cfg.backtest.snapshot_version

    log.info("=== Post-Q predictions temporada %d (versión %s) ===", season, version)

    # Calendario para determinar cuáles rondas están completadas
    schedule = get_schedule(season)
    today    = date.today()
    guard    = LeakageGuard(schedule)

    completed = [r for r in schedule if r.date <= today]
    if args.round_ is not None:
        completed = [r for r in completed if r.round == args.round_]

    if not completed:
        log.info("Sin rondas completadas para procesar")
        return

    log.info("Rondas a procesar: %s", [r.round for r in completed])

    model = StrengthRaceModel(n_sims=args.n_sims)
    n_generated = 0
    n_skipped   = 0
    n_errors    = 0

    for race in completed:
        r = race.round

        # Saltar si ya existe y no se fuerza
        if not args.force_predict and prediction_exists(season, r, version, cutoff="post-q"):
            log.info("R%02d post-q: ya existe — skip", r)
            n_skipped += 1
            continue

        # Snapshot pre-fp1 debe estar congelado
        snap = get_snapshot(season, r, version)
        if not snap.is_frozen:
            log.warning("R%02d: snapshot pre-fp1 no congelado — salta post-q", r)
            n_errors += 1
            continue

        # Construir extra_pace + grid desde clasificación
        extra_pace, grid_positions = _build_qualifying_extra_pace(season, r)

        if extra_pace is None:
            log.warning("R%02d: sin datos de clasificación — no se genera post-q", r)
            n_errors += 1
            continue

        # Generar predicción
        try:
            finish_probs = model.predict(
                snapshot=snap,
                round_=r,
                guard=guard,
                extra_pace=extra_pace,
                grid_positions=grid_positions,
            )
        except Exception as exc:
            log.error("R%02d: error generando post-q: %s", r, exc)
            n_errors += 1
            continue

        if finish_probs.empty:
            log.warning("R%02d: predicción post-q vacía", r)
            n_errors += 1
            continue

        cutoff_date = guard.round_cutoff(season, r)
        pred = RoundPrediction(
            season=season,
            round_=r,
            cutoff_date=cutoff_date,
            version=version,
            finish_probs=finish_probs,
            cutoff_name="post-q",
            model_version=f"{model.version}_post-q",
        )

        try:
            save_prediction(pred, overwrite=args.force_predict)
            n_generated += 1
        except Exception as exc:
            log.error("R%02d: error guardando post-q: %s", r, exc)
            n_errors += 1

    log.info(
        "=== Post-Q completado: %d generadas, %d saltadas, %d errores ===",
        n_generated, n_skipped, n_errors,
    )


if __name__ == "__main__":
    main()
