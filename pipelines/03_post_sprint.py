#!/usr/bin/env python3
"""
Predicciones post-sprint para el backtest walk-forward.

Solo aplica a rondas con fin de semana sprint (R02 China, R04 Miami, R05 Canadá).

Qué hace:
  Para cada ronda sprint completada:
  1. Obtiene vueltas de la sesión Sprint (S) y calcula el ritmo de carrera por piloto.
  2. Construye extra_pace con ese ritmo como señal long_run_pace_s.
  3. Lee el snapshot pre-fp1 congelado.
  4. Genera predicción post-sprint (sin grid_positions — la clasificación aún no ha ocurrido).
  5. Guarda en outputs/predictions/{season}/{version}/round_{N:02d}_post-sprint/.

Point-in-time correctness:
  El cutoff post-sprint usa datos de FP1 + Sprint. La clasificación principal
  (que decide el grid de la carrera) ocurre DESPUÉS del sprint — no se incluye.

Uso:
  python pipelines/03_post_sprint.py --season 2026
  python pipelines/03_post_sprint.py --season 2026 --round 2
  python pipelines/03_post_sprint.py --season 2026 --force-predict
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from gridcast.backtest.leakage_guard import LeakageGuard
from gridcast.backtest.replay import (
    RoundPrediction,
    prediction_exists,
    save_prediction,
)
from gridcast.backtest.snapshot import get_snapshot
from gridcast.config import get_settings
from gridcast.ingest.fastf1_client import get_laps
from gridcast.ingest.jolpica_client import get_schedule
from gridcast.models.prediction_model import StrengthRaceModel

log = logging.getLogger("gridcast.post_sprint")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Predicciones post-sprint — pipeline 03_post_sprint")
    p.add_argument("--season",      type=int, default=cfg.season.active)
    p.add_argument("--round",       type=int, dest="round_", default=None, metavar="N",
                   help="Procesar solo esta ronda (default: todas las sprint completadas)")
    p.add_argument("--force-predict", action="store_true",
                   help="Regenerar aunque la predicción post-sprint ya exista")
    p.add_argument("--n-sims",      type=int, default=None)
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sprint pace builder
# ---------------------------------------------------------------------------

def _build_sprint_extra_pace(season: int, round_: int) -> pd.DataFrame | None:
    """
    Carga vueltas de la sesión Sprint y calcula el ritmo de carrera por piloto.

    Usa la mediana de vueltas representativas (accurate_only=True) como
    proxy de long_run_pace_s. Excluye stints de apertura/cierre.

    Retorna None si no hay datos disponibles.
    """
    try:
        laps = get_laps(season, round_, "S", accurate_only=True)
    except Exception as exc:
        log.warning("R%02d: no se pudo obtener sprint laps: %s", round_, exc)
        return None

    if laps.empty:
        log.warning("R%02d: sprint laps vacías", round_)
        return None

    # Convertir LapTime a segundos
    if "LapTime" not in laps.columns:
        log.warning("R%02d: sin columna LapTime en sprint", round_)
        return None

    laps = laps.copy()
    laps["lap_s"] = pd.to_timedelta(laps["LapTime"], errors="coerce").dt.total_seconds()
    laps = laps.dropna(subset=["lap_s"])
    laps = laps[laps["lap_s"] > 0]

    # Excluir primera y última vuelta (out/in lap más distorsionadas)
    if "LapNumber" in laps.columns:
        max_lap = laps["LapNumber"].max()
        laps = laps[(laps["LapNumber"] > 1) & (laps["LapNumber"] < max_lap)]

    if laps.empty:
        return None

    # Agregar por piloto
    driver_col = "Driver" if "Driver" in laps.columns else "Abbreviation"
    team_col   = "Team"   if "Team"   in laps.columns else "TeamName"

    grp = laps.groupby(driver_col)
    pace_rows = []
    for abbrev, grp_df in grp:
        if len(grp_df) < 3:
            continue
        pace_rows.append({
            "abbreviation":   abbrev,
            "team_name":      grp_df[team_col].iloc[0] if team_col in grp_df.columns else "",
            "long_run_pace_s": float(np.median(grp_df["lap_s"])),
            "long_run_n":      len(grp_df),
            "quali_pace_s":    float("nan"),
            "fp_best_lap_s":   float("nan"),
            "round":           round_,
            "season":          season,
        })

    if not pace_rows:
        log.warning("R%02d: sin datos de ritmo en sprint tras filtrado", round_)
        return None

    extra_pace = pd.DataFrame(pace_rows)
    log.info(
        "R%02d: sprint pace — %d pilotos, mediana mejor %.3f s",
        round_,
        len(extra_pace),
        extra_pace["long_run_pace_s"].min(),
    )
    return extra_pace


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

    log.info("=== Post-Sprint predictions temporada %d (versión %s) ===", season, version)

    schedule = get_schedule(season)
    today    = date.today()
    guard    = LeakageGuard(schedule)

    # Solo rondas sprint completadas
    sprint_rounds = [r for r in schedule if r.has_sprint and r.date <= today]
    if args.round_ is not None:
        sprint_rounds = [r for r in sprint_rounds if r.round == args.round_]

    if not sprint_rounds:
        log.info("Sin rondas sprint completadas para procesar")
        return

    log.info("Rondas sprint a procesar: %s", [r.round for r in sprint_rounds])

    model       = StrengthRaceModel(n_sims=args.n_sims)
    n_generated = 0
    n_skipped   = 0
    n_errors    = 0

    for race in sprint_rounds:
        r = race.round

        if not args.force_predict and prediction_exists(season, r, version, cutoff="post-sprint"):
            log.info("R%02d post-sprint: ya existe — skip", r)
            n_skipped += 1
            continue

        snap = get_snapshot(season, r, version)
        if not snap.is_frozen:
            log.warning("R%02d: snapshot pre-fp1 no congelado — salta post-sprint", r)
            n_errors += 1
            continue

        extra_pace = _build_sprint_extra_pace(season, r)

        if extra_pace is None:
            log.warning("R%02d: sin datos de sprint — no se genera post-sprint", r)
            n_errors += 1
            continue

        try:
            finish_probs = model.predict(
                snapshot=snap,
                round_=r,
                guard=guard,
                extra_pace=extra_pace,
                grid_positions=None,   # clasificación aún no ocurrió
            )
        except Exception as exc:
            log.error("R%02d: error generando post-sprint: %s", r, exc)
            n_errors += 1
            continue

        if finish_probs.empty:
            log.warning("R%02d: predicción post-sprint vacía", r)
            n_errors += 1
            continue

        cutoff_date = guard.round_cutoff(season, r)
        pred = RoundPrediction(
            season=season,
            round_=r,
            cutoff_date=cutoff_date,
            version=version,
            finish_probs=finish_probs,
            cutoff_name="post-sprint",
            model_version=f"{model.version}_post-sprint",
        )

        try:
            save_prediction(pred, overwrite=args.force_predict)
            n_generated += 1
        except Exception as exc:
            log.error("R%02d: error guardando post-sprint: %s", r, exc)
            n_errors += 1

    log.info(
        "=== Post-Sprint completado: %d generadas, %d saltadas, %d errores ===",
        n_generated, n_skipped, n_errors,
    )


if __name__ == "__main__":
    main()
