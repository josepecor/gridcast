#!/usr/bin/env python3
"""
Ajuste del modelo jerárquico de fuerza de piloto / coche / motorista.

Qué hace:
  1. Carga los features de pace de data/processed/pace_features_{season}.parquet.
  2. Filtra a rondas 1..N para garantizar point-in-time correctness.
  3. Ajusta el modelo baseline (within-team normalization) con strength.fit_strength_model().
  4. Guarda las estimaciones en data/processed/strength_estimates_{season}_r{N:02d}.parquet.
     Formato largo: as_of_round, entity_type (driver/team/engine), entity_id, mean, sd.

Prerequisito: 02_build_features.py — pace_features_{season}.parquet en data/processed/.

Uso:
    python pipelines/04_fit_models.py --as-of-round 8
    python pipelines/04_fit_models.py --season 2026 --as-of-round 8 --signal race
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from gridcast.config import get_settings
from gridcast.models.strength import fit_strength_model

log = logging.getLogger("gridcast.models")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Ajuste del modelo de fuerza — pipeline 04")
    p.add_argument("--season",      type=int, default=cfg.season.active)
    p.add_argument("--as-of-round", type=int, required=True, metavar="N",
                   help="Ajustar usando datos de rondas 1..N (leakage-safe)")
    p.add_argument("--signal",      choices=["quali", "race"], default="quali",
                   help="Señal primaria de pace (default: quali)")
    p.add_argument("--min-rounds",  type=int, default=1,
                   help="Mínimo de rondas para incluir un piloto (default: 1)")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_pace_features(season: int) -> pd.DataFrame:
    path = _PROJECT_ROOT / get_settings().paths.data_processed / f"pace_features_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Features no encontradas: {path}\n"
            f"Ejecuta primero: python pipelines/02_build_features.py --season {season}"
        )
    race_features = pd.read_parquet(path)

    # Incluir testing features (round=0) como prior point-in-time seguro.
    # Solo tienen long_run_pace_s — se mapean al mismo schema que pace_features.
    testing_path = _PROJECT_ROOT / get_settings().paths.data_processed / f"testing_features_{season}.parquet"
    if testing_path.exists():
        testing = pd.read_parquet(testing_path).rename(columns={
            "team": "team_name",
            "testing_long_run_s": "long_run_pace_s",
            "testing_laps_n": "long_run_n",
        })
        testing["quali_pace_s"] = float("nan")
        testing["fp_best_lap_s"] = float("nan")
        race_features = pd.concat([testing, race_features], ignore_index=True)

    return race_features


def _save_estimates(estimates: pd.DataFrame, season: int, as_of_round: int) -> Path:
    out_dir = _PROJECT_ROOT / get_settings().paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"strength_estimates_{season}_r{as_of_round:02d}.parquet"
    estimates.to_parquet(out, index=False)
    return out


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

    season      = args.season
    as_of_round = args.as_of_round
    log.info("=== Ajuste del modelo temporada %d hasta R%02d ===", season, as_of_round)

    # --- Cargar y filtrar features (point-in-time correctness) ---
    all_features = _load_pace_features(season)
    features = all_features[all_features["round"] <= as_of_round].copy()
    log.info(
        "Features cargadas: %d filas, rondas %s (filtro ≤ R%02d)",
        len(features),
        sorted(features["round"].unique().tolist()) if not features.empty else [],
        as_of_round,
    )

    if features.empty:
        log.warning(
            "Sin features para temporada %d hasta R%02d. "
            "¿Se ejecutó pipeline 02?",
            season, as_of_round,
        )
        return

    # --- Ajustar modelo ---
    # engine_map: opcional — conecta equipos a su fabricante de motor.
    # Actualmente None (sin datos de motoristas). Añadir cuando se disponga de
    # config/engines.yaml o similar.
    estimates = fit_strength_model(
        pace_features=features,
        as_of_round=as_of_round,
        engine_map=None,
        primary_signal=args.signal,
        min_rounds=args.min_rounds,
    )

    n_drivers = (estimates["entity_type"] == "driver").sum()
    n_teams   = (estimates["entity_type"] == "team").sum()
    log.info(
        "Estimaciones: %d pilotos, %d equipos (total %d filas)",
        n_drivers, n_teams, len(estimates),
    )

    if n_drivers == 0:
        log.warning(
            "Ningún piloto estimado — ¿hay rondas con al menos 2 pilotos del mismo equipo "
            "con datos de %s_pace_s válidos?",
            args.signal,
        )

    # --- Guardar ---
    out = _save_estimates(estimates, season, as_of_round)
    log.info("Estimaciones guardadas: %s", out)

    # --- Resumen de los top pilotos ---
    drivers = estimates[estimates["entity_type"] == "driver"].sort_values("mean", ascending=False)
    if not drivers.empty:
        top5 = drivers.head(5)[["entity_id", "mean", "sd"]].to_string(index=False)
        log.info("Top 5 pilotos por fortaleza:\n%s", top5)

    log.info("=== Ajuste completado ===")


if __name__ == "__main__":
    main()
