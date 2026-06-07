#!/usr/bin/env python3
"""
Ingesta de datos para la temporada activa.

Qué hace:
  1. Obtiene el calendario de Jolpica (fuente de round numbers canónicos).
  2. Para cada ronda completada: carga todas las sesiones via FastF1, populando
     la caché local con vueltas y resultados.
  3. Construye y guarda el driver_session_map (flag es_titular incluido).
  4. Descarga standings de pilotos y constructores de Jolpica.

Prerequisito: ninguno. Es el primer paso del pipeline.

Uso:
    python pipelines/01_ingest.py
    python pipelines/01_ingest.py --season 2026 --round 8
    python pipelines/01_ingest.py --season 2026 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from gridcast.config import get_settings
from gridcast.data.driver_session_map import (
    build_round_map,
    load_season_map,
    save_season_map,
)
from gridcast.ingest.fastf1_client import (
    PRACTICE_SESSIONS,
    get_session,
    get_testing_sessions_info,
    get_testing_laps,
)
from gridcast.ingest.jolpica_client import (
    get_constructor_standings,
    get_driver_standings,
    get_schedule,
)

log = logging.getLogger("gridcast.ingest")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Ingesta datos F1 — pipeline 01")
    p.add_argument("--season", type=int, default=cfg.season.active,
                   help="Temporada (default: %(default)s)")
    p.add_argument("--round", type=int, dest="round_", default=None, metavar="N",
                   help="Ingestar solo esta ronda (default: todas las completadas)")
    p.add_argument("--force", action="store_true",
                   help="Re-ingestar aunque el driver_session_map ya exista para la ronda")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _load_all_sessions(season: int, round_: int, event_format: str) -> None:
    """Carga todas las sesiones de una ronda para poblar la caché FastF1."""
    practice = PRACTICE_SESSIONS[event_format]
    all_sessions = practice + ["Q", "R"]
    if event_format == "sprint_qualifying":
        all_sessions = ["FP1", "SQ", "S", "Q", "R"]

    for stype in all_sessions:
        try:
            get_session(season, round_, stype, load_laps=True)
            log.debug("  R%02d %s — cargado", round_, stype)
        except Exception as exc:
            log.warning("  R%02d %s — no disponible: %s", round_, stype, exc)


def _save_standings(season: int) -> None:
    interim = _PROJECT_ROOT / get_settings().paths.data_interim
    interim.mkdir(parents=True, exist_ok=True)

    try:
        d_st = get_driver_standings(season)
        if d_st:
            pd.DataFrame([vars(s) for s in d_st]).to_parquet(
                interim / f"driver_standings_{season}.parquet", index=False
            )
            log.info("Standings pilotos: %d entradas", len(d_st))
    except Exception as exc:
        log.warning("Standings pilotos no disponibles: %s", exc)

    try:
        c_st = get_constructor_standings(season)
        if c_st:
            pd.DataFrame([vars(s) for s in c_st]).to_parquet(
                interim / f"constructor_standings_{season}.parquet", index=False
            )
            log.info("Standings constructores: %d entradas", len(c_st))
    except Exception as exc:
        log.warning("Standings constructores no disponibles: %s", exc)


def _ingest_testing(season: int, interim_dir: Path) -> None:
    """
    Descarga todas las sesiones de test de pretemporada disponibles en FastF1
    y guarda las vueltas en data/interim/testing_laps_{season}.parquet.

    Detecta automáticamente test_number y day disponibles.
    Si el archivo ya existe y tiene todos los tests conocidos, no re-descarga.
    """
    out_path = interim_dir / f"testing_laps_{season}.parquet"

    tests = get_testing_sessions_info(season)
    if not tests:
        log.info("Tests: ninguno disponible en FastF1 para %d", season)
        return

    log.info("Tests disponibles: %d sesiones — %s",
             len(tests),
             ", ".join(f"T{t['test_number']}D{t['day']} {t['location']}" for t in tests))

    # Si ya tenemos el parquet con el mismo número de sesiones, saltar
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing_keys = set(
            zip(existing["test_number"].tolist(), existing["day"].tolist())
        )
        new_keys = {(t["test_number"], t["day"]) for t in tests}
        if new_keys == existing_keys:
            log.info("Tests: parquet ya al día (%d vueltas) — saltando", len(existing))
            return

    frames: list[pd.DataFrame] = []
    for t in tests:
        try:
            laps = get_testing_laps(season, t["test_number"], t["day"])
            frames.append(laps)
            log.info("  T%dD%d %s: %d vueltas", t["test_number"], t["day"], t["location"], len(laps))
        except Exception as exc:
            log.warning("  T%dD%d: error descargando — %s", t["test_number"], t["day"], exc)

    if not frames:
        log.warning("Tests: ninguna vuelta descargada")
        return

    combined = pd.concat(frames, ignore_index=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    log.info("Tests guardados: %s (%d vueltas, %d sesiones)", out_path, len(combined), len(frames))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    season = args.season
    log.info("=== Ingesta temporada %d ===", season)

    interim = _PROJECT_ROOT / get_settings().paths.data_interim

    # 0. Tests de pretemporada (siempre, independiente de --round)
    if args.round_ is None:
        _ingest_testing(season, interim)

    # 1. Calendario desde Jolpica — siempre la fuente de round numbers
    schedule = get_schedule(season)
    log.info("Calendario: %d rondas", len(schedule))

    today = date.today()
    completed = [r for r in schedule if r.date <= today]
    if args.round_ is not None:
        completed = [r for r in completed if r.round == args.round_]
        if not completed:
            log.error("Ronda %d no encontrada o no completada todavía", args.round_)
            sys.exit(1)

    if not completed:
        log.warning("No hay rondas completadas — nada que ingestar")
        return

    log.info("Rondas a ingestar: %s", [r.round for r in completed])

    fmt_map = {
        r.round: ("sprint_qualifying" if r.has_sprint else "conventional")
        for r in schedule
    }

    # 2. Poblar caché FastF1 + acumular driver_session_map
    new_frames: list[pd.DataFrame] = []
    existing_dsm = load_season_map(season)
    already_done = (
        set(existing_dsm["round"].unique()) if existing_dsm is not None else set()
    )

    for r in completed:
        fmt = fmt_map[r.round]
        log.info("R%02d %s (%s)...", r.round, r.race_name, fmt)

        _load_all_sessions(season, r.round, fmt)

        if r.round in already_done and not args.force:
            log.info("  R%02d: driver_session_map ya existe — saltando (usa --force para regenerar)", r.round)
            continue

        try:
            dsm = build_round_map(season, r.round, fmt)
            if not dsm.empty:
                new_frames.append(dsm)
                log.info("  R%02d: %d filas en driver_session_map", r.round, len(dsm))
        except Exception as exc:
            log.warning("  R%02d: driver_session_map falló: %s", r.round, exc)

    # Combinar con el existente y guardar
    if new_frames:
        to_save_parts = (
            [existing_dsm] if existing_dsm is not None else []
        ) + new_frames
        # Eliminar duplicados si --round fuerza re-ingesta
        combined = pd.concat(to_save_parts, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["season", "round", "session_type", "car_number"], keep="last"
        )
        path = save_season_map(combined, season)
        log.info("driver_session_map guardado: %s (%d filas)", path, len(combined))
    else:
        log.info("driver_session_map sin cambios")

    # 3. Standings desde Jolpica
    _save_standings(season)

    log.info("=== Ingesta completada ===")


if __name__ == "__main__":
    main()
