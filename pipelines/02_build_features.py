#!/usr/bin/env python3
"""
Construcción de features de pace y gaps entre compañeros de equipo.

Qué hace:
  1. Para cada ronda completada carga las vueltas de la sesión de pace
     primaria (FP2 en normal, FP1 en sprint) y las vueltas de clasificación (Q).
  2. Calcula long_run_pace_s (fuel-corrected) y fp_best_lap_s desde práctica.
  3. Calcula quali_pace_s desde la sesión Q.
  4. Calcula gap_to_teammate_race y gap_to_teammate_quali.
  5. Guarda las features en data/processed/pace_features_{season}.parquet.

Prerequisito: 01_ingest.py (caché FastF1 + driver_session_map).

Uso:
    python pipelines/02_build_features.py
    python pipelines/02_build_features.py --season 2026 --round 8
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
from gridcast.features.pace import compute_pace_features
from gridcast.features.teammate_gaps import compute_teammate_gaps
from gridcast.ingest.fastf1_client import get_laps, get_pace_laps
from gridcast.ingest.jolpica_client import get_schedule

_MIN_STINT_LAPS = 5   # mínimo de vueltas consecutivas para considerar una tanda de test como long run

log = logging.getLogger("gridcast.features")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Construcción de features — pipeline 02")
    p.add_argument("--season", type=int, default=cfg.season.active)
    p.add_argument("--round", type=int, dest="round_", default=None, metavar="N",
                   help="Procesar solo esta ronda (default: todas las completadas)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _load_quali_laps(season: int, round_: int) -> pd.DataFrame | None:
    """Carga vueltas de Q. Devuelve None si la sesión no está disponible."""
    try:
        return get_laps(season, round_, "Q", accurate_only=False)
    except Exception as exc:
        log.warning("R%02d Q: no disponible (%s)", round_, exc)
        return None


def _build_testing_features(season: int) -> pd.DataFrame | None:
    """
    Calcula pace medio de test de pretemporada por piloto a partir de
    data/interim/testing_laps_{season}.parquet.

    Solo usa tandas largas (≥ MIN_STINT_LAPS vueltas del mismo compuesto sin pit),
    que son la señal más limpia disponible en testing.

    Devuelve un DataFrame con columnas:
      abbreviation, team, testing_long_run_s, testing_laps_n
    con round=0 para distinguirlo de las rondas reales.
    """
    interim = _PROJECT_ROOT / get_settings().paths.data_interim
    path = interim / f"testing_laps_{season}.parquet"
    if not path.exists():
        return None

    laps = pd.read_parquet(path)
    if laps.empty or "LapTime" not in laps.columns:
        return None

    # Convertir LapTime a segundos
    laps = laps.copy()
    laps["lap_s"] = pd.to_timedelta(laps["LapTime"], errors="coerce").dt.total_seconds()
    laps = laps.dropna(subset=["lap_s", "Driver", "Team", "Stint", "Compound"])

    # Solo tandas largas (stints de ≥ MIN_STINT_LAPS vueltas)
    stint_len = laps.groupby(["Driver", "test_number", "day", "Stint"])["lap_s"].transform("count")
    long_runs = laps[stint_len >= _MIN_STINT_LAPS].copy()

    if long_runs.empty:
        log.warning("Testing: ninguna tanda larga encontrada (≥%d vueltas)", _MIN_STINT_LAPS)
        return None

    # Media por piloto (excluyendo el 10% más lento para robustez)
    def trimmed_mean(x: pd.Series) -> float:
        if len(x) < 3:
            return x.mean()
        q90 = x.quantile(0.90)
        return x[x <= q90].mean()

    agg = (
        long_runs.groupby(["Driver", "Team"])
        .agg(
            testing_long_run_s=("lap_s", trimmed_mean),
            testing_laps_n=("lap_s", "count"),
        )
        .reset_index()
        .rename(columns={"Driver": "abbreviation", "Team": "team"})
    )
    agg["round"] = 0    # ronda 0 = pretemporada
    agg["season"] = season

    log.info(
        "Testing features: %d pilotos, media=%.3fs, total_laps=%d",
        len(agg),
        agg["testing_long_run_s"].mean(),
        agg["testing_laps_n"].sum(),
    )
    return agg


def _load_existing_features(season: int) -> pd.DataFrame | None:
    path = _PROJECT_ROOT / get_settings().paths.data_processed / f"pace_features_{season}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    season = args.season
    log.info("=== Build features temporada %d ===", season)

    # 0. Features de test de pretemporada — se guardan en testing_features_{season}.parquet
    if args.round_ is None:
        testing_feats = _build_testing_features(season)
        if testing_feats is not None:
            out_dir = _PROJECT_ROOT / get_settings().paths.data_processed
            out_dir.mkdir(parents=True, exist_ok=True)
            out_test = out_dir / f"testing_features_{season}.parquet"
            testing_feats.to_parquet(out_test, index=False)
            log.info("Testing features guardadas: %s", out_test)
        else:
            log.info("Testing features: no disponibles (ejecuta 01_ingest.py primero)")

    schedule = get_schedule(season)
    today = date.today()
    completed = [r for r in schedule if r.date <= today]
    if args.round_ is not None:
        completed = [r for r in completed if r.round == args.round_]

    if not completed:
        log.warning("No hay rondas completadas — nada que procesar")
        return

    fmt_map = {
        r.round: ("sprint_qualifying" if r.has_sprint else "conventional")
        for r in schedule
    }

    # Cargar features ya calculadas — se usan al final para evitar sobreescribir
    # rondas existentes cuando se ejecuta con --round N.
    existing_features = _load_existing_features(season)
    rounds_to_process = {r.round for r in completed}

    # Lookup de equipo por piloto de rondas ya procesadas.
    # Se usa para rellenar team_name cuando FastF1 lo devuelve vacío en una sesión concreta.
    team_name_fallback: dict[str, str] = {}
    if existing_features is not None and not existing_features.empty:
        valid = existing_features.dropna(subset=["team_name"])
        valid = valid[valid["team_name"].astype(str).str.strip() != ""]
        if not valid.empty:
            team_name_fallback = (
                valid.sort_values("round")
                .drop_duplicates("abbreviation", keep="last")
                .set_index("abbreviation")["team_name"]
                .to_dict()
            )

    all_features: list[pd.DataFrame] = []

    for r in completed:
        fmt = fmt_map[r.round]
        log.info("R%02d %s (%s)...", r.round, r.race_name, fmt)

        # Vueltas de práctica (FP2 o FP1 según formato)
        try:
            practice_laps = get_pace_laps(season, r.round, fmt, accurate_only=True)
            log.info("  práctica: %d vueltas", len(practice_laps))
        except Exception as exc:
            log.warning("  práctica no disponible: %s — saltando ronda", exc)
            continue

        # Vueltas de clasificación (Q) — opcional, añade quali_pace_s real
        quali_laps = _load_quali_laps(season, r.round)
        if quali_laps is not None:
            log.info("  Q: %d vueltas", len(quali_laps))

        # Features de pace
        try:
            pace = compute_pace_features(practice_laps, quali_laps)
            log.info("  pace features: %d pilotos", len(pace))
        except Exception as exc:
            log.warning("  compute_pace_features falló: %s — saltando ronda", exc)
            continue

        if pace.empty:
            log.warning("  sin features de pace para R%02d", r.round)
            continue

        # Rellenar team_name vacíos antes de calcular gaps — FastF1 devuelve
        # Team="" en algunas sesiones; copiar de la ronda más reciente con valor válido.
        empty_mask = pace["team_name"].isna() | (pace["team_name"].astype(str).str.strip() == "")
        if empty_mask.any():
            pace.loc[empty_mask, "team_name"] = (
                pace.loc[empty_mask, "abbreviation"].map(team_name_fallback)
            )
            n_filled = empty_mask.sum() - pace["team_name"].isna().sum()
            log.warning(
                "  R%02d: team_name vacío en FastF1 — rellenado desde rondas anteriores (%d/%d pilotos)",
                r.round, n_filled, empty_mask.sum(),
            )
        # Actualizar el fallback con los valores válidos de esta ronda
        valid_now = pace.dropna(subset=["team_name"])
        valid_now = valid_now[valid_now["team_name"].astype(str).str.strip() != ""]
        team_name_fallback.update(
            valid_now.set_index("abbreviation")["team_name"].to_dict()
        )

        # Gaps entre compañeros de equipo
        try:
            gaps = compute_teammate_gaps(pace)
            pace = pace.merge(gaps, on=["abbreviation", "round"], how="left")
            log.info(
                "  gaps calculados: race=%.0f%% válidos, quali=%.0f%% válidos",
                pace["gap_to_teammate_race"].notna().mean() * 100,
                pace["gap_to_teammate_quali"].notna().mean() * 100,
            )
        except Exception as exc:
            log.warning("  compute_teammate_gaps falló: %s", exc)
            pace["gap_to_teammate_race"]  = float("nan")
            pace["gap_to_teammate_quali"] = float("nan")

        all_features.append(pace)

    if not all_features:
        log.warning("No se generaron features — revisa que 01_ingest.py se ejecutó")
        return

    # Combinar nuevas features con las existentes de otras rondas.
    # Las rondas procesadas en esta ejecución reemplazan cualquier dato previo.
    parts: list[pd.DataFrame] = []
    if existing_features is not None and not existing_features.empty:
        kept = existing_features[~existing_features["round"].isin(rounds_to_process)]
        if not kept.empty:
            parts.append(kept)
    parts.extend(all_features)

    combined = pd.concat(parts, ignore_index=True).sort_values(
        ["round", "abbreviation"]
    ).reset_index(drop=True)

    # FastF1 devuelve Team="" en algunas sesiones (e.g. Canadá 2026).
    # El equipo de cada piloto es estable durante la temporada, así que
    # propagamos el valor de la ronda más cercana con team_name válido.
    combined["team_name"] = combined["team_name"].replace("", pd.NA)
    combined["team_name"] = (
        combined.sort_values("round")
        .groupby("abbreviation")["team_name"]
        .transform(lambda s: s.ffill().bfill())
    )
    n_fixed = (combined["team_name"].notna() & (combined["team_name"] != "")).sum()
    n_missing = combined["team_name"].isna().sum()
    if n_missing:
        log.warning("team_name vacío para %d filas (ninguna ronda tiene valor válido para ese piloto)", n_missing)
    else:
        log.info("team_name: %d filas con valor válido (ffill/bfill aplicado donde era vacío)", n_fixed)

    out_dir = _PROJECT_ROOT / get_settings().paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pace_features_{season}.parquet"
    combined.to_parquet(out, index=False)
    log.info("Features guardadas: %s (%d filas, %d rondas)", out, len(combined),
             combined["round"].nunique())

    # Resumen de cobertura
    log.info(
        "Cobertura: long_run=%.0f%%, quali=%.0f%%, gaps_race=%.0f%%, gaps_quali=%.0f%%",
        combined["long_run_pace_s"].notna().mean() * 100,
        combined["quali_pace_s"].notna().mean() * 100,
        combined["gap_to_teammate_race"].notna().mean() * 100,
        combined["gap_to_teammate_quali"].notna().mean() * 100,
    )
    log.info("=== Build features completado ===")


if __name__ == "__main__":
    main()
