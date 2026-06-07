#!/usr/bin/env python3
"""
Simulación Monte Carlo del campeonato.

Qué hace:
  1. Carga las estimaciones de fuerza de data/processed/strength_estimates_{season}_r{N}.parquet.
  2. Extrae la fuerza de los pilotos (entity_type == "driver") y la pasa a simulate_championship.
  3. Carga los standings actuales (pilotos y constructores) desde Jolpica.
  4. Deriva el mapa driver→constructor desde los resultados de la última ronda disponible.
  5. Filtra las rondas restantes (round > as_of_round) del calendario.
  6. Ejecuta N simulaciones Monte Carlo via season_sim.simulate_championship().
  7. Guarda los resultados en outputs/championship/{season}/as_of_r{N}/.

Prerequisito: 04_fit_models.py — strength_estimates en data/processed/.

Uso:
    python pipelines/05_simulate_championship.py --as-of-round 8
    python pipelines/05_simulate_championship.py --as-of-round 8 --n-sims 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from gridcast.config import get_settings
from gridcast.ingest.jolpica_client import (
    get_constructor_standings,
    get_driver_standings,
    get_race_results,
    get_schedule,
)
from gridcast.simulation.season_sim import simulate_championship

log = logging.getLogger("gridcast.simulate")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Simulación del campeonato — pipeline 05")
    p.add_argument("--season",      type=int, default=cfg.season.active)
    p.add_argument("--as-of-round", type=int, required=True, metavar="N",
                   help="Simular con datos del modelo hasta R{N}; rondas restantes = round > N")
    p.add_argument("--n-sims",      type=int, default=None,
                   help="Número de simulaciones Monte Carlo (default: settings.simulation.n_sims)")
    p.add_argument("--seed",        type=int, default=None,
                   help="Semilla aleatoria (default: settings.simulation.seed)")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_strength_estimates(season: int, as_of_round: int) -> pd.DataFrame:
    path = (
        _PROJECT_ROOT
        / get_settings().paths.data_processed
        / f"strength_estimates_{season}_r{as_of_round:02d}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Estimaciones no encontradas: {path}\n"
            f"Ejecuta primero: python pipelines/04_fit_models.py --as-of-round {as_of_round}"
        )
    return pd.read_parquet(path)


def _abbrev_to_jolpica(full_name: str, jolpica_ids: set[str]) -> str | None:
    """Mapea un full_name de FastF1 al driver_id de Jolpica (3-step fuzzy match)."""
    last = str(full_name).split()[-1].lower()
    if last in jolpica_ids:
        return last
    candidates = [d for d in jolpica_ids if d.endswith("_" + last)]
    if len(candidates) == 1:
        return candidates[0]
    full_norm = "_".join(str(full_name).lower().split())
    if full_norm in jolpica_ids:
        return full_norm
    return None


def _build_driver_strength(
    estimates: pd.DataFrame,
    season: int,
    jolpica_ids: set[str],
) -> pd.DataFrame:
    """
    Combina fortaleza de piloto (within-team) + fortaleza de equipo (absoluta)
    y mapea las abreviaturas FastF1 a Jolpica driver_ids.

    Input: (as_of_round, entity_type, entity_id, mean, sd) — formato largo de pipeline 04
    Output: (driver_id, mean, sd)  con driver_id en Jolpica IDs
    """
    cfg = get_settings()

    # Cargar driver_session_map para abbrev → full_name + team_name
    dsm_path = _PROJECT_ROOT / cfg.paths.data_interim / f"driver_session_map_{season}.parquet"
    if dsm_path.exists():
        dsm = pd.read_parquet(dsm_path)
        latest = dsm.sort_values("round").groupby("abbreviation").last().reset_index()
        abbrev_full = dict(zip(latest["abbreviation"], latest.get("full_name", latest["abbreviation"])))
        abbrev_team = dict(zip(latest["abbreviation"], latest.get("team_name", "")))
    else:
        abbrev_full = {}
        abbrev_team = {}

    # Fortalezas de piloto y equipo
    drivers = estimates[estimates["entity_type"] == "driver"].set_index("entity_id")
    teams   = estimates[estimates["entity_type"] == "team"].set_index("entity_id")
    team_mean = teams["mean"] if not teams.empty else pd.Series(dtype=float)

    rows = []
    for abbrev, row in drivers.iterrows():
        full_name = abbrev_full.get(abbrev, abbrev)
        jolpica_id = _abbrev_to_jolpica(full_name, jolpica_ids)
        if jolpica_id is None:
            log.debug("Sin Jolpica ID para %s (%s) — omitido", abbrev, full_name)
            continue
        team = abbrev_team.get(abbrev, "")
        t_strength = float(team_mean.get(team, 0.0))
        rows.append({
            "driver_id": jolpica_id,
            "mean":      float(row["mean"]) + t_strength,
            "sd":        float(row["sd"]),
        })

    if not rows:
        log.warning("No se pudo mapear ningún piloto a Jolpica IDs")
        return pd.DataFrame(columns=["driver_id", "mean", "sd"])

    return pd.DataFrame(rows).reset_index(drop=True)


def _get_driver_constructor_map(season: int, as_of_round: int) -> dict[str, str] | None:
    """
    Construye el mapa driver_id → constructor_id desde los resultados de la ronda más reciente
    disponible (as_of_round o anterior si aún no está publicada en Jolpica).

    Necesario para que simulate_championship pueda acumular puntos de constructores.
    Devuelve None si no se pueden cargar los resultados (falla silenciosa).
    """
    for rnd in range(as_of_round, 0, -1):
        try:
            results = get_race_results(season, rnd)
            mapping = {r.driver_id: r.constructor_id for r in results if r.constructor_id}
            if mapping:
                if rnd < as_of_round:
                    log.info(
                        "driver→constructor map: R%02d aún sin resultados en Jolpica — "
                        "usando R%02d (%d pilotos)",
                        as_of_round, rnd, len(mapping),
                    )
                else:
                    log.info(
                        "driver→constructor map: %d pilotos (desde R%02d)", len(mapping), rnd
                    )
                return mapping
        except Exception as exc:
            log.warning("Error obteniendo resultados R%02d: %s", rnd, exc)
    log.warning("No se pudo obtener driver→constructor map — probabilidades de constructor no se calcularán")
    return None


def _save_results(results: dict, season: int, as_of_round: int) -> Path:
    out_dir = (
        _PROJECT_ROOT / get_settings().paths.outputs
        / "championship" / str(season) / f"as_of_r{as_of_round:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # driver_points_dist → parquet (tabla grande)
    if "driver_points_dist" in results and isinstance(results["driver_points_dist"], pd.DataFrame):
        results["driver_points_dist"].to_parquet(out_dir / "driver_points_dist.parquet", index=False)

    # Dicts de probabilidades y puntos esperados → parquet de 2 columnas
    for key in ("driver_title_probs", "constructor_title_probs", "driver_expected_points"):
        if key in results and isinstance(results[key], dict):
            pd.DataFrame(
                list(results[key].items()), columns=["entity_id", "value"]
            ).sort_values("value", ascending=False).to_parquet(
                out_dir / f"{key}.parquet", index=False
            )

    # Resumen JSON legible por humanos
    summary = {
        k: v for k, v in results.items()
        if not isinstance(v, pd.DataFrame)
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Resultados guardados en %s", out_dir)
    return out_dir


def _print_title_probs(probs: dict[str, float], label: str, top_n: int = 10) -> None:
    if not probs:
        return
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    log.info("%s (top %d):", label, top_n)
    for entity_id, prob in sorted_probs[:top_n]:
        log.info("  %-20s  %.1f%%", entity_id, prob * 100)


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

    cfg         = get_settings()
    season      = args.season
    as_of_round = args.as_of_round
    n_sims      = args.n_sims   # None → usa settings.simulation.n_sims en season_sim
    seed        = args.seed     # None → usa settings.simulation.seed en season_sim

    log.info(
        "=== Simulación campeonato %d (modelo hasta R%02d) ===",
        season, as_of_round,
    )

    # --- Fortaleza de pilotos (driver + team, mapeada a Jolpica IDs) ---
    raw_estimates = _load_strength_estimates(season, as_of_round)
    driver_st_tmp  = get_driver_standings(season, after_round=as_of_round)
    jolpica_ids    = {s.driver_id for s in driver_st_tmp}
    driver_strength = _build_driver_strength(raw_estimates, season, jolpica_ids)
    log.info("Fuerza de pilotos: %d pilotos mapeados a Jolpica IDs", len(driver_strength))

    if driver_strength.empty:
        log.error(
            "No hay estimaciones de fuerza para pilotos. "
            "Comprueba la salida de pipeline 04."
        )
        sys.exit(1)

    # --- Calendario y rondas restantes ---
    schedule = get_schedule(season)
    # Rondas restantes = las que no están cubiertas por el modelo (round > as_of_round)
    remaining = [r for r in schedule if r.round > as_of_round]
    log.info(
        "Rondas restantes: %d (rounds %s)",
        len(remaining),
        [r.round for r in remaining],
    )

    # --- Standings actuales (driver_st_tmp ya fue cargado arriba) ---
    driver_st      = driver_st_tmp
    constructor_st = get_constructor_standings(season, after_round=as_of_round)
    log.info(
        "Standings: %d pilotos, %d constructores",
        len(driver_st), len(constructor_st),
    )

    # --- Mapa driver → constructor (para puntos de constructor) ---
    driver_constructor_map = _get_driver_constructor_map(season, as_of_round)

    # --- Simulación ---
    log.info(
        "Iniciando simulación: %s sims, seed=%s",
        n_sims or cfg.simulation.n_sims,
        seed or cfg.simulation.seed,
    )

    results = simulate_championship(
        strength=driver_strength,
        remaining_rounds=remaining,
        driver_standings=driver_st,
        constructor_standings=constructor_st,
        driver_constructor_map=driver_constructor_map,
        n_sims=n_sims,
        seed=seed,
        # car_development_sigma leído de settings.simulation.car_development_sigma
    )

    # --- Mostrar y guardar ---
    _print_title_probs(results["driver_title_probs"],      "Probabilidades de título (pilotos)")
    _print_title_probs(results["constructor_title_probs"], "Probabilidades de título (constructores)")

    out_dir = _save_results(results, season, as_of_round)
    log.info("=== Simulación completada → %s ===", out_dir)


if __name__ == "__main__":
    main()
