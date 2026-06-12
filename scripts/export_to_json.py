#!/usr/bin/env python3
"""
Exporta los outputs de GridCast a JSON para GitHub Pages.

Lee los parquets de outputs/ y los convierte a JSON legibles por
el dashboard estático en docs/data/. Se ejecuta como último paso
del workflow de GitHub Actions.

Qué exporta:
  - Probabilidades de título (pilotos y constructores), latest round
  - Puntos esperados por piloto
  - Métricas de evaluación (accuracy y calibración)
  - Resumen del calendario con rondas completadas
  - manifest.json con metadatos para que el dashboard sepa qué hay disponible

Uso:
    python scripts/export_to_json.py
    python scripts/export_to_json.py --season 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from gridcast.config import get_settings
from gridcast.ingest.jolpica_client import get_schedule

log = logging.getLogger("gridcast.export")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DATA = _PROJECT_ROOT / "docs" / "data"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Export outputs a JSON para GitHub Pages")
    p.add_argument("--season", type=int, default=cfg.season.active)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_parquet_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        log.warning("No se pudo leer %s: %s", path, exc)
        return None


def _df_to_records(df: pd.DataFrame | None) -> list[dict] | None:
    if df is None or df.empty:
        return None
    # Convertir fechas/timestamps a str para JSON
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]", "object"]).columns:
        try:
            df[col] = df[col].astype(str)
        except Exception:
            pass
    return df.to_dict(orient="records")


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("Exportado: %s", path)


def _latest_championship_round(season: int) -> int | None:
    """Devuelve el número de ronda de la simulación de campeonato más reciente."""
    cfg = get_settings()
    champ_dir = _PROJECT_ROOT / cfg.paths.outputs / "championship" / str(season)
    if not champ_dir.exists():
        return None
    dirs = sorted(
        [d for d in champ_dir.iterdir() if d.is_dir() and d.name.startswith("as_of_r")],
        key=lambda d: d.name,
    )
    if not dirs:
        return None
    latest = dirs[-1].name  # "as_of_r08"
    try:
        return int(latest.split("r")[1])
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Exportadores por tipo de dato
# ---------------------------------------------------------------------------

# Puntos F1 2026 (sin vuelta rápida)
_POINTS = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

# Normalización de nombre de equipo FastF1 → team_id del dashboard
_TEAM_ID = {
    "mclaren": "mclaren",
    "red bull racing": "red_bull",
    "ferrari": "ferrari",
    "mercedes": "mercedes",
    "aston martin": "aston_martin",
    "williams": "williams",
    "racing bulls": "rb",
    "rb": "rb",
    "haas f1 team": "haas",
    "haas": "haas",
    "alpine": "alpine",
    "audi": "audi",
    "cadillac": "cadillac",
    "kick sauber": "audi",
}


def _normalize_team(name: str) -> str:
    return _TEAM_ID.get(name.lower().strip(), name.lower().replace(" ", "_"))


def _build_driver_meta(season: int) -> dict[str, dict]:
    """
    Devuelve {abbreviation: {display_name, team, jolpica_id}} usando
    la última aparición de cada piloto en el driver_session_map.
    """
    cfg = get_settings()
    dsm_path = _PROJECT_ROOT / cfg.paths.data_interim / f"driver_session_map_{season}.parquet"
    if not dsm_path.exists():
        return {}
    dsm = pd.read_parquet(dsm_path)
    # Tomar la aparición más reciente por piloto
    latest = dsm.sort_values("round").groupby("abbreviation").last().reset_index()

    meta: dict[str, dict] = {}
    for _, row in latest.iterrows():
        abbrev    = row["abbreviation"]
        full_name = row.get("full_name", abbrev)
        team_name = row.get("team_name", "")
        # display_name = apellido
        parts = str(full_name).split()
        display_name = parts[-1] if parts else abbrev
        meta[abbrev] = {
            "display_name": display_name,
            "full_name": full_name,
            "team": _normalize_team(str(team_name)),
        }
    return meta


def _build_results_lookup(season: int, completed_rounds: list[int]) -> dict[int, dict[str, int]]:
    """
    Devuelve {round: {jolpica_driver_id: finish_position}} para todas las rondas completadas.

    Fuente principal: Jolpica (resultados definitivos, IDs ya en formato Jolpica).
    Fallback: snapshots (para el caso en que Jolpica no esté disponible).
    Los snapshots son point-in-time y pueden no tener la última ronda completada
    (la ronda N solo aparece en el snapshot de N+1), por eso Jolpica es preferible.
    """
    from gridcast.ingest.jolpica_client import get_race_results

    lookup: dict[int, dict[str, int]] = {}

    for r in completed_rounds:
        try:
            results = get_race_results(season, r)
            for res in results:
                if res.position is not None:
                    lookup.setdefault(r, {})[res.driver_id] = res.position
            if r in lookup:
                log.debug("R%02d: %d resultados desde Jolpica", r, len(lookup[r]))
        except Exception as exc:
            log.warning("R%02d: no se pudo obtener resultados de Jolpica: %s", r, exc)

    # Fallback: snapshots para rondas sin datos de Jolpica
    cfg = get_settings()
    snap_base = _PROJECT_ROOT / cfg.paths.snapshots / str(season) / cfg.backtest.snapshot_version
    if snap_base.exists():
        all_snap: list[pd.DataFrame] = []
        for snap_dir in sorted(snap_base.iterdir()):
            rr_path = snap_dir / "race_results.parquet"
            if rr_path.exists():
                try:
                    all_snap.append(pd.read_parquet(rr_path))
                except Exception:
                    pass
        if all_snap:
            combined = pd.concat(all_snap, ignore_index=True)
            combined = combined.dropna(subset=["position"])
            combined = combined.sort_values("round").drop_duplicates(
                subset=["round", "driver_id"], keep="last"
            )
            for _, row in combined.iterrows():
                r = int(row["round"])
                if r not in lookup:
                    lookup.setdefault(r, {})[str(row["driver_id"])] = int(row["position"])

    return lookup


def _abbrev_to_jolpica(full_name: str, jolpica_ids: set[str]) -> str | None:
    """Intenta encontrar el driver_id de jolpica para un full_name de FastF1."""
    last = str(full_name).split()[-1].lower()
    # Primero intento: coincidencia exacta del apellido
    if last in jolpica_ids:
        return last
    # Segundo intento: termina en _apellido (ej. max_verstappen)
    candidates = [d for d in jolpica_ids if d.endswith("_" + last)]
    if len(candidates) == 1:
        return candidates[0]
    # Tercer intento: nombre_apellido completo
    full_norm = "_".join(str(full_name).lower().split())
    if full_norm in jolpica_ids:
        return full_norm
    return None


def _finish_probs_to_driver_rows(
    finish_probs: pd.DataFrame,
    driver_meta: dict[str, dict],
    actual_positions: dict[str, int],   # jolpica_id → position
    jolpica_ids: set[str],
) -> list[dict]:
    """
    Convierte finish_probs (driver_id=abbrev, finish_pos, probability) a la
    lista de pilotos del dashboard con probabilidades y posición real.
    """
    # Agregar por piloto
    rows: list[dict] = []
    for abbrev, grp in finish_probs.groupby("driver_id"):
        prob_by_pos = grp.set_index("finish_pos")["probability"]
        win_prob    = float(prob_by_pos.get(1, 0.0))
        podium_prob = float(sum(prob_by_pos.get(p, 0.0) for p in [1, 2, 3]))
        points_prob = float(sum(prob_by_pos.get(p, 0.0) for p in range(1, 11)))
        exp_pts     = float(sum(
            prob_by_pos.get(pos, 0.0) * pts
            for pos, pts in _POINTS.items()
        ))

        meta = driver_meta.get(abbrev, {"display_name": abbrev, "team": "unknown"})

        # Buscar posición real
        actual_pos = None
        jolpica_id = _abbrev_to_jolpica(meta.get("full_name", abbrev), jolpica_ids)
        if jolpica_id and jolpica_id in actual_positions:
            actual_pos = actual_positions[jolpica_id]

        rows.append({
            "driver_id":       abbrev,
            "display_name":    meta["display_name"],
            "team":            meta["team"],
            "win_prob":        round(win_prob, 4),
            "podium_prob":     round(podium_prob, 4),
            "points_prob":     round(points_prob, 4),
            "expected_points": round(exp_pts, 2),
            "actual_position": actual_pos,
        })

    # Ordenar por probabilidad de victoria descendente
    rows.sort(key=lambda r: r["win_prob"], reverse=True)
    return rows


def _dir_to_cutoff_name(dir_suffix: str) -> str:
    """Mapea sufijo de directorio a nombre canónico de cutoff. 'pre' → 'pre-fp1'."""
    return "pre-fp1" if dir_suffix == "pre" else dir_suffix


def _read_prediction_dir(
    snap_dir: Path,
    driver_meta: dict[str, dict],
    actual_positions: dict[str, int],
    jolpica_ids: set[str],
) -> tuple[str, dict] | None:
    """
    Lee un directorio de predicción (round_NN_<suffix>/) y devuelve
    (cutoff_name, cutoff_payload) o None si hay error.
    """
    parts = snap_dir.name.split("_", 2)  # ["round", "NN", "pre"] o ["round", "NN", "post-q"]
    if len(parts) < 3:
        return None
    dir_suffix = parts[2]
    cutoff_name = _dir_to_cutoff_name(dir_suffix)

    fp_path = snap_dir / "finish_probs.parquet"
    if not fp_path.exists():
        return None

    try:
        finish_probs = pd.read_parquet(fp_path)
    except Exception as exc:
        log.warning("%s: no se pudo leer finish_probs: %s", snap_dir.name, exc)
        return None

    driver_rows = _finish_probs_to_driver_rows(
        finish_probs, driver_meta, actual_positions, jolpica_ids
    )

    # Fecha de generación desde metadata.json del directorio de predicción
    meta_path = snap_dir / "metadata.json"
    generated_at = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            generated_at = meta.get("cutoff_date") or meta.get("generated_at", "")
        except Exception:
            pass

    return cutoff_name, {
        "generated_at": generated_at or "",
        "drivers": driver_rows,
    }


def _build_data_warnings(season: int) -> list[dict]:
    """
    Lee pace_features_{season}.parquet y devuelve los registros sospechosos.

    Cada warning tiene: round, driver, signal ("long_run" | "quali"), value.
    Devuelve [] si no hay flags o el parquet no existe.
    """
    cfg = get_settings()
    path = _PROJECT_ROOT / cfg.paths.data_processed / f"pace_features_{season}.parquet"
    if not path.exists():
        return []

    try:
        pf = pd.read_parquet(path)
    except Exception as exc:
        log.warning("No se pudo leer pace_features para warnings: %s", exc)
        return []

    warnings: list[dict] = []
    signal_map = {
        "long_run_suspect": ("long_run", "long_run_pace_s"),
        "quali_suspect":    ("quali",    "quali_pace_s"),
    }

    for flag_col, (signal_label, value_col) in signal_map.items():
        if flag_col not in pf.columns:
            continue
        flagged = pf[pf[flag_col].fillna(False).astype(bool)].copy()
        for _, row in flagged.iterrows():
            val = row.get(value_col)
            warnings.append({
                "round":   int(row["round"]),
                "driver":  str(row["abbreviation"]),
                "signal":  signal_label,
                "value_s": round(float(val), 3) if pd.notna(val) else None,
            })

    warnings.sort(key=lambda w: (w["round"], w["driver"]))
    if warnings:
        log.info("data_warnings: %d entradas (datos sospechosos)", len(warnings))
    return warnings


def export_predictions(season: int, schedule_data: dict) -> dict:
    """
    Exporta predicciones desde outputs/predictions/ al formato multi-cutoff del dashboard.

    Lee directorios round_NN_pre/, round_NN_post-q/, round_NN_post-sprint/
    y construye el dict de cutoffs para cada ronda.
    """
    cfg = get_settings()
    pred_base = (
        _PROJECT_ROOT / cfg.paths.outputs / "predictions"
        / str(season) / cfg.backtest.snapshot_version
    )

    driver_meta     = _build_driver_meta(season)
    completed_rounds = [r["round"] for r in schedule_data.get("rounds", []) if r.get("completed")]
    results_lookup  = _build_results_lookup(season, completed_rounds)
    jolpica_ids     = {jid for rd in results_lookup.values() for jid in rd}

    sched_by_round = {r["round"]: r for r in schedule_data.get("rounds", [])}

    # Agrupar directorios por número de ronda
    round_dirs: dict[int, list[Path]] = {}
    if pred_base.exists():
        for d in sorted(pred_base.iterdir()):
            if not d.is_dir() or not d.name.startswith("round_"):
                continue
            parts = d.name.split("_", 2)
            if len(parts) < 3:
                continue
            try:
                r = int(parts[1])
            except ValueError:
                continue
            round_dirs.setdefault(r, []).append(d)

    predictions: list[dict] = []

    for round_, dirs in sorted(round_dirs.items()):
        actual_positions = results_lookup.get(round_, {})
        actual_winner    = None
        if actual_positions:
            winner_jid = next((jid for jid, pos in actual_positions.items() if pos == 1), None)
            if winner_jid:
                actual_winner = winner_jid

        # Construir cutoffs disponibles para esta ronda
        cutoffs: dict[str, dict] = {}
        for d in dirs:
            result = _read_prediction_dir(d, driver_meta, actual_positions, jolpica_ids)
            if result is not None:
                cutoff_name, payload = result
                cutoffs[cutoff_name] = payload

        if not cutoffs:
            continue

        sched = sched_by_round.get(round_, {})
        predictions.append({
            "round":         round_,
            "race_name":     sched.get("race_name", f"Round {round_}"),
            "circuit":       sched.get("circuit_name", ""),
            "race_date":     sched.get("date", ""),
            "has_sprint":    sched.get("has_sprint", False),
            "completed":     sched.get("completed", False),
            "actual_winner": actual_winner,
            "cutoffs":       cutoffs,
        })

        log.info("R%02d: cutoffs exportados: %s", round_, sorted(cutoffs.keys()))

    data_warnings = _build_data_warnings(season)

    data = {
        "season": season,
        "data_warnings": data_warnings,
        "predictions": sorted(predictions, key=lambda p: p["round"]),
    }
    _write_json(data, _DOCS_DATA / f"predictions_{season}.json")
    return data


def _jolpica_id_to_display(jolpica_id: str) -> str:
    """Fallback: 'max_verstappen' → 'Verstappen', 'antonelli' → 'Antonelli'."""
    return jolpica_id.split("_")[-1].capitalize()


def _build_jolpica_meta(season: int, jolpica_ids: list[str]) -> dict[str, dict]:
    """
    Mapea jolpica_id → {display_name, team} usando driver_session_map como fuente.
    Fallback: capitaliza el apellido del ID de Jolpica.
    """
    driver_meta = _build_driver_meta(season)
    jolpica_set = set(jolpica_ids)
    result: dict[str, dict] = {}
    for abbrev, meta in driver_meta.items():
        jid = _abbrev_to_jolpica(meta["full_name"], jolpica_set)
        if jid:
            result[jid] = {"display_name": meta["display_name"], "team": meta["team"]}
    for jid in jolpica_ids:
        if jid not in result:
            result[jid] = {
                "display_name": _jolpica_id_to_display(jid),
                "team": "unknown",
            }
    return result


def export_championship(season: int) -> dict | None:
    """Exporta la simulación de campeonato más reciente."""
    round_ = _latest_championship_round(season)
    if round_ is None:
        log.info("Sin simulaciones de campeonato disponibles")
        return None

    cfg = get_settings()
    champ_dir = (
        _PROJECT_ROOT / cfg.paths.outputs / "championship"
        / str(season) / f"as_of_r{round_:02d}"
    )

    driver_probs = _read_parquet_safe(champ_dir / "driver_title_probs.parquet")
    constr_probs = _read_parquet_safe(champ_dir / "constructor_title_probs.parquet")
    driver_pts   = _read_parquet_safe(champ_dir / "driver_expected_points.parquet")

    # Enriquecer registros de pilotos con display_name y team
    driver_probs_records = None
    if driver_probs is not None:
        jolpica_ids = driver_probs["entity_id"].tolist()
        jmeta = _build_jolpica_meta(season, jolpica_ids)
        driver_probs_records = []
        for rec in _df_to_records(driver_probs) or []:
            m = jmeta.get(rec["entity_id"], {})
            rec["display_name"] = m.get("display_name", _jolpica_id_to_display(rec["entity_id"]))
            rec["team"] = m.get("team", "unknown")
            driver_probs_records.append(rec)

    data = {
        "as_of_round": round_,
        "driver_title_probs":       driver_probs_records,
        "constructor_title_probs":  _df_to_records(constr_probs),
        "driver_expected_points":   _df_to_records(driver_pts),
    }

    _write_json(data, _DOCS_DATA / f"championship_{season}.json")
    return data


def export_evaluation(season: int) -> dict | None:
    """Exporta las métricas de evaluación (accuracy + calibración)."""
    cfg = get_settings()
    version = cfg.backtest.snapshot_version
    reports_dir = _PROJECT_ROOT / cfg.paths.outputs / "reports" / str(season)

    accuracy  = _read_parquet_safe(reports_dir / f"accuracy_{version}.parquet")
    per_round = _read_parquet_safe(reports_dir / f"per_round_{version}.parquet")
    reliability = _read_parquet_safe(reports_dir / f"reliability_{version}.parquet")

    # Resumen escalar desde JSON si existe
    summary_path = reports_dir / f"summary_{version}.json"
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            log.warning("No se pudo leer summary.json: %s", exc)

    if accuracy is None and summary is None:
        log.info("Sin datos de evaluación disponibles")
        return None

    data = {
        "version": version,
        "summary": summary,
        "accuracy_by_round": _df_to_records(accuracy),
        "calibration_by_round": _df_to_records(per_round),
        "reliability_diagram": _df_to_records(reliability),
    }

    _write_json(data, _DOCS_DATA / f"evaluation_{season}.json")
    return data


def export_schedule(season: int) -> dict:
    """Exporta el calendario con estado de cada ronda (completada / pendiente)."""
    schedule = get_schedule(season)
    today = date.today()

    rounds = []
    for r in schedule:
        rounds.append({
            "round":        r.round,
            "race_name":    r.race_name,
            "circuit_id":   r.jolpica_circuit_id,
            "circuit_name": r.circuit_name,
            "date":         r.date.isoformat(),
            "has_sprint":   r.has_sprint,
            "completed":    r.date < today,
        })

    data = {
        "season": season,
        "n_rounds": len(rounds),
        "rounds": rounds,
    }
    _write_json(data, _DOCS_DATA / f"schedule_{season}.json")
    return data


def export_manifest(season: int, available: dict) -> None:
    """Escribe manifest.json con qué datos están disponibles para el dashboard."""
    manifest = {
        "season":            season,
        "updated_at":        date.today().isoformat(),
        "has_championship":  available.get("championship") is not None,
        "has_evaluation":    available.get("evaluation") is not None,
        "championship_round": (
            available["championship"].get("as_of_round")
            if available.get("championship") else None
        ),
    }
    _write_json(manifest, _DOCS_DATA / "manifest.json")


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

    season = args.season
    log.info("=== Export a JSON para GitHub Pages (temporada %d) ===", season)

    available: dict = {}
    available["schedule"]     = export_schedule(season)
    available["predictions"]  = export_predictions(season, available["schedule"])
    available["championship"] = export_championship(season)
    available["evaluation"]   = export_evaluation(season)
    export_manifest(season, available)

    exported = sum(1 for v in available.values() if v is not None)
    log.info("=== Export completado: %d datasets exportados ===", exported)


if __name__ == "__main__":
    main()
