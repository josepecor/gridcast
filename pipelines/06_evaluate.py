#!/usr/bin/env python3
"""
Evaluación de predicciones congeladas contra resultados reales.

Qué hace:
  1. Carga todas las predicciones congeladas de outputs/predictions/.
  2. Carga los resultados reales de cada ronda desde Jolpica.
  3. Calcula métricas de calibración completas via evaluate.calibration:
       - Brier score (win, top-3, full)
       - Log-loss (win)
       - Rank correlation (Spearman)
       - ECE + reliability diagram
  4. Calcula accuracy básica por ronda (top-1, top-3).
  5. Guarda informes en outputs/reports/{season}/.

Prerequisito: 03_freeze_history.py con predicciones generadas.

Uso:
    python pipelines/06_evaluate.py
    python pipelines/06_evaluate.py --season 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from gridcast.backtest.replay import RoundPrediction, list_predictions, load_prediction
from gridcast.config import get_settings
from gridcast.evaluate.calibration import evaluate_predictions
from gridcast.evaluate.metrics import per_round_df, summarize
from gridcast.ingest.jolpica_client import get_race_results, get_schedule

log = logging.getLogger("gridcast.evaluate")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Evaluación de predicciones — pipeline 06")
    p.add_argument("--season", type=int, default=cfg.season.active)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top1_driver(finish_probs: pd.DataFrame) -> str | None:
    """Piloto con mayor probabilidad de victoria (position==1 o columna win_prob)."""
    if "driver_id" not in finish_probs.columns:
        return None

    # Formato canónico: (driver_id, finish_pos, probability)
    if "finish_pos" in finish_probs.columns and "probability" in finish_probs.columns:
        win_rows = finish_probs[finish_probs["finish_pos"] == 1]
        if win_rows.empty:
            return None
        return win_rows.sort_values("probability", ascending=False).iloc[0]["driver_id"]

    # Formato compacto: columna de score/probabilidad por piloto
    prob_col = next(
        (c for c in ["win_prob", "probability", "score", "mean"] if c in finish_probs.columns),
        None,
    )
    if prob_col is None:
        return None
    return finish_probs.sort_values(prob_col, ascending=False).iloc[0]["driver_id"]


def _topk_drivers(finish_probs: pd.DataFrame, k: int) -> list[str]:
    """Top-k pilotos por probabilidad de victoria."""
    if "driver_id" not in finish_probs.columns:
        return []

    if "finish_pos" in finish_probs.columns and "probability" in finish_probs.columns:
        win_rows = finish_probs[finish_probs["finish_pos"] == 1]
        if win_rows.empty:
            return []
        return (
            win_rows.sort_values("probability", ascending=False)
            .head(k)["driver_id"]
            .tolist()
        )

    prob_col = next(
        (c for c in ["win_prob", "probability", "score", "mean"] if c in finish_probs.columns),
        None,
    )
    if prob_col is None:
        return []
    return finish_probs.sort_values(prob_col, ascending=False).head(k)["driver_id"].tolist()


def _accuracy_rows(
    predictions: list[RoundPrediction],
    results_by_round: dict[int, list],
    round_date: dict[int, object],
) -> list[dict]:
    rows = []
    for pred in predictions:
        round_ = pred.round_
        actual = results_by_round.get(round_, [])
        winner = next((r.driver_id for r in actual if r.position == 1), None)
        if winner is None:
            log.warning("R%02d: ganador no encontrado en resultados — saltando", round_)
            continue

        top1 = _top1_driver(pred.finish_probs)
        top3 = _topk_drivers(pred.finish_probs, 3)
        hit1 = top1 == winner
        hit3 = winner in top3

        rows.append({
            "round":      round_,
            "date":       round_date.get(round_),
            "winner":     winner,
            "predicted_winner": top1,
            "top1_hit":   hit1,
            "top3_hit":   hit3,
            "model_version": pred.model_version,
        })
        log.info(
            "R%02d: winner=%-12s predicted=%-12s | top-1=%s top-3=%s",
            round_, winner, top1 or "?",
            "✓" if hit1 else "✗",
            "✓" if hit3 else "✗",
        )
    return rows


def _save_reports(
    season: int,
    version: str,
    accuracy_df: pd.DataFrame,
    calib_report: dict | None,
    per_round: pd.DataFrame | None,
) -> Path:
    out_dir = _PROJECT_ROOT / get_settings().paths.outputs / "reports" / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy por ronda
    accuracy_df.to_parquet(out_dir / f"accuracy_{version}.parquet", index=False)

    if calib_report is not None:
        # Reliability diagram
        if "reliability_win" in calib_report and isinstance(
            calib_report["reliability_win"], pd.DataFrame
        ):
            calib_report["reliability_win"].to_parquet(
                out_dir / f"reliability_{version}.parquet", index=False
            )

        # Métricas por ronda
        if per_round is not None and not per_round.empty:
            per_round.to_parquet(out_dir / f"per_round_{version}.parquet", index=False)

        # Resumen escalar → JSON legible
        summary = summarize(calib_report)
        summary["season"] = season
        summary["version"] = version
        (out_dir / f"summary_{version}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    log.info("Informes guardados en %s", out_dir)
    return out_dir


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

    cfg = get_settings()
    season = args.season
    version = cfg.backtest.snapshot_version
    log.info("=== Evaluación temporada %d (versión %s) ===", season, version)

    frozen_rounds = list_predictions(season, version)
    if not frozen_rounds:
        log.warning(
            "Sin predicciones congeladas para temporada %d versión %s. "
            "Ejecuta: python pipelines/03_freeze_history.py --season %d",
            season, version, season,
        )
        return

    log.info("Predicciones disponibles: rondas %s", frozen_rounds)

    schedule = get_schedule(season)
    round_date = {r.round: r.date for r in schedule}

    # Cargar predicciones y resultados reales
    predictions: list[RoundPrediction] = []
    results_by_round: dict[int, list] = {}

    for round_ in frozen_rounds:
        pred = load_prediction(season, round_, version)
        if pred is None:
            log.warning("R%02d: predicción corrupta — saltando", round_)
            continue
        predictions.append(pred)

        try:
            results_by_round[round_] = get_race_results(season, round_)
        except Exception as exc:
            log.warning("R%02d: resultados no disponibles en Jolpica: %s", round_, exc)

    if not predictions:
        log.warning("Sin predicciones válidas para evaluar")
        return

    # --- Accuracy básica ---
    acc_rows = _accuracy_rows(predictions, results_by_round, round_date)
    if not acc_rows:
        log.warning("Sin rondas evaluables (faltan resultados Jolpica)")
        return

    accuracy_df = pd.DataFrame(acc_rows)
    n = len(accuracy_df)
    log.info(
        "Accuracy: top-1=%.0f%% (%d/%d) | top-3=%.0f%% (%d/%d)",
        accuracy_df["top1_hit"].mean() * 100, accuracy_df["top1_hit"].sum(), n,
        accuracy_df["top3_hit"].mean() * 100, accuracy_df["top3_hit"].sum(), n,
    )

    # --- Calibración completa ---
    calib_report = per_round = None
    preds_with_results = [p for p in predictions if p.round_ in results_by_round]

    if preds_with_results:
        try:
            calib_report = evaluate_predictions(preds_with_results, results_by_round)
            per_round = per_round_df(calib_report)
            log.info(
                "Calibración: Brier_win=%.4f | log_loss=%.4f | rank_corr=%.3f | ECE=%.4f",
                calib_report.get("brier_win", float("nan")),
                calib_report.get("log_loss_win", float("nan")),
                calib_report.get("rank_correlation", float("nan")),
                calib_report.get("ece_win", float("nan")),
            )
        except Exception as exc:
            log.warning("Calibración completa falló: %s", exc)

    # --- Guardar ---
    out_dir = _save_reports(season, version, accuracy_df, calib_report, per_round)
    log.info("=== Evaluación completada → %s ===", out_dir)


if __name__ == "__main__":
    main()
