"""
Agregación de métricas de calibración para informes y export.

Complementa calibration.py (que opera vuelta a vuelta y produce scores individuales)
con funciones de resumen multi-ronda orientadas a reporting y el pipeline de export.

Interfaz principal:
    summarize(report)   → dict con métricas escalares (para JSON / GitHub Pages)
    per_round_df(report) → DataFrame con una fila por ronda (para parquet)
"""

from __future__ import annotations

import math

import pandas as pd


def summarize(report: dict) -> dict:
    """
    Convierte la salida de evaluate_predictions() en un dict de métricas escalares.

    Todos los valores son serializables a JSON (float, int, str, None).
    Los NaN se convierten a None para compatibilidad con JSON.

    Args:
        report: Salida de evaluate_predictions() — ver calibration.py.

    Returns:
        Dict con claves: brier_win, brier_top3, log_loss_win, rank_correlation,
        ece_win, n_rounds.
    """
    def _clean(v: object) -> object:
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, float):
            return round(v, 6)
        return v

    keys = ["brier_win", "brier_top3", "log_loss_win", "rank_correlation", "ece_win", "n_rounds"]
    return {k: _clean(report.get(k)) for k in keys}


def per_round_df(report: dict) -> pd.DataFrame:
    """
    Extrae las métricas por ronda del report de evaluate_predictions().

    Returns:
        DataFrame con columnas: round, brier_win, log_loss_win, rank_correlation.
        Vacío si el report no contiene 'per_round'.
    """
    per = report.get("per_round")
    if per is None:
        return pd.DataFrame(
            columns=["round", "brier_win", "log_loss_win", "rank_correlation"]
        )
    if isinstance(per, pd.DataFrame):
        return per.reset_index(drop=True)
    return pd.DataFrame(per)


def naif_brier_win(n_drivers: int = 20) -> float:
    """
    Brier score de un modelo uniforme (baseline de referencia).
    BS_naif ≈ (n-1)/n² — para 20 pilotos ≈ 0.0475.
    """
    return (n_drivers - 1) / (n_drivers ** 2)


def skill_score(brier: float, n_drivers: int = 20) -> float:
    """
    Brier Skill Score respecto al baseline uniforme.
    BSS = 1 - BS / BS_ref. Positivo si el modelo bate al uniforme.
    """
    ref = naif_brier_win(n_drivers)
    if ref == 0:
        return 0.0
    return 1.0 - brier / ref
