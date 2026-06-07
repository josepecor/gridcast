"""
Evaluación de calibración de predicciones probabilísticas de F1.

Principio rector (§1 CLAUDE.md):
  "Lo que importa no es el acierto puntual, es la calibración.
   Una probabilidad del 30% debe cumplirse ~3 de cada 10 veces."

Métricas implementadas:
  Brier score      — (p - y)²; mide calibración y resolución conjuntamente.
  Log-loss         — -log(p_correcto); penaliza más la sobreconfianza incorrecta.
  Reliability curve — bins de probabilidad predicha vs frecuencia observada.
  ECE              — Expected Calibration Error; escalar resumen de la reliability curve.
  Rank correlation — Spearman entre ranking predicho y resultado real.

Formato canónico de finish_probs (ver backtest/replay.py PredictionModel):
  driver_id  finish_pos  probability
  VER        1           0.35
  VER        2           0.20
  NOR        1           0.22
  ...
  (N_drivers × N_positions filas)
  probability suma ≈ 1 por driver_id (distribución de posición de ese piloto)
  probability suma ≈ 1 por finish_pos (exactamente un piloto por posición)

Formatos alternativos aceptados por las funciones de extracción:
  - (driver_id, win_prob)            — solo prob de victoria
  - (driver_id, score / strength)    — fortaleza; se convierte vía softmax
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos internos
# ---------------------------------------------------------------------------

# Pares (prob_predicha, resultado_binario) para reliability diagram
_ProbOutcomePair = tuple[float, int]   # (p_predicted, y_actual ∈ {0, 1})


# ---------------------------------------------------------------------------
# Extracción de probabilidades desde finish_probs
# ---------------------------------------------------------------------------

def extract_win_probs(finish_probs: pd.DataFrame) -> pd.Series:
    """
    Extrae P(ganar) por piloto desde el DataFrame canónico de predicciones.

    Formatos soportados (en orden de preferencia):
      1. (driver_id, finish_pos, probability) — canónico; filtra finish_pos == 1.
      2. (driver_id, win_prob)               — columna directa.
      3. (driver_id, score | strength)       — softmax sobre la columna.

    Returns:
        pd.Series con driver_id como índice y P(win) como valor.
        Suma ≈ 1 para el formato canónico; puede diferir si hay NaN.

    Raises:
        ValueError: si no se puede inferir ninguna probabilidad de victoria.
    """
    if finish_probs.empty:
        return pd.Series(dtype=float)

    if {"driver_id", "finish_pos", "probability"}.issubset(finish_probs.columns):
        win_df = finish_probs[finish_probs["finish_pos"] == 1][["driver_id", "probability"]]
        return win_df.set_index("driver_id")["probability"].rename("win_prob")

    if "driver_id" in finish_probs.columns and "win_prob" in finish_probs.columns:
        return finish_probs.set_index("driver_id")["win_prob"].rename("win_prob")

    for col in ("score", "strength", "mean"):
        if col in finish_probs.columns and "driver_id" in finish_probs.columns:
            log.warning(
                "finish_probs no tiene probabilidades explícitas — "
                "convirtiendo columna '%s' vía softmax (aproximación)", col
            )
            scores = finish_probs.set_index("driver_id")[col].astype(float)
            return _softmax(scores).rename("win_prob")

    raise ValueError(
        "No se puede extraer P(win) de finish_probs. "
        "Columnas disponibles: " + str(list(finish_probs.columns)) + ". "
        "Formato esperado: (driver_id, finish_pos, probability) o (driver_id, win_prob)."
    )


def extract_topk_probs(finish_probs: pd.DataFrame, k: int) -> pd.Series:
    """
    Extrae P(terminar en top-k) por piloto.

    En formato canónico: suma P(finish_pos <= k) por driver_id.
    En formato simplificado: solo usa P(win) (subestimación para k > 1).

    Returns:
        pd.Series con driver_id como índice y P(top-k) como valor.
    """
    if finish_probs.empty:
        return pd.Series(dtype=float)

    if {"driver_id", "finish_pos", "probability"}.issubset(finish_probs.columns):
        topk = finish_probs[finish_probs["finish_pos"] <= k]
        return (
            topk.groupby("driver_id")["probability"]
            .sum()
            .rename(f"top{k}_prob")
        )

    return extract_win_probs(finish_probs).rename(f"top{k}_prob")


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier_score_win(
    finish_probs: pd.DataFrame,
    actual_results: list,
) -> float:
    """
    Brier score para el evento binario "ganar la carrera".

    BS = (1/N) × Σ_i (p_win_i - y_win_i)²

    Rango: [0, 1]. Perfecto = 0. Modelo naïf uniforme ≈ (N-1)/N² ≈ 0.047 con 20 pilotos.

    Args:
        finish_probs:   DataFrame de predicciones (formatos soportados → extract_win_probs).
        actual_results: list[RaceResult] de jolpica_client (solo se usa position == 1).

    Returns:
        float, o NaN si no hay datos suficientes.
    """
    p_win = extract_win_probs(finish_probs)
    if p_win.empty:
        return float("nan")

    winner_id = _actual_winner(actual_results)
    if winner_id is None:
        return float("nan")

    outcomes = pd.Series(0.0, index=p_win.index)
    if winner_id in outcomes.index:
        outcomes[winner_id] = 1.0

    # Pilotos en resultados pero sin predicción → asumir p=0
    return float(((p_win.reindex(outcomes.index, fill_value=0.0) - outcomes) ** 2).mean())


def brier_score_topk(
    finish_probs: pd.DataFrame,
    actual_results: list,
    k: int = 3,
) -> float:
    """
    Brier score para el evento "terminar en top-k".

    Args:
        finish_probs:   DataFrame de predicciones.
        actual_results: list[RaceResult].
        k:              Umbral de posición (k=3 → podio).

    Returns:
        float, o NaN si no hay datos suficientes.
    """
    p_topk = extract_topk_probs(finish_probs, k)
    if p_topk.empty:
        return float("nan")

    topk_ids = {r.driver_id for r in actual_results if r.position is not None and r.position <= k}
    outcomes = pd.Series(
        {d: (1.0 if d in topk_ids else 0.0) for d in p_topk.index}
    )

    return float(((p_topk.reindex(outcomes.index, fill_value=0.0) - outcomes) ** 2).mean())


def brier_score_full(
    finish_probs: pd.DataFrame,
    actual_results: list,
) -> float:
    """
    Brier score multiclase sobre la distribución completa de posición.

    BS_full = (1/N) × Σ_i Σ_j (p_ij - y_ij)²
    donde y_ij = 1 si el piloto i terminó en la posición j.

    Solo aplicable cuando finish_probs está en formato canónico
    (driver_id, finish_pos, probability).

    Returns:
        float, o NaN si finish_probs no tiene el formato completo.
    """
    if not {"driver_id", "finish_pos", "probability"}.issubset(finish_probs.columns):
        log.debug("brier_score_full requiere formato canónico — devolviendo NaN")
        return float("nan")

    # Tabla predicha: matriz drivers × posiciones
    pivot = finish_probs.pivot_table(
        index="driver_id", columns="finish_pos", values="probability", fill_value=0.0
    )

    # Tabla real: matriz binaria drivers × posiciones
    actual = {r.driver_id: r.position for r in actual_results if r.position is not None}
    actual_matrix = pd.DataFrame(0.0, index=pivot.index, columns=pivot.columns)
    for driver_id, pos in actual.items():
        if driver_id in actual_matrix.index and pos in actual_matrix.columns:
            actual_matrix.loc[driver_id, pos] = 1.0

    diff = pivot - actual_matrix
    return float((diff ** 2).values.mean())


# ---------------------------------------------------------------------------
# Log-loss
# ---------------------------------------------------------------------------

def log_loss_win(
    finish_probs: pd.DataFrame,
    actual_results: list,
    clip_min: float = 1e-6,
) -> float:
    """
    Log-loss para el evento binario "ganar".

    LL = -[y_w × log(p_w) + Σ_{i≠w} log(1 - p_i)]

    Args:
        finish_probs:  DataFrame de predicciones.
        actual_results: list[RaceResult].
        clip_min:      Cota inferior de probabilidad (evita log(0)).

    Returns:
        float, o NaN si no hay datos.
    """
    p_win = extract_win_probs(finish_probs)
    if p_win.empty:
        return float("nan")

    winner_id = _actual_winner(actual_results)
    if winner_id is None:
        return float("nan")

    p_clipped = p_win.clip(lower=clip_min, upper=1.0 - clip_min)

    total = 0.0
    for driver_id, p in p_clipped.items():
        y = 1.0 if driver_id == winner_id else 0.0
        total += y * math.log(p) + (1.0 - y) * math.log(1.0 - p)

    return float(-total / len(p_clipped))


# ---------------------------------------------------------------------------
# Reliability diagram y ECE
# ---------------------------------------------------------------------------

def reliability_diagram(
    prob_outcome_pairs: list[_ProbOutcomePair],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Construye los datos de la curva de calibración (reliability diagram).

    Agrupa pares (p_predicha, y_real) en bins de igual anchura y computa
    la probabilidad media predicha y la frecuencia media observada por bin.

    Args:
        prob_outcome_pairs: Lista de (float, int) donde float ∈ [0,1] y int ∈ {0,1}.
        n_bins:             Número de bins. 10 es estándar en la literatura.

    Returns:
        DataFrame con columnas:
          bin_lower    — límite inferior del bin
          bin_upper    — límite superior del bin
          bin_center   — centro del bin (= (lower + upper) / 2)
          mean_pred    — media de probabilidades predichas en el bin
          mean_obs     — fracción de outcomes positivos en el bin (frecuencia observada)
          count        — número de muestras en el bin
          gap          — |mean_pred - mean_obs| (calibration gap del bin)
    """
    if not prob_outcome_pairs:
        return pd.DataFrame(columns=[
            "bin_lower", "bin_upper", "bin_center",
            "mean_pred", "mean_obs", "count", "gap",
        ])

    probs   = np.array([p for p, _ in prob_outcome_pairs])
    outcomes = np.array([y for _, y in prob_outcome_pairs])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Último bin incluye el 1.0 exacto
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bin_lower":  lo,
                "bin_upper":  hi,
                "bin_center": (lo + hi) / 2,
                "mean_pred":  float("nan"),
                "mean_obs":   float("nan"),
                "count":      0,
                "gap":        float("nan"),
            })
            continue

        mean_pred = float(probs[mask].mean())
        mean_obs  = float(outcomes[mask].mean())
        rows.append({
            "bin_lower":  lo,
            "bin_upper":  hi,
            "bin_center": (lo + hi) / 2,
            "mean_pred":  mean_pred,
            "mean_obs":   mean_obs,
            "count":      count,
            "gap":        abs(mean_pred - mean_obs),
        })

    return pd.DataFrame(rows)


def expected_calibration_error(diag: pd.DataFrame) -> float:
    """
    Expected Calibration Error (ECE): media ponderada del calibration gap.

    ECE = Σ_b (n_b / N) × |mean_pred_b - mean_obs_b|

    Args:
        diag: Salida de reliability_diagram().

    Returns:
        float en [0, 1]. 0 = perfectamente calibrado.
    """
    valid = diag.dropna(subset=["mean_pred", "mean_obs", "count"])
    if valid.empty:
        return float("nan")

    total = valid["count"].sum()
    ece = float((valid["count"] / total * valid["gap"]).sum())
    return ece


def collect_win_pairs(
    predictions: list,
    results_by_round: dict[int, list],
) -> list[_ProbOutcomePair]:
    """
    Recopila todos los pares (p_win_i_ronda, y_win_i_ronda) de múltiples rondas.

    Usado para construir la reliability diagram con suficientes datos
    (una sola carrera tiene solo 20 puntos; se necesitan varias rondas).

    Args:
        predictions:      list[RoundPrediction].
        results_by_round: {round_: list[RaceResult]}.

    Returns:
        Lista de (p_predicha, y_real) para todos los drivers × rondas.
    """
    pairs: list[_ProbOutcomePair] = []

    for pred in predictions:
        results = results_by_round.get(pred.round_)
        if results is None:
            log.debug("Sin resultados reales para R%02d — saltando", pred.round_)
            continue

        try:
            p_win = extract_win_probs(pred.finish_probs)
        except ValueError as exc:
            log.warning("R%02d: no se pudo extraer win_probs: %s", pred.round_, exc)
            continue

        winner_id = _actual_winner(results)
        known_drivers = {r.driver_id for r in results}

        for driver_id, p in p_win.items():
            if driver_id not in known_drivers:
                continue
            y = 1 if driver_id == winner_id else 0
            pairs.append((float(p), y))

    return pairs


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def rank_correlation(
    finish_probs: pd.DataFrame,
    actual_results: list,
) -> float:
    """
    Correlación de Spearman entre el ranking predicho y el resultado real.

    El ranking predicho se obtiene ordenando por P(win) descendente.
    Para el formato canónico se usa P(finish_pos == 1).

    Returns:
        float en [-1, 1]. 1.0 = predicción perfecta del orden. NaN si datos insuficientes.
    """
    try:
        p_win = extract_win_probs(finish_probs)
    except ValueError:
        return float("nan")

    if len(p_win) < 2:
        return float("nan")

    actual_pos = {r.driver_id: r.position for r in actual_results if r.position is not None}
    common = [d for d in p_win.index if d in actual_pos]

    if len(common) < 2:
        return float("nan")

    pred_rank   = p_win[common].rank(ascending=False)
    actual_rank = pd.Series({d: actual_pos[d] for d in common})

    return float(_spearman(pred_rank, actual_rank))


# ---------------------------------------------------------------------------
# Punto de entrada principal (usado por pipeline 06)
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: list,
    results_by_round: dict[int, list],
    n_bins: int = 10,
) -> dict:
    """
    Evalúa un conjunto de predicciones congeladas contra los resultados reales.

    Args:
        predictions:      list[RoundPrediction] de backtest/replay.py.
        results_by_round: {round_number: list[RaceResult]} de jolpica_client.
        n_bins:           Bins para la reliability diagram.

    Returns:
        dict con:
          'brier_win'         — Brier score medio para P(win) (todas las rondas)
          'brier_top3'        — Brier score medio para P(top-3)
          'log_loss_win'      — Log-loss medio para P(win)
          'rank_correlation'  — Spearman medio entre ranking predicho y real
          'ece_win'           — Expected Calibration Error para P(win)
          'per_round'         — DataFrame con métricas por ronda (round_, bs_win, bs_top3, ...)
          'reliability_win'   — DataFrame de reliability diagram para P(win)
          'n_rounds'          — número de rondas evaluadas
    """
    per_round_rows: list[dict] = []
    win_pairs: list[_ProbOutcomePair] = []

    for pred in sorted(predictions, key=lambda p: p.round_):
        results = results_by_round.get(pred.round_)
        if results is None:
            log.warning("R%02d: sin resultados reales — omitida en evaluación", pred.round_)
            continue

        row: dict = {"round_": pred.round_, "cutoff_date": pred.cutoff_date}

        # --- Brier win ---
        bs_win = brier_score_win(pred.finish_probs, results)
        row["bs_win"] = bs_win

        # --- Brier top-3 ---
        bs_top3 = brier_score_topk(pred.finish_probs, results, k=3)
        row["bs_top3"] = bs_top3

        # --- Brier full (solo si hay formato canónico) ---
        bs_full = brier_score_full(pred.finish_probs, results)
        row["bs_full"] = bs_full

        # --- Log-loss win ---
        ll_win = log_loss_win(pred.finish_probs, results)
        row["log_loss_win"] = ll_win

        # --- Rank correlation ---
        rho = rank_correlation(pred.finish_probs, results)
        row["rank_corr"] = rho

        per_round_rows.append(row)

        # Acumular pares para reliability diagram
        try:
            p_win = extract_win_probs(pred.finish_probs)
            winner_id = _actual_winner(results)
            known = {r.driver_id for r in results}
            for driver_id, p in p_win.items():
                if driver_id in known:
                    y = 1 if driver_id == winner_id else 0
                    win_pairs.append((float(p), y))
        except ValueError:
            pass

    per_round_df = pd.DataFrame(per_round_rows)
    reliability  = reliability_diagram(win_pairs, n_bins=n_bins)
    ece          = expected_calibration_error(reliability)

    def _safe_mean(col: str) -> float:
        if per_round_df.empty or col not in per_round_df.columns:
            return float("nan")
        return float(per_round_df[col].dropna().mean())

    result = {
        "brier_win":        _safe_mean("bs_win"),
        "brier_top3":       _safe_mean("bs_top3"),
        "log_loss_win":     _safe_mean("log_loss_win"),
        "rank_correlation": _safe_mean("rank_corr"),
        "ece_win":          ece,
        "per_round":        per_round_df,
        "reliability_win":  reliability,
        "n_rounds":         len(per_round_rows),
    }

    if per_round_rows:
        log.info(
            "Evaluación: %d rondas | BS_win=%.4f | LL_win=%.4f | "
            "Spearman=%.3f | ECE=%.4f",
            len(per_round_rows),
            result["brier_win"],
            result["log_loss_win"],
            result["rank_correlation"],
            ece,
        )
    else:
        log.warning("No se pudo evaluar ninguna ronda (sin resultados o sin predicciones).")

    return result


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _actual_winner(actual_results: list) -> str | None:
    """Devuelve driver_id del piloto que terminó en posición 1, o None."""
    for r in actual_results:
        if r.position == 1:
            return r.driver_id
    return None


def _softmax(scores: pd.Series) -> pd.Series:
    """Softmax sobre una Serie de fortalezas → distribución de probabilidad."""
    vals = scores.values.astype(float)
    vals = vals - vals.max()  # estabilidad numérica
    exp_vals = np.exp(vals)
    return pd.Series(exp_vals / exp_vals.sum(), index=scores.index)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Correlación de Spearman entre dos Series alineadas."""
    n = len(x)
    if n < 2:
        return float("nan")
    rx = x.rank()
    ry = y.rank()
    d2 = ((rx - ry) ** 2).sum()
    return float(1.0 - 6.0 * d2 / (n * (n ** 2 - 1)))
