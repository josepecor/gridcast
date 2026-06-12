"""
Adaptador PredictionModel → usa strengths + race_sim para generar finish_probs.

Implementa el protocolo PredictionModel de backtest/replay.py.
El modelo lee pace_features del snapshot congelado, ajusta las fortalezas via
fit_all() (within-team normalization), y corre simulate_race() para producir
una distribución de probabilidad sobre el orden de llegada.

Columnas de salida (formato canónico):
    driver_id   — abbreviation del piloto (de FastF1)
    finish_pos  — posición (1..n_drivers)
    probability — P(driver termina en esa posición)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.backtest.leakage_guard import LeakageGuard
from gridcast.backtest.snapshot import Snapshot
from gridcast.config import get_settings
from gridcast.models.strength import fit_all
from gridcast.simulation.race_sim import simulate_race

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _testing_features_path(season: int) -> Path:
    return _PROJECT_ROOT / "data" / "processed" / f"testing_features_{season}.parquet"


def _map_testing_to_pace_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map testing_features columns to pace_features schema (round=0 rows)."""
    mapped = df.rename(columns={
        "team": "team_name",
        "testing_long_run_s": "long_run_pace_s",
        "testing_laps_n": "long_run_n",
    }).copy()
    mapped["quali_pace_s"] = float("nan")
    mapped["fp_best_lap_s"] = float("nan")
    keep = ["season", "round", "abbreviation", "team_name",
            "long_run_pace_s", "long_run_n", "fp_best_lap_s", "quali_pace_s"]
    return mapped[[c for c in keep if c in mapped.columns]]


class StrengthRaceModel:
    """
    Modelo de predicción baseline: within-team strength + Plackett-Luce.

    Lee `pace_features` del snapshot congelado, ajusta el modelo de fuerza
    (baseline de normalización intra-equipo) y devuelve la distribución de
    probabilidad sobre el orden de llegada.

    Args:
        primary_signal: 'quali' (default) o 'race' — señal de pace primaria.
        n_sims:         Número de simulaciones por ronda. None → settings.
        seed:           Semilla. None → settings.
    """

    def __init__(
        self,
        primary_signal: str = "quali",
        n_sims: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._primary_signal = primary_signal
        cfg = get_settings()
        self._n_sims = n_sims or cfg.simulation.n_sims
        self._seed   = seed   or cfg.simulation.seed

    @property
    def version(self) -> str:
        return f"strength_race_v1_{self._primary_signal}"

    def predict(
        self,
        snapshot: Snapshot,
        round_: int,
        guard: LeakageGuard,
        extra_pace: pd.DataFrame | None = None,
        grid_positions: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """
        Genera predicciones para la ronda `round_`.

        Args:
            snapshot:       Estado congelado hasta el corte de round_.
            round_:         Número de ronda a predecir.
            guard:          LeakageGuard para validaciones adicionales.
            extra_pace:     Pace features adicionales de esta ronda (qualifying/sprint).
                            Mismas columnas que pace_features. Si tiene quali_pace_s no-NaN,
                            se usa primary_signal='quali'.
            grid_positions: dict {abbreviation → grid_pos (1=pole)}. Si se pasa, se aplica
                            un pequeño modificador de fortaleza por posición de salida.

        Returns:
            DataFrame con columnas: driver_id, finish_pos, probability.
        """
        # Pace features del snapshot (puede estar vacío en R01)
        if "pace_features" in snapshot.frames:
            pace = snapshot.read_frame("pace_features")
        else:
            pace = pd.DataFrame()

        # Testing features (round=0): prior point-in-time seguro para cualquier ronda.
        # Solo tienen long_run_pace_s → cuando se usan, cambiamos la señal a "race".
        season = snapshot.metadata.get("season", 2026)
        t_path = _testing_features_path(season)
        primary_signal = self._primary_signal
        if t_path.exists():
            testing = _map_testing_to_pace_schema(pd.read_parquet(t_path))
            if not testing.empty:
                pace = (
                    pd.concat([testing, pace], ignore_index=True)
                    if not pace.empty
                    else testing
                )
                primary_signal = "race"  # testing solo tiene long_run_pace_s
                log.info(
                    "R%02d: testing features incluidas (%d pilotos, round=0)",
                    round_, len(testing),
                )

        # Extra pace (qualifying/sprint de esta ronda): señal más informativa.
        # Si tiene quali_pace_s, cambia la señal a "quali" (mejor predictor).
        if extra_pace is not None and not extra_pace.empty:
            pace = pd.concat([pace, extra_pace], ignore_index=True) if not pace.empty else extra_pace
            has_quali = (
                "quali_pace_s" in extra_pace.columns
                and extra_pace["quali_pace_s"].notna().any()
            )
            if has_quali:
                primary_signal = "quali"
            log.info(
                "R%02d: extra_pace incluido (%d filas, signal=%s)",
                round_, len(extra_pace), primary_signal,
            )

        if pace.empty:
            log.warning(
                "R%02d: sin pace_features ni testing_features — "
                "ejecuta 01_ingest.py y 02_build_features.py",
                round_,
            )
            return pd.DataFrame(columns=["driver_id", "finish_pos", "probability"])

        # Ajustar fortalezas con datos disponibles en el corte
        estimates = fit_all(
            pace,
            engine_map=None,
            primary_signal=primary_signal,
        )
        driver_est = estimates["driver"]

        if driver_est.empty:
            log.warning(
                "R%02d: sin estimaciones de fuerza (¿suficientes compañeros de equipo?)",
                round_,
            )
            return pd.DataFrame(columns=["driver_id", "finish_pos", "probability"])

        driver_ids = driver_est["driver_id"].tolist()

        # Combinar fortaleza de piloto (within-team) + fortaleza de equipo (absoluta).
        # strength.py ya devuelve escala "higher = better" (negación de lap time incorporada).
        # No negar aquí: race_sim espera "higher = better" directamente.
        team_est = estimates["team"]
        team_strength_map = (
            team_est.set_index("team_id")["mean"]
            if not team_est.empty
            else pd.Series(dtype=float)
        )
        # Asignación driver → equipo (ronda más reciente con team_name no vacío)
        driver_team = (
            pace.sort_values("round")[["abbreviation", "team_name"]]
            .dropna(subset=["team_name"])
            .pipe(lambda d: d[d["team_name"].astype(str).str.strip() != ""])
            .drop_duplicates("abbreviation", keep="last")
            .set_index("abbreviation")["team_name"]
        )
        d_strength_map = driver_est.set_index("driver_id")["mean"]
        total_strengths = [
            float(d_strength_map.get(drv, 0.0))
            + float(team_strength_map.get(driver_team.get(drv, ""), 0.0))
            for drv in driver_ids
        ]
        strengths = np.array(total_strengths, dtype=float)
        # No usar strength_sds para muestreo posterior en la predicción.
        # Los strength.py devuelve estimaciones puntuales; para n_obs=1 el sd
        # sería _WIDE_PRIOR_SD=2.0s, mucho mayor que el spread real del campo
        # (±0.3s) y daría a pilotos con pocas observaciones probabilidades
        # absurdamente altas. El ruido Gumbel (noise_scale=0.5) ya captura
        # la variabilidad dentro de carrera.
        strength_sds = None

        # Grid position modifier: P1 +boost, Plast -boost (±0.15s total range).
        # Captura ventaja de posición de salida independiente del ritmo puro.
        if grid_positions:
            n_d = len(driver_ids)
            scale = 0.15  # rango total ±0.15s
            for i, drv in enumerate(driver_ids):
                gp = grid_positions.get(drv)
                if gp is not None:
                    strengths[i] += scale * (1.0 - 2.0 * (gp - 1) / max(n_d - 1, 1))

        rng = np.random.default_rng(
            np.random.SeedSequence([self._seed, round_])
        )

        result = simulate_race(
            driver_ids=driver_ids,
            strengths=strengths,
            strength_sds=strength_sds,
            n_sims=self._n_sims,
            rng=rng,
        )

        # Formato canónico: (driver_id, finish_pos, probability)
        dist = result.finish_position_distribution()   # [n_drivers, n_drivers]
        rows = []
        for d_idx, drv in enumerate(driver_ids):
            for pos in range(len(driver_ids)):
                rows.append({
                    "driver_id":   drv,
                    "finish_pos":  pos + 1,
                    "probability": float(dist[d_idx, pos]),
                })

        log.info(
            "R%02d: predicción generada para %d pilotos (%d sims, señal=%s)",
            round_, len(driver_ids), self._n_sims, self._primary_signal,
        )
        return pd.DataFrame(rows)
