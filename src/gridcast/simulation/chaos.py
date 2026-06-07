"""
Caos de carrera: safety car, lluvia y DNF boost.

Propósito (§4.5 CLAUDE.md):
  "Sin caos, el favorito gana el 99% → irreal. El caos es feature, no bug."

El módulo envuelve race_sim.simulate_race() añadiendo eventos aleatorios que la
distribución Plackett-Luce pura no captura porque afectan a todos los pilotos a la vez:

  Safety car — reagrupa el campo; ventaja de ritmo se reduce temporalmente.
    Modelado: multiplica noise_scale por sc_noise_multiplier.
    Probabilidad media: ~55% de las carreras de F1 tienen al menos un SC.

  Lluvia — comprime las diferencias de ritmo entre coches; cuida en el que el
    talento del piloto pesa más que la aerodinámica del coche.
    Modelado: comprime la dispersión de strengths + aumenta noise_scale.
    Probabilidad media: ~10% (varía mucho por circuito).

  DNF boost 2026 — reglamento nuevo → mayor incertidumbre de fiabilidad.
    Modelado: incremento flat de dnf_rates por encima de la tasa histórica.

Implementación vectorizada:
  Las n_sims simulaciones se agrupan por escenario de caos (SC×lluvia → 4 grupos).
  Cada grupo llama una vez a simulate_race() con los parámetros ajustados.
  Los resultados se reensamblan en los índices originales.

Extensiones futuras:
  - Ajuste por circuito (Monaco baja SC probability, Spa alta rain probability).
  - Incidentes de primera vuelta (elevación puntual de DNF en vuelta 1).
  - SC virtual (VSC) como categoría intermedia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from gridcast.simulation.race_sim import RaceSimResult, simulate_race

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parámetros de caos
# ---------------------------------------------------------------------------

@dataclass
class ChaosParameters:
    """
    Parámetros configurables para los eventos de caos de carrera.

    Los defaults están calibrados sobre estadísticas de F1 2018-2025.
    Ajustar por circuito cuando se disponga de datos suficientes.
    """
    # --- Probabilidades base ---
    sc_probability: float = 0.55
    """P(safety car aparece en la carrera). ~55% en F1 moderno."""

    rain_probability: float = 0.10
    """P(lluvia significativa que afecta al ritmo). Media ~10%; varía mucho por circuito."""

    # --- Efectos del safety car sobre noise_scale ---
    sc_noise_multiplier: float = 1.35
    """Factor multiplicativo de noise_scale bajo safety car.
    > 1 → mayor sorpresa; el campo se reagrupa y el restart es caótico."""

    # --- Efectos de la lluvia ---
    rain_noise_multiplier: float = 1.25
    """Factor multiplicativo de noise_scale en lluvia. Añade incertidumbre al ritmo."""

    rain_strength_compression: float = 0.65
    """Factor de compresión de la dispersión de strengths en lluvia.
    0.65 → se reduce el 35% de la diferencia entre el más rápido y el más lento.
    Modela que en lluvia los pilotos puntúan más que los coches."""

    # --- DNF boost ---
    dnf_rate_boost: float = 0.02
    """Incremento flat de P(DNF) por encima de la tasa histórica del piloto.
    En 2026 se eleva por el debut del nuevo reglamento de motores."""


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def simulate_race_with_chaos(
    driver_ids: list[str],
    strengths: np.ndarray,
    strength_sds: np.ndarray | None = None,
    dnf_rates: np.ndarray | None = None,
    grid_penalties: np.ndarray | None = None,
    pit_lane_starts: np.ndarray | None = None,
    is_sprint: bool = False,
    n_sims: int = 10_000,
    base_noise_scale: float = 0.5,
    params: ChaosParameters | None = None,
    rng: np.random.Generator | None = None,
    strength_penalty_per_grid_pos: float = 0.02,
    dev_noise: np.ndarray | None = None,
) -> RaceSimResult:
    """
    Simula N realizaciones de una carrera incorporando eventos de caos.

    Diferencia respecto a simulate_race():
      - Cada simulación sortea independientemente si hay SC y/o lluvia.
      - Los parámetros de la carrera (noise_scale, strengths, dnf_rates) se ajustan
        según el escenario de caos antes de llamar a simulate_race().
      - El DNF boost de 2026 se aplica siempre (independientemente de SC/lluvia).

    Args:
        driver_ids:       Lista de identificadores de piloto (indexados en los arrays).
        strengths:        [n_drivers] fortaleza media (higher = better, misma escala
                          que simulate_race: typical range ≈ ±3 en unidades de pace_s).
        strength_sds:     [n_drivers] incertidumbre de la estimación. None = puntual.
        dnf_rates:        [n_drivers] P(DNF) base por piloto. El boost de 2026 se suma.
        grid_penalties:   [n_drivers] posiciones de penalización de parrilla (≥0).
        pit_lane_starts:  [n_drivers] True si el piloto sale desde el pit lane.
        is_sprint:        True para sprint race (tabla de puntos diferente).
        n_sims:           Número de simulaciones Monte Carlo.
        base_noise_scale: σ del ruido Gumbel antes de modificadores de caos.
        params:           Parámetros de caos. None → defaults de ChaosParameters.
        rng:              RNG de NumPy. None → seed de settings.yaml.
        strength_penalty_per_grid_pos: Reducción de fortaleza por posición de penalización.

    Returns:
        RaceSimResult con los resultados de las n_sims simulaciones, con caos.
    """
    if params is None:
        params = ChaosParameters()

    if rng is None:
        from gridcast.config import get_settings
        rng = np.random.default_rng(get_settings().simulation.seed)

    n_drivers = len(driver_ids)
    strengths = np.asarray(strengths, dtype=float)

    dnf_base = (
        np.asarray(dnf_rates, dtype=float)
        if dnf_rates is not None
        else np.zeros(n_drivers)
    )

    # Sortear caos independientemente por simulación
    has_sc   = rng.random(n_sims) < params.sc_probability    # [n_sims] bool
    has_rain = rng.random(n_sims) < params.rain_probability  # [n_sims] bool

    # Acumuladores — se rellenarán por grupo de escenario
    all_orders = np.empty((n_sims, n_drivers), dtype=np.intp)
    all_dnf    = np.empty((n_sims, n_drivers), dtype=bool)
    all_points = np.empty((n_sims, n_drivers), dtype=float)

    # Cuatro escenarios: SC ∈ {F,T} × lluvia ∈ {F,T}
    for sc in (False, True):
        for rain in (False, True):
            mask = (has_sc == sc) & (has_rain == rain)
            k    = int(mask.sum())
            if k == 0:
                continue

            noise_scale   = base_noise_scale
            adj_strengths = strengths.copy()
            adj_dnf       = np.clip(dnf_base + params.dnf_rate_boost, 0.0, 1.0)

            if sc:
                noise_scale *= params.sc_noise_multiplier

            if rain:
                noise_scale *= params.rain_noise_multiplier
                mean_s = adj_strengths.mean()
                adj_strengths = (
                    mean_s
                    + (adj_strengths - mean_s) * params.rain_strength_compression
                )

            log.debug(
                "caos grupo SC=%s rain=%s: %d sims, noise=%.3f, "
                "strength_spread=%.3f",
                sc, rain, k, noise_scale,
                float((adj_strengths - adj_strengths.mean()).std()),
            )

            # dev_noise es [n_sims, n_drivers]; extraer filas del grupo actual
            group_dev = dev_noise[mask] if dev_noise is not None else None

            result = simulate_race(
                driver_ids=driver_ids,
                strengths=adj_strengths,
                strength_sds=strength_sds,
                dnf_rates=adj_dnf,
                grid_penalties=grid_penalties,
                pit_lane_starts=pit_lane_starts,
                is_sprint=is_sprint,
                n_sims=k,
                noise_scale=noise_scale,
                strength_penalty_per_grid_pos=strength_penalty_per_grid_pos,
                rng=rng,
                additional_strength_offset=group_dev,
            )

            idx = np.where(mask)[0]
            all_orders[idx] = result.finishing_orders
            all_dnf[idx]    = result.dnf_flags
            all_points[idx] = result.points

    return RaceSimResult(
        driver_ids=driver_ids,
        finishing_orders=all_orders,
        dnf_flags=all_dnf,
        points=all_points,
        is_sprint=is_sprint,
    )
