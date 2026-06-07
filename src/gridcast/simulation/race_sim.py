"""
Simulador de carrera: distribución de orden de llegada mediante Plackett-Luce.

Modelo (§4.3 CLAUDE.md):
  Para cada piloto: pace_estimada + ruido Gumbel → orden de llegada.
  El truco de Gumbel implementa exactamente el muestreo de Plackett-Luce:
    argmax(strength_i + Gumbel(0, σ)) ~ Plackett-Luce(exp(strength_i / σ))

Estructura de la simulación:
  1. Muestrear fortalezas desde la distribución posterior (Normal(mean, sd))
     para capturar la incertidumbre en las estimaciones del modelo jerárquico.
  2. Aplicar penalizaciones de parrilla como reducción de fortaleza efectiva.
  3. Añadir ruido Gumbel para modelar variabilidad de carrera.
  4. Ordenar → orden de llegada "puro" (sin caos).
  5. Sortear DNFs independientemente por piloto (Bernoulli).
  6. Pilotos DNF pasan al final del orden; solo los clasificados puntúan.
  7. Aplicar tabla de puntos (scoring.py).

Caos externo (safety car, lluvia, incidentes vuelta 1) está en simulation/chaos.py
y se aplica ANTES de llamar a simulate_race(), ajustando strengths/dnf_rates.

Convención de signos para strengths:
  Higher strength = better expected finishing position.
  Velocidad de vuelta (en segundos) → strength = -lap_time_s.
  El modelo jerárquico (models/strength.py) devuelve directamente esta escala.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gridcast.simulation.scoring import (
    build_race_points_table,
    build_sprint_points_table,
    score_finishing_orders_batch,
)

log = logging.getLogger(__name__)

# Número máximo de drivers que reconoce el simulador
_MAX_DRIVERS = 30


# ---------------------------------------------------------------------------
# Resultado de simulación
# ---------------------------------------------------------------------------

@dataclass
class RaceSimResult:
    """
    Resultado de N simulaciones Monte Carlo de una carrera.

    Convención de índices:
      finishing_orders[s, pos] = driver_idx del piloto en posición `pos`
                                 de la simulación `s` (pos 0 = ganador).
      dnf_flags[s, d]          = True si el driver_idx `d` abandonó en `s`.
      points[s, d]             = puntos obtenidos por driver_idx `d` en `s`.
    """
    driver_ids: list[str]
    finishing_orders: np.ndarray   # [n_sims, n_drivers]
    dnf_flags: np.ndarray          # [n_sims, n_drivers]
    points: np.ndarray             # [n_sims, n_drivers]
    is_sprint: bool

    # -------------------------------------------------------------------
    # Propiedades de resumen — O(n_sims) computadas on-demand
    # -------------------------------------------------------------------

    @property
    def n_sims(self) -> int:
        return self.finishing_orders.shape[0]

    @property
    def n_drivers(self) -> int:
        return len(self.driver_ids)

    def win_probability(self) -> np.ndarray:
        """P(driver_idx i gana) — vector [n_drivers]."""
        winners = self.finishing_orders[:, 0]
        return np.bincount(winners, minlength=self.n_drivers) / self.n_sims

    def podium_probability(self) -> np.ndarray:
        """P(driver_idx i termina en el podio) — vector [n_drivers]."""
        counts = np.zeros(self.n_drivers, dtype=float)
        for pos in range(min(3, self.n_drivers)):
            np.add.at(counts, self.finishing_orders[:, pos], 1)
        return counts / self.n_sims

    def top_k_probability(self, k: int) -> np.ndarray:
        """P(driver_idx i termina en el top k) — vector [n_drivers]."""
        k = min(k, self.n_drivers)
        counts = np.zeros(self.n_drivers, dtype=float)
        for pos in range(k):
            np.add.at(counts, self.finishing_orders[:, pos], 1)
        return counts / self.n_sims

    def dnf_probability(self) -> np.ndarray:
        """P(driver_idx i abandona) — vector [n_drivers]."""
        return self.dnf_flags.mean(axis=0)

    def expected_points(self) -> np.ndarray:
        """E[puntos] por driver_idx — vector [n_drivers]."""
        return self.points.mean(axis=0)

    def finish_position_distribution(self) -> np.ndarray:
        """
        P(driver_idx i termina en posición pos) — matriz [n_drivers, n_drivers].
        dist[i, pos] = fracción de sims donde driver i terminó en posición pos.
        Incluye posiciones de DNF (al final del orden de llegada).
        """
        dist = np.zeros((self.n_drivers, self.n_drivers), dtype=float)
        for pos in range(self.n_drivers):
            np.add.at(dist[:, pos], self.finishing_orders[:, pos], 1)
        return dist / self.n_sims

    def to_summary_df(self) -> pd.DataFrame:
        """DataFrame por piloto: win_prob, podium_prob, dnf_prob, expected_points."""
        return pd.DataFrame({
            "driver_id":      self.driver_ids,
            "win_prob":       self.win_probability(),
            "podium_prob":    self.podium_probability(),
            "dnf_prob":       self.dnf_probability(),
            "expected_points": self.expected_points(),
        })


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def simulate_race(
    driver_ids: list[str],
    strengths: np.ndarray,
    strength_sds: np.ndarray | None = None,
    dnf_rates: np.ndarray | None = None,
    grid_penalties: np.ndarray | None = None,
    pit_lane_starts: np.ndarray | None = None,
    is_sprint: bool = False,
    n_sims: int = 10_000,
    noise_scale: float = 0.5,
    strength_penalty_per_grid_pos: float = 0.02,
    rng: np.random.Generator | None = None,
    additional_strength_offset: np.ndarray | None = None,
) -> RaceSimResult:
    """
    Simula N realizaciones de una carrera con Plackett-Luce.

    Args:
        driver_ids:      Lista de n_drivers identificadores de piloto. El índice
                         en esta lista corresponde a driver_idx en los arrays de salida.
        strengths:       [n_drivers] — fortaleza media estimada (escala log: higher = better).
                         Típicamente -pace_s normalizado o salida de models/strength.py.
        strength_sds:    [n_drivers] — desviación estándar de la estimación de fortaleza.
                         Si None, se usan las fortalezas como valores puntuales (sin
                         muestreo de incertidumbre del modelo).
        dnf_rates:       [n_drivers] — P(abandono) por piloto.
                         Si None, todos los pilotos terminan siempre.
        grid_penalties:  [n_drivers] — puestos de penalización de parrilla (0 = sin penalización).
                         Se aplican como reducción de fortaleza efectiva.
        pit_lane_starts: [n_drivers] — True si el piloto sale desde el pit lane.
                         La penalización equivalente es 1.5 × grid_penalties positivos.
        is_sprint:       True para usar la tabla de puntos de sprint.
        n_sims:          Número de simulaciones Monte Carlo.
        noise_scale:     Escala σ del ruido Gumbel (variabilidad de carrera).
                         Mayor σ = más aleatoriedad. Valor por defecto calibrado para F1.
        strength_penalty_per_grid_pos: Reducción de fortaleza por posición de penalización
                         de parrilla. Ajustable empíricamente con datos observados.
        rng:             Generador de números aleatorios de NumPy. Si None, se crea uno
                         con la semilla de settings.yaml (reproducible).

    Returns:
        RaceSimResult con finishing_orders, dnf_flags, points y métodos de resumen.

    Raises:
        ValueError: si los arrays de entrada tienen longitudes inconsistentes.
    """
    driver_ids = list(driver_ids)
    n_drivers  = len(driver_ids)

    if n_drivers == 0:
        raise ValueError("driver_ids no puede estar vacío")
    if n_drivers > _MAX_DRIVERS:
        raise ValueError(f"Demasiados drivers: {n_drivers} > {_MAX_DRIVERS}")

    strengths = np.asarray(strengths, dtype=float)
    _check_shape(strengths, (n_drivers,), "strengths")

    if strength_sds is not None:
        strength_sds = np.asarray(strength_sds, dtype=float)
        _check_shape(strength_sds, (n_drivers,), "strength_sds")

    if dnf_rates is not None:
        dnf_rates = np.asarray(dnf_rates, dtype=float)
        _check_shape(dnf_rates, (n_drivers,), "dnf_rates")
        if np.any((dnf_rates < 0) | (dnf_rates > 1)):
            raise ValueError("dnf_rates debe estar en [0, 1]")
    else:
        dnf_rates = np.zeros(n_drivers)

    if grid_penalties is not None:
        grid_penalties = np.asarray(grid_penalties, dtype=float)
        _check_shape(grid_penalties, (n_drivers,), "grid_penalties")
    else:
        grid_penalties = np.zeros(n_drivers)

    if pit_lane_starts is not None:
        pit_lane_starts = np.asarray(pit_lane_starts, dtype=bool)
        _check_shape(pit_lane_starts, (n_drivers,), "pit_lane_starts")
    else:
        pit_lane_starts = np.zeros(n_drivers, dtype=bool)

    if rng is None:
        from gridcast.config import get_settings
        rng = np.random.default_rng(get_settings().simulation.seed)

    # 1. Penalizaciones de parrilla → reducción de fortaleza efectiva
    effective_strengths = _apply_grid_penalties(
        strengths, grid_penalties, pit_lane_starts, strength_penalty_per_grid_pos
    )

    # 2. Muestrear fortalezas desde la posterior (si hay incertidumbre)
    sampled_strengths = _sample_strengths(effective_strengths, strength_sds, n_sims, rng)

    # 2b. Offset de desarrollo de coche por simulación (season_sim la pasa para
    #     reflejar incertidumbre sobre la jerarquía de coches durante la temporada).
    #     Shape [n_sims, n_drivers]; se suma elemento a elemento.
    if additional_strength_offset is not None:
        sampled_strengths = sampled_strengths + additional_strength_offset

    # 3. Plackett-Luce: añadir ruido Gumbel y ordenar
    finishing_orders = _plackett_luce(sampled_strengths, noise_scale, rng)  # [n_sims, n_drivers]

    # 4. DNFs: Bernoulli independiente por piloto
    dnf_flags = _sample_dnfs(dnf_rates, n_sims, rng)  # [n_sims, n_drivers]

    # 5. DNFed drivers al final del orden (orden relativo entre finishers conservado)
    finishing_orders, n_finishers = _apply_dnfs(finishing_orders, dnf_flags)

    # 6. Puntos
    pts_table = (
        build_sprint_points_table(n_drivers)
        if is_sprint
        else build_race_points_table(n_drivers)
    )
    points = score_finishing_orders_batch(finishing_orders, pts_table, n_finishers)

    return RaceSimResult(
        driver_ids=driver_ids,
        finishing_orders=finishing_orders,
        dnf_flags=dnf_flags,
        points=points,
        is_sprint=is_sprint,
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _apply_grid_penalties(
    strengths: np.ndarray,
    grid_penalties: np.ndarray,
    pit_lane_starts: np.ndarray,
    penalty_per_pos: float,
) -> np.ndarray:
    """Reduce la fortaleza de pilotos con penalización de parrilla."""
    effective = strengths.copy()
    # Penalización base: posiciones × peso
    effective -= grid_penalties * penalty_per_pos
    # Penalización adicional por salida desde el pit lane (1.5× total)
    effective -= pit_lane_starts * grid_penalties * penalty_per_pos * 0.5
    return effective


def _sample_strengths(
    strengths: np.ndarray,
    sds: np.ndarray | None,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Muestrea fortalezas desde Normal(mean, sd) si se proveen SDs.

    Returns:
        [n_sims, n_drivers]
    """
    if sds is not None:
        return rng.normal(
            loc=strengths[np.newaxis, :],
            scale=np.maximum(sds[np.newaxis, :], 1e-9),  # evitar sd=0
            size=(n_sims, len(strengths)),
        )
    return np.broadcast_to(strengths, (n_sims, len(strengths))).copy()


def _plackett_luce(
    strengths_batch: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Muestrea un orden de llegada Plackett-Luce por simulación.

    Truco de Gumbel: argmax(strength_i + Gumbel(0, σ)) ~ PL(exp(strength_i / σ)).

    Args:
        strengths_batch: [n_sims, n_drivers]
        noise_scale:     σ del ruido Gumbel.

    Returns:
        finishing_orders: [n_sims, n_drivers] — driver_idx en cada posición.
    """
    gumbel = rng.gumbel(loc=0.0, scale=noise_scale, size=strengths_batch.shape)
    noisy  = strengths_batch + gumbel
    # argsort descendente: posición 0 = driver con mayor noisy_strength (ganador)
    return np.argsort(-noisy, axis=1, kind="stable")


def _sample_dnfs(
    dnf_rates: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Muestrea flags de DNF independientes por piloto y simulación.

    Returns:
        dnf_flags: [n_sims, n_drivers] bool
    """
    return rng.random((n_sims, len(dnf_rates))) < dnf_rates[np.newaxis, :]


def _apply_dnfs(
    finishing_orders: np.ndarray,
    dnf_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mueve los pilotos DNF al final del orden de llegada.

    Los finishers mantienen su orden relativo de Plackett-Luce.
    Los DNFs se colocan en el resto de posiciones (su orden entre sí es arbitrario).

    Args:
        finishing_orders: [n_sims, n_drivers] — orden PL, antes de DNFs.
        dnf_flags:        [n_sims, n_drivers] — True si driver_idx i DNFed en sim s.

    Returns:
        (reordered, n_finishers)
          reordered:   [n_sims, n_drivers] — finishers primero, DNFs al final.
          n_finishers: [n_sims] int — número de pilotos clasificados por simulación.
    """
    n_sims, n_drivers = finishing_orders.shape

    # dnf_at_pos[s, pos] = True si el piloto en la posición `pos` de la sim `s` DNFó
    # Necesitamos indexar dnf_flags (driver-indexed) con finishing_orders (pos-indexed)
    row_idx = np.arange(n_sims)[:, np.newaxis]      # [n_sims, 1]
    dnf_at_pos = dnf_flags[row_idx, finishing_orders]  # [n_sims, n_drivers]

    # Clave de ordenación: (DNF_flag, posición_PL) → stable sort los DNF al final
    # Como finishing_orders ya está ordenado por posición PL, un stable sort
    # por dnf_at_pos mantiene el orden relativo dentro de cada grupo.
    sort_key = dnf_at_pos.astype(np.int8)  # 0 = finisher, 1 = DNF
    reordered = np.empty_like(finishing_orders)
    for s in range(n_sims):
        # stable sort: finishers (0) preceden a DNFs (1), orden interno conservado
        order_within = np.argsort(sort_key[s], kind="stable")
        reordered[s] = finishing_orders[s, order_within]

    n_finishers = (n_drivers - dnf_flags.sum(axis=1)).astype(np.intp)
    return reordered, n_finishers


def _check_shape(arr: np.ndarray, expected: tuple, name: str) -> None:
    if arr.shape != expected:
        raise ValueError(
            f"{name}: forma esperada {expected}, recibida {arr.shape}"
        )
