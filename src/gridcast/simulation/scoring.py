"""
Asignación de puntos F1 2026.

Reglas (§3.1 CLAUDE.md, fuente: reglamento deportivo FIA):
  Carrera  (top 10): 25, 18, 15, 12, 10, 8, 6, 4, 2, 1
  Sprint   (top  8):  8,  7,  6,  5,  4, 3, 2, 1
  Sin punto de vuelta rápida (eliminado desde 2025).

Dos niveles de API:
  1. Escalar / por carrera: funciones para verificación manual y tests.
  2. Vectorizado por lote: para simulaciones Monte Carlo (n_sims × n_drivers).

Convención de índices usada por race_sim:
  finishing_orders[sim, pos] = driver_idx que termina en la posición `pos`
    → pos 0 = ganador, pos 1 = segundo, ...
    → drivers después de n_finishers[sim] son DNF y no puntúan
"""

from __future__ import annotations

import numpy as np

from gridcast.config import get_scoring


# ---------------------------------------------------------------------------
# Tablas de puntos (arrays por posición, 0-indexed)
# ---------------------------------------------------------------------------

def build_race_points_table(n_drivers: int) -> np.ndarray:
    """
    Array de puntos por posición de carrera.

    Returns:
        table[pos] = puntos para el piloto que termina en la posición `pos` (0-indexed).
        Posiciones sin puntos (pos ≥ 10) valen 0.
    """
    cfg = get_scoring()
    table = np.zeros(n_drivers, dtype=float)
    for pos_1indexed, pts in cfg.race.positions.items():
        idx = pos_1indexed - 1
        if idx < n_drivers:
            table[idx] = float(pts)
    return table


def build_sprint_points_table(n_drivers: int) -> np.ndarray:
    """
    Array de puntos por posición de carrera al sprint (top 8 puntúan).
    """
    cfg = get_scoring()
    table = np.zeros(n_drivers, dtype=float)
    for pos_1indexed, pts in cfg.sprint.positions.items():
        idx = pos_1indexed - 1
        if idx < n_drivers:
            table[idx] = float(pts)
    return table


# ---------------------------------------------------------------------------
# Scoring escalar (una sola carrera)
# ---------------------------------------------------------------------------

def score_finishing_order(
    finishing_order: np.ndarray,
    points_table: np.ndarray,
    n_finishers: int | None = None,
) -> np.ndarray:
    """
    Asigna puntos a cada driver_idx a partir del orden de llegada.

    Args:
        finishing_order: [n_drivers] — driver_idx en cada posición (pos 0 = ganador).
        points_table:    [n_drivers] — points_table[pos] = puntos para esa posición.
        n_finishers:     Número de pilotos clasificados (los demás no puntúan).
                         None → todos los pilotos se consideran clasificados.

    Returns:
        points[driver_idx] = puntos obtenidos por ese piloto.
    """
    n = len(finishing_order)
    effective_table = points_table.copy()
    if n_finishers is not None and n_finishers < n:
        effective_table[n_finishers:] = 0.0

    points = np.zeros(n, dtype=float)
    for pos, driver_idx in enumerate(finishing_order):
        points[driver_idx] = effective_table[pos]
    return points


def points_for_position(position_1indexed: int, is_sprint: bool = False) -> float:
    """
    Puntos para una posición concreta. Útil para verificación manual.

    Args:
        position_1indexed: Posición de llegada (1 = ganador).
        is_sprint:         True para usar tabla de sprint.

    Returns:
        Puntos correspondientes; 0 si la posición no puntúa.
    """
    cfg = get_scoring()
    table = cfg.sprint.positions if is_sprint else cfg.race.positions
    return float(table.get(position_1indexed, 0))


# ---------------------------------------------------------------------------
# Scoring vectorizado (batch de simulaciones)
# ---------------------------------------------------------------------------

def score_finishing_orders_batch(
    finishing_orders: np.ndarray,
    points_table: np.ndarray,
    n_finishers: np.ndarray | None = None,
) -> np.ndarray:
    """
    Asigna puntos para un lote de simulaciones de forma completamente vectorizada.

    Implementación O(n_sims × n_drivers) sin bucles Python. Requiere numpy ≥ 1.15
    para `np.put_along_axis`.

    Args:
        finishing_orders: [n_sims, n_drivers] — finishing_orders[s, pos] = driver_idx
                          del piloto en posición `pos` en la simulación `s`.
        points_table:     [n_drivers] — puntos por posición (0-indexed).
        n_finishers:      [n_sims] — número de pilotos clasificados por simulación.
                          Drivers en posición ≥ n_finishers[s] no puntúan.
                          None → todos son clasificados.

    Returns:
        points[s, driver_idx] = puntos del piloto driver_idx en simulación s.
    """
    n_sims, n_drivers = finishing_orders.shape
    n_pts = len(points_table)

    # Tabla extendida a n_drivers (posiciones fuera de la zona de puntos valen 0)
    pts_extended = np.zeros(n_drivers, dtype=float)
    pts_extended[:n_pts] = points_table[:n_drivers]

    # ranks[s, d] = posición del driver d en la simulación s (0 = ganador)
    ranks = _inverse_permutation_batch(finishing_orders)

    # points_by_pos[s, d] = pts_extended[rank de d en s]
    points_by_pos = pts_extended[ranks]

    # Anular puntos de pilotos DNF
    if n_finishers is not None:
        dnf_mask = ranks >= n_finishers[:, np.newaxis]   # [n_sims, n_drivers]
        points_by_pos = points_by_pos.copy()
        points_by_pos[dnf_mask] = 0.0

    return points_by_pos


def _inverse_permutation_batch(perm: np.ndarray) -> np.ndarray:
    """
    Calcula la permutación inversa de un lote de permutaciones.

    Si perm[s, pos] = driver, entonces rank[s, driver] = pos.

    Args:
        perm: [n_sims, n_drivers] — permutación de driver_idx por posición.

    Returns:
        rank: [n_sims, n_drivers] — posición de cada driver_idx.
    """
    n_sims, n_drivers = perm.shape
    rank = np.empty_like(perm)
    np.put_along_axis(
        rank,
        perm,
        np.broadcast_to(np.arange(n_drivers, dtype=perm.dtype), perm.shape),
        axis=1,
    )
    return rank
