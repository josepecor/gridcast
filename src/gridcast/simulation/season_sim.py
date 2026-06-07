"""
Simulación Monte Carlo del campeonato F1.

Qué hace (§4.5 CLAUDE.md):
  Para cada carrera restante → race_sim (con caos) → puntos.
  Repite N veces → distribución de puntos totales al final de la temporada.
  Resultados: P(título pilotos), P(título constructores), E[puntos], distribución.

Fines de semana sprint:
  Cada ronda sprint simula DOS carreras: sprint (tabla 8→1) + carrera principal (25→1).

Inputs esperados:
  strength DataFrame — columnas: driver_id (str), mean (float), sd (float, opcional).
                       Salida de models/strength.py. Si sd está ausente, se usa 1.0.
  driver_standings   — list[DriverStanding] de jolpica_client.
  constructor_standings — list[ConstructorStanding] de jolpica_client.
  remaining_rounds   — list[RoundInfo] de jolpica_client (rondas no disputadas aún).
  driver_constructor_map — dict[str, str]: driver_id → constructor_id.
                       Necesario para acumular puntos de constructores.
                       Si None, constructor_title_probs estará vacío.

Procesado en chunks:
  Se usan chunk_size (settings.yaml) simulaciones a la vez para controlar la memoria.
  Cada chunk usa su propia sub-secuencia aleatoria derivada del seed principal.
  Esto garantiza reproducibilidad independientemente de chunk_size.

Interfaz pública principal (usada por pipelines/05_simulate_championship.py):
  simulate_championship(strength, remaining_rounds, driver_standings,
                        constructor_standings, ..., n_sims, seed) -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.config import get_settings
from gridcast.ingest.jolpica_client import (
    ConstructorStanding,
    DriverStanding,
    RoundInfo,
)
from gridcast.simulation.chaos import ChaosParameters, simulate_race_with_chaos

log = logging.getLogger(__name__)

# Tasa de DNF por defecto cuando no se pasan datos de fiabilidad
_DEFAULT_DNF_RATE: float = 0.08

# σ del ruido Gumbel para Plackett-Luce (variabilidad de carrera)
_DEFAULT_NOISE_SCALE: float = 0.5

# sd de fortaleza para pilotos sin estimación del modelo
_UNKNOWN_STRENGTH_SD: float = 2.0


# ---------------------------------------------------------------------------
# Resultado de la simulación de campeonato
# ---------------------------------------------------------------------------

@dataclass
class SeasonSimResult:
    """
    Resultado de N simulaciones Monte Carlo del campeonato.

    Convención:
      driver_points_sims[s, d]    = puntos adicionales del piloto d en las rondas
                                    restantes durante la simulación s.
      driver_points_start[d]      = puntos ya acumulados por el piloto d en standings.
      driver_total[s, d]          = driver_points_start[d] + driver_points_sims[s, d].
    """
    driver_ids: list[str]
    constructor_ids: list[str]
    driver_points_sims: np.ndarray        # [n_sims, n_drivers]
    constructor_points_sims: np.ndarray   # [n_sims, n_constructors] (puede ser vacío)
    driver_points_start: np.ndarray       # [n_drivers]
    constructor_points_start: np.ndarray  # [n_constructors]
    n_rounds_simulated: int

    # ------------------------------------------------------------------
    # Propiedades derivadas
    # ------------------------------------------------------------------

    @property
    def n_sims(self) -> int:
        return self.driver_points_sims.shape[0]

    @property
    def n_drivers(self) -> int:
        return len(self.driver_ids)

    @property
    def n_constructors(self) -> int:
        return len(self.constructor_ids)

    @property
    def driver_total(self) -> np.ndarray:
        """[n_sims, n_drivers] — total de puntos de temporada por piloto."""
        return self.driver_points_start[np.newaxis, :] + self.driver_points_sims

    @property
    def constructor_total(self) -> np.ndarray:
        """[n_sims, n_constructors] — total de puntos de temporada por constructor."""
        return (
            self.constructor_points_start[np.newaxis, :]
            + self.constructor_points_sims
        )

    # ------------------------------------------------------------------
    # Probabilidades de título
    # ------------------------------------------------------------------

    def driver_title_probability(self) -> dict[str, float]:
        """
        P(piloto gana el campeonato) para cada driver_id.

        El ganador de cada simulación es el piloto con más puntos totales al final.
        En caso de empate en puntos, el ganador real se determina por número de
        victorias (no modelado aquí → el primer piloto en el array gana el tie).
        """
        total = self.driver_total                         # [n_sims, n_drivers]
        winners = total.argmax(axis=1)                    # [n_sims]
        counts = np.bincount(winners, minlength=self.n_drivers)
        return dict(zip(self.driver_ids, (counts / self.n_sims).tolist()))

    def constructor_title_probability(self) -> dict[str, float]:
        """P(constructor gana el campeonato). Vacío si no hay driver_constructor_map."""
        if self.n_constructors == 0:
            return {}
        total = self.constructor_total
        winners = total.argmax(axis=1)
        counts = np.bincount(winners, minlength=self.n_constructors)
        return dict(zip(self.constructor_ids, (counts / self.n_sims).tolist()))

    # ------------------------------------------------------------------
    # Puntos esperados
    # ------------------------------------------------------------------

    def driver_expected_total_points(self) -> dict[str, float]:
        """E[puntos totales de temporada] por piloto."""
        return dict(zip(self.driver_ids, self.driver_total.mean(axis=0).tolist()))

    def driver_expected_additional_points(self) -> dict[str, float]:
        """E[puntos adicionales de rondas restantes] por piloto."""
        return dict(zip(self.driver_ids, self.driver_points_sims.mean(axis=0).tolist()))

    # ------------------------------------------------------------------
    # Número mágico
    # ------------------------------------------------------------------

    def magic_number(
        self,
        driver_id: str,
        max_pts_per_round: float = 33.0,
    ) -> float | None:
        """
        Diferencia de puntos que necesita el líder para matemáticamente clinchar.

        Fórmula clásica (determinista):
          max_achievable_by_2nd = 2nd_pts_now + remaining_rounds × max_per_round
          magic_number = leader_pts - max_achievable_by_2nd

        Si magic_number > 0 → el piloto ya ha clinchado matemáticamente.
        Si magic_number ≤ 0 → quedan posibilidades para el resto.

        Args:
            driver_id:          Piloto para el que calcular el número mágico.
            max_pts_per_round:  Puntos máximos posibles por ronda.
                                Sprint weekend: 8 (sprint) + 25 (race) = 33.
                                Normal weekend: 25. Default=33 (peor caso conservador).

        Returns:
            float si driver_id está en la lista; None si no se encuentra.
        """
        if driver_id not in self.driver_ids:
            return None
        d_idx = self.driver_ids.index(driver_id)
        leader_pts = float(self.driver_points_start[d_idx])

        max_second = max(
            float(self.driver_points_start[i])
            for i in range(self.n_drivers)
            if i != d_idx
        ) + self.n_rounds_simulated * max_pts_per_round

        return leader_pts - max_second

    # ------------------------------------------------------------------
    # DataFrames de resumen
    # ------------------------------------------------------------------

    def to_driver_summary_df(self) -> pd.DataFrame:
        """DataFrame ordenado por probabilidad de título descendente."""
        probs  = self.driver_title_probability()
        exp_pt = self.driver_expected_total_points()
        df = pd.DataFrame({
            "driver_id":      self.driver_ids,
            "current_points": self.driver_points_start.tolist(),
            "title_prob":     [probs[d] for d in self.driver_ids],
            "expected_total": [exp_pt[d] for d in self.driver_ids],
        })
        return df.sort_values("title_prob", ascending=False).reset_index(drop=True)

    def to_constructor_summary_df(self) -> pd.DataFrame:
        """DataFrame de constructores. Vacío si no hay mapeo driver→constructor."""
        if self.n_constructors == 0:
            return pd.DataFrame(
                columns=["constructor_id", "current_points", "title_prob"]
            )
        probs = self.constructor_title_probability()
        df = pd.DataFrame({
            "constructor_id": self.constructor_ids,
            "current_points": self.constructor_points_start.tolist(),
            "title_prob":     [probs.get(c, 0.0) for c in self.constructor_ids],
        })
        return df.sort_values("title_prob", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# API pública (interfaz usada por pipeline 05)
# ---------------------------------------------------------------------------

def simulate_championship(
    strength: pd.DataFrame,
    remaining_rounds: list[RoundInfo],
    driver_standings: list[DriverStanding],
    constructor_standings: list[ConstructorStanding],
    driver_constructor_map: dict[str, str] | None = None,
    dnf_rates: dict[str, float] | None = None,
    chaos_params: ChaosParameters | None = None,
    n_sims: int | None = None,
    seed: int | None = None,
    car_development_sigma: float | None = None,
) -> dict:
    """
    Simula el campeonato restante N veces y devuelve distribuciones de probabilidad.

    Args:
        strength:              DataFrame con fortalezas del modelo.
                               Columnas requeridas: driver_id, mean.
                               Columnas opcionales: sd (default 1.0 si ausente).
        remaining_rounds:      Rondas no disputadas aún (jolpica_client.get_schedule()).
        driver_standings:      Standings actuales de pilotos (jolpica_client.get_driver_standings()).
        constructor_standings: Standings actuales de constructores.
        driver_constructor_map: driver_id → constructor_id para asignar puntos de
                               constructor. Si None, constructor_title_probs = {}.
        dnf_rates:             driver_id → P(DNF) por carrera. Si None, usa default 8%.
        chaos_params:          Parámetros de caos. None → defaults de ChaosParameters.
        n_sims:                Número de simulaciones. None → settings.simulation.n_sims.
        seed:                  Semilla aleatoria. None → settings.simulation.seed.

    Returns:
        dict con:
          'driver_title_probs'    : {driver_id: float}
          'constructor_title_probs': {constructor_id: float}
          'driver_expected_points' : {driver_id: float}
          'driver_points_dist'     : pd.DataFrame [n_sims × n_drivers] de puntos totales
    """
    cfg = get_settings()
    if n_sims is None:
        n_sims = cfg.simulation.n_sims
    if seed is None:
        seed = cfg.simulation.seed
    if chaos_params is None:
        chaos_params = ChaosParameters()
    if car_development_sigma is None:
        car_development_sigma = cfg.simulation.car_development_sigma

    # SeedSequence garantiza sub-secuencias reproducibles e independientes por chunk
    ss = np.random.SeedSequence(seed)

    # --- Construir arrays alineados ---
    driver_ids, strengths, strength_sds, driver_pts_start = _build_driver_arrays(
        strength, driver_standings
    )
    n_drivers = len(driver_ids)

    if n_drivers == 0:
        log.warning("Sin pilotos — devolviendo resultado vacío")
        return {
            "driver_title_probs":     {},
            "constructor_title_probs": {},
            "driver_expected_points":  {},
            "driver_points_dist":      pd.DataFrame(),
        }

    constructor_ids, driver_constr_idx, constr_pts_start = _build_constructor_arrays(
        constructor_standings, driver_constructor_map, driver_ids
    )

    dnf_arr = _build_dnf_array(dnf_rates, driver_ids)

    n_rounds = len(remaining_rounds)
    log.info(
        "Simulando %d rondas × %d sims × %d pilotos ...",
        n_rounds, n_sims, n_drivers,
    )

    if n_rounds == 0:
        log.info("Sin rondas restantes — standings actuales son el resultado final")
        result = SeasonSimResult(
            driver_ids=driver_ids,
            constructor_ids=constructor_ids,
            driver_points_sims=np.zeros((n_sims, n_drivers)),
            constructor_points_sims=np.zeros((n_sims, max(len(constructor_ids), 1))),
            driver_points_start=driver_pts_start,
            constructor_points_start=constr_pts_start,
            n_rounds_simulated=0,
        )
        return _to_output_dict(result, driver_ids, driver_pts_start, n_sims,
                               has_constructor_map=(driver_constructor_map is not None))

    # --- Simulación en chunks ---
    chunk_size = cfg.simulation.chunk_size
    n_chunks = (n_sims + chunk_size - 1) // chunk_size
    chunk_seeds = ss.spawn(n_chunks)

    all_driver_pts = np.zeros((n_sims, n_drivers))

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end   = min(chunk_start + chunk_size, n_sims)
        chunk_n     = chunk_end - chunk_start
        chunk_rng   = np.random.default_rng(chunk_seeds[chunk_idx])

        chunk_pts = _simulate_remaining_rounds(
            driver_ids=driver_ids,
            strengths=strengths,
            strength_sds=strength_sds,
            dnf_rates=dnf_arr,
            remaining_rounds=remaining_rounds,
            n_sims=chunk_n,
            noise_scale=_DEFAULT_NOISE_SCALE,
            chaos_params=chaos_params,
            rng=chunk_rng,
            car_development_sigma=car_development_sigma,
        )
        all_driver_pts[chunk_start:chunk_end] = chunk_pts

    # --- Puntos de constructor (suma de pilotos del equipo) ---
    n_constr = len(constructor_ids)
    all_constr_pts = np.zeros((n_sims, max(n_constr, 1)))
    if driver_constr_idx is not None and n_constr > 0:
        for c_idx in range(n_constr):
            d_mask = driver_constr_idx == c_idx
            if d_mask.any():
                all_constr_pts[:, c_idx] = all_driver_pts[:, d_mask].sum(axis=1)

    result = SeasonSimResult(
        driver_ids=driver_ids,
        constructor_ids=constructor_ids,
        driver_points_sims=all_driver_pts,
        constructor_points_sims=all_constr_pts[:, :max(n_constr, 1)],
        driver_points_start=driver_pts_start,
        constructor_points_start=constr_pts_start,
        n_rounds_simulated=n_rounds,
    )

    log.info(
        "Simulación completada. Líder tentativo: %s (%.1f%% de títulos)",
        _leading_driver(result), _leading_prob(result),
    )

    return _to_output_dict(result, driver_ids, driver_pts_start, n_sims,
                           has_constructor_map=(driver_constructor_map is not None))


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _simulate_remaining_rounds(
    driver_ids: list[str],
    strengths: np.ndarray,
    strength_sds: np.ndarray | None,
    dnf_rates: np.ndarray,
    remaining_rounds: list[RoundInfo],
    n_sims: int,
    noise_scale: float,
    chaos_params: ChaosParameters,
    rng: np.random.Generator,
    car_development_sigma: float = 0.0,
) -> np.ndarray:
    """
    Simula todas las rondas restantes para un chunk de n_sims simulaciones.

    car_development_sigma: σ del drift de desarrollo del coche. Se samplea una vez
    por simulación (mismo valor para todas las carreras de esa sim), representando
    la incertidumbre sobre la evolución de la jerarquía de coches durante la temporada.
    Con más carreras restantes el drift acumula más efecto sobre los puntos finales
    → la incertidumbre del campeonato aumenta de forma emergente.

    Returns:
        [n_sims, n_drivers] — puntos adicionales acumulados sobre las rondas restantes.
    """
    n_drivers = len(driver_ids)
    chunk_pts = np.zeros((n_sims, n_drivers))

    # dev_noise[s, d]: offset de fortaleza para el piloto d en la simulación s.
    # Se samplea ANTES del bucle de carreras → misma jerarquía de coches en toda la sim.
    dev_noise: np.ndarray | None = None
    if car_development_sigma > 0.0:
        dev_noise = rng.standard_normal((n_sims, n_drivers)) * car_development_sigma

    for round_info in remaining_rounds:
        if round_info.has_sprint:
            sprint = simulate_race_with_chaos(
                driver_ids=driver_ids,
                strengths=strengths,
                strength_sds=strength_sds,
                dnf_rates=dnf_rates,
                is_sprint=True,
                n_sims=n_sims,
                base_noise_scale=noise_scale,
                params=chaos_params,
                rng=rng,
                dev_noise=dev_noise,
            )
            chunk_pts += sprint.points

        race = simulate_race_with_chaos(
            driver_ids=driver_ids,
            strengths=strengths,
            strength_sds=strength_sds,
            dnf_rates=dnf_rates,
            is_sprint=False,
            n_sims=n_sims,
            base_noise_scale=noise_scale,
            params=chaos_params,
            rng=rng,
            dev_noise=dev_noise,
        )
        chunk_pts += race.points

    return chunk_pts


def _build_driver_arrays(
    strength_df: pd.DataFrame,
    driver_standings: list[DriverStanding],
) -> tuple[list[str], np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Construye arrays alineados de driver_ids, strengths, strength_sds y puntos actuales.

    Los pilotos en standings son la lista canónica; si un piloto no tiene estimación
    de fortaleza, se usa mean=0.0 con sd=_UNKNOWN_STRENGTH_SD (mucha incertidumbre).

    Returns:
        (driver_ids, strengths[n], strength_sds[n], pts_start[n])
    """
    # Índice de strength por driver_id
    str_index: dict[str, dict] = {}
    if not strength_df.empty and "driver_id" in strength_df.columns:
        for _, row in strength_df.iterrows():
            str_index[row["driver_id"]] = row.to_dict()

    driver_ids: list[str] = []
    strengths_list: list[float] = []
    sds_list: list[float] = []
    pts_list: list[float] = []

    for standing in driver_standings:
        driver_ids.append(standing.driver_id)
        pts_list.append(float(standing.points))

        entry = str_index.get(standing.driver_id, {})
        strengths_list.append(float(entry.get("mean", 0.0)))
        sds_list.append(float(entry.get("sd", _UNKNOWN_STRENGTH_SD)))

        if standing.driver_id not in str_index:
            log.debug(
                "Piloto %s no tiene estimación de fortaleza → mean=0, sd=%.1f",
                standing.driver_id, _UNKNOWN_STRENGTH_SD,
            )

    # strength.py ya devuelve escala "higher = better" (negación de lap time incorporada).
    # race_sim espera "mayor = mejor" directamente — no negar.
    strengths = np.array(strengths_list)
    strength_sds = np.array(sds_list)
    pts_start = np.array(pts_list)

    # Si todos los sd son _UNKNOWN_STRENGTH_SD (sin modelo), devolver None para sds
    # → simulate_race usará valores puntuales (menos ruido por estimación)
    all_unknown = np.all(strength_sds == _UNKNOWN_STRENGTH_SD)
    if all_unknown:
        log.warning(
            "Ningún piloto tiene sd de fortaleza — usando estimaciones puntuales. "
            "Ejecuta 04_fit_models.py para obtener incertidumbre del modelo."
        )
        strength_sds_out = None
    else:
        strength_sds_out = strength_sds

    return driver_ids, strengths, strength_sds_out, pts_start


def _build_constructor_arrays(
    constructor_standings: list[ConstructorStanding],
    driver_constructor_map: dict[str, str] | None,
    driver_ids: list[str],
) -> tuple[list[str], np.ndarray | None, np.ndarray]:
    """
    Construye la lista de constructores y el mapeo driver→constructor_idx.

    Returns:
        (constructor_ids, driver_constr_idx, constr_pts_start)
        driver_constr_idx[d] = índice en constructor_ids del piloto d.
                               -1 si el piloto no está mapeado.
        constr_pts_start[c]  = puntos actuales del constructor c.
    """
    if not constructor_standings:
        return [], None, np.array([])

    constructor_ids = [cs.constructor_id for cs in constructor_standings]
    constr_idx      = {c: i for i, c in enumerate(constructor_ids)}
    constr_pts      = np.array([float(cs.points) for cs in constructor_standings])

    if driver_constructor_map is None:
        log.debug("Sin driver_constructor_map — puntos de constructor no calculados")
        return constructor_ids, None, constr_pts

    driver_constr_idx = np.full(len(driver_ids), -1, dtype=np.intp)
    for d_idx, d_id in enumerate(driver_ids):
        c_id = driver_constructor_map.get(d_id)
        if c_id is not None and c_id in constr_idx:
            driver_constr_idx[d_idx] = constr_idx[c_id]
        else:
            log.debug("Piloto %s sin mapeo a constructor", d_id)

    return constructor_ids, driver_constr_idx, constr_pts


def _build_dnf_array(
    dnf_rates: dict[str, float] | None,
    driver_ids: list[str],
) -> np.ndarray:
    """
    Construye el array de tasas de DNF alineado con driver_ids.

    Si dnf_rates es None o un piloto no aparece, usa _DEFAULT_DNF_RATE.
    """
    if dnf_rates is None:
        return np.full(len(driver_ids), _DEFAULT_DNF_RATE)

    return np.array([
        dnf_rates.get(d, _DEFAULT_DNF_RATE)
        for d in driver_ids
    ])


def _to_output_dict(
    result: SeasonSimResult,
    driver_ids: list[str],
    driver_pts_start: np.ndarray,
    n_sims: int,
    has_constructor_map: bool = True,
) -> dict:
    """Convierte SeasonSimResult al dict esperado por pipeline 05."""
    driver_pts_dist = pd.DataFrame(
        result.driver_total,
        columns=driver_ids,
    )
    # Sin mapeo driver→constructor no se pueden calcular probabilidades de título
    # de constructor basadas en puntos simulados — devolver vacío como documenta la API.
    constr_title = result.constructor_title_probability() if has_constructor_map else {}
    return {
        "driver_title_probs":      result.driver_title_probability(),
        "constructor_title_probs": constr_title,
        "driver_expected_points":  result.driver_expected_total_points(),
        "driver_points_dist":      driver_pts_dist,
    }


def _leading_driver(result: SeasonSimResult) -> str:
    probs = result.driver_title_probability()
    if not probs:
        return "—"
    return max(probs, key=lambda d: probs[d])


def _leading_prob(result: SeasonSimResult) -> float:
    probs = result.driver_title_probability()
    if not probs:
        return 0.0
    return max(probs.values()) * 100
