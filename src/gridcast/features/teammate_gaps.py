"""
Gaps entre compañeros de equipo — backbone de la separación coche / piloto.

La única comparación con coche constante es la de los dos pilotos del mismo equipo.
Encadenando estos gaps a través de varias temporadas (red de compañeros compartidos)
se puede situar a todos los pilotos en una escala común, independientemente de la
fuerza relativa de los coches.

Dos gaps (§4.1, §7 CLAUDE.md):
  gap_to_teammate_quali — diferencia en quali_pace_s. Señal más limpia porque
                          elimina ruido de estrategia y caos de carrera.
  gap_to_teammate_race  — diferencia en long_run_pace_s. Más ruidoso pero
                          recoge dimensiones distintas (gestión de neumáticos, etc.).

Convención de signo: negativo = el piloto es más rápido que su compañero.
Los dos gaps son simétricos: gap_A + gap_B = 0 si ambos tienen datos válidos.

Input: salida de features/pace.compute_pace_features()
Output: mismo índice (abbreviation × round), columnas gap_to_teammate_*
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

_REQUIRED_COLS = {"abbreviation", "team_name", "round"}

# Si un piloto es más de N segundos más lento que su compañero de equipo,
# el dato se considera sospechoso (¿problema de coche, accidente en sesión?).
_INTRA_TEAM_SUSPECT_THRESHOLD = {
    "long_run_pace_s": 3.0,
    "quali_pace_s":    1.5,
}


def compute_teammate_gaps(pace_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula gaps entre compañeros de equipo para cada ronda.

    Args:
        pace_features: Salida de pace.compute_pace_features(). Columnas requeridas:
                       abbreviation, team_name, round.
                       Columnas usadas si presentes: long_run_pace_s, quali_pace_s.

    Returns:
        DataFrame con una fila por piloto × ronda, columnas:
          abbreviation, round,
          gap_to_teammate_race   (NaN si alguno de los dos no tiene long_run_pace_s)
          gap_to_teammate_quali  (NaN si alguno de los dos no tiene quali_pace_s)

    Raises:
        ValueError: si pace_features no tiene las columnas mínimas requeridas.
    """
    missing = _REQUIRED_COLS - set(pace_features.columns)
    if missing:
        raise ValueError(
            f"pace_features le faltan columnas: {missing}. "
            "Usa la salida de features.pace.compute_pace_features()."
        )

    if pace_features.empty:
        return pd.DataFrame(
            columns=[
                "abbreviation", "round",
                "gap_to_teammate_race", "gap_to_teammate_quali",
            ]
        )

    has_race  = "long_run_pace_s" in pace_features.columns
    has_quali = "quali_pace_s"    in pace_features.columns

    rows: list[dict] = []

    for (round_, team), group in pace_features.groupby(["round", "team_name"]):
        group = group.reset_index(drop=True)
        n = len(group)

        if n == 1:
            log.debug(
                "R%s equipo '%s': un solo piloto (%s) — gaps = NaN",
                round_, team, group.iloc[0]["abbreviation"],
            )
            rows.append(_nan_row(group.iloc[0]["abbreviation"], round_))
            continue

        if n > 2:
            log.warning(
                "R%s equipo '%s': %d pilotos detectados (esperado 2) — "
                "usando los primeros dos",
                round_, team, n,
            )
            group = group.head(2)

        a, b = group.iloc[0], group.iloc[1]

        gap_race  = _gap(a, b, "long_run_pace_s") if has_race  else float("nan")
        gap_quali = _gap(a, b, "quali_pace_s")    if has_quali else float("nan")

        rows.append({
            "abbreviation":         a["abbreviation"],
            "round":                round_,
            "gap_to_teammate_race":  gap_race,
            "gap_to_teammate_quali": gap_quali,
        })
        rows.append({
            "abbreviation":         b["abbreviation"],
            "round":                round_,
            "gap_to_teammate_race":  _neg(gap_race),
            "gap_to_teammate_quali": _neg(gap_quali),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "abbreviation", "round",
                "gap_to_teammate_race", "gap_to_teammate_quali",
            ]
        )

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gap(a: pd.Series, b: pd.Series, col: str) -> float:
    va, vb = a.get(col, float("nan")), b.get(col, float("nan"))
    if pd.notna(va) and pd.notna(vb):
        return float(va) - float(vb)
    return float("nan")


def _neg(v: float) -> float:
    return -v if pd.notna(v) else float("nan")


def _nan_row(abbreviation: str, round_) -> dict:
    return {
        "abbreviation":         abbreviation,
        "round":                round_,
        "gap_to_teammate_race":  float("nan"),
        "gap_to_teammate_quali": float("nan"),
    }


def flag_intra_team_suspects(pace_features: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas `long_run_suspect` y `quali_suspect` al DataFrame.

    Para cada par ronda × equipo con exactamente 2 pilotos con datos válidos:
    si la diferencia entre compañeros supera el umbral (_INTRA_TEAM_SUSPECT_THRESHOLD),
    el piloto más lento se marca como sospechoso para ese signal.

    El flag indica que el valor de pace de ese piloto puede no reflejar
    el rendimiento real del coche (ej. coche dañado, sesión comprometida)
    y no debe usarse como señal intra-equipo en el modelo de fuerza.

    Returns:
        pace_features con las dos nuevas columnas booleanas añadidas.
    """
    df = pace_features.copy()
    df["long_run_suspect"] = False
    df["quali_suspect"]    = False

    col_flag_map = {
        "long_run_pace_s": "long_run_suspect",
        "quali_pace_s":    "quali_suspect",
    }

    for signal_col, flag_col in col_flag_map.items():
        if signal_col not in df.columns:
            continue
        threshold = _INTRA_TEAM_SUSPECT_THRESHOLD[signal_col]

        for (round_, team), group in df.groupby(["round", "team_name"]):
            valid = group.dropna(subset=[signal_col])
            if len(valid) != 2:
                continue

            a, b = valid.iloc[0], valid.iloc[1]
            gap = abs(float(a[signal_col]) - float(b[signal_col]))

            if gap > threshold:
                slower_abbr = (
                    a["abbreviation"]
                    if float(a[signal_col]) > float(b[signal_col])
                    else b["abbreviation"]
                )
                mask = (df["round"] == round_) & (df["abbreviation"] == slower_abbr)
                df.loc[mask, flag_col] = True
                log.warning(
                    "R%02d %s: gap intra-equipo %.2fs en %s (umbral %.1fs) — "
                    "%s marcado como suspect",
                    round_, team, gap, signal_col, threshold, slower_abbr,
                )

    n_lr = int(df["long_run_suspect"].sum())
    n_q  = int(df["quali_suspect"].sum())
    if n_lr or n_q:
        log.warning(
            "Flags suspect totales: long_run=%d, quali=%d",
            n_lr, n_q,
        )

    return df
