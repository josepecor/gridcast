# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Estado del proyecto (junio 2026): bootstrapping.** Solo existe este fichero. Todo lo del §6 (árbol de directorios), §7 (tablas) y §8 (comandos) es el **diseño objetivo**, no código existente. Al arrancar una sesión, verifica qué existe realmente con `find . -type f`.

> Este fichero es la **fuente de verdad** del proyecto. Si algo del código contradice este documento, gana este documento (o actualiza este documento en el mismo PR).

---

## 1. Qué es esto

Sistema de predicción para el **Campeonato del Mundo de F1 2026**. Dos productos:

1. **Predicción por carrera** → distribución de probabilidad sobre el orden de llegada de cada piloto (no un único resultado).
2. **Simulación del campeonato** (la pieza diferenciadora) → Monte Carlo de la temporada restante para obtener probabilidades de título de pilotos y constructores, puntos esperados, y escenarios ("número mágico", carrera en la que se decide, etc.).

La predicción por carrera **no es un producto separado**: es el motor que alimenta la simulación. Un único modelo de "ritmo + incertidumbre + riesgo de abandono" sirve para las dos cosas.

### Principio rector
Lo que importa **no es el acierto puntual, es la calibración**. Una probabilidad del 30% debe cumplirse ~3 de cada 10 veces. Todo se evalúa con Brier score / log-loss y curvas de fiabilidad. Un modelo que "acierta el ganador" pero está mal calibrado es inútil para la simulación.

---

## 2. El handicap de 2026 (leer antes de tocar nada)

Hay **dos** condicionantes que atraviesan todo el diseño:

### 2.1. Reseteo reglamentario total
2026 es el mayor cambio de reglas de la historia de la F1: chasis nuevo, motores nuevos (reparto ~50/50 combustión/eléctrico, sin MGU-H), aerodinámica activa en vez de DRS, 11 equipos (debuta Cadillac), reorganización de motoristas.

**Consecuencia para el modelo:** los datos históricos 2022-2025 valen muy poco para la **fuerza del coche**. La habilidad del **piloto** sí transfiere. Por tanto:
- Priors de piloto → arrancan informados desde 2022-2025.
- Priors de coche → arrancan **anchos** (mucha incertidumbre) y se aprenden rápido desde los test de pretemporada + primeras carreras de 2026.

### 2.2. La temporada ya ha empezado → backtest con congelado point-in-time
El proyecto se construye con la temporada en marcha, pero el objetivo es tener un **registro completo desde la ronda 1**. Esto se hace con un **backtest walk-forward**, NO inventando predicciones.

**Regla absoluta — point-in-time correctness (sin leakage):**
Para cada ronda ya disputada, la predicción se genera usando **exclusivamente datos disponibles antes del corte de esa ronda** (antes de su primera sesión, o antes de cada sesión si se predice a nivel sesión). Después se **congela** como artefacto inmutable.

- Un `leakage_guard` debe **fallar (assert)** si cualquier fila con fecha ≥ corte entra en el set de features de esa ronda.
- Las predicciones congeladas son honestas: son lo que el modelo *habría* producido con la información de entonces.
- Nunca se reescribe un snapshot ya congelado. Si cambia el modelo, se versiona y se re-genera **todo** el histórico bajo la nueva versión, etiquetado.

Esto es lo que hace que el registro retrospectivo sea legítimo y evaluable, en vez de un ajuste a posteriori.

---

## 3. Reglas de dominio (hechos que el código debe respetar)

### 3.1. Sistema de puntos (config/scoring.yaml)
- **Carrera (top 10):** 25, 18, 15, 12, 10, 8, 6, 4, 2, 1.
- **NO existe punto por vuelta rápida** (eliminado desde 2025). No lo modeles.
- **Sprint (top 8):** 8, 7, 6, 5, 4, 3, 2, 1.

### 3.2. Formatos de fin de semana (src/gridcast/data/sessions.py)
El pipeline DEBE detectar el formato y adaptarse:
- **Normal:** FP1, FP2, FP3, Clasificación, Carrera.
- **Sprint:** FP1, Sprint Quali, Sprint, Clasificación, Carrera → **solo UNA sesión de libres**.

### 3.3. Valor predictivo por sesión y diferencia entre formatos

**Fin de semana normal:** FP1 → FP2 → FP3 → Q → R
**Fin de semana sprint:** FP1 → SQ → S → Q → R  ← **FP1 es la ÚNICA práctica**

| Sesión | Normal | Sprint | Notas |
|---|---|---|---|
| FP1 | Existe | Existe | Ver regla rookies abajo |
| FP2 | Existe ★ | **NO existe** | Sesión de referencia en normal |
| FP3 | Existe | **NO existe** | Prep clasificación en normal |
| SQ (Sprint Quali) | NO existe | Existe | |
| S (Sprint) | NO existe | Existe | |

**Regla de rookies en FP1 — solo aplica en fines de semana normales:**
La FIA obliga a equipos a rodar novatos hasta 4 sesiones de FP1 por temporada. En fines de semana sprint **no aplica**: FP1 es la única práctica y los equipos necesitan al titular para preparar el coche. Por tanto:
- `es_titular = False` solo puede ocurrir en FP1 de formato `conventional`
- FP1 sprint → siempre `es_titular = True`, no hace falta verificar
- Obligatorio: tabla `driver_session_map`. Las vueltas con `es_titular = False` se **excluyen o ponderan a la baja** para estimar habilidad del piloto, pero **pueden** usarse para estimar el coche.

**Sesión de pace primaria por formato:**
- Normal → **FP2** (sim quali + sim carrera en la misma sesión; el ritmo de tanda larga es el predictor más fuerte)
- Sprint → **FP1** (única opción; datos más escasos y ruidosos que FP2 en normal)

### 3.5. Cutoffs de predicción por fin de semana

Solo se predice la **carrera principal**. Los cutoffs son los momentos en que se genera una nueva predicción, usando exclusivamente datos disponibles en ese instante.

**Fin de semana normal (2 cutoffs):**

| Cutoff | Datos disponibles | Salto de información |
|---|---|---|
| `pre-fp1` | Historial + test + carreras anteriores de la temporada | Baseline pre-finde |
| `post-q` | + ritmo FP1/FP2/FP3 + parrilla provisional de clasificación | Gran salto: grid conocido |

**Fin de semana sprint (3 cutoffs):**

| Cutoff | Datos disponibles | Salto de información |
|---|---|---|
| `pre-fp1` | Historial + test + carreras anteriores | Baseline pre-finde |
| `post-sprint` | + ritmo FP1 + resultado del sprint | Señal de ritmo real en condiciones de carrera |
| `post-q` | + parrilla provisional de clasificación | Grid conocido |

**Limitación conocida del cutoff `post-q` — parrilla no totalmente final:**

La parrilla que ofrecen FastF1/Jolpica tras la clasificación es la parrilla **provisional**, no la definitiva. Entre la clasificación y la salida pueden ocurrir:

1. **Penalizaciones de stewards** (incidentes en clasificación) → desplazan posiciones. FastF1 las recoge en los resultados de carrera (`GridPosition`), disponibles solo tras la carrera.
2. **Penalizaciones de componentes PU** confirmadas después de quali → ídem.
3. **Salida desde el pit lane** por problema técnico en los últimos minutos antes de la salida → imposible de capturar automáticamente; ocurre cuando se cierra el pit lane.

**Decisión de diseño:** el cutoff `post-q` usa la parrilla provisional de clasificación como proxy. Las penalizaciones conocidas de componentes se aplican si están registradas en `component_usage`. Las salidas desde boxes de último momento se tratan como ruido del modelo (contribuyen a la varianza del chaos, no se intentan predecir). No es un error del sistema: es una limitación honesta documentada.

### 3.4. Componentes de unidad de potencia y penalizaciones (config/components.yaml)
Los límites son **datos de referencia fijos** (hardcode), no hay que capturarlos. Asignación por piloto en 2026 (incluye el "+1 bonus" por ser año de motor nuevo):

| Componente | Asignación 2026 | Cae a en 2027 |
|---|---|---|
| Motor de combustión (ICE) | 4 (3+1) | 3 |
| Turbocompresor (TC) | 4 (3+1) | 3 |
| Escape (EX) | 4 (3+1) | 3 |
| MGU-K | 3 (2+1) | 2 |
| Energy Store (ES) | 3 (2+1) | 2 |
| Control Electronics (CE) | 3 (2+1) | 2 |
| MGU-H | — eliminado en 2026 — | — |

- Caja de cambios: límites propios, modelar aparte.
- Un fabricante en su primer año suministrando recibe asignación bonus adicional.
- **Penalizaciones:** primer elemento extra de un tipo = **10 puestos**; siguientes = **5 puestos**; acumulativas; si la sanción total supera **15 puestos** → salida desde el **fondo de parrilla**.

**Captura del uso acumulado:** NO está en datasets limpios (FastF1/Jolpica no lo dan). Opciones, por orden de robustez:
1. Modelar el **riesgo de penalización** como hazard (ver 4.4) — recomendado para la simulación.
2. Acumular eventos de penalización observados (recuperables de resultados/prensa).
3. (Opcional, frágil) Scrapear documentos técnicos de la FIA por evento.

---

## 4. Arquitectura del modelo

### 4.1. Separación coche vs piloto (el núcleo)
Descomposición del rendimiento:

```
rendimiento = efecto_equipo + efecto_piloto + (piloto × tipo_circuito) + ruido

efecto_equipo = efecto_chasis + efecto_motorista + ejecución_operativa
```

- **Modelo jerárquico / mixed-effects** (piloto y equipo como efectos aleatorios). Da partial pooling (rookies se encogen a la media) e incertidumbre por estimación.
- **Backbone = teammate gaps**: la única comparación con coche constante son los dos pilotos del mismo equipo. Se encadenan a través de la red de compañeros (varias temporadas) para situar a todos en una escala común.
- **Usa preferentemente el gap en CLASIFICACIÓN**, no en carrera: menos ruido de estrategia/caos → señal de piloto más limpia.
- **`efecto_motorista` es compartido** entre equipos cliente del mismo motor. En 2026 esto es probablemente la variable más predictiva: si un motorista clava la PU, sube a todos sus clientes a la vez.

### 4.2. Fuerza de coche en el tiempo
Variable, no constante: los coches evolucionan durante la temporada. Modelar como **estado que se actualiza carrera a carrera** (state-space / rolling). Arranca ancho en 2026 y aprende rápido.

### 4.3. Modelo de carrera → distribución de orden de llegada
Por piloto y GP: `pace_estimada(equipo, piloto, circuito) + ruido + prob_DNF`.
Para samplear órdenes respetando que es una permutación: **Plackett-Luce** (asignar "fuerza", sumar ruido, ordenar).

### 4.4. Riesgo de penalización (varianza realista)
Probabilidad de que un piloto tome sanción en un GP = función de: kilometraje sobre componentes actuales + tasa de fiabilidad del equipo/motorista + momento de temporada (las sanciones se concentran al final). En 2026 ↑ por motores nuevos.

### 4.5. Simulación
- **race_sim:** para una carrera, samplea N órdenes desde el modelo + caos.
- **chaos:** safety car, lluvia, incidentes 1ª vuelta, DNF, **penalización de parrilla** (se sortea ANTES de la carrera y modifica la parrilla; luego se corre desde ahí).
- **season_sim:** para cada carrera restante, race_sim + reparto de puntos; acumula sobre la temporada; repetir 10k–50k veces → probabilidades de título.
- Sin caos, el favorito gana el 99% → irreal. El caos es feature, no bug.

---

## 5. Stack

- **Python 3.11+**, gestión con `uv` (o `pip` + `pyproject.toml`).
- **Datos — dos fuentes con roles distintos:**
  - `fastf1` → **fuente primaria de todo dato de sesión**: lap times, sectores, telemetría, tandas largas, resultados de sesión (posiciones, DNF, estado). FastF1 lee directamente `livetiming.formula1.com` (API oficial F1), disponible minutos/horas después de cada sesión. Caché local en `data/raw/fastf1_cache/`.
  - **Jolpica-F1 API** → **solo calendario y standings históricos**. Actualiza *una vez por fin de semana (lunes post-carrera)*; no usar para datos de sesión en caliente. Usarlo para: calendario (round numbers, fechas de corte), standings acumulados como verificación. `jolpica_id` del circuito es la clave de join estable.
  - (Opcional) documentos FIA → uso de componentes/penalizaciones.
- **Datos tabulares:** `pandas` / `polars`, almacenamiento en `parquet`. `duckdb` o `sqlite` para el store point-in-time.
- **Modelado:** `numpyro` o `pymc` para el jerárquico bayesiano (necesitamos incertidumbre para el Monte Carlo). Baseline rápido: `statsmodels` MixedLM.
- **Simulación:** `numpy` (vectorizado).
- **Config:** `yaml` + `pydantic` para validar.
- **Evaluación:** métricas propias (Brier, log-loss) + curvas de calibración.
- **Viz/entrega:** `plotly`/`matplotlib`; opcional dashboard `streamlit`.
- **Tests:** `pytest`.

---

## 6. Árbol de directorios

```
gridcast/
├── CLAUDE.md                      # este fichero
├── README.md
├── pyproject.toml
├── .env.example                  # claves API si hicieran falta; NO subir .env
├── config/
│   ├── settings.yaml             # rutas, temporada activa, nº simulaciones, seeds
│   ├── scoring.yaml              # puntos carrera + sprint (sin vuelta rápida)
│   ├── components.yaml           # asignaciones PU 2026 + reglas de penalización
│   └── circuits.yaml            # metadatos por circuito (tipo, sensibilidad a potencia/carga)
├── data/
│   ├── raw/                      # inmutable; pulls crudos
│   │   ├── fastf1_cache/
│   │   └── jolpica/
│   ├── interim/                 # limpiado por sesión
│   ├── processed/               # tablas de features
│   └── snapshots/               # ESTADOS CONGELADOS point-in-time
│       └── 2026/
│           ├── round_01_pre/    # estado conocido ANTES de la ronda 1
│           ├── round_02_pre/
│           └── ...
├── src/
│   └── gridcast/
│       ├── __init__.py
│       ├── ingest/
│       │   ├── fastf1_client.py
│       │   ├── jolpica_client.py
│       │   ├── fia_documents.py      # scrape uso PU/penalizaciones (best-effort)
│       │   └── schedule.py           # calendario + formato de cada finde
│       ├── data/
│       │   ├── sessions.py           # tipos de sesión + formato normal/sprint
│       │   ├── driver_session_map.py # quién condujo realmente (flag rookie FP1)
│       │   ├── tyres.py
│       │   └── weather.py
│       ├── features/
│       │   ├── pace.py               # ritmo tanda larga / quali
│       │   ├── teammate_gaps.py      # backbone separación coche/piloto
│       │   ├── reliability.py
│       │   └── penalties.py          # presupuesto componentes / riesgo sanción
│       ├── models/
│       │   ├── strength.py           # jerárquico piloto/equipo/motorista
│       │   ├── race_model.py         # distribución de orden de llegada
│       │   ├── plackett_luce.py      # sampler de rankings
│       │   └── priors.py             # priors de reseteo 2026
│       ├── simulation/
│       │   ├── race_sim.py
│       │   ├── season_sim.py
│       │   ├── chaos.py              # safety car/lluvia/DNF/penalización
│       │   └── scoring.py            # aplica puntos
│       ├── backtest/
│       │   ├── snapshot.py           # construye estados congelados
│       │   ├── replay.py             # walk-forward sobre rondas pasadas
│       │   └── leakage_guard.py      # assert: nada de futuro
│       ├── evaluate/
│       │   ├── calibration.py        # brier, log-loss, reliability curves
│       │   └── metrics.py
│       └── cli.py
├── pipelines/
│   ├── 01_ingest.py
│   ├── 02_build_features.py
│   ├── 03_freeze_history.py          # backfill: snapshot + predice rondas pasadas
│   ├── 04_fit_models.py
│   ├── 05_simulate_championship.py
│   └── 06_evaluate.py
├── outputs/
│   ├── predictions/                  # predicciones congeladas por carrera
│   ├── championship/                 # probabilidades de título en el tiempo
│   └── reports/
├── notebooks/
│   └── exploration/
├── tests/
│   ├── test_leakage_guard.py         # crítico
│   ├── test_scoring.py
│   └── test_session_format.py
└── scripts/
```

---

## 7. Modelo de datos (tablas clave)

- **sessions** — `season, round, session_type, weekend_format, date_utc, circuit_id`
- **driver_session_map** — `season, round, session_type, car_no, driver_id, es_titular` *(separa rookie FP1)*
- **laps** — `..., driver_id, lap_time, sector_1/2/3, compound, tyre_age, fuel_corrected_flag, stint_id`
- **pace_features** — por (round, driver): `quali_pace, long_run_pace, gap_to_teammate_quali, gap_to_teammate_race`
- **results** — `season, round, driver_id, grid, finish_pos, status (DNF/...), points`
- **component_usage** — `season, driver_id, component_type, units_used, units_allowed`
- **strength_estimates** — `as_of_round, driver_id|team_id|engine_id, mean, sd` *(salida del modelo jerárquico)*
- **snapshots/** — para cada `round_NN_pre`: copia inmutable de las features y los estimates disponibles en el corte.

---

## 8. Comandos / flujo de trabajo

> Comandos objetivo (aún no hay `pyproject.toml`; créalo antes de `uv sync`).

```bash
# setup
uv sync                       # o: pip install -e .

# pipeline completo
python pipelines/01_ingest.py --season 2026
python pipelines/02_build_features.py --season 2026
python pipelines/03_freeze_history.py --season 2026     # backfill point-in-time
python pipelines/04_fit_models.py --as-of-round <N>
python pipelines/05_simulate_championship.py --as-of-round <N> --n-sims 30000
python pipelines/06_evaluate.py --season 2026

# tests (el de leakage es obligatorio antes de cualquier commit que toque backtest)
pytest tests/ -q
pytest tests/test_leakage_guard.py -q   # solo leakage
pytest tests/ -q -k "test_scoring"      # un test concreto
```

---

## 9. Convenciones y reglas para Claude Code

- **No introducir leakage jamás.** Cualquier feature de la ronda N solo puede usar datos con fecha < corte(N). Si dudas, pasa por `leakage_guard`.
- **No reescribir snapshots congelados.** Cambios de modelo → nueva versión + re-generación etiquetada de todo el histórico.
- **Predicciones = distribuciones**, no valores puntuales. Cualquier salida lleva su incertidumbre.
- **Calibración primero.** Antes de "mejorar el acierto", comprobar Brier/log-loss y la curva de fiabilidad.
- **Reglas de dominio de la sección 3 son invariantes**: tests que las protejan (puntos, formato de finde, flag de titular en FP1).
- Config en YAML validado con pydantic; **nada de constantes mágicas** dispersas por el código.
- Datos crudos en `data/raw/` son **inmutables**; todo lo derivado se regenera.
- Seeds fijos para reproducibilidad de las simulaciones (configurable).

---

## 10. Roadmap sugerido

1. **Ingesta + caché** (FastF1 + Jolpica) y tabla `driver_session_map` con el flag de titular.
2. **Features de pace** (tanda larga + quali) y **teammate gaps**.
3. **Backtest harness** con `leakage_guard` y snapshots → congelar histórico 2026.
4. **Modelo de fuerza** jerárquico (baseline MixedLM → bayesiano) con efecto motorista compartido.
5. **race_sim + Plackett-Luce + chaos** (incluida penalización de parrilla).
6. **season_sim** → probabilidades de título + "embudo de incertidumbre" en el tiempo.
7. **Evaluación de calibración** y reporte.
8. (Opcional) dashboard.

---

## 11. Referencia rápida de hechos 2026

- Temporada arrancó en Australia (8 marzo 2026), 24 rondas, 11 equipos.
- Puntos: 25-18-15-12-10-8-6-4-2-1; **sin** punto de vuelta rápida; sprint 8→1.
- PU sin MGU-H; reparto ~50/50; combustible 100% sostenible; aero activa (X/Z mode) en vez de DRS.
- Asignación PU con +1 bonus (ver tabla §3.4); penalización 10/5/fondo>15.
- FP1 con rodaje obligatorio de rookies (hasta 4/temporada) → flag de titular imprescindible.
