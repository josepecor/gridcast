# GridCast

F1 2026 championship forecast. Two outputs:

1. **Race predictions** — probability distribution over finishing order for each driver, per race.
2. **Championship simulation** — Monte Carlo over the remaining season to produce title probabilities for drivers and constructors, expected points, and scenarios.

The race model feeds the season simulation. A single "pace + uncertainty + DNF risk" model serves both.

### Design principle

What matters is **calibration**, not point accuracy. A 30% probability should happen ~3 out of 10 times. Everything is evaluated with Brier score / log-loss and reliability curves.

---

## 2026 context

2026 is the largest regulation reset in F1 history: new chassis, new hybrid PU (no MGU-H, ~50/50 combustion/electric), active aero replacing DRS, 11 teams (Cadillac debut). Historical data from 2022–2025 is nearly useless for car pace — the model starts with wide priors on team strength and learns fast from pre-season testing and early races.

The season is already underway. All historical predictions are generated via **walk-forward backtest with point-in-time correctness** — each round's prediction uses only data available before that round's first session. No leakage.

---

## Setup

```bash
# Install (requires Python 3.11+)
uv sync

# Or with pip
pip install -e .
```

---

## Pipeline

Run steps in order. Each step is idempotent.

```bash
# 1. Ingest: FastF1 cache + driver_session_map + Jolpica standings
python pipelines/01_ingest.py --season 2026

# 2. Features: pace per round (long_run_pace_s, quali_pace_s, intra-team gaps)
python pipelines/02_build_features.py --season 2026

# 3. Snapshots (point-in-time) + predictions (pre-fp1, post-q, post-sprint)
python pipelines/03_freeze_history.py --season 2026 --force-predict
python pipelines/03_post_q.py --season 2026 --force-predict
python pipelines/03_post_sprint.py --season 2026 --force-predict

# 4. Fit strength model (driver / team / engine hierarchy)
python pipelines/04_fit_models.py --as-of-round 5

# 5. Championship simulation (Monte Carlo, 30k runs)
python pipelines/05_simulate_championship.py --as-of-round 5 --n-sims 30000

# 6. Evaluate calibration
python pipelines/06_evaluate.py --season 2026

# Export to JSON for the dashboard
python scripts/export_to_json.py --season 2026
```

The GitHub Actions workflow (`update.yml`) runs the full pipeline every Monday after race results land on Jolpica.

---

## Model

**Driver / team separation** (§4.1 CLAUDE.md):

```
performance = team_effect + driver_effect + noise
team_effect = chassis + engine + operational
```

The only unconfounded comparison is two teammates in the same car. Chaining teammate gaps across seasons places every driver on a common scale. Qualifying pace is the primary signal (less strategy noise than race pace).

**Car strength** is a time-varying state — starts wide in 2026 and updates race by race.

**Race distribution**: Plackett-Luce — assign strengths, add Gumbel noise, sort. Gives a proper probability distribution over finishing order.

**Season simulation**: for each remaining race, sample N finishing orders → assign points → accumulate over the season → repeat 10k–50k times → title probabilities.

---

## Point-in-time correctness

`LeakageGuard` asserts that no row with date ≥ cutoff(round N) enters the feature set for round N. Violated assertions fail hard. Frozen snapshots are never overwritten — model changes require bumping `backtest.snapshot_version` in `config/settings.yaml` and regenerating the full historical backtest.

---

## Tests

```bash
pytest tests/ -q

# Critical: leakage guard must pass before any backtest-touching commit
pytest tests/test_leakage_guard.py -q
```

---

## Scoring (2026)

| Type | Points |
|------|--------|
| Race top 10 | 25 · 18 · 15 · 12 · 10 · 8 · 6 · 4 · 2 · 1 |
| Sprint top 8 | 8 · 7 · 6 · 5 · 4 · 3 · 2 · 1 |
| Fastest lap | — eliminated since 2025 |

---

## Stack

- **Python 3.11+**, managed with `uv`
- **FastF1** — primary source for all session data (lap times, sectors, results)
- **Jolpica-F1 API** — calendar and standings only
- **pandas / pyarrow** — tabular data, stored as Parquet
- **numpy** — vectorized simulation
- **statsmodels** — baseline hierarchical model (MixedLM); bayesian upgrade via `pymc` / `numpyro` optional
- **pydantic** — config validation
- **pytest** — tests
