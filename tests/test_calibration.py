"""
Tests de evaluate/calibration.py

Sin llamadas a red. RaceResult y RoundPrediction sintéticos.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridcast.evaluate.calibration import (
    brier_score_full,
    brier_score_topk,
    brier_score_win,
    collect_win_pairs,
    evaluate_predictions,
    expected_calibration_error,
    extract_topk_probs,
    extract_win_probs,
    log_loss_win,
    rank_correlation,
    reliability_diagram,
)


# ---------------------------------------------------------------------------
# Factories de fixtures
# ---------------------------------------------------------------------------

def _race_result(driver_id: str, position: int | None) -> Any:
    class _RR:
        pass
    r = _RR()
    r.driver_id = driver_id
    r.position  = position
    r.position_text = str(position) if position else "R"
    r.status    = "Finished" if position else "Retired"
    r.points    = 0.0
    return r


def _results(order: list[str]) -> list:
    """Crea una lista de RaceResult donde order[0] ganó, etc."""
    return [_race_result(d, i + 1) for i, d in enumerate(order)]


def _finish_probs_canonical(probs: dict[str, list[float]]) -> pd.DataFrame:
    """
    Construye finish_probs en formato canónico.
    probs = {driver_id: [p_pos1, p_pos2, ...]}
    """
    rows = []
    for driver_id, ps in probs.items():
        for pos, p in enumerate(ps, start=1):
            rows.append({"driver_id": driver_id, "finish_pos": pos, "probability": p})
    return pd.DataFrame(rows)


def _finish_probs_win(win_probs: dict[str, float]) -> pd.DataFrame:
    rows = [{"driver_id": d, "win_prob": p} for d, p in win_probs.items()]
    return pd.DataFrame(rows)


def _finish_probs_score(scores: dict[str, float]) -> pd.DataFrame:
    rows = [{"driver_id": d, "score": s} for d, s in scores.items()]
    return pd.DataFrame(rows)


class _FakePrediction:
    def __init__(self, round_: int, finish_probs: pd.DataFrame):
        self.round_       = round_
        self.finish_probs = finish_probs
        self.cutoff_date  = date(2026, 3, round_)


# ---------------------------------------------------------------------------
# extract_win_probs
# ---------------------------------------------------------------------------

class TestExtractWinProbs:
    def test_canonical_format(self):
        fp = _finish_probs_canonical({
            "VER": [0.4, 0.3, 0.2, 0.1],
            "NOR": [0.3, 0.4, 0.2, 0.1],
        })
        probs = extract_win_probs(fp)
        assert probs["VER"] == pytest.approx(0.4)
        assert probs["NOR"] == pytest.approx(0.3)

    def test_win_prob_format(self):
        fp = _finish_probs_win({"VER": 0.35, "NOR": 0.25, "HAM": 0.20})
        probs = extract_win_probs(fp)
        assert probs["VER"] == pytest.approx(0.35)
        assert set(probs.index) == {"VER", "NOR", "HAM"}

    def test_score_format_softmax(self):
        fp = _finish_probs_score({"VER": 2.0, "NOR": 1.0, "HAM": 0.0})
        probs = extract_win_probs(fp)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert probs["VER"] > probs["NOR"] > probs["HAM"]

    def test_empty_returns_empty(self):
        fp = pd.DataFrame(columns=["driver_id", "finish_pos", "probability"])
        result = extract_win_probs(fp)
        assert result.empty

    def test_no_recognizable_cols_raises(self):
        fp = pd.DataFrame({"driver_id": ["VER"], "unknown_col": [0.5]})
        with pytest.raises(ValueError, match="P\\(win\\)"):
            extract_win_probs(fp)

    def test_canonical_sums_to_one(self):
        """La suma de P(win) sobre todos los pilotos debe ser ≈ 1."""
        fp = _finish_probs_canonical({
            "VER": [0.4, 0.35, 0.25],
            "NOR": [0.35, 0.40, 0.25],
            "HAM": [0.25, 0.25, 0.50],
        })
        probs = extract_win_probs(fp)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# extract_topk_probs
# ---------------------------------------------------------------------------

class TestExtractTopkProbs:
    def test_top1_equals_win_prob(self):
        fp = _finish_probs_canonical({
            "VER": [0.4, 0.35, 0.25],
            "NOR": [0.35, 0.40, 0.25],
            "HAM": [0.25, 0.25, 0.50],
        })
        top1 = extract_topk_probs(fp, k=1).sort_index()
        win  = extract_win_probs(fp).rename("top1_prob").sort_index()
        pd.testing.assert_series_equal(top1, win)

    def test_top2_sum_of_positions(self):
        fp = _finish_probs_canonical({
            "VER": [0.4, 0.3, 0.3],
            "NOR": [0.6, 0.2, 0.2],
        })
        top2 = extract_topk_probs(fp, k=2)
        assert top2["VER"] == pytest.approx(0.4 + 0.3, abs=1e-9)
        assert top2["NOR"] == pytest.approx(0.6 + 0.2, abs=1e-9)

    def test_empty_returns_empty(self):
        fp = pd.DataFrame(columns=["driver_id", "finish_pos", "probability"])
        assert extract_topk_probs(fp, k=3).empty


# ---------------------------------------------------------------------------
# brier_score_win
# ---------------------------------------------------------------------------

class TestBrierScoreWin:
    def test_perfect_prediction(self):
        """Proba 1.0 para el ganador y 0.0 para los demás → BS = 0."""
        fp = _finish_probs_win({"VER": 1.0, "NOR": 0.0, "HAM": 0.0})
        results = _results(["VER", "NOR", "HAM"])
        assert brier_score_win(fp, results) == pytest.approx(0.0, abs=1e-9)

    def test_worst_prediction(self):
        """Proba 0.0 para el ganador y demás distribuida → BS > 0."""
        fp = _finish_probs_win({"VER": 0.0, "NOR": 0.5, "HAM": 0.5})
        results = _results(["VER", "NOR", "HAM"])
        bs = brier_score_win(fp, results)
        assert bs > 0

    def test_uniform_prediction(self):
        """Predicción uniforme para N pilotos: BS = (N-1)/N²."""
        n = 4
        p = 1.0 / n
        fp = _finish_probs_win({f"d{i}": p for i in range(n)})
        results = _results([f"d{i}" for i in range(n)])
        expected = (n - 1) / n ** 2
        assert brier_score_win(fp, results) == pytest.approx(expected, abs=1e-9)

    def test_no_winner_in_results_returns_nan(self):
        fp = _finish_probs_win({"VER": 0.5, "NOR": 0.5})
        results = [_race_result("VER", None), _race_result("NOR", None)]
        assert math.isnan(brier_score_win(fp, results))

    def test_empty_finish_probs_returns_nan(self):
        results = _results(["VER", "NOR"])
        fp = pd.DataFrame(columns=["driver_id", "win_prob"])
        assert math.isnan(brier_score_win(fp, results))

    def test_lower_is_better(self):
        results = _results(["VER", "NOR", "HAM"])
        fp_good = _finish_probs_win({"VER": 0.7, "NOR": 0.2, "HAM": 0.1})
        fp_bad  = _finish_probs_win({"VER": 0.1, "NOR": 0.5, "HAM": 0.4})
        assert brier_score_win(fp_good, results) < brier_score_win(fp_bad, results)

    def test_canonical_format_works(self):
        fp = _finish_probs_canonical({"VER": [0.8, 0.2], "NOR": [0.2, 0.8]})
        results = _results(["VER", "NOR"])
        bs = brier_score_win(fp, results)
        assert 0.0 <= bs <= 1.0


# ---------------------------------------------------------------------------
# brier_score_topk
# ---------------------------------------------------------------------------

class TestBrierScoreTopk:
    def test_perfect_podium_prediction(self):
        fp = _finish_probs_canonical({
            "VER": [1.0, 0.0, 0.0],
            "NOR": [0.0, 1.0, 0.0],
            "HAM": [0.0, 0.0, 1.0],
            "RUS": [0.0, 0.0, 0.0],
        })
        results = _results(["VER", "NOR", "HAM", "RUS"])
        bs = brier_score_topk(fp, results, k=3)
        assert bs == pytest.approx(0.0, abs=1e-9)

    def test_top1_equals_bs_win(self):
        fp = _finish_probs_win({"VER": 0.5, "NOR": 0.3, "HAM": 0.2})
        results = _results(["VER", "NOR", "HAM"])
        assert brier_score_topk(fp, results, k=1) == pytest.approx(
            brier_score_win(fp, results), abs=1e-9
        )

    def test_empty_probs_returns_nan(self):
        fp = pd.DataFrame(columns=["driver_id", "win_prob"])
        assert math.isnan(brier_score_topk(fp, _results(["VER"]), k=3))


# ---------------------------------------------------------------------------
# brier_score_full
# ---------------------------------------------------------------------------

class TestBrierScoreFull:
    def test_perfect_prediction(self):
        fp = _finish_probs_canonical({
            "VER": [1.0, 0.0],
            "NOR": [0.0, 1.0],
        })
        results = _results(["VER", "NOR"])
        assert brier_score_full(fp, results) == pytest.approx(0.0, abs=1e-9)

    def test_non_canonical_returns_nan(self):
        fp = _finish_probs_win({"VER": 0.6, "NOR": 0.4})
        assert math.isnan(brier_score_full(fp, _results(["VER", "NOR"])))

    def test_worse_than_perfect(self):
        fp_perfect = _finish_probs_canonical({
            "VER": [1.0, 0.0],
            "NOR": [0.0, 1.0],
        })
        fp_uniform = _finish_probs_canonical({
            "VER": [0.5, 0.5],
            "NOR": [0.5, 0.5],
        })
        results = _results(["VER", "NOR"])
        assert brier_score_full(fp_perfect, results) < brier_score_full(fp_uniform, results)


# ---------------------------------------------------------------------------
# log_loss_win
# ---------------------------------------------------------------------------

class TestLogLossWin:
    def test_perfect_prediction_near_zero(self):
        fp = _finish_probs_win({"VER": 1.0 - 1e-7, "NOR": 1e-7})
        results = _results(["VER", "NOR"])
        ll = log_loss_win(fp, results)
        assert ll < 0.05  # casi 0 dado el clip

    def test_overconfident_wrong_has_high_loss(self):
        """Alta confianza en el piloto equivocado → log-loss alto."""
        fp_conf  = _finish_probs_win({"VER": 0.99, "NOR": 0.01})
        fp_uninf = _finish_probs_win({"VER": 0.5,  "NOR": 0.5 })
        results  = _results(["NOR", "VER"])  # NOR ganó
        assert log_loss_win(fp_conf, results) > log_loss_win(fp_uninf, results)

    def test_clipping_avoids_inf(self):
        fp = _finish_probs_win({"VER": 0.0, "NOR": 1.0})
        results = _results(["VER", "NOR"])
        ll = log_loss_win(fp, results, clip_min=1e-6)
        assert not math.isinf(ll)
        assert not math.isnan(ll)

    def test_empty_probs_returns_nan(self):
        fp = pd.DataFrame(columns=["driver_id", "win_prob"])
        assert math.isnan(log_loss_win(fp, _results(["VER"])))

    def test_no_winner_returns_nan(self):
        fp = _finish_probs_win({"VER": 0.5, "NOR": 0.5})
        results = [_race_result("VER", None), _race_result("NOR", None)]
        assert math.isnan(log_loss_win(fp, results))


# ---------------------------------------------------------------------------
# reliability_diagram
# ---------------------------------------------------------------------------

class TestReliabilityDiagram:
    def test_empty_input_returns_empty(self):
        df = reliability_diagram([])
        assert df.empty

    def test_output_columns(self):
        pairs = [(0.3, 1), (0.7, 0), (0.5, 1)]
        df = reliability_diagram(pairs, n_bins=5)
        assert set(df.columns) >= {"bin_lower", "bin_upper", "bin_center", "mean_pred", "mean_obs", "count", "gap"}

    def test_n_bins_rows(self):
        pairs = [(i / 100, i % 2) for i in range(100)]
        df = reliability_diagram(pairs, n_bins=10)
        assert len(df) == 10

    def test_bin_covers_0_to_1(self):
        pairs = [(0.0, 0), (1.0, 1)]
        df = reliability_diagram(pairs, n_bins=5)
        assert df["bin_lower"].min() == pytest.approx(0.0)
        assert df["bin_upper"].max() == pytest.approx(1.0)

    def test_perfectly_calibrated_model(self):
        """Un modelo perfectamente calibrado: mean_pred ≈ mean_obs en cada bin."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, 5000)
        outcomes = rng.binomial(1, probs)
        pairs = list(zip(probs.tolist(), outcomes.tolist()))
        df = reliability_diagram(pairs, n_bins=10).dropna()
        # En expectativa, |mean_pred - mean_obs| < 0.05 por bin con 5000 muestras
        assert (df["gap"] < 0.07).all()

    def test_overconfident_model_gaps_large(self):
        """Modelo que siempre predice 0.9 pero solo acierta el 30%."""
        pairs = [(0.9, 1)] * 30 + [(0.9, 0)] * 70
        df = reliability_diagram(pairs, n_bins=5)
        populated = df.dropna()
        assert (populated["gap"] > 0.4).any()

    def test_gap_equals_abs_diff(self):
        pairs = [(0.4, 1), (0.4, 0), (0.4, 1)]
        df = reliability_diagram(pairs, n_bins=5).dropna()
        for _, row in df.iterrows():
            assert row["gap"] == pytest.approx(abs(row["mean_pred"] - row["mean_obs"]), abs=1e-9)

    def test_empty_bins_have_nan(self):
        """Bins sin muestras tienen mean_pred, mean_obs = NaN."""
        pairs = [(0.05, 1), (0.05, 0)]  # solo el primer bin tiene datos
        df = reliability_diagram(pairs, n_bins=10)
        empty_bins = df[df["count"] == 0]
        assert empty_bins["mean_pred"].isna().all()
        assert empty_bins["mean_obs"].isna().all()


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------

class TestExpectedCalibrationError:
    def test_perfectly_calibrated_ece_near_zero(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 10000)
        outcomes = rng.binomial(1, probs)
        diag = reliability_diagram(list(zip(probs.tolist(), outcomes.tolist())), n_bins=10)
        ece = expected_calibration_error(diag)
        assert ece < 0.03

    def test_overconfident_ece_high(self):
        pairs = [(0.95, 0)] * 100  # siempre predice 0.95, nunca acierta
        diag = reliability_diagram(pairs, n_bins=10)
        ece = expected_calibration_error(diag)
        assert ece > 0.5

    def test_empty_diagram_returns_nan(self):
        diag = reliability_diagram([], n_bins=10)
        assert math.isnan(expected_calibration_error(diag))

    def test_ece_in_0_1(self):
        pairs = [(0.3, 1), (0.7, 0), (0.5, 1), (0.2, 0)] * 20
        diag = reliability_diagram(pairs, n_bins=5)
        ece = expected_calibration_error(diag)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# rank_correlation
# ---------------------------------------------------------------------------

class TestRankCorrelation:
    def test_perfect_order(self):
        """Predicción con VER >> NOR >> HAM y ese mismo orden en resultado → rho ≈ 1."""
        fp = _finish_probs_win({"VER": 0.6, "NOR": 0.3, "HAM": 0.1})
        results = _results(["VER", "NOR", "HAM"])
        rho = rank_correlation(fp, results)
        assert rho == pytest.approx(1.0, abs=1e-9)

    def test_reverse_order(self):
        """Predicción inversa al resultado → rho = -1."""
        fp = _finish_probs_win({"VER": 0.1, "NOR": 0.3, "HAM": 0.6})
        results = _results(["VER", "NOR", "HAM"])
        rho = rank_correlation(fp, results)
        assert rho == pytest.approx(-1.0, abs=1e-9)

    def test_single_driver_returns_nan(self):
        fp = _finish_probs_win({"VER": 1.0})
        results = _results(["VER"])
        assert math.isnan(rank_correlation(fp, results))

    def test_no_common_drivers_returns_nan(self):
        fp = _finish_probs_win({"VER": 0.6, "NOR": 0.4})
        results = _results(["HAM", "RUS"])
        assert math.isnan(rank_correlation(fp, results))

    def test_between_minus1_and_1(self):
        fp = _finish_probs_win({"VER": 0.4, "NOR": 0.35, "HAM": 0.15, "RUS": 0.10})
        results = _results(["NOR", "HAM", "VER", "RUS"])
        rho = rank_correlation(fp, results)
        assert -1.0 <= rho <= 1.0


# ---------------------------------------------------------------------------
# collect_win_pairs
# ---------------------------------------------------------------------------

class TestCollectWinPairs:
    def test_correct_number_of_pairs(self):
        """3 rondas × 3 pilotos = 9 pares."""
        preds = [
            _FakePrediction(r, _finish_probs_win({"VER": 0.5, "NOR": 0.3, "HAM": 0.2}))
            for r in range(1, 4)
        ]
        results_map = {r: _results(["VER", "NOR", "HAM"]) for r in range(1, 4)}
        pairs = collect_win_pairs(preds, results_map)
        assert len(pairs) == 9

    def test_winner_has_outcome_1(self):
        fp = _finish_probs_win({"VER": 0.7, "NOR": 0.3})
        pred = _FakePrediction(1, fp)
        results_map = {1: _results(["VER", "NOR"])}
        pairs = collect_win_pairs([pred], results_map)
        ver_pairs = [(p, y) for p, y in pairs if abs(p - 0.7) < 1e-9]
        assert any(y == 1 for _, y in ver_pairs)

    def test_missing_round_skipped(self):
        fp = _finish_probs_win({"VER": 0.6, "NOR": 0.4})
        pred = _FakePrediction(1, fp)
        results_map = {99: _results(["VER", "NOR"])}  # ronda 1 no tiene resultados
        pairs = collect_win_pairs([pred], results_map)
        assert len(pairs) == 0

    def test_probs_are_floats_and_outcomes_binary(self):
        fp = _finish_probs_win({"VER": 0.6, "NOR": 0.4})
        pred = _FakePrediction(1, fp)
        pairs = collect_win_pairs([pred], {1: _results(["VER", "NOR"])})
        for p, y in pairs:
            assert isinstance(p, float)
            assert y in (0, 1)


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------

class TestEvaluatePredictions:
    def _make_preds_and_results(self, n_rounds: int = 3):
        preds = []
        results_map = {}
        for r in range(1, n_rounds + 1):
            fp = _finish_probs_win({"VER": 0.5, "NOR": 0.3, "HAM": 0.2})
            preds.append(_FakePrediction(r, fp))
            results_map[r] = _results(["VER", "NOR", "HAM"])
        return preds, results_map

    def test_returns_required_keys(self):
        preds, results = self._make_preds_and_results(3)
        out = evaluate_predictions(preds, results)
        assert set(out.keys()) >= {
            "brier_win", "brier_top3", "log_loss_win",
            "rank_correlation", "ece_win",
            "per_round", "reliability_win", "n_rounds",
        }

    def test_n_rounds_correct(self):
        preds, results = self._make_preds_and_results(4)
        out = evaluate_predictions(preds, results)
        assert out["n_rounds"] == 4

    def test_per_round_df_shape(self):
        preds, results = self._make_preds_and_results(3)
        out = evaluate_predictions(preds, results)
        assert len(out["per_round"]) == 3

    def test_per_round_has_columns(self):
        preds, results = self._make_preds_and_results(2)
        out = evaluate_predictions(preds, results)
        assert {"round_", "bs_win", "bs_top3", "log_loss_win", "rank_corr"}.issubset(
            out["per_round"].columns
        )

    def test_brier_win_in_0_1(self):
        preds, results = self._make_preds_and_results(5)
        out = evaluate_predictions(preds, results)
        assert 0.0 <= out["brier_win"] <= 1.0

    def test_empty_predictions_returns_nan_metrics(self):
        out = evaluate_predictions([], {})
        assert out["n_rounds"] == 0
        assert math.isnan(out["brier_win"])

    def test_missing_results_round_excluded(self):
        fp = _finish_probs_win({"VER": 0.6, "NOR": 0.4})
        preds = [_FakePrediction(1, fp), _FakePrediction(2, fp)]
        results_map = {1: _results(["VER", "NOR"])}  # ronda 2 sin resultados
        out = evaluate_predictions(preds, results_map)
        assert out["n_rounds"] == 1

    def test_good_model_lower_brier_than_bad(self):
        """Modelo que siempre acierta tiene BS menor que el que siempre falla."""
        results_map = {1: _results(["VER", "NOR", "HAM"])}

        fp_good = _finish_probs_win({"VER": 0.9, "NOR": 0.07, "HAM": 0.03})
        fp_bad  = _finish_probs_win({"VER": 0.03, "NOR": 0.07, "HAM": 0.9})

        out_good = evaluate_predictions([_FakePrediction(1, fp_good)], results_map)
        out_bad  = evaluate_predictions([_FakePrediction(1, fp_bad)],  results_map)

        assert out_good["brier_win"] < out_bad["brier_win"]
        assert out_good["log_loss_win"] < out_bad["log_loss_win"]


# ---------------------------------------------------------------------------
# Import para math.isnan en los tests
# ---------------------------------------------------------------------------

import math
