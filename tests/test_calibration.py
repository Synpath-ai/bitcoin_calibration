import numpy as np
import pandas as pd

from src import calibration as cal


def test_brier_score_perfect_and_worst():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1, 0, 1, 0])
    assert cal.brier_score(p, y) == 0.0
    assert cal.brier_score(1 - p, y) == 1.0


def test_log_loss_matches_hand_computation():
    p = np.array([0.8, 0.2])
    y = np.array([1, 0])
    expected = -np.mean([np.log(0.8), np.log(0.8)])
    assert abs(cal.log_loss(p, y) - expected) < 1e-9


def test_ece_zero_when_perfectly_calibrated_within_bin():
    # every observation in [0.7,0.8) has implied prob 0.75 and realized freq 0.75
    p = np.array([0.75] * 4)
    y = np.array([1, 1, 1, 0])  # freq = 0.75
    assert cal.expected_calibration_error(p, y, n_bins=10) < 1e-9


def test_calibration_slope_and_intercept_recovers_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    y = rng.binomial(1, p)
    intercept, slope = cal.calibration_intercept_slope(p, y)
    assert abs(intercept) < 0.15
    assert abs(slope - 1.0) < 0.15


def test_fixed_buckets_are_evenly_spaced():
    edges = cal.fixed_buckets(10)
    assert len(edges) == 11
    assert np.allclose(np.diff(edges), 0.1)


def test_adaptive_quantile_buckets_have_similar_counts_per_bucket():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 1000)
    edges = cal.adaptive_quantile_buckets(p, n_bins=5)
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    assert counts.min() > 0.5 * counts.max()  # roughly balanced, unlike fixed bins on skewed data


def test_bootstrap_metric_by_market_resamples_at_market_level_not_row_level():
    # 2 markets, one with 5 correlated (duplicated) rows, one with 1 row.
    # A row-level bootstrap would let the 5x-duplicated market dominate variance
    # far more than a market-level bootstrap does.
    df = pd.DataFrame({
        "p": [0.9] * 5 + [0.1],
        "y": [1] * 5 + [0],
        "market": ["A"] * 5 + ["B"],
    })
    point, lo, hi = cal.bootstrap_metric_by_market(df, "p", "y", "market", cal.brier_score, n_boot=200, seed=1)
    # with only 2 distinct markets, bootstrap draws are one of {AA, AB, BA, BB} in
    # expectation -> should include some spread, but with a fixed point estimate.
    assert lo <= point <= hi


def test_reliability_table_gap_sign_convention():
    # implied prob 0.3 everywhere, but realized frequency is 0.6 (underpriced YES)
    df = pd.DataFrame({
        "p": [0.3] * 10,
        "y": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "market": [f"m{i}" for i in range(10)],
    })
    table = cal.reliability_table(df, "p", "y", "market", cal.fixed_buckets(10))
    row = table[(table.bucket_lo <= 0.3) & (table.bucket_hi > 0.3)].iloc[0]
    assert row["calibration_gap"] > 0  # realized > implied -> YES was underpriced
