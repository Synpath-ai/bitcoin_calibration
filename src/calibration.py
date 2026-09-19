"""Calibration metrics for probabilistic forecasts against binary outcomes.

Every bootstrap routine here resamples at the MARKET/DATE level (not the raw
observation level) because observations from the same market/date at
different horizons are not independent draws -- they describe the same
underlying coin-flip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    pc = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    n = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def calibration_intercept_slope(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y ~ intercept + slope * logit(p) via unregularized logistic regression
    (the standard 'calibration-in-the-large' + 'calibration slope' diagnostic).
    Slope=1, intercept=0 is perfect calibration."""
    from sklearn.linear_model import LogisticRegression
    pc = np.clip(p, EPS, 1 - EPS)
    logit_p = np.log(pc / (1 - pc)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e12, solver="lbfgs")
    lr.fit(logit_p, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def fixed_buckets(n_bins: int = 10) -> np.ndarray:
    return np.linspace(0, 1, n_bins + 1)


def adaptive_quantile_buckets(p: np.ndarray, n_bins: int = 10) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p, qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def reliability_table(df: pd.DataFrame, p_col: str, y_col: str, market_col: str,
                       bin_edges: np.ndarray) -> pd.DataFrame:
    """One row per non-empty bucket: n_obs, n_independent_markets, avg implied prob,
    realized frequency, calibration gap, and a normal-approximation 95% CI on the
    realized frequency (using the number of INDEPENDENT MARKETS, not raw
    observations, as the effective sample size -- conservative, since horizons
    from the same market are correlated)."""
    d = df[[p_col, y_col, market_col]].copy()
    d.columns = ["p", "y", "market"]
    idx = np.clip(np.digitize(d["p"], bin_edges) - 1, 0, len(bin_edges) - 2)
    d["bucket"] = idx

    rows = []
    for b in sorted(d["bucket"].unique()):
        sub = d[d["bucket"] == b]
        n_obs = len(sub)
        n_mkt = sub["market"].nunique()
        avg_p = sub["p"].mean()
        # realized frequency at market granularity: average outcome per distinct market
        # (a market appearing at multiple horizons in the same bucket is not double counted)
        mkt_level = sub.groupby("market")["y"].mean()
        realized_freq = mkt_level.mean()
        gap = realized_freq - avg_p
        se = np.sqrt(max(realized_freq * (1 - realized_freq), EPS) / max(n_mkt, 1))
        ci_lo, ci_hi = realized_freq - 1.96 * se, realized_freq + 1.96 * se
        rows.append({
            "bucket": b, "bucket_lo": bin_edges[b], "bucket_hi": bin_edges[b + 1],
            "n_obs": n_obs, "n_independent_markets": n_mkt,
            "avg_implied_prob": avg_p, "realized_up_frequency": realized_freq,
            "calibration_gap": gap, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "statistically_significant": bool((ci_lo > 0) or (ci_hi < 0)) if n_mkt >= 20 else False,
        })
    return pd.DataFrame(rows)


def group_gap_table(df: pd.DataFrame, group_col: str, p_col: str, y_col: str, market_col: str) -> pd.DataFrame:
    """Like reliability_table but grouping by an arbitrary pre-binned categorical
    column (e.g. a volatility regime or volume quartile) instead of the implied
    probability itself. Used for the mispricing-by-regime breakdowns."""
    d = df[[group_col, p_col, y_col, market_col]].dropna(subset=[group_col]).copy()
    d.columns = ["group", "p", "y", "market"]
    rows = []
    for g, sub in d.groupby("group", observed=True):
        n_obs = len(sub)
        n_mkt = sub["market"].nunique()
        avg_p = sub["p"].mean()
        mkt_level = sub.groupby("market")["y"].mean()
        realized_freq = mkt_level.mean()
        gap = realized_freq - avg_p
        se = np.sqrt(max(realized_freq * (1 - realized_freq), EPS) / max(n_mkt, 1))
        ci_lo, ci_hi = realized_freq - 1.96 * se, realized_freq + 1.96 * se
        rows.append({
            "group": g, "n_obs": n_obs, "n_independent_markets": n_mkt,
            "avg_implied_prob": avg_p, "realized_up_frequency": realized_freq,
            "calibration_gap": gap, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "statistically_significant": bool((ci_lo > 0) or (ci_hi < 0)) if n_mkt >= 20 else False,
        })
    return pd.DataFrame(rows)


def bootstrap_metric_by_market(df: pd.DataFrame, p_col: str, y_col: str, market_col: str,
                                metric_fn, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Block-bootstrap: resample distinct markets WITH replacement, keep all of that
    market's observations each draw, recompute the metric. Returns (point_estimate,
    ci_lo, ci_hi) at 95%."""
    rng = np.random.default_rng(seed)
    markets = df[market_col].unique()
    point = metric_fn(df[p_col].values, df[y_col].values)

    boot_vals = []
    grouped = {m: np.asarray(idx) for m, idx in df.groupby(market_col).indices.items()}
    p_arr, y_arr = df[p_col].values, df[y_col].values
    for _ in range(n_boot):
        sampled = rng.choice(markets, size=len(markets), replace=True)
        rows = np.concatenate([grouped[m] for m in sampled])
        boot_vals.append(metric_fn(p_arr[rows], y_arr[rows]))
    boot_vals = np.array(boot_vals)
    return float(point), float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))
