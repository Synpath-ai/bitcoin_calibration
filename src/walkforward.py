"""Chronological walk-forward evaluation.

For every test market (sorted by its own start_date), the model is trained
ONLY on observations belonging to markets whose resolution ("end_date") is
strictly before the test market's start_date -- i.e. the training market must
have already resolved before the test market even opened. This is checked
directly per test market (not assumed monotonic), which is important here
because Polymarket keeps 2-3 daily BTC markets open concurrently.

A minimum training-pool size (in independent resolved markets) gates when
walk-forward predictions begin; markets before that point are recorded as
`insufficient_history` rather than silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as mdl

DEFAULT_MIN_TRAIN_MARKETS = 30


def run_walkforward(obs_df: pd.DataFrame, markets_df: pd.DataFrame,
                     model_names: list[str] | None = None,
                     min_train_markets: int = DEFAULT_MIN_TRAIN_MARKETS,
                     refit_every_n_markets: int = 1) -> tuple[pd.DataFrame, dict]:
    """
    obs_df: fixed-horizon observations (must include 'slug', 'market_date').
    markets_df: one row per market with 'slug', 'start_date', 'end_date' (both
        tz-aware timestamps), sorted or not (sorted internally).
    refit_every_n_markets: refit models every N test markets instead of every
        single one, purely to control runtime; predictions are still only ever
        made with a model trained strictly on markets resolved before the
        *current batch's* earliest test market opened (still leakage-free,
        just slightly more conservative/stale than refitting every market).
    """
    model_names = model_names or list(mdl.MODEL_REGISTRY.keys())

    mkts = markets_df.sort_values("start_date").reset_index(drop=True)
    # Compare as naive UTC int64 nanoseconds throughout -- avoids tz-aware vs
    # tz-naive comparison errors between pandas Timestamp (from iterrows) and
    # numpy datetime64 (from .values), since every source timestamp is already UTC.
    start_naive = mkts["start_date"].dt.tz_convert(None)
    end_naive = mkts["end_date"].dt.tz_convert(None)
    end_sorted_idx = np.argsort(end_naive.values)
    end_sorted_vals = end_naive.values[end_sorted_idx]

    predictions = []
    skipped = []

    pending_batch = []
    last_trained_models = None
    last_train_pool_slugs = None

    for i, row in mkts.iterrows():
        cutoff = start_naive.iloc[i]
        # all markets with end_date < cutoff (strict) via searchsorted on sorted end dates
        k = np.searchsorted(end_sorted_vals, cutoff, side="left")
        train_slugs = set(mkts.loc[end_sorted_idx[:k], "slug"])

        if len(train_slugs) < min_train_markets:
            skipped.append({"slug": row["slug"], "reason": "insufficient_history",
                             "n_train_markets_available": len(train_slugs)})
            continue

        need_refit = (
            last_trained_models is None
            or len(pending_batch) >= refit_every_n_markets
            or train_slugs != last_train_pool_slugs
        )
        if need_refit and (last_trained_models is None or train_slugs != last_train_pool_slugs):
            train_df = obs_df[obs_df["slug"].isin(train_slugs)]
            fitted = {}
            for name in model_names:
                fitted[name] = mdl.MODEL_REGISTRY[name]().fit(train_df)
            last_trained_models = fitted
            last_train_pool_slugs = train_slugs
            pending_batch = []
        pending_batch.append(row["slug"])

        test_obs = obs_df[obs_df["slug"] == row["slug"]]
        if test_obs.empty:
            continue
        for name, model in last_trained_models.items():
            preds = model.predict(test_obs)
            for (_, obs_row), pred in zip(test_obs.iterrows(), preds):
                predictions.append({
                    "slug": row["slug"],
                    "market_date": obs_row["market_date"],
                    "observation_timestamp": obs_row["observation_timestamp"],
                    "horizon_hours": obs_row["horizon_hours"],
                    "model": name,
                    "model_probability": float(pred),
                    "raw_probability": float(obs_row["yes_probability"]),
                    "outcome_up": int(obs_row["outcome_up"]),
                    "n_train_markets": len(train_slugs),
                })

    pred_df = pd.DataFrame(predictions)
    diagnostics = {
        "n_test_markets_used": mkts["slug"].nunique() - len(skipped),
        "n_test_markets_skipped": len(skipped),
        "skipped": skipped,
        "min_train_markets": min_train_markets,
    }
    return pred_df, diagnostics
