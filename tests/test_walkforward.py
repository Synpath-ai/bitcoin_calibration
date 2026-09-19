import pandas as pd

from src import walkforward as wf


def _synthetic_dataset(n_markets=60):
    """n_markets sequential daily markets, each with a single 1h-horizon observation,
    outcome alternating so both classes are present (needed to fit logistic models)."""
    rows, mkt_rows = [], []
    base = pd.Timestamp("2025-01-01T12:00:00Z")
    for i in range(n_markets):
        end = base + pd.Timedelta(days=i)
        start = end - pd.Timedelta(hours=30)
        slug = f"mkt-{i:03d}"
        outcome = i % 2
        rows.append({
            "slug": slug, "market_date": end.date().isoformat(),
            "observation_timestamp": (end - pd.Timedelta(hours=1)).isoformat(),
            "horizon_hours": 1, "yes_probability": 0.5 + (0.1 if outcome else -0.1),
            "outcome_up": outcome, "market_volume": 10000.0, "market_liquidity": 0.0,
            "btc_return_15m": 0.0, "btc_return_1h": 0.0, "btc_return_6h": 0.0,
            "btc_realized_vol_6h": 0.001, "btc_spot_volume_1h": 100.0,
        })
        mkt_rows.append({"slug": slug, "start_date": start, "end_date": end})
    return pd.DataFrame(rows), pd.DataFrame(mkt_rows)


def test_training_pool_never_includes_markets_resolving_after_test_market_opens():
    obs_df, mkts_df = _synthetic_dataset(40)
    pred_df, diag = wf.run_walkforward(obs_df, mkts_df, model_names=["raw_probability"],
                                        min_train_markets=5)
    end_lookup = dict(zip(mkts_df["slug"], mkts_df["end_date"]))
    start_lookup = dict(zip(mkts_df["slug"], mkts_df["start_date"]))

    # Reconstruct, for each predicted test market, whether n_train_markets is
    # consistent with "only markets that resolved before this market opened".
    for slug, n_train in pred_df[["slug", "n_train_markets"]].drop_duplicates().itertuples(index=False):
        test_open = start_lookup[slug]
        true_train_count = sum(1 for s, e in end_lookup.items() if e < test_open)
        assert n_train == true_train_count


def test_insufficient_history_markets_are_skipped_not_predicted():
    obs_df, mkts_df = _synthetic_dataset(10)
    pred_df, diag = wf.run_walkforward(obs_df, mkts_df, model_names=["raw_probability"],
                                        min_train_markets=100)  # impossible to satisfy
    assert pred_df.empty
    assert diag["n_test_markets_skipped"] == 10
    assert all(s["reason"] == "insufficient_history" for s in diag["skipped"])


def test_walkforward_predictions_are_chronologically_sorted_by_construction():
    obs_df, mkts_df = _synthetic_dataset(30)
    pred_df, _ = wf.run_walkforward(obs_df, mkts_df, model_names=["raw_probability"], min_train_markets=5)
    ts = pd.to_datetime(pred_df["observation_timestamp"])
    # training pool size must be non-decreasing as we move forward through time
    # (markets keep resolving, so the available-to-train-on pool only grows)
    n_train_by_time = pred_df.assign(ts=ts).sort_values("ts")["n_train_markets"].values
    assert all(n_train_by_time[i] <= n_train_by_time[i + 1] for i in range(len(n_train_by_time) - 1))
