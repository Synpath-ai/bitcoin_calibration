"""Execution-aware backtest for the extended hourly BTC Up/Down sample.

Same strategies and single live-measured execution config as the daily
backtest (scripts/07_backtest.py -- see its docstring), but with risk settings
adapted to a market that only lasts ~1 hour: min_hours_to_resolution is cut
from the daily default (0.25h = 15min, which would silently exclude this
family's 5min/2min horizons entirely) down to ~1 minute.
"""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import backtest as bt
from src import backtest_metrics as bm
from src import execution as ex
from src import risk as rk

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

STRATEGIES = {
    "raw_probability_strategy": "raw_probability",
    "calibration_only_strategy": "logistic_calibration",
    "full_feature_strategy": "full_feature_model",
}
SCENARIO = "live"
HOURLY_HORIZON_ORDER = [45 / 60, 30 / 60, 15 / 60, 5 / 60, 2 / 60]  # hours, longest first


def main():
    pred_df = pd.read_parquet(PROCESSED / "hourly_walkforward_predictions.parquet")
    obs_df = pd.read_parquet(PROCESSED / "hourly_observations.parquet")

    feat_cols = ["slug", "horizon_hours", "observation_timestamp",
                 "btc_realized_vol_6h", "btc_return_1h", "news_data_available",
                 "market_volume", "market_liquidity"]
    pred_df = pred_df.merge(obs_df[feat_cols], on=["slug", "horizon_hours", "observation_timestamp"], how="left")

    live_measured = ex.measure_live_execution_costs("hourly")
    print(f"Live-measured execution costs (hourly, n={live_measured['n_snapshots']} check-ins): "
          f"half-spread ${live_measured['mean_half_spread']:.4f}, "
          f"touch depth ${live_measured['touch_depth_usd']:,.0f}, "
          f"per-cent depth ${live_measured['depth_per_cent_usd']:,.0f}\n")
    ex_cfg = ex.ExecutionConfig.from_live_measurements("hourly")
    risk_cfg = rk.RiskConfig(min_hours_to_resolution=1 / 60)  # allow the 2min horizon

    all_summaries, all_trades, all_equity, all_breakdowns = [], {}, {}, {}

    for strat_label, model_name in STRATEGIES.items():
        result = bt.run_backtest(pred_df, model_name, SCENARIO, ex_cfg, risk_cfg,
                                  horizon_order=HOURLY_HORIZON_ORDER)
        summary = bm.summarize(result)
        summary["strategy"] = strat_label
        all_summaries.append(summary)

        trades_df = bm.trades_to_df(result["trades"])
        key = f"{strat_label}__{SCENARIO}"
        all_trades[key] = trades_df
        all_equity[key] = pd.DataFrame(result["equity_curve"])

        if not trades_df.empty:
            trades_df = trades_df.copy()
            trades_df["horizon_minutes"] = (trades_df["horizon_hours"] * 60).round()
            all_breakdowns[f"{strat_label}__ttr"] = bm.breakdown(trades_df, "horizon_minutes")
            all_breakdowns[f"{strat_label}__prob_bucket"] = bm.breakdown(
                trades_df, "market_probability", bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
            all_breakdowns[f"{strat_label}__exec_quality"] = bm.breakdown(trades_df, "execution_quality")

        print(f"{strat_label:28s} n_signals={summary['n_signals']:5d} "
              f"n_fills={summary.get('n_fills',0):5d} net_pnl={summary.get('net_pnl',0):10.2f} "
              f"hit_rate={summary.get('hit_rate', float('nan')):.3f}")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(PROCESSED / "hourly_backtest_summary.csv", index=False)
    with open(PROCESSED / "hourly_backtest_trades.pkl", "wb") as f:
        pickle.dump(all_trades, f)
    with open(PROCESSED / "hourly_backtest_equity.pkl", "wb") as f:
        pickle.dump(all_equity, f)
    with open(PROCESSED / "hourly_backtest_breakdowns.pkl", "wb") as f:
        pickle.dump(all_breakdowns, f)

    print("\nSaved hourly_backtest_summary.csv, hourly_backtest_trades.pkl, "
          "hourly_backtest_equity.pkl, hourly_backtest_breakdowns.pkl")


if __name__ == "__main__":
    main()
