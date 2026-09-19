import pandas as pd
import pytest

from src import backtest as bt
from src import backtest_metrics as bm
from src import execution as ex
from src import risk as rk


def _pred_row(slug, horizon, obs_ts, model_prob, raw_prob, outcome, volume=200_000, liquidity=0):
    return {
        "slug": slug, "market_date": obs_ts[:10], "observation_timestamp": obs_ts,
        "horizon_hours": horizon, "model": "test_model", "model_probability": model_prob,
        "raw_probability": raw_prob, "outcome_up": outcome, "n_train_markets": 50,
        "market_volume": volume, "market_liquidity": liquidity,
    }


def _two_market_dataset():
    """Market A: model sees a large mispriced YES edge, YES resolves True (win).
    Market B: model sees a mispriced NO edge (raw prob high), NO resolves True
    i.e. outcome_up=0 (win for buy_no)."""
    rows = [
        _pred_row("mktA", 1, "2025-01-01T11:00:00Z", model_prob=0.85, raw_prob=0.55, outcome=1),
        _pred_row("mktB", 1, "2025-01-02T11:00:00Z", model_prob=0.15, raw_prob=0.50, outcome=0),
    ]
    return pd.DataFrame(rows)


def test_evaluate_signal_picks_positive_edge_side():
    pred_df = _two_market_dataset()
    ex_cfg = ex.ExecutionConfig(fee_bps=0)
    risk_cfg = rk.RiskConfig(min_model_edge=0.03)
    row = pred_df.iloc[0]
    sig = bt.evaluate_signal(row, ex_cfg, risk_cfg, "base", nominal_trade_size=300)
    assert sig is not None
    assert sig.side == "buy_yes"
    assert sig.net_expected_edge > risk_cfg.min_model_edge


def test_no_signal_when_edge_below_minimum():
    pred_df = _two_market_dataset()
    ex_cfg = ex.ExecutionConfig(fee_bps=0)
    risk_cfg = rk.RiskConfig(min_model_edge=0.90)  # impossible bar
    row = pred_df.iloc[0]
    sig = bt.evaluate_signal(row, ex_cfg, risk_cfg, "base", nominal_trade_size=300)
    assert sig is None


def test_backtest_pnl_accounting_yes_win():
    pred_df = pd.DataFrame([_pred_row("mktA", 1, "2025-01-01T11:00:00Z",
                                        model_prob=0.85, raw_prob=0.55, outcome=1)])
    ex_cfg = ex.ExecutionConfig(fee_bps=0, base_half_spread_bps=1, min_half_spread_bps=1,
                                 max_half_spread_bps=50, liquidity_coeff=0)
    risk_cfg = rk.RiskConfig(min_model_edge=0.01, fractional_kelly_cap=1.0, max_risk_per_market=1.0,
                              max_participation_rate=1.0, max_trade_size=1e9,
                              max_total_capital_deployed=1.0, min_market_volume=0)
    result = bt.run_backtest(pred_df, "test_model", "base", ex_cfg, risk_cfg)
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["side"] == "buy_yes"
    assert trade["won"] is True
    expected_gross_payout = trade["n_contracts"] * 1.0
    assert abs(trade["gross_payout"] - expected_gross_payout) < 1e-9
    assert abs(trade["pnl_net"] - (expected_gross_payout - trade["stake"])) < 1e-6


def test_backtest_pnl_accounting_no_win():
    pred_df = pd.DataFrame([_pred_row("mktB", 1, "2025-01-02T11:00:00Z",
                                        model_prob=0.15, raw_prob=0.50, outcome=0)])
    ex_cfg = ex.ExecutionConfig(fee_bps=0, base_half_spread_bps=1, min_half_spread_bps=1,
                                 max_half_spread_bps=50, liquidity_coeff=0)
    risk_cfg = rk.RiskConfig(min_model_edge=0.01, fractional_kelly_cap=1.0, max_risk_per_market=1.0,
                              max_participation_rate=1.0, max_trade_size=1e9,
                              max_total_capital_deployed=1.0, min_market_volume=0)
    result = bt.run_backtest(pred_df, "test_model", "base", ex_cfg, risk_cfg)
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["side"] == "buy_no"
    assert trade["won"] is True  # outcome_up=0 means the NO leg won


def test_backtest_losing_trade_loses_full_stake():
    # outcome flips against the winning side used above
    pred_df = pd.DataFrame([_pred_row("mktA", 1, "2025-01-01T11:00:00Z",
                                        model_prob=0.85, raw_prob=0.55, outcome=0)])
    ex_cfg = ex.ExecutionConfig(fee_bps=0, base_half_spread_bps=1, min_half_spread_bps=1,
                                 max_half_spread_bps=50, liquidity_coeff=0)
    risk_cfg = rk.RiskConfig(min_model_edge=0.01, fractional_kelly_cap=1.0, max_risk_per_market=1.0,
                              max_participation_rate=1.0, max_trade_size=1e9,
                              max_total_capital_deployed=1.0, min_market_volume=0)
    result = bt.run_backtest(pred_df, "test_model", "base", ex_cfg, risk_cfg)
    trade = result["trades"][0]
    assert trade["won"] is False
    assert trade["gross_payout"] == 0.0
    assert abs(trade["pnl_net"] + trade["stake"]) < 1e-6  # lost exactly the stake


def test_no_fill_when_market_volume_below_minimum():
    pred_df = pd.DataFrame([_pred_row("mktA", 1, "2025-01-01T11:00:00Z",
                                        model_prob=0.85, raw_prob=0.55, outcome=1, volume=10)])
    ex_cfg = ex.ExecutionConfig(fee_bps=0)
    risk_cfg = rk.RiskConfig(min_model_edge=0.01, min_market_volume=100_000)
    result = bt.run_backtest(pred_df, "test_model", "base", ex_cfg, risk_cfg)
    assert len(result["trades"]) == 0
    assert len(result["signals"]) == 0  # volume gate is checked before signal evaluation


def test_fees_reduce_net_pnl_relative_to_zero_fee():
    pred_df = pd.DataFrame([_pred_row("mktA", 1, "2025-01-01T11:00:00Z",
                                        model_prob=0.85, raw_prob=0.55, outcome=1)])
    risk_cfg = rk.RiskConfig(min_model_edge=0.01, fractional_kelly_cap=1.0, max_risk_per_market=1.0,
                              max_participation_rate=1.0, max_trade_size=1e9,
                              max_total_capital_deployed=1.0, min_market_volume=0)
    no_fee = ex.ExecutionConfig(fee_bps=0, base_half_spread_bps=1, min_half_spread_bps=1,
                                 max_half_spread_bps=50, liquidity_coeff=0)
    with_fee = ex.ExecutionConfig(fee_bps=200, base_half_spread_bps=1, min_half_spread_bps=1,
                                   max_half_spread_bps=50, liquidity_coeff=0)
    r1 = bt.run_backtest(pred_df, "test_model", "base", no_fee, risk_cfg)
    r2 = bt.run_backtest(pred_df, "test_model", "base", with_fee, risk_cfg)
    assert r2["trades"][0]["pnl_net"] < r1["trades"][0]["pnl_net"]
    assert r2["trades"][0]["fee_cost"] > 0


def test_fractional_horizon_hours_survive_backtest_without_truncation():
    """Regression test: sub-hourly market families (e.g. horizon_hours=0.0333 for a
    2-minute horizon) must not have their horizon truncated to 0/int anywhere in the
    backtest -- this previously broke resolution-time offsetting and per-horizon
    breakdowns for the hourly BTC market extension."""
    pred_df = pd.DataFrame([_pred_row("mktA", 0.0333, "2025-01-01T11:58:00Z",
                                        model_prob=0.85, raw_prob=0.55, outcome=1)])
    ex_cfg = ex.ExecutionConfig(fee_bps=0, base_half_spread_bps=1, min_half_spread_bps=1,
                                 max_half_spread_bps=50, liquidity_coeff=0)
    risk_cfg = rk.RiskConfig(min_model_edge=0.01, fractional_kelly_cap=1.0, max_risk_per_market=1.0,
                              max_participation_rate=1.0, max_trade_size=1e9,
                              max_total_capital_deployed=1.0, min_market_volume=0,
                              min_hours_to_resolution=0.0)
    result = bt.run_backtest(pred_df, "test_model", "base", ex_cfg, risk_cfg,
                              horizon_order=[0.0333])
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    sig = trade["signal"]
    assert abs(sig.horizon_hours - 0.0333) < 1e-6  # not truncated to 0
    # resolution time must be ~2 minutes after open, not immediate (int(0.0333)=0 bug)
    holding_seconds = (pd.Timestamp(trade["resolved_at"]) - pd.Timestamp(trade["opened_at"])).total_seconds()
    assert 100 < holding_seconds < 140


def test_higher_scenario_cost_reduces_or_matches_number_of_signals():
    pred_df = _two_market_dataset()
    risk_cfg = rk.RiskConfig(min_model_edge=0.05)
    opt_result = bt.run_backtest(pred_df, "test_model", "optimistic", ex.ExecutionConfig(), risk_cfg)
    cons_result = bt.run_backtest(pred_df, "test_model", "conservative", ex.ExecutionConfig(), risk_cfg)
    assert len(cons_result["signals"]) <= len(opt_result["signals"])
