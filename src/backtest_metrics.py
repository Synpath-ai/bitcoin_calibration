"""Summary statistics and regime breakdowns for a completed backtest run."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_TRADES_FOR_SHARPE = 30


def trades_to_df(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        sig = t.get("signal")
        row = {
            "slug": t["slug"], "side": t["side"], "stake": t["stake"],
            "buy_price": t["buy_price"], "n_contracts": t["n_contracts"],
            "opened_at": t["opened_at"], "resolved_at": t.get("resolved_at"),
            "won": t["won"], "gross_payout": t["gross_payout"], "fee_cost": t["fee_cost"],
            "pnl_net": t["pnl_net"], "pnl_gross": t["gross_payout"] - t["stake"],
        }
        if sig is not None:
            row.update({
                "horizon_hours": sig.horizon_hours, "gross_edge": sig.gross_edge,
                "net_expected_edge": sig.net_expected_edge, "slippage": sig.slippage,
                "fees_expected": sig.fees, "market_volume": sig.market_volume,
                "market_liquidity": sig.market_liquidity, "execution_quality": sig.execution_quality,
                "market_probability": sig.market_probability,
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    df["holding_period_hours"] = (
        pd.to_datetime(df["resolved_at"], utc=True) - pd.to_datetime(df["opened_at"], utc=True)
    ).dt.total_seconds() / 3600.0
    return df


def summarize(result: dict) -> dict:
    trades_df = trades_to_df(result["trades"])
    n_signals = len(result["signals"])
    n_trades = len(trades_df)
    n_unfilled = len(result["unfilled"])

    if trades_df.empty:
        return {
            "model": result["model"], "scenario": result["scenario"],
            "n_signals": n_signals, "n_fills": 0, "fill_rate": 0.0,
            "gross_pnl": 0.0, "net_pnl": 0.0, "final_equity": result["final_equity"],
            "note": "no trades executed",
        }

    gross_pnl = trades_df["pnl_gross"].sum()
    net_pnl = trades_df["pnl_net"].sum()
    total_stake = trades_df["stake"].sum()
    hit_rate = trades_df["won"].mean()

    eq = pd.DataFrame(result["equity_curve"])
    max_dd = eq["drawdown"].max() if not eq.empty else 0.0

    # Sharpe on per-trade net returns (stake-normalized), NOT annualized (holding
    # periods vary and overlap); flagged with an explicit sample-size warning.
    per_trade_return = trades_df["pnl_net"] / trades_df["stake"]
    sharpe = per_trade_return.mean() / per_trade_return.std(ddof=1) if per_trade_return.std(ddof=1) > 0 else np.nan
    sharpe_warning = n_trades < MIN_TRADES_FOR_SHARPE

    summary = {
        "model": result["model"], "scenario": result["scenario"],
        "n_signals": n_signals, "n_fills": n_trades,
        "fill_rate": n_trades / n_signals if n_signals else 0.0,
        "n_unfilled": n_unfilled,
        "gross_pnl": float(gross_pnl), "net_pnl": float(net_pnl),
        "return_on_deployed_capital": float(net_pnl / total_stake) if total_stake else 0.0,
        "final_equity": result["final_equity"],
        "total_return_on_initial_capital": float(
            (result["final_equity"] - result["risk_cfg"].initial_capital) / result["risk_cfg"].initial_capital),
        "max_drawdown": float(max_dd),
        "sharpe_per_trade": float(sharpe) if not np.isnan(sharpe) else None,
        "sharpe_sample_size_warning": sharpe_warning,
        "hit_rate": float(hit_rate),
        "avg_gross_edge": float(trades_df["gross_edge"].mean()),
        "avg_net_expected_edge": float(trades_df["net_expected_edge"].mean()),
        "total_fees": float(trades_df["fee_cost"].sum()),
        "total_slippage_cost_est": float((trades_df["slippage"] * trades_df["stake"]).sum()),
        "turnover": float(total_stake),
        "avg_position_size": float(trades_df["stake"].mean()),
        "avg_holding_period_hours": float(trades_df["holding_period_hours"].mean()),
    }
    return summary


def breakdown(trades_df: pd.DataFrame, by: str, bins=None, labels=None) -> pd.DataFrame:
    if trades_df.empty or by not in trades_df.columns:
        return pd.DataFrame()
    d = trades_df.copy()
    if bins is not None:
        d["_bucket"] = pd.cut(d[by], bins=bins, labels=labels)
        group_col = "_bucket"
    else:
        group_col = by
    agg = d.groupby(group_col, observed=True).agg(
        n_trades=("pnl_net", "size"),
        gross_pnl=("pnl_gross", "sum"),
        net_pnl=("pnl_net", "sum"),
        hit_rate=("won", "mean"),
        avg_stake=("stake", "mean"),
    ).reset_index()
    return agg
