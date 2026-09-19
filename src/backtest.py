"""
Execution-aware, walk-forward backtest.

Processes each market's observations in true chronological order (horizon
24h -> 1h). At most one position is opened per market -- the first horizon at
which the net-of-cost edge clears the minimum-edge bar and all risk checks
pass -- so a single coin-flip market never contributes correlated risk at
multiple horizons. YES and NO are modeled as separate long-only contracts
(no short-selling / no costless "sell YES" assumed, per the execution
assumptions in README.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import execution as ex
from . import risk as rk

HORIZON_ORDER = [24, 12, 6, 3, 1]


@dataclass
class Signal:
    slug: str
    horizon_hours: float
    observation_timestamp: str
    side: str
    model_probability: float
    market_probability: float
    estimated_buy_price: float
    estimated_sell_price: float
    gross_edge: float
    fees: float
    slippage: float
    net_expected_edge: float
    market_volume: float
    market_liquidity: float
    execution_quality: str


def evaluate_signal(row: pd.Series, ex_cfg: ex.ExecutionConfig, risk_cfg: rk.RiskConfig,
                     scenario: str, nominal_trade_size: float) -> Signal | None:
    """Evaluate BUY YES vs BUY NO for one observation; return the better-edge side
    if it clears minimum_edge, else None. Uses only this observation's own fields
    (all already point-in-time by construction of observations.parquet)."""
    mp = row["model_probability"]
    mkt_p = row["raw_probability"]
    vol, liq = row["market_volume"], row["market_liquidity"]

    best = None
    for side, model_p_side in (("buy_yes", mp), ("buy_no", 1 - mp)):
        q = ex.quote_estimated_execution(side, mkt_p, nominal_trade_size, vol, liq, ex_cfg, scenario)
        if q.unfilled:
            continue   # book cannot absorb this order -- no signal, not a free fill
        gross_edge = model_p_side - q.estimated_buy_price
        fee_cost = q.estimated_buy_price * (q.fee_bps / 10_000.0)
        slip_cost = q.estimated_buy_price * (q.slippage_bps / 10_000.0)
        net_edge = gross_edge - fee_cost - slip_cost
        if net_edge > risk_cfg.min_model_edge:
            if best is None or net_edge > best.net_expected_edge:
                best = Signal(
                    slug=row["slug"], horizon_hours=float(row["horizon_hours"]),
                    observation_timestamp=row["observation_timestamp"], side=side,
                    model_probability=model_p_side, market_probability=q.mid_price,
                    estimated_buy_price=q.estimated_buy_price, estimated_sell_price=q.estimated_sell_price,
                    gross_edge=gross_edge, fees=fee_cost, slippage=slip_cost, net_expected_edge=net_edge,
                    market_volume=vol, market_liquidity=liq, execution_quality=q.quality_tier,
                )
    return best


def run_backtest(pred_df: pd.DataFrame, model_name: str, scenario: str,
                  ex_cfg: ex.ExecutionConfig | None = None, risk_cfg: rk.RiskConfig | None = None,
                  horizon_order: list[float] = HORIZON_ORDER) -> dict:
    ex_cfg = ex_cfg or ex.ExecutionConfig()
    risk_cfg = risk_cfg or rk.RiskConfig()

    sub = pred_df[pred_df["model"] == model_name].copy()
    sub["observation_timestamp"] = pd.to_datetime(sub["observation_timestamp"], utc=True)

    # resolution time = obs_ts(h) + h hours, recovered from any one row per market.
    # horizon_hours may be fractional (e.g. 0.25 = 15 minutes) for sub-daily market
    # families, so this must NOT truncate to int.
    resolve_map = {}
    for slug, g in sub.groupby("slug"):
        row = g.iloc[0]
        resolve_map[slug] = row["observation_timestamp"] + pd.to_timedelta(float(row["horizon_hours"]), unit="h")

    portfolio = rk.PortfolioState(risk_cfg)
    all_signals, trades, unfilled = [], [], []

    markets_sorted = sorted(sub["slug"].unique(), key=lambda s: resolve_map[s])

    # event queue: process opens in observation-time order, settle at resolution.
    # (positions never overlap in a way that matters for this model since we
    # settle immediately when we reach each market's resolution time in the loop)
    pending_settlement = []  # list of (resolve_time, slug, outcome_up)

    for slug in markets_sorted:
        g = sub[sub["slug"] == slug].sort_values(
            "horizon_hours", key=lambda s: s.map({h: i for i, h in enumerate(horizon_order)}))

        # settle any positions that should have resolved by the time this market's
        # earliest observation occurs
        now = g["observation_timestamp"].min()
        _settle_due(pending_settlement, now, portfolio, ex_cfg, trades)

        outcome_up = int(g.iloc[0]["outcome_up"])
        opened = False
        for _, row in g.iterrows():
            if opened:
                break
            hours_to_res = row["horizon_hours"]
            if hours_to_res < risk_cfg.min_hours_to_resolution:
                continue
            if row["market_volume"] < risk_cfg.min_market_volume:
                continue

            # Quote at the size we could actually trade, not the nominal cap: with
            # the depth model in play, quoting at max_trade_size would report every
            # order as unfillable and suppress all signals.
            nominal = min(risk_cfg.max_trade_size, ex.max_fillable_usd(ex_cfg, scenario))
            sig = evaluate_signal(row, ex_cfg, risk_cfg, scenario, nominal_trade_size=nominal)
            if sig is None:
                continue
            all_signals.append(sig)

            if not portfolio.can_open_new_position():
                unfilled.append({**sig.__dict__, "reason": "risk_limit_blocked"})
                continue

            stake = portfolio.size_position(sig.model_probability, sig.estimated_buy_price,
                                             sig.market_volume, sig.market_liquidity,
                                             max_depth_usd=ex.max_fillable_usd(ex_cfg, scenario))
            if stake < 1.0:
                unfilled.append({**sig.__dict__, "reason": "position_size_zero"})
                continue

            n_contracts = stake / sig.estimated_buy_price
            portfolio.open_position(slug, sig.side, stake, sig.estimated_buy_price, n_contracts,
                                     row["observation_timestamp"], resolve_map[slug], meta={"signal": sig})
            pending_settlement.append((resolve_map[slug], slug, outcome_up))
            portfolio.mark_equity(row["observation_timestamp"])
            opened = True

    # settle anything left
    _settle_due(pending_settlement, pd.Timestamp.max.tz_localize("UTC"), portfolio, ex_cfg, trades)

    return {
        "model": model_name, "scenario": scenario,
        "signals": all_signals, "trades": trades, "unfilled": unfilled,
        "equity_curve": portfolio.equity_curve, "final_equity": portfolio.equity,
        "risk_cfg": risk_cfg, "ex_cfg": ex_cfg,
    }


def _settle_due(pending_settlement, now, portfolio, ex_cfg, trades):
    still_pending = []
    for resolve_time, slug, outcome_up in pending_settlement:
        if resolve_time <= now and slug in portfolio.open_positions:
            trade = portfolio.settle_position(slug, outcome_up, ex_cfg.fee_bps)
            trade["resolved_at"] = resolve_time
            trades.append(trade)
        elif slug in portfolio.open_positions:
            still_pending.append((resolve_time, slug, outcome_up))
    pending_settlement[:] = still_pending
