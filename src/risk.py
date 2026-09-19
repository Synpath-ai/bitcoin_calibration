"""Position sizing and portfolio risk constraints."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskConfig:
    initial_capital: float = 100_000.0
    max_risk_per_market: float = 0.02          # fraction of capital
    max_total_capital_deployed: float = 0.60   # fraction of capital simultaneously at risk
    max_participation_rate: float = 0.10       # of market volume (also enforced in execution.py)
    max_trade_size: float = 5_000.0            # absolute $ cap per trade
    max_simultaneous_positions: int = 25
    min_market_volume: float = 5_000.0
    min_model_edge: float = 0.03               # minimum_edge in the buy rule
    min_hours_to_resolution: float = 0.25      # stop opening positions inside this window
    fractional_kelly_cap: float = 0.10         # "0.10 x Kelly size"
    max_drawdown_stop: float = 0.25            # halt new positions if drawdown exceeds this


def kelly_fraction(model_prob: float, price: float) -> float:
    """Kelly fraction of bankroll for a binary contract costing `price`, paying $1
    on win with probability `model_prob`. f* = (p - c) / (1 - c), clipped to [0, 1]."""
    if price >= 0.999 or price <= 0.001:
        return 0.0
    f = (model_prob - price) / (1.0 - price)
    return float(np.clip(f, 0.0, 1.0))


@dataclass
class PortfolioState:
    cfg: RiskConfig
    cash: float = field(init=False)
    equity: float = field(init=False)
    peak_equity: float = field(init=False)
    open_positions: dict = field(default_factory=dict)   # slug -> position dict
    equity_curve: list = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.cfg.initial_capital
        self.equity = self.cfg.initial_capital
        self.peak_equity = self.cfg.initial_capital

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def deployed_capital(self) -> float:
        return sum(p["stake"] for p in self.open_positions.values())

    def can_open_new_position(self) -> bool:
        if self.drawdown >= self.cfg.max_drawdown_stop:
            return False
        if len(self.open_positions) >= self.cfg.max_simultaneous_positions:
            return False
        if self.deployed_capital >= self.cfg.max_total_capital_deployed * self.equity:
            return False
        return True

    def size_position(self, model_prob: float, buy_price: float,
                       market_volume: float, market_liquidity: float,
                       max_depth_usd: float | None = None) -> float:
        """Conservative min-of-everything position sizing, in dollars of stake.

        `max_depth_usd` is what the order book can actually absorb (see
        execution.max_fillable_usd). It is a distinct constraint from the
        volume-based participation cap: volume is a flow measured over the
        market's whole life, whereas depth is the stock resting on the book right
        now, and live sampling showed the latter to be far smaller. Passing None
        keeps the pre-depth-model behaviour."""
        kelly_f = kelly_fraction(model_prob, buy_price)
        kelly_size = self.cfg.fractional_kelly_cap * kelly_f * self.equity

        # NOTE: Gamma's `market_liquidity` field is structurally 0 for every CLOB
        # order-book market in this dataset (it reflects AMM-pool liquidity, which
        # this market type doesn't use -- see data_quality_report.md). Both caps
        # therefore fall back to total market volume as the best available depth
        # proxy when liquidity is unreported, rather than zeroing out every trade.
        depth_proxy = max(market_liquidity, market_volume, 0.0)
        per_market_cap = self.cfg.max_risk_per_market * self.equity
        liquidity_cap = self.cfg.max_participation_rate * depth_proxy
        participation_cap = self.cfg.max_participation_rate * max(market_volume, 0.0)
        remaining_capital = max(self.cfg.max_total_capital_deployed * self.equity - self.deployed_capital, 0.0)

        caps = [kelly_size, liquidity_cap, participation_cap, per_market_cap,
                remaining_capital, self.cfg.max_trade_size]
        if max_depth_usd is not None:
            caps.append(max_depth_usd)
        return max(min(caps), 0.0)

    def open_position(self, slug: str, side: str, stake: float, buy_price: float,
                       n_contracts: float, opened_at, resolves_at, meta: dict):
        self.cash -= stake
        self.open_positions[slug] = {
            "slug": slug, "side": side, "stake": stake, "buy_price": buy_price,
            "n_contracts": n_contracts, "opened_at": opened_at, "resolves_at": resolves_at,
            **meta,
        }

    def settle_position(self, slug: str, outcome_up: int, fee_bps: float) -> dict:
        pos = self.open_positions.pop(slug)
        won = (outcome_up == 1) if pos["side"] == "buy_yes" else (outcome_up == 0)
        gross_payout = pos["n_contracts"] * 1.0 if won else 0.0
        fee_cost = pos["stake"] * (fee_bps / 10_000.0)
        pnl_net = gross_payout - pos["stake"] - fee_cost
        self.cash += gross_payout - fee_cost
        self.equity = self.cash + sum(p["stake"] for p in self.open_positions.values())
        self.peak_equity = max(self.peak_equity, self.equity)
        return {**pos, "won": won, "gross_payout": gross_payout, "fee_cost": fee_cost, "pnl_net": pnl_net}

    def mark_equity(self, ts):
        self.equity = self.cash + sum(p["stake"] for p in self.open_positions.values())
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append({"timestamp": ts, "equity": self.equity, "cash": self.cash,
                                   "deployed": self.deployed_capital, "drawdown": self.drawdown})
