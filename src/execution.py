"""
Execution-cost model.

Tier hierarchy actually available to this project:
  1. actual_book        -- real bid/ask depth. Only obtainable for markets that
                            are CURRENTLY OPEN (CLOB /book endpoint). Used by
                            the live paper trader, never by the historical
                            backtest (Polymarket's public API does not expose
                            historical order-book snapshots).
  2. trade_proxy         -- conservative quantile of trades in a short window
                            after the signal. Implemented for completeness /
                            future use, but Polymarket's public trades
                            endpoints either require an API key (clob
                            /trades) or returned no rows for historical
                            markets during this project's data collection
                            (data-api.polymarket.com /trades) -- see
                            reports/data_quality_report.md. Historical
                            backtests therefore fall through to tier 3.
  3. estimated_execution -- transparent spread + slippage model, a function of
                            the market's realized liquidity/volume. This is
                            the tier used for EVERY historical backtest fill
                            in this project, labeled accordingly.

Every historical trade in this project is therefore labeled
`estimated_execution` -- never `actual_book` -- and every backtest number
should be read with that caveat (see research_memo.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper_trading"

TIER_ACTUAL_BOOK = "actual_book"
TIER_TRADE_PROXY = "trade_proxy"
TIER_ESTIMATED = "estimated_execution"


@dataclass
class ExecutionConfig:
    # spread model: half_spread_bps = base_bps + liquidity_coeff / sqrt(liquidity_usd + 1)
    base_half_spread_bps: float = 50.0        # 0.50% min half-spread, these are thin binary markets
    liquidity_coeff: float = 5000.0
    min_half_spread_bps: float = 25.0
    max_half_spread_bps: float = 1500.0        # clip at 15% -- deep-tail illiquid markets

    # slippage: extra cost from participating in a fraction of daily volume
    slippage_coeff_bps: float = 300.0          # bps of extra cost at 100% participation
    max_participation_rate: float = 0.10       # never assume >10% of a day's volume is tradable

    fee_bps: float = 0.0    # Polymarket charges no standard taker fee on these markets (documented
                             # assumption -- see README); kept configurable for sensitivity analysis.

    # --- Absolute spread floor -------------------------------------------------
    # The proportional model above scales the spread with the contract price, but
    # real books on these markets quote a roughly FIXED 1-cent spread regardless of
    # level (tick-size driven; Polymarket's own UI reports "Spread: 1c"). The
    # proportional form therefore understates cost badly in the tails -- at a 5c
    # contract it charged ~0.03c one-way against ~0.5c actually quoted, ~17x light.
    # This floor is an observed market fact, so it is NOT relaxed by the optimistic
    # scenario: you cannot trade inside the quoted spread.
    min_half_spread_abs: float = 0.005          # $0.005 = half of the observed 1c spread

    # --- Order-book depth / capacity ------------------------------------------
    # Position size is limited by what the book can actually absorb, which is not
    # the same as a share of daily volume. Live books sampled at trade-relevant
    # horizons showed only a few hundred dollars within a cent of the touch, while
    # the volume-based model permitted ~$2,000 positions. The book is modelled as
    # `touch_depth_usd` fillable at the touch, plus `depth_per_cent_usd` for each
    # additional cent walked, refusing to walk deeper than `max_walk_cents`.
    # Defaults are anchored to live samples (buy_depth_1c ~$100-350,
    # buy_depth_5c ~$430-800) and are deliberately scenario-scaled, because the
    # true depth distribution at trade time is still being measured -- the live
    # paper trader logs it on every check-in. See README "Order-book depth".
    touch_depth_usd: float = 300.0
    depth_per_cent_usd: float = 150.0
    max_walk_cents: int = 3

    # scenario multipliers on spread & slippage
    scenario_multipliers: dict = None
    # scenario multipliers on DEPTH -- inverted sense: optimistic means a deeper,
    # more forgiving book, conservative means a thinner one.
    depth_multipliers: dict = None

    def __post_init__(self):
        if self.scenario_multipliers is None:
            self.scenario_multipliers = {"optimistic": 0.5, "base": 1.0, "conservative": 2.0}
        if self.depth_multipliers is None:
            self.depth_multipliers = {"optimistic": 2.0, "base": 1.0, "conservative": 0.5}

    @classmethod
    def from_live_measurements(cls, family: str, paper_dir: Path | None = None, **overrides) -> "ExecutionConfig":
        """Build a config whose spread floor and order-book depth are the CURRENT
        live-measured averages for `family` -- one number, not an
        optimistic/base/conservative sweep -- per explicit direction replacing the
        old approach of a fixed depth constant hand-picked once from a single
        Aug28-Sep8 snapshot and then bracketed with 0.5x/2x multipliers. This
        reads paper_trading/mark_to_market*.jsonl fresh on every call (see
        measure_live_execution_costs), so re-running the backtest after the live
        system has logged more check-ins picks up a more current, more
        representative average automatically -- there is no cached/frozen value
        to go stale."""
        m = measure_live_execution_costs(family, paper_dir)
        return cls(
            min_half_spread_abs=m["mean_half_spread"],
            touch_depth_usd=m["touch_depth_usd"],
            depth_per_cent_usd=m["depth_per_cent_usd"],
            scenario_multipliers={"live": 1.0},
            depth_multipliers={"live": 1.0},
            **overrides,
        )


def measure_live_execution_costs(family: str, paper_dir: Path | None = None) -> dict:
    """Average spread and order-book depth actually observed by the live paper
    trader for `family`, computed fresh from paper_trading/mark_to_market*.jsonl
    every time this is called. Pools both legs of the market (YES's buy_depth_*
    and NO's no_buy_depth_*) into one estimate, since the depth model in this
    module is a single generic constant applied to whichever side is traded, not
    a per-side parameter.

    touch_depth_usd is the mean notional within 1c of the touch (YES's
    buy_depth_1c and NO's no_buy_depth_1c pooled). depth_per_cent_usd is the mean
    ADDITIONAL notional per further cent walked -- YES contributes
    (buy_depth_5c - buy_depth_1c) / 4 per snapshot (4 more cents recorded, 2c
    through 5c) and NO contributes (no_buy_depth_2c - no_buy_depth_1c) / 1 (NO
    only ever had 1c/2c logged, see paper_trader.OrderBookSummary)."""
    paper_dir = paper_dir or PAPER_DIR
    fname = "mark_to_market.jsonl" if family == "daily" else "mark_to_market_hourly.jsonl"
    path = paper_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"No live check-in log at {path} -- need at least some live "
                                 f"paper-trade history for family={family!r} to measure execution costs from.")

    spreads, touch_depths, per_cent_depths = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("status") == "no_active_market":
            continue
        spread = d.get("spread")
        b1, b5 = d.get("buy_depth_1c"), d.get("buy_depth_5c")
        n1, n2 = d.get("no_buy_depth_1c"), d.get("no_buy_depth_2c")
        if spread:
            spreads.append(spread)
        if b1 is not None:
            touch_depths.append(b1)
        if n1 is not None:
            touch_depths.append(n1)
        if b1 is not None and b5 is not None and b5 >= b1:
            per_cent_depths.append((b5 - b1) / 4.0)
        if n1 is not None and n2 is not None and n2 >= n1:
            per_cent_depths.append(n2 - n1)

    if not spreads or not touch_depths:
        raise ValueError(f"Live log for family={family!r} has no rows with real order-book data yet "
                          "-- cannot measure execution costs from it.")

    return {
        "n_snapshots": len(spreads),
        "mean_half_spread": float(np.mean(spreads)) / 2.0,
        "touch_depth_usd": float(np.mean(touch_depths)),
        "depth_per_cent_usd": float(np.mean(per_cent_depths)) if per_cent_depths else 0.0,
    }


def max_fillable_usd(cfg: ExecutionConfig, scenario: str) -> float:
    """Largest notional the modelled book can absorb without walking further than
    `max_walk_cents` past the touch."""
    m = cfg.depth_multipliers[scenario]
    return cfg.touch_depth_usd * m + cfg.depth_per_cent_usd * m * cfg.max_walk_cents


def walk_book(size_usd: float, touch_price: float, cfg: ExecutionConfig,
               scenario: str) -> tuple[float, float] | None:
    """Fill `size_usd` against a linear book starting at `touch_price`.

    Returns (volume_weighted_avg_price, cents_walked), or None if the order cannot
    be filled within `max_walk_cents` -- callers must treat None as UNFILLED rather
    than silently filling at a better price."""
    m = cfg.depth_multipliers[scenario]
    touch_depth, per_cent = cfg.touch_depth_usd * m, cfg.depth_per_cent_usd * m

    remaining, contracts, cents = size_usd, 0.0, 0
    while remaining > 1e-9:
        if cents > cfg.max_walk_cents:
            return None
        px = min(touch_price + cents * 0.01, 0.999)
        take = min(remaining, touch_depth if cents == 0 else per_cent)
        if take <= 0:
            return None
        contracts += take / px
        remaining -= take
        cents += 1
    if contracts <= 0:
        return None
    return size_usd / contracts, float(cents - 1)


def half_spread_bps(liquidity_usd: float, cfg: ExecutionConfig, scenario: str) -> float:
    liquidity_usd = max(liquidity_usd, 0.0)
    raw = cfg.base_half_spread_bps + cfg.liquidity_coeff / np.sqrt(liquidity_usd + 1.0)
    raw = float(np.clip(raw, cfg.min_half_spread_bps, cfg.max_half_spread_bps))
    return raw * cfg.scenario_multipliers[scenario]


def slippage_bps(trade_size_usd: float, market_volume_usd: float, cfg: ExecutionConfig, scenario: str) -> float:
    if market_volume_usd <= 0:
        return cfg.max_half_spread_bps  # no volume data -> most conservative assumption
    participation = trade_size_usd / market_volume_usd
    raw = cfg.slippage_coeff_bps * participation
    return raw * cfg.scenario_multipliers[scenario]


@dataclass
class ExecutionQuote:
    side: str                # "buy_yes" or "buy_no"
    mid_price: float
    estimated_buy_price: float
    estimated_sell_price: float
    half_spread_bps: float
    slippage_bps: float
    fee_bps: float
    quality_tier: str
    participation_capped: bool
    cents_walked: float = 0.0   # how far past the touch the fill had to walk
    unfilled: bool = False      # book could not absorb the order within max_walk_cents


def quote_estimated_execution(side: str, yes_mid: float, trade_size_usd: float,
                               market_volume_usd: float, market_liquidity_usd: float,
                               cfg: ExecutionConfig, scenario: str = "base") -> ExecutionQuote:
    """side: 'buy_yes' or 'buy_no'. Mid for the traded leg: yes_mid for buy_yes,
    (1-yes_mid) for buy_no (long-only instruments, no short-selling assumed --
    see README 'Execution assumptions').

    Gamma's `liquidityNum` field is structurally 0 for every one of these CLOB
    order-book markets (it reflects AMM pool liquidity, which this market type
    doesn't use -- confirmed across all 2,520 observations, see
    data_quality_report.md). Total market volume is therefore used as the
    depth proxy for the spread model too, with `market_liquidity_usd` kept as
    a separate, distinct input so a future data source that does populate it
    doesn't require an interface change."""
    leg_mid = yes_mid if side == "buy_yes" else (1.0 - yes_mid)
    leg_mid = float(np.clip(leg_mid, 0.001, 0.999))

    depth_proxy = max(market_liquidity_usd, market_volume_usd)

    # Half-spread: the greater of the proportional model and the absolute floor the
    # real book actually quotes. The floor is not scenario-relaxed -- you cannot
    # trade inside the quoted spread no matter how optimistic the assumptions.
    hs_proportional = leg_mid * half_spread_bps(depth_proxy, cfg, scenario) / 10_000.0
    half_spread = max(hs_proportional, cfg.min_half_spread_abs)

    max_size = cfg.max_participation_rate * market_volume_usd if market_volume_usd > 0 else 0.0
    capped = trade_size_usd > max_size
    eff_size = min(trade_size_usd, max_size) if max_size > 0 else trade_size_usd

    # Walking the book replaces the old volume-based slippage term -- keeping both
    # would double-count the same cost. `unfilled` propagates a genuine no-fill.
    touch_buy = float(np.clip(leg_mid + half_spread, 0.001, 0.999))
    walked = walk_book(eff_size, touch_buy, cfg, scenario) if eff_size > 0 else (touch_buy, 0.0)
    if walked is None:
        # Price the no-fill at the WORST level we would have had to reach, not the
        # touch: returning the touch here would make an unfillable order look
        # cheaper than a fillable one, so any caller that overlooked `unfilled`
        # would silently prefer the thinner book. Erring worst-case keeps the
        # optimistic < base < conservative ordering honest either way.
        buy_price = float(np.clip(touch_buy + cfg.max_walk_cents * 0.01, 0.001, 0.999))
        cents_walked, unfilled = float(cfg.max_walk_cents), True
    else:
        buy_price, cents_walked = walked
        unfilled = False

    sell_price = float(np.clip(leg_mid - half_spread, 0.001, 0.999))
    buy_price = float(np.clip(buy_price, 0.001, 0.999))

    # Report the realised costs in bps of the leg mid so downstream accounting and
    # the dashboard's cost attribution stay meaningful.
    hs_bps_eff = half_spread / leg_mid * 10_000.0
    slip_bps_eff = max(buy_price - touch_buy, 0.0) / leg_mid * 10_000.0

    return ExecutionQuote(
        side=side, mid_price=leg_mid, estimated_buy_price=buy_price, estimated_sell_price=sell_price,
        half_spread_bps=hs_bps_eff, slippage_bps=slip_bps_eff, fee_bps=cfg.fee_bps,
        quality_tier=TIER_ESTIMATED, participation_capped=capped,
        cents_walked=cents_walked, unfilled=unfilled,
    )


def trade_proxy_fill(side: str, trades_after_signal: list[dict], quantile: float = 0.75) -> float | None:
    """Tier-2 helper (for future use / live venues with a trades feed): a BUY fill
    uses a conservative HIGH quantile of trade prices in the execution window; a
    SELL fill uses a conservative LOW quantile. Returns None ('unfilled') if no
    eligible trades occurred in the window -- caller must not use later prices to
    revise the trading signal itself, only to determine whether/at-what-price a fill
    happened."""
    if not trades_after_signal:
        return None
    prices = sorted(t["price"] for t in trades_after_signal)
    q = quantile if side.startswith("buy") else (1 - quantile)
    idx = int(np.clip(round(q * (len(prices) - 1)), 0, len(prices) - 1))
    return float(prices[idx])
