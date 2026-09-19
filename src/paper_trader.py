"""
Live paper trader for the current active BTC Up/Down market -- either the
**daily** family or the **hourly** extended-sample family (see
src/market_matching_hourly.py for why the latter needs its own discovery path).

Reads REAL live data (Gamma market metadata, CLOB order book, Binance spot
price) and produces a calibrated fair value + a simulated (never real) trade
recommendation. No authenticated orders are ever placed -- this module only
calls public read endpoints.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import btc_client as btc_c
from . import execution as ex
from . import market_matching as mm
from . import market_matching_hourly as mmh
from . import models as mdl
from . import news_client as nc
from . import polymarket_client as pc
from . import risk as rk
from . import social_client as sc

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
PAPER_DIR = ROOT / "paper_trading"
PAPER_DIR.mkdir(exist_ok=True)

FAMILIES = ("daily", "hourly")

# The backtest evaluates each market at exactly these horizons and opens at most
# one position per market, at the FIRST that clears the edge bar. The live trader
# wakes every 10 minutes, so without this grid it evaluates a daily market ~288
# times instead of 5 and enters at whatever moment the edge first happens to
# clear -- observed live entries sat at 23.3-23.8h, points the backtest never
# looks at. That is a different strategy, so the grid is enforced here too.
BACKTEST_HORIZONS_HOURS = {
    "daily": [24.0, 12.0, 6.0, 3.0, 1.0],
    "hourly": [45 / 60, 30 / 60, 15 / 60, 5 / 60, 2 / 60],
}
# A 10-minute cadence cannot land exactly on a horizon, so accept a window around
# it. Half the cadence keeps windows from overlapping and makes at most one
# check-in per horizon eligible.
HORIZON_TOLERANCE_HOURS = 5 / 60


def nearest_backtest_horizon(family: str, hours_to_resolution: float) -> float | None:
    """The backtest horizon this check-in corresponds to, or None if we are between
    horizons and the backtest would not have evaluated at all.

    Matches the CLOSEST horizon rather than the first within tolerance: the hourly
    grid puts its 5-minute and 2-minute points only 3 minutes apart, so a
    first-match with a 5-minute window silently assigned 2-minute check-ins to the
    5-minute horizon.

    Note the cadence limit this exposes -- a 10-minute wake-up cannot resolve
    horizons 3 minutes apart, so in practice the hourly family covers its
    45/30/15-minute horizons and only opportunistically the tighter two."""
    horizons = BACKTEST_HORIZONS_HOURS[family]
    best = min(horizons, key=lambda h: abs(hours_to_resolution - h))
    return best if abs(hours_to_resolution - best) <= HORIZON_TOLERANCE_HOURS else None


def portfolio_state_path(family: str) -> Path:
    return PAPER_DIR / f"portfolio_{family}.json"


def load_portfolio(family: str, cfg: rk.RiskConfig) -> rk.PortfolioState:
    """Restore the portfolio across check-ins.

    Previously a fresh PortfolioState was constructed on every run, so equity was
    always the full initial capital and there were never any open positions. That
    silently disabled every portfolio-level risk control the backtest enforces --
    max_simultaneous_positions, max_total_capital_deployed and max_drawdown_stop
    can only bind if state persists between trades."""
    pf = rk.PortfolioState(cfg)
    path = portfolio_state_path(family)
    if not path.exists():
        return pf
    st = json.loads(path.read_text())
    pf.cash = st["cash"]
    pf.equity = st["equity"]
    pf.peak_equity = st["peak_equity"]
    pf.open_positions = st.get("open_positions", {})
    return pf


def save_portfolio(family: str, pf: rk.PortfolioState) -> None:
    portfolio_state_path(family).write_text(json.dumps({
        "cash": pf.cash, "equity": pf.equity, "peak_equity": pf.peak_equity,
        "open_positions": pf.open_positions,
    }, default=str, indent=2))


def settle_due_positions(family: str, pf: rk.PortfolioState, fee_bps: float) -> list[dict]:
    """Close out any open position whose market has resolved, mirroring the
    backtest's settlement so realized P&L accrues to the same portfolio the risk
    limits are computed against."""
    now = pd.Timestamp.now(tz="UTC")
    settled = []
    for slug, pos in list(pf.open_positions.items()):
        if pd.Timestamp(pos["resolves_at"]) > now:
            continue
        outcome = _resolved_outcome(slug, family)
        if outcome is None:
            continue      # resolved but oracle not final yet -- leave it open
        settled.append(pf.settle_position(slug, outcome, fee_bps))
    return settled


def _resolved_outcome(slug: str, family: str) -> int | None:
    """1 if Up won, 0 if Down won, None if not yet unambiguously resolved."""
    try:
        markets = pc.get_market_by_slug(slug)
    except RuntimeError:
        return None
    for m in markets:
        if m.get("slug") != slug or not m.get("closed"):
            continue
        try:
            prices = json.loads(m.get("outcomePrices") or "[]")
            up, down = float(prices[0]), float(prices[1])
        except (ValueError, IndexError):
            return None
        if up >= 0.99 and down <= 0.01:
            return 1
        if down >= 0.99 and up <= 0.01:
            return 0
    return None


def paper_log_path(family: str) -> Path:
    return PAPER_DIR / (f"paper_trades.jsonl" if family == "daily" else f"paper_trades_{family}.jsonl")


def mtm_log_path(family: str) -> Path:
    return PAPER_DIR / (f"mark_to_market.jsonl" if family == "daily" else f"mark_to_market_{family}.jsonl")


@dataclass
class OrderBookSummary:
    best_bid: float | None
    best_ask: float | None
    mid: float | None
    spread: float | None
    bid_depth_top5: float
    ask_depth_top5: float
    # Executable depth near the touch, in USD notional. This is what actually
    # constrains position size: `buy_depth_1c` is how much can be bought without
    # paying more than 1 cent above the best ask. The backtest's slippage model
    # (a function of *market volume*) is not the same thing and may be far more
    # optimistic -- these fields exist to measure the real constraint at the
    # moments the strategy would actually trade, which no historical Polymarket
    # data source exposes. See README "Order-book depth".
    buy_depth_1c: float = 0.0
    buy_depth_2c: float = 0.0
    buy_depth_5c: float = 0.0
    sell_depth_1c: float = 0.0
    sell_depth_2c: float = 0.0
    sell_depth_5c: float = 0.0
    n_ask_levels: int = 0
    n_bid_levels: int = 0


def find_current_active_market(family: str = "daily") -> dict | None:
    """Discover currently open BTC Up/Down markets in the given family and return
    the one resolving soonest (the 'current' market a trader would actually act
    on). `family` is 'daily' or 'hourly'."""
    if family == "hourly":
        # only need a small recent window -- the currently-open market(s) are
        # always among the most-recently-created (ascending=False, default).
        accepted, _ = mmh.discover_hourly_markets(max_events=200, closed=False)
    else:
        accepted, _ = mm.discover_daily_markets(closed=False)
    now = pd.Timestamp.now(tz="UTC")
    open_now = [c for c in accepted if not c.closed and
                pd.Timestamp(c.start_date) <= now < pd.Timestamp(c.end_date)]
    if not open_now:
        return None
    open_now.sort(key=lambda c: pd.Timestamp(c.end_date))
    return asdict(open_now[0])


def summarize_book(book: dict) -> OrderBookSummary:
    bids = sorted(book.get("bids", []), key=lambda x: -float(x["price"]))
    asks = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
    best_bid = float(bids[0]["price"]) if bids else None
    best_ask = float(asks[0]["price"]) if asks else None
    mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
    spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
    bid_depth = sum(float(b["size"]) for b in bids[:5])
    ask_depth = sum(float(a["size"]) for a in asks[:5])

    def notional_within(levels, touch, cents, side):
        """USD notional available without going more than `cents` past the touch."""
        if touch is None:
            return 0.0
        limit = touch + cents / 100 if side == "buy" else touch - cents / 100
        total = 0.0
        for lv in levels:
            px = float(lv["price"])
            if (side == "buy" and px <= limit + 1e-9) or (side == "sell" and px >= limit - 1e-9):
                total += px * float(lv["size"])
        return total

    return OrderBookSummary(
        best_bid, best_ask, mid, spread, bid_depth, ask_depth,
        buy_depth_1c=notional_within(asks, best_ask, 1, "buy"),
        buy_depth_2c=notional_within(asks, best_ask, 2, "buy"),
        buy_depth_5c=notional_within(asks, best_ask, 5, "buy"),
        sell_depth_1c=notional_within(bids, best_bid, 1, "sell"),
        sell_depth_2c=notional_within(bids, best_bid, 2, "sell"),
        sell_depth_5c=notional_within(bids, best_bid, 5, "sell"),
        n_ask_levels=len(asks), n_bid_levels=len(bids),
    )


def walk_real_book(stake: float, best_ask: float, depth_1c: float,
                    depth_2c: float, depth_5c: float) -> float | None:
    """VWAP entry price walked against the ACTUAL recorded depth bands (1c/2c/5c
    notional-within-band, as measured from the real live order book), not a flat
    fill at best_ask. Each band's contribution is priced at its midpoint (e.g.
    the 0-1c tranche at best_ask+0.005) since only the cumulative notional per
    band is available here, not each individual price level -- a reasonable
    approximation, not a true level-by-level VWAP.

    Returns None if `stake` cannot be filled within the 5c band actually
    recorded, matching the project's 'unfilled, not silently filled at a good
    price' rule.

    Validated first in scripts/21_rescore_live_with_fresh_model.py by rescoring
    92 historical live check-ins: the walk shrank stake-weighted returns from
    +23.74% to +21.89% and dropped 8/92 trades whose edge existed only at the
    flat touch price and vanished once the true fill cost was included. Applied
    here to the production entry path for the first time -- previously
    run_live_paper_trade recorded the touch price (best_ask) as if it were the
    fill, which understates entry cost and overstates edge on every trade."""
    tranches = [
        (depth_1c, best_ask + 0.005),
        (max(depth_2c - depth_1c, 0.0), best_ask + 0.015),
        (max(depth_5c - depth_2c, 0.0), best_ask + 0.035),
    ]
    remaining, total_contracts = stake, 0.0
    for size, price in tranches:
        take = min(remaining, size)
        if take <= 0:
            continue
        total_contracts += take / price
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9 or total_contracts <= 0:
        return None
    return stake / total_contracts   # VWAP


def get_latest_btc_snapshot() -> dict:
    now = pd.Timestamp.now(tz="UTC")
    lo = now - pd.Timedelta(hours=8)  # only need up to the 6h-lookback features, plus buffer
    start_ms, end_ms = int(lo.timestamp() * 1000), int(now.timestamp() * 1000)

    source = "binance"
    try:
        df = btc_c.fetch_klines("BTCUSDT", "5m", start_ms, end_ms)
    except RuntimeError:
        # Binance.com geo-blocks requests from many datacenter/cloud IPs (e.g. GitHub
        # Actions' hosted runners) -- fall back to Coinbase's public candles for the
        # LIVE snapshot only (the historical/training data stays Binance-only). See
        # src/btc_client.py module docstring.
        source = "coinbase_fallback"
        df = btc_c.fetch_coinbase_klines_5m(start_ms, end_ms)

    if df.empty:
        return {"available": False, "source": source}
    df = df.sort_values("timestamp")
    close = df["close"].values
    ts = df["timestamp"].values
    now_px = close[-1]

    def ret_over(hours):
        cutoff = now - pd.Timedelta(hours=hours)
        past = df[df["timestamp"] <= cutoff]
        if past.empty:
            return float("nan")
        return float(now_px / past["close"].iloc[-1] - 1.0)

    log_ret = np.diff(np.log(close[-73:])) if len(close) > 2 else np.array([])
    return {
        "available": True, "source": source, "btc_spot_price": float(now_px),
        "btc_return_15m": ret_over(0.25), "btc_return_1h": ret_over(1), "btc_return_6h": ret_over(6),
        "btc_realized_vol_6h": float(np.std(log_ret, ddof=1)) if len(log_ret) > 1 else float("nan"),
        "btc_spot_volume_1h": float(df[df["timestamp"] > now - pd.Timedelta(hours=1)]["volume"].sum()),
        "as_of": now.isoformat(),
    }


def train_latest_models(family: str = "daily") -> dict:
    """Train on EVERY resolved market available in this family (walk-forward's
    final/expanding window) -- valid because 'now' is after every historical
    resolution used."""
    obs_name = "observations.parquet" if family == "daily" else "hourly_observations.parquet"
    obs_df = pd.read_parquet(PROCESSED / obs_name)
    fitted = {}
    for name in ["raw_probability", "logistic_calibration", "isotonic_calibration", "full_feature_model"]:
        fitted[name] = mdl.MODEL_REGISTRY[name]().fit(obs_df)
    return fitted, obs_df


def run_live_paper_trade(family: str = "daily", risk_cfg: rk.RiskConfig | None = None,
                          ex_cfg: ex.ExecutionConfig | None = None) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    # the hourly market only runs ~1h end to end; the daily default 15-minute
    # "stop opening positions" buffer would silently veto most of its lifetime.
    default_min_hours = 1 / 60 if family == "hourly" else rk.RiskConfig().min_hours_to_resolution
    risk_cfg = risk_cfg or rk.RiskConfig(min_hours_to_resolution=default_min_hours)
    ex_cfg = ex_cfg or ex.ExecutionConfig()
    mtm_path, ptrades_path = mtm_log_path(family), paper_log_path(family)

    market = find_current_active_market(family)
    if market is None:
        result = {"status": "no_active_market", "family": family,
                   "message": f"No qualifying current {family} Bitcoin Up/Down market found."}
        _append_jsonl(mtm_path, {"timestamp": pd.Timestamp.now(tz='UTC').isoformat(), **result})
        return result

    up_token, down_token = market["clob_token_ids"]
    try:
        up_book_raw = pc.get_order_book(up_token)
        down_book_raw = pc.get_order_book(down_token)
        up_book = summarize_book(up_book_raw)
        down_book = summarize_book(down_book_raw)
        book_ok = up_book.mid is not None
    except RuntimeError:
        up_book = down_book = OrderBookSummary(None, None, None, None, 0, 0)
        book_ok = False

    btc_snap = get_latest_btc_snapshot()
    now = pd.Timestamp.now(tz="UTC")
    news_feat = nc.get_news_features(now, use_live_gdelt=True)
    social_feat = sc.get_social_features(now)

    resolution_time = pd.Timestamp(market["end_date"])
    hours_to_resolution = (resolution_time - now).total_seconds() / 3600.0

    raw_prob = up_book.mid if book_ok else float(market["outcome_prices"][0]) if market.get("outcome_prices") else None

    models, obs_df = train_latest_models(family)
    feature_row = pd.DataFrame([{
        "yes_probability": raw_prob if raw_prob is not None else 0.5,
        "horizon_hours": max(hours_to_resolution, 0.0),
        "btc_return_15m": btc_snap.get("btc_return_15m", np.nan),
        "btc_return_1h": btc_snap.get("btc_return_1h", np.nan),
        "btc_return_6h": btc_snap.get("btc_return_6h", np.nan),
        "btc_realized_vol_6h": btc_snap.get("btc_realized_vol_6h", np.nan),
        "btc_spot_volume_1h": btc_snap.get("btc_spot_volume_1h", np.nan),
        "market_volume": market["volume"],
        "market_liquidity": market["liquidity"],
    }])

    model_probs = {}
    for name, model in models.items():
        try:
            model_probs[name] = float(model.predict(feature_row)[0])
        except Exception as exc:  # noqa: BLE001 -- surface as unavailable, don't crash the live trader
            model_probs[name] = None

    calibrated_fair_value = model_probs.get("full_feature_model") or model_probs.get("logistic_calibration") or raw_prob

    # Restore the portfolio and settle anything that has resolved, so the risk
    # limits below are evaluated against real accumulated state.
    portfolio = load_portfolio(family, risk_cfg)
    just_settled = settle_due_positions(family, portfolio, ex_cfg.fee_bps)

    # Only act at the horizons the backtest evaluates, and only once per market.
    horizon = nearest_backtest_horizon(family, hours_to_resolution)
    already_open = market["slug"] in portfolio.open_positions

    # execution: real book if available (actual_book tier), else estimated fallback
    signal = None
    checks = {"has_order_book": book_ok, "market_volume_ok": market["volume"] >= risk_cfg.min_market_volume,
              "hours_to_resolution_ok": hours_to_resolution >= risk_cfg.min_hours_to_resolution,
              "at_backtest_horizon": horizon is not None,
              "position_already_open": already_open,
              "portfolio_can_open": portfolio.can_open_new_position()}

    if book_ok and calibrated_fair_value is not None and horizon is not None \
            and not already_open and portfolio.can_open_new_position():
        for side, model_p, book in (("buy_yes", calibrated_fair_value, up_book),
                                     ("buy_no", 1 - calibrated_fair_value, down_book)):
            if book.best_ask is None:
                continue
            buy_price = book.best_ask
            fee_cost = buy_price * (ex_cfg.fee_bps / 10_000.0)
            gross_edge = model_p - buy_price
            net_edge = gross_edge - fee_cost
            checks[f"{side}_net_edge"] = net_edge
            if net_edge > risk_cfg.min_model_edge and all([checks["market_volume_ok"], checks["hours_to_resolution_ok"]]):
                if signal is None or net_edge > signal["net_expected_edge"]:
                    signal = {
                        "side": side, "model_probability": model_p, "estimated_buy_price": buy_price,
                        "gross_edge": gross_edge, "fees": fee_cost, "net_expected_edge": net_edge,
                        "execution_quality": ex.TIER_ACTUAL_BOOK,
                    }

    recommended_action = "NO_TRADE"
    stake = 0.0
    if signal is not None:
        book = up_book if signal["side"] == "buy_yes" else down_book
        # Cap by what the observed book can absorb within the same walk the
        # backtest allows, rather than leaving size unconstrained by depth.
        stake = portfolio.size_position(signal["model_probability"], signal["estimated_buy_price"],
                                         market["volume"], market["liquidity"],
                                         max_depth_usd=book.buy_depth_5c or None)
        if stake >= 1.0:
            # `signal` above was picked and sized at the flat touch price
            # (best_ask), same as the backtest's signal-detection step. Now walk
            # the REAL recorded depth bands for this stake and re-check the edge
            # at the true fill cost -- a stake that clears the bar at the touch
            # may not once the walk is included. See walk_real_book().
            vwap = walk_real_book(stake, signal["estimated_buy_price"],
                                   book.buy_depth_1c, book.buy_depth_2c, book.buy_depth_5c)
            if vwap is None:
                recommended_action = "NO_TRADE (unfilled at walked depth)"
                checks["walked_vwap"] = None
            else:
                fee_cost = vwap * (ex_cfg.fee_bps / 10_000.0)
                walked_edge = signal["model_probability"] - vwap - fee_cost
                checks["walked_vwap"] = vwap
                checks["walked_net_edge"] = walked_edge
                if walked_edge <= risk_cfg.min_model_edge:
                    recommended_action = "NO_TRADE (edge erased by walk)"
                else:
                    signal["touch_price"] = signal["estimated_buy_price"]
                    signal["estimated_buy_price"] = vwap
                    signal["fees"] = fee_cost
                    signal["net_expected_edge"] = walked_edge
                    portfolio.open_position(
                        market["slug"], signal["side"], stake, vwap,
                        stake / vwap, now.isoformat(),
                        market["end_date"], meta={"horizon_hours": horizon, "entry_edge": walked_edge,
                                                   "touch_price": signal["touch_price"]})
                    recommended_action = signal["side"].upper()
        else:
            recommended_action = "NO_TRADE (size=0)"

    portfolio.mark_equity(now.isoformat())
    save_portfolio(family, portfolio)

    result = {
        "status": "ok", "family": family, "timestamp": now.isoformat(),
        "market": {"slug": market["slug"], "question": market["question"],
                    "resolution_time": market["end_date"], "hours_to_resolution": hours_to_resolution,
                    "volume": market["volume"], "liquidity": market["liquidity"]},
        "order_book": {"up": asdict(up_book), "down": asdict(down_book)},
        "raw_implied_probability": raw_prob,
        "model_probabilities": model_probs,
        "calibrated_fair_value": calibrated_fair_value,
        "btc_snapshot": btc_snap,
        "news_features": news_feat.as_dict(),
        "social_features": social_feat.as_dict(),
        "signal": signal,
        "risk_checks": checks,
        "recommended_action": recommended_action,
        "paper_stake": stake,
        "horizon_hours": horizon,
        "portfolio": {"equity": portfolio.equity, "cash": portfolio.cash,
                       "open_positions": len(portfolio.open_positions),
                       "deployed": portfolio.deployed_capital, "drawdown": portfolio.drawdown},
        "settled_this_run": [{"slug": t["slug"], "won": t["won"], "pnl_net": t["pnl_net"]}
                              for t in just_settled],
    }

    _append_jsonl(mtm_path, {
        "timestamp": now.isoformat(), "family": family, "slug": market["slug"],
        "hours_to_resolution": hours_to_resolution,
        "raw_probability": raw_prob, "calibrated_fair_value": calibrated_fair_value,
        "recommended_action": recommended_action, "paper_stake": stake,
        "market_volume": market["volume"],
        # Real executable depth at this instant. The point of logging it is to
        # find out how large a position these markets can actually absorb at the
        # moments the strategy would trade -- the backtest sized positions off
        # market *volume*, which a spot check suggested may overstate what the
        # book can fill by an order of magnitude. See README "Order-book depth".
        "best_bid": up_book.best_bid, "best_ask": up_book.best_ask,
        "spread": up_book.spread,
        "buy_depth_1c": up_book.buy_depth_1c,
        "buy_depth_2c": up_book.buy_depth_2c,
        "buy_depth_5c": up_book.buy_depth_5c,
        "sell_depth_1c": up_book.sell_depth_1c,
        "no_buy_depth_1c": down_book.buy_depth_1c,
        "no_buy_depth_2c": down_book.buy_depth_2c,
    })
    if recommended_action not in ("NO_TRADE",) and not recommended_action.startswith("NO_TRADE"):
        _append_jsonl(ptrades_path, {"timestamp": now.isoformat(), **{k: v for k, v in result.items()
                                                                          if k not in ("order_book",)}})

    return result


def _append_jsonl(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
