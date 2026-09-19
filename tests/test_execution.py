import numpy as np

from src import execution as ex


def test_spread_widens_as_liquidity_shrinks():
    cfg = ex.ExecutionConfig()
    tight = ex.half_spread_bps(1_000_000, cfg, "base")
    wide = ex.half_spread_bps(100, cfg, "base")
    assert wide > tight


def test_spread_is_clipped_to_configured_bounds():
    cfg = ex.ExecutionConfig()
    assert ex.half_spread_bps(1e12, cfg, "base") >= cfg.min_half_spread_bps
    assert ex.half_spread_bps(0, cfg, "base") <= cfg.max_half_spread_bps


def test_scenario_ordering_optimistic_lt_base_lt_conservative():
    cfg = ex.ExecutionConfig()
    liq, vol = 50_000, 200_000
    # size chosen to be fillable in every scenario, so this compares prices rather
    # than fill/no-fill (the unfilled ordering is covered separately below)
    size = 200
    opt = ex.quote_estimated_execution("buy_yes", 0.5, size, vol, liq, cfg, "optimistic")
    base = ex.quote_estimated_execution("buy_yes", 0.5, size, vol, liq, cfg, "base")
    cons = ex.quote_estimated_execution("buy_yes", 0.5, size, vol, liq, cfg, "conservative")
    assert not (opt.unfilled or base.unfilled or cons.unfilled)
    assert opt.estimated_buy_price <= base.estimated_buy_price <= cons.estimated_buy_price


def test_thinner_scenario_never_looks_cheaper_even_when_unfilled():
    """An unfillable order must not be priced better than a fillable one -- a caller
    that ignored the `unfilled` flag would otherwise prefer the thinner book."""
    cfg = ex.ExecutionConfig()
    size = 1200   # fills only under the optimistic (deepest) book
    opt = ex.quote_estimated_execution("buy_yes", 0.5, size, 200_000, 50_000, cfg, "optimistic")
    cons = ex.quote_estimated_execution("buy_yes", 0.5, size, 200_000, 50_000, cfg, "conservative")
    assert not opt.unfilled and cons.unfilled
    assert cons.estimated_buy_price >= opt.estimated_buy_price


def test_order_beyond_max_walk_is_unfilled():
    cfg = ex.ExecutionConfig()
    assert ex.walk_book(ex.max_fillable_usd(cfg, "base") + 1, 0.50, cfg, "base") is None
    assert ex.walk_book(ex.max_fillable_usd(cfg, "base") - 1, 0.50, cfg, "base") is not None


def test_small_order_fills_at_the_touch_without_walking():
    cfg = ex.ExecutionConfig()
    px, cents = ex.walk_book(cfg.touch_depth_usd * 0.5, 0.505, cfg, "base")
    assert cents == 0 and abs(px - 0.505) < 1e-9


def test_larger_orders_pay_a_worse_average_price():
    cfg = ex.ExecutionConfig()
    small = ex.walk_book(100, 0.505, cfg, "base")[0]
    big = ex.walk_book(700, 0.505, cfg, "base")[0]
    assert big > small


def test_absolute_spread_floor_dominates_in_the_tails():
    """Real books quote ~1c regardless of level, so a 5c contract must not be
    charged a proportionally tiny spread."""
    cfg = ex.ExecutionConfig()
    q = ex.quote_estimated_execution("buy_yes", 0.05, 50, 500_000, 0, cfg, "base")
    assert q.estimated_buy_price - q.mid_price >= cfg.min_half_spread_abs - 1e-9


def test_buy_no_leg_uses_complement_of_yes_mid():
    cfg = ex.ExecutionConfig()
    q = ex.quote_estimated_execution("buy_no", 0.7, 1000, 200_000, 50_000, cfg, "base")
    assert abs(q.mid_price - 0.3) < 1e-9


def test_participation_cap_flags_oversized_trades():
    cfg = ex.ExecutionConfig(max_participation_rate=0.10)
    q_small = ex.quote_estimated_execution("buy_yes", 0.5, 100, 100_000, 50_000, cfg, "base")
    q_large = ex.quote_estimated_execution("buy_yes", 0.5, 50_000, 100_000, 50_000, cfg, "base")
    assert not q_small.participation_capped
    assert q_large.participation_capped


def test_zero_volume_falls_back_to_maximally_conservative_slippage():
    cfg = ex.ExecutionConfig()
    s = ex.slippage_bps(1000, 0.0, cfg, "base")
    assert s == cfg.max_half_spread_bps


def test_execution_prices_stay_within_valid_probability_range():
    cfg = ex.ExecutionConfig()
    q = ex.quote_estimated_execution("buy_yes", 0.99, 1000, 1, 1, cfg, "conservative")
    assert 0.0 < q.estimated_buy_price <= 1.0
    assert 0.0 <= q.estimated_sell_price < 1.0


def test_trade_proxy_fill_uses_conservative_quantiles():
    trades = [{"price": p} for p in [0.40, 0.45, 0.50, 0.55, 0.60]]
    buy_fill = ex.trade_proxy_fill("buy_yes", trades, quantile=0.75)
    sell_fill = ex.trade_proxy_fill("sell_yes", trades, quantile=0.75)
    assert buy_fill >= 0.55   # BUY uses a conservative HIGH quantile
    assert sell_fill <= 0.45  # SELL uses a conservative LOW quantile


def test_trade_proxy_fill_returns_none_when_no_eligible_trades():
    assert ex.trade_proxy_fill("buy_yes", [], quantile=0.75) is None
