from src import risk as rk


def test_kelly_fraction_zero_when_no_edge():
    assert rk.kelly_fraction(0.5, 0.5) == 0.0
    assert rk.kelly_fraction(0.4, 0.5) == 0.0  # negative edge clipped to 0


def test_kelly_fraction_positive_edge():
    f = rk.kelly_fraction(0.6, 0.5)
    assert 0 < f <= 1.0


def test_kelly_fraction_matches_closed_form():
    p, c = 0.7, 0.4
    expected = (p - c) / (1 - c)
    assert abs(rk.kelly_fraction(p, c) - expected) < 1e-9


def test_position_size_never_exceeds_max_trade_size():
    cfg = rk.RiskConfig(max_trade_size=500.0, initial_capital=1_000_000)
    pf = rk.PortfolioState(cfg)
    size = pf.size_position(model_prob=0.99, buy_price=0.1, market_volume=10_000_000, market_liquidity=0)
    assert size <= 500.0


def test_position_size_respects_participation_limit():
    cfg = rk.RiskConfig(max_participation_rate=0.05, max_trade_size=1e9, initial_capital=1e9)
    pf = rk.PortfolioState(cfg)
    size = pf.size_position(model_prob=0.9, buy_price=0.1, market_volume=1000, market_liquidity=0)
    assert size <= 0.05 * 1000 + 1e-6


def test_position_size_respects_per_market_risk_cap():
    cfg = rk.RiskConfig(max_risk_per_market=0.01, initial_capital=100_000, max_trade_size=1e9,
                         max_participation_rate=1.0)
    pf = rk.PortfolioState(cfg)
    size = pf.size_position(model_prob=0.95, buy_price=0.05, market_volume=1e9, market_liquidity=1e9)
    assert size <= 0.01 * 100_000 + 1e-6


def test_cannot_open_new_position_beyond_max_simultaneous_positions():
    cfg = rk.RiskConfig(max_simultaneous_positions=2)
    pf = rk.PortfolioState(cfg)
    pf.open_position("m1", "buy_yes", 100, 0.5, 200, "t0", "t1", meta={})
    pf.open_position("m2", "buy_yes", 100, 0.5, 200, "t0", "t1", meta={})
    assert not pf.can_open_new_position()


def test_max_drawdown_stop_blocks_new_positions():
    cfg = rk.RiskConfig(max_drawdown_stop=0.20, initial_capital=100_000)
    pf = rk.PortfolioState(cfg)
    pf.equity = 100_000
    pf.peak_equity = 100_000
    pf.equity = 75_000  # 25% drawdown > 20% stop
    assert not pf.can_open_new_position()


def test_max_total_capital_deployed_blocks_new_positions():
    cfg = rk.RiskConfig(max_total_capital_deployed=0.10, initial_capital=10_000)
    pf = rk.PortfolioState(cfg)
    pf.open_position("m1", "buy_yes", 1500, 0.5, 3000, "t0", "t1", meta={})  # 15% > 10% cap
    assert not pf.can_open_new_position()
