import numpy as np
import pandas as pd
import pytest

from src import news_client as nc
from src import paper_trader as pt
from src import polymarket_client as pc
from src import risk as rk
from src import social_client as sc


def test_log_paths_differ_by_family_and_stay_backward_compatible():
    # daily keeps the original (pre-multi-family) filenames so existing history isn't orphaned
    assert pt.mtm_log_path("daily").name == "mark_to_market.jsonl"
    assert pt.paper_log_path("daily").name == "paper_trades.jsonl"
    assert pt.mtm_log_path("hourly").name == "mark_to_market_hourly.jsonl"
    assert pt.paper_log_path("hourly").name == "paper_trades_hourly.jsonl"


def test_run_live_paper_trade_rejects_unknown_family():
    with pytest.raises(ValueError):
        pt.run_live_paper_trade(family="weekly")


# Depth figures below are taken from a real Polymarket order book screenshot
# (best bid 30c / best ask 31c, "Spread: 1c" per Polymarket's own UI).
REAL_BOOK = {
    "asks": [{"price": "0.31", "size": "541.27"}, {"price": "0.32", "size": "605.95"},
             {"price": "0.33", "size": "151.24"}, {"price": "0.34", "size": "176.71"}],
    "bids": [{"price": "0.30", "size": "86.46"}, {"price": "0.29", "size": "143.17"},
             {"price": "0.28", "size": "143.17"}, {"price": "0.27", "size": "76.77"}],
}


def test_summarize_book_touch_and_spread():
    b = pt.summarize_book(REAL_BOOK)
    assert b.best_bid == 0.30
    assert b.best_ask == 0.31
    assert abs(b.spread - 0.01) < 1e-9      # 1 cent, matching Polymarket's UI
    assert abs(b.mid - 0.305) < 1e-9


def test_buy_depth_counts_only_levels_within_the_cent_band():
    b = pt.summarize_book(REAL_BOOK)
    # within 1c of the 0.31 touch -> only the 0.31 and 0.32 levels qualify
    assert abs(b.buy_depth_1c - (0.31 * 541.27 + 0.32 * 605.95)) < 1e-6
    # within 2c additionally admits 0.33
    assert abs(b.buy_depth_2c - (0.31 * 541.27 + 0.32 * 605.95 + 0.33 * 151.24)) < 1e-6
    # depth is monotonically non-decreasing as the band widens
    assert b.buy_depth_1c <= b.buy_depth_2c <= b.buy_depth_5c


def test_sell_depth_walks_bids_downward_not_upward():
    b = pt.summarize_book(REAL_BOOK)
    # selling into bids at 0.30 and 0.29 is within 1c of the 0.30 touch
    assert abs(b.sell_depth_1c - (0.30 * 86.46 + 0.29 * 143.17)) < 1e-6
    assert b.sell_depth_1c <= b.sell_depth_2c <= b.sell_depth_5c


def test_depth_is_notional_usd_not_share_count():
    b = pt.summarize_book(REAL_BOOK)
    shares_1c = 541.27 + 605.95
    # notional must be strictly less than the share count here, since every
    # price is well below $1 -- guards against regressing to a raw size sum
    assert b.buy_depth_1c < shares_1c
    assert b.n_ask_levels == 4 and b.n_bid_levels == 4


def test_empty_book_yields_zero_depth_not_a_crash():
    b = pt.summarize_book({"asks": [], "bids": []})
    assert b.best_ask is None and b.best_bid is None
    assert b.buy_depth_1c == 0.0 and b.sell_depth_1c == 0.0


# --- alignment with the backtest -------------------------------------------
# The live trader wakes every 10 minutes while the backtest evaluates each market
# at five fixed horizons and opens at most one position per market. Without the
# checks below the two run different strategies and their results are not
# comparable -- which is exactly what happened: live daily entries landed at
# 23.3-23.8h, points the backtest never evaluates.

def test_only_trades_at_backtest_horizons():
    for h in [24.0, 12.0, 6.0, 3.0, 1.0]:
        assert pt.nearest_backtest_horizon("daily", h) == h
    # between horizons the backtest would not have looked, so neither should live
    assert pt.nearest_backtest_horizon("daily", 23.5) is None
    assert pt.nearest_backtest_horizon("daily", 18.0) is None
    assert pt.nearest_backtest_horizon("daily", 2.0) is None


def test_horizon_tolerance_admits_a_ten_minute_cadence():
    """A 10-minute wake-up cannot land exactly on a horizon, so a window is needed
    -- but it must be narrow enough that only one check-in per horizon qualifies."""
    assert pt.nearest_backtest_horizon("daily", 24.0 - 4 / 60) == 24.0
    assert pt.nearest_backtest_horizon("daily", 24.0 + 4 / 60) == 24.0
    assert pt.nearest_backtest_horizon("daily", 24.0 - 9 / 60) is None
    assert 2 * pt.HORIZON_TOLERANCE_HOURS <= 10 / 60 + 1e-9


def test_hourly_family_uses_its_own_minute_scale_grid():
    assert pt.nearest_backtest_horizon("hourly", 45 / 60) == 45 / 60
    assert pt.nearest_backtest_horizon("hourly", 2 / 60) == 2 / 60
    # a daily-scale horizon is meaningless for a market that only lasts an hour
    assert pt.nearest_backtest_horizon("hourly", 24.0) is None


def test_portfolio_state_round_trips(tmp_path, monkeypatch):
    """Risk limits can only bind if state survives between check-ins."""
    monkeypatch.setattr(pt, "PAPER_DIR", tmp_path)
    cfg = rk.RiskConfig()
    pf = pt.load_portfolio("daily", cfg)
    assert pf.open_positions == {} and pf.equity == cfg.initial_capital

    pf.open_position("mkt-1", "buy_yes", 500.0, 0.5, 1000.0, "t0", "t1", meta={})
    pt.save_portfolio("daily", pf)

    restored = pt.load_portfolio("daily", cfg)
    assert "mkt-1" in restored.open_positions
    assert restored.cash == pf.cash
    assert restored.deployed_capital == 500.0


def test_restored_portfolio_enforces_position_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "PAPER_DIR", tmp_path)
    cfg = rk.RiskConfig(max_simultaneous_positions=2)
    pf = pt.load_portfolio("daily", cfg)
    pf.open_position("a", "buy_yes", 10.0, 0.5, 20.0, "t0", "t1", meta={})
    pf.open_position("b", "buy_yes", 10.0, 0.5, 20.0, "t0", "t1", meta={})
    pt.save_portfolio("daily", pf)
    assert pt.load_portfolio("daily", cfg).can_open_new_position() is False


# --- walk_real_book: VWAP entry pricing against real recorded depth bands ---

def test_walk_real_book_fills_within_first_band():
    # $10 stake, all of it inside the 1c band -> priced at the band-1 midpoint
    vwap = pt.walk_real_book(10.0, best_ask=0.40, depth_1c=50.0, depth_2c=50.0, depth_5c=50.0)
    assert vwap == pytest.approx(0.405)   # best_ask + 0.005


def test_walk_real_book_spans_multiple_bands_worse_than_touch():
    # depth_1c=20 forces the remaining $30 of a $50 stake into the pricier 2c band
    vwap = pt.walk_real_book(50.0, best_ask=0.40, depth_1c=20.0, depth_2c=60.0, depth_5c=60.0)
    assert vwap is not None
    assert vwap > 0.405   # worse than a full fill at the band-1 price alone
    assert vwap < 0.415   # but better than the band-2 midpoint alone (0.415)


def test_walk_real_book_returns_none_when_stake_exceeds_recorded_depth():
    # only $30 total sits within 5c of the touch -- a $100 stake cannot fill
    assert pt.walk_real_book(100.0, best_ask=0.40, depth_1c=10.0, depth_2c=20.0, depth_5c=30.0) is None


# --- production entry path: must price fills at the walked VWAP, not best_ask ---

class _NullFeat:
    def as_dict(self):
        return {}


class _FixedModel:
    def __init__(self, prob):
        self.prob = prob

    def predict(self, df):
        return np.array([self.prob])


def _stub_common(monkeypatch, tmp_path, market, up_book_raw, down_book_raw, model_prob):
    monkeypatch.setattr(pt, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(pt, "find_current_active_market", lambda family: market)
    monkeypatch.setattr(pc, "get_order_book",
                         lambda token: up_book_raw if token == "UP" else down_book_raw)
    monkeypatch.setattr(pt, "get_latest_btc_snapshot", lambda: {"available": False, "source": "test"})
    monkeypatch.setattr(nc, "get_news_features", lambda *a, **k: _NullFeat())
    monkeypatch.setattr(sc, "get_social_features", lambda *a, **k: _NullFeat())
    monkeypatch.setattr(pt, "train_latest_models",
                         lambda family: ({"full_feature_model": _FixedModel(model_prob)}, None))


def _make_market(minutes_to_resolution):
    return {
        "slug": "test-market", "question": "Test?", "clob_token_ids": ["UP", "DOWN"],
        "start_date": "2026-01-01T00:00:00+00:00",
        "end_date": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=minutes_to_resolution)).isoformat(),
        "volume": 50_000.0, "liquidity": 50_000.0, "outcome_prices": [0.5, 0.5],
    }


def test_opened_position_is_priced_at_walked_vwap_not_touch(tmp_path, monkeypatch):
    """Regression: run_live_paper_trade used to record best_ask as if it were
    the fill price. A real book that thins out past the touch must produce an
    opened position priced worse than best_ask, matching walk_real_book()."""
    market = _make_market(30)   # -> the hourly 0.5h horizon
    up_book_raw = {
        "asks": [{"price": "0.10", "size": "200"}, {"price": "0.11", "size": "200"},
                 {"price": "0.12", "size": "200"}],
        "bids": [{"price": "0.09", "size": "200"}],
    }
    down_book_raw = {"asks": [{"price": "0.95", "size": "1"}], "bids": [{"price": "0.05", "size": "1"}]}
    _stub_common(monkeypatch, tmp_path, market, up_book_raw, down_book_raw, model_prob=0.55)

    result = pt.run_live_paper_trade("hourly")

    assert result["recommended_action"] == "BUY_YES"
    assert result["signal"]["side"] == "buy_yes"
    touch_price = 0.10
    assert result["signal"]["touch_price"] == pytest.approx(touch_price)
    assert result["signal"]["estimated_buy_price"] > touch_price   # walked, not flat best_ask

    pf = pt.load_portfolio("hourly", rk.RiskConfig())
    pos = pf.open_positions["test-market"]
    assert pos["buy_price"] == pytest.approx(result["signal"]["estimated_buy_price"])
    assert pos["buy_price"] > touch_price
    # contracts must be consistent with stake / walked price, not stake / touch price
    assert pos["n_contracts"] == pytest.approx(pos["stake"] / pos["buy_price"])


def test_edge_that_only_exists_at_touch_price_is_not_traded(tmp_path, monkeypatch):
    """A thin book can show an inviting edge at the flat best_ask that evaporates
    once the real depth is walked -- that must produce NO_TRADE, not a position
    opened at a price the book could never actually have filled."""
    market = _make_market(30)
    up_book_raw = {
        # tiny size at the touch, then a big block only reachable near the 5c
        # boundary -- walking a stake capped by the outer band pushes the VWAP
        # well past the model's edge.
        "asks": [{"price": "0.10", "size": "50"}, {"price": "0.14", "size": "714300"}],
        "bids": [{"price": "0.09", "size": "50"}],
    }
    down_book_raw = {"asks": [{"price": "0.95", "size": "1"}], "bids": [{"price": "0.05", "size": "1"}]}
    _stub_common(monkeypatch, tmp_path, market, up_book_raw, down_book_raw, model_prob=0.145)

    result = pt.run_live_paper_trade("hourly")

    assert result["recommended_action"] == "NO_TRADE (edge erased by walk)"
    assert result["risk_checks"]["walked_net_edge"] <= rk.RiskConfig().min_model_edge
    pf = pt.load_portfolio("hourly", rk.RiskConfig())
    assert pf.open_positions == {}
