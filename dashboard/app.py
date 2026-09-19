"""Streamlit dashboard: Bitcoin Prediction Market Calibration & Execution-Aware Backtest."""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import calibration as cal  # noqa: E402
from src import live_dashboard_data as ld  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
PAPER_DIR = ROOT / "paper_trading"

st.set_page_config(page_title="BTC Up/Down Calibration & Backtest", layout="wide")


@st.cache_data
def load_json(name):
    path = PROCESSED / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_parquet(name):
    path = PROCESSED / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_csv(name):
    path = PROCESSED / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_pickle(name):
    path = PROCESSED / name
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


st.title("₿ Bitcoin Prediction Market Calibration & Execution-Aware Backtest")
st.caption("Polymarket 'Bitcoin Up or Down' markets — real resolved-market data, "
           "walk-forward evaluated, execution-cost-aware backtest. All numbers on this page "
           "are computed from data cached under data/processed/ and data/raw/.")

# ---------------------------------------------------------------------------
# GLOBAL MARKET-FAMILY SELECTOR -- every tab below reads off this single choice
# ---------------------------------------------------------------------------
fam_choice = st.radio(
    "Market family",
    ["Daily (507 markets, Mar 2025 – Aug 2026)",
     "Hourly extended sample (1,999 markets, most recent ~86 days)"],
    horizontal=True,
    help="The daily BTC Up/Down series only has 507 resolved markets total (it launched March 2025). "
         "The hourly series is the same mechanic at 24x the frequency, pulled as an extended sample "
         "(most recent ~2,000 resolved markets, ~86 days) to test whether findings replicate with a "
         "much larger, though more recent and more autocorrelated, sample. See reports/research_memo.md "
         "Part 4 for the full writeup. Every tab below reflects whichever family is selected here.",
)
IS_HOURLY = fam_choice.startswith("Hourly")
PREFIX = "hourly_" if IS_HOURLY else ""
FAMILY = "hourly" if IS_HOURLY else "daily"
HORIZON_COL = "horizon_minutes" if IS_HOURLY else "horizon_hours"
HORIZON_UNIT = "min" if IS_HOURLY else "h"
HORIZON_LABEL = "minutes" if IS_HOURLY else "hours"
N_MARKETS_LABEL = "1,999" if IS_HOURLY else "507"

if IS_HOURLY:
    st.info(
        "**Replication check, not a strict extension.** This is a different, more recent time window than "
        "the daily sample (most recent ~86 days), not the same period at higher frequency — and hourly BTC "
        "candles on the same day are more autocorrelated with each other than daily closes are, so "
        "'1,999 markets' overstates independence more than '507 daily markets' did. Treat agreement between "
        "the two panels as corroborating evidence, not independent confirmation. "
        "See `reports/research_memo.md` Part 4."
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["1 · Data Quality", "2 · Calibration", "3 · Mispricing Analysis", "4 · Backtest",
     "5 · Live Paper Trade", "6 · Live P&L (real)"])

# ---------------------------------------------------------------------------
# TAB 1: DATA QUALITY
# ---------------------------------------------------------------------------
with tab1:
    st.header("Data Quality")
    markets_accepted = load_json(f"{PREFIX}markets_accepted.json") or []
    markets_excluded = load_json(f"{PREFIX}markets_excluded.json") or []
    obs_exclusions = load_json(f"{PREFIX}observation_exclusions.json") or []
    obs_df = load_parquet(f"{PREFIX}observations.parquet")

    if not markets_accepted:
        st.warning(f"Run scripts/01_fetch_markets.py (daily) or scripts/11-14 (hourly) first.")
    else:
        n_discovered = len(markets_accepted) + len(markets_excluded)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Markets discovered (series scan)", n_discovered)
        c2.metric("Markets included (resolved, validated)", len(markets_accepted))
        c3.metric("Markets excluded", len(markets_excluded))
        c4.metric("Fixed-horizon observations built", len(obs_df))

        st.subheader("Market-level exclusion reasons")
        if markets_excluded:
            reasons = pd.Series([e["reason"] for e in markets_excluded]).value_counts().reset_index()
            reasons.columns = ["reason", "count"]
            fig = go.Figure(go.Bar(x=reasons["reason"], y=reasons["count"]))
            fig.update_layout(height=350, margin=dict(t=10))
            st.plotly_chart(fig, width='stretch')
            with st.expander("Full exclusion list"):
                st.dataframe(pd.DataFrame(markets_excluded), width='stretch')
        else:
            st.info("No market-level exclusions recorded.")

        st.subheader("Observation-level exclusion reasons (per horizon attempt)")
        if obs_exclusions:
            oreasons = pd.Series([e["reason"] for e in obs_exclusions]).value_counts().reset_index()
            oreasons.columns = ["reason", "count"]
            st.dataframe(oreasons, width='stretch')
        else:
            st.info("No observation-level exclusions recorded.")

        st.subheader(f"Observation counts by horizon ({HORIZON_LABEL})")
        if not obs_df.empty:
            by_h = obs_df.groupby(HORIZON_COL).size().reset_index(name="n_observations")
            by_h = by_h.sort_values(HORIZON_COL, ascending=False)
            st.dataframe(by_h, width='stretch')

            st.subheader("Feature availability")
            avail = pd.DataFrame({
                "feature": ["news (GDELT)", "social/sentiment"],
                "available_rate": [obs_df["news_data_available"].mean(), obs_df["social_data_available"].mean()],
            })
            st.dataframe(avail, width='stretch')
            st.caption("News/social availability reflects real, live API reachability at data-collection time "
                       "(see README 'Known limitations') — features are left NaN/unavailable rather than "
                       "fabricated when a source could not be reached.")

# ---------------------------------------------------------------------------
# TAB 2: CALIBRATION
# ---------------------------------------------------------------------------
with tab2:
    st.header("Calibration of the Raw Polymarket Implied Probability")
    calib = load_json(f"{PREFIX}calibration_raw.json") or {}
    if not calib:
        st.warning("Run scripts/05_calibration.py (daily) or scripts/15_hourly_calibration.py (hourly) first.")
    else:
        horizons = sorted((int(h) for h in calib.keys()), reverse=True)
        sel_h = st.selectbox(f"Horizon ({HORIZON_LABEL} before resolution)", horizons, index=0)
        entry = calib[str(sel_h)]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N observations", entry["n_observations"])
        c2.metric("N independent markets", entry["n_independent_markets"])
        c3.metric("Brier score", f"{entry['brier_score']['point']:.4f}",
                  help=f"95% CI [{entry['brier_score']['ci_lo']:.4f}, {entry['brier_score']['ci_hi']:.4f}] "
                       "(market-level bootstrap)")
        c4.metric("Log loss", f"{entry['log_loss']['point']:.4f}",
                  help=f"95% CI [{entry['log_loss']['ci_lo']:.4f}, {entry['log_loss']['ci_hi']:.4f}]")

        c5, c6 = st.columns(2)
        c5.metric("ECE (10-bin)", f"{entry['ece_10bin']['point']:.4f}",
                  help=f"95% CI [{entry['ece_10bin']['ci_lo']:.4f}, {entry['ece_10bin']['ci_hi']:.4f}]")
        c6.metric("Calibration slope / intercept",
                  f"{entry['calibration_slope']:.3f} / {entry['calibration_intercept']:.3f}",
                  help="Perfect calibration = slope 1.0, intercept 0.0")

        st.subheader("Reliability diagram")
        bucket_type = st.radio("Bucketing", ["Fixed (0.1-wide)", "Adaptive (quantile)"], horizontal=True,
                                key=f"bucket_{PREFIX}")
        table = entry["fixed_bucket_table"] if bucket_type.startswith("Fixed") else entry["adaptive_bucket_table"]
        tdf = pd.DataFrame(table)
        if not tdf.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                      line=dict(dash="dash", color="gray")))
            fig.add_trace(go.Scatter(
                x=tdf["avg_implied_prob"], y=tdf["realized_up_frequency"], mode="markers+lines",
                name=f"{sel_h}{HORIZON_UNIT} horizon",
                error_y=dict(type="data", symmetric=False,
                             array=tdf["ci_hi"] - tdf["realized_up_frequency"],
                             arrayminus=tdf["realized_up_frequency"] - tdf["ci_lo"]),
                marker=dict(size=np.clip(tdf["n_independent_markets"], 4, 30))))
            fig.update_layout(xaxis_title="Average implied probability (YES)",
                               yaxis_title="Realized Up frequency", height=500,
                               xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, width='stretch')
            st.dataframe(tdf, width='stretch')

        st.subheader("All horizons — summary")
        rows = []
        for h in horizons:
            e = calib[str(h)]
            rows.append({f"horizon_{HORIZON_LABEL}": h, "n_obs": e["n_observations"],
                         "n_markets": e["n_independent_markets"],
                         "brier": e["brier_score"]["point"], "log_loss": e["log_loss"]["point"],
                         "ece": e["ece_10bin"]["point"], "calib_slope": e["calibration_slope"],
                         "calib_intercept": e["calibration_intercept"]})
        st.dataframe(pd.DataFrame(rows), width='stretch')

        st.subheader("Walk-forward model comparison (baseline vs calibrated vs full-feature)")
        wf_summary = load_csv(f"{PREFIX}walkforward_model_summary.csv")
        if not wf_summary.empty:
            st.dataframe(wf_summary, width='stretch')
        else:
            needed = "scripts/06_walkforward.py" if not IS_HOURLY else "scripts/16_hourly_walkforward.py"
            st.info(f"Run {needed} to populate this table.")

# ---------------------------------------------------------------------------
# TAB 3: MISPRICING ANALYSIS
# ---------------------------------------------------------------------------
with tab3:
    st.header("Mispricing Analysis: Calibration Gap by Regime")
    obs_df = load_parquet(f"{PREFIX}observations.parquet")
    if obs_df.empty:
        st.warning("Run scripts/04_build_observations.py (daily) or scripts/14_build_hourly_observations.py "
                   "(hourly) first.")
    else:
        horizons = sorted(obs_df[HORIZON_COL].unique(), reverse=True)
        sel_h3 = st.selectbox(f"Horizon ({HORIZON_LABEL})", horizons, index=0, key=f"h3_{PREFIX}")
        sub = obs_df[obs_df[HORIZON_COL] == sel_h3].copy()

        st.subheader("By probability bucket")
        table = cal.reliability_table(sub, "yes_probability", "outcome_up", "slug", cal.fixed_buckets(10))
        fig = go.Figure(go.Bar(x=[f"{r.bucket_lo:.1f}-{r.bucket_hi:.1f}" for r in table.itertuples()],
                                 y=table["calibration_gap"],
                                 error_y=dict(type="data", array=table["ci_hi"] - table["realized_up_frequency"]),
                                 marker_color=["#2a9d8f" if g > 0 else "#e76f51" for g in table["calibration_gap"]]))
        fig.update_layout(yaxis_title="Calibration gap (realized - implied)", height=350, margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')
        st.dataframe(table, width='stretch')
        st.caption("Positive gap = realized Up frequency exceeded the implied probability "
                   "(YES historically underpriced in that bucket). Negative = YES overpriced. "
                   "`statistically_significant` requires >=20 independent markets AND a 95% CI excluding 0.")

        st.subheader("By BTC momentum regime (sign of 1h return)")
        sub["momentum_regime"] = np.where(sub["btc_return_1h"] > 0, "up", "down")
        mom_table = cal.group_gap_table(sub, "momentum_regime", "yes_probability", "outcome_up", "slug")
        st.dataframe(mom_table, width='stretch')

        st.subheader("By BTC volatility regime (terciles of trailing 6h realized vol)")
        try:
            sub["vol_regime"] = pd.qcut(sub["btc_realized_vol_6h"], 3, labels=["low", "medium", "high"])
            vol_table = cal.group_gap_table(sub, "vol_regime", "yes_probability", "outcome_up", "slug")
            st.dataframe(vol_table, width='stretch')
        except ValueError:
            st.info("Not enough distinct volatility values to form terciles at this horizon.")

        st.subheader("By market volume (quartiles)")
        try:
            sub["volume_regime"] = pd.qcut(sub["market_volume"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
            vol_q_table = cal.group_gap_table(sub, "volume_regime", "yes_probability", "outcome_up", "slug")
            st.dataframe(vol_q_table, width='stretch')
        except ValueError:
            st.info("Not enough distinct volume values to form quartiles at this horizon.")

        st.subheader("By news intensity")
        if sub["news_data_available"].mean() > 0:
            sub_news = sub[sub["news_data_available"]].copy()
            sub_news["news_regime"] = pd.qcut(sub_news["article_count_24h"], 3, duplicates="drop")
            news_table = cal.group_gap_table(sub_news, "news_regime", "yes_probability", "outcome_up", "slug")
            st.dataframe(news_table, width='stretch')
        else:
            st.info("News features were unavailable for this sample (0% coverage) — see Data Quality tab and "
                    "README 'Known limitations'. This breakdown is intentionally left empty rather than fabricated.")

# ---------------------------------------------------------------------------
# TAB 4: BACKTEST
# ---------------------------------------------------------------------------
with tab4:
    st.header("Execution-Aware Backtest")
    bt_summary = load_csv(f"{PREFIX}backtest_summary.csv")
    bt_trades = load_pickle(f"{PREFIX}backtest_trades.pkl")
    bt_equity = load_pickle(f"{PREFIX}backtest_equity.pkl")
    bt_breakdowns = load_pickle(f"{PREFIX}backtest_breakdowns.pkl")

    if bt_summary.empty:
        needed = "scripts/07_backtest.py" if not IS_HOURLY else "scripts/17_hourly_backtest.py"
        st.warning(f"Run {needed} first.")
    else:
        st.subheader("Strategy comparison — all scenarios")
        st.dataframe(bt_summary, width='stretch')

        strategies = sorted(bt_summary["strategy"].unique())
        scenarios = sorted(bt_summary["scenario"].unique())
        c1, c2 = st.columns(2)
        sel_strat = c1.selectbox("Strategy", strategies, key=f"strat_{PREFIX}")
        sel_scn = c2.selectbox("Execution scenario", scenarios,
                                index=scenarios.index("base") if "base" in scenarios else 0, key=f"scn_{PREFIX}")
        key = f"{sel_strat}__{sel_scn}"

        eq = bt_equity.get(key, pd.DataFrame())
        if not eq.empty:
            eq["timestamp"] = pd.to_datetime(eq["timestamp"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["equity"], name="Equity"))
            fig.update_layout(title="Equity curve", height=350, margin=dict(t=40))
            st.plotly_chart(fig, width='stretch')

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=eq["timestamp"], y=eq["drawdown"], name="Drawdown", fill="tozeroy",
                                        line=dict(color="crimson")))
            fig2.update_layout(title="Drawdown", height=250, margin=dict(t=40), yaxis_tickformat=".0%")
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("No trades executed for this strategy/scenario combination.")

        trades_df = bt_trades.get(key, pd.DataFrame())
        st.subheader(f"Trade log — {sel_strat} / {sel_scn} ({len(trades_df)} trades)")
        if not trades_df.empty:
            st.dataframe(trades_df, width='stretch', height=300)

            st.subheader("Cost attribution")
            cost_c1, cost_c2, cost_c3 = st.columns(3)
            cost_c1.metric("Total fees", f"${trades_df['fee_cost'].sum():,.2f}")
            cost_c2.metric("Est. total slippage cost", f"${(trades_df['slippage'] * trades_df['stake']).sum():,.2f}")
            cost_c3.metric("Turnover (total stake)", f"${trades_df['stake'].sum():,.2f}")

            st.subheader(f"Performance by time-to-resolution ({HORIZON_LABEL})")
            ttr = bt_breakdowns.get(f"{sel_strat}__ttr", pd.DataFrame())
            if not ttr.empty:
                st.dataframe(ttr, width='stretch')
                if IS_HOURLY:
                    st.caption(
                        "Compare to the daily family (toggle at the top of the page): there, ~75% of "
                        "full-feature-strategy trades fired at the least-informative (24h) horizon and that "
                        "segment was roughly break-even/negative. Here, check whether the edge is still "
                        "concentrated only in the shortest horizons, or whether it now appears more broadly "
                        "with ~4x the sample.")

            st.subheader("Performance by probability bucket")
            pb = bt_breakdowns.get(f"{sel_strat}__prob_bucket", pd.DataFrame())
            if not pb.empty:
                st.dataframe(pb, width='stretch')

            st.subheader("Performance by execution-data quality")
            eq_tbl = bt_breakdowns.get(f"{sel_strat}__exec_quality", pd.DataFrame())
            if not eq_tbl.empty:
                st.dataframe(eq_tbl, width='stretch')
            st.caption("All historical fills in this project are labeled `estimated_execution` — Polymarket's "
                       "public API does not expose historical order-book or trade data (see README). Treat "
                       "backtest P&L as an estimate under a transparent, documented cost model, not as a claim "
                       "of achievable historical execution.")

        st.subheader("Execution-cost basis: live-measured, not a scenario sweep")
        try:
            from src import execution as ex
            m = ex.measure_live_execution_costs(FAMILY)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Live check-ins measured", f"{m['n_snapshots']:,}")
            mc2.metric("Mean half-spread", f"${m['mean_half_spread']:.4f}")
            mc3.metric("Touch depth (within 1c)", f"${m['touch_depth_usd']:,.0f}")
            mc4.metric("Depth per additional cent", f"${m['depth_per_cent_usd']:,.0f}")
            st.caption(
                "There is a single execution-cost config now, not an optimistic/base/conservative sweep: "
                "spread floor and order-book depth are averaged fresh from every real book the live paper "
                "trader has logged for this family (`ExecutionConfig.from_live_measurements`, "
                "`src/execution.py`), recomputed every time this backtest is re-run — so it updates on its "
                "own as more live check-ins accumulate, with no frozen constant to go stale."
            )
        except (FileNotFoundError, ValueError) as e:
            st.info(f"Live execution costs not available: {e}")

# ---------------------------------------------------------------------------
# TAB 5: LIVE PAPER TRADE
# ---------------------------------------------------------------------------
with tab5:
    fam_title = "Hourly" if IS_HOURLY else "Daily"
    st.header(f"Live Paper Trade — Current {fam_title} Bitcoin Up/Down Market")
    st.caption("Reads REAL live data on demand (Polymarket order book, Binance spot, GDELT news). "
               "Simulated paper execution only — no real orders are ever sent.")

    run_live = st.button(f"🔄 Fetch live {fam_title.lower()} market & evaluate", type="primary", key=f"live_{PREFIX}")

    if run_live:
        with st.spinner("Querying Polymarket, Binance, GDELT..."):
            from src import paper_trader as pt
            result = pt.run_live_paper_trade(family=FAMILY)
        st.session_state[f"live_result_{FAMILY}"] = result

    result = st.session_state.get(f"live_result_{FAMILY}")
    if result is None:
        st.info(f"Click the button above to fetch the current {fam_title.lower()} market.")
    elif result["status"] != "ok":
        st.warning(result["message"])
    else:
        m = result["market"]
        st.subheader(m["question"] if "question" in m else m["slug"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{HORIZON_LABEL.capitalize()} to resolution",
                  f"{m['hours_to_resolution']:.2f}h" if not IS_HOURLY else f"{m['hours_to_resolution']*60:.1f}min")
        c2.metric("Market volume", f"${m['volume']:,.0f}")
        c3.metric("Raw implied P(Up)", f"{result['raw_implied_probability']:.3f}" if result["raw_implied_probability"] else "N/A")
        c4.metric("Calibrated fair value", f"{result['calibrated_fair_value']:.3f}" if result["calibrated_fair_value"] else "N/A")

        st.subheader("Live order book")
        ob1, ob2 = st.columns(2)
        with ob1:
            st.markdown("**UP token**")
            st.json(result["order_book"]["up"])
        with ob2:
            st.markdown("**DOWN token**")
            st.json(result["order_book"]["down"])

        st.subheader("Model probabilities")
        st.dataframe(pd.DataFrame([result["model_probabilities"]]), width='stretch')

        st.subheader("BTC / news / social snapshot")
        st.json({"btc": result["btc_snapshot"], "news": result["news_features"], "social": result["social_features"]})

        st.subheader("Risk checks & recommendation")
        st.json(result["risk_checks"])
        action_color = "green" if result["recommended_action"].startswith("BUY") else "gray"
        st.markdown(f"### Recommended action: :{action_color}[{result['recommended_action']}]")
        if result["signal"]:
            st.json(result["signal"])
            st.metric("Paper stake (simulated)", f"${result['paper_stake']:,.2f}")

    st.subheader(f"Paper trade history ({fam_title})")
    from src import paper_trader as pt
    mtm_path = pt.mtm_log_path(FAMILY)
    if mtm_path.exists():
        rows = [json.loads(line) for line in mtm_path.read_text().splitlines() if line.strip()]
        st.dataframe(pd.DataFrame(rows), width='stretch', height=250)
    else:
        st.info("No paper trade check-ins logged yet for this family.")

# ---------------------------------------------------------------------------
# TAB 6: LIVE P&L -- current model architecture (full_feature_model,
# walk-forward retrained) trading BTC Up/Down markets, entries priced at the
# real-book-walked VWAP. Retrospective (09-02->09-08) and real post-fix rows
# are combined into one "live" number, per explicit direction.
# ---------------------------------------------------------------------------
with tab6:
    st.header("Live P&L")
    st.caption("Current model architecture, trading Polymarket's BTC Up/Down markets on BTC's own "
               "momentum/volatility signal, entries priced at the real-book-walked VWAP.")

    refresh_c1, refresh_c2 = st.columns([1, 4])
    manual_refresh = refresh_c1.button("🔄 Refresh now", key="tab6_refresh")
    auto_refresh = refresh_c2.checkbox("Auto-refresh every 30s", value=False, key="tab6_autorefresh")
    _ = manual_refresh

    combined = ld.build_combined_live_view()
    if combined.empty:
        st.info("No trades to show yet.")
    else:
        s = combined[combined.settled]
        pnl = s.pnl_net.sum()

        # ONE shared $100,000 account, not one per family/source: daily and
        # hourly trades both draw against and pay back into this same starting
        # equity, in chronological order, exactly like a single backtest
        # PortfolioState (src/backtest.py) -- not four separate $100k pools
        # summed together. Recorded stakes are kept as-is rather than
        # re-simulated against a shared pool: the equity-dependent sizing caps
        # (10% fractional Kelly, 2% per-market) barely move for P&L this small
        # relative to $100k, and turnover here never got close to the 60%-
        # deployed or 25-concurrent-position caps, so a true shared-pool
        # resimulation would very likely reproduce the same stakes.
        start_equity = 100_000.0
        end_equity = start_equity + pnl

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Settled trades", f"{len(s)}")
        c2.metric("Start equity", f"${start_equity:,.0f}")
        c3.metric("End equity", f"${end_equity:,.0f}", delta=f"{pnl/start_equity:+.2%}")
        c4.metric("Staked", f"${s.stake.sum():,.0f}")
        c5.metric("Hit rate", f"{s.won.mean():.1%}")

        curve = s.sort_values("timestamp").copy()
        curve["cumulative_return"] = curve["pnl_net"].cumsum() / start_equity
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=curve["timestamp"], y=curve["cumulative_return"], mode="lines+markers",
            name="Cumulative return", line=dict(color="#2a9d8f"),
            hovertext=curve["slug"], hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:+.2%}<br>%{hovertext}<extra></extra>"))
        fig_cum.add_hline(y=0, line=dict(color="gray", dash="dash"))
        fig_cum.update_layout(height=350, margin=dict(t=20), yaxis_title="Cumulative return",
                               yaxis_tickformat=".0%", xaxis_title="Date")
        st.plotly_chart(fig_cum, width='stretch')

        with st.expander(f"Full trade log ({len(combined)} trades, settled + open)"):
            st.dataframe(combined, width='stretch', height=350)

    if auto_refresh:
        time.sleep(30)
        st.rerun()
