"""Chronological walk-forward evaluation for the extended hourly BTC Up/Down sample."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import calibration as cal
from src import walkforward as wf

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def main():
    obs_df = pd.read_parquet(PROCESSED / "hourly_observations.parquet")
    markets = json.loads((PROCESSED / "hourly_markets_accepted.json").read_text())
    mkts_df = pd.DataFrame(markets)[["slug", "start_date", "end_date"]]
    mkts_df["start_date"] = pd.to_datetime(mkts_df["start_date"], utc=True, format="ISO8601")
    mkts_df["end_date"] = pd.to_datetime(mkts_df["end_date"], utc=True, format="ISO8601")
    mkts_df = mkts_df[mkts_df["slug"].isin(obs_df["slug"].unique())]

    print(f"Running walk-forward over {len(mkts_df)} hourly markets, {len(obs_df)} observations...")
    # much larger N than the daily series -> require a bigger seed window before
    # trusting predictions, but still small relative to the ~2000-market sample.
    pred_df, diag = wf.run_walkforward(obs_df, mkts_df, min_train_markets=100)
    print(f"Test markets used: {diag['n_test_markets_used']}, skipped (insufficient history): "
          f"{diag['n_test_markets_skipped']}")

    pred_df.to_parquet(PROCESSED / "hourly_walkforward_predictions.parquet")
    (PROCESSED / "hourly_walkforward_diagnostics.json").write_text(json.dumps(diag, indent=2, default=str))

    summary = []
    for model_name, g in pred_df.groupby("model"):
        for h, gh in g.groupby("horizon_hours"):
            p, y = gh["model_probability"].values, gh["outcome_up"].values
            brier_pt, brier_lo, brier_hi = cal.bootstrap_metric_by_market(
                gh, "model_probability", "outcome_up", "slug", cal.brier_score)
            ll_pt, _, _ = cal.bootstrap_metric_by_market(
                gh, "model_probability", "outcome_up", "slug", cal.log_loss)
            intercept, slope = cal.calibration_intercept_slope(p, y)
            summary.append({
                "model": model_name, "horizon_minutes": round(float(h) * 60), "n": len(gh),
                "n_markets": gh["slug"].nunique(),
                "brier": brier_pt, "brier_ci_lo": brier_lo, "brier_ci_hi": brier_hi,
                "log_loss": ll_pt, "calib_intercept": intercept, "calib_slope": slope,
            })
    summary_df = pd.DataFrame(summary).sort_values(["horizon_minutes", "model"])
    summary_df.to_csv(PROCESSED / "hourly_walkforward_model_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
