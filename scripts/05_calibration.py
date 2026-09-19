"""Calibration analysis of the RAW Polymarket implied probability, per horizon."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import calibration as cal

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    df = pd.read_parquet(PROCESSED / "observations.parquet")
    results = {}

    for h in sorted(df["horizon_hours"].unique(), reverse=True):
        sub = df[df["horizon_hours"] == h].copy()
        p, y = sub["yes_probability"].values, sub["outcome_up"].values

        brier_pt, brier_lo, brier_hi = cal.bootstrap_metric_by_market(
            sub, "yes_probability", "outcome_up", "slug", cal.brier_score)
        ll_pt, ll_lo, ll_hi = cal.bootstrap_metric_by_market(
            sub, "yes_probability", "outcome_up", "slug", cal.log_loss)
        ece_pt, ece_lo, ece_hi = cal.bootstrap_metric_by_market(
            sub, "yes_probability", "outcome_up", "slug",
            lambda pp, yy: cal.expected_calibration_error(pp, yy, n_bins=10))
        intercept, slope = cal.calibration_intercept_slope(p, y)

        fixed_table = cal.reliability_table(sub, "yes_probability", "outcome_up", "slug",
                                             cal.fixed_buckets(10))
        adaptive_edges = cal.adaptive_quantile_buckets(p, n_bins=6)
        adaptive_table = cal.reliability_table(sub, "yes_probability", "outcome_up", "slug",
                                                adaptive_edges)

        results[str(h)] = {
            "n_observations": len(sub),
            "n_independent_markets": sub["slug"].nunique(),
            "brier_score": {"point": brier_pt, "ci_lo": brier_lo, "ci_hi": brier_hi},
            "log_loss": {"point": ll_pt, "ci_lo": ll_lo, "ci_hi": ll_hi},
            "ece_10bin": {"point": ece_pt, "ci_lo": ece_lo, "ci_hi": ece_hi},
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "fixed_bucket_table": fixed_table.to_dict(orient="records"),
            "adaptive_bucket_table": adaptive_table.to_dict(orient="records"),
        }

        print(f"\n=== Horizon {h}h ===  n={len(sub)} markets={sub['slug'].nunique()}")
        print(f"Brier: {brier_pt:.4f} [{brier_lo:.4f}, {brier_hi:.4f}]")
        print(f"LogLoss: {ll_pt:.4f} [{ll_lo:.4f}, {ll_hi:.4f}]")
        print(f"ECE(10bin): {ece_pt:.4f} [{ece_lo:.4f}, {ece_hi:.4f}]")
        print(f"Calibration intercept={intercept:.3f} slope={slope:.3f}")

    (PROCESSED / "calibration_raw.json").write_text(json.dumps(results, indent=2, default=str))

    # also a constant-50% baseline for comparison, all horizons pooled
    y_all = df["outcome_up"].values
    const_brier = cal.brier_score(np.full_like(y_all, 0.5, dtype=float), y_all)
    const_ll = cal.log_loss(np.full_like(y_all, 0.5, dtype=float), y_all)
    print(f"\nConstant-50% baseline (pooled all horizons): Brier={const_brier:.4f} LogLoss={const_ll:.4f}")

    summary = {
        "constant_50_baseline": {"brier": const_brier, "log_loss": const_ll},
        "horizons": sorted(int(h) for h in results.keys()),
    }
    (PROCESSED / "calibration_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
