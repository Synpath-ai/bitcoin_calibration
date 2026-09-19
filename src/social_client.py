"""
Optional social/attention adapter. Kept intentionally separate from
`news_client` (GDELT-style media coverage) so social-media activity is never
mislabeled as a general news/sentiment proxy, per the project spec.

No credentialed API (e.g. Reddit OAuth) is available in this environment, and
Reddit's unauthenticated JSON endpoints only expose current hot/new listings,
not a genuine historical time series -- using them here would mean silently
building "historical" features out of whatever posts happen to be live today,
which is indistinguishable from fabrication. Rather than do that, this
adapter is a clean, honest no-op: it always reports `social_data_available =
False` unless a user supplies a CSV of real, timestamped social activity.

The main calibration/backtest pipeline must run correctly with this adapter
returning nothing -- and it does (see observation_builder.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SocialFeatures:
    social_data_available: bool
    activity_level: float = float("nan")
    activity_change: float = float("nan")
    sentiment: float = float("nan")
    engagement: float = float("nan")

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def load_social_csv(path: str) -> pd.DataFrame:
    """Expected columns: timestamp (ISO8601 UTC), activity, sentiment, engagement."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def get_social_features(as_of: pd.Timestamp, csv_df: pd.DataFrame | None = None) -> SocialFeatures:
    if csv_df is None:
        return SocialFeatures(social_data_available=False)
    window = csv_df[(csv_df.timestamp <= as_of) & (csv_df.timestamp > as_of - pd.Timedelta(hours=24))]
    prior = csv_df[(csv_df.timestamp <= as_of - pd.Timedelta(hours=24)) &
                    (csv_df.timestamp > as_of - pd.Timedelta(hours=48))]
    if window.empty:
        return SocialFeatures(social_data_available=False)
    return SocialFeatures(
        social_data_available=True,
        activity_level=float(window["activity"].sum()),
        activity_change=float(window["activity"].sum() - prior["activity"].sum()) if not prior.empty else float("nan"),
        sentiment=float(window["sentiment"].mean()) if "sentiment" in window else float("nan"),
        engagement=float(window["engagement"].mean()) if "engagement" in window else float("nan"),
    )
