"""Baseline / calibration models compared in the walk-forward evaluation.

All fitting happens on a caller-supplied *training* DataFrame only -- these
functions never see the full dataset, which is what keeps the walk-forward
harness (src/walkforward.py) leakage-free. Preprocessing (scaling) is folded
into the same fit call so it too only ever sees the training window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6

# Feature set for the "full" regularized-logistic model. News/social columns are
# included in the observation schema but excluded here because they were
# unavailable for every observation in this run (see reports/data_quality_report.md) --
# a feature that is 100% missing in the training window cannot be fit or scaled
# meaningfully. Re-include them once a working news/social feed is available.
FULL_FEATURE_COLS = [
    "yes_probability", "horizon_hours",
    "btc_return_15m", "btc_return_1h", "btc_return_6h",
    "btc_realized_vol_6h", "btc_spot_volume_1h",
    "market_volume", "market_liquidity",
]


def logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, EPS, 1 - EPS)
    return np.log(pc / (1 - pc))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class ConstantModel:
    name = "constant_50"

    def fit(self, train_df: pd.DataFrame):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 0.5)


class RawProbabilityModel:
    name = "raw_probability"

    def fit(self, train_df: pd.DataFrame):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["yes_probability"].values.astype(float)


class LogisticCalibrationModel:
    """Platt scaling: y ~ sigmoid(a + b * logit(raw_prob)), fit on TRAIN only."""
    name = "logistic_calibration"

    def __init__(self):
        self.lr: LogisticRegression | None = None

    def fit(self, train_df: pd.DataFrame):
        z = logit(train_df["yes_probability"].values).reshape(-1, 1)
        y = train_df["outcome_up"].values
        if len(np.unique(y)) < 2:
            self.lr = None
            return self
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")
        self.lr.fit(z, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.lr is None:
            return df["yes_probability"].values.astype(float)
        z = logit(df["yes_probability"].values).reshape(-1, 1)
        return self.lr.predict_proba(z)[:, 1]


class IsotonicCalibrationModel:
    name = "isotonic_calibration"

    def __init__(self):
        self.iso: IsotonicRegression | None = None

    def fit(self, train_df: pd.DataFrame):
        x = train_df["yes_probability"].values
        y = train_df["outcome_up"].values
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso.fit(x, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.iso is None:
            return df["yes_probability"].values.astype(float)
        return self.iso.predict(df["yes_probability"].values)


class RegularizedLogisticModel:
    """L2-regularized logistic regression on raw prob + time-to-resolution +
    BTC momentum/vol/volume + market volume/liquidity. Scaling is fit inside
    the same training window (Pipeline), never on the full dataset."""
    name = "full_feature_model"

    def __init__(self, feature_cols: list[str] | None = None, C: float = 1.0):
        self.feature_cols = feature_cols or FULL_FEATURE_COLS
        self.C = C
        self.pipe: Pipeline | None = None

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()
        X["yes_probability"] = logit(X["yes_probability"].values)  # model in logit space
        return X.values.astype(float)

    def fit(self, train_df: pd.DataFrame):
        X = self._X(train_df)
        y = train_df["outcome_up"].values
        if len(np.unique(y)) < 2:
            self.pipe = None
            return self
        self.pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, solver="lbfgs", max_iter=500)),
        ])
        self.pipe.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipe is None:
            return df["yes_probability"].values.astype(float)
        return self.pipe.predict_proba(self._X(df))[:, 1]


MODEL_REGISTRY = {
    "constant_50": ConstantModel,
    "raw_probability": RawProbabilityModel,
    "logistic_calibration": LogisticCalibrationModel,
    "isotonic_calibration": IsotonicCalibrationModel,
    "full_feature_model": RegularizedLogisticModel,
}
