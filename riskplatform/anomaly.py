"""Unsupervised / statistical anomaly detection (labels are NEVER used to fit these)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# behaviour features whose extremes indicate unusual activity; transformed so they are roughly symmetric
ROBUST_Z_SPEC = {  # feature -> transform
    "amount_logz": lambda s: s,
    "cnt_1h": np.log1p,
    "cnt_24h_vs_daily_rate": np.log1p,
    "dist_from_home_km": np.log1p,
    "implied_speed_kmh": np.log1p,
    "small_cnt_30m": np.log1p,
    "amount_to_merch_mean": lambda s: np.log1p(s),
}
IFOREST_FEATURES = ["log_amount", "cnt_1h", "cnt_24h", "amount_logz", "amount_to_mean", "dist_from_home_km",
                    "implied_speed_kmh", "hour_local", "is_online", "new_device", "new_merchant_for_cust", "merch_fraud_rate_matured",
                    "small_cnt_30m", "distinct_merchants_24h"]


class RobustZ:
    """Median/MAD z-scores per feature (fit on reference data); score = mean of the top-2 positive z's."""

    def fit(self, f: pd.DataFrame) -> "RobustZ":
        self.params_ = {}
        for col, tf in ROBUST_Z_SPEC.items():
            x = pd.Series(tf(f[col].astype(float))).replace([np.inf, -np.inf], np.nan).dropna()
            med = x.median()
            mad = (x - med).abs().median()
            if mad < 1e-9:  # mostly-constant feature (e.g. counts): fall back to mean abs deviation, then to std
                mad = max((x - med).abs().mean() / 1.2533, x.std(), 1e-3) / 1.4826
            self.params_[col] = (float(med), float(1.4826 * mad))
        return self

    def z(self, f: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for col, tf in ROBUST_Z_SPEC.items():
            med, scale = self.params_[col]
            cols[col] = ((pd.Series(tf(f[col].astype(float)), index=f.index) - med) / scale).clip(-50, 50).fillna(0.0)
        return pd.DataFrame(cols, index=f.index)

    def score(self, f: pd.DataFrame) -> np.ndarray:
        z = np.sort(self.z(f).clip(lower=0).values, axis=1)[:, -2:]
        return z.mean(axis=1)


class IsoForest:
    def __init__(self, seed: int = 0, n_estimators: int = 200):
        self.model = IsolationForest(n_estimators=n_estimators, contamination="auto", random_state=seed, n_jobs=-1)

    def _x(self, f: pd.DataFrame) -> np.ndarray:
        return f[IFOREST_FEATURES].astype(float).fillna(-1.0).values

    def fit(self, f: pd.DataFrame) -> "IsoForest":
        self.model.fit(self._x(f))
        return self

    def score(self, f: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self._x(f))  # higher = more anomalous
