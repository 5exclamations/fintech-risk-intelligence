"""Reason codes.

Two complementary sources, merged for every scored transaction:
  * rule hits (exact, human-written conditions) and
  * model attributions: group-wise *occlusion*. For each interpretable feature group we replace the group's features
    by the typical value of a legitimate training transaction and measure how much the model's log-odds fall.
    This is model-agnostic, needs no extra dependency, and groups correlated features so the codes read like
    analyst language (e.g. "VELOCITY") instead of 50 raw features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_GROUPS: dict[str, list[str]] = {
    "VELOCITY": ["cnt_10m", "cnt_1h", "cnt_6h", "cnt_24h", "cnt_7d", "sum_1h", "sum_24h", "sum_7d", "small_cnt_30m",
                 "online_cnt_1h", "distinct_merchants_24h", "cnt_24h_vs_daily_rate", "hours_since_last"],
    "AMOUNT_DEVIATION": ["amount", "log_amount", "cust_prior_mean", "cust_prior_max", "amount_to_mean", "amount_to_max",
                         "amount_logz", "amount_gt_prior_max", "amount_to_merch_mean"],
    "GEO_ANOMALY": ["dist_from_last_km", "implied_speed_kmh", "dist_from_home_km", "is_foreign", "new_country"],
    "NEW_DEVICE": ["new_device", "device_prior_cnt"],
    "MERCHANT_RISK": ["merch_fraud_rate_matured", "merch_prior_n", "merch_prior_mean", "merch_cnt_1h", "merch_cnt_24h", "merch_age_days"],
    "UNFAMILIAR_MERCHANT": ["new_merchant_for_cust", "new_category_for_cust", "cust_category_share", "category_code", "merch_is_online"],
    "TIME_PATTERN": ["hour_local", "dow_local", "is_night", "is_weekend", "cust_night_share", "cust_hourband_share"],
    "CUSTOMER_PROFILE": ["cust_prior_n", "cust_channel_share", "is_online", "is_atm", "account_age_days", "is_credit"],
}

TEMPLATES = {
    "VELOCITY": lambda r: f"{r.get('cnt_1h', np.nan):.0f} txns in the last hour / {r.get('cnt_24h', np.nan):.0f} in 24h "
                          f"({r.get('small_cnt_30m', np.nan):.0f} micro-payments in 30 min)",
    "AMOUNT_DEVIATION": lambda r: f"amount ${r.get('amount', np.nan):,.2f} is {r.get('amount_to_mean', np.nan):.1f}x the customer's average",
    "GEO_ANOMALY": lambda r: f"{r.get('dist_from_home_km', np.nan):,.0f} km from home; {r.get('dist_from_last_km', np.nan):,.0f} km from the "
                             f"previous transaction (implied {r.get('implied_speed_kmh', np.nan):,.0f} km/h)",
    "NEW_DEVICE": lambda r: "device never used by this customer before",
    "MERCHANT_RISK": lambda r: f"merchant's matured-label fraud rate is {100 * r.get('merch_fraud_rate_matured', np.nan):.1f}%",
    "UNFAMILIAR_MERCHANT": lambda r: "merchant/category unfamiliar for this customer",
    "TIME_PATTERN": lambda r: f"unusual time of day (local hour {r.get('hour_local', np.nan):.0f}, customer night share "
                              f"{100 * r.get('cust_night_share', np.nan):.0f}%)",
    "CUSTOMER_PROFILE": lambda r: "limited customer history / unusual channel for this customer",
}


def baseline_values(train_features: pd.DataFrame, legit_mask: np.ndarray) -> dict[str, float]:
    """Typical legitimate value per feature (median), used as the 'occluded' value."""
    med = train_features.loc[legit_mask].median(numeric_only=True)
    return {k: (float(v) if pd.notna(v) else 0.0) for k, v in med.items()}


def group_contributions(predict_logit, X: pd.DataFrame, baseline: dict[str, float]) -> pd.DataFrame:
    """Drop in model log-odds when each group is replaced by its baseline. Rows align with X; columns = groups."""
    full = predict_logit(X)
    out = {}
    for g, cols in FEATURE_GROUPS.items():
        cols = [c for c in cols if c in X.columns]
        Xo = X.copy()
        for c in cols:
            Xo[c] = baseline[c]
        out[g] = full - predict_logit(Xo)
    return pd.DataFrame(out, index=X.index)


def model_reasons(contrib_row: pd.Series, feature_row: pd.Series, top_k: int = 3, min_contrib: float = 0.15) -> list[dict]:
    top = contrib_row.sort_values(ascending=False)
    reasons = []
    for g, v in top.items():
        if v < min_contrib or len(reasons) >= top_k:
            break
        reasons.append({"code": g, "source": "model", "contribution": round(float(v), 3), "detail": TEMPLATES[g](feature_row)})
    return reasons
