"""Point-in-time feature engineering.

Leakage rules enforced here (and unit-tested in ``tests/test_features.py``):

1. A feature for transaction *t* may only use rows with ``ts < t.ts`` (strictly earlier). The row itself and
   rows sharing its timestamp are never used.
2. Label-derived features (merchant fraud rate) may only use labels that have *matured*:
   a row's label is treated as known only ``label_delay_days`` after that row's own timestamp.
3. The generator's ground-truth tables (``synthetic_truth*``) are never read here.

The same function is used for batch training and for online scoring (the API passes a customer + merchant
history slice and the new transaction), so there is no training/serving skew in the feature code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SPLIT

K = np.int64(1 << 34)  # group stride for composite (group, timestamp) sort keys; epoch seconds < 2^34
EARTH_KM = 6371.0088
CATEGORIES = ["grocery", "restaurant", "fuel", "retail", "electronics", "travel", "entertainment", "online_services",
              "gift_cards", "crypto_exchange", "utilities", "healthcare", "atm_cash"]
MERCHANT_PRIOR_RATE = 0.01   # shrinkage target for merchant fraud rate
MERCHANT_PRIOR_K = 200.0

NUMERIC_FEATURES = [
    # transaction
    "amount", "log_amount", "is_online", "is_atm", "hour_local", "dow_local", "is_night", "is_weekend", "category_code",
    # velocity (customer)
    "cnt_10m", "cnt_1h", "cnt_6h", "cnt_24h", "cnt_7d", "sum_1h", "sum_24h", "sum_7d", "small_cnt_30m", "online_cnt_1h",
    "distinct_merchants_24h",
    # customer history / deviations
    "cust_prior_n", "cust_prior_mean", "cust_prior_max", "amount_to_mean", "amount_to_max", "amount_logz", "amount_gt_prior_max",
    "hours_since_last", "dist_from_last_km", "implied_speed_kmh", "dist_from_home_km", "is_foreign", "new_country",
    "new_merchant_for_cust", "new_category_for_cust", "cust_channel_share", "cust_category_share", "cust_hourband_share",
    "cust_night_share", "new_device", "device_prior_cnt", "cnt_24h_vs_daily_rate",
    # merchant history
    "merch_prior_n", "merch_prior_mean", "amount_to_merch_mean", "merch_cnt_1h", "merch_cnt_24h", "merch_age_days",
    "merch_fraud_rate_matured", "merch_is_online",
    # account
    "account_age_days", "is_credit",
]
CATEGORICAL_FEATURES = ["category_code"]


# --------------------------------------------------------------------------- group-wise prior aggregates
def _prior(g: np.ndarray, ts: np.ndarray, values: tuple[np.ndarray, ...] = (), window: int | None = None,
           mask: np.ndarray | None = None):
    """For each row: count (and sums of ``values``) of rows in the same group with ts STRICTLY earlier.

    ``window`` restricts to ``ts - window <= ts_j < ts``. ``mask`` restricts which rows are counted (e.g. only
    online rows). Returns (count, sums..., prev_idx) in the original row order; prev_idx = latest earlier row
    in the group (or -1).
    """
    n = len(g)
    order = np.lexsort((ts, g))
    gs, tss = g[order].astype(np.int64), ts[order].astype(np.int64)
    key = gs * K + tss
    pos = np.searchsorted(key, key, side="left")
    start = np.searchsorted(key, gs * K, side="left") if window is None else np.searchsorted(key, key - window, side="left")
    m = np.ones(n) if mask is None else mask[order].astype(float)
    cm = np.concatenate([[0.0], np.cumsum(m)])
    cnt = cm[pos] - cm[start]
    outs = [cnt]
    for v in values:
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v[order].astype(float)) * m)])
        outs.append(cs[pos] - cs[start])
    first = np.searchsorted(key, gs * K, side="left")
    prev_sorted = np.where(pos > first, pos - 1, -1)
    prev = np.where(prev_sorted >= 0, order[np.maximum(prev_sorted, 0)], -1)
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    return [o[inv] for o in outs] + [prev[inv]]


def _distinct_in_window(g: np.ndarray, item: np.ndarray, ts: np.ndarray, window: int) -> np.ndarray:
    """Distinct ``item`` values among earlier rows of the same group in (ts-window, ts). O(n * rows_in_window)."""
    order = np.lexsort((ts, g))
    gs, tss, it = g[order].astype(np.int64), ts[order].astype(np.int64), item[order]
    key = gs * K + tss
    pos = np.searchsorted(key, key, side="left")
    lo = np.searchsorted(key, key - window, side="left")
    out = np.zeros(len(g))
    for i in np.where(pos - lo > 0)[0]:
        out[i] = len(np.unique(it[lo[i]:pos[i]]))
    inv = np.empty(len(g), dtype=np.int64)
    inv[order] = np.arange(len(g))
    return out[inv]


def _epoch_s(s: pd.Series) -> np.ndarray:
    """Epoch seconds (float, NaN for missing) independent of pandas' datetime resolution."""
    return (pd.to_datetime(s) - pd.Timestamp("1970-01-01")).dt.total_seconds().values


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _codes(*cols) -> np.ndarray:
    """Dense integer code for the combination of one or more columns."""
    if len(cols) == 1:
        return pd.factorize(pd.Series(cols[0]).astype(str))[0]
    return pd.factorize(pd.Series(cols[0]).astype(str) + "|" + pd.Series(cols[1]).astype(str))[0]


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


# --------------------------------------------------------------------------------------- main entry
def build_features(tx: pd.DataFrame, customers: pd.DataFrame, accounts: pd.DataFrame, merchants: pd.DataFrame,
                   locations: pd.DataFrame, label_delay_days: int = SPLIT.label_delay_days) -> pd.DataFrame:
    """Return one feature row per input transaction (same order as ``tx``).

    ``tx`` needs: txn_id, customer_id, account_id, merchant_id, location_id, ts_epoch, amount, channel, device_id,
    hour_local, dow_local, and (for label-derived features) is_fraud. Rows with is_fraud NaN are treated as unlabeled.
    """
    d = tx.reset_index(drop=True).copy()
    d["_o"] = np.arange(len(d))
    d = d.sort_values(["ts_epoch", "txn_id"], kind="stable").reset_index(drop=True)  # cummax etc. need time order
    n = len(d)
    ts = d.ts_epoch.values.astype(np.int64)
    amt = d.amount.values.astype(float)
    d = d.merge(merchants[["merchant_id", "category", "is_online", "onboarded_at"]].rename(columns={"is_online": "m_online"}),
                on="merchant_id", how="left")
    d = d.merge(locations[["location_id", "lat", "lon", "country"]], on="location_id", how="left")
    d = d.merge(customers[["customer_id", "home_location_id"]], on="customer_id", how="left")
    home = locations.set_index("location_id")
    d["home_lat"] = home.lat.reindex(d.home_location_id.values).values
    d["home_lon"] = home.lon.reindex(d.home_location_id.values).values
    d["home_country"] = home.country.reindex(d.home_location_id.values).values
    d = d.merge(accounts[["account_id", "account_type", "opened_at"]], on="account_id", how="left")
    assert len(d) == n, "reference tables must have unique keys"

    f = pd.DataFrame(index=np.arange(n))
    f["txn_id"] = d.txn_id.values
    f["amount"] = amt
    f["log_amount"] = np.log1p(amt)
    f["is_online"] = (d.channel.values == "online").astype(int)
    f["is_atm"] = (d.channel.values == "atm").astype(int)
    f["hour_local"] = d.hour_local.values
    f["dow_local"] = d.dow_local.values
    f["is_night"] = (d.hour_local.values <= 5).astype(int)
    f["is_weekend"] = (d.dow_local.values >= 5).astype(int)
    cat = pd.Categorical(d.category, categories=CATEGORIES)
    f["category_code"] = cat.codes.astype(float)
    f.loc[f.category_code < 0, "category_code"] = np.nan

    c = _codes(d.customer_id.values)
    lamt = np.log1p(amt)
    # --- velocity windows (customer)
    for name, w in [("10m", 600), ("1h", 3600), ("6h", 21600), ("24h", 86400), ("7d", 604800)]:
        out = _prior(c, ts, (amt,), window=w)
        f[f"cnt_{name}"] = out[0]
        if name in ("1h", "24h", "7d"):
            f[f"sum_{name}"] = out[1]
    f["small_cnt_30m"] = _prior(c, ts, window=1800, mask=amt < 5.0)[0]
    f["online_cnt_1h"] = _prior(c, ts, window=3600, mask=(d.channel.values == "online"))[0]
    f["distinct_merchants_24h"] = _distinct_in_window(c, d.merchant_id.values.astype(str), ts, 86400)
    # --- customer history
    n_prior, s_amt, s_lamt, s_lamt2, prev = _prior(c, ts, (amt, lamt, lamt ** 2))
    f["cust_prior_n"] = n_prior
    mean = _safe_div(s_amt, n_prior)
    f["cust_prior_mean"] = mean
    gmax = pd.Series(amt).groupby(c).cummax().values  # running max INCLUDING self ...
    prev_max = np.where(prev >= 0, gmax[np.maximum(prev, 0)], np.nan)  # ... read at the previous row => excludes self
    f["cust_prior_max"] = prev_max
    f["amount_to_mean"] = _safe_div(amt, mean)
    f["amount_to_max"] = _safe_div(amt, prev_max)
    f["amount_gt_prior_max"] = np.where(n_prior >= 5, (amt > prev_max).astype(float), np.nan)
    mu_l = _safe_div(s_lamt, n_prior)
    var_l = np.maximum(_safe_div(s_lamt2, n_prior) - mu_l ** 2, 0)
    sd_l = np.sqrt(var_l)
    f["amount_logz"] = np.where(n_prior >= 5, (lamt - mu_l) / np.maximum(sd_l, 0.25), np.nan)
    has_prev = prev >= 0
    pidx = np.maximum(prev, 0)
    dt = np.where(has_prev, ts - ts[pidx], np.nan)
    f["hours_since_last"] = dt / 3600
    lat, lon = d.lat.values.astype(float), d.lon.values.astype(float)
    dist_last = np.where(has_prev, _haversine(lat[pidx], lon[pidx], lat, lon), np.nan)
    f["dist_from_last_km"] = dist_last
    f["implied_speed_kmh"] = np.where(has_prev, dist_last / np.maximum(dt / 3600, 1 / 60), np.nan)
    f["dist_from_home_km"] = _haversine(d.home_lat.values.astype(float), d.home_lon.values.astype(float), lat, lon)
    f["is_foreign"] = (d.country.values != d.home_country.values).astype(float)
    f.loc[d.home_country.isna().values, "is_foreign"] = np.nan

    has_hist = n_prior > 0
    f["new_country"] = np.where(has_hist, (_prior(_codes(d.customer_id.values, d.country.values), ts)[0] == 0).astype(float), np.nan)
    f["new_merchant_for_cust"] = np.where(has_hist, (_prior(_codes(d.customer_id.values, d.merchant_id.values), ts)[0] == 0).astype(float), np.nan)
    f["new_category_for_cust"] = np.where(has_hist, (_prior(_codes(d.customer_id.values, d.category.values), ts)[0] == 0).astype(float), np.nan)
    f["cust_channel_share"] = _safe_div(_prior(_codes(d.customer_id.values, d.channel.values), ts)[0], n_prior)
    f["cust_category_share"] = _safe_div(_prior(_codes(d.customer_id.values, d.category.values), ts)[0], n_prior)
    band = (d.hour_local.values // 4).astype(int)
    f["cust_hourband_share"] = _safe_div(_prior(_codes(d.customer_id.values, band), ts)[0], n_prior)
    night = (d.hour_local.values <= 5).astype(float)
    f["cust_night_share"] = _safe_div(_prior(c, ts, mask=night > 0)[0], n_prior)
    dev = d.device_id.fillna("").values.astype(str)
    has_dev = dev != ""
    dcnt = _prior(_codes(d.customer_id.values, dev), ts)[0]
    f["device_prior_cnt"] = np.where(has_dev, dcnt, np.nan)
    f["new_device"] = np.where(has_dev & has_hist, (dcnt == 0).astype(float), np.nan)
    first_ts = pd.Series(ts).groupby(c).transform("min").values
    obs_days = np.maximum((ts - first_ts) / 86400, 1.0)
    daily_rate = _safe_div(n_prior, np.where(has_hist, obs_days, 0))
    f["cnt_24h_vs_daily_rate"] = _safe_div(f.cnt_24h.values, np.maximum(daily_rate, 0.05))

    # --- merchant history
    m = _codes(d.merchant_id.values)
    mn, ms, mprev = _prior(m, ts, (amt,))
    f["merch_prior_n"] = mn
    mm = _safe_div(ms, mn)
    f["merch_prior_mean"] = mm
    f["amount_to_merch_mean"] = _safe_div(amt, mm)
    f["merch_cnt_1h"] = _prior(m, ts, window=3600)[0]
    f["merch_cnt_24h"] = _prior(m, ts, window=86400)[0]
    onboard = _epoch_s(d.onboarded_at)
    f["merch_age_days"] = np.maximum((ts - onboard) / 86400, 0)
    f["merch_is_online"] = d.m_online.values.astype(float)
    f["merch_fraud_rate_matured"] = _matured_rate(m, ts, d.is_fraud.values, label_delay_days)
    # --- account
    f["account_age_days"] = np.maximum((ts - _epoch_s(d.opened_at)) / 86400, 0)
    f["is_credit"] = np.where(d.account_type.isna().values, np.nan, (d.account_type.values == "credit_card").astype(float))
    f = f[["txn_id"] + NUMERIC_FEATURES]
    return f.iloc[np.argsort(d["_o"].values, kind="stable")].reset_index(drop=True)  # back to the caller's row order


def _matured_rate(m: np.ndarray, ts: np.ndarray, is_fraud: np.ndarray, delay_days: int) -> np.ndarray:
    """Shrunk merchant fraud rate from labels that were already *known* at scoring time (ts_j + delay <= ts_i)."""
    lab = pd.Series(is_fraud).astype(float)
    labelled = ~lab.isna().values
    fr = np.nan_to_num(lab.values)
    mat = ts + int(delay_days) * 86400
    sel = np.where(labelled)[0]
    order = sel[np.lexsort((mat[sel], m[sel]))]
    key = m[order].astype(np.int64) * K + mat[order]
    cf = np.concatenate([[0.0], np.cumsum(fr[order])])
    q = m.astype(np.int64) * K + ts
    hi = np.searchsorted(key, q, side="right")           # labels matured at or before this row's time
    lo = np.searchsorted(key, m.astype(np.int64) * K, side="left")
    n_mat, n_fraud = (hi - lo).astype(float), cf[hi] - cf[lo]
    return (n_fraud + MERCHANT_PRIOR_K * MERCHANT_PRIOR_RATE) / (n_mat + MERCHANT_PRIOR_K)
