"""Synthetic financial-transaction generator.

EVERYTHING produced here is synthetic. The assumptions are listed in ``docs/synthetic_data_assumptions.md``
and mirrored in comments below. Fraud patterns are *invented* simulations of well-known fraud typologies;
they do not reproduce the statistics of any real institution, so model metrics measured on this data must
not be read as real-world performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GEN, GenConfig

# (city, country, lat, lon, utc_offset_hours, domestic, population_weight)
_CITIES = [
    ("New York", "US", 40.71, -74.01, -5, 1, 8.4), ("Los Angeles", "US", 34.05, -118.24, -8, 1, 4.0),
    ("Chicago", "US", 41.88, -87.63, -6, 1, 2.7), ("Houston", "US", 29.76, -95.37, -6, 1, 2.3),
    ("Phoenix", "US", 33.45, -112.07, -7, 1, 1.6), ("Philadelphia", "US", 39.95, -75.17, -5, 1, 1.6),
    ("San Antonio", "US", 29.42, -98.49, -6, 1, 1.5), ("San Diego", "US", 32.72, -117.16, -8, 1, 1.4),
    ("Dallas", "US", 32.78, -96.80, -6, 1, 1.3), ("San Jose", "US", 37.34, -121.89, -8, 1, 1.0),
    ("Austin", "US", 30.27, -97.74, -6, 1, 1.0), ("Jacksonville", "US", 30.33, -81.66, -5, 1, 0.9),
    ("Columbus", "US", 39.96, -83.00, -5, 1, 0.9), ("Charlotte", "US", 35.23, -80.84, -5, 1, 0.9),
    ("Seattle", "US", 47.61, -122.33, -8, 1, 0.75), ("Denver", "US", 39.74, -104.99, -7, 1, 0.72),
    ("Boston", "US", 42.36, -71.06, -5, 1, 0.68), ("Atlanta", "US", 33.75, -84.39, -5, 1, 0.5),
    ("Miami", "US", 25.76, -80.19, -5, 1, 0.45), ("Minneapolis", "US", 44.98, -93.27, -6, 1, 0.43),
    ("Toronto", "CA", 43.65, -79.38, -5, 0, 0), ("London", "GB", 51.51, -0.13, 0, 0, 0),
    ("Paris", "FR", 48.86, 2.35, 1, 0, 0), ("Berlin", "DE", 52.52, 13.40, 1, 0, 0),
    ("Madrid", "ES", 40.42, -3.70, 1, 0, 0), ("Mexico City", "MX", 19.43, -99.13, -6, 0, 0),
    ("Sao Paulo", "BR", -23.55, -46.63, -3, 0, 0), ("Lagos", "NG", 6.52, 3.38, 1, 0, 0),
    ("Mumbai", "IN", 19.08, 72.88, 5.5, 0, 0), ("Tokyo", "JP", 35.68, 139.69, 9, 0, 0),
    ("Sydney", "AU", -33.87, 151.21, 10, 0, 0), ("Dubai", "AE", 25.20, 55.27, 4, 0, 0),
    ("Singapore", "SG", 1.35, 103.82, 8, 0, 0), ("Bucharest", "RO", 44.43, 26.10, 2, 0, 0),
    ("Istanbul", "TR", 41.01, 28.98, 3, 0, 0), ("Johannesburg", "ZA", -26.20, 28.05, 2, 0, 0),
    ("Bangkok", "TH", 13.76, 100.50, 7, 0, 0), ("Manila", "PH", 14.60, 120.98, 8, 0, 0),
]

# name: (mean log-amount shift, share, online-eligibility factor)
_CATEGORIES = {
    "grocery": (0.0, 0.20, 0.15), "restaurant": (-0.1, 0.15, 0.25), "fuel": (-0.1, 0.08, 0.0),
    "retail": (0.3, 0.12, 0.6), "electronics": (1.2, 0.03, 0.8), "travel": (1.3, 0.03, 0.9),
    "entertainment": (0.0, 0.07, 0.7), "online_services": (-0.4, 0.10, 1.0), "gift_cards": (0.4, 0.01, 0.9),
    "crypto_exchange": (1.0, 0.005, 1.0), "utilities": (0.8, 0.06, 0.85), "healthcare": (0.5, 0.04, 0.2),
    "atm_cash": (0.5, 0.04, 0.0),
}
_CAT_NAMES = list(_CATEGORIES)

_SEGMENTS = {  # weight, relative rate, mu(log $), sigma, online share, night-owl share, trip propensity
    "student": (0.15, 0.80, 2.7, 0.80, 0.45, 0.25, 0.02),
    "young_professional": (0.28, 1.30, 3.3, 0.85, 0.40, 0.10, 0.06),
    "family": (0.27, 1.30, 3.5, 0.80, 0.30, 0.04, 0.04),
    "affluent": (0.12, 1.15, 4.2, 1.00, 0.35, 0.05, 0.14),
    "retiree": (0.18, 0.75, 3.2, 0.70, 0.12, 0.01, 0.05),
}

_HOUR_BASE = np.array([1, .5, .3, .2, .2, .4, 1, 2, 3, 3.5, 4, 5, 6, 5, 4, 4, 4.5, 5.5, 6, 5, 4, 3, 2, 1.5], float)
_HOUR_NIGHT = np.array([5, 4, 3, 2, 1, .5, .3, .3, .5, 1, 1.5, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 6, 6], float)


def locations() -> pd.DataFrame:
    df = pd.DataFrame(_CITIES, columns=["city", "country", "lat", "lon", "utc_offset", "is_domestic", "pop_w"])
    df.insert(0, "location_id", np.arange(1, len(df) + 1))
    return df


def _cdf_sample(rng, probs: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Sample one column index per row from a (n_rows_unique, k) probability matrix."""
    cdf = np.cumsum(probs, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random(len(rows))
    return np.minimum((cdf[rows] < u[:, None]).sum(1), probs.shape[1] - 1)


def _customers(rng, cfg: GenConfig, loc: pd.DataFrame) -> pd.DataFrame:
    n = cfg.n_customers
    names = list(_SEGMENTS)
    seg = rng.choice(names, n, p=[_SEGMENTS[s][0] for s in names])
    dom = loc[loc.is_domestic == 1]
    home = rng.choice(dom.location_id.values, n, p=dom.pop_w.values / dom.pop_w.sum())
    existing = rng.random(n) < 0.75
    start_day = np.where(existing, -rng.integers(30, 2000, n), rng.integers(0, max(cfg.days - 45, 1), n))
    rate = np.array([_SEGMENTS[s][1] for s in seg]) * rng.lognormal(0, 0.5, n)
    rate *= cfg.target_txn_per_customer_day / rate.mean()
    t0 = pd.Timestamp(cfg.start)
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)], "segment": seg, "home_location_id": home,
        "age": np.clip(rng.normal(np.select([seg == "student", seg == "retiree"], [22, 68], 40), 9), 18, 90).astype(int),
        "signup_date": t0 + pd.to_timedelta(start_day, unit="D"),
        "kyc_risk_rating": rng.choice(["low", "medium", "high"], n, p=[0.8, 0.17, 0.03]),
        "_rate": rate, "_start_day": start_day,
    })


def _merchants(rng, cfg: GenConfig, loc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cat_p = np.array([v[1] for v in _CATEGORIES.values()]); cat_p /= cat_p.sum()
    mandatory = ["grocery", "restaurant", "fuel", "retail", "atm_cash"]
    k_dom = max(6, round(0.65 * cfg.n_merchants / (20 + 18 / 3)))  # 18 per domestic city at the default 700 merchants
    for r in loc.itertuples():
        k = k_dom if r.is_domestic else max(5, k_dom // 3)
        cats = mandatory + list(rng.choice(_CAT_NAMES, k - len(mandatory), p=cat_p))
        rows += [(r.location_id, c, 0) for c in cats]
    n_online = max(cfg.n_merchants - len(rows), 0)
    on_cats = [c for c in _CAT_NAMES if _CATEGORIES[c][2] > 0]
    p = np.array([_CATEGORIES[c][1] * _CATEGORIES[c][2] for c in on_cats]); p /= p.sum()
    hq = loc[loc.is_domestic == 1].location_id.values
    rows += [(int(rng.choice(hq)), c, 1) for c in rng.choice(on_cats, n_online, p=p)]
    m = pd.DataFrame(rows, columns=["location_id", "category", "is_online"])
    m.insert(0, "merchant_id", [f"M{i:05d}" for i in range(len(m))])
    m["name"] = [f"{c.replace('_', ' ').title()} #{i}" for i, c in enumerate(m.category)]
    m["_pop"] = rng.lognormal(0, 1.0, len(m))
    new = rng.random(len(m)) < 0.20
    m["_start_day"] = np.where(new, rng.integers(10, 320, len(m)), -rng.integers(30, 3000, len(m)))
    m["_compromised"] = (rng.random(len(m)) < cfg.compromised_merchant_share) & (m.is_online == 1) | (
        rng.random(len(m)) < cfg.compromised_merchant_share / 4)
    return m


def _pick_merchants(rng, merchants, loc_ids, cats, online, day):
    """Choose a merchant per transaction (vectorised by (location, category, channel) group)."""
    out = np.empty(len(loc_ids), dtype=object)
    m = merchants
    df = pd.DataFrame({"loc": loc_ids, "cat": cats, "on": online, "day": day, "i": np.arange(len(loc_ids))})
    for (l, c, o), g in df.groupby(["loc", "cat", "on"]):
        sel = m[(m.category == c) & (m.is_online == 1)] if o else m[(m.location_id == l) & (m.category == c) & (m.is_online == 0)]
        if len(sel) == 0:  # fallback: any physical merchant in that city
            sel = m[(m.location_id == l) & (m.is_online == 0)]
        for _ in range(4):  # re-draw rows whose merchant was not yet onboarded on that day
            w = sel._pop.values
            picked = sel.iloc[rng.choice(len(sel), len(g), p=w / w.sum())]
            ok = picked._start_day.values <= g.day.values
            out[g.i.values[ok]] = picked.merchant_id.values[ok]
            g = g[~ok]
            if g.empty:
                break
        if not g.empty:  # give up: use the oldest merchant of the group
            out[g.i.values] = sel.sort_values("_start_day").merchant_id.values[0]
    return out


def _legit(rng, cfg: GenConfig, cust, merch, loc) -> pd.DataFrame:
    n_c = len(cust)
    t0 = pd.Timestamp(cfg.start)
    a = np.maximum(cust._start_day.values, 0)
    active = cfg.days - a
    n_tx = rng.poisson(cust._rate.values * active)
    ci = np.repeat(np.arange(n_c), n_tx)
    # day-of-year weights: weekend uplift + holiday-season volume uplift (a deliberate, documented seasonality)
    days = np.arange(cfg.days)
    dow = ((t0.dayofweek + days) % 7)
    w = np.array([1, 1, 1, 1, 1.15, 1.3, 1.1])[dow] * np.where((days >= 325) & (days <= 358), 1.35, 1.0)
    cw = np.concatenate([[0], np.cumsum(w)])
    lo, hi = cw[a][ci], cw[cfg.days]
    day = np.minimum(np.searchsorted(cw, lo + rng.random(len(ci)) * (hi - lo), side="right") - 1, cfg.days - 1)
    day = np.maximum(day, a[ci])
    # hour of day (local), customer-specific mix of "day" and "night owl" profiles
    seg = cust.segment.values
    night = np.array([_SEGMENTS[s][5] for s in seg])
    night = np.clip(night * rng.lognormal(0, 0.4, n_c), 0, 0.9)
    base = _HOUR_BASE / _HOUR_BASE.sum(); nt = _HOUR_NIGHT / _HOUR_NIGHT.sum()
    hp = (1 - night)[:, None] * base[None] + night[:, None] * nt[None]
    hour = _cdf_sample(rng, hp, ci)
    sec = hour * 3600 + rng.integers(0, 3600, len(ci))
    # categories, channel
    share = np.array([v[1] for v in _CATEGORIES.values()])
    pref = share[None] * rng.lognormal(0, 0.6, (n_c, len(share)))
    cat = _cdf_sample(rng, pref, ci)
    on_f = np.array([v[2] for v in _CATEGORIES.values()])[cat]
    on_share = np.array([_SEGMENTS[s][4] for s in seg])[ci]
    online = (rng.random(len(ci)) < on_share * on_f * 1.8).astype(int)
    cat_names = np.array(_CAT_NAMES)[cat]
    # location: home, or trip city
    cur_loc = cust.home_location_id.values[ci].copy()
    trip_p = np.array([_SEGMENTS[s][6] for s in seg])
    order = np.argsort(ci, kind="stable"); starts = np.searchsorted(ci[order], np.arange(n_c))
    ends = np.append(starts[1:], len(ci))
    all_loc = loc.location_id.values; dom_loc = loc[loc.is_domestic == 1].location_id.values
    for c in np.where(rng.random(n_c) < np.minimum(trip_p * 4, 0.9))[0]:
        idx = order[starts[c]:ends[c]]
        for _ in range(rng.integers(1, 4)):
            s = rng.integers(max(a[c], 0), cfg.days - 1); e = s + rng.integers(3, 10)
            dest = rng.choice(all_loc if rng.random() < 0.3 else dom_loc)
            cur_loc[idx[(day[idx] >= s) & (day[idx] <= e)]] = dest
    merch_id = _pick_merchants(rng, merch, cur_loc, cat_names, online, day)
    # shopping-trip bursts (legit hard negatives): 4% of transactions spawn 1-3 follow-ups within 2 hours
    burst = rng.random(len(ci)) < 0.04
    rep = np.where(burst, rng.integers(1, 4, len(ci)), 0)
    bi = np.repeat(np.arange(len(ci)), rep)
    b_cat = rng.choice(["retail", "restaurant", "entertainment", "grocery"], len(bi))
    b_sec = sec[bi] + rng.integers(120, 7200, len(bi))
    b_day = day[bi] + b_sec // 86400
    keep = b_day < cfg.days
    bi, b_cat, b_sec, b_day = bi[keep], b_cat[keep], b_sec[keep], b_day[keep]
    b_merch = _pick_merchants(rng, merch, cur_loc[bi], b_cat, np.zeros(len(bi), int), b_day)
    ci = np.concatenate([ci, ci[bi]]); day = np.concatenate([day, b_day]); sec = np.concatenate([sec, b_sec % 86400])
    cat_names = np.concatenate([cat_names, b_cat]); online = np.concatenate([online, np.zeros(len(bi), int)])
    cur_loc = np.concatenate([cur_loc, cur_loc[bi]]); merch_id = np.concatenate([merch_id, b_merch])
    # amounts
    mu = np.array([_SEGMENTS[s][2] for s in seg])[ci] + np.array([_CATEGORIES[c][0] for c in cat_names])
    sg = np.array([_SEGMENTS[s][3] for s in seg])[ci]
    amt = rng.lognormal(mu, sg * 0.75)
    amt *= np.where(rng.random(len(amt)) < 0.006, rng.lognormal(1.5, 0.5, len(amt)), 1.0)  # legit big purchases
    holiday = (day >= cfg.drift_day + 25) & np.isin(cat_names, ["retail", "electronics", "entertainment", "gift_cards", "travel"])
    amt = np.where(holiday, amt * 1.25, amt)  # covariate drift: holiday spending
    amt = np.clip(amt, 0.5, 8000)
    amt = np.where(cat_names == "atm_cash", np.maximum(np.round(amt / 20) * 20, 20), np.round(amt, 2))
    # devices: online -> device; one device-switch event per customer with prob .25 (legit "new device")
    switch_day = np.where(rng.random(n_c) < 0.25, rng.integers(20, cfg.days, n_c), 10**6)
    dev_k = (day >= switch_day[ci]).astype(int)
    dev = np.where(online == 1, np.array([f"D{c:05d}_{k}" for c, k in zip(ci, dev_k)], dtype=object), None)
    channel = np.where(cat_names == "atm_cash", "atm", np.where(online == 1, "online", "card_present"))
    utc_off = loc.set_index("location_id").utc_offset
    ts = t0 + pd.to_timedelta(day, unit="D") + pd.to_timedelta(sec, unit="s") - pd.to_timedelta(utc_off.reindex(cur_loc).values, unit="h")
    # account: customer has 1-3 spend accounts; pick one
    return pd.DataFrame({"_ci": ci, "ts": ts, "amount": amt, "merchant_id": merch_id, "location_id": cur_loc,
                         "channel": channel, "device_id": dev, "category": cat_names})


def _accounts(rng, cfg, cust) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp(cfg.start)
    for i, (cid, sd) in enumerate(zip(cust.customer_id.values, cust._start_day.values)):
        k = rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15])
        types = ["debit", "credit_card", "debit"][:k] if k < 3 else ["debit", "credit_card", "credit_card"]
        for j, ty in enumerate(types):
            opened = t0 + pd.Timedelta(days=int(sd) - (0 if j == 0 else int(rng.integers(0, 400))))
            rows.append((cid, ty, opened, float(rng.choice([2000, 5000, 10000, 20000])) if ty == "credit_card" else None, i))
    a = pd.DataFrame(rows, columns=["customer_id", "account_type", "opened_at", "credit_limit", "_ci"])
    a.insert(0, "account_id", [f"A{i:06d}" for i in range(len(a))])
    return a


def generate(cfg: GenConfig = GEN) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    loc = locations()
    cust = _customers(rng, cfg, loc)
    merch = _merchants(rng, cfg, loc)
    acc = _accounts(rng, cfg, cust)
    legit = _legit(rng, cfg, cust, merch, loc)
    legit["is_fraud_true"] = 0; legit["pattern"] = "legit"

    # account for each legit txn
    acc_by_c = acc.groupby("_ci").account_id.apply(list).to_dict()
    legit["account_id"] = [acc_by_c[c][rng.integers(len(acc_by_c[c]))] for c in legit._ci.values]
    fraud = _inject_fraud(rng, cfg, cust, merch, loc, acc, legit)
    tx = pd.concat([legit, fraud], ignore_index=True)
    tx["customer_id"] = cust.customer_id.values[tx._ci.values]
    tx = tx[(tx.ts >= pd.Timestamp(cfg.start)) & (tx.ts < pd.Timestamp(cfg.start) + pd.Timedelta(days=cfg.days))]
    tx = tx.sort_values(["ts", "customer_id"], kind="stable").reset_index(drop=True)
    tx.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(tx))])
    # label noise (documented): undiscovered fraud and false disputes
    is_f = tx.is_fraud_true.values.astype(bool)
    missed = is_f & (rng.random(len(tx)) < cfg.label_noise_missed_fraud)
    false_d = (~is_f) & (rng.random(len(tx)) < cfg.label_noise_false_dispute)
    tx["is_fraud"] = ((is_f & ~missed) | false_d).astype(int)
    tx["pattern"] = np.where(false_d, "false_dispute", tx.pattern)
    off = loc.set_index("location_id").utc_offset.reindex(tx.location_id).values
    local = tx.ts + pd.to_timedelta(off, unit="h")
    tx["hour_local"] = local.dt.hour.values
    tx["dow_local"] = local.dt.dayofweek.values  # 0 = Monday
    tx["txn_date"] = tx.ts.dt.strftime("%Y-%m-%d")
    tx["ts_epoch"] = (tx.ts - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64")
    tx["amount"] = tx.amount.round(2)

    truth = tx[["txn_id", "pattern", "is_fraud_true"]].rename(columns={"pattern": "true_pattern"})
    truth_m = merch[["merchant_id", "_compromised"]].rename(columns={"_compromised": "is_compromised"})
    transactions = tx[["txn_id", "account_id", "customer_id", "merchant_id", "location_id", "ts", "txn_date", "hour_local",
                       "dow_local", "ts_epoch", "amount", "channel", "device_id", "is_fraud"]].copy()
    first = transactions.groupby("merchant_id").ts.min().dt.normalize()
    merch["onboarded_at"] = [min(pd.Timestamp(cfg.start) + pd.Timedelta(days=int(d)), first.get(m, pd.Timestamp(cfg.start) + pd.Timedelta(days=int(d))))
                             for m, d in zip(merch.merchant_id, merch._start_day)]
    customers = cust[["customer_id", "segment", "age", "home_location_id", "signup_date", "kyc_risk_rating"]]
    accounts = acc[["account_id", "customer_id", "account_type", "opened_at", "credit_limit"]]
    merchants = merch[["merchant_id", "name", "category", "location_id", "is_online", "onboarded_at"]]
    locs = loc[["location_id", "city", "country", "lat", "lon", "utc_offset", "is_domestic"]]
    return {"locations": locs, "customers": customers, "accounts": accounts, "merchants": merchants,
            "transactions": transactions, "synthetic_truth": truth, "synthetic_truth_merchants": truth_m}


# ----------------------------------------------------------------------------------------------- fraud
def _inject_fraud(rng, cfg, cust, merch, loc, acc, legit) -> pd.DataFrame:
    t0 = pd.Timestamp(cfg.start)
    warm = 30
    n_c = len(cust)
    L = loc.set_index("location_id")
    med = legit.groupby("_ci").amount.median().reindex(range(n_c)).fillna(25.0).values
    acc_by_c = acc.groupby("_ci").account_id.apply(list).to_dict()
    idx_by_c = legit.groupby("_ci").indices
    cust_start = np.maximum(cust._start_day.values, 0)
    comp_w = np.where(merch._compromised, 8.0, 1.0)
    merch = merch.assign(_w=merch._pop * comp_w)
    online_m = merch[merch.is_online == 1]
    phys_m = merch[merch.is_online == 0]
    by_cat_online = {c: g for c, g in online_m.groupby("category")}
    phys_by_loc = {l: g for l, g in phys_m.groupby("location_id")}
    out: list[dict] = []

    def victim(day_min=warm, day_max=cfg.days - 1):
        for _ in range(50):
            c = int(rng.integers(n_c))
            lo = max(day_min, cust_start[c] + 10)
            if lo < day_max and c in acc_by_c and len(idx_by_c.get(c, [])) >= 3:
                return c, int(rng.integers(lo, day_max))
        raise RuntimeError("no victim")

    def far_city(home_id, foreign_p):
        cands = loc[(loc.is_domestic == 0) if rng.random() < foreign_p else (loc.is_domestic == 1)]
        cands = cands[cands.location_id != home_id]
        return int(rng.choice(cands.location_id.values))

    def pick_online(cat_w: dict):
        cats = list(cat_w); p = np.array(list(cat_w.values())); p = p / p.sum()
        cat = rng.choice(cats, p=p)
        g = by_cat_online.get(cat, online_m)
        w = g._w.values
        return g.merchant_id.values[rng.choice(len(g), p=w / w.sum())], cat

    def emit(c, ts, amount, merchant, location, channel, device, pattern, category):
        out.append(dict(_ci=c, ts=ts, amount=float(np.clip(amount, 0.5, 9000)), merchant_id=merchant, location_id=int(location),
                        channel=channel, device_id=device, category=category, is_fraud_true=1, pattern=pattern,
                        account_id=acc_by_c[c][rng.integers(len(acc_by_c[c]))]))

    ato_cats = {"electronics": .3, "gift_cards": .2, "crypto_exchange": .15, "travel": .1, "retail": .15, "online_services": .1}
    ord_cats = {"online_services": .3, "retail": .3, "entertainment": .2, "electronics": .1, "travel": .1}
    ev = 0
    for _ in range(cfg.n_ato):
        c, d = victim(); ev += 1
        subtle = rng.random() < 0.30
        n = int(rng.integers(1, 4) if subtle else rng.integers(3, 13))
        t = t0 + pd.Timedelta(days=d, seconds=int(rng.integers(0, 86400)))
        city = far_city(cust.home_location_id.values[c], 0.0 if subtle else 0.6)
        for k in range(n):
            t = t + pd.Timedelta(minutes=float(rng.exponential(25 if subtle else 8)))
            m, cat = pick_online(ord_cats if subtle else ato_cats)
            a = med[c] * rng.lognormal(0.3 if subtle else 1.2, 0.5)
            emit(c, t, a, m, city, "online", f"F{ev:05d}", "account_takeover", cat)

    for _ in range(cfg.n_card_testing):
        c, d = victim(); ev += 1
        t = t0 + pd.Timedelta(days=d, seconds=int(rng.integers(0, 86400)))
        city = far_city(cust.home_location_id.values[c], 0.5)
        n_small = int(rng.integers(6, 26))
        offs = np.sort(rng.uniform(0, rng.uniform(3, 20), n_small))
        for o in offs:
            m, cat = pick_online({"online_services": .5, "entertainment": .3, "retail": .2})
            emit(c, t + pd.Timedelta(minutes=float(o)), rng.uniform(0.5, 4.99), m, city, "online", f"F{ev:05d}", "card_testing", cat)
        for j in range(int(rng.integers(1, 3))):
            m, cat = pick_online({"electronics": .5, "gift_cards": .3, "travel": .2})
            emit(c, t + pd.Timedelta(minutes=float(offs[-1] + rng.uniform(5, 30))), rng.uniform(150, 900), m, city, "online", f"F{ev:05d}", "card_testing", cat)

    cp = legit[legit.channel == "card_present"]
    cp_idx = cp.groupby("_ci").indices
    cp_vals = cp.reset_index(drop=True)
    for _ in range(cfg.n_stolen_card):
        for _try in range(30):
            c, d = victim()
            if c in cp_idx and len(cp_idx[c]) > 0:
                break
        else:
            continue
        anchor = cp_vals.iloc[int(rng.choice(cp_idx[c]))]
        t = anchor.ts + pd.Timedelta(minutes=float(rng.uniform(20, 300)))
        h = L.loc[anchor.location_id]
        far = loc[(np.hypot(loc.lat - h.lat, loc.lon - h.lon) > 8)]
        city = int(rng.choice(far.location_id.values))
        g = phys_by_loc[city]
        for k in range(int(rng.integers(1, 5))):
            t = t + pd.Timedelta(minutes=float(rng.uniform(5, 60)))
            row = g.iloc[rng.choice(len(g), p=g._w.values / g._w.sum())]
            ch = "atm" if row.category == "atm_cash" else "card_present"
            a = med[c] * rng.lognormal(1.0, 0.6)
            emit(c, t, np.round(a / 20) * 20 if ch == "atm" else a, row.merchant_id, city, ch, None, "stolen_card", row.category)

    comp_ids = set(merch.merchant_id[merch._compromised])
    comp_tx = legit[legit.merchant_id.isin(comp_ids)]
    if len(comp_tx):
        pick = comp_tx.sample(min(cfg.n_low_and_slow, len(comp_tx)), random_state=int(rng.integers(1 << 30)))
        for r in pick.rename(columns={'_ci': 'ci'}).itertuples():
            t = r.ts + pd.Timedelta(days=float(rng.uniform(7, 28)))
            c = int(r.ci)
            if c not in acc_by_c:
                continue
            ev += 1
            dev = f"F{ev:05d}" if rng.random() < 0.6 else f"D{c:05d}_0"
            city = int(cust.home_location_id.values[c]) if rng.random() < 0.6 else far_city(cust.home_location_id.values[c], 0.2)
            for k in range(int(rng.integers(1, 3))):
                m, cat = pick_online(ord_cats)
                emit(c, t + pd.Timedelta(minutes=float(k * rng.uniform(10, 200))), med[c] * rng.lognormal(0.3, 0.5), m, city, "online", dev, "low_and_slow", cat)

    for _ in range(cfg.n_gift_card_cashout):  # NEW pattern after the drift day (simulates concept drift)
        c, d = victim(cfg.drift_day, cfg.days - 1); ev += 1
        t = t0 + pd.Timedelta(days=d, seconds=int(rng.integers(0, 86400)))
        for k in range(int(rng.integers(2, 5))):
            t = t + pd.Timedelta(hours=float(rng.uniform(1, 6)))
            m, cat = pick_online({"gift_cards": .5, "crypto_exchange": .2, "online_services": .3})
            emit(c, t, rng.uniform(100, 400), m, cust.home_location_id.values[c], "online", f"F{ev:05d}", "gift_card_cashout", cat)
    return pd.DataFrame(out)
