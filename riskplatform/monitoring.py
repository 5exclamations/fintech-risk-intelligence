"""Monitoring: input-distribution drift (PSI), score drift and delayed-label performance tracking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Features that grow monotonically with calendar time by construction (history accumulates), so PSI against an
# earlier reference would fire permanently. They are excluded from drift tests (documented in docs/monitoring.md).
MONOTONIC_FEATURES = ["cust_prior_n", "merch_prior_n", "account_age_days", "merch_age_days", "device_prior_cnt",
                      "cust_prior_mean", "cust_prior_max", "merch_prior_mean", "merch_fraud_rate_matured"]
PSI_WARN, PSI_ALERT = 0.10, 0.25   # conventional rule-of-thumb thresholds
PERF_REL_DROP_WARN, PERF_REL_DROP_ALERT = 0.15, 0.30


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with reference-quantile bins and a separate NaN bin."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    rn, cn = np.isnan(ref), np.isnan(cur)
    rv, cv = ref[~rn], cur[~cn]
    if len(rv) == 0 or len(cv) == 0:
        return float("nan")
    edges = np.unique(np.quantile(rv, np.linspace(0, 1, bins + 1)[1:-1]))
    rc = np.bincount(np.searchsorted(edges, rv, side="right"), minlength=len(edges) + 1).astype(float)
    cc = np.bincount(np.searchsorted(edges, cv, side="right"), minlength=len(edges) + 1).astype(float)
    rc = np.append(rc, rn.sum()); cc = np.append(cc, cn.sum())
    eps = 1e-4
    rp, cp = np.maximum(rc / rc.sum(), eps), np.maximum(cc / cc.sum(), eps)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def status(v: float) -> str:
    return "alert" if v >= PSI_ALERT else "warn" if v >= PSI_WARN else "ok"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(ref[f].values, cur[f].values)} for f in features]
    out = pd.DataFrame(rows)
    out["status"] = out.psi.map(status)
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def windowed(ev: pd.DataFrame, ref: pd.DataFrame, features: list[str], p_ref: np.ndarray, window_days: int = 14,
             threshold: float = 0.1, label_delay_days: int = 14, as_of_day: int | None = None) -> dict:
    """Drift + performance per ``window_days`` block of ``ev`` (needs: day, p_model, is_fraud, features).

    Performance is only reported for windows whose labels are fully mature at ``as_of_day`` (default: last day in ev) -
    this mirrors production, where chargebacks arrive weeks late and input drift is the *early* warning signal.
    """
    as_of = int(ev.day.max()) if as_of_day is None else as_of_day
    d0 = int(ev.day.min())
    ev = ev.assign(win=(ev.day - d0) // window_days)
    drift_rows, perf_rows = [], []
    features = [f for f in features if f not in MONOTONIC_FEATURES]
    for w, g in ev.groupby("win"):
        if g.day.nunique() < 0.7 * window_days:  # partial window (e.g. the label-maturity gap): too noisy to test
            continue
        start, end = d0 + int(w) * window_days, d0 + int(w + 1) * window_days - 1
        fd = feature_drift(ref, g, features)
        score_psi = psi(p_ref, g.p_model.values)
        top = fd.head(3)
        drift_rows.append({"window": int(w), "start_day": start, "end_day": end, "n_txn": len(g), "score_psi": score_psi,
                           "score_status": status(score_psi), "max_feature_psi": float(fd.psi.max()),
                           "n_features_warn": int((fd.psi >= PSI_WARN).sum()), "n_features_alert": int((fd.psi >= PSI_ALERT).sum()),
                           "top_drifting_features": ", ".join(f"{r.feature} ({r.psi:.2f})" for r in top.itertuples()),
                           "alert_rate": float((g.p_model >= threshold).mean()), "mean_score": float(g.p_model.mean())})
        matured = end + label_delay_days <= as_of
        row = {"window": int(w), "start_day": start, "end_day": end, "labels_mature": bool(matured), "fraud_rate": float(g.is_fraud.mean())}
        if matured and g.is_fraud.sum() >= 5:
            y, a = g.is_fraud.values.astype(bool), (g.p_model.values >= threshold)
            tp = int((y & a).sum())
            row.update(pr_auc=float(average_precision_score(y, g.p_model)), precision=tp / max(a.sum(), 1), recall=tp / max(y.sum(), 1))
        perf_rows.append(row)
    return {"drift": pd.DataFrame(drift_rows), "performance": pd.DataFrame(perf_rows)}


def summarize(drift: pd.DataFrame, perf: pd.DataFrame, baseline_pr_auc: float, baseline_recall: float) -> dict:
    """Overall traffic-light status plus human-readable findings."""
    findings, level = [], "ok"

    def bump(l):
        nonlocal level
        level = max(level, l, key=["ok", "warn", "alert"].index)

    last = drift.iloc[-1]
    if last.n_features_alert:
        findings.append(f"{int(last.n_features_alert)} feature(s) with PSI >= {PSI_ALERT} in latest window: {last.top_drifting_features}")
        bump("alert")
    elif last.n_features_warn:
        findings.append(f"{int(last.n_features_warn)} feature(s) with PSI >= {PSI_WARN} in latest window: {last.top_drifting_features}")
        bump("warn")
    if last.score_status != "ok":
        findings.append(f"model-score distribution shifted (PSI {last.score_psi:.2f})")
        bump(last.score_status)
    m = perf[perf.labels_mature & perf.pr_auc.notna()]
    if len(m):
        cur = m.iloc[-1]
        drop = 1 - cur.pr_auc / baseline_pr_auc if baseline_pr_auc else 0
        rdrop = 1 - cur.recall / baseline_recall if baseline_recall else 0
        if max(drop, rdrop) >= PERF_REL_DROP_ALERT:
            findings.append(f"performance down vs validation baseline (PR-AUC {cur.pr_auc:.2f} vs {baseline_pr_auc:.2f}; recall {cur.recall:.2f} vs {baseline_recall:.2f})")
            bump("alert")
        elif max(drop, rdrop) >= PERF_REL_DROP_WARN:
            findings.append(f"performance slipping vs validation baseline (PR-AUC {cur.pr_auc:.2f} vs {baseline_pr_auc:.2f})")
            bump("warn")
    if not findings:
        findings.append("no material drift or performance degradation detected")
    return {"status": level, "findings": findings, "latest_window": int(last.window)}
