"""Model evaluation: ranking metrics, operating-point metrics, confusion matrix, false-positive analysis,
business cost, review workload and block bootstraps."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import COST, CostConfig


def ranking_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "base_rate": float(y.mean())}
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)), "base_rate": float(y.mean())}


def confusion(y: np.ndarray, alert: np.ndarray) -> dict:
    y, a = np.asarray(y).astype(bool), np.asarray(alert).astype(bool)
    tp, fp, fn, tn = int((y & a).sum()), int((~y & a).sum()), int((y & ~a).sum()), int((~y & ~a).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0, "alert_rate": (tp + fp) / len(y),
            "false_discovery_rate": fp / (tp + fp) if tp + fp else 0.0}


def business_cost(y: np.ndarray, alert: np.ndarray, amount: np.ndarray, cost: CostConfig = COST) -> dict:
    """Expected cost of an alerting policy.

    caught fraud  -> review cost only (loss prevented)
    false alert   -> review cost + customer-friction cost
    missed fraud  -> loss_given_missed * amount + fixed chargeback fee
    """
    y, a, amt = np.asarray(y).astype(bool), np.asarray(alert).astype(bool), np.asarray(amount, float)
    review = cost.review_cost * a.sum()
    friction = cost.friction_cost * (a & ~y).sum()
    fraud_loss = (cost.loss_given_missed * amt[y & ~a]).sum() + cost.fraud_fixed_cost * (y & ~a).sum()
    prevented = amt[y & a].sum() * cost.loss_given_missed
    return {"review_cost": float(review), "friction_cost": float(friction), "missed_fraud_cost": float(fraud_loss),
            "total_cost": float(review + friction + fraud_loss), "fraud_prevented": float(prevented)}


def pick_threshold_by_cost(y, s, amount, cost: CostConfig = COST, n_grid: int = 400) -> tuple[float, float]:
    """Cost-minimising score threshold on a (validation) set."""
    y, s, amount = np.asarray(y).astype(bool), np.asarray(s, float), np.asarray(amount, float)
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.9999, n_grid)))
    best = (np.inf, grid[-1])
    no_alert = business_cost(y, np.zeros_like(y), amount, cost)["total_cost"]
    for t in grid:
        c = business_cost(y, s >= t, amount, cost)["total_cost"]
        if c < best[0]:
            best = (c, t)
    return (float(best[1]), float(best[0])) if best[0] < no_alert else (float("inf"), float(no_alert))


def pick_threshold_precision(y, s, min_precision: float = 0.9, min_alerts: int = 20) -> float:
    prec, rec, thr = precision_recall_curve(y, s)
    ok = np.where((prec[:-1] >= min_precision) & (np.arange(len(thr)) >= 0))[0]
    for i in ok:  # lowest threshold that keeps precision >= target with enough alerts
        if (np.asarray(s) >= thr[i]).sum() >= min_alerts:
            return float(thr[i])
    return float("inf")


def workload(n_alerts: int, n_days: float, cost: CostConfig = COST) -> dict:
    per_day = n_alerts / max(n_days, 1e-9)
    hours = per_day * cost.review_minutes_per_alert / 60
    return {"alerts_per_day": per_day, "review_hours_per_day": hours, "analysts_needed": hours / cost.analyst_hours_per_day}


def at_budget(y, s, k: int) -> dict:
    """Precision/recall when only the top-k scored transactions are reviewed (review-capacity view)."""
    y, s = np.asarray(y).astype(bool), np.asarray(s)
    k = int(min(max(k, 1), len(s)))
    top = np.argpartition(-s, k - 1)[:k]
    tp = int(y[top].sum())
    return {"k": k, "precision": tp / k, "recall": tp / max(int(y.sum()), 1)}


def false_positive_analysis(df: pd.DataFrame, alert: np.ndarray) -> dict:
    """Where do false alerts concentrate? ``df`` needs: is_fraud, channel, category, amount, cust_prior_n, segment, customer_id."""
    d = df.assign(alert=np.asarray(alert).astype(bool))
    legit, fp = d[d.is_fraud == 0], d[(d.is_fraud == 0) & d.alert]

    def rate(by):
        t = legit.groupby(by, observed=True).size().rename("legit_txns").to_frame()
        t["false_positives"] = fp.groupby(by, observed=True).size()
        t = t.fillna(0)
        t["fp_rate_pct"] = 100 * t.false_positives / t.legit_txns
        t["share_of_fps_pct"] = 100 * t.false_positives / max(len(fp), 1)
        return t.reset_index().round(3).to_dict("records")

    amt_bucket = pd.cut(d.amount, [0, 25, 100, 500, 1e9], labels=["<25", "25-100", "100-500", "500+"])
    hist_bucket = pd.cut(d.cust_prior_n.fillna(0), [-1, 9, 49, 1e9], labels=["new (<10 txns)", "established (10-49)", "mature (50+)"])
    per_cust = fp.groupby("customer_id").size()
    out = {
        "n_false_positives": int(len(fp)), "n_legit": int(len(legit)),
        "by_channel": rate(legit.channel), "by_category": rate(legit.category),
        "by_segment": rate(legit.segment), "by_amount_bucket": rate(amt_bucket[legit.index]),
        "by_customer_history": rate(hist_bucket[legit.index]),
        "customers_with_fp": int(len(per_cust)),
        "pct_fps_from_customers_with_3plus_fps": float(100 * per_cust[per_cust >= 3].sum() / max(len(fp), 1)),
        "top_fp_customers": per_cust.sort_values(ascending=False).head(5).to_dict(),
    }
    if "rule_hits" in d:
        out["top_fp_rule_hits"] = fp.rule_hits.str.split("|").explode().replace("", np.nan).value_counts().head(8).to_dict()
    return out


def block_bootstrap_pr_auc(y, s, block: np.ndarray, n_boot: int = 200, seed: int = 0) -> dict:
    """Day-block bootstrap CI for PR-AUC (resampling whole days preserves within-day dependence)."""
    rng = np.random.default_rng(seed)
    y, s, block = np.asarray(y), np.asarray(s), np.asarray(block)
    ub = np.unique(block)
    idx_by = {b: np.where(block == b)[0] for b in ub}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ub, len(ub), replace=True)
        ii = np.concatenate([idx_by[b] for b in pick])
        if y[ii].sum() > 0:
            vals.append(average_precision_score(y[ii], s[ii]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"pr_auc_ci95": [float(lo), float(hi)], "n_boot": n_boot}
