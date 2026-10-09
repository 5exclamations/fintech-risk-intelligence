"""Streamlit analytics dashboard.  streamlit run dashboard/app.py   (reads files written by the pipeline)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from riskplatform.config import ARTIFACT_DIR, COST  # noqa: E402
from riskplatform.evaluation import business_cost, confusion, workload  # noqa: E402

st.set_page_config(page_title="Transaction Risk Intelligence", layout="wide", page_icon="🛡️")


@st.cache_data
def load():
    m = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    sc = pd.read_parquet(ARTIFACT_DIR / "scored" / "scored.parquet")
    mon = {k: pd.read_csv(ARTIFACT_DIR / "monitoring" / f"{k}.csv") for k in ["drift_windows", "performance_windows", "feature_psi_latest"]}
    summ = json.loads((ARTIFACT_DIR / "monitoring" / "summary.json").read_text())
    an = {p.stem: pd.read_csv(p) for p in sorted((ARTIFACT_DIR / "analytics").glob("*.csv"))} if (ARTIFACT_DIR / "analytics").exists() else {}
    return m, sc, mon, summ, an


if not (ARTIFACT_DIR / "metrics.json").exists():
    st.error("No artifacts found. Run `python -m riskplatform.pipeline all` first.")
    st.stop()
M, SC, MON, SUMM, AN = load()
TEST = SC[SC.split == "test"].copy()
thr = M["model"]["thresholds"]

st.title("🛡️ Financial Transaction Risk Intelligence")
st.warning("**SYNTHETIC DATA.** Every number below comes from a simulated dataset with invented fraud patterns. "
           "It demonstrates the methodology and must not be read as real-world performance.", icon="⚠️")

with st.sidebar:
    st.header("Operating point")
    t_review = st.slider("Review threshold (calibrated P(fraud))", 0.005, 0.9, float(min(max(thr["review"], 0.005), 0.9)), 0.005)
    st.caption(f"Cost-optimal on validation: {thr['review']:.3f}")
    st.header("Cost assumptions")
    review_cost = st.number_input("Review cost / alert ($)", 0.0, 100.0, COST.review_cost)
    friction = st.number_input("Customer friction / false alert ($)", 0.0, 100.0, COST.friction_cost)
    fixed = st.number_input("Fixed cost / missed fraud ($)", 0.0, 500.0, COST.fraud_fixed_cost)
    mins = st.number_input("Minutes per review", 1.0, 60.0, COST.review_minutes_per_alert)
cost_cfg = type(COST)(review_cost=review_cost, friction_cost=friction, fraud_fixed_cost=fixed, review_minutes_per_alert=mins)
y, p, amt = TEST.is_fraud.values, TEST.p_model.values, TEST.amount.values
alert = p >= t_review
cm = confusion(y, alert)
cost = business_cost(y, alert, amt, cost_cfg)
base = business_cost(y, np.zeros(len(y), bool), amt, cost_cfg)
days = TEST.day.nunique()
wl = workload(int(alert.sum()), days, cost_cfg)

tabs = st.tabs(["Overview", "Risk distribution", "Suspicious transactions", "Fraud trends", "Model performance",
                "False positives", "Workload & cost", "Monitoring", "SQL analytics"])

with tabs[0]:
    c = st.columns(5)
    c[0].metric("Precision", f"{cm['precision']:.1%}"); c[1].metric("Recall", f"{cm['recall']:.1%}")
    c[2].metric("F1", f"{cm['f1']:.2f}"); c[3].metric("PR-AUC", f"{M['methods_test']['Gradient boosting (calibrated)']['pr_auc']:.3f}")
    c[4].metric("False-positive rate", f"{cm['false_positive_rate']:.3%}")
    c = st.columns(4)
    c[0].metric("Alerts / day", f"{wl['alerts_per_day']:.0f}"); c[1].metric("Analysts needed", f"{wl['analysts_needed']:.1f}")
    c[2].metric("Total cost vs no model", f"${cost['total_cost']:,.0f}", f"{cost['total_cost'] - base['total_cost']:+,.0f}", delta_color="inverse")
    c[3].metric("Fraud $ prevented", f"${cost['fraud_prevented']:,.0f}")
    st.caption(f"Test window: days {M['splits']['test']['days'][0]}–{M['splits']['test']['days'][1]} · {len(TEST):,} transactions · "
               f"label fraud rate {y.mean():.2%} · model: {M['model']['name']} (selected on validation PR-AUC)")

with tabs[1]:
    TEST["label"] = np.where(TEST.is_fraud == 1, "fraud-labelled", "legitimate")
    fig = px.histogram(TEST, x="p_model", color="label", nbins=60, log_y=True, barmode="overlay", opacity=0.7,
                       labels={"p_model": "calibrated risk score"}, title="Risk score distribution (log count)")
    fig.add_vline(x=t_review, line_dash="dash", annotation_text="review"); st.plotly_chart(fig, use_container_width=True)
    b = TEST.assign(band=np.select([p >= thr["block"], p >= t_review], ["block", "review"], "allow")).band.value_counts()
    st.plotly_chart(px.bar(b.reset_index(), x="band", y="count", title="Decision bands (block threshold from validation)"), use_container_width=True)

with tabs[2]:
    f1, f2, f3 = st.columns(3)
    ch = f1.multiselect("Channel", sorted(TEST.channel.unique()), sorted(TEST.channel.unique()))
    cat = f2.multiselect("Category", sorted(TEST.category.dropna().unique()))
    top_n = f3.slider("Rows", 10, 500, 100)
    s = TEST[alert & TEST.channel.isin(ch)]
    if cat:
        s = s[s.category.isin(cat)]
    s = s.sort_values("p_model", ascending=False).head(top_n)
    s = s.assign(reason_codes=s.reasons.map(lambda r: ", ".join(x["code"] for x in json.loads(r)) if r else ""))
    st.dataframe(s[["txn_id", "ts", "customer_id", "merchant_id", "category", "channel", "amount", "p_model", "band", "reason_codes", "is_fraud"]]
                 .rename(columns={"is_fraud": "label_is_fraud"}), use_container_width=True, hide_index=True)
    if len(s):
        pick = st.selectbox("Explain transaction", s.txn_id)
        for r in json.loads(s[s.txn_id == pick].reasons.iloc[0] or "[]"):
            st.write(f"**{r['code']}** ({r['source']}) — {r['detail']}")

with tabs[3]:
    d = SC.groupby("day").agg(n=("txn_id", "size"), fraud=("is_fraud", "sum"), alerts=("p_model", lambda x: (x >= t_review).sum())).reset_index()
    d["fraud_rate_7d"] = d.fraud.rolling(7).sum() / d.n.rolling(7).sum()
    st.plotly_chart(px.line(d, x="day", y="fraud_rate_7d", title="Fraud-labelled rate (7-day rolling), validation + test"), use_container_width=True)
    st.plotly_chart(px.bar(d, x="day", y=["fraud", "alerts"], barmode="group", title="Daily fraud labels vs alerts"), use_container_width=True)
    by = SC.groupby("category").agg(n=("txn_id", "size"), rate=("is_fraud", "mean")).reset_index().sort_values("rate", ascending=False)
    st.plotly_chart(px.bar(by, x="category", y="rate", title="Fraud-label rate by merchant category"), use_container_width=True)
    pat = TEST[TEST.is_fraud == 1].true_pattern.value_counts().reset_index()
    st.caption("Synthetic-only view: generator ground-truth pattern of fraud-labelled test transactions.")
    st.plotly_chart(px.pie(pat, names="true_pattern", values="count"), use_container_width=True)

with tabs[4]:
    mt = pd.DataFrame(M["methods_test"]).T[["pr_auc", "roc_auc", "precision_at_budget", "recall_at_budget"]]
    st.subheader("Detection methods on the held-out test window"); st.dataframe(mt.style.format("{:.3f}"), use_container_width=True)
    op = M["test_operating_point"]
    st.caption(f"PR-AUC 95% day-block bootstrap CI: {op['pr_auc_ci95'][0]:.3f}–{op['pr_auc_ci95'][1]:.3f}")
    c1, c2 = st.columns(2)
    pr, rc, _ = precision_recall_curve(y, p)
    fig = go.Figure(go.Scatter(x=rc, y=pr, mode="lines", name="model"))
    fig.update_layout(title="Precision–recall curve (test)", xaxis_title="recall", yaxis_title="precision"); c1.plotly_chart(fig, use_container_width=True)
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    c2.plotly_chart(px.imshow(z, text_auto=True, x=["no alert", "alert"], y=["legit", "fraud"], title="Confusion matrix at selected threshold",
                              color_continuous_scale="Blues"), use_container_width=True)
    cal = pd.DataFrame(M["calibration"])
    fig = px.scatter(cal, x="mean_pred", y="observed", title="Calibration (test, 8 equal-size score bins)")
    fig.add_shape(type="line", x0=0, y0=0, x1=cal.mean_pred.max(), y1=cal.mean_pred.max(), line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.bar(pd.DataFrame(M["by_30d_block"]), x="first_day", y=["pr_auc", "precision", "recall"], barmode="group",
                           title="Performance by 30-day block (new fraud pattern appears at drift day)"), use_container_width=True)
    st.subheader("Recall by fraud pattern (synthetic-only diagnostic)")
    st.dataframe(pd.DataFrame(M["synthetic_only_diagnostics"]["recall_by_true_pattern"]).T, use_container_width=True)

with tabs[5]:
    fp = M["false_positive_analysis"]
    st.metric("False positives at the *validated* review threshold", fp["n_false_positives"])
    st.caption("Breakdowns below use the cost-optimal threshold from validation (not the slider).")
    for k, title in [("by_channel", "Channel"), ("by_category", "Category"), ("by_amount_bucket", "Amount"), ("by_customer_history", "Customer history"), ("by_segment", "Segment")]:
        df = pd.DataFrame(fp[k]); st.markdown(f"**False-positive rate by {title.lower()}**")
        st.plotly_chart(px.bar(df, x=df.columns[0], y="fp_rate_pct", hover_data=["false_positives", "legit_txns", "share_of_fps_pct"]), use_container_width=True)
    st.write("Top rule hits among false positives:", fp.get("top_fp_rule_hits", {}))
    st.write(f"{fp['pct_fps_from_customers_with_3plus_fps']:.1f}% of false positives come from customers flagged 3+ times (friction concentration).")
    st.info(f"Synthetic-only: {M['synthetic_only_diagnostics']['share_of_false_positives_that_are_undiscovered_fraud']:.1%} of 'false positives' are "
            "actually fraud the label process never discovered.")

with tabs[6]:
    c = st.columns(3)
    c[0].metric("Alerts per day", f"{wl['alerts_per_day']:.0f}"); c[1].metric("Review hours / day", f"{wl['review_hours_per_day']:.1f}")
    c[2].metric("Analysts (6.5 h/day)", f"{wl['analysts_needed']:.1f}")
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.9995, 120)))
    rows = [{"threshold": t, **{k: v for k, v in business_cost(y, p >= t, amt, cost_cfg).items() if k == "total_cost"},
             "analysts": workload(int((p >= t).sum()), days, cost_cfg)["analysts_needed"], "recall": confusion(y, p >= t)["recall"]} for t in grid]
    cdf = pd.DataFrame(rows)
    fig = px.line(cdf, x="threshold", y="total_cost", title="Total business cost vs review threshold"); fig.add_vline(x=t_review, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.line(cdf, x="analysts", y="recall", title="Recall achievable for a given review team size"), use_container_width=True)
    st.table(pd.DataFrame({"": ["No model", "Model @ selected threshold"], "Total cost": [base["total_cost"], cost["total_cost"]],
                           "Missed-fraud cost": [base["missed_fraud_cost"], cost["missed_fraud_cost"]],
                           "Review + friction": [0, cost["review_cost"] + cost["friction_cost"]]}).set_index("").style.format("${:,.0f}"))

with tabs[7]:
    st.metric("Monitoring status", SUMM["status"].upper())
    for f in SUMM["findings"]:
        st.write("• " + f)
    dw = MON["drift_windows"]
    st.plotly_chart(px.line(dw, x="start_day", y=["max_feature_psi", "score_psi"], markers=True, title="Drift per 14-day window (PSI; 0.10 warn, 0.25 alert)")
                    .add_hline(y=0.1, line_dash="dot").add_hline(y=0.25, line_dash="dash"), use_container_width=True)
    pw = MON["performance_windows"].dropna(subset=["pr_auc"])
    st.plotly_chart(px.line(pw, x="start_day", y=["pr_auc", "precision", "recall"], markers=True, title="Performance on windows with mature labels"), use_container_width=True)
    st.plotly_chart(px.bar(dw, x="start_day", y="alert_rate", title="Alert rate per window at validated threshold"), use_container_width=True)
    st.dataframe(MON["feature_psi_latest"].head(15), use_container_width=True, hide_index=True)

with tabs[8]:
    if not AN:
        st.info("Run the pipeline to generate SQL analytics outputs.")
    for name, df in AN.items():
        with st.expander(name, expanded=False):
            st.code((Path(__file__).resolve().parents[1] / "sql" / "analytics" / f"{name}.sql").read_text(), language="sql")
            st.dataframe(df, use_container_width=True, hide_index=True)
