"""Render docs/architecture.png (dev-only; needs matplotlib)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 7.5)); ax.set_xlim(0, 14); ax.set_ylim(0, 7.5); ax.axis("off")
C = {"data": "#dbeafe", "ml": "#dcfce7", "serve": "#fef9c3", "ops": "#fee2e2"}


def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=C[c], ec="#334155", lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)


def arrow(a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3))


ax.text(7, 7.2, "Financial Transaction Risk Intelligence Platform  (all data synthetic)", ha="center", fontsize=13, weight="bold")
box(0.3, 5.2, 2.6, 1.2, "Synthetic generator\n(customers, accounts, merchants,\nlocations, transactions, fraud typologies)", "data")
box(3.6, 5.2, 2.6, 1.2, "PostgreSQL / SQLite\nschema.sql + labels\n+ synthetic_truth (isolated)", "data")
box(6.9, 5.2, 2.6, 1.2, "SQL analytics library\n11 portable queries\n(window functions)", "data")
box(10.2, 5.2, 3.4, 1.2, "Dashboard (Streamlit)\nrisk distribution · suspicious txns · trends\nperformance · FP · workload · monitoring", "serve")
box(3.6, 3.2, 2.6, 1.3, "Point-in-time features\n(strictly-past windows,\nmatured labels only)", "ml")
box(6.9, 3.2, 2.6, 1.3, "Detectors\nrules · robust z · IsolationForest\nlogreg · gradient boosting", "ml")
box(10.2, 3.2, 3.4, 1.3, "Time-split train / validate / test\nPlatt calibration · cost-optimal threshold\nevaluation + cost + FP analysis", "ml")
box(3.6, 1.1, 2.6, 1.3, "model.joblib + model_meta.json\nmetrics.json · scored.parquet", "ops")
box(6.9, 1.1, 2.6, 1.3, "FastAPI scoring service\nPOST /score · GET /score/{id}\nrisk score + reason codes", "serve")
box(10.2, 1.1, 3.4, 1.3, "Monitoring\nPSI drift · score drift\ndelayed-label performance", "ops")
for a, b in [((2.95, 5.8), (3.55, 5.8)), ((6.25, 5.8), (6.85, 5.8)), ((9.55, 5.8), (10.15, 5.8)), ((4.9, 5.15), (4.9, 4.55)),
             ((6.25, 3.85), (6.85, 3.85)), ((9.55, 3.85), (10.15, 3.85)), ((11.9, 3.15), (11.9, 2.45)), ((10.15, 1.75), (9.55, 1.75)),
             ((6.85, 1.75), (6.25, 1.75)), ((4.9, 3.15), (4.9, 2.45)), ((11.9, 4.55), (11.9, 5.15))]:
    arrow(a, b)
ax.text(0.3, 0.3, "Docker Compose: db → pipeline (one-shot) → api + dashboard · GitHub Actions: ruff, pytest (SQLite + PostgreSQL), pipeline smoke run",
        fontsize=9, color="#334155")
fig.savefig("docs/architecture.png", dpi=140, bbox_inches="tight")
