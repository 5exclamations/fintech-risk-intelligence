# Monitoring

Run as part of the pipeline (`riskplatform/monitoring.py`) → `artifacts/monitoring/` and the dashboard's *Monitoring* tab; API: `GET /monitoring`.

| Signal | Method | Thresholds |
|---|---|---|
| Input drift | PSI per feature (10 reference-quantile bins + NaN bin) per 14-day window vs the last 60 training days | warn ≥ 0.10, alert ≥ 0.25 |
| Score drift | PSI of calibrated score vs the same reference | same |
| Alert rate / mean score | per window at the validated threshold | visual |
| Performance | PR-AUC, precision, recall per window **only when labels are mature** (window end + 14 d ≤ as-of day) compared with the validation baseline | warn: −15% relative, alert: −30% |

Design decisions
* **Labels arrive late**, so input/score drift is the early warning; performance windows lag by the label delay.
* Features that grow with calendar time *by construction* (history counters, merchant matured rate; `MONOTONIC_FEATURES`) are excluded; otherwise PSI fires forever.
* Partial windows (< 70% of days, e.g. the label-gap) are skipped.

What it shows on the synthetic data: the final windows warn on `cnt_7d`/`sum_7d` (holiday volume — a genuine covariate shift injected by the generator). The post-day-300
`gift_card_cashout` pattern is *not* visible in aggregate PSI or aggregate PR-AUC (it is ~0.3% of transactions), but is clearly visible in per-pattern recall (21%) — an example of why
aggregate monitors need segment-level and label-feedback checks. Retraining triggers: any alert-level PSI, performance alert, or scheduled refresh.
