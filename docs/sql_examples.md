# SQL analytics

Eleven queries in [`sql/analytics/`](../sql/analytics) use only portable SQL (CTEs, window functions incl. `RANGE BETWEEN n PRECEDING`,
`NTILE`, `RANK`) and run **unchanged on PostgreSQL 17 and SQLite** — both are exercised by the test-suite / CI.
`python -m riskplatform.pipeline analytics` runs them and writes CSVs to `artifacts/analytics/`. Outputs below are from the synthetic dataset.

| File | Question | Technique |
|---|---|---|
| `01_transaction_frequency` | How many customers transact 1–2 / 3–5 / … times a month? | CTE + bucket + `SUM() OVER ()` |
| `02_daily_volume` | Daily volume, amount, fraud-label rate | trailing 7-day `ROWS` window |
| `03_spending_patterns` | Spend by category × channel | share-of-total window |
| `04_customer_segmentation` | Behavioural RFM segments and their fraud exposure | `NTILE(4)` |
| `05_unusual_behavior` | Transactions > 5× customer's running mean or in a ≥4-txn/hour burst | `ROWS … 1 PRECEDING`, `RANGE 3600 PRECEDING` (past-only) |
| `06_unusual_behavior_summary` | Do those SQL flags concentrate fraud? | |
| `07_merchant_risk` | Merchant ranking with empirical-Bayes smoothing | `RANK()` |
| `08_merchant_category_risk` | Category-level rate and exposure | |
| `09_time_anomalies` | Fraud-rate lift by local hour | |
| `10_daily_volume_anomalies` | Days > 3σ from the trailing 28-day baseline (σ via E[x²]−E[x]²) | named `WINDOW` |
| `11_customer_velocity_peaks` | Customers with the highest 1-hour velocity | `RANGE` window |

## Examples

**Unusual behaviour (05/06)** — a burst alone is a strong signal in this synthetic world; amount spikes alone are weak:
```
amount_spike velocity_burst   n_txn  n_fraud fraud_rate_pct
0            0               399970   2332    0.58
0            1                 1019    925   90.78
1            0                 4390    176    4.01
1            1                   91     87   95.60
```
**Time anomalies (09)** — fraud-rate lift vs portfolio is 4–6× between 02:00 and 05:00 local time (the generator makes fraud
time-uniform in UTC, while legitimate activity is daytime-heavy).

**Merchant risk (07)** — the top merchants (smoothed rate 7–24%) are the latently "compromised" online merchants the generator planted:
```
M00495 Online Services #495  n=863 fraud=256 raw 29.7% smoothed 24.3%
M00491 Crypto Exchange #491  n=705 fraud=133 raw 18.9% smoothed 14.9%
```
**Customer segmentation (04)**
```
rfm_segment  customers avg_txns avg_spend fraud_txn_rate_pct
loyal            973    156.3    8555      0.939
occasional       812     75.1    3118      0.979
dormant          691     75.9    3419      1.176
champions        524    267.0   15799      0.628
```
**Daily volume anomalies (10)** — flags 2025-11-22/23 (z = 5.7 / 3.8): the generator's holiday volume uplift.

> Analytical SQL (07, 09 …) uses *all* labels for descriptive purposes. The ML merchant feature (`merch_fraud_rate_matured`) only uses labels
> that were already mature at scoring time — do not reuse the descriptive queries as model features.

Portability notes: derived columns (`txn_date`, `hour_local`, `ts_epoch`) avoid `EXTRACT`/`strftime`; `ROUND` is applied to `CAST(… AS NUMERIC)` (PostgreSQL has no `ROUND(double, int)`);
SQLite gets a registered `sqrt()` function in `db.get_engine`.
