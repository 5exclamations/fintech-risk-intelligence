# Modeling methodology

> All numbers are from the **synthetic** dataset. They show the method works end-to-end; they are not forecasts for real portfolios.

## 1. Problem framing
Binary classification of each transaction at authorisation time: *will this transaction later be labelled fraud (chargeback/dispute)?*
Output = calibrated probability → decision band (`allow` / `review` / `block`) + reason codes.

## 2. Leakage prevention (the part that usually breaks fraud models)
| Risk | Control | Test |
|---|---|---|
| Features see the future | Every window/aggregate uses rows with `ts` **strictly before** the transaction (`features._prior`, one composite-key `searchsorted` implementation). Same-second ties are excluded. | `test_no_future_leakage` recomputes features on a truncated, shuffled table and requires identical values |
| Feature uses the row's own label | label-derived feature only via matured labels | `test_own_label_never_used` |
| Label not yet known | A label is visible only `label_delay_days` (14) after its transaction (merchant fraud-rate feature **and** splits) | `test_label_maturity_delay` |
| Random split | Strict time split with gaps (below) | `test_time_split_has_no_overlap_and_matured_labels` |
| Tuning on test | Model selection, calibration and thresholds use *validation only*; test is scored once | pipeline structure |
| Generator ground truth | stored in separate tables, never read by features | `test_ground_truth_not_in_transactions` |
| Train/serve skew | API calls the same `build_features` on a history slice | `test_replay_matches_batch_scores` (online == offline to 1e-6) |

## 3. Time-based split (`SplitConfig`, 365-day dataset)
```
day:  0      30                 216   230            276   290                    364
      |warmup|------ TRAIN ------|gap  |--- VALID ----|gap  |------ TEST ----------|
```
* warm-up (30 d): history only, never trained/evaluated on (cold start).
* train = [30, 216): every label is mature at T1 = day 230. Rows in [216, 230) are dropped (labels immature).
* validation = [230, 276): mature at T2 = day 290. Used for model selection, calibration, thresholds.
* test = [290, 365): untouched until the end. It contains the **drift period** (new fraud pattern after day 300, holiday spending).

## 4. Detection methods compared (same test rows)
1. **Rules** – 9 hand-written rules with prior confidences combined by noisy-OR (`rules.py`).
2. **Statistical anomaly** – median/MAD robust z-scores on 7 behavioural features; score = mean of the top-2 positive z.
3. **Isolation Forest** – unsupervised on 14 behavioural features. 1 and 2 never see labels.
4. **Logistic regression** – signed-log transform, median imputation + indicators, class-balanced.
5. **Histogram gradient boosting** – 350 iterations, lr 0.06, 31 leaves, L2 = 1, native NaN / categorical handling. **Selected** on validation PR-AUC.

## 5. Calibration and operating points
* Platt scaling (sigmoid on the model logit) fitted on validation → score ≈ P(fraud label). Test calibration plot in the dashboard.
* **Review threshold** = argmin of business cost on validation (see below). **Block threshold** = lowest threshold with ≥90% validation precision.
* Base rates move (0.79% validation → 1.23% test), so probabilities drift upward — a reason to monitor and recalibrate.

## 6. Business cost
For each decision (all assumptions illustrative, configurable in `CostConfig`):
* alert on fraud → review cost ($6), loss prevented
* alert on legitimate → review ($6) + customer friction ($3)
* missed fraud → full amount + $25 chargeback fee

Baselines: no model, alert everything. Dashboard sliders recompute cost, workload and recall for any threshold / assumption.

## 7. Results on the synthetic test window (75 days, 103,469 txns, 1,276 fraud labels)
| Method | PR-AUC | ROC-AUC | Precision @ same alert budget | Recall @ same budget |
|---|---|---|---|---|
| Rules (noisy-OR) | 0.469 | 0.814 | 0.552 | 0.470 |
| Robust z-score | 0.490 | 0.906 | 0.576 | 0.490 |
| Isolation Forest | 0.668 | 0.939 | 0.715 | 0.609 |
| Logistic regression | 0.684 | 0.938 | 0.684 | 0.582 |
| **Gradient boosting** | **0.730** (95% CI 0.694–0.767, day-block bootstrap) | 0.930 | 0.802 | 0.683 |

Operating point (review threshold 0.0746): TP 871 · FP 215 · FN 405 · TN 101,978 → **precision 80.2%, recall 68.3%, F1 0.738, FPR 0.21%**,
~14 alerts/day (~0.22 analyst FTE). Cost $80.3k vs $181.4k with no model (−55.7%) — *under invented cost assumptions*.
Block band (≥0.575): precision 94.4%, recall 55.2%.

Honest reading:
* Account takeover and card testing are caught at 100% **because the simulator makes them bursty**; real attackers adapt.
* The post-drift `gift_card_cashout` pattern the model never saw: recall **21%** (mean score 0.08). `low_and_slow` recall 72%. That gap is exactly what monitoring must surface.
* 22.8% of "false positives" are actually undiscovered fraud (label noise) — a synthetic-only insight; in real life, FP counts are overstated by undiscovered fraud too.
* False positives concentrate in online transactions (76.7% of FPs; FP rate 0.66% vs 0.07% card-present) and in newer customers (0.85% vs 0.18% for 50+ txns).

## 8. Reproducibility
`python -m riskplatform.pipeline all` (seeded: generator seed 42, model seed 42) regenerates every artifact. Same results on SQLite and PostgreSQL (verified: identical metrics).
Config snapshots (splits, cost assumptions, seeds) are stored in `artifacts/model/model_meta.json`.
