# Model card — Transaction Risk Model v1.0.0

**⚠️ Trained and evaluated on synthetic data only. Not validated for any real-world use.**

| | |
|---|---|
| Model | `HistGradientBoostingClassifier` + Platt calibration (selected over logistic regression on validation PR-AUC: 0.659 vs 0.580) |
| Task | Binary: transaction later labelled fraud (chargeback/dispute) |
| Inputs | 52 point-in-time features (see data dictionary); no PII, no ground-truth columns |
| Output | calibrated probability, band (`allow`/`review`/`block`), reason codes |
| Training data | days 30–215 of simulated year, 191,243 txns (0.78% fraud labels) |
| Validation | days 230–275 (calibration, thresholds, model choice) |
| Test | days 290–364, 103,469 txns, 1.23% fraud labels |
| Seeds | generator 42, model 42 |

## Performance (test, synthetic)
PR-AUC 0.730 (95% CI 0.694–0.767) · at review threshold 0.0746: precision 80.2%, recall 68.3%, F1 0.738, FPR 0.21%, ~14 alerts/day.
By 30-day block PR-AUC 0.74 / 0.72 / 0.74 — flat on aggregate even though the unseen `gift_card_cashout` pattern has only 21% recall.

## Intended use
Demonstration of methodology: time-safe feature engineering, evaluation, cost-based thresholds, explainability, monitoring.

## Out-of-scope / not recommended
Real credit, lending, or account-closure decisions; any use without re-training on real data and independent validation; automated decline without human appeal path.

## Known limitations
* The simulator defines the signal: velocity, device and geography effects are far cleaner than in reality. Expect real PR-AUC to be lower and thresholds to differ.
* Hand-set cost assumptions drive the operating point.
* Label noise is modelled coarsely; real chargeback delays vary by network and reason code.
* Single seed / single simulated year; no confidence interval for training randomness (only a day-block bootstrap for the test sample).
* Occlusion reason codes are model-specific explanations, not causal.
* Fairness: customer personas/age exist but were not analysed for disparate impact; none of them is a model feature except through behaviour.

## Monitoring & retraining
PSI + delayed-label windows (see `monitoring.md`); retrain on alert or on a schedule; re-fit calibration whenever the base rate shifts.
