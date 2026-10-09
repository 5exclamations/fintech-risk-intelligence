# Model limitations and intended use

> **This repository is a demonstration built on synthetic data. It is not a production fraud decision system and must not be used to approve, decline,
> block, report or investigate real customers or transactions.**

## 1. Synthetic ≠ real
* The fraud signal is whatever the simulator author wrote (bursts, new devices, impossible travel, planted risky merchants). Real fraud is adversarial, more varied and often indistinguishable from legitimate behaviour; expect materially lower PR-AUC and different optimal thresholds.
* Costs (review $6, friction $3, chargeback fee $25), label delay (14 days), label noise (5% / 0.15%), base rate (≈1%) and population are assumptions, not measurements.
* One simulated year, one generator seed, one model seed. The reported day-block bootstrap CI only covers test-sampling noise, not data-generation or training randomness.

## 2. Poor recall on unseen fraud patterns (the most important limitation)
The test window contains a fraud typology (`gift_card_cashout`, 270 transactions, introduced after simulated day 300) that did not exist in training.

| Pattern (test, synthetic ground truth) | Recall at the review threshold |
|---|---|
| account_takeover | 100% |
| card_testing | 100% |
| stolen_card | 93% |
| low_and_slow (designed to look normal) | 72% |
| **gift_card_cashout (never seen in training)** | **21%** (mean score 0.08) |

Why it matters
* A supervised model detects what it has seen. A new typology with moderate amounts, domestic IP and low velocity looks like ordinary behaviour; the unsupervised detectors did not rescue it either.
* **Aggregate metrics hide it.** Overall PR-AUC stayed ≈0.72–0.74 across 30-day blocks because the new pattern is a small share of transactions; PSI on inputs did not alert either. Only per-pattern recall — which requires ground truth that real systems don't have — exposes it. In production the signal would arrive late, through chargebacks, analyst feedback or threat intelligence.
* Recall on the simplest simulated patterns (100%) is an artifact of how bursty the simulator makes them; do not generalise it.

Mitigations a real system would need (not implemented): segment-level and typology-level monitoring, analyst-feedback and rapid-label loops, challenger models and scheduled retraining, rules that can be deployed in hours, graph/consortium features, and adversarial testing.

## 3. Other limitations
* Occlusion-based reason codes describe model behaviour, not causality; groups interact so contributions are not additive.
* Calibration is fitted on one validation window; the base rate moved 0.79% → 1.23% in test, so probabilities are optimistic/pessimistic when prevalence shifts.
* Fairness and disparate impact were not assessed; customer age/segment are not features, but behavioural proxies may correlate with protected attributes in real data.
* Feature serving reads full history from an in-memory table (demo); production needs a low-latency feature store and exactly-once event handling.
* Isolation Forest and robust-z detectors are unsupervised but still tuned on one synthetic world.
* No privacy, retention, consent, model-risk-management or regulatory (e.g. adverse-action) work has been done.
