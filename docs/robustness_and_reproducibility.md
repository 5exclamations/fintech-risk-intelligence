# Robustness checks, slice screen and run provenance

> All numbers below are from **synthetic** data (seed 42, default config). They show the *method*; they are not real-world performance.
> Artifacts are written by `python -m riskplatform.pipeline all` (`metrics.json`, `run_manifest.json`) and `... pipeline backtest` (`backtest.json`).

## 1. Rolling-origin backtest (`riskplatform/backtest.py`)
The headline single split cannot say how results depend on *when* the model is trained. The backtest repeats the identical procedure
(train on matured labels → calibrate/threshold on validation → score the following 30 days) at four origins, 30 days apart.

| Origin (sim. day) | Test days | Frauds | PR-AUC | Rules PR-AUC | Precision | Recall | Cost saving |
|---|---|---|---|---|---|---|---|
| 200 | 200–229 | 177 | 0.563 | 0.188 | 0.763 | 0.565 | 62.6% |
| 230 | 230–259 | 293 | 0.692 | 0.432 | 0.739 | 0.706 | 72.2% |
| 260 | 260–289 | 315 | 0.736 | 0.447 | 0.773 | 0.759 | 75.9% |
| 290 (production split) | 290–319 | 485 | 0.736 | 0.470 | 0.862 | 0.682 | 51.0% |

Mean PR-AUC 0.681 ± 0.071 (min 0.563, max 0.736). Reading it: (a) performance rises with more training history, so the headline 0.730 is not a lucky
point but is at the upper end of the range; (b) the earliest origin is materially worse (fewer labelled frauds); (c) the production-origin fold covers
only the first 30 test days (before the unseen gift-card typology matters much), which is why it differs from the 75-day headline. Folds share
data and a generator seed, so the spread is **not** a confidence interval for real-world performance.

## 2. Slice screen (`riskplatform/slices.py`, `metrics.json → slice_performance`)
Recall / false-positive rate by customer segment, channel, merchant category, amount band and customer history, with flags where recall is < 70% of overall
(0.683) or FPR is > 1.5× overall (0.21%). Slices with < 30 frauds (recall) or < 500 legitimate rows (FPR) are never flagged. Findings on the test window:
* **Low recall:** `gift_cards` (0.32), `crypto_exchange` (0.47), `grocery` (0.40, mostly low-and-slow fraud designed to look normal). The first is the unseen typology from `limitations.md §2` — visible here *without* ground-truth pattern labels, which is the realistic monitoring route.
* **High FPR:** `online` channel, `electronics`, `online_services`, `entertainment`, `travel`, `affluent` segment and the 100–500 amount band — i.e. the review burden concentrates on card-not-present, higher-ticket activity.
* This is a screening tool, **not a fairness audit**: no protected attributes exist in the data.

## 3. Calibration summary
Test Brier 0.0052 vs 0.0122 for a constant predictor; quantile-bin ECE 0.0026. Mean predicted rate 1.01% vs observed 1.23% — probabilities are slightly
optimistic because the fraud base rate rose after the validation window (see `limitations.md §3`).

## 4. Run manifest (`artifacts/run_manifest.json`)
Records git commit and dirty flag, Python/OS, library versions, a content hash of the modelling dataset (order-independent), a hash of the generator/split/cost config, and the headline
synthetic metrics. Use it to answer "which code, data and libraries produced this model?" and to detect silent drift in the environment.
`constraints.txt` pins the dependency versions of the verified run (`pip install -r requirements-dev.txt -c constraints.txt`).

## 5. Windows
`scripts/dev.ps1` mirrors the Makefile (`setup | all | backtest | test | lint | api | dashboard`); `.gitattributes` forces LF so golden files and hashes are identical on
Windows and Linux; CI has a `windows-latest` job. Python 3.12 is recommended; 3.14+ may lack wheels for some dependencies.
