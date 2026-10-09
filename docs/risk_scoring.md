# Risk scoring explanation

## Output
`POST /score` returns:
```json
{
  "risk_score": 0.634,            // calibrated P(fraud label) — estimated on synthetic data
  "risk_band": "block",           // allow < 0.0746 <= review < 0.5748 <= block
  "reasons": [
    {"code": "R06_RISKY_MERCHANT", "source": "rule",  "detail": "merchant's historical (matured-label) fraud rate is >= 5%"},
    {"code": "NEW_DEVICE",        "source": "model", "contribution": 4.99, "detail": "device never used by this customer before"},
    {"code": "VELOCITY",          "source": "model", "contribution": 2.26, "detail": "0 txns in the last hour / 2 in 24h ..."}
  ],
  "rule_score": 0.30, "model_version": "1.0.0", "notice": "Model trained on synthetic data ..."
}
```
`GET /score/{txn_id}` replays a stored transaction using only earlier history (used for audits and for the parity test).

## How the score is made
1. Look up the customer's and the merchant's history strictly before the transaction time.
2. Build the same 52 features as in training.
3. Gradient-boosting probability → Platt calibration → band via validation-derived thresholds.

## Reason codes
Two sources are merged, rule hits first:
* **Rules (R01–R09)** — exact, human-readable conditions (e.g. `R04_IMPOSSIBLE_TRAVEL`: ≥500 km and ≥900 km/h from the previous transaction).
* **Model attributions** — group occlusion. For each of 8 feature groups (`VELOCITY`, `AMOUNT_DEVIATION`, `GEO_ANOMALY`, `NEW_DEVICE`, `MERCHANT_RISK`,
  `UNFAMILIAR_MERCHANT`, `TIME_PATTERN`, `CUSTOMER_PROFILE`) we replace the group's features with the median of legitimate training transactions and report the **drop in log-odds**.
  Top-3 groups with a drop > 0.15 are returned with a templated sentence filled with the actual values.

Caveats: occlusion shows what the model *used*, not causal truth; contributions across groups don't sum exactly to the score (interactions); thresholds are tied to the synthetic base rate.

## Using the bands (suggested policy)
`allow` → approve · `review` → step-up authentication or analyst queue · `block` → decline. Thresholds are cost-derived; change `CostConfig` and retrain to move them.
