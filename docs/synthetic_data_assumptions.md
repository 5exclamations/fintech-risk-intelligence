# Synthetic data assumptions

**Everything in this repository is simulated.** No real customer, merchant, or transaction is involved, and the fraud
"typologies" are invented simplifications. This page lists every assumption so reviewers can judge what the results can
and cannot mean. All parameters live in [`riskplatform/config.py`](../riskplatform/config.py) and
[`riskplatform/generate.py`](../riskplatform/generate.py); the generator is deterministic for a given seed.

## Scale (defaults)
3,000 customers · 5,103 accounts · 700 merchants · 38 cities · 365 days (2025-01-01 → 2025-12-31) · ~405k transactions ·
label fraud rate 0.87% (validation 0.79%, test 1.23%).

## Legitimate behaviour
| Aspect | Assumption |
|---|---|
| Customers | 5 personas (student, young professional, family, affluent, retiree) with different transaction rates, log-normal ticket sizes, online share, night-owl share, and trip propensity. 75% existed before the window, 25% sign up during it (cold start). |
| Volume | Poisson counts per customer; weekend uplift (×1.1–1.3); holiday-season volume uplift ×1.35 (days 325–358). |
| Time of day | Customer-specific mix of a daytime profile and a late-night profile, in *local* time at the transaction location. |
| Merchants | 13 categories; physical merchants live in cities, online merchants are location-less; popularity is log-normal; 20% of merchants onboard mid-year. |
| Channels | `card_present`, `online`, `atm`. Online transactions carry a device id; a customer's device changes once for 25% of customers (a legitimate "new device"). |
| Location | Home city for most transactions; 1–3 trips (3–9 days) for travel-prone customers, 30% of trips abroad. For online transactions the location is the IP-geolocated city. |
| Hard negatives | 4% of transactions spawn a "shopping burst" of 1–3 extra transactions within 2 h; 0.6% of purchases are 4–5× larger than usual; holiday spending raises retail/electronics/gift-card/travel/entertainment amounts by 25% after day 325 (covariate drift). |

## Fraud typologies (event → several transactions)
| Pattern | Events | Mechanics (invented) |
|---|---|---|
| `account_takeover` | 150 | New device, online, far/foreign IP, 3–12 transactions at ~8 min spacing, amounts ≈3× the victim's median, high-risk categories. **30% are "subtle"**: 1–3 transactions, typical amounts, domestic IP. |
| `card_testing` | 60 | 6–25 micro-payments ($0.50–$4.99) in 3–20 min at online merchants, then 1–2 large purchases. |
| `stolen_card` | 150 | Card-present/ATM use 20 min–5 h after one of the victim's real purchases, in a city > ~900 km away (impossible travel). |
| `low_and_slow` | 420 | Merchant-compromise follow-up 7–28 days after the victim used a compromised merchant: 1–2 online purchases with typical amounts, 60% new device. Designed to be hard. |
| `gift_card_cashout` | 90 | **Only emitted after day 300** (concept drift): 2–4 moderate purchases ($100–$400) over hours, domestic IP, gift cards/crypto. |

4% of merchants (plus a few physical ones) are latently "compromised" and are 8× more likely to be picked by fraud, which
creates a learnable merchant-risk signal.

## Label process
* **Delay** – a chargeback label is only *known* 14 days after the transaction (`SplitConfig.label_delay_days`). Splits and label-derived features respect this.
* **Noise** – 5% of fraud is never discovered (labelled legitimate); 0.15% of legitimate transactions are labelled fraud (disputes / friendly fraud).
* The generator's true pattern is stored in `synthetic_truth` and **never** used for features or training; it is only joined after training for diagnostics that are impossible in real life.

## What this data deliberately does not model
Merchant-category-code granularity, card-network/BIN data, issuer authorisation data (CVV/AVS/3-DS), IP reputation, device fingerprint
quality, consumer identity networks (shared devices/addresses between customers), fraud-ring graph structure, adversarial adaptation,
seasonality beyond the holiday bump, regulatory/PII handling, and the true base rate of any real portfolio.
