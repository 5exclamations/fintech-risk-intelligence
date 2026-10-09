# Data dictionary

All tables are defined in [`sql/schema.sql`](../sql/schema.sql) (PostgreSQL 13+ / SQLite 3.35+). All data is synthetic.

## `locations`
| column | type | description |
|---|---|---|
| location_id | int PK | |
| city, country | text | 38 cities (20 domestic "US", 18 foreign) |
| lat, lon | float | used for haversine distances |
| utc_offset | float | hours, used to derive local time |
| is_domestic | int | 1 = domestic market |

## `customers`
customer_id (PK) · segment (generator persona) · age · home_location_id (FK) · signup_date · kyc_risk_rating (low/medium/high)

## `accounts`
account_id (PK) · customer_id (FK) · account_type (`debit`, `credit_card`) · opened_at · credit_limit (credit cards only)

## `merchants`
merchant_id (PK) · name · category (13 values) · location_id (FK; HQ city for online merchants) · is_online · onboarded_at

## `transactions`
| column | description |
|---|---|
| txn_id (PK) | `T0000000…`, ordered by time |
| account_id, customer_id, merchant_id | FKs |
| location_id | where the transaction happened (POS city, or IP-geolocated city for online) |
| ts | UTC timestamp |
| txn_date, hour_local, dow_local, ts_epoch | derived convenience columns (UTC date, local hour 0-23, local weekday 0=Mon, epoch seconds) kept so analytical SQL is dialect-neutral |
| amount | transaction amount (currency-less) |
| channel | `card_present` \| `online` \| `atm` |
| device_id | NULL for card-present / ATM |
| is_fraud | **label** – chargeback/dispute outcome; noisy and delayed (see assumptions) |

## `synthetic_truth`, `synthetic_truth_merchants`
Generator ground truth (`true_pattern`, `is_fraud_true`, `is_compromised`). **Isolated on purpose**; never read by feature code
(`riskplatform/features.py`, `modeling.make_dataset`). Only `pipeline.step_train_evaluate` joins it *after* training for synthetic-only diagnostics.

## Model features (`riskplatform/features.py → NUMERIC_FEATURES`, 52 columns)
| group | features |
|---|---|
| Transaction | amount, log_amount, is_online, is_atm, hour_local, dow_local, is_night (≤05h), is_weekend, category_code |
| Velocity (customer, strictly-past windows) | cnt_10m / 1h / 6h / 24h / 7d, sum_1h / 24h / 7d, small_cnt_30m (<$5), online_cnt_1h, distinct_merchants_24h, cnt_24h_vs_daily_rate |
| Amount deviation | cust_prior_mean, cust_prior_max, amount_to_mean, amount_to_max, amount_logz (log-amount z-score vs the customer's history), amount_gt_prior_max |
| Geo | dist_from_last_km, implied_speed_kmh, hours_since_last, dist_from_home_km, is_foreign, new_country |
| Customer history | cust_prior_n, new_merchant_for_cust, new_category_for_cust, cust_channel_share, cust_category_share, cust_hourband_share, cust_night_share, new_device, device_prior_cnt |
| Merchant history | merch_prior_n, merch_prior_mean, amount_to_merch_mean, merch_cnt_1h, merch_cnt_24h, merch_age_days, merch_is_online, merch_fraud_rate_matured (shrunk, **mature labels only**) |
| Account | account_age_days, is_credit |

Missing history is `NaN` (handled natively by gradient boosting; median-imputed with indicator for logistic regression).
