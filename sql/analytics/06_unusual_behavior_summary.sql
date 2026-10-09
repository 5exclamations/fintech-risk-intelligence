-- 06 How well do the simple SQL flags from 05 separate fraud-labelled from other transactions?
WITH w AS (
    SELECT t.is_fraud, t.amount,
           COUNT(*)      OVER (PARTITION BY t.customer_id ORDER BY t.ts_epoch ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_n,
           AVG(t.amount) OVER (PARTITION BY t.customer_id ORDER BY t.ts_epoch ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_mean,
           COUNT(*)      OVER (PARTITION BY t.customer_id ORDER BY t.ts_epoch RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING)  AS prior_1h
    FROM transactions t
),
f AS (
    SELECT is_fraud,
           CASE WHEN prior_n >= 10 AND amount > 5 * prior_mean THEN 1 ELSE 0 END AS amount_spike,
           CASE WHEN prior_1h >= 4 THEN 1 ELSE 0 END                              AS velocity_burst
    FROM w
)
SELECT amount_spike, velocity_burst,
       COUNT(*)                 AS n_txn,
       SUM(is_fraud)            AS n_fraud,
       ROUND(CAST(100.0 * AVG(is_fraud) AS NUMERIC), 2) AS fraud_rate_pct
FROM f
GROUP BY amount_spike, velocity_burst
ORDER BY amount_spike, velocity_burst;
