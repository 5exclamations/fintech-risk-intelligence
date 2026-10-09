-- 05 Unusual transaction behaviour, computed with window functions that only look at the PAST of each customer
-- (ROWS ... 1 PRECEDING / RANGE ... 1 PRECEDING), i.e. the same point-in-time discipline used for ML features.
-- Flags: amount > 5x the customer's running mean, or >= 4 prior txns in the previous hour.
-- Parameters: :limit
WITH w AS (
    SELECT t.txn_id, t.customer_id, t.ts, t.amount, t.channel, t.is_fraud,
           COUNT(*)      OVER (PARTITION BY t.customer_id ORDER BY t.ts_epoch ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_n,
           AVG(t.amount) OVER (PARTITION BY t.customer_id ORDER BY t.ts_epoch ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_mean,
           COUNT(*)      OVER (PARTITION BY t.customer_id ORDER BY t.ts_epoch RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING)  AS prior_1h
    FROM transactions t
)
SELECT txn_id, customer_id, ts, ROUND(CAST(amount AS NUMERIC), 2) AS amount, channel,
       ROUND(CAST(prior_mean AS NUMERIC), 2) AS prior_mean, ROUND(CAST(amount / prior_mean AS NUMERIC), 1) AS amount_vs_mean, prior_1h,
       CASE WHEN amount > 5 * prior_mean AND prior_1h >= 4 THEN 'spike+burst'
            WHEN amount > 5 * prior_mean THEN 'amount_spike' ELSE 'velocity_burst' END AS flag,
       is_fraud
FROM w
WHERE (prior_n >= 10 AND amount > 5 * prior_mean) OR prior_1h >= 4
ORDER BY amount / prior_mean DESC
LIMIT :limit;
