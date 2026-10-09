-- 09 Time-based anomalies, part 1: fraud rate by local hour vs portfolio (lift).
WITH tot AS (SELECT 1.0 * SUM(is_fraud) / COUNT(*) AS base_rate FROM transactions)
SELECT t.hour_local,
       COUNT(*)                           AS n_txn,
       SUM(t.is_fraud)                    AS n_fraud,
       ROUND(CAST(100.0 * AVG(t.is_fraud) AS NUMERIC), 3)  AS fraud_rate_pct,
       ROUND(CAST(AVG(t.is_fraud) / tot.base_rate AS NUMERIC), 2) AS lift_vs_portfolio
FROM transactions t CROSS JOIN tot
GROUP BY t.hour_local, tot.base_rate
ORDER BY t.hour_local;
