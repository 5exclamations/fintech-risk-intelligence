-- 08 Category-level risk and exposure.
SELECT m.category, m.is_online,
       COUNT(*)                        AS n_txn,
       SUM(t.is_fraud)                 AS n_fraud,
       ROUND(CAST(100.0 * AVG(t.is_fraud) AS NUMERIC), 3) AS fraud_rate_pct,
       ROUND(CAST(SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END) AS NUMERIC), 0) AS fraud_amount
FROM transactions t JOIN merchants m ON m.merchant_id = t.merchant_id
GROUP BY m.category, m.is_online
HAVING COUNT(*) >= 200
ORDER BY fraud_rate_pct DESC;
