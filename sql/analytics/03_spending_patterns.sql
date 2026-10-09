-- 03 Spending patterns by merchant category and channel.
SELECT m.category,
       t.channel,
       COUNT(*)                       AS n_txn,
       ROUND(CAST(AVG(t.amount) AS NUMERIC), 2)        AS avg_amount,
       ROUND(CAST(SUM(t.amount) AS NUMERIC), 0)        AS total_amount,
       ROUND(CAST(100.0 * SUM(t.amount) / SUM(SUM(t.amount)) OVER () AS NUMERIC), 2) AS pct_of_spend,
       ROUND(CAST(100.0 * AVG(t.is_fraud) AS NUMERIC), 3) AS fraud_rate_pct
FROM transactions t
JOIN merchants m ON m.merchant_id = t.merchant_id
GROUP BY m.category, t.channel
ORDER BY total_amount DESC;
