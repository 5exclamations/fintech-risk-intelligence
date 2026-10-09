-- 01 Transaction frequency: how often do customers transact? (portable ANSI SQL)
-- Output: distribution of customers by monthly transaction count (all months combined).
WITH per_cust_month AS (
    SELECT customer_id,
           SUBSTR(txn_date, 1, 7) AS month,
           COUNT(*)               AS n_txn
    FROM transactions
    GROUP BY customer_id, SUBSTR(txn_date, 1, 7)
)
SELECT CASE WHEN n_txn <= 2  THEN '1-2'
            WHEN n_txn <= 5  THEN '3-5'
            WHEN n_txn <= 10 THEN '6-10'
            WHEN n_txn <= 20 THEN '11-20'
            ELSE '21+' END            AS txns_per_month_bucket,
       COUNT(*)                       AS customer_months,
       ROUND(CAST(100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS NUMERIC), 2) AS pct_of_customer_months
FROM per_cust_month
GROUP BY 1
ORDER BY MIN(n_txn);
