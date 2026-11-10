SELECT
    "Contract",
    COUNT(*) AS customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate_percent
FROM customers_raw
GROUP BY "Contract"
ORDER BY churn_rate_percent DESC;
