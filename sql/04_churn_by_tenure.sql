WITH customer_buckets AS (
    SELECT
        CASE
            WHEN CAST("tenure" AS INTEGER) BETWEEN 0 AND 12 THEN '0-12 months'
            WHEN CAST("tenure" AS INTEGER) BETWEEN 13 AND 24 THEN '13-24 months'
            WHEN CAST("tenure" AS INTEGER) BETWEEN 25 AND 48 THEN '25-48 months'
            WHEN CAST("tenure" AS INTEGER) BETWEEN 49 AND 72 THEN '49-72 months'
        END AS tenure_bucket,
        "Churn"
    FROM customers_raw
)
SELECT
    tenure_bucket,
    COUNT(*) AS customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate_percent
FROM customer_buckets
WHERE tenure_bucket IS NOT NULL
GROUP BY tenure_bucket
ORDER BY CASE tenure_bucket
    WHEN '0-12 months' THEN 1
    WHEN '13-24 months' THEN 2
    WHEN '25-48 months' THEN 3
    WHEN '49-72 months' THEN 4
END;
