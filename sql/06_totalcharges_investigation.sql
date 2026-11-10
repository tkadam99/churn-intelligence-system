-- Query A: count rows with blank TotalCharges
SELECT COUNT(*) AS blank_total_charges
FROM customers_raw
WHERE TRIM("TotalCharges") = '';

-- Query B: inspect rows with blank TotalCharges
SELECT "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"
FROM customers_raw
WHERE TRIM("TotalCharges") = '';
