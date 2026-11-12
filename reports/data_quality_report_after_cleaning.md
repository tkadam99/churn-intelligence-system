# Data Quality Report After Cleaning

Date: 2026-11-12

Data state: after clean_total_charges
Whitespace-only TotalCharges values set to 0: 11

## Summary

- Cleaned-data checks passed: 7
- Cleaned-data checks failed: 0

## Check results

| Check | Raw-data result | Cleaned-data result |
| --- | --- | --- |
| check_no_blank_strings | FAIL | PASS |
| check_unique_customer_ids | PASS | PASS |
| check_no_duplicate_rows | PASS | PASS |
| check_tenure_range | PASS | PASS |
| check_monthly_charges_positive | PASS | PASS |
| check_senior_citizen_values | PASS | PASS |
| check_allowed_category_values | PASS | PASS |

## Details of checks still failing

None.
