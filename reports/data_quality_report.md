# Data Quality Report

Date: 2026-11-10

## Summary

- Checks passed: 6
- Checks failed: 1

## Check results

- **FAIL - check_no_blank_strings:** TotalCharges: 11
- **PASS - check_unique_customer_ids:** No duplicate customerID values found.
- **PASS - check_no_duplicate_rows:** No fully duplicate rows found.
- **PASS - check_tenure_range:** 0 values outside 0-72; 0 missing or non-numeric values.
- **PASS - check_monthly_charges_positive:** 0 values are not greater than 0; 0 missing or non-numeric values.
- **PASS - check_senior_citizen_values:** All values are 0 or 1.
- **PASS - check_allowed_category_values:** All category values are allowed.
