from dataclasses import dataclass

import pandas as pd

from src.load_data import load_raw_data


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def check_no_blank_strings(df: pd.DataFrame) -> CheckResult:
    problem_counts: dict[str, int] = {}
    text_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in text_columns:
        blank_count = df[column].map(
            lambda value: isinstance(value, str) and not value.strip()
        ).sum()
        if blank_count:
            problem_counts[column] = int(blank_count)

    if problem_counts:
        details = ", ".join(
            f"{column}: {count}" for column, count in problem_counts.items()
        )
    else:
        details = "No empty or whitespace-only strings found."

    return CheckResult(
        name="check_no_blank_strings",
        passed=not problem_counts,
        details=details,
    )


def check_unique_customer_ids(df: pd.DataFrame) -> CheckResult:
    duplicate_count = int(df["customerID"].duplicated().sum())
    details = (
        f"{duplicate_count} duplicate customerID rows after the first occurrence."
        if duplicate_count
        else "No duplicate customerID values found."
    )
    return CheckResult(
        name="check_unique_customer_ids",
        passed=duplicate_count == 0,
        details=details,
    )


def check_no_duplicate_rows(df: pd.DataFrame) -> CheckResult:
    duplicate_count = int(df.duplicated().sum())
    details = (
        f"{duplicate_count} fully duplicate rows after the first occurrence."
        if duplicate_count
        else "No fully duplicate rows found."
    )
    return CheckResult(
        name="check_no_duplicate_rows",
        passed=duplicate_count == 0,
        details=details,
    )


def check_tenure_range(df: pd.DataFrame) -> CheckResult:
    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    invalid_count = int(tenure.isna().sum())
    out_of_range_count = int(((tenure < 0) | (tenure > 72)).sum())
    passed = invalid_count == 0 and out_of_range_count == 0
    details = (
        f"{out_of_range_count} values outside 0-72; "
        f"{invalid_count} missing or non-numeric values."
    )
    return CheckResult(
        name="check_tenure_range",
        passed=passed,
        details=details,
    )


def check_monthly_charges_positive(df: pd.DataFrame) -> CheckResult:
    charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    invalid_count = int(charges.isna().sum())
    non_positive_count = int((charges <= 0).sum())
    passed = invalid_count == 0 and non_positive_count == 0
    details = (
        f"{non_positive_count} values are not greater than 0; "
        f"{invalid_count} missing or non-numeric values."
    )
    return CheckResult(
        name="check_monthly_charges_positive",
        passed=passed,
        details=details,
    )


def check_senior_citizen_values(df: pd.DataFrame) -> CheckResult:
    invalid_count = int((~df["SeniorCitizen"].isin([0, 1])).sum())
    details = (
        f"{invalid_count} values are not 0 or 1."
        if invalid_count
        else "All values are 0 or 1."
    )
    return CheckResult(
        name="check_senior_citizen_values",
        passed=invalid_count == 0,
        details=details,
    )


if __name__ == "__main__":
    data = load_raw_data()
    checks = [
        check_no_blank_strings(data),
        check_unique_customer_ids(data),
        check_no_duplicate_rows(data),
        check_tenure_range(data),
        check_monthly_charges_positive(data),
        check_senior_citizen_values(data),
    ]
    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name}: {status} - {result.details}")
