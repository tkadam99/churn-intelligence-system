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


def check_allowed_category_values(df: pd.DataFrame) -> CheckResult:
    allowed_values = {
        "Churn": {"Yes", "No"},
        "Contract": {"Month-to-month", "One year", "Two year"},
        "gender": {"Male", "Female"},
        "Partner": {"Yes", "No"},
        "Dependents": {"Yes", "No"},
        "PhoneService": {"Yes", "No"},
        "PaperlessBilling": {"Yes", "No"},
    }
    unexpected_values = {}

    for column, allowed in allowed_values.items():
        invalid = df.loc[~df[column].isin(allowed), column]
        counts = invalid.value_counts(dropna=False)
        if not counts.empty:
            unexpected_values[column] = ", ".join(
                f"{'<missing>' if pd.isna(value) else value} ({count})"
                for value, count in counts.items()
            )

    details = (
        "; ".join(
            f"{column}: {values}" for column, values in unexpected_values.items()
        )
        if unexpected_values
        else "All category values are allowed."
    )
    return CheckResult(
        name="check_allowed_category_values",
        passed=not unexpected_values,
        details=details,
    )


def run_all_checks(df: pd.DataFrame) -> list[CheckResult]:
    return [
        check_no_blank_strings(df),
        check_unique_customer_ids(df),
        check_no_duplicate_rows(df),
        check_tenure_range(df),
        check_monthly_charges_positive(df),
        check_senior_citizen_values(df),
        check_allowed_category_values(df),
    ]


if __name__ == "__main__":
    checks = run_all_checks(load_raw_data())
    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name}: {status} - {result.details}")
