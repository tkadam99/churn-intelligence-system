import pandas as pd

from src.prepare_data import load_prepared_data, split_data
from src.quality_checks import CheckResult


def check_no_row_overlap(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> CheckResult:
    overlapping_indices = X_train.index.intersection(X_test.index)
    details = (
        f"Overlapping row indices: {list(overlapping_indices)}."
        if len(overlapping_indices)
        else "No row indices appear in both sets."
    )
    return CheckResult(
        name="check_no_row_overlap",
        passed=len(overlapping_indices) == 0,
        details=details,
    )


def check_target_not_in_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> CheckResult:
    locations = [
        name
        for name, features in [("X_train", X_train), ("X_test", X_test)]
        if "Churn" in features.columns
    ]
    details = (
        f"Churn found in {', '.join(locations)}."
        if locations
        else "Churn is not in either feature set."
    )
    return CheckResult(
        name="check_target_not_in_features",
        passed=not locations,
        details=details,
    )


def check_no_perfect_predictor(
    X_train: pd.DataFrame, y_train: pd.Series, threshold: float = 0.9
) -> CheckResult:
    correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    top_correlations = correlations.head(3)
    high_correlations = correlations[correlations > threshold]

    top_details = ", ".join(
        f"{feature}: {correlation:.3f}"
        for feature, correlation in top_correlations.items()
    )
    details = f"Top absolute correlations: {top_details}."
    if not high_correlations.empty:
        flagged = ", ".join(
            f"{feature}: {correlation:.3f}"
            for feature, correlation in high_correlations.items()
        )
        details += f" Above threshold ({threshold:.2f}): {flagged}."

    return CheckResult(
        name="check_no_perfect_predictor",
        passed=high_correlations.empty,
        details=details,
    )


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, X_test, y_train, _ = split_data(prepared_data)
    results = [
        check_no_row_overlap(X_train, X_test),
        check_target_not_in_features(X_train, X_test),
        check_no_perfect_predictor(X_train, y_train),
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name}: {status} - {result.details}")
