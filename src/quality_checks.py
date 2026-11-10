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


if __name__ == "__main__":
    result = check_no_blank_strings(load_raw_data())
    print(result)
