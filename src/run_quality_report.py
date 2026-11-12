import argparse
from datetime import date
from pathlib import Path

from src.clean_data import clean_total_charges
from src.load_data import load_raw_data
from src.quality_checks import run_all_checks


def write_raw_report() -> None:
    checks = run_all_checks(load_raw_data())
    passed_count = sum(result.passed for result in checks)
    failed_count = len(checks) - passed_count

    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name}: {status} - {result.details}")

    report_lines = [
        "# Data Quality Report",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Summary",
        "",
        f"- Checks passed: {passed_count}",
        f"- Checks failed: {failed_count}",
        "",
        "## Check results",
        "",
    ]
    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        report_lines.append(f"- **{status} - {result.name}:** {result.details}")

    report_path = (
        Path(__file__).resolve().parents[1] / "reports" / "data_quality_report.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report saved to {report_path.relative_to(report_path.parents[1])}")


def write_cleaned_report() -> None:
    raw_data = load_raw_data()
    raw_checks = run_all_checks(raw_data)
    blank_total_charges_count = int(
        raw_data["TotalCharges"]
        .astype("string")
        .str.strip()
        .eq("")
        .fillna(False)
        .sum()
    )
    cleaned_checks = run_all_checks(clean_total_charges(raw_data))
    passed_count = sum(result.passed for result in cleaned_checks)
    failed_count = len(cleaned_checks) - passed_count

    report_lines = [
        "# Data Quality Report After Cleaning",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "Data state: after clean_total_charges",
        (
            "Whitespace-only TotalCharges values set to 0: "
            f"{blank_total_charges_count}"
        ),
        "",
        "## Summary",
        "",
        f"- Cleaned-data checks passed: {passed_count}",
        f"- Cleaned-data checks failed: {failed_count}",
        "",
        "## Check results",
        "",
        "| Check | Raw-data result | Cleaned-data result |",
        "| --- | --- | --- |",
    ]
    for raw_result, cleaned_result in zip(raw_checks, cleaned_checks):
        raw_status = "PASS" if raw_result.passed else "FAIL"
        cleaned_status = "PASS" if cleaned_result.passed else "FAIL"
        report_lines.append(
            f"| {raw_result.name} | {raw_status} | {cleaned_status} |"
        )

    remaining_failures = [
        result for result in cleaned_checks if not result.passed
    ]
    report_lines.extend(["", "## Details of checks still failing", ""])
    if remaining_failures:
        report_lines.extend(
            f"- **{result.name}:** {result.details}"
            for result in remaining_failures
        )
    else:
        report_lines.append("None.")

    report_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "data_quality_report_after_cleaning.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Data state: after clean_total_charges")
    print(
        "Whitespace-only TotalCharges values set to 0: "
        f"{blank_total_charges_count}"
    )
    print(f"Cleaned-data checks passed: {passed_count}")
    print(f"Cleaned-data checks failed: {failed_count}")
    print("\nCheck results:")
    print("| Check | Raw-data result | Cleaned-data result |")
    print("| --- | --- | --- |")
    for raw_result, cleaned_result in zip(raw_checks, cleaned_checks):
        raw_status = "PASS" if raw_result.passed else "FAIL"
        cleaned_status = "PASS" if cleaned_result.passed else "FAIL"
        print(f"| {raw_result.name} | {raw_status} | {cleaned_status} |")
    print("\nDetails of checks still failing:")
    if remaining_failures:
        for result in remaining_failures:
            print(f"- {result.name}: {result.details}")
    else:
        print("None.")
    print(f"Report saved to {report_path.relative_to(report_path.parents[1])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cleaned",
        action="store_true",
        help="Compare raw checks with checks after cleaning TotalCharges.",
    )
    args = parser.parse_args()
    if args.cleaned:
        write_cleaned_report()
    else:
        write_raw_report()


if __name__ == "__main__":
    main()
