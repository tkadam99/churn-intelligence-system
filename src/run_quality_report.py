from datetime import date
from pathlib import Path

from src.load_data import load_raw_data
from src.quality_checks import run_all_checks


def main() -> None:
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


if __name__ == "__main__":
    main()
