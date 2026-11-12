import json
from pathlib import Path

import joblib
import pandas as pd

from src.baseline import build_logistic_pipeline
from src.explainability import compute_shap_values
from src.error_analysis import gender, segment_error_table, tenure_bucket
from src.load_data import load_raw_data
from src.prepare_data import load_prepared_data, split_data
from src.threshold_analysis import get_oof_probabilities


def load_saved_results(project_root):
    results_directory = project_root / "reports"
    return {
        "test": json.loads(
            (results_directory / "final_test_results.json").read_text(
                encoding="utf-8"
            )
        ),
        "intervals": json.loads(
            (results_directory / "final_test_confidence_intervals.json").read_text(
                encoding="utf-8"
            )
        ),
        "config": json.loads(
            (project_root / "models" / "final_config.json").read_text(
                encoding="utf-8"
            )
        ),
    }


def compute_descriptive_tables(raw_data):
    data = raw_data.copy()
    data["churned"] = data["Churn"].eq("Yes").astype(int)
    data["tenure_bucket"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        right=True,
        ordered=True,
    )
    return {
        "overall": float(data["churned"].mean()),
        "contract": data.groupby("Contract", observed=True)["churned"].mean(),
        "tenure": data.groupby("tenure_bucket", observed=True)["churned"].mean(),
        "payment_method": data.groupby("PaymentMethod", observed=True)[
            "churned"
        ].mean(),
        "gender": data.groupby("gender", observed=True)["churned"].mean(),
    }


def compute_oof_recall_tables(threshold, project_root):
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    probabilities = get_oof_probabilities(
        build_logistic_pipeline(),
        X_train,
        y_train,
    )
    saved_pipeline = joblib.load(
        project_root / "models" / "final_logistic_pipeline.joblib"
    )
    shap_table, _ = compute_shap_values(saved_pipeline, X_train)
    top_features = (
        shap_table.abs().mean().sort_values(ascending=False).head(15).index
    )
    return {
        "tenure": segment_error_table(
            tenure_bucket(X_train),
            y_train,
            probabilities,
            threshold,
        )[["segment", "recall"]],
        "gender": segment_error_table(
            gender(X_train),
            y_train,
            probabilities,
            threshold,
        )[["segment", "recall"]],
        "gender_in_top_15_shap": "gender_Male" in top_features,
    }


def turn_numbers_into_text(saved_results, descriptive, oof_tables):
    test = saved_results["test"]
    intervals = saved_results["intervals"]["metrics"]
    config = saved_results["config"]
    threshold = config["threshold"]
    recall_interval = intervals["recall"]
    precision_interval = intervals["precision"]

    def format_table(series, label, source):
        rows = [f"| {label} | Churn rate |", "| --- | ---: |"]
        for category, rate in series.items():
            rows.append(f"| {category} | {rate:.1%} |")
            number_sources.append(
                (f"{label}: {category} churn rate", f"{rate:.1%}", source)
            )
        return "\n".join(rows)

    number_sources = [
        (
            "Frozen cutoff",
            f"{threshold:.2f}",
            "models/final_config.json: threshold",
        ),
        (
            "Overall raw-data churn rate",
            f"{descriptive['overall']:.1%}",
            "Full raw data: pandas mean of Churn == Yes",
        ),
        (
            "Test customers",
            f"{test['n_test_rows']:,}",
            "reports/final_test_results.json: n_test_rows",
        ),
        (
            "Actual test churners",
            f"{test['n_test_churners']:,}",
            "reports/final_test_results.json: n_test_churners",
        ),
        (
            "Test customers flagged",
            f"{test['flagged_customers']:,}",
            "reports/final_test_results.json: flagged_customers",
        ),
        (
            "Actual churners caught",
            f"{test['true_positives']:,}",
            "reports/final_test_results.json: true_positives",
        ),
        (
            "Actual churners missed",
            f"{test['false_negatives']:,}",
            "reports/final_test_results.json: false_negatives",
        ),
        (
            "Flagged customers who did not churn",
            f"{test['false_positives']:,}",
            "reports/final_test_results.json: false_positives",
        ),
        (
            "Test recall",
            f"{test['recall']:.1%}",
            "reports/final_test_results.json: recall",
        ),
        (
            "Test precision",
            f"{test['precision']:.1%}",
            "reports/final_test_results.json: precision",
        ),
        (
            "Recall interval low and high",
            (
                f"{recall_interval['interval_low']:.1%} to "
                f"{recall_interval['interval_high']:.1%}"
            ),
            "reports/final_test_confidence_intervals.json: recall interval",
        ),
        (
            "Precision interval low and high",
            (
                f"{precision_interval['interval_low']:.1%} to "
                f"{precision_interval['interval_high']:.1%}"
            ),
            "reports/final_test_confidence_intervals.json: precision interval",
        ),
    ]

    contract_table = format_table(
        descriptive["contract"],
        "Contract",
        "Full raw data: churn rate grouped by Contract",
    )
    tenure_table = format_table(
        descriptive["tenure"],
        "Tenure (months)",
        "Full raw data: churn rate grouped by requested tenure buckets",
    )
    payment_table = format_table(
        descriptive["payment_method"],
        "Payment method",
        "Full raw data: churn rate grouped by PaymentMethod",
    )
    female_churn_rate = descriptive["gender"].loc["Female"]
    male_churn_rate = descriptive["gender"].loc["Male"]
    gender_in_top_15 = oof_tables["gender_in_top_15_shap"]
    gender_shap_statement = (
        "appears" if gender_in_top_15 else "does not appear"
    )
    number_sources.extend(
        [
            (
                "Raw-data female churn rate",
                f"{female_churn_rate:.1%}",
                "Full raw data: pandas churn rate grouped by gender",
            ),
            (
                "Raw-data male churn rate",
                f"{male_churn_rate:.1%}",
                "Full raw data: pandas churn rate grouped by gender",
            ),
            (
                "gender_Male among top 15 mean absolute SHAP features",
                "Yes" if gender_in_top_15 else "No",
                "Saved pipeline with X_train only: mean absolute SHAP ranking",
            ),
            (
                "Features in SHAP importance ranking",
                "15",
                "Requested top-15 ranking computed from saved pipeline and X_train",
            ),
        ]
    )

    oof_tenure_rows = []
    for row in oof_tables["tenure"].itertuples(index=False):
        oof_tenure_rows.append(f"| {row.segment} months | {row.recall:.1%} |")
        number_sources.append(
            (
                f"Out-of-fold training recall, tenure {row.segment} months",
                f"{row.recall:.1%}",
                "Out-of-fold training predictions: segment_error_table and tenure_bucket",
            )
        )
    oof_gender_values = {}
    for row in oof_tables["gender"].itertuples(index=False):
        oof_gender_values[row.segment] = row.recall
        number_sources.append(
            (
                f"Out-of-fold training recall, gender {row.segment}",
                f"{row.recall:.1%}",
                "Out-of-fold training predictions: segment_error_table and gender",
            )
        )
    number_sources.extend(
        [
            (
                "Tenure bucket boundaries",
                "0-12, 13-24, 25-48, 49-72 months",
                "Requested raw-data and out-of-fold tenure grouping",
            ),
            (
                "Confidence interval level",
                "95%",
                "reports/final_test_confidence_intervals.json: percentile interval convention",
            ),
        ]
    )
    report = f"""# Customer churn model: stakeholder report

## Purpose

This project asks whether customer information can help identify customers who may leave, so a team can review those predictions and consider appropriate support. The report separates patterns seen in the customer data from how well the frozen model performed on previously unseen customers.

## Key findings about who churns

Across the full raw dataset, the observed churn rate was **{descriptive["overall"]:.1%}**. The tables below show churn rates by contract, months of tenure, and payment method. These are descriptive patterns in this dataset, not explanations of what caused a customer to leave.

{contract_table}

{tenure_table}

{payment_table}

## How well the model works on customers it has never seen

The final evaluation used **{test["n_test_rows"]:,}** held-out customers, including **{test["n_test_churners"]:,}** who actually churned. At the frozen cutoff of **{threshold:.2f}**, the model flagged **{test["flagged_customers"]:,}** customers. It caught **{test["true_positives"]:,}** actual churners and missed **{test["false_negatives"]:,}**. Of the customers flagged, **{test["false_positives"]:,}** did not churn.

**Recall** means the share of actual churners the model caught: test recall was **{test["recall"]:.1%}**, with a 95% interval from **{recall_interval["interval_low"]:.1%}** to **{recall_interval["interval_high"]:.1%}**. **Precision** means the share of flagged customers who actually churned: test precision was **{test["precision"]:.1%}**, with a 95% interval from **{precision_interval["interval_low"]:.1%}** to **{precision_interval["interval_high"]:.1%}**. These intervals describe uncertainty from sampling customers for this test evaluation; they do not include variation from training a new model.

## The trade-off

The frozen cutoff is **{threshold:.2f}**: a customer at or above that predicted probability is flagged for review. A lower cutoff can catch more potential churners, but it can also flag more customers who would not churn and may receive an unnecessary contact. A higher cutoff generally means fewer contacts but can miss more churners. The appropriate balance depends on the real cost of contacting customers and the value of retaining them; business owners must provide those inputs before choosing a cutoff for operational use. No dollar savings are estimated here.

## Limits and cautions

- The data is a public dataset, not company customer data. Results may not transfer to a particular company.
- The group rates are patterns, not causes. They do not prove why customers leave.
- The raw-data churn rate was **{female_churn_rate:.1%}** for female customers and **{male_churn_rate:.1%}** for male customers. In the training-only ranking of the 15 largest mean absolute SHAP values, the gender feature {gender_shap_statement}. A next version should drop it and then be evaluated again.
- Some input columns duplicate one another. When inputs overlap, individual feature contributions should not be read as isolated effects.
- On out-of-fold training predictions, recall for customers with 49-72 months of tenure was **{oof_tables["tenure"].loc[oof_tables["tenure"]["segment"] == "49-72", "recall"].iloc[0]:.1%}**. This figure comes from training data split into folds, where each row is predicted by a fold model that did not train on that row; it is not a test-set subgroup result.
- Gender fairness was only given a first look by comparing out-of-fold training recall: female **{oof_gender_values["Female"]:.1%}**, male **{oof_gender_values["Male"]:.1%}**. Similarity or difference in this one measure is not proof of fairness.
- The confidence intervals cover test-sample variation only. They do not account for retraining the model, changes over time, different company populations, or every source of uncertainty.

| Training group | Share of actual churners caught (recall) |
| --- | ---: |
{chr(10).join(oof_tenure_rows)}

## Suggested next steps

- Test a next model version without gender, and assess it using an appropriate evaluation plan.
- Review duplicated and overlapping input columns in a future development step; do not change this frozen model based on the test results.
- Investigate why long-standing customers are often missed, using training analysis first and a separate evaluation plan for any proposed changes.
- Ask business owners to define contact costs, customer value, and acceptable contact volume before selecting an operational cutoff.
- Validate the approach on suitable company data and continue monitoring prediction quality and fairness before using it to guide customer outreach.
"""
    return report, number_sources


def write_report(report_text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    saved_results = load_saved_results(project_root)
    descriptive_tables = compute_descriptive_tables(load_raw_data())
    oof_tables = compute_oof_recall_tables(
        saved_results["config"]["threshold"],
        project_root,
    )
    report_text, number_sources = turn_numbers_into_text(
        saved_results,
        descriptive_tables,
        oof_tables,
    )
    report_path = project_root / "reports" / "stakeholder_report.md"
    write_report(report_text, report_path)

    print(report_text)
    print(f"\nSaved report: {report_path}")
    print("\nNumber source table:")
    print("| Number or measure in report | Report value | Source |")
    print("| --- | ---: | --- |")
    for label, value, source in number_sources:
        print(f"| {label} | {value} | {source} |")
