import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline import build_logistic_pipeline
from src.prepare_data import load_prepared_data, split_data
from src.threshold_analysis import get_oof_probabilities


def segment_error_table(segment, y_true, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    actual = np.asarray(y_true)
    rows = []

    if isinstance(segment.dtype, pd.CategoricalDtype):
        segment_values = [
            value for value in segment.cat.categories if segment.eq(value).any()
        ]
    else:
        segment_values = segment.dropna().drop_duplicates().tolist()

    for value in segment_values:
        in_segment = segment.eq(value).to_numpy()
        segment_actual = actual[in_segment]
        segment_predictions = predictions[in_segment]
        customers = len(segment_actual)
        actual_churners = int((segment_actual == 1).sum())
        flagged_customers = int(segment_predictions.sum())
        true_positives = int(
            ((segment_actual == 1) & (segment_predictions == 1)).sum()
        )

        rows.append(
            {
                "segment": value,
                "customers": customers,
                "actual_churners": actual_churners,
                "actual_churn_rate": actual_churners / customers,
                "flagged_customers": flagged_customers,
                "true_positives": true_positives,
                "false_negatives": actual_churners - true_positives,
                "recall": (
                    true_positives / actual_churners
                    if actual_churners
                    else np.nan
                ),
                "precision": (
                    true_positives / flagged_customers
                    if flagged_customers
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "segment",
            "customers",
            "actual_churners",
            "actual_churn_rate",
            "flagged_customers",
            "true_positives",
            "false_negatives",
            "recall",
            "precision",
        ],
    )


def tenure_bucket(X):
    labels = ["0-12", "13-24", "25-48", "49-72"]
    buckets = pd.cut(
        X["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=labels,
        right=True,
    )
    return pd.Series(buckets, index=X.index, name="tenure_bucket")


def contract_type(X):
    contract = np.select(
        [
            X["Contract_One year"].eq(1),
            X["Contract_Two year"].eq(1),
        ],
        ["One year", "Two year"],
        default="Month-to-month",
    )
    return pd.Series(
        pd.Categorical(
            contract,
            categories=["Month-to-month", "One year", "Two year"],
            ordered=True,
        ),
        index=X.index,
        name="contract_type",
    )


def gender(X):
    values = np.where(X["gender_Male"].eq(1), "Male", "Female")
    return pd.Series(
        pd.Categorical(values, categories=["Female", "Male"], ordered=True),
        index=X.index,
        name="gender",
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    model = build_logistic_pipeline()
    probabilities = get_oof_probabilities(model, X_train, y_train)
    config = json.loads(
        (project_root / "models" / "final_config.json").read_text(
            encoding="utf-8"
        )
    )
    threshold = config["threshold"]

    print(f"Out-of-fold training error analysis (frozen threshold: {threshold:.2f})")
    segments = [
        ("Tenure bucket", tenure_bucket(X_train)),
        ("Contract type", contract_type(X_train)),
        ("Gender", gender(X_train)),
    ]
    tables = {}
    for name, segment in segments:
        table = segment_error_table(segment, y_train, probabilities, threshold)
        tables[name] = table
        print(f"\n{name}:")
        print(table.round(3).to_string(index=False))

    total_false_negatives = int(tables["Tenure bucket"]["false_negatives"].sum())
    print(f"\nTotal false negatives: {total_false_negatives}")
    for name in ["Tenure bucket", "Contract type"]:
        table = tables[name]
        shares = table[["segment", "false_negatives"]].copy()
        shares["share_of_false_negatives"] = (
            shares["false_negatives"] / total_false_negatives
            if total_false_negatives
            else np.nan
        )
        print(f"\nShare of false negatives by {name.lower()}:")
        print(shares.round(3).to_string(index=False))
