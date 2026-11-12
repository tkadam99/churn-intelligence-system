import math

import pandas as pd

from src.error_analysis import contract_type, segment_error_table, tenure_bucket


def test_tenure_bucket_includes_each_boundary_in_the_expected_group():
    features = pd.DataFrame(
        {"tenure": [0, 12, 13, 24, 25, 48, 49, 72]}
    )

    buckets = tenure_bucket(features)

    assert buckets.astype(object).tolist() == [
        "0-12",
        "0-12",
        "13-24",
        "13-24",
        "25-48",
        "25-48",
        "49-72",
        "49-72",
    ]


def test_contract_type_uses_indicator_columns_and_month_to_month_default():
    features = pd.DataFrame(
        {
            "Contract_One year": [0, 1, 0],
            "Contract_Two year": [0, 0, 1],
        }
    )

    types = contract_type(features)

    assert types.astype(object).tolist() == [
        "Month-to-month",
        "One year",
        "Two year",
    ]


def test_segment_error_table_counts_and_nan_precision_are_hand_checkable():
    segments = pd.Series(
        pd.Categorical(
            ["A", "A", "A", "B", "B"],
            categories=["A", "B"],
            ordered=True,
        )
    )
    y_true = [1, 1, 0, 1, 0]
    probabilities = [0.9, 0.2, 0.8, 0.1, 0.3]

    # A: 3 customers, 2 churners, 2 flagged, 1 true positive, 1 missed.
    # B: 2 customers, 1 churner, no flags, no true positives, 1 missed.
    table = segment_error_table(
        segments,
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert table["customers"].tolist() == [3, 2]
    assert table["actual_churners"].tolist() == [2, 1]
    assert table["flagged_customers"].tolist() == [2, 0]
    assert table["true_positives"].tolist() == [1, 0]
    assert table["false_negatives"].tolist() == [1, 1]
    assert table.loc[0, "precision"] == 0.5
    assert math.isnan(table.loc[1, "precision"])
