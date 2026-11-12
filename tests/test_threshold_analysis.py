import pandas as pd
import pytest

from src.threshold_analysis import add_net_benefit, threshold_table


def test_threshold_table_calculates_metrics_and_handles_no_flags():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    probabilities = [0.9, 0.8, 0.4, 0.6, 0.3, 0.2, 0.1, 0.05]

    table = threshold_table(y_true, probabilities, [0.5, 0.3, 0.95])

    at_half = table.loc[table["threshold"] == 0.5].iloc[0]
    assert at_half["flagged_customers"] == 3
    assert at_half["precision"] == pytest.approx(2 / 3)
    assert at_half["recall"] == pytest.approx(2 / 3)

    at_point_three = table.loc[table["threshold"] == 0.3].iloc[0]
    assert at_point_three["flagged_customers"] == 5
    assert at_point_three["precision"] == pytest.approx(0.6)
    assert at_point_three["recall"] == pytest.approx(1.0)

    at_point_nine_five = table.loc[table["threshold"] == 0.95].iloc[0]
    assert at_point_nine_five["flagged_customers"] == 0
    assert at_point_nine_five["precision"] == 0


def test_add_net_benefit_calculates_counts_and_benefit():
    table = pd.DataFrame(
        [{"flagged_customers": 5, "recall": 1.0}]
    )

    result = add_net_benefit(
        table,
        n_churners=3,
        contact_cost=5,
        retention_success_rate=0.3,
        customer_value=500,
    )

    row = result.iloc[0]
    assert row["true_positives"] == 3
    assert row["false_positives"] == 2
    assert row["net_benefit"] == 425


def test_add_net_benefit_returns_copy_without_modifying_input():
    table = pd.DataFrame(
        [{"flagged_customers": 5, "recall": 1.0}]
    )
    original = table.copy(deep=True)

    result = add_net_benefit(
        table,
        n_churners=3,
        contact_cost=5,
        retention_success_rate=0.3,
        customer_value=500,
    )

    pd.testing.assert_frame_equal(table, original)
    assert result is not table
