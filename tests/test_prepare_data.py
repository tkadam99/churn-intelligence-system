import pandas as pd

from src.prepare_data import split_data


def balanced_data():
    return pd.DataFrame(
        {
            "feature_a": range(100),
            "feature_b": [value % 3 for value in range(100)],
            "Churn": [1] * 25 + [0] * 75,
        },
        index=range(100, 200),
    )


def test_target_is_separated_from_features():
    # Arrange
    data = balanced_data()

    # Act
    X_train, X_test, y_train, y_test = split_data(data)

    # Assert
    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns
    pd.testing.assert_series_equal(y_train, data.loc[y_train.index, "Churn"])
    pd.testing.assert_series_equal(y_test, data.loc[y_test.index, "Churn"])


def test_split_preserves_row_count_and_test_share():
    # Arrange
    data = balanced_data()
    test_size = 0.2

    # Act
    X_train, X_test, y_train, y_test = split_data(data, test_size=test_size)

    # Assert
    assert len(X_train) + len(X_test) == len(data)
    assert len(y_train) + len(y_test) == len(data)
    assert len(X_test) / len(data) == test_size


def test_train_and_test_feature_columns_match():
    # Arrange
    data = balanced_data()

    # Act
    X_train, X_test, _, _ = split_data(data)

    # Assert
    assert X_train.columns.equals(X_test.columns)


def test_no_row_index_is_in_both_splits():
    # Arrange
    data = balanced_data()

    # Act
    X_train, X_test, _, _ = split_data(data)

    # Assert
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_stratification_keeps_churn_rates_close():
    # Arrange
    data = balanced_data()
    original_rate = data["Churn"].mean()

    # Act
    _, _, y_train, y_test = split_data(data)

    # Assert
    assert abs(y_train.mean() - original_rate) <= 0.02
    assert abs(y_test.mean() - original_rate) <= 0.02


def test_random_state_repeats_split_and_different_seed_changes_it():
    # Arrange
    data = balanced_data()

    # Act
    first = split_data(data, random_state=42)
    repeated = split_data(data, random_state=42)
    different = split_data(data, random_state=7)

    # Assert
    for first_part, repeated_part in zip(first, repeated):
        if isinstance(first_part, pd.Series):
            pd.testing.assert_series_equal(first_part, repeated_part)
        else:
            pd.testing.assert_frame_equal(first_part, repeated_part)
    assert not first[0].index.equals(different[0].index)
