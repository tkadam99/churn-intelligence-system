import pandas as pd
import pytest

from src.encode_data import encode_yes_no_columns


ENCODED_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]


def valid_data():
    return pd.DataFrame({
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "No"],
        "PaperlessBilling": ["No", "Yes"],
        "Churn": ["No", "Yes"],
        "tenure": [12, 24],
        "Contract": ["One year", "Two year"],
    })


def test_yes_no_values_are_encoded_in_all_five_columns():
    # Arrange
    data = valid_data()

    # Act
    encoded = encode_yes_no_columns(data)

    # Assert
    assert encoded[ENCODED_COLUMNS].values.tolist() == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1],
    ]


def test_encoded_columns_have_integer_dtype():
    # Arrange
    data = valid_data()

    # Act
    encoded = encode_yes_no_columns(data)

    # Assert
    assert all(
        pd.api.types.is_integer_dtype(encoded[column].dtype)
        for column in ENCODED_COLUMNS
    )


def test_unexpected_value_raises_value_error_with_column_name():
    # Arrange
    data = valid_data()
    data.loc[0, "Partner"] = "Yess"

    # Act and Assert
    with pytest.raises(ValueError, match="Partner"):
        encode_yes_no_columns(data)


def test_missing_value_raises_value_error():
    # Arrange
    data = valid_data()
    data.loc[0, "Churn"] = None

    # Act and Assert
    with pytest.raises(ValueError, match="Churn"):
        encode_yes_no_columns(data)


def test_input_dataframe_is_not_modified():
    # Arrange
    data = valid_data()
    before = data.copy(deep=True)

    # Act
    encode_yes_no_columns(data)

    # Assert
    pd.testing.assert_frame_equal(data, before)


def test_unencoded_columns_are_unchanged():
    # Arrange
    data = valid_data()
    other_columns_before = data[["tenure", "Contract"]].copy(deep=True)

    # Act
    encoded = encode_yes_no_columns(data)

    # Assert
    pd.testing.assert_frame_equal(
        encoded[["tenure", "Contract"]],
        other_columns_before,
    )
