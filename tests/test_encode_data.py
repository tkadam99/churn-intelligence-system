import pandas as pd
import pytest

from src.encode_data import encode_categorical_columns, encode_yes_no_columns


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


def categorical_data():
    return pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "gender": ["Female", "Male", "Female"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
        ],
        "tenure": [1, 12, 24],
        "Churn": [0, 1, 0],
    })


def test_customer_id_is_dropped():
    # Arrange
    data = categorical_data()

    # Act
    encoded = encode_categorical_columns(data)

    # Assert
    assert "customerID" not in encoded.columns


def test_categorical_columns_are_replaced_with_expected_dummies():
    # Arrange
    data = categorical_data()
    categorical_columns = [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    # Act
    encoded = encode_categorical_columns(data)

    # Assert
    assert all(column not in encoded.columns for column in categorical_columns)
    assert "Contract_One year" in encoded.columns
    assert "Contract_Two year" in encoded.columns
    assert "Contract_Month-to-month" not in encoded.columns


def test_new_dummy_columns_are_binary_integers():
    # Arrange
    data = categorical_data()
    categorical_columns = [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    # Act
    encoded = encode_categorical_columns(data)
    dummy_columns = [
        column for column in encoded.columns
        if any(column.startswith(f"{name}_") for name in categorical_columns)
    ]

    # Assert
    assert dummy_columns
    assert all(
        pd.api.types.is_integer_dtype(encoded[column].dtype)
        for column in dummy_columns
    )
    assert all(set(encoded[column].unique()).issubset({0, 1}) for column in dummy_columns)


def test_missing_categorical_column_raises_value_error():
    # Arrange
    data = categorical_data().drop(columns=["MultipleLines"])

    # Act and Assert
    with pytest.raises(ValueError, match="MultipleLines"):
        encode_categorical_columns(data)


def test_categorical_encoder_does_not_modify_input():
    # Arrange
    data = categorical_data()
    before = data.copy(deep=True)

    # Act
    encode_categorical_columns(data)

    # Assert
    pd.testing.assert_frame_equal(data, before)


def test_non_encoded_columns_are_unchanged():
    # Arrange
    data = categorical_data()
    other_columns_before = data[["tenure", "Churn"]].copy(deep=True)

    # Act
    encoded = encode_categorical_columns(data)

    # Assert
    pd.testing.assert_frame_equal(
        encoded[["tenure", "Churn"]],
        other_columns_before,
    )
