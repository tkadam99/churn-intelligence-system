import pandas as pd

from src.clean_data import clean_total_charges


def test_whitespace_total_charges_becomes_zero():
    # Arrange
    data = pd.DataFrame({"TotalCharges": [" "]})

    # Act
    cleaned = clean_total_charges(data)

    # Assert
    assert cleaned.loc[0, "TotalCharges"] == 0.0


def test_numeric_string_becomes_float_value():
    # Arrange
    data = pd.DataFrame({"TotalCharges": ["29.85"]})

    # Act
    cleaned = clean_total_charges(data)

    # Assert
    assert cleaned.loc[0, "TotalCharges"] == 29.85


def test_total_charges_dtype_is_float():
    # Arrange
    data = pd.DataFrame({"TotalCharges": ["29.85", " "]})

    # Act
    cleaned = clean_total_charges(data)

    # Assert
    assert cleaned["TotalCharges"].dtype == float


def test_input_dataframe_is_not_modified():
    # Arrange
    data = pd.DataFrame({"TotalCharges": [" ", "29.85"], "tenure": [0, 1]})
    before = data.copy(deep=True)

    # Act
    clean_total_charges(data)

    # Assert
    pd.testing.assert_frame_equal(data, before)


def test_row_count_is_unchanged():
    # Arrange
    data = pd.DataFrame({"TotalCharges": [" ", "29.85"], "tenure": [0, 1]})

    # Act
    cleaned = clean_total_charges(data)

    # Assert
    assert len(cleaned) == len(data)


def test_other_columns_are_unchanged():
    # Arrange
    data = pd.DataFrame({
        "TotalCharges": [" ", "29.85"],
        "tenure": [0, 1],
        "customerID": ["A", "B"],
    })
    other_columns_before = data.drop(columns=["TotalCharges"]).copy(deep=True)

    # Act
    cleaned = clean_total_charges(data)

    # Assert
    pd.testing.assert_frame_equal(
        cleaned.drop(columns=["TotalCharges"]),
        other_columns_before,
    )
