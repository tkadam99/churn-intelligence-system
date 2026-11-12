import pandas as pd

from src.clean_data import clean_total_charges
from src.quality_checks import (
    check_allowed_category_values,
    check_monthly_charges_positive,
    check_no_blank_strings,
    check_no_duplicate_rows,
    check_senior_citizen_values,
    check_tenure_range,
    check_unique_customer_ids,
)


def test_no_blank_strings_passes():
    # Arrange
    data = pd.DataFrame({"name": ["Alex", "Sam"]})

    # Act
    result = check_no_blank_strings(data)

    # Assert
    assert result.passed is True


def test_empty_string_fails_and_names_column():
    # Arrange
    data = pd.DataFrame({"notes": [""]})

    # Act
    result = check_no_blank_strings(data)

    # Assert
    assert result.passed is False
    assert "notes" in result.details


def test_whitespace_only_string_fails():
    # Arrange
    data = pd.DataFrame({"notes": [" "]})

    # Act
    result = check_no_blank_strings(data)

    # Assert
    assert result.passed is False


def test_clean_total_charges_fixes_whitespace_only_quality_failure():
    data = pd.DataFrame({"TotalCharges": [" "]})

    raw_result = check_no_blank_strings(data)
    cleaned_result = check_no_blank_strings(clean_total_charges(data))

    assert raw_result.passed is False
    assert cleaned_result.passed is True


def test_missing_value_and_numbers_pass():
    # Arrange
    data = pd.DataFrame({"value": [None, 3, 7]})

    # Act
    result = check_no_blank_strings(data)

    # Assert
    assert result.passed is True


def test_unique_customer_ids_passes():
    # Arrange
    data = pd.DataFrame({"customerID": ["A", "B"]})

    # Act
    result = check_unique_customer_ids(data)

    # Assert
    assert result.passed is True


def test_duplicate_customer_ids_fail():
    # Arrange
    data = pd.DataFrame({"customerID": ["A", "A"]})

    # Act
    result = check_unique_customer_ids(data)

    # Assert
    assert result.passed is False
    assert "1 duplicate customerID" in result.details


def test_no_duplicate_rows_passes():
    # Arrange
    data = pd.DataFrame({"value": [1, 2]})

    # Act
    result = check_no_duplicate_rows(data)

    # Assert
    assert result.passed is True


def test_duplicate_rows_fail():
    # Arrange
    data = pd.DataFrame({"value": [1, 1]})

    # Act
    result = check_no_duplicate_rows(data)

    # Assert
    assert result.passed is False
    assert "1 fully duplicate rows" in result.details


def test_tenure_range_passes():
    # Arrange
    data = pd.DataFrame({"tenure": [0, 36, 72]})

    # Act
    result = check_tenure_range(data)

    # Assert
    assert result.passed is True


def test_tenure_above_range_fails():
    # Arrange
    data = pd.DataFrame({"tenure": [80]})

    # Act
    result = check_tenure_range(data)

    # Assert
    assert result.passed is False
    assert "1 values outside 0-72" in result.details


def test_monthly_charges_positive_passes():
    # Arrange
    data = pd.DataFrame({"MonthlyCharges": [18.25, 50.0]})

    # Act
    result = check_monthly_charges_positive(data)

    # Assert
    assert result.passed is True


def test_zero_monthly_charges_fail():
    # Arrange
    data = pd.DataFrame({"MonthlyCharges": [0]})

    # Act
    result = check_monthly_charges_positive(data)

    # Assert
    assert result.passed is False
    assert "1 values are not greater than 0" in result.details


def test_senior_citizen_values_pass():
    # Arrange
    data = pd.DataFrame({"SeniorCitizen": [0, 1]})

    # Act
    result = check_senior_citizen_values(data)

    # Assert
    assert result.passed is True


def test_unexpected_senior_citizen_value_fails():
    # Arrange
    data = pd.DataFrame({"SeniorCitizen": [2]})

    # Act
    result = check_senior_citizen_values(data)

    # Assert
    assert result.passed is False
    assert "1 values are not 0 or 1" in result.details


def test_allowed_category_values_pass():
    # Arrange
    data = pd.DataFrame({
        "Churn": ["Yes"],
        "Contract": ["Month-to-month"],
        "gender": ["Female"],
        "Partner": ["No"],
        "Dependents": ["Yes"],
        "PhoneService": ["No"],
        "PaperlessBilling": ["Yes"],
    })

    # Act
    result = check_allowed_category_values(data)

    # Assert
    assert result.passed is True


def test_unexpected_category_value_fails():
    # Arrange
    data = pd.DataFrame({
        "Churn": ["Maybe"],
        "Contract": ["Month-to-month"],
        "gender": ["Female"],
        "Partner": ["No"],
        "Dependents": ["Yes"],
        "PhoneService": ["No"],
        "PaperlessBilling": ["Yes"],
    })

    # Act
    result = check_allowed_category_values(data)

    # Assert
    assert result.passed is False
    assert "Churn: Maybe (1)" in result.details
