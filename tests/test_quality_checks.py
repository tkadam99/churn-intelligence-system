import pandas as pd

from src.quality_checks import check_no_blank_strings


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


def test_missing_value_and_numbers_pass():
    # Arrange
    data = pd.DataFrame({"value": [None, 3, 7]})

    # Act
    result = check_no_blank_strings(data)

    # Assert
    assert result.passed is True
