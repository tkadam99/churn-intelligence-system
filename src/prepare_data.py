from src.clean_data import clean_total_charges
from src.encode_data import encode_categorical_columns, encode_yes_no_columns
from src.load_data import load_raw_data


def load_prepared_data():
    """Prepare customer data in order:

    1. Load the raw CSV data.
    2. Clean TotalCharges.
    3. Encode Yes/No columns.
    4. One-hot encode multi-category columns and drop customerID.
    """
    data = load_raw_data()
    data = clean_total_charges(data)
    data = encode_yes_no_columns(data)
    return encode_categorical_columns(data)


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    missing_rows = int(prepared_data.isna().any(axis=1).sum())

    print(f"Shape: {prepared_data.shape}")
    print(f"Rows with any missing value: {missing_rows}")
    print("\nData type counts:")
    print(prepared_data.dtypes.value_counts().to_string())
    print("\nChurn value counts:")
    print(prepared_data["Churn"].value_counts().sort_index().to_string())
