import pandas as pd

from src.clean_data import clean_total_charges
from src.load_data import load_raw_data


def encode_yes_no_columns(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
    ]
    values = {"No": 0, "Yes": 1}

    for column in columns:
        unexpected = ~encoded[column].isin(values)
        if unexpected.any():
            bad_value = encoded.loc[unexpected, column].iloc[0]
            raise ValueError(f"Unexpected value in {column}: {bad_value!r}")
        encoded[column] = encoded[column].map(values).astype(int)

    # Churn is the target to predict; the other encoded columns are input features.
    return encoded


if __name__ == "__main__":
    raw_data = load_raw_data()
    cleaned_data = clean_total_charges(raw_data)
    encoded_data = encode_yes_no_columns(cleaned_data)
    columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
    ]

    print("Data types before encoding:")
    print(cleaned_data[columns].dtypes.to_string())
    print("\nData types after encoding:")
    print(encoded_data[columns].dtypes.to_string())
    print("\nChurn value counts after encoding:")
    print(encoded_data["Churn"].value_counts().sort_index().to_string())
    print(f"\nRows before: {len(raw_data)}")
    print(f"Rows after: {len(encoded_data)}")
