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


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
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

    for column in categorical_columns:
        if column not in encoded.columns:
            raise ValueError(f"Required categorical column is missing: {column}")

    # customerID is an identifier, not a feature, and a model could memorize it.
    encoded = encoded.drop(columns=["customerID"])

    # The dropped category is implied when all other one-hot category columns are 0.
    return pd.get_dummies(
        encoded,
        columns=categorical_columns,
        dtype=int,
        drop_first=True,
    )


if __name__ == "__main__":
    raw_data = load_raw_data()
    cleaned_data = clean_total_charges(raw_data)
    encoded_data = encode_yes_no_columns(cleaned_data)
    final_data = encode_categorical_columns(encoded_data)

    print(f"Shape before encoding: {raw_data.shape}")
    print(f"Shape after encoding: {final_data.shape}")
    print("\nColumn names after encoding:")
    print(final_data.columns.tolist())
    print(f"\nFeature columns (excluding Churn): {len(final_data.columns) - 1}")
    print("\nData type counts after encoding:")
    print(final_data.dtypes.value_counts().to_string())
