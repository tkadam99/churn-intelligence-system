import pandas as pd
from sklearn.model_selection import train_test_split

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


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    features = df.drop(columns=["Churn"])
    target = df["Churn"]
    # Stratifying keeps the churn/no-churn proportions similar in both sets.
    # A fixed random_state makes the split repeatable across runs.
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, X_test, y_train, y_test = split_data(prepared_data)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    for name, target in [("y_train", y_train), ("y_test", y_test)]:
        churners = int(target.sum())
        churn_rate = target.mean() * 100
        print(f"{name} churners: {churners}")
        print(f"{name} churn rate: {churn_rate:.2f}%")

    same_columns = X_train.columns.equals(X_test.columns)
    target_excluded = "Churn" not in X_train.columns and "Churn" not in X_test.columns
    print(f"X_train and X_test have the same columns: {same_columns}")
    print(f"Churn is excluded from X_train and X_test: {target_excluded}")
