from pathlib import Path

import pandas as pd


def load_raw_data() -> pd.DataFrame:
    data_folder = Path(__file__).resolve().parents[1] / "data" / "raw"
    csv_files = sorted(data_folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_folder}. Add one raw CSV file and try again."
        )
    if len(csv_files) > 1:
        raise ValueError(f"Expected one CSV file in {data_folder}, found {len(csv_files)}.")

    return pd.read_csv(csv_files[0])


if __name__ == "__main__":
    data = load_raw_data()
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print("Column names:")
    print(data.columns.tolist())
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(data.head(5).to_string(index=False))
    print("\nMissing values per column:")
    print(data.isna().sum().to_string())
