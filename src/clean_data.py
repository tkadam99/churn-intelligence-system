import pandas as pd

from src.load_data import load_raw_data
from src.quality_checks import run_all_checks


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    blank_mask = (
        cleaned["TotalCharges"]
        .astype("string")
        .str.strip()
        .eq("")
        .fillna(False)
    )

    # Assumption: all 11 blank rows have tenure 0, so no billing cycle has completed.
    cleaned.loc[blank_mask, "TotalCharges"] = "0"
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"]).astype(float)
    return cleaned


if __name__ == "__main__":
    data = load_raw_data()
    blank_mask = (
        data["TotalCharges"].astype("string").str.strip().eq("").fillna(False)
    )
    cleaned_data = clean_total_charges(data)

    print(f"TotalCharges dtype before: {data['TotalCharges'].dtype}")
    print(f"TotalCharges dtype after: {cleaned_data['TotalCharges'].dtype}")
    print(f"Rows before: {len(data)}")
    print(f"Rows after: {len(cleaned_data)}")
    print("\nAffected rows after cleaning:")
    print(
        cleaned_data.loc[
            blank_mask, ["customerID", "tenure", "TotalCharges"]
        ].to_string(index=False)
    )

    print("\nQuality checks on cleaned data:")
    for result in run_all_checks(cleaned_data):
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name}: {status} - {result.details}")
