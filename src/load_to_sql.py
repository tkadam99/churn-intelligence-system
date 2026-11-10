from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db import get_engine
from src.load_data import load_raw_data


def load_csv_to_sql() -> int:
    project_root = Path(__file__).resolve().parents[1]
    create_table_sql = (project_root / "sql" / "01_create_table.sql").read_text(
        encoding="utf-8"
    )
    data = load_raw_data().astype("string")
    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(text(create_table_sql))
        connection.execute(text("TRUNCATE TABLE customers_raw"))
        data.to_sql(
            "customers_raw",
            con=connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

    return len(data)


if __name__ == "__main__":
    for run_number in (1, 2):
        loaded_rows = load_csv_to_sql()
        with get_engine().connect() as connection:
            row_count = connection.execute(
                text("SELECT COUNT(*) FROM customers_raw")
            ).scalar_one()
            print(
                f"Run {run_number}: loaded {loaded_rows} CSV rows; "
                f"SELECT COUNT(*) = {row_count}"
            )
            if run_number == 2:
                result = connection.execute(
                    text("SELECT * FROM customers_raw LIMIT 3")
                )
                preview = pd.DataFrame(result.fetchall(), columns=result.keys())
                print("First 3 rows:")
                print(preview.to_string(index=False))
