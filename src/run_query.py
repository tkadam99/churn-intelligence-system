import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db import get_engine


def run_query_file(path: str | Path) -> pd.DataFrame:
    sql = Path(path).read_text(encoding="utf-8")
    with get_engine().connect() as connection:
        return pd.read_sql_query(text(sql), connection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SQL file and print its results.")
    parser.add_argument("path", help="Path to a .sql file")
    arguments = parser.parse_args()
    print(run_query_file(arguments.path).to_string(index=False))
