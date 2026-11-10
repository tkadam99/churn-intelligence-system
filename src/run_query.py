import argparse
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db import get_engine


def run_query_file(path: str | Path, query_label: str | None = None) -> pd.DataFrame:
    sql = Path(path).read_text(encoding="utf-8")
    sections = re.split(r"(?m)^\s*--\s*Query\s+([A-Z])\b.*$", sql)
    queries = {
        sections[index].upper(): sections[index + 1].strip()
        for index in range(1, len(sections), 2)
    }

    if queries:
        if query_label is None:
            raise ValueError("This SQL file has multiple queries; choose one with --query.")
        try:
            sql = queries[query_label.upper()]
        except KeyError as error:
            available = ", ".join(queries)
            raise ValueError(
                f"Query {query_label!r} was not found. Available queries: {available}."
            ) from error
    elif query_label is not None:
        raise ValueError("No labeled queries were found in this SQL file.")

    with get_engine().connect() as connection:
        return pd.read_sql_query(text(sql), connection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SQL file and print its results.")
    parser.add_argument("path", help="Path to a .sql file")
    parser.add_argument("--query", help="Letter of a labeled query, such as A or B")
    arguments = parser.parse_args()
    print(
        run_query_file(arguments.path, arguments.query).to_string(index=False)
    )
