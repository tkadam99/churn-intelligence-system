import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL


def get_engine() -> Engine:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    required = ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing database settings in .env: {', '.join(missing)}")

    database_url = URL.create(
        "postgresql+psycopg2",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        database=os.environ["POSTGRES_DB"],
    )
    return create_engine(database_url)


if __name__ == "__main__":
    with get_engine().connect() as connection:
        version = connection.execute(text("SELECT version()")).scalar_one()
        print(f"PostgreSQL version: {version}")
