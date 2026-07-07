import os
import sqlite3

import pandas as pd

from src.exception import CustomException

MISSING_DATABASE_HINT = (
    "Inference log database not found at '{path}'. It is created by the "
    "inference API (src/serving/api.py) on its first prediction -- run the demo first."
)


def ensure_database_exists(database_path: str) -> None:
    """Raise a clear error when the inference log database does not exist yet."""
    if not os.path.exists(database_path):
        raise CustomException(MISSING_DATABASE_HINT.format(path=database_path))


def open_read_only_connection(database_path: str) -> sqlite3.Connection:
    """Open a read-only SQLite connection so no tool can mutate the log."""
    try:
        ensure_database_exists(database_path)
        return sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    except CustomException:
        raise
    except Exception as error:
        raise CustomException(str(error)) from error


def fetch_dataframe(
    database_path: str, query: str, parameters: tuple = ()
) -> pd.DataFrame:
    """Run a parameterized read-only query and return the result as a dataframe."""
    try:
        with open_read_only_connection(database_path) as connection:
            return pd.read_sql_query(query, connection, params=parameters)
    except CustomException:
        raise
    except Exception as error:
        raise CustomException(str(error)) from error


def fetch_table_columns(database_path: str, table_name: str) -> set[str]:
    """The actual column names of one table, straight from the schema.

    Used to validate sensor/feature names against what a deployment actually
    logs before building a query: SQLite falls back to treating an unmatched
    double-quoted identifier as a string literal instead of erroring, so an
    unchecked column name silently returns garbage rather than failing loudly.
    """
    try:
        with open_read_only_connection(database_path) as connection:
            cursor = connection.execute(f"SELECT * FROM {table_name} LIMIT 0")
            return {description[0] for description in cursor.description}
    except CustomException:
        raise
    except Exception as error:
        raise CustomException(str(error)) from error
