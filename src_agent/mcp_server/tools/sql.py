import sqlite3
from collections.abc import Callable

from src.exception import CustomException
from src_agent.mcp_server.tools.database import open_read_only_connection

ALLOWED_QUERY_PREFIXES = ("select", "with")

READ_ONLY_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_FUNCTION,
}


def _build_table_authorizer(
    allowed_tables: list[str],
) -> Callable[[int, str | None, str | None, str | None, str | None], int]:
    """Build a SQLite authorizer that only permits reads on allowlisted tables."""

    def authorize(
        action: int,
        first_argument: str | None,
        second_argument: str | None,
        database_name: str | None,
        trigger_name: str | None,
    ) -> int:
        if action in READ_ONLY_ALLOWED_ACTIONS:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if first_argument in allowed_tables:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_DENY

    return authorize


def _validate_query_shape(query: str) -> str:
    """Reject multi-statement input and anything that is not a SELECT query."""
    normalized_query = query.strip().rstrip(";").strip()
    if not normalized_query:
        raise CustomException("Query is empty")
    if ";" in normalized_query:
        raise CustomException("Only a single SQL statement is allowed")
    if not normalized_query.lower().startswith(ALLOWED_QUERY_PREFIXES):
        raise CustomException("Only SELECT queries are allowed")
    return normalized_query


def run_read_only_query(
    database_path: str,
    query: str,
    allowed_tables: list[str],
    max_rows: int,
) -> dict:
    """Run a guarded SELECT against the inference log and return rows as dicts.

    Guard rails: read-only connection, single-statement SELECT/CTE only, a
    SQLite authorizer that denies reads outside allowed_tables, and a hard
    row limit with a truncation flag.
    """
    validated_query = _validate_query_shape(query)
    try:
        with open_read_only_connection(database_path) as connection:
            connection.set_authorizer(_build_table_authorizer(allowed_tables))
            cursor = connection.execute(validated_query)
            column_names = [description[0] for description in cursor.description]
            fetched_rows = cursor.fetchmany(max_rows + 1)
    except CustomException:
        raise
    except Exception as error:
        raise CustomException(str(error)) from error
    truncated = len(fetched_rows) > max_rows
    returned_rows = fetched_rows[:max_rows]
    return {
        "columns": column_names,
        "rows": [list(row) for row in returned_rows],
        "row_count": len(returned_rows),
        "truncated": truncated,
    }
