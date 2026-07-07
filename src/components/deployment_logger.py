"""Minute-rotated deployment log shared by the inference API and the agent.

Every prediction (app/api.py) and every chat turn (src_agent/api.py) appends
one structured, human-readable line to <log_directory>/<YYYY-MM-DD_HH_MM>.log.
Plain text keeps the deployment history consumable by ops tooling and the
RAG knowledge base without touching either service's SQLite store, and
minute-sized files keep each file a small, coherent chunk for retrieval
instead of one large multi-hour blob.
"""

import os
from datetime import UTC, datetime

from src.exception import CustomException

LOG_FILE_TIMESTAMP_FORMAT = "%Y-%m-%d_%H_%M"
QUESTION_LOG_PREVIEW_LENGTH = 200
ANSWER_LOG_PREVIEW_LENGTH = 200
# Marks a chat-turn line (as opposed to a prediction line) within a log
# file -- the RAG ingestion reader (src_agent/rag/readers/text.py) matches
# on this to drop chat lines before embedding, so a copilot answer never
# gets re-surfaced as if it were primary evidence for a later question.
CHAT_EVENT_LOG_MARKER = " chat backend="


def current_log_file_path(log_directory: str) -> str:
    """Path of the log file for the current UTC minute."""
    file_name = datetime.now(UTC).strftime(LOG_FILE_TIMESTAMP_FORMAT) + ".log"
    return os.path.join(log_directory, file_name)


def log_prediction_event(
    log_directory: str,
    engine_id: int,
    cycle: int,
    sensor_readings: dict[str, float],
    predicted_life_ratio: float,
    life_ratio_threshold: float,
) -> None:
    """Append one structured prediction line to the current log file.

    Predictions below life_ratio_threshold are logged at WARNING with a
    near_failure marker so scanning tools (and retrieval) can find them.
    """
    near_failure = predicted_life_ratio < life_ratio_threshold
    readings_text = " ".join(
        f"{sensor_name}={float(value):.4f}"
        for sensor_name, value in sensor_readings.items()
    )
    level = "WARNING" if near_failure else "INFO"
    line = (
        f"{level} prediction engine_id={engine_id} cycle={cycle} "
        f"predicted_life_ratio={predicted_life_ratio:.6f} "
        f"near_failure={str(near_failure).lower()} {readings_text}"
    )
    _append_log_line(log_directory, line)


def log_chat_event(
    log_directory: str,
    backend_name: str,
    question: str,
    tool_names: list[str],
    answer: str,
) -> None:
    """Append one structured line for a completed Diagnostic Copilot turn."""
    unique_tool_names = list(dict.fromkeys(tool_names))
    line = (
        f"INFO{CHAT_EVENT_LOG_MARKER}{backend_name} "
        f'question="{_preview(question, QUESTION_LOG_PREVIEW_LENGTH)}" '
        f"tools={','.join(unique_tool_names) or 'none'} "
        f'answer="{_preview(answer, ANSWER_LOG_PREVIEW_LENGTH)}"'
    )
    _append_log_line(log_directory, line)


def log_chat_error_event(
    log_directory: str, backend_name: str, question: str, error_message: str
) -> None:
    """Append one structured line for a Diagnostic Copilot turn that failed."""
    line = (
        f"ERROR{CHAT_EVENT_LOG_MARKER}{backend_name} "
        f'question="{_preview(question, QUESTION_LOG_PREVIEW_LENGTH)}" '
        f'error="{_preview(error_message, ANSWER_LOG_PREVIEW_LENGTH)}"'
    )
    _append_log_line(log_directory, line)


def _preview(text: str, max_length: int) -> str:
    """Single-line, length-capped rendering of free text for a log line."""
    single_line = " ".join(text.split())
    if len(single_line) <= max_length:
        return single_line
    return single_line[: max_length - 1] + "…"


def _append_log_line(log_directory: str, line: str) -> None:
    """Append one already-formatted line, prefixed with its own timestamp."""
    try:
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        os.makedirs(log_directory, exist_ok=True)
        with open(current_log_file_path(log_directory), "a") as log_file:
            log_file.write(f"{timestamp} {line}\n")
    except Exception as error:
        raise CustomException(str(error)) from error
