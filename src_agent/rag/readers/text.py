"""Plain-text reader for files with no header structure to preserve.

Used for .txt corpus files and the inference API's hourly .log deployment
files. Each file becomes one whole-file document, left for the splitting
stage to size-split.
"""

from pathlib import Path

from langchain_core.documents import Document

from src.components.deployment_logger import CHAT_EVENT_LOG_MARKER
from src.exception import CustomException

DEPLOYMENT_LOG_SUFFIX = ".log"


class TextReader:
    """Reads one plain-text file into a single document."""

    def read(self, file_path: Path) -> list[Document]:
        """Read one text file into a single whole-file document.

        Deployment logs have their chat-turn lines dropped first (see
        _without_chat_lines): otherwise a copilot answer gets logged,
        embedded into the knowledge base, and later surfaces as a
        "source" for a similar question -- the model then paraphrases
        its own past answer instead of the primary evidence, and the UI
        cites a log file that looks unrelated to the user. Prediction
        telemetry lines stay; they're legitimate deployment context.
        """
        try:
            file_text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise CustomException(
                f"Cannot read text file '{file_path}': {error}"
            ) from error
        is_deployment_log = file_path.suffix == DEPLOYMENT_LOG_SUFFIX
        if is_deployment_log:
            file_text = _without_chat_lines(file_text)
        if not file_text.strip():
            return []
        source_type = "deployment_log" if is_deployment_log else "document"
        return [
            Document(
                page_content=file_text,
                metadata={"source": file_path.name, "source_type": source_type},
            )
        ]


def _without_chat_lines(file_text: str) -> str:
    """Drop every chat-turn line, keeping only prediction telemetry lines."""
    return "\n".join(
        line for line in file_text.splitlines() if CHAT_EVENT_LOG_MARKER not in line
    )
