"""Markdown reader that splits a file into one document per section header.

Header-based sections keep semantically complete units (a whole sensor-table
section, a whole how-to block) together, which retrieves better than
fixed-size character chunks for structured documents.
"""

import re
from pathlib import Path

from langchain_core.documents import Document

from src.exception import CustomException
from src_agent.rag.document_metadata import NO_SPLIT_METADATA_KEY

HEADER_PATTERN = re.compile(r"^#{1,6}\s+(?P<title>.+)$")
PREAMBLE_SECTION_TITLE = "preamble"


class MarkdownReader:
    """Reads a markdown file into one document per section header.

    Content before the first header becomes a "preamble" section. With
    no_split=True (the default) each document carries a no_split marker so
    the ingest pipeline keeps sections whole instead of re-splitting them
    by character count.
    """

    def __init__(self, no_split: bool = True) -> None:
        self.no_split = no_split

    def read(self, file_path: Path) -> list[Document]:
        """Read and split one markdown file into section documents."""
        try:
            markdown_text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise CustomException(
                f"Cannot read markdown file '{file_path}': {error}"
            ) from error
        return [
            Document(
                page_content=section_text,
                metadata={
                    "source": file_path.name,
                    "source_type": "document",
                    "section": section_title,
                    NO_SPLIT_METADATA_KEY: self.no_split,
                },
            )
            for section_title, section_text in self._split_sections(markdown_text)
        ]

    def _split_sections(self, markdown_text: str) -> list[tuple[str, str]]:
        """Return (title, text) per header section, including the header line."""
        sections: list[tuple[str, str]] = []
        current_title = PREAMBLE_SECTION_TITLE
        current_lines: list[str] = []
        for line in markdown_text.splitlines():
            header_match = HEADER_PATTERN.match(line)
            if header_match:
                self._append_section(sections, current_title, current_lines)
                current_title = header_match.group("title").strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        self._append_section(sections, current_title, current_lines)
        return sections

    @staticmethod
    def _append_section(
        sections: list[tuple[str, str]], title: str, lines: list[str]
    ) -> None:
        """Add a finished section, skipping ones with no actual content."""
        section_text = "\n".join(lines).strip()
        if section_text:
            sections.append((title, section_text))
