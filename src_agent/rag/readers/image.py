"""Vision-captioning reader for plot images in the knowledge base.

Each image becomes one document whose page_content is a retrieval-friendly
caption generated once at ingest time; the source image path is kept in
metadata so the UI can render the actual plot alongside the caption text.
"""

import base64
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.exception import CustomException
from src_agent.rag.document_metadata import NO_SPLIT_METADATA_KEY

CAPTION_PROMPT = (
    "This image comes from a predictive-maintenance machine learning project "
    "(remaining-useful-life prediction on NASA CMAPSS turbofan data). Describe "
    "what the image shows -- the kind of plot or screenshot, its axes or key "
    "elements, and any notable pattern or takeaway -- in two to four sentences, "
    "so the description can be retrieved later by semantic search. If it has "
    "labeled data points or bars (e.g. per-engine or per-feature values), read "
    "the axis labels precisely and name the specific ones at the extremes (e.g. "
    "which exact engine ID or feature has the highest/lowest value) instead of "
    "describing the pattern only in general terms -- a question naming a "
    "specific id or value should be answerable from the caption alone."
)


class ImageCaptionReader:
    """Reads one image into a single captioned document, via a vision model."""

    def __init__(self, caption_model: BaseChatModel) -> None:
        self.caption_model = caption_model

    def read(self, file_path: Path) -> list[Document]:
        """Caption one image and wrap it as a single, whole document."""
        caption_text = self._caption(file_path)
        return [
            Document(
                page_content=f"Plot '{file_path.name}': {caption_text}",
                metadata={
                    "source": file_path.name,
                    "source_type": "plot",
                    "image_path": str(file_path),
                    NO_SPLIT_METADATA_KEY: True,
                },
            )
        ]

    def _caption(self, image_path: Path) -> str:
        try:
            encoded_image = base64.b64encode(image_path.read_bytes()).decode()
            message = HumanMessage(
                content=[
                    {"type": "text", "text": CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ]
            )
            return self.caption_model.invoke([message]).text
        except Exception as error:
            raise CustomException(
                f"Captioning {image_path.name} failed: {error}"
            ) from error
