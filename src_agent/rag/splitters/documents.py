"""Document-level splitting stage of the ingest pipeline.

Takes whatever documents the reading stage produced and size-splits any
that need it, leaving documents the reader already sized correctly (a
markdown section, an image caption) untouched.
"""

from langchain_core.documents import Document

from src_agent.rag.document_metadata import NO_SPLIT_METADATA_KEY
from src_agent.rag.splitters.sentences import SentenceSplitter


def split_documents(
    documents: list[Document], splitter: SentenceSplitter
) -> list[Document]:
    """Split each document with splitter, except those marked no_split."""
    result: list[Document] = []
    for document in documents:
        if document.metadata.get(NO_SPLIT_METADATA_KEY):
            result.append(_without_no_split_marker(document))
        else:
            result.extend(_split_single_document(document, splitter))
    return result


def _split_single_document(
    document: Document, splitter: SentenceSplitter
) -> list[Document]:
    return [
        Document(page_content=chunk, metadata=dict(document.metadata))
        for chunk in splitter.split_text(document.page_content)
    ]


def _without_no_split_marker(document: Document) -> Document:
    metadata = dict(document.metadata)
    metadata.pop(NO_SPLIT_METADATA_KEY, None)
    return Document(page_content=document.page_content, metadata=metadata)
