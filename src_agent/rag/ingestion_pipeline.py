"""Build the multimodal knowledge base for the Diagnostic Copilot.

Three stages: read the configured document paths into per-file documents
(readers/), split any document that still needs size-splitting
(splitters/), then embed and upsert everything into the Chroma server.
The SQLite inference-log database is never a valid input -- only the
inference API's rendered .log files (and every other configured path) are.

Idempotent: document ids derive from source + position within that source,
so re-running refreshes content in place instead of duplicating it.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.exception import CustomException
from src.logger import logging
from src_agent.backends.base import (
    build_caption_chat_model,
    build_embeddings,
    resolve_embedder_name,
)
from src_agent.config import BackendsConfig, RagConfig
from src_agent.rag.readers.image import ImageCaptionReader
from src_agent.rag.readers.markdown import MarkdownReader
from src_agent.rag.readers.text import TextReader
from src_agent.rag.splitters.documents import split_documents
from src_agent.rag.splitters.sentences import SentenceSplitter

EMBEDDING_MODEL_METADATA_KEY = "embedding_model"
CHUNK_SIZE = 1000
UPSERT_BATCH_SIZE = 16
READING_PHASE = "reading"
EMBEDDING_PHASE = "embedding"
# Fixed source label for the single synthetic "pointer" document -- vector
# search can't derive facts like "which file is newest" from ordinary content
# chunks (embeddings don't encode recency), so this is upserted with a
# stable id, kept current by the caller, and named to read well in a
# references list.
LATEST_LOG_POINTER_SOURCE = "deployment_log_index"

# (phase, completed_units, total_units) -- called as ingestion progresses so a
# caller (the agent API) can surface a percentage while a run is in flight.
ProgressCallback = Callable[[str, int, int], None]


class DocumentReaderDispatcher:
    """Read stage: resolves configured paths and dispatches each file to
    the reader for its suffix (markdown, plain text/log, or image)."""

    MARKDOWN_SUFFIXES: ClassVar[set[str]] = {".md"}
    TEXT_SUFFIXES: ClassVar[set[str]] = {".txt", ".log"}
    IMAGE_SUFFIXES: ClassVar[set[str]] = {".png"}

    def __init__(
        self,
        backends_configuration: BackendsConfig,
        openai_api_key: str | None = None,
    ) -> None:
        self._backends_configuration = backends_configuration
        self._openai_api_key = openai_api_key
        self._image_reader: ImageCaptionReader | None = None

    def read_all(
        self,
        document_paths: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[Document]:
        """Read every file resolved from document_paths into documents.

        Reports "reading" progress per file (not per document) -- captioning
        an image is the slow step here, and it happens once per file.
        """
        documents: list[Document] = []
        resolved_files = self._resolve_files(document_paths)
        for completed, file_path in enumerate(resolved_files, start=1):
            documents.extend(self.read_one(file_path))
            if progress_callback:
                progress_callback(READING_PHASE, completed, len(resolved_files))
        logging.info(f"Read {len(documents)} documents from {document_paths}")
        return documents

    def _resolve_files(self, document_paths: list[str]) -> list[Path]:
        """Expand files and (non-recursive) directories into a flat file list."""
        resolved_files = []
        for entry in document_paths:
            entry_path = Path(entry)
            if entry_path.is_dir():
                resolved_files.extend(
                    sorted(path for path in entry_path.iterdir() if path.is_file())
                )
            elif entry_path.is_file():
                resolved_files.append(entry_path)
            else:
                logging.warning(f"Knowledge-base path not found, skipping: {entry}")
        return resolved_files

    def read_one(self, file_path: Path) -> list[Document]:
        """Dispatch one file to the reader for its suffix.

        Public so callers can incrementally ingest a single new file (e.g. a
        freshly rotated deployment log) without re-reading the whole corpus.
        """
        if file_path.suffix in self.MARKDOWN_SUFFIXES:
            return MarkdownReader().read(file_path)
        if file_path.suffix in self.TEXT_SUFFIXES:
            return TextReader().read(file_path)
        if file_path.suffix in self.IMAGE_SUFFIXES:
            return self._get_image_reader().read(file_path)
        logging.warning(f"Skipping unsupported knowledge-base file: {file_path}")
        return []

    def _get_image_reader(self) -> ImageCaptionReader:
        """Defer building the vision chat model until an image needs it.

        Uses the default backend's caption model (None resolves to it, same
        as the knowledge base's embedder), so an Ollama-only setup captions
        images locally instead of requiring an OpenAI key.
        """
        if self._image_reader is None:
            caption_model = build_caption_chat_model(
                None, self._backends_configuration, self._openai_api_key
            )
            self._image_reader = ImageCaptionReader(caption_model)
        return self._image_reader


class IngestionPipeline:
    """Orchestrates read -> split -> embed/upsert into the Chroma server."""

    def __init__(
        self,
        rag_configuration: RagConfig,
        backends_configuration: BackendsConfig,
        openai_api_key: str | None = None,
    ) -> None:
        self.rag_configuration = rag_configuration
        self.backends_configuration = backends_configuration
        self._openai_api_key = openai_api_key
        self._reader = DocumentReaderDispatcher(backends_configuration, openai_api_key)
        self._splitter = SentenceSplitter(chunk_size=CHUNK_SIZE)
        self._vector_store: Chroma | None = None

    def run(
        self, reset: bool = False, progress_callback: ProgressCallback | None = None
    ) -> int:
        """Run the full pipeline and return the number of documents upserted."""
        documents = self.split(self.read(progress_callback))
        vector_store = self._connect(reset)
        self._upsert_in_batches(vector_store, documents, progress_callback)
        logging.info(
            f"Upserted {len(documents)} documents into "
            f"'{self.rag_configuration.collection_name}'"
        )
        return len(documents)

    def _upsert_in_batches(
        self,
        vector_store: Chroma,
        documents: list[Document],
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Embed and upsert in fixed-size batches so progress can be reported
        between batches instead of only once the whole corpus is embedded."""
        document_ids = self.build_document_ids(documents)
        total = len(documents)
        for start in range(0, total, UPSERT_BATCH_SIZE):
            end = min(start + UPSERT_BATCH_SIZE, total)
            vector_store.add_documents(
                documents[start:end], ids=document_ids[start:end]
            )
            if progress_callback:
                progress_callback(EMBEDDING_PHASE, end, total)

    def ingest_file(self, file_path: Path) -> int:
        """Read, split, and upsert one file's documents, on their own.

        For incrementally picking up a single newly written file (e.g. one
        freshly rotated deployment log) without re-reading the rest of the
        corpus. Reuses the connection from a prior run()/ingest_file() call,
        or connects fresh (without resetting) if this is the first call.
        """
        documents = self.split(self._reader.read_one(file_path))
        if not documents:
            return 0
        vector_store = self._connect(reset=False)
        vector_store.add_documents(documents, ids=self.build_document_ids(documents))
        logging.info(
            f"Incrementally ingested {len(documents)} documents from {file_path}"
        )
        return len(documents)

    def upsert_latest_log_pointer(self, latest_log_file_name: str) -> None:
        """Keep one small synthetic document naming the newest log file.

        "What's the latest log file" can't be answered by ordinary semantic
        search over the log chunks themselves -- nothing in their content
        says which one is newest. This document exists purely to hold that
        fact; a fixed source name gives it a stable id, so re-calling this
        (e.g. after every incremental log ingest) updates it in place.
        """
        document = Document(
            page_content=(
                f"The most recently written deployment log file is "
                f"'{latest_log_file_name}' in monitor/logs/. Log files are "
                f"named <YYYY-MM-DD_HH_MM>.log and rotate every minute, so "
                f"this is also the latest deployment activity (predictions "
                f"and chat turns) reflected in the knowledge base."
            ),
            metadata={"source": LATEST_LOG_POINTER_SOURCE, "source_type": "index"},
        )
        vector_store = self._connect(reset=False)
        vector_store.add_documents([document], ids=self.build_document_ids([document]))

    def read(
        self, progress_callback: ProgressCallback | None = None
    ) -> list[Document]:
        """Stage 1: read every configured document path into documents."""
        return self._reader.read_all(
            self.rag_configuration.document_paths, progress_callback
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Stage 2: size-split any document that still needs it.

        The in-house splitter is deliberate: langchain's sentence-aware
        alternative (SentenceTransformersTokenTextSplitter) drags in torch
        just to count tokens with a tokenizer that doesn't match our
        embedders.
        """
        return split_documents(documents, self._splitter)

    def _connect(self, reset: bool) -> Chroma:
        """Stage 3: reuse the cached Chroma connection, or open a fresh one.

        Cached (not rebuilt per call) so repeated ingest_file() calls from a
        background poller don't reconnect to Chroma every time; reset=True
        always reconnects since it recreates the collection.
        """
        if self._vector_store is not None and not reset:
            return self._vector_store
        self._vector_store = self.build_vector_store(reset)
        return self._vector_store

    def build_vector_store(self, reset: bool) -> Chroma:
        """Connect to Chroma and open (or recreate) the collection."""
        try:
            chroma_client = chromadb.HttpClient(
                host=self.rag_configuration.chroma_host,
                port=self.rag_configuration.chroma_port,
            )
            chroma_client.heartbeat()
        except Exception as error:
            raise CustomException(
                f"Chroma server unreachable at {self.rag_configuration.chroma_host}:"
                f"{self.rag_configuration.chroma_port} -- start it first: {error}"
            ) from error
        if reset:
            self._delete_collection_if_present(chroma_client)
        return Chroma(
            client=chroma_client,
            collection_name=self.rag_configuration.collection_name,
            embedding_function=build_embeddings(
                None, self.backends_configuration, self._openai_api_key
            ),
            collection_metadata={
                EMBEDDING_MODEL_METADATA_KEY: resolve_embedder_name(
                    None, self.backends_configuration
                )
            },
        )

    def _delete_collection_if_present(self, chroma_client) -> None:
        """Drop the collection so ingest starts from a clean slate."""
        collection_name = self.rag_configuration.collection_name
        existing_names = [
            collection.name for collection in chroma_client.list_collections()
        ]
        if collection_name in existing_names:
            chroma_client.delete_collection(collection_name)
            logging.info(f"Deleted existing collection '{collection_name}'")

    @staticmethod
    def build_document_ids(documents: list[Document]) -> list[str]:
        """Stable ids keyed by source + position within that source.

        Per-source positions keep one file's ids independent of every other
        file, so adding or removing a source never shifts existing ids -- a
        re-ingest upserts changed chunks in place instead of duplicating them.
        """
        position_by_source: dict[str, int] = {}
        document_ids = []
        for document in documents:
            source_name = document.metadata["source"]
            position = position_by_source.get(source_name, 0)
            position_by_source[source_name] = position + 1
            key = f"{source_name}:{position}"
            document_ids.append(hashlib.sha1(key.encode()).hexdigest())
        return document_ids
