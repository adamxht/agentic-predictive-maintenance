"""Retrieval side of the knowledge base: search the Chroma collection built
by IngestionPipeline and expose it to the agent as the knowledge_search tool.
"""

import chromadb
from langchain_chroma import Chroma
from langchain_core.tools import StructuredTool

from src.exception import CustomException
from src_agent.backends.base import build_embeddings, resolve_embedder_name
from src_agent.config import BackendsConfig, RagConfig

EMBEDDING_MODEL_METADATA_KEY = "embedding_model"

KNOWLEDGE_SEARCH_DESCRIPTION = (
    "Search the project knowledge base: CMAPSS sensor descriptions (what each "
    "sensor physically measures, plausible ranges, degradation behavior), "
    "project documentation, and captioned training/evaluation plots. Use it "
    "to ground claims about sensor physics and model behavior. Returns "
    "relevant passages; any matching plot is shown to the user automatically. "
    "Not for live predictions or deployment history -- use run_sql and the "
    "other analytics tools for those."
)


class RetrievalPipeline:
    """Wraps a verified Chroma collection and exposes it as a search tool."""

    def __init__(
        self,
        rag_configuration: RagConfig,
        backends_configuration: BackendsConfig,
        chroma_client=None,
        openai_api_key: str | None = None,
    ) -> None:
        self.rag_configuration = rag_configuration
        self.backends_configuration = backends_configuration
        self._chroma_client = chroma_client or chromadb.HttpClient(
            host=rag_configuration.chroma_host, port=rag_configuration.chroma_port
        )
        self.verify_collection(
            self._chroma_client,
            rag_configuration.collection_name,
            resolve_embedder_name(None, backends_configuration),
        )
        self._vector_store = Chroma(
            client=self._chroma_client,
            collection_name=rag_configuration.collection_name,
            embedding_function=build_embeddings(
                None, backends_configuration, openai_api_key
            ),
        )

    @staticmethod
    def verify_collection(
        chroma_client, collection_name: str, expected_embedding_model: str
    ):
        """Fetch the collection and refuse to serve stale embeddings.

        A collection embedded with a different model would return
        garbage-similarity results silently, so a mismatch is a hard error
        pointing at re-ingestion.
        """
        try:
            collection = chroma_client.get_collection(collection_name)
        except Exception as error:
            raise CustomException(
                f"Knowledge base collection '{collection_name}' not found -- run "
                f"`python scripts/run_rag_ingest.py` first: {error}"
            ) from error
        stamped_model = (collection.metadata or {}).get(EMBEDDING_MODEL_METADATA_KEY)
        if stamped_model != expected_embedding_model:
            raise CustomException(
                f"Knowledge base was embedded with '{stamped_model}' but the "
                f"config expects '{expected_embedding_model}' -- re-run "
                "`python scripts/run_rag_ingest.py --reset`"
            )
        return collection

    def search(self, query: str) -> dict:
        """Search sensor documentation and captioned plots for a query."""
        matches = self._vector_store.similarity_search(
            query, k=self.rag_configuration.top_k
        )
        return {
            "passages": [
                {
                    "source": match.metadata.get("source", "unknown"),
                    "text": match.page_content,
                    "image_path": match.metadata.get("image_path"),
                }
                for match in matches
            ]
        }

    def as_tool(self) -> StructuredTool:
        """Expose this pipeline's search as the knowledge_search agent tool."""
        return StructuredTool.from_function(
            func=self.search,
            name="knowledge_search",
            description=KNOWLEDGE_SEARCH_DESCRIPTION,
        )
