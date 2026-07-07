import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.exception import CustomException
from src_agent.config import OpenAiBackendConfig

API_KEY_ENVIRONMENT_VARIABLE = "OPENAI_API_KEY"


def _resolve_api_key(api_key: str | None) -> str:
    """Prefer a per-request key over the environment/.env variable."""
    resolved_api_key = api_key or os.environ.get(API_KEY_ENVIRONMENT_VARIABLE)
    if not resolved_api_key:
        raise CustomException(
            "No OpenAI API key: provide one with the request or set "
            f"{API_KEY_ENVIRONMENT_VARIABLE} in the environment/.env"
        )
    return resolved_api_key


def build_chat_model(
    backend_configuration: OpenAiBackendConfig, api_key: str | None = None
) -> ChatOpenAI:
    """Build a ChatOpenAI model for this backend's configured chat model."""
    return ChatOpenAI(
        model=backend_configuration.model, api_key=_resolve_api_key(api_key)
    )


def build_embeddings(
    backend_configuration: OpenAiBackendConfig, api_key: str | None = None
) -> OpenAIEmbeddings:
    """Build the OpenAI embeddings model for the knowledge base."""
    return OpenAIEmbeddings(
        model=backend_configuration.embedder, api_key=_resolve_api_key(api_key)
    )
