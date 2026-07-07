from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.exception import CustomException
from src_agent.config import BackendsConfig

OPENAI_BACKEND = "openai"
OLLAMA_BACKEND = "ollama"


def _resolve_backend_name(
    backend_name: str | None, backends_configuration: BackendsConfig
) -> str:
    resolved_name = backend_name or backends_configuration.default
    if resolved_name not in (OPENAI_BACKEND, OLLAMA_BACKEND):
        raise CustomException(
            f"Unknown backend '{resolved_name}'; expected one of "
            f"['{OPENAI_BACKEND}', '{OLLAMA_BACKEND}']"
        )
    return resolved_name


def build_chat_model(
    backend_name: str | None,
    backends_configuration: BackendsConfig,
    openai_api_key: str | None = None,
) -> BaseChatModel:
    """Build the chat model for the requested backend (or the configured default).

    openai_api_key only applies to the OpenAI backend; a key sent with the
    request wins over the OPENAI_API_KEY environment variable.
    """
    from src_agent.backends import ollama, openai

    resolved_name = _resolve_backend_name(backend_name, backends_configuration)
    if resolved_name == OPENAI_BACKEND:
        return openai.build_chat_model(backends_configuration.openai, openai_api_key)
    return ollama.build_chat_model(backends_configuration.ollama)


def build_caption_chat_model(
    backend_name: str | None,
    backends_configuration: BackendsConfig,
    openai_api_key: str | None = None,
) -> BaseChatModel:
    """Build the vision-captioning chat model for the requested backend (or
    the configured default).

    Ingestion captions knowledge-base images with a backend's caption_model
    role rather than its chat model role, so this swaps that in before
    delegating to the same per-backend builders as build_chat_model.
    """
    from src_agent.backends import ollama, openai

    resolved_name = _resolve_backend_name(backend_name, backends_configuration)
    if resolved_name == OPENAI_BACKEND:
        caption_configuration = backends_configuration.openai.model_copy(
            update={"model": backends_configuration.openai.caption_model}
        )
        return openai.build_chat_model(caption_configuration, openai_api_key)
    caption_configuration = backends_configuration.ollama.model_copy(
        update={"model": backends_configuration.ollama.caption_model}
    )
    return ollama.build_chat_model(caption_configuration)


def build_embeddings(
    backend_name: str | None,
    backends_configuration: BackendsConfig,
    openai_api_key: str | None = None,
) -> Embeddings:
    """Build the embeddings model of the requested backend (default: configured).

    The knowledge base embeds and retrieves with the default backend's
    embedder; the collection's embedding-model stamp guards against reading
    a collection built with a different one.
    """
    from src_agent.backends import ollama, openai

    resolved_name = _resolve_backend_name(backend_name, backends_configuration)
    if resolved_name == OPENAI_BACKEND:
        return openai.build_embeddings(backends_configuration.openai, openai_api_key)
    return ollama.build_embeddings(backends_configuration.ollama)


def resolve_embedder_name(
    backend_name: str | None, backends_configuration: BackendsConfig
) -> str:
    """The embedder model name the requested (or default) backend would use."""
    resolved_name = _resolve_backend_name(backend_name, backends_configuration)
    if resolved_name == OPENAI_BACKEND:
        return backends_configuration.openai.embedder
    return backends_configuration.ollama.embedder
