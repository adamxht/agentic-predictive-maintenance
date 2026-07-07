from langchain_ollama import ChatOllama, OllamaEmbeddings

from src_agent.config import OllamaBackendConfig


def build_chat_model(backend_configuration: OllamaBackendConfig) -> ChatOllama:
    """Build a ChatOllama model pointed at the configured local Ollama server."""
    return ChatOllama(
        model=backend_configuration.model,
        base_url=backend_configuration.base_url,
    )


def build_embeddings(backend_configuration: OllamaBackendConfig) -> OllamaEmbeddings:
    """Build the local Ollama embeddings model for the knowledge base."""
    return OllamaEmbeddings(
        model=backend_configuration.embedder,
        base_url=backend_configuration.base_url,
    )
