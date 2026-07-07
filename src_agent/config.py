from collections.abc import Mapping

import yaml
from pydantic import BaseModel, Field, model_validator

from src.exception import CustomException

DEFAULT_ALLOWED_TABLES = ["inference_readings", "inference_shap_values"]


class LifePhaseBandsConfig(BaseModel):
    """Predicted life_ratio thresholds that split an engine's life into phases.

    A cycle is "early" when its predicted life_ratio is above
    early_minimum_life_ratio, "late" when below late_maximum_life_ratio, and
    "mid" otherwise. Bands are based on the model's own prediction because
    live engines are censored: their true maximum cycle is unknown.
    """

    early_minimum_life_ratio: float = 0.7
    late_maximum_life_ratio: float = 0.3

    @model_validator(mode="after")
    def validate_band_ordering(self) -> "LifePhaseBandsConfig":
        """Ensure the late threshold sits below the early threshold."""
        if self.late_maximum_life_ratio >= self.early_minimum_life_ratio:
            raise ValueError(
                "late_maximum_life_ratio must be below early_minimum_life_ratio"
            )
        return self


class SqlToolConfig(BaseModel):
    """Guard rails for the read-only SQL tool."""

    allowed_tables: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ALLOWED_TABLES)
    )
    max_rows: int = 200


class DriftToolConfig(BaseModel):
    """Settings for comparing live readings against the training distribution."""

    default_window_size: int = 20
    z_score_alert_threshold: float = 4.0


class McpServerConfig(BaseModel):
    """Network settings for the analytics MCP server."""

    host: str = "0.0.0.0"
    port: int = 8200


class OpenAiBackendConfig(BaseModel):
    """Settings for the OpenAI chat backend."""

    model: str = "gpt-5-mini"
    caption_model: str = "gpt-5-mini"
    embedder: str = "text-embedding-3-small"


class OllamaBackendConfig(BaseModel):
    """Settings for the local Ollama chat backend."""

    model: str = "qwen3.5:9b"
    base_url: str = "http://localhost:11434"
    embedder: str = "qwen3-embedding:4b"
    caption_model: str = "qwen3.5:9b"


class BackendsConfig(BaseModel):
    """Which backends are available and which one is the default.

    Every backend carries its own model roles: chat model, embedder (the
    default backend's embedder is what the knowledge base uses, for both
    ingest and retrieval), and a vision caption model used once at
    knowledge-base ingest time -- also picked from the default backend, so
    an Ollama-only setup never needs an OpenAI key to build the knowledge
    base.
    """

    default: str = "openai"
    openai: OpenAiBackendConfig = Field(default_factory=OpenAiBackendConfig)
    ollama: OllamaBackendConfig = Field(default_factory=OllamaBackendConfig)


class AgentApiConfig(BaseModel):
    """Network and loop settings for the agent chat API."""

    host: str = "0.0.0.0"
    port: int = 8300
    mcp_server_url: str = "http://localhost:8200/mcp"
    recursion_limit: int = 25


# monitor/logs is deliberately not here: chat turns are logged to the same
# rotating files as predictions, and indexing them would let the copilot
# retrieve (and paraphrase) its own past answers as if they were primary
# evidence. Add it back only with src_agent/rag/readers/text.py's chat-line
# filtering in mind -- see _deployment_logs_are_in_rag_corpus in api.py,
# which gates the incremental log-watcher on this same list.
DEFAULT_DOCUMENT_PATHS = [
    "rag_documents",
    "README.md",
]


class RagConfig(BaseModel):
    """Settings for the multimodal knowledge base (Chroma server mode).

    document_paths entries may be files or directories. Directories are
    scanned non-recursively: markdown/text/log files are chunked as text
    (.log files tagged as deployment logs) and PNGs are captioned by the
    vision model. Model choices live under backends, not here.
    """

    enabled: bool = True
    chroma_host: str = "localhost"
    chroma_port: int = 8100
    collection_name: str = "cmapss_knowledge"
    top_k: int = 4
    document_paths: list[str] = Field(
        default_factory=lambda: list(DEFAULT_DOCUMENT_PATHS)
    )


class TracingConfig(BaseModel):
    """OpenLIT tracing settings; disabled by default and always fail-open."""

    enabled: bool = False
    otlp_endpoint: str = "http://localhost:4318"
    application_name: str = "diagnostic-copilot"


TRACING_ENABLED_ENVIRONMENT_VARIABLE = "AGENT_TRACING_ENABLED"
TRACING_OTLP_ENDPOINT_ENVIRONMENT_VARIABLE = "AGENT_TRACING_OTLP_ENDPOINT"


def apply_tracing_environment_overrides(
    tracing_configuration: TracingConfig, environment: Mapping[str, str]
) -> TracingConfig:
    """Let two env vars toggle tracing without a second Docker config file.

    The two agent compose variants (plain vs. with-tracing) share one image
    and one config; they only differ in these two environment variables.
    """
    updates: dict[str, bool | str] = {}
    if TRACING_ENABLED_ENVIRONMENT_VARIABLE in environment:
        updates["enabled"] = (
            environment[TRACING_ENABLED_ENVIRONMENT_VARIABLE].lower() == "true"
        )
    if TRACING_OTLP_ENDPOINT_ENVIRONMENT_VARIABLE in environment:
        updates["otlp_endpoint"] = environment[
            TRACING_OTLP_ENDPOINT_ENVIRONMENT_VARIABLE
        ]
    return tracing_configuration.model_copy(update=updates)


class AgentServiceConfig(BaseModel):
    """Settings shared by the MCP analytics server and the agent service."""

    database_path: str = "monitor/inference_log.db"
    training_statistics_path: str = "configs/agent/training_statistics.json"
    # Same directory app/api.py writes prediction events to, and RagConfig's
    # default document_paths scans -- chat and prediction history end up in
    # the same rotating log files, one shared deployment record for RAG.
    deployment_log_directory: str = "monitor/logs"
    life_phase_bands: LifePhaseBandsConfig = Field(default_factory=LifePhaseBandsConfig)
    sql_tool: SqlToolConfig = Field(default_factory=SqlToolConfig)
    drift_tool: DriftToolConfig = Field(default_factory=DriftToolConfig)
    mcp_server: McpServerConfig = Field(default_factory=McpServerConfig)
    agent_api: AgentApiConfig = Field(default_factory=AgentApiConfig)
    backends: BackendsConfig = Field(default_factory=BackendsConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)


def load_agent_service_config(config_path: str) -> AgentServiceConfig:
    """Load and validate the agent service config from a YAML file."""
    try:
        with open(config_path) as config_file:
            raw_configuration = yaml.safe_load(config_file) or {}
        return AgentServiceConfig(**raw_configuration)
    except Exception as error:
        raise CustomException(str(error)) from error
