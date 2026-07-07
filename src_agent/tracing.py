from src.logger import logging
from src_agent.config import TracingConfig


def initialize_tracing(tracing_configuration: TracingConfig) -> None:
    """Initialize OpenLIT auto-instrumentation; never breaks the service.

    Disabled config, a missing openlit package, or an unreachable collector
    all degrade to running without traces (OpenTelemetry exports drop
    silently), so chat keeps working regardless of the tracing stack.
    """
    if not tracing_configuration.enabled:
        logging.info("LLM tracing disabled by config")
        return
    try:
        import openlit

        openlit.init(
            otlp_endpoint=tracing_configuration.otlp_endpoint,
            application_name=tracing_configuration.application_name,
        )
        logging.info(
            f"OpenLIT tracing enabled -> {tracing_configuration.otlp_endpoint}"
        )
    except Exception as error:
        logging.warning(f"Tracing unavailable, continuing without it: {error}")
