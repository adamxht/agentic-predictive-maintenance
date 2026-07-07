"""FastAPI service exposing the Diagnostic Copilot over SSE.

Stateless by design: the client resends the full conversation each request
(mirroring the inference API, which is stateless about engine windows), so
this service holds no sessions -- only the tool connections built at startup.
"""

import asyncio
import contextvars
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from src.components import deployment_logger
from src.logger import logging
from src_agent import agent as copilot
from src_agent.backends.base import build_chat_model
from src_agent.channels import SideChannel, wrap_tools_with_side_channel
from src_agent.config import (
    apply_tracing_environment_overrides,
    load_agent_service_config,
)
from src_agent.schemas import ChatRequest
from src_agent.tracing import initialize_tracing

CONFIG_PATH_ENVIRONMENT_VARIABLE = "AGENT_CONFIG_PATH"
DEFAULT_CONFIG_PATH = "configs/agent/default.yaml"

RAG_STATUS_DISABLED = "disabled"
RAG_STATUS_INGESTING = "ingesting"
RAG_STATUS_READY = "ready"
RAG_STATUS_FAILED = "failed"

# How often to check monitor/logs/ for a newly rotated file worth ingesting.
DEPLOYMENT_LOG_POLL_SECONDS = 30
# Don't retry a failed ingest on every single chat message while it's still
# broken (e.g. genuinely no key available yet) -- only this often.
RAG_RETRY_COOLDOWN_SECONDS = 30
# Reading (file loads + image captioning) vs embedding/upsert, when combining
# the two phases into one overall percentage. Captioning dominates reading's
# wall-clock cost, so this is a reasonable approximation, not a precise ETA.
READING_PHASE_WEIGHT = 0.5

load_dotenv()
configuration = load_agent_service_config(
    os.environ.get(CONFIG_PATH_ENVIRONMENT_VARIABLE, DEFAULT_CONFIG_PATH)
)
configuration.tracing = apply_tracing_environment_overrides(
    configuration.tracing, os.environ
)
initialize_tracing(configuration.tracing)


async def _load_mcp_tools() -> list:
    """Connect the MCP analytics tools.

    Failures leave the service up (health endpoint keeps working) with /chat
    returning 503, so a missing MCP server is diagnosable from the outside
    instead of crash-looping the container.
    """
    try:
        tools = await copilot.load_analytics_tools(
            configuration.agent_api.mcp_server_url
        )
        logging.info(f"Loaded {len(tools)} MCP tools")
        return tools
    except Exception as error:
        logging.error(f"Could not load MCP tools: {error}")
        return []


def _run_ingest_and_build_knowledge_search_tool(
    openai_api_key: str | None, progress_callback
):
    """Ingest the knowledge base and build its tool; runs off the event loop.

    Non-destructive (reset=False): ingest's deterministic document ids make
    re-running safe to repeat on every startup rather than wiping and
    rebuilding the whole collection each time. Returns the IngestionPipeline
    too so the caller can reuse its Chroma connection for later incremental
    per-file ingests instead of reconnecting every time. openai_api_key lets
    a caller retry with a key supplied on a request rather than only ever
    reading OPENAI_API_KEY from the environment.
    """
    from src_agent.rag.ingestion_pipeline import IngestionPipeline
    from src_agent.rag.retrieval_pipeline import RetrievalPipeline

    ingestion_pipeline = IngestionPipeline(
        configuration.rag, configuration.backends, openai_api_key
    )
    ingestion_pipeline.run(reset=False, progress_callback=progress_callback)
    tool = RetrievalPipeline(
        configuration.rag, configuration.backends, openai_api_key=openai_api_key
    ).as_tool()
    return ingestion_pipeline, tool


def _current_deployment_log_files() -> set[Path]:
    """The .log files presently in the deployment log directory, if any."""
    log_directory = Path(configuration.deployment_log_directory)
    if not log_directory.is_dir():
        return set()
    return set(log_directory.glob("*.log"))


def _deployment_logs_are_in_rag_corpus() -> bool:
    """Whether the deployment log directory is one of the configured RAG
    document_paths.

    The incremental log-watcher and "latest log" pointer document only make
    sense when it is -- otherwise there's nothing to keep in sync, and
    deliberately so: chat turns are logged to the same files as
    predictions, and re-ingesting a copilot's own past answers would let it
    cite itself as if it were primary evidence instead of the underlying
    data (see src_agent/rag/readers/text.py's chat-line filtering, which
    only matters at all if this is ever turned back on).
    """
    return configuration.deployment_log_directory in configuration.rag.document_paths


def _make_progress_recorder(application: FastAPI):
    """Build a progress_callback recording an overall ingest percentage on
    application.state as (phase, completed, total) callbacks arrive from
    IngestionPipeline.run, so /health can report it while a build is running.
    """
    from src_agent.rag.ingestion_pipeline import EMBEDDING_PHASE, READING_PHASE

    application.state.rag_ingest_percent = 0
    phase_fractions = {READING_PHASE: 0.0, EMBEDDING_PHASE: 0.0}

    def record(phase: str, completed: int, total: int) -> None:
        phase_fractions[phase] = completed / total if total else 1.0
        overall = (
            READING_PHASE_WEIGHT * phase_fractions[READING_PHASE]
            + (1 - READING_PHASE_WEIGHT) * phase_fractions[EMBEDDING_PHASE]
        )
        application.state.rag_ingest_percent = round(overall * 100)

    return record


async def _activate_knowledge_base(
    application: FastAPI, openai_api_key: str | None
) -> None:
    """Run one knowledge-base ingest attempt; on success, register the
    knowledge_search tool and start watching for new deployment logs.

    Shared by the startup task and by a later request-triggered retry: a
    failed attempt (e.g. no OpenAI key yet) just leaves rag_status FAILED for
    the next caller to retry, rather than being fatal to the service.
    """
    application.state.rag_status = RAG_STATUS_INGESTING
    progress_callback = _make_progress_recorder(application)
    try:
        ingestion_pipeline, tool = await asyncio.to_thread(
            _run_ingest_and_build_knowledge_search_tool,
            openai_api_key,
            progress_callback,
        )
    except Exception as error:
        application.state.rag_status = RAG_STATUS_FAILED
        logging.warning(f"knowledge_search disabled: {error}")
        return
    application.state.base_tools.append(tool)
    application.state.ingestion_pipeline = ingestion_pipeline
    application.state.rag_status = RAG_STATUS_READY
    application.state.rag_ingest_percent = 100
    logging.info("knowledge_search tool enabled")
    if _deployment_logs_are_in_rag_corpus():
        application.state.ingested_log_files = _current_deployment_log_files()
        await _update_latest_log_pointer(application)
        application.state.log_watch_task = asyncio.create_task(
            _watch_deployment_logs(application)
        )


async def _ingest_knowledge_base_in_background(application: FastAPI) -> None:
    """Populate the knowledge base after startup without blocking the API
    from serving the analytics tools in the meantime.
    """
    await _activate_knowledge_base(application, openai_api_key=None)


async def _retry_rag_ingestion_if_needed(
    application: FastAPI, openai_api_key: str | None
) -> None:
    """Retry a previously failed knowledge-base ingest using this request's
    key, e.g. one pasted into the UI after the service already started with
    none in the environment.

    Startup only ever tries once with OPENAI_API_KEY from the environment, so
    without this a setup that gets its key later (or only ever means to use
    the ollama backend once it owns its own caption model) would stay stuck
    at rag_status=failed until the process restarts. Cooldown- and
    lock-gated so a still-broken setup doesn't redo the attempt on every
    single chat message.
    """
    if application.state.rag_status != RAG_STATUS_FAILED:
        return
    if getattr(application.state, "rag_retrying", False):
        return
    last_attempt = getattr(application.state, "rag_last_retry_at", 0.0)
    if time.monotonic() - last_attempt < RAG_RETRY_COOLDOWN_SECONDS:
        return
    application.state.rag_retrying = True
    application.state.rag_last_retry_at = time.monotonic()
    try:
        await _activate_knowledge_base(application, openai_api_key)
    finally:
        application.state.rag_retrying = False


async def _update_latest_log_pointer(application: FastAPI) -> None:
    """Refresh the "latest log file" pointer document from the tracked set.

    Ordinary semantic search can't answer "what's the latest log file" --
    nothing in a log chunk's content says whether it's the newest one -- so
    this keeps a small dedicated document naming it, upserted in place.
    """
    log_files = application.state.ingested_log_files
    if not log_files:
        return
    latest_file_name = max(log_files, key=lambda path: path.name).name
    await asyncio.to_thread(
        application.state.ingestion_pipeline.upsert_latest_log_pointer,
        latest_file_name,
    )


async def _watch_deployment_logs(application: FastAPI) -> None:
    """Poll the deployment log directory and incrementally ingest any newly
    rotated file, so recent chat/prediction activity becomes searchable
    without waiting for a full rebuild. Runs for the life of the process.
    """
    while True:
        await asyncio.sleep(DEPLOYMENT_LOG_POLL_SECONDS)
        new_files = (
            _current_deployment_log_files() - application.state.ingested_log_files
        )
        for file_path in sorted(new_files):
            await _ingest_one_new_log_file(application, file_path)
        if new_files:
            await _update_latest_log_pointer(application)


async def _ingest_one_new_log_file(application: FastAPI, file_path: Path) -> None:
    """Ingest a single newly discovered log file, surfacing the same
    "ingesting" status the UI already shows during the initial build.
    """
    application.state.rag_status = RAG_STATUS_INGESTING
    try:
        await asyncio.to_thread(
            application.state.ingestion_pipeline.ingest_file, file_path
        )
        application.state.ingested_log_files.add(file_path)
        logging.info(f"Ingested new deployment log: {file_path.name}")
    except Exception as error:
        logging.warning(
            f"Skipping {file_path.name}, incremental ingest failed: {error}"
        )
    finally:
        application.state.rag_status = RAG_STATUS_READY


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.base_tools = await _load_mcp_tools()
    application.state.rag_status = RAG_STATUS_DISABLED
    if configuration.rag.enabled:
        application.state.rag_status = RAG_STATUS_INGESTING
        application.state.rag_ingest_task = asyncio.create_task(
            _ingest_knowledge_base_in_background(application)
        )
    yield


app = FastAPI(title="Diagnostic Copilot API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness check: tool count, the knowledge base's ingest status, and
    (while ingesting) an approximate completion percentage for its build."""
    return {
        "status": "ok",
        "tools_loaded": len(app.state.base_tools),
        "rag_status": app.state.rag_status,
        "rag_ingest_percent": getattr(app.state, "rag_ingest_percent", None),
    }


@app.get("/backends")
def backends() -> dict:
    """The chat backends a client can pick from, the default, and each
    backend's configured chat model name -- e.g. so a UI can show the user
    exactly which model is about to answer, not just "ollama" vs "openai"."""
    return {
        "available": ["ollama", "openai"],
        "default": configuration.backends.default,
        "models": {
            "ollama": configuration.backends.ollama.model,
            "openai": configuration.backends.openai.model,
        },
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    """Run one agent turn over the supplied conversation, streamed as SSE."""
    if not app.state.base_tools:
        raise HTTPException(
            status_code=503,
            detail="Agent tools unavailable -- is the MCP server running?",
        )
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")

    if configuration.rag.enabled and app.state.rag_status == RAG_STATUS_FAILED:
        # Fire-and-forget: this turn answers with whatever tools are already
        # available; a retry that succeeds enables knowledge_search starting
        # with the next turn instead of needing a restart. Kept on app.state
        # so the task isn't garbage-collected mid-run. Runs in a fresh,
        # empty Context (not the default copy-of-caller's) so it doesn't
        # inherit this request's active tracing span -- sharing one across
        # two concurrently-running tasks corrupts OpenTelemetry's context
        # stack (a "Failed to detach context" error from a span opened in
        # one task and closed from the other).
        app.state.rag_retry_task = asyncio.create_task(
            _retry_rag_ingestion_if_needed(app, request.openai_api_key),
            context=contextvars.Context(),
        )

    side_channel = SideChannel()
    resolved_backend_name = request.backend or configuration.backends.default
    chat_model = build_chat_model(
        request.backend, configuration.backends, request.openai_api_key
    )
    tools = wrap_tools_with_side_channel(app.state.base_tools, side_channel)
    agent = copilot.build_agent(chat_model, tools)
    conversation = [(message.role, message.content) for message in request.messages]
    question = conversation[-1][1]

    async def event_stream():
        final_payload = None
        try:
            events = copilot.stream_agent_events(
                agent,
                conversation,
                side_channel,
                configuration.agent_api.recursion_limit,
            )
            async for event_name, payload in events:
                if event_name == "final":
                    final_payload = payload
                yield {"event": event_name, "data": json.dumps(payload)}
        except Exception as error:
            logging.error(f"Agent turn failed: {error}")
            deployment_logger.log_chat_error_event(
                configuration.deployment_log_directory,
                resolved_backend_name,
                question,
                str(error),
            )
            yield {"event": "error", "data": json.dumps({"message": str(error)})}
            return
        if final_payload is not None:
            deployment_logger.log_chat_event(
                configuration.deployment_log_directory,
                resolved_backend_name,
                question,
                [entry["tool_name"] for entry in final_payload["tool_trace"]],
                final_payload["message"],
            )

    return EventSourceResponse(event_stream())


def main() -> None:
    """Start the Diagnostic Copilot API."""
    uvicorn.run(
        app, host=configuration.agent_api.host, port=configuration.agent_api.port
    )


if __name__ == "__main__":
    main()
