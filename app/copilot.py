"""Diagnostic Copilot chat panel for the Streamlit demo.

Talks to the agent API (src_agent/api.py) over SSE: streams the answer
tokens live, shows each tool call as it happens, then renders the final
payload -- ChartSpecs as altair charts, retrieved plot images, and the
investigation trace. Degrades to a hint when the agent service is down, so
the base demo never depends on it.
"""

import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

AGENT_API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8300")
CHAT_TIMEOUT_SECONDS = 300
CHAT_HISTORY_HEIGHT_PIXELS = 480
OFFLINE_HINT = (
    "The Diagnostic Copilot service is offline. Start the MCP server and the "
    "agent API (see README) to enable it."
)
# Mirrors src_agent/api.py's RAG_STATUS_* constants -- that's the source of
# truth, this side just compares against the string it sends over /health.
RAG_STATUS_INGESTING = "ingesting"
RAG_STATUS_FAILED = "failed"
INGESTING_HINT = "Building knowledge base (other features still work)"
INGESTING_STATUS_POLL_SECONDS = 3
OLLAMA_RAG_CAVEAT = (
    "Knowledge-base search (`knowledge_search`) isn't available right now. "
    "It's still built with the server's *default* backend (OpenAI) even "
    "when you chat with Ollama here, so without an OpenAI key on the server "
    "it may never populate -- Ollama answers can't cite the knowledge base "
    "until that's resolved."
)


@st.cache_data(ttl=INGESTING_STATUS_POLL_SECONDS, show_spinner=False)
def fetch_agent_status(agent_api_url: str) -> dict | None:
    """Probe the agent API health endpoint; None when unreachable."""
    try:
        response = requests.get(f"{agent_api_url}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_available_backends(agent_api_url: str) -> dict:
    """Fetch which chat backends the agent API offers and its default."""
    response = requests.get(f"{agent_api_url}/backends", timeout=5)
    response.raise_for_status()
    return response.json()


def _render_ingesting_status(agent_api_url: str) -> None:
    """Spinner banner while the knowledge base is still building.

    Re-fetches on its own poll timer (see run_every above) rather than the
    outer render's cached status, so it self-clears within a few seconds of
    ingestion finishing instead of needing a page refresh or interaction.
    """
    agent_status = fetch_agent_status(agent_api_url)
    if agent_status is not None and agent_status.get("rag_status") == (
        RAG_STATUS_INGESTING
    ):
        percent = agent_status.get("rag_ingest_percent")
        label = INGESTING_HINT
        if percent is not None:
            label = f"{INGESTING_HINT} -- {percent}%"
        st.status(label, state="running", type="compact")


def _is_agent_ready(agent_status: dict | None) -> bool:
    """Whether the panel has enough to show more than the offline/no-tools hint."""
    return agent_status is not None and agent_status.get("tools_loaded", 0) > 0


def _watch_for_agent_startup(agent_api_url: str) -> None:
    """While showing the offline/no-tools hint, poll until the agent service
    (MCP server + agent API) actually comes up, then force a full rerun.

    Without this, the hint would only ever be re-checked on the next user
    interaction (Streamlit doesn't rerun a script on its own) -- so if both
    servers finish starting up while a user is just looking at the page, it
    would keep claiming they're offline until something like a manual
    refresh happened to trigger a fresh check. The "was it ready before"
    guard keeps this a one-shot transition instead of rerunning forever.
    """
    if st.session_state.get("copilot_agent_was_ready", False):
        return
    if _is_agent_ready(fetch_agent_status(agent_api_url)):
        st.session_state.copilot_agent_was_ready = True
        st.rerun()


def render_copilot_panel() -> None:
    """Render the full copilot chat panel (or its offline/degraded hints)."""
    st.fragment(run_every=INGESTING_STATUS_POLL_SECONDS)(_watch_for_agent_startup)(
        AGENT_API_URL
    )

    agent_status = fetch_agent_status(AGENT_API_URL)
    if agent_status is None:
        st.info(OFFLINE_HINT)
        return
    if agent_status.get("tools_loaded", 0) == 0:
        st.warning(
            "The copilot is running but has no tools -- is the MCP server "
            "(and Chroma) up? Restart the agent API once they are."
        )
        return
    # A plain check here would only ever refresh on the next user
    # interaction (Streamlit doesn't rerun a script on its own), so the
    # banner would keep showing long after ingestion actually finished.
    # st.fragment(run_every=...) polls independently of the rest of the page.
    st.fragment(run_every=INGESTING_STATUS_POLL_SECONDS)(_render_ingesting_status)(
        AGENT_API_URL
    )

    backend_name, openai_api_key = _render_backend_controls()
    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []

    # Placed before the history container (and this panel lives inside a
    # st.tabs() tab, where chat_input renders inline rather than pinned to
    # the viewport) so it stays put at the top instead of drifting further
    # down the page as the conversation grows.
    question = st.chat_input("Ask about an engine - e.g. 'What is the status of engine 75?'")

    # autoscroll=True (rather than the height+chat_message auto-detect
    # default) so the box reliably jumps to the newest message instead of
    # staying scrolled at the top when history is re-rendered on a rerun.
    with st.container(height=CHAT_HISTORY_HEIGHT_PIXELS, border=True, autoscroll=True):
        for history_entry in st.session_state.copilot_history:
            _render_history_entry(history_entry)
        if question:
            _run_agent_turn(question, backend_name, openai_api_key)


def _render_backend_controls() -> tuple[str, str | None]:
    """Backend picker plus the API-key field the OpenAI backend requires."""
    st.caption(
        "Disclaimer: LLM responses are not always perfect, verify important findings "
        "against the underlying data before acting on them."
    )
    backends = fetch_available_backends(AGENT_API_URL)
    picker_column, key_column = st.columns([1, 2])
    with picker_column:
        backend_name = st.selectbox(
            "Backend (openai recommended)",
            backends["available"],
            index=backends["available"].index(backends["default"]),
            help=(
                "OpenAI: generally better responses, costs per request. "
                "Ollama: free and local, but response quality is not guaranteed."
            ),
        )
        st.caption(f"Model: `{backends['models'][backend_name]}`")
    openai_api_key = None
    if backend_name == "openai":
        with key_column:
            openai_api_key = (
                st.text_input(
                    "OpenAI API key (will use .env's key if found)",
                    type="password",
                    help="Sent only with your requests, never stored. Leave "
                    "empty to use the server's configured key, if any.",
                )
                or None
            )
    elif backend_name == "ollama":
        _render_ollama_rag_caveat_if_unavailable()
    return backend_name, openai_api_key


def _render_ollama_rag_caveat_if_unavailable() -> None:
    """Warn that knowledge_search may be unusable while chatting with Ollama.

    The knowledge base is ingested with whichever backend is the server's
    *default* (see configs/agent/*.yaml), not the backend picked in this UI,
    so an Ollama-only setup with no OpenAI key never gets one built.
    """
    agent_status = fetch_agent_status(AGENT_API_URL)
    if agent_status is not None and agent_status.get("rag_status") == RAG_STATUS_FAILED:
        st.warning(OLLAMA_RAG_CAVEAT)


def _run_agent_turn(
    question: str, backend_name: str, openai_api_key: str | None
) -> None:
    """Send one question to the agent and stream its investigation live."""
    st.session_state.copilot_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        activity = st.status("Investigating...", expanded=False)
        answer_placeholder = st.empty()
        final_payload = _consume_event_stream(
            question, backend_name, openai_api_key, activity, answer_placeholder
        )
    if final_payload is not None:
        st.session_state.copilot_history.append(
            {
                "role": "assistant",
                "content": final_payload["message"],
                "charts": final_payload["charts"],
                "sources": final_payload["sources"],
                "tool_trace": final_payload["tool_trace"],
            }
        )
        st.rerun()


def _consume_event_stream(
    question: str,
    backend_name: str,
    openai_api_key: str | None,
    activity,
    answer_placeholder,
) -> dict | None:
    """Drive the SSE stream, updating the live widgets; returns the final payload."""
    request_payload = {
        "messages": _conversation_payload(question),
        "backend": backend_name,
        "openai_api_key": openai_api_key,
    }
    streamed_text = ""
    try:
        for event_name, event_data in _stream_chat_events(request_payload):
            if event_name == "tool_start":
                activity.update(label=f"Running {event_data['tool_name']}...")
                activity.write(f"`{event_data['tool_name']}` {event_data['arguments']}")
            elif event_name == "token":
                streamed_text += event_data["content"]
                answer_placeholder.markdown(streamed_text + "▌")
            elif event_name == "error":
                activity.update(label="Investigation failed", state="error")
                st.error(event_data["message"])
                return None
            elif event_name == "final":
                activity.update(label="Investigation complete", state="complete")
                return event_data
    except requests.RequestException as error:
        activity.update(label="Investigation failed", state="error")
        st.error(f"Could not reach the copilot: {error}")
    return None


def _conversation_payload(question: str) -> list[dict]:
    """The stateless full-history messages array, ending with the new question."""
    return [
        {"role": entry["role"], "content": entry["content"]}
        for entry in st.session_state.copilot_history
        if entry["role"] in ("user", "assistant")
    ] + [{"role": "user", "content": question}]


def _stream_chat_events(request_payload: dict):
    """Yield (event_name, data) pairs from the agent API's SSE response."""
    with requests.post(
        f"{AGENT_API_URL}/chat",
        json=request_payload,
        stream=True,
        timeout=CHAT_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        event_name = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("event:"):
                event_name = raw_line.split(":", 1)[1].strip()
            elif raw_line.startswith("data:") and event_name:
                yield event_name, json.loads(raw_line.split(":", 1)[1])


EMPTY_RESPONSE_FALLBACK = (
    "_The model returned an empty response. This happens occasionally with "
    "local models. Please try asking again._"
)


def _render_history_entry(history_entry: dict) -> None:
    """Render one past turn: text, references, charts, retrieved plots, trace."""
    with st.chat_message(history_entry["role"]):
        st.markdown(history_entry["content"].strip() or EMPTY_RESPONSE_FALLBACK)
        _render_references(history_entry.get("sources", []))
        for chart_specification in history_entry.get("charts", []):
            _render_chart_specification(chart_specification)
        _render_retrieved_images(history_entry.get("sources", []))
        _render_tool_trace(history_entry.get("tool_trace", []))


def _render_references(sources: list[dict]) -> None:
    """List the unique knowledge-base sources behind an answer, if any.

    Only knowledge_search populates "sources", so this is the tell that RAG
    was actually used to ground the answer (as opposed to just the
    deterministic analytics tools).
    """
    if not sources:
        return
    unique_source_names = list(dict.fromkeys(source["source"] for source in sources))
    reference_list = ", ".join(f"`{name}`" for name in unique_source_names)
    st.caption(f"**References:** {reference_list}")


def _render_chart_specification(chart_specification: dict) -> None:
    """Render a ChartSpec from the agent as an altair chart."""
    x_label = chart_specification["x_axis"]["label"]
    long_format_rows = [
        {x_label: x_value, "value": y_value, "series": series["name"]}
        for series in chart_specification["series"]
        for x_value, y_value in zip(
            chart_specification["x_axis"]["values"], series["values"], strict=False
        )
        if y_value is not None
    ]
    if not long_format_rows:
        return
    chart_dataframe = pd.DataFrame(long_format_rows)
    x_type = "Q" if chart_specification["chart_type"] == "line" else "N"
    mark = (
        alt.Chart(chart_dataframe).mark_line()
        if chart_specification["chart_type"] == "line"
        else alt.Chart(chart_dataframe).mark_bar()
    )
    st.caption(chart_specification["title"])
    st.altair_chart(
        mark.encode(
            x=alt.X(f"{x_label}:{x_type}"),
            y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
            color="series:N",
        ).properties(height=220),
        width="stretch",
    )


def _render_retrieved_images(sources: list[dict]) -> None:
    """Show each retrieved plot image once, with its source name."""
    seen_paths = set()
    for source in sources:
        image_path = source.get("image_path")
        if image_path and image_path not in seen_paths and os.path.exists(image_path):
            seen_paths.add(image_path)
            st.image(image_path, caption=source["source"])


def _render_tool_trace(tool_trace: list[dict]) -> None:
    """Collapsible list of the tool calls behind an answer."""
    if not tool_trace:
        return
    with st.expander(f"Investigation trace ({len(tool_trace)} tool calls)"):
        for step in tool_trace:
            st.markdown(f"**{step['tool_name']}** `{step['arguments']}`")
            if step["output_preview"]:
                st.code(step["output_preview"], language="text")
