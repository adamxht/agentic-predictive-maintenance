"""The Diagnostic Copilot agent: a LangChain tool-calling agent over the
analytics MCP server, streamed as (event_name, payload) pairs for SSE."""

from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src_agent.channels import SIDE_CHANNEL_TOOL_TAG, SideChannel
from src_agent.prompts import DIAGNOSTIC_COPILOT_SYSTEM_PROMPT
from src_agent.schemas import ChatResponsePayload, ToolTraceEntry

ANALYTICS_SERVER_NAME = "cmapss-analytics"
TOOL_OUTPUT_PREVIEW_LENGTH = 400


async def load_analytics_tools(mcp_server_url: str) -> list[BaseTool]:
    """Discover the analytics tools exposed by the MCP server."""
    mcp_client = MultiServerMCPClient(
        {
            ANALYTICS_SERVER_NAME: {
                "transport": "streamable_http",
                "url": mcp_server_url,
            }
        }
    )
    return await mcp_client.get_tools()


def build_agent(chat_model: BaseChatModel, tools: list[BaseTool]):
    """Assemble the tool-calling agent with the copilot system prompt."""
    return create_agent(
        chat_model, tools, system_prompt=DIAGNOSTIC_COPILOT_SYSTEM_PROMPT
    )


async def stream_agent_events(
    agent: Any,
    conversation_messages: list[tuple[str, str]],
    side_channel: SideChannel,
    recursion_limit: int,
) -> AsyncIterator[tuple[str, dict]]:
    """Run one agent turn, yielding SSE-ready (event_name, payload) pairs.

    Emits token events for the streamed answer, tool_start/tool_end around
    each tool call, chart/sources events as the side channel fills, and one
    final event carrying the assembled ChatResponsePayload.
    """
    answer_parts: list[str] = []
    tool_trace: list[ToolTraceEntry] = []
    emitted_chart_count = 0
    emitted_source_count = 0

    event_stream = agent.astream_events(
        {"messages": conversation_messages},
        version="v2",
        config={"recursion_limit": recursion_limit},
    )
    async for event in event_stream:
        event_kind = event["event"]
        if event_kind == "on_chat_model_stream":
            token_text = _chunk_text(event["data"]["chunk"])
            if token_text:
                answer_parts.append(token_text)
                yield "token", {"content": token_text}
        elif event_kind == "on_chat_model_end":
            final_message = event["data"].get("output")
            if _message_has_tool_calls(final_message):
                answer_parts.clear()
            elif not answer_parts:
                # Non-streaming models emit no chunks; recover the text here.
                answer_parts.append(_chunk_text(final_message))
        elif event_kind == "on_tool_start" and _is_surface_tool_event(event):
            arguments = _json_safe(event["data"].get("input"))
            tool_trace.append(
                ToolTraceEntry(
                    tool_name=event["name"], arguments=arguments, output_preview=""
                )
            )
            yield "tool_start", {"tool_name": event["name"], "arguments": arguments}
        elif event_kind == "on_tool_end" and _is_surface_tool_event(event):
            preview = _output_preview(event["data"].get("output"))
            if tool_trace:
                tool_trace[-1].output_preview = preview
            yield "tool_end", {"tool_name": event["name"], "output_preview": preview}
            for chart in side_channel.charts[emitted_chart_count:]:
                yield "chart", chart.model_dump()
            emitted_chart_count = len(side_channel.charts)
            for source in side_channel.sources[emitted_source_count:]:
                yield "sources", source.model_dump()
            emitted_source_count = len(side_channel.sources)

    final_payload = ChatResponsePayload(
        message="".join(answer_parts).strip(),
        charts=side_channel.charts,
        sources=side_channel.sources,
        tool_trace=tool_trace,
    )
    yield "final", final_payload.model_dump()


def _is_surface_tool_event(event: dict) -> bool:
    """Only the tagged wrapper tools count; their nested inner runs do not."""
    return SIDE_CHANNEL_TOOL_TAG in (event.get("tags") or [])


def _chunk_text(chunk: Any) -> str:
    """Extract plain text from a streamed chunk (string or block-list content)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _message_has_tool_calls(message: Any) -> bool:
    """True when an intermediate model message only sets up tool calls."""
    return bool(getattr(message, "tool_calls", None))


def _output_preview(tool_output: Any) -> str:
    """Truncate a tool's output (already model-safe) for the UI trace."""
    content = getattr(tool_output, "content", tool_output)
    text = content if isinstance(content, str) else str(content)
    if len(text) <= TOOL_OUTPUT_PREVIEW_LENGTH:
        return text
    return text[:TOOL_OUTPUT_PREVIEW_LENGTH] + "..."


def _json_safe(value: Any) -> dict:
    """Coerce tool-call arguments into a JSON-serializable dict."""
    if isinstance(value, dict):
        return {key: _scalar_or_text(item) for key, item in value.items()}
    return {"input": _scalar_or_text(value)}


def _scalar_or_text(value: Any) -> Any:
    if isinstance(value, str | int | float | bool | list | dict) or value is None:
        return value
    return str(value)
