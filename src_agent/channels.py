"""Split tool results into what the LLM sees vs what only the UI should get.

Chart tools return a full ChartSpec plus a text digest; retrieval returns
passages plus image paths. The model only ever receives the text parts --
raw series data and file paths stay out of its context and are collected on
a per-request SideChannel for the streaming endpoint to forward to the UI.
"""

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from src.logger import logging
from src_agent.schemas import ChartSpec, RetrievedSource

CHART_SPECIFICATION_KEY = "chart_specification"
PASSAGES_KEY = "passages"

# Marks the outer wrapper tools so the event stream can ignore the nested
# inner-tool runs (both carry the same tool name).
SIDE_CHANNEL_TOOL_TAG = "side-channel-tool"


class SideChannel:
    """Per-request collector for artifacts that bypass the LLM context."""

    def __init__(self) -> None:
        self.charts: list[ChartSpec] = []
        self.sources: list[RetrievedSource] = []


def wrap_tools_with_side_channel(
    tools: list[BaseTool], side_channel: SideChannel
) -> list[BaseTool]:
    """Wrap every tool so its result is split before reaching the model."""
    return [_wrap_single_tool(tool, side_channel) for tool in tools]


def _wrap_single_tool(tool: BaseTool, side_channel: SideChannel) -> BaseTool:
    """Return a delegate tool that filters the wrapped tool's output."""

    async def invoke_and_split(**tool_arguments: Any) -> str:
        raw_result = await tool.ainvoke(tool_arguments)
        return split_result_for_model(raw_result, side_channel)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=invoke_and_split,
        tags=[SIDE_CHANNEL_TOOL_TAG],
    )


def split_result_for_model(raw_result: Any, side_channel: SideChannel) -> str:
    """Route UI-only artifacts to the side channel and return the model's text."""
    parsed_result = _parse_result(raw_result)
    if not isinstance(parsed_result, dict):
        return _as_text(raw_result)
    if CHART_SPECIFICATION_KEY in parsed_result:
        return _split_chart_result(parsed_result, side_channel)
    if PASSAGES_KEY in parsed_result:
        return _split_retrieval_result(parsed_result, side_channel)
    return _as_text(raw_result)


def _split_chart_result(parsed_result: dict, side_channel: SideChannel) -> str:
    """Collect the chart spec for the UI; the model gets only the digest."""
    try:
        side_channel.charts.append(ChartSpec(**parsed_result[CHART_SPECIFICATION_KEY]))
    except Exception as error:
        logging.warning(f"Discarding malformed chart specification: {error}")
    return parsed_result.get("digest", "Chart rendered for the user.")


def _split_retrieval_result(parsed_result: dict, side_channel: SideChannel) -> str:
    """Collect sources (with image paths) for the UI; the model gets the text."""
    passage_texts = []
    for passage in parsed_result[PASSAGES_KEY]:
        source = RetrievedSource(**passage)
        side_channel.sources.append(source)
        passage_texts.append(f"[{source.source}] {source.text}")
    if not passage_texts:
        return "No relevant knowledge-base passages found."
    return "\n\n".join(passage_texts)


def _parse_result(raw_result: Any) -> Any:
    """Best-effort structural parse of a tool result.

    Native tools return dicts; MCP tools return either JSON text or a list
    of content blocks whose text is JSON.
    """
    if isinstance(raw_result, dict):
        return raw_result
    text = _content_text(raw_result)
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _content_text(raw_result: Any) -> str:
    """Flatten a string or MCP content-block list into plain text."""
    if isinstance(raw_result, str):
        return raw_result
    if isinstance(raw_result, list):
        return "".join(
            block.get("text", "")
            for block in raw_result
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _as_text(raw_result: Any) -> str:
    """Render a tool result as the plain text the model will read."""
    text = _content_text(raw_result)
    if text:
        return text
    return json.dumps(raw_result, default=str)
