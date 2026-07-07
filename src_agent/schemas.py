from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChartAxis(BaseModel):
    """The x axis of a chart: a label plus one value per data point."""

    label: str
    values: list[float | int | str]


class ChartSeries(BaseModel):
    """One named series of y values, aligned with the chart's x axis."""

    name: str
    values: list[float | None]


class ChartAnnotation(BaseModel):
    """A labelled marker at one x position (e.g. a drift-onset cycle)."""

    x_value: float | int | str
    label: str


class ChartSpec(BaseModel):
    """A renderer-agnostic chart description consumed by the Streamlit UI.

    Charts travel as data, never as image files: the UI renders the spec with
    Plotly, and only a compact text digest of the same data reaches the LLM.
    """

    chart_type: Literal["line", "bar"]
    title: str
    x_axis: ChartAxis
    series: list[ChartSeries]
    annotations: list[ChartAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_series_lengths(self) -> "ChartSpec":
        """Ensure every series has exactly one value per x-axis position."""
        axis_length = len(self.x_axis.values)
        for single_series in self.series:
            if len(single_series.values) != axis_length:
                raise ValueError(
                    f"Series '{single_series.name}' has {len(single_series.values)} "
                    f"values but the x axis has {axis_length}"
                )
        return self


class ChartToolResult(BaseModel):
    """What the render_chart tool returns: the spec for the UI, digest for the LLM."""

    chart_specification: ChartSpec
    digest: str


class ChatMessage(BaseModel):
    """One turn of the conversation as the stateless client resends it."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """A stateless chat request: full history plus the backend to answer with."""

    messages: list[ChatMessage]
    backend: str | None = None
    openai_api_key: str | None = None


class ToolTraceEntry(BaseModel):
    """One tool invocation as shown in the UI's investigation trace."""

    tool_name: str
    arguments: dict
    output_preview: str


class RetrievedSource(BaseModel):
    """One knowledge-base passage surfaced to the UI alongside the answer."""

    source: str
    text: str
    image_path: str | None = None


class ChatResponsePayload(BaseModel):
    """The complete result of one agent turn, sent as the final stream event."""

    message: str
    charts: list[ChartSpec] = Field(default_factory=list)
    sources: list[RetrievedSource] = Field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
