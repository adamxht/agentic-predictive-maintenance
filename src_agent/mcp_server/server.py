import os

from mcp.server.fastmcp import FastMCP

from src.exception import CustomException
from src.logger import logging
from src_agent.config import load_agent_service_config
from src_agent.mcp_server.tools import charts, drift, predictions, shap_trends, sql

CONFIG_PATH_ENVIRONMENT_VARIABLE = "AGENT_CONFIG_PATH"
DEFAULT_CONFIG_PATH = "configs/agent/default.yaml"

configuration = load_agent_service_config(
    os.environ.get(CONFIG_PATH_ENVIRONMENT_VARIABLE, DEFAULT_CONFIG_PATH)
)
training_statistics = drift.load_training_statistics(
    configuration.training_statistics_path
)

mcp_application = FastMCP(
    "cmapss-analytics",
    host=configuration.mcp_server.host,
    port=configuration.mcp_server.port,
)


def _as_cycle_range(cycle_range: list[int] | None) -> tuple[int, int] | None:
    """Validate an optional [start, end] cycle range argument."""
    if cycle_range is None:
        return None
    if len(cycle_range) != 2:
        raise CustomException("cycle_range must be [start_cycle, end_cycle]")
    return (cycle_range[0], cycle_range[1])


@mcp_application.tool()
def get_shap_evidence_profile(engine_id: int) -> dict:
    """Report which evidence drives one engine's predictions per life phase.

    Splits the engine's logged cycles into early/mid/late phases (by predicted
    life_ratio) and reports, per phase, the share of total absolute SHAP
    carried by the `cycle` feature vs sensor-derived features. Use this first
    to judge trust: in cycle-dominated phases (early/late life) the prediction
    is blind to sensor anomalies; in sensor-driven phases (mid life) SHAP
    trends are credible degradation evidence. Also returns the engine's
    current phase and latest predicted life_ratio.
    """
    return shap_trends.get_shap_evidence_profile(
        configuration.database_path, engine_id, configuration.life_phase_bands
    )


@mcp_application.tool()
def get_shap_trend(
    engine_id: int,
    feature_names: list[str] | None = None,
    cycle_range: list[int] | None = None,
) -> dict:
    """Report how each feature's SHAP contribution evolves for one engine.

    Returns, per feature: mean absolute SHAP, the latest signed SHAP value,
    first-half vs second-half mean absolute SHAP over the window, and a
    rising/falling/flat direction. Optionally restrict to specific
    feature_names and/or a [start, end] cycle_range.
    """
    return shap_trends.get_shap_trend(
        configuration.database_path,
        engine_id,
        feature_names,
        _as_cycle_range(cycle_range),
    )


@mcp_application.tool()
def compare_to_training_distribution(
    engine_id: int,
    sensor_names: list[str] | None = None,
    window_size: int | None = None,
) -> dict:
    """Check whether an engine's recent raw readings drifted out of distribution.

    Computes z-scores of the recent window mean and latest value of each raw
    sensor against the training data's per-sensor mean/std, flagging sensors
    whose |z| reaches the alert threshold. This is the deterministic
    out-of-distribution detector: use it to distinguish sensor faults or data
    drift from normal readings, independent of what the model predicts.
    Defaults to all sensors and the configured window size.
    """
    return drift.compare_to_training_distribution(
        configuration.database_path,
        training_statistics,
        engine_id,
        sensor_names,
        window_size or configuration.drift_tool.default_window_size,
        configuration.drift_tool.z_score_alert_threshold,
    )


@mcp_application.tool()
def get_prediction_trend(
    engine_id: int,
    cycle_range: list[int] | None = None,
    recent_window_size: int = 10,
) -> dict:
    """Report an engine's predicted life_ratio per cycle plus a trend summary.

    The summary includes the latest prediction, the change over the recent
    window, and the min/max seen. Use it to contrast prediction stability
    with sensor behavior (e.g. a sensor anomaly the prediction ignored).
    """
    return predictions.get_prediction_trend(
        configuration.database_path,
        engine_id,
        _as_cycle_range(cycle_range),
        recent_window_size,
    )


@mcp_application.tool()
def run_sql(query: str) -> dict:
    """Run one read-only SELECT query against the inference log database.

    Tables: inference_readings (one row per prediction: engine_id, cycle, one
    column per raw sensor, predicted_life_ratio, timestamp) and
    inference_shap_values (one row per feature per prediction: engine_id,
    cycle, feature_name, shap_value, timestamp). Only single-statement
    SELECT/CTE queries on these tables are allowed; results are row-limited.
    Prefer the dedicated tools; use this for anything they cannot answer.
    """
    return sql.run_read_only_query(
        configuration.database_path,
        query,
        configuration.sql_tool.allowed_tables,
        configuration.sql_tool.max_rows,
    )


@mcp_application.tool()
def render_chart(
    chart_kind: str,
    engine_id: int,
    feature_names: list[str] | None = None,
    cycle_range: list[int] | None = None,
) -> dict:
    """Build a chart for the UI and return only a text digest of it to you.

    chart_kind is one of: sensor_trend (raw sensor values per cycle; requires
    feature_names), prediction_trend (predicted life_ratio per cycle),
    shap_trend (SHAP contribution per cycle, optionally filtered by
    feature_names), drift_z_scores (bar chart of each sensor's window-mean
    z-score vs training). The full chart is shown to the user automatically;
    reason from the returned digest only.
    """
    chart_result = charts.render_chart(
        configuration.database_path,
        training_statistics,
        configuration.drift_tool,
        chart_kind,
        engine_id,
        feature_names,
        _as_cycle_range(cycle_range),
    )
    return chart_result.model_dump()


def main() -> None:
    """Start the analytics MCP server over streamable HTTP."""
    logging.info(
        f"Starting cmapss-analytics MCP server on "
        f"{configuration.mcp_server.host}:{configuration.mcp_server.port}"
    )
    mcp_application.run(transport="streamable-http")


if __name__ == "__main__":
    main()
