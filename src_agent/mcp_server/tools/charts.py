import pandas as pd

from src.exception import CustomException
from src_agent.config import DriftToolConfig
from src_agent.mcp_server.tools import drift as drift_tool
from src_agent.mcp_server.tools.database import fetch_dataframe, fetch_table_columns
from src_agent.mcp_server.tools.predictions import fetch_prediction_series
from src_agent.mcp_server.tools.sensor_names import resolve_feature_names
from src_agent.mcp_server.tools.shap_trends import fetch_shap_values_with_predictions
from src_agent.schemas import ChartAxis, ChartSeries, ChartSpec, ChartToolResult

INFERENCE_READINGS_TABLE = "inference_readings"
SENSOR_TREND_CHART = "sensor_trend"
PREDICTION_TREND_CHART = "prediction_trend"
SHAP_TREND_CHART = "shap_trend"
DRIFT_Z_SCORES_CHART = "drift_z_scores"
CHART_KINDS = [
    SENSOR_TREND_CHART,
    PREDICTION_TREND_CHART,
    SHAP_TREND_CHART,
    DRIFT_Z_SCORES_CHART,
]


def render_chart(
    database_path: str,
    training_statistics: dict[str, dict[str, float]],
    drift_settings: DriftToolConfig,
    chart_kind: str,
    engine_id: int,
    feature_names: list[str] | None = None,
    cycle_range: tuple[int, int] | None = None,
) -> ChartToolResult:
    """Build a ChartSpec plus text digest for one of the supported chart kinds.

    The spec is rendered by the Streamlit UI; only the digest is meant for
    the LLM, so raw series data never enters the model context.
    """
    if feature_names:
        feature_names = resolve_feature_names(feature_names)
    if chart_kind == SENSOR_TREND_CHART:
        return _build_sensor_trend(database_path, engine_id, feature_names, cycle_range)
    if chart_kind == PREDICTION_TREND_CHART:
        return _build_prediction_trend(database_path, engine_id, cycle_range)
    if chart_kind == SHAP_TREND_CHART:
        return _build_shap_trend(database_path, engine_id, feature_names, cycle_range)
    if chart_kind == DRIFT_Z_SCORES_CHART:
        return _build_drift_z_scores(
            database_path, training_statistics, drift_settings, engine_id
        )
    raise CustomException(
        f"Unknown chart kind '{chart_kind}'. Supported kinds: {CHART_KINDS}"
    )


def _build_sensor_trend(
    database_path: str,
    engine_id: int,
    sensor_names: list[str] | None,
    cycle_range: tuple[int, int] | None,
) -> ChartToolResult:
    """Line chart of raw sensor values per cycle for one engine."""
    if not sensor_names:
        raise CustomException("sensor_trend requires feature_names (sensor columns)")
    _ensure_columns_are_logged(database_path, sensor_names)
    quoted_columns = ", ".join(f'"{name}"' for name in sensor_names)
    query = (
        f"SELECT cycle, {quoted_columns} FROM inference_readings WHERE engine_id = ?"
    )
    parameters: list = [engine_id]
    if cycle_range:
        query += " AND cycle BETWEEN ? AND ?"
        parameters.extend(cycle_range)
    query += " ORDER BY cycle"
    readings_df = fetch_dataframe(database_path, query, tuple(parameters))
    _ensure_chart_data(readings_df, engine_id)
    series = [
        ChartSeries(name=name, values=[float(v) for v in readings_df[name]])
        for name in sensor_names
    ]
    return _assemble_line_chart(
        f"Raw sensor trend -- engine {engine_id}", readings_df["cycle"], series
    )


def _build_prediction_trend(
    database_path: str, engine_id: int, cycle_range: tuple[int, int] | None
) -> ChartToolResult:
    """Line chart of the predicted life_ratio per cycle for one engine."""
    prediction_series_df = fetch_prediction_series(
        database_path, engine_id, cycle_range
    )
    _ensure_chart_data(prediction_series_df, engine_id)
    series = [
        ChartSeries(
            name="predicted_life_ratio",
            values=[float(v) for v in prediction_series_df["predicted_life_ratio"]],
        )
    ]
    return _assemble_line_chart(
        f"Predicted life ratio -- engine {engine_id}",
        prediction_series_df["cycle"],
        series,
    )


def _build_shap_trend(
    database_path: str,
    engine_id: int,
    feature_names: list[str] | None,
    cycle_range: tuple[int, int] | None,
) -> ChartToolResult:
    """Line chart of each feature's SHAP contribution per cycle for one engine."""
    shap_values_df = fetch_shap_values_with_predictions(
        database_path, engine_id, feature_names, cycle_range
    )
    _ensure_chart_data(shap_values_df, engine_id)
    shap_wide_df = shap_values_df.pivot_table(
        index="cycle", columns="feature_name", values="shap_value"
    ).sort_index()
    series = [
        ChartSeries(
            name=str(feature_name),
            values=[
                None if pd.isna(value) else float(value)
                for value in shap_wide_df[feature_name]
            ],
        )
        for feature_name in shap_wide_df.columns
    ]
    return _assemble_line_chart(
        f"SHAP contribution trend -- engine {engine_id}",
        shap_wide_df.index.to_series(),
        series,
    )


def _build_drift_z_scores(
    database_path: str,
    training_statistics: dict[str, dict[str, float]],
    drift_settings: DriftToolConfig,
    engine_id: int,
) -> ChartToolResult:
    """Bar chart of each sensor's recent window-mean z-score vs training."""
    drift_report = drift_tool.compare_to_training_distribution(
        database_path,
        training_statistics,
        engine_id,
        sensor_names=None,
        window_size=drift_settings.default_window_size,
        alert_threshold=drift_settings.z_score_alert_threshold,
    )
    if not drift_report["sensors"]:
        raise CustomException(f"No logged readings for engine {engine_id}")
    sensor_summaries = drift_report["sensors"]
    specification = ChartSpec(
        chart_type="bar",
        title=f"Sensor drift z-scores -- engine {engine_id}",
        x_axis=ChartAxis(
            label="sensor", values=[entry["sensor"] for entry in sensor_summaries]
        ),
        series=[
            ChartSeries(
                name="window_mean_z_score",
                values=[entry["window_mean_z_score"] for entry in sensor_summaries],
            )
        ],
    )
    return ChartToolResult(
        chart_specification=specification,
        digest=_build_drift_digest(specification.title, sensor_summaries),
    )


def _ensure_chart_data(chart_data_df: pd.DataFrame, engine_id: int) -> None:
    """Fail with a clear message when there is nothing to chart."""
    if chart_data_df.empty:
        raise CustomException(
            f"No logged data for engine {engine_id} -- cannot render chart"
        )


def _ensure_columns_are_logged(database_path: str, sensor_names: list[str]) -> None:
    """Fail loudly on a name that isn't a real column, before it reaches SQL.

    An unmatched quoted identifier is not an error in SQLite -- it silently
    falls back to a string literal -- so an unchecked name would otherwise
    return garbage instead of a clear failure.
    """
    logged_columns = fetch_table_columns(database_path, INFERENCE_READINGS_TABLE)
    unlogged_names = [name for name in sensor_names if name not in logged_columns]
    if unlogged_names:
        raise CustomException(
            f"Not logged sensor column(s): {unlogged_names}. Logged sensors: "
            f"{sorted(logged_columns)}"
        )


def _assemble_line_chart(
    title: str, cycle_values: pd.Series, series: list[ChartSeries]
) -> ChartToolResult:
    """Wrap cycle-indexed series into a line ChartSpec with a stats digest."""
    specification = ChartSpec(
        chart_type="line",
        title=title,
        x_axis=ChartAxis(label="cycle", values=[int(v) for v in cycle_values]),
        series=series,
    )
    return ChartToolResult(
        chart_specification=specification,
        digest=_build_series_digest(title, series),
    )


def _build_series_digest(title: str, series: list[ChartSeries]) -> str:
    """Build the compact per-series stats line meant for the LLM context."""
    series_parts = []
    for single_series in series:
        present_values = [v for v in single_series.values if v is not None]
        if not present_values:
            series_parts.append(f"{single_series.name}: no data")
            continue
        series_parts.append(
            f"{single_series.name}: latest={present_values[-1]:.4f} "
            f"mean={sum(present_values) / len(present_values):.4f} "
            f"min={min(present_values):.4f} max={max(present_values):.4f}"
        )
    return f"{title}. " + "; ".join(series_parts)


def _build_drift_digest(title: str, sensor_summaries: list[dict]) -> str:
    """Digest for the drift bar chart, calling out out-of-distribution sensors."""
    out_of_distribution_sensors = [
        entry["sensor"] for entry in sensor_summaries if entry["out_of_distribution"]
    ]
    flagged_text = (
        f"out-of-distribution sensors: {', '.join(out_of_distribution_sensors)}"
        if out_of_distribution_sensors
        else "no sensors out of distribution"
    )
    score_parts = [
        f"{entry['sensor']}={entry['window_mean_z_score']:.2f}"
        for entry in sensor_summaries
        if entry["window_mean_z_score"] is not None
    ]
    return f"{title}. {flagged_text}. window-mean z-scores: {'; '.join(score_parts)}"
