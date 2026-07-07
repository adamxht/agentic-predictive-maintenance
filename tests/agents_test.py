import asyncio
import contextlib
import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from src.components import deployment_logger, inference_store
from src.exception import CustomException
from src_agent.agent import build_agent, stream_agent_events
from src_agent.backends.base import build_caption_chat_model, build_chat_model
from src_agent.channels import (
    SideChannel,
    split_result_for_model,
    wrap_tools_with_side_channel,
)
from src_agent.config import (
    BackendsConfig,
    DriftToolConfig,
    LifePhaseBandsConfig,
    RagConfig,
    TracingConfig,
    apply_tracing_environment_overrides,
    load_agent_service_config,
)
from src_agent.mcp_server.tools import charts, drift, predictions, shap_trends, sql
from src_agent.rag.ingestion_pipeline import DocumentReaderDispatcher, IngestionPipeline
from src_agent.rag.readers.markdown import MarkdownReader
from src_agent.rag.readers.text import TextReader
from src_agent.rag.retrieval_pipeline import RetrievalPipeline
from src_agent.rag.splitters.sentences import SentenceSplitter
from src_agent.schemas import ChartAxis, ChartSeries, ChartSpec
from src_agent.tracing import initialize_tracing

SENSOR_COLUMNS = ["T24", "Ps30", "T2"]
ALLOWED_TABLES = ["inference_readings", "inference_shap_values"]

ENGINE_ONE_ID = 1
ENGINE_ONE_PREDICTIONS = [0.9, 0.8, 0.75, 0.5, 0.5, 0.5, 0.2, 0.1, 0.05]
ENGINE_ONE_T24_READINGS = [20.0] * 8 + [0.0]
ENGINE_ONE_PHASES = ["early"] * 3 + ["mid"] * 3 + ["late"] * 3
SHAP_VALUES_BY_PHASE = {
    "early": {"cycle": 0.8, "T24": 0.1, "Ps30": 0.1},
    "mid": {"cycle": 0.2, "T24": 0.5, "Ps30": 0.3},
    "late": {"cycle": 0.9, "T24": 0.05, "Ps30": 0.05},
}


@pytest.fixture
def life_phase_bands() -> LifePhaseBandsConfig:
    return LifePhaseBandsConfig()


@pytest.fixture
def training_statistics() -> dict[str, dict[str, float]]:
    return {
        "T24": {"mean": 20.0, "std": 2.0, "min": 10.0, "max": 30.0},
        "Ps30": {"mean": 47.0, "std": 1.0, "min": 44.0, "max": 50.0},
        "T2": {"mean": 518.67, "std": 0.0, "min": 518.67, "max": 518.67},
        # In the stats file (covers all raw CMAPSS sensors) but never logged
        # by this fixture's engine -- exercises the not-a-logged-column path.
        "Nc": {"mean": 8100.0, "std": 15.0, "min": 8000.0, "max": 8200.0},
    }


@pytest.fixture
def inference_log_database(tmp_path) -> str:
    """A real inference log database seeded with one nine-cycle engine."""
    database_path = str(tmp_path / "inference_log.db")
    inference_store.initialize_database(database_path, SENSOR_COLUMNS)
    for cycle_index, predicted_life_ratio in enumerate(ENGINE_ONE_PREDICTIONS):
        cycle = cycle_index + 1
        inference_store.log_inference_reading(
            database_path,
            ENGINE_ONE_ID,
            cycle,
            {
                "T24": ENGINE_ONE_T24_READINGS[cycle_index],
                "Ps30": 47.0,
                "T2": 518.67,
            },
            predicted_life_ratio,
        )
        # numpy scalars mimic what the serving API actually passes in.
        inference_store.log_shap_values(
            database_path,
            ENGINE_ONE_ID,
            cycle,
            {
                feature_name: numpy.float32(shap_value)
                for feature_name, shap_value in SHAP_VALUES_BY_PHASE[
                    ENGINE_ONE_PHASES[cycle_index]
                ].items()
            },
        )
    return database_path


def test_life_phase_band_boundaries_and_validation(life_phase_bands):
    assert shap_trends.assign_life_phase(0.71, life_phase_bands) == "early"
    assert shap_trends.assign_life_phase(0.7, life_phase_bands) == "mid"
    assert shap_trends.assign_life_phase(0.3, life_phase_bands) == "mid"
    assert shap_trends.assign_life_phase(0.29, life_phase_bands) == "late"
    with pytest.raises(ValidationError):
        LifePhaseBandsConfig(early_minimum_life_ratio=0.3, late_maximum_life_ratio=0.7)


def test_evidence_profile_phase_shares_and_ranges(
    inference_log_database, life_phase_bands
):
    profile = shap_trends.get_shap_evidence_profile(
        inference_log_database, ENGINE_ONE_ID, life_phase_bands
    )

    shares_by_phase = {
        phase["phase"]: phase["cycle_feature_share"] for phase in profile["phases"]
    }
    assert shares_by_phase["early"] == pytest.approx(0.8)
    assert shares_by_phase["mid"] == pytest.approx(0.2)
    assert shares_by_phase["late"] == pytest.approx(0.9)
    for phase in profile["phases"]:
        assert phase["cycle_feature_share"] + phase["sensor_feature_share"] == (
            pytest.approx(1.0)
        )
    ranges_by_phase = {
        phase["phase"]: phase["cycle_range"] for phase in profile["phases"]
    }
    assert ranges_by_phase == {"early": [1, 3], "mid": [4, 6], "late": [7, 9]}


def test_evidence_profile_current_phase_and_top_sensor(
    inference_log_database, life_phase_bands
):
    profile = shap_trends.get_shap_evidence_profile(
        inference_log_database, ENGINE_ONE_ID, life_phase_bands
    )

    assert profile["current_phase"] == "late"
    assert profile["latest_cycle"] == 9
    assert profile["latest_predicted_life_ratio"] == pytest.approx(0.05)
    mid_phase = next(phase for phase in profile["phases"] if phase["phase"] == "mid")
    assert mid_phase["top_sensor_features"][0] == {
        "feature": "T24",
        "share": pytest.approx(0.5),
    }


def test_shap_trend_direction_rising_and_falling(inference_log_database):
    rising_trend = shap_trends.get_shap_trend(
        inference_log_database, ENGINE_ONE_ID, ["T24"], (1, 6)
    )
    falling_trend = shap_trends.get_shap_trend(
        inference_log_database, ENGINE_ONE_ID, ["T24"], (4, 9)
    )

    assert rising_trend["features"][0]["direction"] == "rising"
    assert falling_trend["features"][0]["direction"] == "falling"


def test_shap_trend_mean_and_feature_filter(inference_log_database):
    trend = shap_trends.get_shap_trend(
        inference_log_database, ENGINE_ONE_ID, ["T24", "Ps30"]
    )

    assert {entry["feature"] for entry in trend["features"]} == {"T24", "Ps30"}
    t24_summary = next(
        entry for entry in trend["features"] if entry["feature"] == "T24"
    )
    expected_mean = (0.1 * 3 + 0.5 * 3 + 0.05 * 3) / 9
    assert t24_summary["mean_absolute_shap"] == pytest.approx(expected_mean)


def test_drift_z_scores_and_out_of_distribution_flags(
    inference_log_database, training_statistics
):
    report = drift.compare_to_training_distribution(
        inference_log_database,
        training_statistics,
        ENGINE_ONE_ID,
        ["T24", "Ps30"],
        3,
        4.0,
    )

    t24_summary, ps30_summary = report["sensors"]
    assert t24_summary["latest_z_score"] == pytest.approx((0.0 - 20.0) / 2.0)
    expected_window_mean = (20.0 + 20.0 + 0.0) / 3
    assert t24_summary["window_mean_z_score"] == pytest.approx(
        (expected_window_mean - 20.0) / 2.0
    )
    assert t24_summary["out_of_distribution"] is True
    assert ps30_summary["out_of_distribution"] is False
    assert report["cycle_range"] == [7, 9]


def test_drift_zero_std_sensor_yields_none_z_score(
    inference_log_database, training_statistics
):
    report = drift.compare_to_training_distribution(
        inference_log_database, training_statistics, ENGINE_ONE_ID, ["T2"], 3, 4.0
    )

    t2_summary = report["sensors"][0]
    assert t2_summary["latest_z_score"] is None
    assert t2_summary["out_of_distribution"] is False


def test_feature_names_resolve_case_insensitively_across_tools(
    inference_log_database, training_statistics
):
    """Tool-calling models sometimes lowercase parameters (e.g. 'ps30'); the
    dict/dataframe lookups behind drift, charts, and SHAP trend all need the
    canonical CMAPSS spelling, so each tool must correct the case itself."""
    drift_report = drift.compare_to_training_distribution(
        inference_log_database, training_statistics, ENGINE_ONE_ID, ["ps30"], 3, 4.0
    )
    assert drift_report["sensors"][0]["sensor"] == "Ps30"
    assert drift_report["unknown_sensors"] == []

    chart_result = charts.render_chart(
        inference_log_database,
        training_statistics,
        DriftToolConfig(),
        "sensor_trend",
        ENGINE_ONE_ID,
        feature_names=["ps30"],
    )
    assert chart_result.chart_specification.series[0].name == "Ps30"

    shap_trend = shap_trends.get_shap_trend(
        inference_log_database, ENGINE_ONE_ID, ["ps30"]
    )
    assert shap_trend["features"][0]["feature"] == "Ps30"


def test_drift_unknown_sensors(inference_log_database, training_statistics):
    report = drift.compare_to_training_distribution(
        inference_log_database,
        training_statistics,
        ENGINE_ONE_ID,
        ["T24", "bogus_sensor"],
        3,
        4.0,
    )
    assert report["unknown_sensors"] == ["bogus_sensor"]
    assert [entry["sensor"] for entry in report["sensors"]] == ["T24"]

    with pytest.raises(CustomException):
        drift.compare_to_training_distribution(
            inference_log_database,
            training_statistics,
            ENGINE_ONE_ID,
            ["bogus"],
            3,
            4.0,
        )


def test_drift_excludes_unlogged_sensors_from_training_stats(
    inference_log_database, training_statistics
):
    """The stats file covers every raw CMAPSS sensor, but a deployment may
    log only a subset; "Nc" is in training_statistics but was never logged
    by inference_log_database, so it must not reach the SQL builder (a
    genuinely unmatched quoted identifier silently returns garbage instead
    of erroring in SQLite)."""
    default_report = drift.compare_to_training_distribution(
        inference_log_database, training_statistics, ENGINE_ONE_ID, None, 3, 4.0
    )
    assert {entry["sensor"] for entry in default_report["sensors"]} == {
        "T24",
        "Ps30",
        "T2",
    }

    explicit_report = drift.compare_to_training_distribution(
        inference_log_database,
        training_statistics,
        ENGINE_ONE_ID,
        ["T24", "Nc"],
        3,
        4.0,
    )
    assert [entry["sensor"] for entry in explicit_report["sensors"]] == ["T24"]
    assert explicit_report["unknown_sensors"] == ["Nc"]


def test_training_statistics_loading_and_missing_inputs(tmp_path, training_statistics):
    statistics_path = str(tmp_path / "training_statistics.json")
    with open(statistics_path, "w") as statistics_file:
        json.dump({"statistics": training_statistics}, statistics_file)
    assert drift.load_training_statistics(statistics_path) == training_statistics

    with pytest.raises(CustomException):
        drift.compare_to_training_distribution(
            str(tmp_path / "missing.db"), training_statistics, 1, ["T24"], 3, 4.0
        )
    with pytest.raises(CustomException):
        drift.load_training_statistics(str(tmp_path / "missing.json"))
    with pytest.raises(CustomException):
        load_agent_service_config(str(tmp_path / "missing.yaml"))


def test_deployment_logger_writes_minute_rotated_prediction_lines(tmp_path):
    log_directory = str(tmp_path / "logs")

    deployment_logger.log_prediction_event(
        log_directory, 75, 132, {"T24": 642.12, "Ps30": numpy.float32(47.3)}, 0.43, 0.1
    )
    deployment_logger.log_prediction_event(
        log_directory, 75, 133, {"T24": 641.9, "Ps30": 47.1}, 0.05, 0.1
    )

    log_path = deployment_logger.current_log_file_path(log_directory)
    assert os.path.basename(log_path) == (
        datetime.now(UTC).strftime("%Y-%m-%d_%H_%M") + ".log"
    )
    with open(log_path) as log_file:
        healthy_line, near_failure_line = log_file.read().strip().splitlines()
    assert " INFO prediction engine_id=75 cycle=132 " in healthy_line
    assert "predicted_life_ratio=0.430000" in healthy_line
    assert "near_failure=false" in healthy_line
    assert "T24=642.1200" in healthy_line and "Ps30=47.3000" in healthy_line
    assert " WARNING prediction engine_id=75 cycle=133 " in near_failure_line
    assert "near_failure=true" in near_failure_line


def test_deployment_logger_writes_chat_lines_with_truncated_previews(tmp_path):
    log_directory = str(tmp_path / "logs")
    long_question = "why " * 100
    long_answer = "because " * 100

    deployment_logger.log_chat_event(
        log_directory,
        "openai",
        long_question,
        ["get_shap_evidence_profile", "knowledge_search", "knowledge_search"],
        long_answer,
    )
    deployment_logger.log_chat_error_event(
        log_directory, "ollama", "is engine 75 ok?", "MCP server unreachable"
    )

    with open(deployment_logger.current_log_file_path(log_directory)) as log_file:
        chat_line, error_line = log_file.read().strip().splitlines()

    assert " INFO chat backend=openai " in chat_line
    assert "tools=get_shap_evidence_profile,knowledge_search" in chat_line
    assert len(chat_line) < len(long_question) + len(long_answer)
    assert chat_line.count("why") < 100
    assert " ERROR chat backend=ollama " in error_line
    assert 'question="is engine 75 ok?"' in error_line
    assert 'error="MCP server unreachable"' in error_line


def test_sql_select_returns_rows(inference_log_database):
    result = sql.run_read_only_query(
        inference_log_database,
        "SELECT engine_id, cycle FROM inference_readings ORDER BY cycle",
        ALLOWED_TABLES,
        200,
    )

    assert result["columns"] == ["engine_id", "cycle"]
    assert result["row_count"] == len(ENGINE_ONE_PREDICTIONS)
    assert result["truncated"] is False


@pytest.mark.parametrize(
    "forbidden_query",
    [
        "INSERT INTO inference_readings (engine_id, cycle) VALUES (1, 1)",
        "DROP TABLE inference_readings",
        "PRAGMA table_info(inference_readings)",
        "SELECT 1; SELECT 2",
        "WITH staging AS (SELECT 1) "
        "INSERT INTO inference_readings (engine_id, cycle) VALUES (1, 99)",
    ],
)
def test_sql_rejects_forbidden_statements(inference_log_database, forbidden_query):
    with pytest.raises(CustomException):
        sql.run_read_only_query(
            inference_log_database, forbidden_query, ALLOWED_TABLES, 200
        )


def test_sql_rejects_non_allowlisted_tables(inference_log_database):
    with sqlite3.connect(inference_log_database) as connection:
        connection.execute("CREATE TABLE secrets (value TEXT)")

    with pytest.raises(CustomException):
        sql.run_read_only_query(
            inference_log_database, "SELECT * FROM secrets", ALLOWED_TABLES, 200
        )
    with pytest.raises(CustomException):
        sql.run_read_only_query(
            inference_log_database, "SELECT * FROM sqlite_master", ALLOWED_TABLES, 200
        )


def test_sql_enforces_row_limit_with_truncation_flag(inference_log_database):
    result = sql.run_read_only_query(
        inference_log_database,
        "SELECT cycle FROM inference_readings",
        ALLOWED_TABLES,
        5,
    )

    assert result["row_count"] == 5
    assert result["truncated"] is True


def test_prediction_trend_summary_and_range_filter(inference_log_database):
    trend = predictions.get_prediction_trend(
        inference_log_database, ENGINE_ONE_ID, recent_window_size=3
    )
    assert len(trend["series"]) == len(ENGINE_ONE_PREDICTIONS)
    assert trend["summary"]["latest_predicted_life_ratio"] == pytest.approx(0.05)
    assert trend["summary"]["recent_change"] == pytest.approx(0.05 - 0.2)
    assert trend["summary"]["latest_cycle"] == 9

    filtered_trend = predictions.get_prediction_trend(
        inference_log_database, ENGINE_ONE_ID, cycle_range=(4, 6)
    )
    assert [entry["cycle"] for entry in filtered_trend["series"]] == [4, 5, 6]


def test_empty_engine_returns_messages(
    inference_log_database, training_statistics, life_phase_bands
):
    unknown_engine_id = 99

    profile = shap_trends.get_shap_evidence_profile(
        inference_log_database, unknown_engine_id, life_phase_bands
    )
    assert profile["phases"] == []
    assert "No logged SHAP values" in profile["message"]

    drift_report = drift.compare_to_training_distribution(
        inference_log_database, training_statistics, unknown_engine_id, ["T24"], 3, 4.0
    )
    assert drift_report["sensors"] == []
    assert "No logged readings" in drift_report["message"]

    trend = predictions.get_prediction_trend(inference_log_database, unknown_engine_id)
    assert trend["series"] == []
    assert "No logged predictions" in trend["message"]


def test_sensor_trend_chart_spec_and_digest(
    inference_log_database, training_statistics
):
    chart_result = charts.render_chart(
        inference_log_database,
        training_statistics,
        DriftToolConfig(),
        "sensor_trend",
        ENGINE_ONE_ID,
        feature_names=["T24", "Ps30"],
    )

    specification = chart_result.chart_specification
    assert specification.chart_type == "line"
    assert len(specification.x_axis.values) == len(ENGINE_ONE_PREDICTIONS)
    assert [series.name for series in specification.series] == ["T24", "Ps30"]
    assert specification.series[0].values == ENGINE_ONE_T24_READINGS
    for expected_token in ("latest=", "mean=", "min=", "max="):
        assert expected_token in chart_result.digest


def test_prediction_and_shap_trend_chart_specs(
    inference_log_database, training_statistics
):
    prediction_chart = charts.render_chart(
        inference_log_database,
        training_statistics,
        DriftToolConfig(),
        "prediction_trend",
        ENGINE_ONE_ID,
    )
    assert prediction_chart.chart_specification.series[0].values == pytest.approx(
        ENGINE_ONE_PREDICTIONS
    )

    shap_chart = charts.render_chart(
        inference_log_database,
        training_statistics,
        DriftToolConfig(),
        "shap_trend",
        ENGINE_ONE_ID,
    )
    series_names = {series.name for series in shap_chart.chart_specification.series}
    assert series_names == {"cycle", "T24", "Ps30"}


def test_drift_z_scores_chart_flags_drifted_sensor(
    inference_log_database, training_statistics
):
    chart_result = charts.render_chart(
        inference_log_database,
        training_statistics,
        DriftToolConfig(),
        "drift_z_scores",
        ENGINE_ONE_ID,
    )

    specification = chart_result.chart_specification
    assert specification.chart_type == "bar"
    assert set(specification.x_axis.values) == {"T24", "Ps30", "T2"}
    assert "out-of-distribution sensors: T24" in chart_result.digest


def test_chart_error_cases(inference_log_database, training_statistics):
    drift_settings = DriftToolConfig()
    with pytest.raises(CustomException):
        charts.render_chart(
            inference_log_database,
            training_statistics,
            drift_settings,
            "sensor_trend",
            ENGINE_ONE_ID,
        )
    with pytest.raises(CustomException):
        charts.render_chart(
            inference_log_database,
            training_statistics,
            drift_settings,
            "pie_chart",
            ENGINE_ONE_ID,
        )
    with pytest.raises(CustomException):
        charts.render_chart(
            inference_log_database,
            training_statistics,
            drift_settings,
            "prediction_trend",
            99,
        )
    with pytest.raises(CustomException):
        charts.render_chart(
            inference_log_database,
            training_statistics,
            drift_settings,
            "sensor_trend",
            ENGINE_ONE_ID,
            feature_names=["Nc"],
        )
    with pytest.raises(ValidationError):
        ChartSpec(
            chart_type="line",
            title="broken",
            x_axis=ChartAxis(label="cycle", values=[1, 2, 3]),
            series=[ChartSeries(name="T24", values=[1.0, 2.0])],
        )
    round_trip_specification = ChartSpec(
        chart_type="line",
        title="round trip",
        x_axis=ChartAxis(label="cycle", values=[1, 2]),
        series=[ChartSeries(name="T24", values=[1.0, None])],
    )
    assert ChartSpec(**round_trip_specification.model_dump()) == (
        round_trip_specification
    )


def test_markdown_reader_splits_by_section_with_no_split_flag(tmp_path):
    markdown_path = tmp_path / "sensors.md"
    markdown_path.write_text(
        "Intro line before any header.\n\n"
        "# Sensor glossary\n\nPs30 is static pressure.\n\n"
        "## Physical plausibility\n\nPressures are never zero.\n"
    )

    documents = MarkdownReader().read(markdown_path)

    assert [document.metadata["section"] for document in documents] == [
        "preamble",
        "Sensor glossary",
        "Physical plausibility",
    ]
    assert documents[1].page_content.startswith("# Sensor glossary")
    assert all(document.metadata["no_split"] is True for document in documents)
    assert all(document.metadata["source"] == "sensors.md" for document in documents)

    sentence_chunks = SentenceSplitter(chunk_size=50, overlap_sentences=1).split_text(
        "First sentence here. Second sentence here. Third one arrives now."
    )
    assert sentence_chunks == [
        "First sentence here. Second sentence here.",
        "Second sentence here. Third one arrives now.",
    ]


def test_text_reader_drops_chat_lines_from_deployment_logs(tmp_path):
    log_directory = str(tmp_path)
    deployment_logger.log_prediction_event(
        log_directory, engine_id=75, cycle=117, sensor_readings={"T24": 643.33},
        predicted_life_ratio=0.39, life_ratio_threshold=0.1,
    )
    deployment_logger.log_chat_event(
        log_directory, "openai", "which engine has the highest error",
        ["knowledge_search"], "engine 93 has the highest test-set MAE",
    )
    deployment_logger.log_chat_error_event(
        log_directory, "ollama", "hi", "No OpenAI API key",
    )
    log_path = Path(deployment_logger.current_log_file_path(log_directory))

    documents = TextReader().read(log_path)

    assert len(documents) == 1
    page_content = documents[0].page_content
    assert "prediction engine_id=75" in page_content
    assert "chat backend=" not in page_content
    assert "engine 93 has the highest test-set MAE" not in page_content
    assert documents[0].metadata["source_type"] == "deployment_log"


def test_text_reader_returns_no_documents_for_a_chat_only_log_file(tmp_path):
    log_directory = str(tmp_path)
    deployment_logger.log_chat_event(
        log_directory, "openai", "hi", [], "Hello!",
    )
    log_path = Path(deployment_logger.current_log_file_path(log_directory))

    assert TextReader().read(log_path) == []


def test_load_agent_service_config_defaults(tmp_path):
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("database_path: 'custom/inference.db'\n")

    configuration = load_agent_service_config(str(config_path))

    assert configuration.database_path == "custom/inference.db"
    assert configuration.life_phase_bands.early_minimum_life_ratio == 0.7
    assert configuration.sql_tool.max_rows == 200
    assert configuration.mcp_server.port == 8200


def test_tracing_environment_overrides_toggle_without_a_second_config_file():
    base_tracing = TracingConfig()

    unchanged = apply_tracing_environment_overrides(base_tracing, {})
    assert unchanged == base_tracing

    enabled = apply_tracing_environment_overrides(
        base_tracing,
        {
            "AGENT_TRACING_ENABLED": "true",
            "AGENT_TRACING_OTLP_ENDPOINT": "http://openlit:4318",
        },
    )
    assert enabled.enabled is True
    assert enabled.otlp_endpoint == "http://openlit:4318"
    assert base_tracing.enabled is False, "override must not mutate the input"


def test_backend_resolution_and_tracing_noop(monkeypatch):
    backends_configuration = BackendsConfig()

    request_key_model = build_chat_model("openai", backends_configuration, "req-key")
    assert request_key_model.openai_api_key.get_secret_value() == "req-key"

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    environment_key_model = build_chat_model("openai", backends_configuration, None)
    assert environment_key_model.openai_api_key.get_secret_value() == "env-key"

    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(CustomException):
        build_chat_model("openai", backends_configuration, None)
    with pytest.raises(CustomException):
        build_chat_model("no_such_backend", backends_configuration, None)

    initialize_tracing(TracingConfig(enabled=False))


def test_side_channel_splits_charts_and_sources_from_model_text():
    side_channel = SideChannel()
    chart_result = json.dumps(
        {
            "chart_specification": {
                "chart_type": "line",
                "title": "Predicted life ratio -- engine 1",
                "x_axis": {"label": "cycle", "values": [1, 2]},
                "series": [{"name": "predicted_life_ratio", "values": [0.9, 0.8]}],
            },
            "digest": "Predicted life ratio: latest=0.8",
        }
    )
    retrieval_result = {
        "passages": [
            {
                "source": "cmapss_sensors.md",
                "text": "Ps30 is static pressure at the HPC outlet.",
                "image_path": None,
            },
            {
                "source": "xgboost_shap_bar.png",
                "text": "SHAP bar plot showing Ps30 importance.",
                "image_path": "images/xgboost_shap_bar.png",
            },
        ]
    }

    # MCP tools deliver their JSON inside a content-block list.
    chart_text = split_result_for_model(
        [{"type": "text", "text": chart_result}], side_channel
    )
    retrieval_text = split_result_for_model(retrieval_result, side_channel)
    passthrough_text = split_result_for_model({"rows": [[1, 2]]}, side_channel)

    assert chart_text == "Predicted life ratio: latest=0.8"
    assert "values" not in chart_text
    assert [chart.title for chart in side_channel.charts] == [
        "Predicted life ratio -- engine 1"
    ]
    assert "Ps30 is static pressure" in retrieval_text
    assert "images/xgboost_shap_bar.png" not in retrieval_text
    assert side_channel.sources[1].image_path == "images/xgboost_shap_bar.png"
    assert json.loads(passthrough_text) == {"rows": [[1, 2]]}


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed sequence of AI messages; tool binding is a no-op."""

    scripted_messages: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **keyword_arguments):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **keyword_arguments):
        next_message = self.scripted_messages.pop(0)
        return ChatResult(generations=[ChatGeneration(message=next_message)])


def test_agent_streams_tool_events_charts_and_final_payload():
    def render_chart(chart_kind: str, engine_id: int) -> str:
        return json.dumps(
            {
                "chart_specification": {
                    "chart_type": "line",
                    "title": f"{chart_kind} -- engine {engine_id}",
                    "x_axis": {"label": "cycle", "values": [1, 2]},
                    "series": [{"name": "predicted_life_ratio", "values": [0.9, 0.8]}],
                },
                "digest": "latest=0.8",
            }
        )

    chart_tool = StructuredTool.from_function(
        func=render_chart, name="render_chart", description="Render a chart."
    )
    scripted_model = ScriptedChatModel(
        scripted_messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "render_chart",
                        "args": {"chart_kind": "prediction_trend", "engine_id": 1},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Engine 1 is degrading normally."),
        ]
    )
    side_channel = SideChannel()
    wrapped_tools = wrap_tools_with_side_channel([chart_tool], side_channel)
    copilot_agent = build_agent(scripted_model, wrapped_tools)

    async def collect_events():
        return [
            event
            async for event in stream_agent_events(
                copilot_agent, [("user", "How is engine 1?")], side_channel, 25
            )
        ]

    events = asyncio.run(collect_events())

    event_names = [name for name, _ in events]
    assert event_names.count("tool_start") == 1
    assert event_names.count("tool_end") == 1
    assert event_names.count("chart") == 1
    assert event_names[-1] == "final"
    final_payload = events[-1][1]
    assert final_payload["message"] == "Engine 1 is degrading normally."
    assert final_payload["tool_trace"][0]["tool_name"] == "render_chart"
    assert final_payload["charts"][0]["title"] == "prediction_trend -- engine 1"


def test_background_ingest_enables_knowledge_search_on_success(monkeypatch, tmp_path):
    import src_agent.api as api_module

    fake_tool = object()
    fake_pipeline = object()
    monkeypatch.setattr(
        api_module,
        "_run_ingest_and_build_knowledge_search_tool",
        lambda openai_api_key, progress_callback: (fake_pipeline, fake_tool),
    )
    fake_app = SimpleNamespace(
        state=SimpleNamespace(base_tools=[], rag_status=api_module.RAG_STATUS_INGESTING)
    )

    asyncio.run(api_module._ingest_knowledge_base_in_background(fake_app))

    assert fake_app.state.base_tools == [fake_tool]
    assert fake_app.state.ingestion_pipeline is fake_pipeline
    assert fake_app.state.rag_status == api_module.RAG_STATUS_READY
    # Deployment logs aren't in the default document_paths (see
    # _deployment_logs_are_in_rag_corpus), so nothing should be watching
    # them -- a chat turn's own answer must never re-enter the corpus.
    assert not hasattr(fake_app.state, "log_watch_task")


def test_background_ingest_watches_logs_when_they_are_in_the_rag_corpus(
    monkeypatch, tmp_path
):
    import src_agent.api as api_module

    fake_tool = object()
    fake_pipeline = object()
    monkeypatch.setattr(
        api_module,
        "_run_ingest_and_build_knowledge_search_tool",
        lambda openai_api_key, progress_callback: (fake_pipeline, fake_tool),
    )
    log_directory = str(tmp_path / "does_not_exist")
    monkeypatch.setattr(
        api_module.configuration, "deployment_log_directory", log_directory
    )
    monkeypatch.setattr(api_module.configuration.rag, "document_paths", [log_directory])
    fake_app = SimpleNamespace(
        state=SimpleNamespace(base_tools=[], rag_status=api_module.RAG_STATUS_INGESTING)
    )

    async def run_and_cancel_watcher():
        await api_module._ingest_knowledge_base_in_background(fake_app)
        fake_app.state.log_watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await fake_app.state.log_watch_task

    asyncio.run(run_and_cancel_watcher())

    assert fake_app.state.ingested_log_files == set()
    assert fake_app.state.rag_status == api_module.RAG_STATUS_READY


def test_background_ingest_failure_leaves_analytics_tools_usable(monkeypatch):
    import src_agent.api as api_module

    def _raise_ingest_error(openai_api_key, progress_callback):
        raise CustomException("chroma unreachable")

    monkeypatch.setattr(
        api_module, "_run_ingest_and_build_knowledge_search_tool", _raise_ingest_error
    )
    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            base_tools=["existing_mcp_tool"],
            rag_status=api_module.RAG_STATUS_INGESTING,
        )
    )

    asyncio.run(api_module._ingest_knowledge_base_in_background(fake_app))

    assert fake_app.state.base_tools == ["existing_mcp_tool"]
    assert fake_app.state.rag_status == api_module.RAG_STATUS_FAILED


def test_current_deployment_log_files_lists_logs_and_handles_missing_directory(
    monkeypatch, tmp_path
):
    import src_agent.api as api_module

    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    (log_directory / "2026-07-07_14_10.log").write_text("a")
    (log_directory / "2026-07-07_14_11.log").write_text("b")
    (log_directory / "not_a_log.txt").write_text("c")
    monkeypatch.setattr(
        api_module.configuration, "deployment_log_directory", str(log_directory)
    )

    assert api_module._current_deployment_log_files() == {
        log_directory / "2026-07-07_14_10.log",
        log_directory / "2026-07-07_14_11.log",
    }

    monkeypatch.setattr(
        api_module.configuration,
        "deployment_log_directory",
        str(tmp_path / "does_not_exist"),
    )

    assert api_module._current_deployment_log_files() == set()


def test_ingest_one_new_log_file_tracks_success_and_skips_failure(tmp_path):
    import src_agent.api as api_module

    ingested_paths = []

    class FakePipeline:
        def __init__(self, should_fail: bool) -> None:
            self.should_fail = should_fail

        def ingest_file(self, file_path):
            if self.should_fail:
                raise CustomException("chroma unreachable")
            ingested_paths.append(file_path)

    success_log_file = tmp_path / "2026-07-07_14_10.log"
    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            ingestion_pipeline=FakePipeline(should_fail=False),
            ingested_log_files=set(),
            rag_status=api_module.RAG_STATUS_READY,
        )
    )

    asyncio.run(api_module._ingest_one_new_log_file(fake_app, success_log_file))

    assert ingested_paths == [success_log_file]
    assert fake_app.state.ingested_log_files == {success_log_file}
    assert fake_app.state.rag_status == api_module.RAG_STATUS_READY

    failure_log_file = tmp_path / "2026-07-07_14_11.log"
    fake_app.state.ingestion_pipeline = FakePipeline(should_fail=True)

    asyncio.run(api_module._ingest_one_new_log_file(fake_app, failure_log_file))

    assert fake_app.state.ingested_log_files == {success_log_file}  # unchanged
    assert fake_app.state.rag_status == api_module.RAG_STATUS_READY


def test_update_latest_log_pointer_picks_max_filename_or_skips_when_empty(tmp_path):
    import src_agent.api as api_module

    called_with = []

    class FakePipeline:
        def upsert_latest_log_pointer(self, latest_log_file_name):
            called_with.append(latest_log_file_name)

    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            ingestion_pipeline=FakePipeline(),
            ingested_log_files={
                tmp_path / "2026-07-07_14_10.log",
                tmp_path / "2026-07-07_14_49.log",
                tmp_path / "2026-07-07_08_00.log",
            },
        )
    )

    asyncio.run(api_module._update_latest_log_pointer(fake_app))

    assert called_with == ["2026-07-07_14_49.log"]

    fake_app.state.ingested_log_files = set()

    asyncio.run(api_module._update_latest_log_pointer(fake_app))

    assert called_with == ["2026-07-07_14_49.log"]  # unchanged: skipped, not re-called


def test_retrieval_pipeline_refuses_embedding_model_mismatch():
    class FakeCollection:
        @property
        def metadata(self):
            return {"embedding_model": "some-other-model"}

    class FakeChromaClient:
        def get_collection(self, collection_name):
            return FakeCollection()

    with pytest.raises(CustomException):
        RetrievalPipeline.verify_collection(
            FakeChromaClient(), "cmapss_knowledge", "qwen3-embedding:4b"
        )

    class FakeEmptyClient:
        def get_collection(self, collection_name):
            raise ValueError("collection does not exist")

    with pytest.raises(CustomException):
        RetrievalPipeline.verify_collection(
            FakeEmptyClient(), "cmapss_knowledge", "qwen3-embedding:4b"
        )


def test_build_caption_chat_model_uses_each_backend_own_caption_model():
    backends_configuration = BackendsConfig()

    openai_caption_model = build_caption_chat_model(
        "openai", backends_configuration, "req-key"
    )
    assert openai_caption_model.model_name == "gpt-5-mini"

    # Must not require an OpenAI key at all -- that's the whole point of an
    # Ollama-owned caption model.
    ollama_caption_model = build_caption_chat_model("ollama", backends_configuration)
    assert ollama_caption_model.model == "qwen3.5:9b"


def test_document_reader_dispatcher_reports_reading_progress_per_file(tmp_path):
    (tmp_path / "a.md").write_text("# A\ncontent a")
    (tmp_path / "b.txt").write_text("content b")
    (tmp_path / "c.txt").write_text("content c")

    progress_calls = []
    dispatcher = DocumentReaderDispatcher(BackendsConfig())

    dispatcher.read_all(
        [str(tmp_path)],
        progress_callback=lambda phase, completed, total: progress_calls.append(
            (phase, completed, total)
        ),
    )

    assert progress_calls == [
        ("reading", 1, 3),
        ("reading", 2, 3),
        ("reading", 3, 3),
    ]


def test_ingestion_pipeline_upsert_reports_embedding_progress_in_batches():
    from langchain_core.documents import Document

    pipeline = IngestionPipeline(RagConfig(), BackendsConfig())
    documents = [
        Document(page_content=f"doc {i}", metadata={"source": "s"}) for i in range(20)
    ]

    class FakeVectorStore:
        def __init__(self):
            self.batches = []

        def add_documents(self, docs, ids):
            self.batches.append(len(docs))

    fake_vector_store = FakeVectorStore()
    progress_calls = []

    pipeline._upsert_in_batches(
        fake_vector_store,
        documents,
        progress_callback=lambda phase, completed, total: progress_calls.append(
            (phase, completed, total)
        ),
    )

    assert fake_vector_store.batches == [16, 4]
    assert progress_calls == [("embedding", 16, 20), ("embedding", 20, 20)]


def test_progress_recorder_weights_reading_and_embedding_phases_evenly():
    import src_agent.api as api_module

    fake_app = SimpleNamespace(state=SimpleNamespace())
    record = api_module._make_progress_recorder(fake_app)
    assert fake_app.state.rag_ingest_percent == 0

    record("reading", 5, 10)
    assert fake_app.state.rag_ingest_percent == 25  # 50% * halfway through reading

    record("reading", 10, 10)
    assert fake_app.state.rag_ingest_percent == 50  # reading done, embedding at 0%

    record("embedding", 10, 20)
    assert fake_app.state.rag_ingest_percent == 75  # reading 100% + embedding 50%

    record("embedding", 20, 20)
    assert fake_app.state.rag_ingest_percent == 100


@pytest.mark.parametrize(
    "state_overrides",
    [
        pytest.param({"rag_status": "ready"}, id="not_failed"),
        pytest.param(
            {"rag_status": "failed", "rag_retrying": True}, id="already_retrying"
        ),
        pytest.param(
            {
                "rag_status": "failed",
                "rag_retrying": False,
                "rag_last_retry_at": time.monotonic(),
            },
            id="within_cooldown",
        ),
    ],
)
def test_retry_rag_ingestion_is_skipped(monkeypatch, state_overrides):
    import src_agent.api as api_module

    def _fail_if_called(application, openai_api_key):
        raise AssertionError("must not attempt a retry")

    monkeypatch.setattr(api_module, "_activate_knowledge_base", _fail_if_called)
    fake_app = SimpleNamespace(state=SimpleNamespace(**state_overrides))

    asyncio.run(api_module._retry_rag_ingestion_if_needed(fake_app, None))


def test_retry_rag_ingestion_attempts_and_clears_flag_afterwards(monkeypatch):
    import src_agent.api as api_module

    calls = []

    async def _fake_activate(application, openai_api_key):
        calls.append(openai_api_key)
        assert application.state.rag_retrying is True
        application.state.rag_status = api_module.RAG_STATUS_READY

    monkeypatch.setattr(api_module, "_activate_knowledge_base", _fake_activate)
    fake_app = SimpleNamespace(
        state=SimpleNamespace(rag_status=api_module.RAG_STATUS_FAILED)
    )

    asyncio.run(api_module._retry_rag_ingestion_if_needed(fake_app, "typed-in-key"))

    assert calls == ["typed-in-key"]
    assert fake_app.state.rag_retrying is False
    assert fake_app.state.rag_status == api_module.RAG_STATUS_READY
