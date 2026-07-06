"""Streamlit demo: replays raw sensor readings as a live feed against the
stateless inference API (app/api.py), one simulated cycle at a time. Shows
the model's live life-ratio prediction, SHAP feature importance, and raw
(unscaled) sensor trends -- plus a button to simulate sensor drift.

Uses the raw training file (not the test set) for the demo engines, because
the test set is censored -- it never actually reaches failure (e.g. one
engine only has data up to cycle 88) -- while the training file has each
engine's full, uncensored run to actual failure.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import altair as alt
import pandas as pd
import requests
import streamlit as st

from src.components import data_ingestion, feature_engineering
from src.configs.data_pipeline_config_schema import load_data_preparation_config
from src.const import SENSOR_NAMES
from src.logger import logging

API_BASE_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000")
DATA_CONFIG_PATH = "configs/data_transformation/default.yaml"
DEMO_ENGINE_IDS = [75, 25, 26]
DEFAULT_CYCLE_DURATION_SECONDS = 10
MIN_CYCLE_DURATION_SECONDS = 1
MAX_CYCLE_DURATION_SECONDS = 120

st.set_page_config(page_title="Real-Time Inference Demo", layout="wide")


@st.cache_data(show_spinner=False)
def load_demo_engine_data() -> dict[int, pd.DataFrame]:
    """Load raw (uncensored) readings for the demo engines from the training file.

    These engine ids exist directly in the raw training file regardless of
    which train/validation split they landed in during data preparation, so
    no split reproduction is needed -- just read and filter the raw file.
    """
    data_config = load_data_preparation_config(DATA_CONFIG_PATH)
    raw_dataframe = data_ingestion.load_raw_sensor_readings(
        data_config.paths.raw_data_path, SENSOR_NAMES
    )
    demo_dataframe = raw_dataframe[raw_dataframe["engine_id"].isin(DEMO_ENGINE_IDS)]
    return {
        engine_id: group.sort_values("cycle").reset_index(drop=True)
        for engine_id, group in demo_dataframe.groupby("engine_id")
    }


@st.cache_data(show_spinner=False)
def fetch_serving_config(api_base_url: str) -> dict:
    """Fetch the required window length and feature columns from the inference API."""
    response = requests.get(f"{api_base_url}/config", timeout=5)
    response.raise_for_status()
    return response.json()


def _initialize_session_state() -> None:
    """Set up per-engine simulation state on first load only."""
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.engine_id = DEMO_ENGINE_IDS[0]
    st.session_state.is_running = False
    st.session_state.cycle_duration_seconds = DEFAULT_CYCLE_DURATION_SECONDS
    st.session_state.drift_feature = None
    st.session_state.cycle_pointer = {}
    st.session_state.buffer = {}
    st.session_state.raw_history = {}
    st.session_state.predictions = {}
    st.session_state.last_error = {}
    st.session_state.last_advance_time = {}
    for engine_id in DEMO_ENGINE_IDS:
        _reset_engine_state(engine_id)


def _reset_engine_state(engine_id: int) -> None:
    """Reset one engine's accumulated simulation state to its defaults."""
    st.session_state.cycle_pointer[engine_id] = 0
    st.session_state.buffer[engine_id] = []
    st.session_state.raw_history[engine_id] = []
    st.session_state.predictions[engine_id] = []
    st.session_state.last_error[engine_id] = None
    st.session_state.last_advance_time[engine_id] = 0.0


def _reset_engine(engine_id: int) -> None:
    """Reset one engine's state and stop the simulation."""
    _reset_engine_state(engine_id)
    st.session_state.is_running = False


def _call_predict_api(engine_id: int, window: list[dict]) -> None:
    """Send a reading window to the inference API and record the outcome."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"engine_id": engine_id, "readings": window},
            timeout=5,
        )
        response.raise_for_status()
        st.session_state.predictions[engine_id].append(response.json())
        st.session_state.last_error[engine_id] = None
    except requests.HTTPError as error:
        detail = error.response.json().get("detail", str(error))
        st.session_state.last_error[engine_id] = detail
    except requests.RequestException as error:
        st.session_state.last_error[engine_id] = str(error)


def _advance_one_cycle(
    engine_id: int, sensor_columns: list[str], required_window_length: int
) -> None:
    """Send the next cycle's reading (with any active drift applied) to the API."""
    source_dataframe = st.session_state.engine_dataframes[engine_id]
    pointer = st.session_state.cycle_pointer[engine_id]
    if pointer >= len(source_dataframe):
        st.session_state.is_running = False
        return

    row = source_dataframe.iloc[pointer]
    reading_values = {column: float(row[column]) for column in sensor_columns}
    if st.session_state.drift_feature:
        reading_values[st.session_state.drift_feature] = 0.0
    cycle_number = int(row["cycle"])
    logging.info(f"Engine {engine_id} cycle {cycle_number} reading: {reading_values}")

    buffer = st.session_state.buffer[engine_id]
    buffer.append({"cycle": cycle_number, "values": reading_values})
    window = buffer[-required_window_length:]

    _call_predict_api(engine_id, window)
    st.session_state.raw_history[engine_id].append(
        {"cycle": cycle_number, **reading_values}
    )
    st.session_state.cycle_pointer[engine_id] = pointer + 1


def _advance_one_cycle_if_due(
    engine_id: int,
    sensor_columns: list[str],
    required_window_length: int,
    cycle_duration_seconds: float,
) -> None:
    """Advance one cycle only if a full interval has actually elapsed.

    Without this gate, any full script rerun (e.g. tweaking the "Seconds per
    cycle" control) would advance a cycle just by re-entering _live_section
    while running, since that check doesn't know why it was rerun.
    """
    now = time.monotonic()
    last_advance_time = st.session_state.last_advance_time.get(engine_id, 0.0)
    if now - last_advance_time < cycle_duration_seconds:
        return
    st.session_state.last_advance_time[engine_id] = now
    _advance_one_cycle(engine_id, sensor_columns, required_window_length)


def _render_controls(plotted_sensor_columns: list[str]) -> None:
    """Render the engine picker, start/stop, drift, interval, and reset controls."""
    engine_column, action_column, interval_column, drift_column, reset_column = (
        st.columns(5)
    )
    with engine_column:
        selected_engine = st.selectbox(
            "Engine",
            DEMO_ENGINE_IDS,
            key="engine_selector",
            index=DEMO_ENGINE_IDS.index(st.session_state.engine_id),
        )
        if selected_engine != st.session_state.engine_id:
            st.session_state.engine_id = selected_engine
            st.session_state.is_running = False

    with action_column:
        button_label = "Stop" if st.session_state.is_running else "Start"
        if st.button(button_label, key="start_stop_button"):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()

    with interval_column:
        st.session_state.cycle_duration_seconds = st.number_input(
            "Seconds per cycle",
            min_value=MIN_CYCLE_DURATION_SECONDS,
            max_value=MAX_CYCLE_DURATION_SECONDS,
            value=st.session_state.cycle_duration_seconds,
            step=1,
            key="cycle_duration_input",
        )

    with drift_column:
        drift_options = ["None", *plotted_sensor_columns]
        if st.session_state.drift_feature not in drift_options:
            st.session_state.drift_feature = None
        drift_choice = st.selectbox(
            "Drift feature",
            drift_options,
            key="drift_selector",
            index=drift_options.index(st.session_state.drift_feature or "None"),
        )
        st.session_state.drift_feature = (
            None if drift_choice == "None" else drift_choice
        )

    with reset_column:
        st.write("")
        if st.button("Reset engine"):
            _reset_engine(st.session_state.engine_id)
            st.rerun()

    if st.session_state.drift_feature:
        st.warning(
            f"Drift active: **{st.session_state.drift_feature}** is being forced "
            "to 0.0 in every new reading."
        )


def _render_prediction(engine_id: int, life_ratio_threshold: float) -> None:
    """Render the latest life-ratio prediction bar and SHAP bar chart."""
    error_message = st.session_state.last_error[engine_id]
    if error_message:
        st.info(f"No prediction for the latest cycle yet: {error_message}")

    predictions = st.session_state.predictions[engine_id]
    if not predictions:
        st.info("No prediction yet -- press Start.")
        return

    latest = predictions[-1]
    predicted_life_ratio = latest["predicted_life_ratio"]
    st.metric("Cycle", latest["cycle"])
    st.progress(
        min(max(predicted_life_ratio, 0.0), 1.0),
        text=f"Predicted life ratio: {predicted_life_ratio:.3f}",
    )
    if predicted_life_ratio < life_ratio_threshold:
        st.error(
            f"Engine failure predicted: life ratio {predicted_life_ratio:.3f} is "
            f"below the failure threshold ({life_ratio_threshold:.2f})."
        )

    shap_values = latest["shap_values"]
    shap_dataframe = pd.DataFrame(
        {"feature": list(shap_values.keys()), "shap_value": list(shap_values.values())}
    ).sort_values("shap_value", key=abs, ascending=False)
    st.caption("SHAP feature contributions for this cycle's prediction")
    st.bar_chart(shap_dataframe.set_index("feature")["shap_value"])


def _build_zoomed_line_chart(
    raw_dataframe: pd.DataFrame, feature_name: str
) -> alt.Chart:
    """Build a line chart whose y-axis is zoomed to the feature's own data range.

    st.line_chart forces the y-axis to include zero, which flattens sensors
    whose real variation is tiny relative to their absolute value (e.g. NRc
    hovering around 47.1-47.5) into an invisible, dead-flat line.
    """
    return (
        alt.Chart(raw_dataframe)
        .mark_line()
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y(f"{feature_name}:Q", title=None, scale=alt.Scale(zero=False)),
        )
        .properties(height=150)
    )


def _render_feature_trends(engine_id: int, plotted_sensor_columns: list[str]) -> None:
    """Render each sensor the model actually uses as its own raw (unscaled) chart."""
    raw_history = st.session_state.raw_history[engine_id]
    if not raw_history:
        return

    st.caption(
        "Raw sensor trends (unscaled, model-used sensors only, y-axis zoomed to "
        "each sensor's own range)"
    )
    raw_dataframe = pd.DataFrame(raw_history)
    columns_per_row = 3
    for row_start in range(0, len(plotted_sensor_columns), columns_per_row):
        row_columns = st.columns(columns_per_row)
        row_features = plotted_sensor_columns[row_start : row_start + columns_per_row]
        for column_widget, feature_name in zip(row_columns, row_features, strict=False):
            with column_widget:
                st.caption(feature_name)
                chart = _build_zoomed_line_chart(raw_dataframe, feature_name)
                st.altair_chart(chart, width="stretch")


def _live_section(
    sensor_columns: list[str],
    plotted_sensor_columns: list[str],
    required_window_length: int,
    life_ratio_threshold: float,
) -> None:
    """Advance the simulation (if running and due) and redraw the live dashboard."""
    engine_id = st.session_state.engine_id
    if st.session_state.is_running:
        _advance_one_cycle_if_due(
            engine_id,
            sensor_columns,
            required_window_length,
            st.session_state.cycle_duration_seconds,
        )
    _render_prediction(engine_id, life_ratio_threshold)
    _render_feature_trends(engine_id, plotted_sensor_columns)


def main() -> None:
    """Wire up the CMAPSS real-time inference demo Streamlit app."""
    st.title("Real-Time Predictive Maintenance Demo")
    st.caption(
        "Replays raw engine readings (which run to actual failure) as a live "
        "sensor feed against the stateless inference API, one simulated cycle every "
        "N seconds (configurable)."
    )

    try:
        serving_config = fetch_serving_config(API_BASE_URL)
    except requests.RequestException as error:
        st.error(f"Could not reach the inference API at {API_BASE_URL}: {error}")
        st.stop()
        return

    _initialize_session_state()
    if "engine_dataframes" not in st.session_state:
        st.session_state.engine_dataframes = load_demo_engine_data()

    sensor_columns = serving_config["sensor_columns"]
    plotted_sensor_columns = feature_engineering.derive_base_sensor_names(
        serving_config["selected_features"]
    )
    required_window_length = serving_config["required_window_length"]
    life_ratio_threshold = serving_config["life_ratio_threshold"]

    _render_controls(plotted_sensor_columns)

    # Re-wrapped every rerun so the fragment's auto-refresh interval always
    # reflects the current "Seconds per cycle" control.
    live_section = st.fragment(run_every=st.session_state.cycle_duration_seconds)(
        _live_section
    )
    live_section(
        sensor_columns,
        plotted_sensor_columns,
        required_window_length,
        life_ratio_threshold,
    )


if __name__ == "__main__":
    main()
