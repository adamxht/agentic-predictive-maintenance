import numpy as np
import pandas as pd
import pytest

from src.components import data_ingestion, feature_engineering
from src.config_schema import TargetConfig
from src.utils import get_sensor_columns


@pytest.fixture
def raw_multi_engine_dataframe() -> pd.DataFrame:
    """Build a small two-engine dataframe with one sensor column."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "T24": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        }
    )


def test_add_remaining_useful_life_computes_correct_rul(raw_multi_engine_dataframe):
    result_dataframe = feature_engineering.add_remaining_useful_life(
        raw_multi_engine_dataframe
    )

    assert result_dataframe["RUL"].tolist() == [2, 1, 0, 2, 1, 0]


def test_add_life_ratio_is_bounded_between_zero_and_one(raw_multi_engine_dataframe):
    dataframe_with_rul = feature_engineering.add_remaining_useful_life(
        raw_multi_engine_dataframe
    )

    result_dataframe = feature_engineering.add_life_ratio(dataframe_with_rul)

    assert result_dataframe["life_ratio"].tolist() == pytest.approx(
        [2 / 3, 1 / 3, 0.0, 2 / 3, 1 / 3, 0.0]
    )
    assert result_dataframe["life_ratio"].between(0, 1).all()


def test_target_config_derives_column_name_from_type():
    assert TargetConfig(type="rul").column_name == "RUL"
    assert TargetConfig(type="life_ratio").column_name == "life_ratio"


def test_drop_unused_columns_ignores_missing_columns(raw_multi_engine_dataframe):
    result_dataframe = feature_engineering.drop_unused_columns(
        raw_multi_engine_dataframe, ["T24", "column_that_does_not_exist"]
    )

    assert "T24" not in result_dataframe.columns
    assert list(result_dataframe.columns) == ["engine_id", "cycle"]


@pytest.fixture
def multi_engine_dataframe_with_missing_values() -> pd.DataFrame:
    """Build a two-engine dataframe with interior, leading, and trailing gaps."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 5, 1, 2, 3],
            "T24": [10.0, np.nan, 30.0, np.nan, np.nan, np.nan, 200.0, 300.0],
        }
    )


def test_handle_missing_sensor_values_interpolates_and_fills_edges(
    multi_engine_dataframe_with_missing_values,
):
    result_dataframe = feature_engineering.handle_missing_sensor_values(
        multi_engine_dataframe_with_missing_values, ["T24"]
    )

    engine_one_values = result_dataframe.loc[
        result_dataframe["engine_id"] == 1, "T24"
    ].tolist()
    engine_two_values = result_dataframe.loc[
        result_dataframe["engine_id"] == 2, "T24"
    ].tolist()

    assert engine_one_values == pytest.approx([10.0, 20.0, 30.0, 30.0, 30.0])
    assert engine_two_values == pytest.approx([200.0, 200.0, 300.0])
    assert not result_dataframe["T24"].isna().any()


def test_handle_missing_sensor_values_does_not_leak_across_engines(
    multi_engine_dataframe_with_missing_values,
):
    result_dataframe = feature_engineering.handle_missing_sensor_values(
        multi_engine_dataframe_with_missing_values, ["T24"]
    )

    engine_two_first_value = result_dataframe.loc[
        (result_dataframe["engine_id"] == 2) & (result_dataframe["cycle"] == 1), "T24"
    ].item()

    assert engine_two_first_value == pytest.approx(200.0)


def test_get_sensor_columns_excludes_identifier_and_target_columns(
    raw_multi_engine_dataframe,
):
    dataframe_with_rul = feature_engineering.add_remaining_useful_life(
        raw_multi_engine_dataframe
    )

    sensor_columns = get_sensor_columns(dataframe_with_rul)

    assert sensor_columns == ["T24"]


def test_fit_and_apply_sensor_scaler_uses_train_statistics_only(
    raw_multi_engine_dataframe,
):
    train_dataframe = raw_multi_engine_dataframe[
        raw_multi_engine_dataframe["engine_id"] == 1
    ]
    validation_dataframe = raw_multi_engine_dataframe[
        raw_multi_engine_dataframe["engine_id"] == 2
    ]

    scaler = feature_engineering.fit_sensor_scaler(train_dataframe, ["T24"])
    scaled_train_dataframe = feature_engineering.apply_sensor_scaler(
        train_dataframe, scaler, ["T24"]
    )
    scaled_validation_dataframe = feature_engineering.apply_sensor_scaler(
        validation_dataframe, scaler, ["T24"]
    )

    assert scaled_train_dataframe["T24"].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaler.mean_[0] == pytest.approx(20.0)
    assert scaled_validation_dataframe["T24"].mean() != pytest.approx(0.0, abs=1e-9)


def test_add_rolling_window_features_does_not_leak_across_engines(
    raw_multi_engine_dataframe,
):
    result_dataframe = feature_engineering.add_rolling_window_features(
        raw_multi_engine_dataframe, ["T24"], window_size=2
    )

    engine_one_rolling_means = result_dataframe.loc[
        result_dataframe["engine_id"] == 1, "T24_roll_mean"
    ].tolist()

    assert engine_one_rolling_means == [10.0, 15.0, 25.0]


def test_add_lag_features_shifts_within_engine_and_introduces_leading_nan(
    raw_multi_engine_dataframe,
):
    result_dataframe = feature_engineering.add_lag_features(
        raw_multi_engine_dataframe, ["T24"], [1]
    )

    engine_one_lag_values = result_dataframe.loc[
        result_dataframe["engine_id"] == 1, "T24_lag1"
    ]

    assert engine_one_lag_values.isna().tolist() == [True, False, False]
    assert engine_one_lag_values.dropna().tolist() == [10.0, 20.0]


def test_select_top_features_respects_top_k():
    training_dataframe = pd.DataFrame(
        {
            "RUL": [10, 8, 6, 4, 2, 0],
            "strong_signal": [0, 1, 2, 3, 4, 5],
            "weak_signal": [1, 1, 1, 1, 1, 2],
        }
    )

    selected_features = feature_engineering.select_top_features(
        training_dataframe, target_column="RUL", top_k=1
    )

    assert selected_features == ["strong_signal"]


def test_apply_feature_selection_keeps_identifier_columns(raw_multi_engine_dataframe):
    dataframe_with_rul = feature_engineering.add_remaining_useful_life(
        raw_multi_engine_dataframe
    )

    result_dataframe = feature_engineering.apply_feature_selection(
        dataframe_with_rul, selected_features=["T24"], target_column="RUL"
    )

    assert set(result_dataframe.columns) == {"RUL", "T24", "cycle", "engine_id"}


def test_split_train_validation_by_engine_has_no_engine_overlap(
    raw_multi_engine_dataframe,
):
    train_dataframe, validation_dataframe = (
        data_ingestion.split_train_validation_by_engine(
            raw_multi_engine_dataframe, test_size=0.5, random_state=0
        )
    )

    train_engine_ids = set(train_dataframe["engine_id"])
    validation_engine_ids = set(validation_dataframe["engine_id"])

    assert train_engine_ids.isdisjoint(validation_engine_ids)
    assert train_engine_ids | validation_engine_ids == {1, 2}
