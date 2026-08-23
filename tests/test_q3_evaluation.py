from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q3.evaluation import (
    _complete_operating_days,
    _residual_blocks,
    long_gap_backtest,
    metric_table,
    residual_block_intervals,
    stratified_metric_table,
)
from q3_fixtures import synthetic_regular_frame


def test_complete_operating_days_ignore_unavailable_numeric_injection():
    missing = synthetic_regular_frame(periods=48)
    unavailable_time = missing.index[5]
    missing.loc[unavailable_time, "treated_ntu"] = np.nan
    missing.loc[unavailable_time, "target_available"] = False
    missing.loc[unavailable_time, "missing_treated_ntu"] = True
    injected = missing.copy()
    injected.loc[unavailable_time, "treated_ntu"] = 999999.0

    missing_days = _complete_operating_days(missing)
    injected_days = _complete_operating_days(injected)

    pd.testing.assert_index_equal(missing_days, injected_days)
    assert pd.Timestamp("2025-01-01") not in injected_days


def prediction_long_fixture():
    rows = []
    models = ("季节朴素", "机理专家", "LightGBM", "GRU", "MoE")
    horizons = (2, 4, 6, 8, 10, 12)
    for model_number, model in enumerate(models):
        for horizon in horizons:
            for sample in range(20):
                timestamp = pd.Timestamp("2025-07-01 07:00") + pd.Timedelta(hours=2 * sample)
                actual = 0.25 + sample / 100
                rows.append(
                    {
                        "model": model,
                        "origin": timestamp,
                        "horizon": horizon,
                        "actual": actual,
                        "prediction": actual + (model_number + 1) / 1000,
                        "hour": timestamp.hour,
                        "weekday": timestamp.weekday(),
                        "season": "雨季",
                        "regime": "平稳工况" if sample % 2 == 0 else "原水突变",
                    }
                )
    return pd.DataFrame(rows)


def oof_residual_fixture():
    rows = []
    days = pd.date_range("2025-01-01", periods=42, freq="D")
    for day_number, operating_date in enumerate(days):
        for sample in range(12):
            rows.append(
                {
                    "operating_date": operating_date,
                    "timestamp": operating_date + pd.Timedelta(hours=7 + 2 * sample),
                    "residual": (day_number % 5 - 2) / 100 + sample / 1000,
                }
            )
    return pd.DataFrame(rows)


def coded_residual_fixture(days=40, residual_offset=0.0):
    rows = []
    for day_number, operating_date in enumerate(pd.date_range("2025-01-01", periods=days, freq="D")):
        for position in range(12):
            rows.append(
                {
                    "operating_date": operating_date,
                    "timestamp": operating_date + pd.Timedelta(hours=7 + 2 * position),
                    "residual": residual_offset + 100 * day_number + position,
                }
            )
    return pd.DataFrame(rows)


def test_metric_table_reports_every_model_and_horizon():
    table = metric_table(prediction_long_fixture())

    assert set(table["预测步长/小时"]) == {2, 4, 6, 8, 10, 12}
    assert {"季节朴素", "机理专家", "LightGBM", "GRU", "MoE"}.issubset(table["模型"])
    assert {"RMSE", "MAE", "R2", "样本数"}.issubset(table.columns)
    assert set(table["样本数"]) == {20}
    assert len(table) == 30


def test_metric_table_keeps_a_partial_two_hour_table_in_hours_and_calculates_exact_metrics():
    table = metric_table(
        pd.DataFrame(
            {
                "model": ["MoE"] * 3,
                "horizon": [2] * 3,
                "actual": [1.0, 2.0, 3.0],
                "prediction": [0.0, 2.0, 5.0],
            }
        )
    )

    row = table.iloc[0]
    assert row["预测步长/小时"] == 2
    assert row["样本数"] == 3
    assert row["MAE"] == pytest.approx(1.0)
    assert row["RMSE"] == pytest.approx(np.sqrt(5 / 3))
    assert row["R2"] == pytest.approx(-1.5)


def test_metric_table_returns_nan_r2_for_singleton_and_constant_actual_groups():
    table = metric_table(
        pd.DataFrame(
            {
                "model": ["MoE", "MoE", "GRU"],
                "horizon": [2, 2, 4],
                "actual": [1.0, 1.0, 2.0],
                "prediction": [0.0, 2.0, 2.5],
            }
        )
    )

    assert table.loc[table["预测步长/小时"] == 2, "R2"].isna().all()
    assert np.isnan(table.loc[table["预测步长/小时"] == 4, "R2"].iloc[0])


def test_stratified_metrics_preserve_model_horizon_and_four_operating_strata():
    source = pd.DataFrame(
        {
            "model": ["MoE"] * 4,
            "origin": pd.to_datetime(["2025-07-01 07:00", "2025-07-06 21:00", "2025-07-01 09:00", "2025-07-06 19:00"]),
            "horizon": [2, 2, 2, 2],
            "actual": [1.0, 1.0, 1.0, 1.0],
            "prediction": [1.0, 1.1, 0.9, 1.0],
            "hour": [7, 21, 9, 19],
            "weekday": [1, 6, 1, 6],
            "season": ["雨季", "旱季", "雨季", "旱季"],
            "regime": ["平稳工况", "原水突变", "矾量调整", "平稳工况"],
        }
    )
    table = stratified_metric_table(source)

    assert {"时段", "工作日", "季节", "工况"} == set(table["分层维度"])
    assert {"模型", "预测步长/小时", "分层", "RMSE", "MAE", "R2", "样本数"}.issubset(table.columns)
    assert set(table.loc[table["分层维度"] == "时段", "分层"]) == {"07:00--19:00", "夜间"}
    assert set(table.loc[table["分层维度"] == "工作日", "分层"]) == {"工作日", "周末"}
    assert set(table.loc[table["分层维度"] == "季节", "分层"]) == {"雨季", "旱季"}
    assert set(table.loc[table["分层维度"] == "工况", "分层"]) == {"平稳工况", "原水突变", "矾量调整"}


def test_residual_block_intervals_are_ordered_and_use_exactly_200_resamples():
    intervals = residual_block_intervals(oof_residual_fixture(), point=np.full(21, 0.3))

    assert len(intervals) == 21
    assert set(intervals["重复次数"]) == {200}
    assert np.all(intervals["95%下限"] <= intervals["80%下限"])
    assert np.all(intervals["80%下限"] <= intervals["预测值"])
    assert np.all(intervals["预测值"] <= intervals["80%上限"])
    assert np.all(intervals["80%上限"] <= intervals["95%上限"])


def test_residual_blocks_are_complete_aligned_twenty_one_point_paths():
    blocks = _residual_blocks(coded_residual_fixture())

    expected = np.array(
        [100 * day + position for day in (0, 3, 6) for position in range(7)],
        dtype=float,
    )
    assert len(blocks) == 34
    np.testing.assert_allclose(blocks[0], expected)
    assert all(block.shape == (21,) for block in blocks)


def test_residual_blocks_accept_original_twelve_rows_per_day_without_time_columns():
    source = coded_residual_fixture().drop(columns="timestamp")
    blocks = _residual_blocks(source)

    expected = np.array(
        [100 * day + position for day in (0, 3, 6) for position in range(7)],
        dtype=float,
    )
    np.testing.assert_allclose(blocks[0], expected)


def test_residual_intervals_use_fixed_seed_centered_path_quantiles_and_all_bounds():
    point = np.full(21, 0.3)
    source = coded_residual_fixture()
    first = residual_block_intervals(source, point)
    second = residual_block_intervals(source, point)
    paths = np.asarray(_residual_blocks(source), dtype=float)
    generator = np.random.default_rng(2026)
    samples = np.array([paths[generator.integers(len(paths))] - np.median(paths, axis=0) for _ in range(200)])
    lower_80, upper_80 = np.quantile(samples, (0.10, 0.90), axis=0)
    lower_95, upper_95 = np.quantile(samples, (0.025, 0.975), axis=0)

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_allclose(first["80%下限"], point + np.minimum(lower_80, 0.0))
    np.testing.assert_allclose(first["80%上限"], point + np.maximum(upper_80, 0.0))
    np.testing.assert_allclose(first["95%下限"], point + np.minimum(lower_95, 0.0))
    np.testing.assert_allclose(first["95%上限"], point + np.maximum(upper_95, 0.0))


def test_residual_intervals_center_one_sided_residuals_around_the_point_forecast():
    source = coded_residual_fixture(residual_offset=10.0)
    source["residual"] = 10.0
    intervals = residual_block_intervals(source, np.full(21, 0.3))

    assert np.all(intervals["95%下限"] <= intervals["80%下限"])
    assert np.all(intervals["80%下限"] <= intervals["预测值"])
    assert np.all(intervals["预测值"] <= intervals["80%上限"])
    assert np.all(intervals["80%上限"] <= intervals["95%上限"])
    np.testing.assert_allclose(intervals["预测值"], intervals["80%下限"])
    np.testing.assert_allclose(intervals["预测值"], intervals["95%上限"])


def test_residual_intervals_require_forty_consecutive_operating_days():
    with pytest.raises(ValueError, match="forty"):
        residual_block_intervals(coded_residual_fixture(days=39), np.full(21, 0.3))


def test_residual_blocks_reject_duplicate_or_incomplete_daily_coordinates():
    duplicate = pd.concat((coded_residual_fixture(), coded_residual_fixture().iloc[[0]]), ignore_index=True)
    incomplete = coded_residual_fixture().drop(index=2)

    with pytest.raises(ValueError, match="duplicate"):
        residual_block_intervals(duplicate, np.full(21, 0.3))
    with pytest.raises(ValueError, match="complete"):
        residual_block_intervals(incomplete, np.full(21, 0.3))


def test_long_gap_backtest_masks_operating_day_intervals_and_never_refits_inside_gap():
    calls = []

    class MechanisticStub:
        def fit(self, frame, train_end):
            calls.append(("mechanistic", pd.Timestamp(train_end), frame["treated_ntu"].copy()))
            return self

        def fill_target_history(self, frame):
            return frame["treated_ntu"].ffill().fillna(0.0)

        def predict(self, frame, origins):
            return np.full((len(origins), 6), 0.30)

    class TreeStub:
        def fit(self, frame, origins, targets):
            calls.append(("tree", pd.DatetimeIndex(origins).max(), frame["treated_ntu"].copy()))
            self.expected_available_end = pd.DatetimeIndex(origins).max() + pd.Timedelta(hours=12)
            return self

        def predict(self, frame, origins, filled_target=None):
            assert filled_target is not None
            assert self.available_target_end_ == self.expected_available_end
            return np.full((len(origins), 6), 0.31)

    class GruStub:
        def fit(self, frame, origins, targets, filled_target=None):
            calls.append(("gru", pd.DatetimeIndex(origins).max(), frame["treated_ntu"].copy()))
            assert filled_target is not None
            self.expected_available_end = pd.DatetimeIndex(origins).max() + pd.Timedelta(hours=12)
            return self

        def predict(self, frame, origins, filled_target=None):
            assert filled_target is not None
            assert self.available_target_end_ == self.expected_available_end
            return np.full((len(origins), 6), 0.32)

    frame = synthetic_regular_frame(periods=1200)
    result = long_gap_backtest(frame, (MechanisticStub, TreeStub, GruStub))

    assert set(result["gap_days"]) == {10, 20, 28}
    assert set(result["model"]) == {"季节朴素", "机理专家", "LightGBM", "GRU", "MoE"}
    assert (result["prediction"] >= 0).all()
    assert len(calls) == 9
    assert (result["fit_end"] < result["gap_start"]).all()
    assert (result["target_timestamp"] <= result["gap_end"]).all()
    assert (result["origin"] > result["fit_end"]).all()
    assert (result["training_label_end"] < result["gap_start"]).all()

    for _, gap in result.groupby("gap_days"):
        masked = gap["gap_start"].iloc[0]
        fit_calls = [call for call in calls if call[1] < masked]
        assert len(fit_calls) >= 3
        for _, _, target in fit_calls[-3:]:
            assert target.loc[gap["gap_start"].iloc[0]:gap["gap_end"].iloc[0]].isna().all()
