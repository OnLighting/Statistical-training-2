from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from question4_eda import build_daily_features, classify_q1, fit_fuzzy_clusters, fuse_grades


class FixedMoEPredictor:
    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return pd.Series(1.8, index=frame.index)


class LowMoEPredictor:
    def __call__(self, frame):
        return pd.Series(0.5, index=frame.index)


class WindowMoEPredictor:
    def __init__(self):
        self.rows = 0

    def __call__(self, frame):
        self.rows = len(frame)
        values = pd.Series(np.nan, index=frame.index)
        values.iloc[24 : len(frame) - 6] = 1.2
        return values


def twelve_point_frame():
    timestamps = pd.date_range("2026-01-02 07:00", periods=12, freq="2h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "treated_ntu": [0.6, 1.2, 1.5, np.nan, 0.7, 1.1, 1.3, 0.8, 0.9, 1.4, 1.6, 0.5],
        }
    )


def test_daily_features_keep_observations_and_mark_moe_values_with_exact_mfdl():
    predictor = FixedMoEPredictor()

    points, daily = build_daily_features(twelve_point_frame(), predictor)

    assert predictor.calls == 1
    assert points.loc[points.index[2], "treated_ntu"] == 1.5
    assert points.loc[points.index[2], "treated_ntu来源"] == "实测"
    assert points.loc[points.index[3], "treated_ntu"] == 1.8
    assert points.loc[points.index[3], "treated_ntu来源"] == "MoE预测"
    row = daily.iloc[0]
    assert row["M"] == 0.8
    assert row["F"] == 7 / 12
    assert row["D"] == 6
    assert row["L"] == 5.8


def test_daily_features_rebuilds_the_twelve_point_grid_and_missing_breaks_runs():
    data = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02 07:00",
                "2026-01-02 09:00",
                "2026-01-02 09:00",
                "2026-01-02 10:00",
                "2026-01-02 11:00",
            ],
            "treated_ntu": [1.2, 1.1, np.nan, 9.0, 1.3],
        }
    )

    points, daily = build_daily_features(data, LowMoEPredictor())

    assert points["timestamp"].tolist() == list(pd.date_range("2026-01-02 07:00", periods=12, freq="2h"))
    assert points.loc[1, "treated_ntu"] == 0.5
    assert points.loc[1, "treated_ntu来源"] == "MoE预测"
    assert daily.loc[0, "F"] == 2 / 12
    assert daily.loc[0, "D"] == 2


def test_daily_features_uses_full_default_context_to_fill_a_continuous_month():
    timestamps = pd.date_range("2026-01-01 07:00", periods=24 + 28 * 12 + 6, freq="2h")
    data = pd.DataFrame({"timestamp": timestamps, "treated_ntu": 0.5})
    target = data.iloc[24 : 24 + 28 * 12].copy()
    data.loc[target.index, "treated_ntu"] = np.nan
    target["treated_ntu"] = np.nan
    predictor = WindowMoEPredictor()

    points, daily = build_daily_features(data, predictor, target_data=target)

    assert predictor.rows == len(data)
    assert len(points) == 28 * 12
    assert (points["treated_ntu来源"] == "MoE预测").sum() == 28 * 12
    assert daily["MoE预测观测数"].eq(12).all()


def test_daily_features_fails_when_the_context_cannot_produce_finite_moe_values():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-02-01 07:00", periods=12, freq="2h"),
            "treated_ntu": np.nan,
        }
    )

    with pytest.raises(ValueError, match="无法由MoE回填"):
        build_daily_features(data, WindowMoEPredictor())


def test_fuzzy_clusters_use_only_complete_observed_2025_exceedance_days():
    daily = pd.DataFrame(
        {
            "运行日期": pd.date_range("2025-01-01", periods=4, freq="D"),
            "有效观测数": [12, 12, 12, 11],
            "实测观测数": [12, 12, 12, 11],
            "M": [0.2, 1.1, 2.2, 8.0],
            "F": [1 / 12, 3 / 12, 6 / 12, 1.0],
            "D": [2, 6, 12, 22],
            "L": [2, 4, 8, 22],
        }
    )

    model = fit_fuzzy_clusters(daily)
    classified = classify_q1(daily, model)

    assert model["train_rows"] == 3
    assert model["centers"].shape == (3, 4)
    assert set(classified.loc[:2, "聚类风险等级"]).issubset({1, 2, 3})
    assert classified.loc[3, "聚类风险等级"] in {1, 2, 3}


def test_grade_fusion_keeps_safety_zero_and_never_downgrades_baseline():
    daily = pd.DataFrame(
        {
            "日最大NTU": [1.0, 1.5, 2.2, 2.0],
            "M": [0.0, 0.5, 1.2, 1.0],
            "D": [0, 2, 2, 7],
            "聚类风险等级": [3, 3, 1, 1],
        }
    )

    result = fuse_grades(daily)

    assert result["基准风险等级"].tolist() == [0, 1, 2, 3]
    assert result["最终风险等级"].tolist() == [0, 3, 2, 3]


def test_grade_fusion_uses_baseline_when_cluster_grade_is_not_present():
    result = fuse_grades(pd.DataFrame({"日最大NTU": [1.5], "M": [0.5], "D": [2]}))

    assert result["最终风险等级"].tolist() == [1]
