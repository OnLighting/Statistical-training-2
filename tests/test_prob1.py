from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import plot1
from prob1 import (
    build_forecast_output,
    chronological_split_by_day,
    fit_and_predict_models,
    model_interpretation_tables,
    quantitative_feature_screening,
)


def test_select_log1p_features_uses_skewness_and_kurtosis_thresholds():
    frame = pd.DataFrame(
        {
            "heavy": [0.0] * 99 + [100.0],
            "regular": np.tile([0.0, 1.0, 2.0, 3.0], 25),
        }
    )

    selector = getattr(plot1, "select_log1p_features", lambda data, columns: [])
    selected = selector(frame, ["heavy", "regular"])

    assert selected == ["heavy"]


def test_apply_log1p_features_transforms_only_requested_columns():
    frame = pd.DataFrame({"heavy": [0.0, 3.0], "regular": [10.0, 20.0]})

    transformer = getattr(plot1, "apply_log1p_features", lambda data, columns: data.copy())
    transformed = transformer(frame, ["heavy"])

    np.testing.assert_allclose(transformed["heavy"], [0.0, np.log(4.0)])
    np.testing.assert_allclose(transformed["regular"], [10.0, 20.0])
    np.testing.assert_allclose(frame["heavy"], [0.0, 3.0])


def test_chronological_split_by_day_reserves_the_last_complete_days():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "operating_date": np.repeat(dates.date, 2),
            "timestamp": pd.date_range("2026-01-01 07:00", periods=12, freq="12h"),
            "treated_ntu": np.arange(12, dtype=float),
        }
    )

    train, test = chronological_split_by_day(frame, test_days=2)

    assert sorted(pd.unique(test["operating_date"])) == list(dates[-2:].date)
    assert max(train["operating_date"]) < min(test["operating_date"])
    assert len(train) == 8
    assert len(test) == 4


def test_fit_and_predict_models_returns_three_aligned_finite_forecasts():
    rng = np.random.default_rng(2026)
    timestamps = pd.date_range("2025-01-01 07:00", periods=144, freq="2h")
    feature_a = np.sin(np.arange(144) * 2 * np.pi / 12)
    feature_b = np.linspace(0, 1, 144)
    target = 0.3 + 0.08 * feature_a + 0.04 * feature_b + rng.normal(0, 0.005, 144)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "treated_ntu": target,
        }
    )
    train = frame.iloc[:120].copy()
    test = frame.iloc[120:].copy()

    predictions, _ = fit_and_predict_models(
        train,
        test,
        feature_columns=["feature_a", "feature_b"],
        random_state=2026,
    )

    assert set(predictions) == {"稳健周期时序回归", "SARIMAX", "随机森林"}
    for values in predictions.values():
        assert len(values) == len(test)
        assert np.isfinite(values).all()


def test_predictions_do_not_use_ground_truth_from_the_forecast_period():
    timestamps = pd.date_range("2025-01-01 07:00", periods=144, freq="2h")
    signal = 0.3 + 0.05 * np.sin(np.arange(144) * 2 * np.pi / 12)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature": np.cos(np.arange(144) * 2 * np.pi / 12),
            "treated_ntu": signal,
        }
    )
    train = frame.iloc[:120].copy()
    test = frame.iloc[120:].copy()
    changed_test = test.copy()
    changed_test["treated_ntu"] = 99.0

    original, _ = fit_and_predict_models(train, test, feature_columns=["feature"])
    changed, _ = fit_and_predict_models(train, changed_test, feature_columns=["feature"])

    for model_name in original:
        np.testing.assert_allclose(original[model_name], changed[model_name])


def test_quantitative_feature_screening_combines_three_methods():
    rng = np.random.default_rng(2026)
    n = 240
    signal = np.linspace(0, 3, n) + rng.normal(0, 0.05, n)
    noise = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "noise": noise,
            "treated_ntu": 0.4 * signal + rng.normal(0, 0.02, n),
        }
    )

    result = quantitative_feature_screening(
        frame, ["signal", "noise"], top_n=1, random_state=2026
    )

    assert {
        "变量",
        "差分Spearman绝对值",
        "LASSO绝对系数",
        "随机森林重要性",
        "综合排名",
        "是否入选",
    }.issubset(result.columns)
    assert result.iloc[0]["变量"] == "signal"
    assert bool(result.iloc[0]["是否入选"])
    assert not bool(result.iloc[1]["是否入选"])


def test_forecast_output_contains_only_date_time_and_prediction():
    future = pd.DataFrame(
        {
            "operating_date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
            "timestamp": pd.to_datetime(["2026-02-01 07:00", "2026-02-02 09:00"]),
        },
        index=[4, 7],
    )
    predictions = {
        "稳健周期时序回归": np.array([0.31, 0.32]),
        "SARIMAX": np.array([9.91, 9.92]),
        "随机森林": np.array([8.81, 8.82]),
    }

    result = build_forecast_output(future, future.iloc[[1]], predictions)

    assert result.columns.tolist() == ["日期", "时间", "预测NTU"]
    assert result.to_dict("records") == [
        {
            "日期": "2026-02-02",
            "时间": "09:00",
            "预测NTU": 0.32,
        }
    ]


def test_robust_coefficient_table_converts_coefficients_to_original_scale():
    robust = SimpleNamespace(named_steps={
        "scaler": SimpleNamespace(
            mean_=np.array([10.0, 20.0]),
            scale_=np.array([4.0, 5.0]),
        ),
        "model": SimpleNamespace(
            intercept_=0.7,
            coef_=np.array([2.0, -3.0]),
        ),
    })
    forest = SimpleNamespace(named_steps={
        "model": SimpleNamespace(feature_importances_=np.array([0.6, 0.4]))
    })
    sarimax = SimpleNamespace(param_names=["ar.L1"], params=np.array([0.2]))
    artifacts = {
        "特征名称": ["hour_sin", "hour_cos"],
        "稳健周期时序回归": robust,
        "随机森林": forest,
        "SARIMAX": sarimax,
    }

    coefficients, _, _ = model_interpretation_tables(artifacts)

    assert coefficients.columns.tolist() == [
        "变量名", "均值", "标准差", "标准化系数", "原始尺度系数", "影响方向"
    ]
    assert coefficients.iloc[0].to_dict() == {
        "变量名": "截距",
        "均值": 0.0,
        "标准差": 1.0,
        "标准化系数": 0.7,
        "原始尺度系数": 7.7,
        "影响方向": "正向",
    }
    np.testing.assert_allclose(
        coefficients.loc[1:, "原始尺度系数"].to_numpy(), [0.5, -0.6]
    )
