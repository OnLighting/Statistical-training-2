from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import question2_eda as question2
from question2_eda import (
    arx_parameter_table,
    build_dynamic_design,
    chronological_split,
    create_delayed_scatter_figure,
    create_dynamic_overview_figure,
    fit_dynamic_models,
    identify_input_lags,
)


def test_identify_input_lags_recovers_a_four_hour_delay():
    rng = np.random.default_rng(2026)
    n = 360
    input_signal = rng.normal(size=n)
    target = np.roll(input_signal, 2) + rng.normal(scale=0.03, size=n)
    target[:2] = np.nan
    frame = pd.DataFrame(
        {"raw_water_ntu": input_signal, "filtered_ntu": target},
        index=pd.date_range("2025-01-01", periods=n, freq="2h"),
    )

    _, best = identify_input_lags(
        frame,
        input_columns=["raw_water_ntu"],
        max_lag_steps=6,
        seasonal_period=12,
    )

    assert best.iloc[0]["滞后阶数"] == 2
    assert best.iloc[0]["滞后小时"] == 4


def test_build_dynamic_design_aligns_input_and_target_lags():
    frame = pd.DataFrame(
        {
            "raw_water_ntu": [10.0, 20.0, 30.0, 40.0],
            "filtered_ntu": [0.1, 0.2, 0.3, 0.4],
            "is_backwash_event": [0.0, 0.0, 1.0, 0.0],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="2h"),
    )

    design = build_dynamic_design(
        frame,
        input_lags={"raw_water_ntu": 2},
        include_target_lag=True,
    )

    assert design.index.tolist() == frame.index[2:].tolist()
    assert design["raw_water_ntu_lag2"].tolist() == [10.0, 20.0]
    assert design["filtered_ntu_lag1"].tolist() == [0.2, 0.3]
    assert design["is_backwash_event"].tolist() == [1.0, 0.0]


def test_chronological_split_keeps_future_observations_out_of_training():
    frame = pd.DataFrame(
        {"filtered_ntu": np.arange(10, dtype=float)},
        index=pd.date_range("2025-01-01", periods=10, freq="2h"),
    )

    train, test = chronological_split(frame, test_fraction=0.2)

    assert len(train) == 8
    assert len(test) == 2
    assert train.index.max() < test.index.min()


def test_model_comparison_uses_only_timestamps_available_to_every_design():
    rng = np.random.default_rng(2026)
    n = 180
    frame = pd.DataFrame(
        {
            "raw_water_ntu": rng.normal(size=n),
            "raw_water_ph": rng.normal(size=n),
            "alum_dosage": rng.normal(size=n),
            "raw_water_flow": rng.normal(size=n),
            "filtered_ntu": rng.normal(size=n),
            "is_backwash_event": np.zeros(n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="2h"),
    )
    frame.loc[frame.index[30], "raw_water_flow"] = np.nan
    frame.loc[frame.index[34], "raw_water_flow"] = np.nan

    metrics, _, _, predictions, _ = fit_dynamic_models(
        frame,
        {
            "raw_water_ntu": 1,
            "raw_water_ph": 1,
            "alum_dosage": 2,
            "raw_water_flow": 3,
        },
    )

    assert len(metrics) == 3
    assert not predictions.empty
    assert predictions.notna().all().all()


def test_lag_correlations_writes_only_the_selected_lag_table(tmp_path, monkeypatch):
    rng = np.random.default_rng(2026)
    n = 180
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="2h"),
            "raw_water_ntu": rng.normal(size=n),
            "raw_water_ph": rng.normal(size=n),
            "alum_dosage": rng.normal(size=n),
            "raw_water_flow": rng.normal(size=n),
            "filtered_ntu": rng.normal(size=n),
            "is_backwash_event": np.zeros(n),
        }
    )
    monkeypatch.setattr(question2, "OUTPUT_DIR", tmp_path)

    question2.lag_correlations(data, max_lag=3)

    assert (tmp_path / "表2_初步时滞识别结果.csv").is_file()
    assert not (tmp_path / "表1_候选时滞相关系数.csv").exists()


def test_delayed_scatter_figure_has_one_panel_for_each_input():
    rng = np.random.default_rng(2026)
    frame = pd.DataFrame(
        {
            "raw_water_ntu": rng.normal(size=60),
            "raw_water_ph": rng.normal(size=60),
            "alum_dosage": rng.normal(size=60),
            "raw_water_flow": rng.normal(size=60),
            "filtered_ntu": rng.normal(size=60),
        },
        index=pd.date_range("2025-01-01", periods=60, freq="2h"),
    )
    best_lags = pd.DataFrame(
        {
            "变量": [
                "raw_water_ntu",
                "raw_water_ph",
                "alum_dosage",
                "raw_water_flow",
            ],
            "滞后阶数": [1, 1, 2, 2],
            "季节差分后相关系数": [0.4, 0.3, -0.2, 0.1],
        }
    )

    figure = create_delayed_scatter_figure(frame, best_lags)

    assert len(figure.axes) == 4
    assert all(axis.get_xlabel() for axis in figure.axes)
    plt.close(figure)


def test_dynamic_overview_has_four_requested_time_series_panels():
    timestamps = pd.date_range("2025-01-01", periods=48, freq="2h")
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "raw_water_ntu": np.linspace(10, 20, len(timestamps)),
            "raw_water_ph": np.linspace(6.9, 7.1, len(timestamps)),
            "raw_water_flow": np.linspace(40, 50, len(timestamps)),
            "filtered_ntu": np.linspace(0.04, 0.08, len(timestamps)),
        }
    )

    figure = create_dynamic_overview_figure(data)

    assert len(figure.axes) == 4
    assert [axis.get_ylabel() for axis in figure.axes] == [
        "原水浊度/NTU",
        "原水pH",
        "原水流量",
        "滤后水浊度/NTU",
    ]
    assert all(len(axis.lines) == 1 for axis in figure.axes)
    plt.close(figure)


def test_parameter_output_keeps_only_arx_rows():
    parameters = pd.DataFrame(
        {
            "模型": ["无时滞多元回归", "分布滞后模型(DLM)", "带外生输入自回归(ARX)"],
            "参数": ["const", "raw_water_ntu_lag1", "filtered_ntu_lag1"],
            "标准化系数": [0.1, 0.2, 0.3],
        }
    )

    result = arx_parameter_table(parameters)

    assert result.to_dict("records") == [
        {
            "模型": "带外生输入自回归(ARX)",
            "参数": "filtered_ntu_lag1",
            "标准化系数": 0.3,
        }
    ]
