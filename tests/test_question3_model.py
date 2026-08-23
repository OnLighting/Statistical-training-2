import ast
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

question3_model = (
    importlib.import_module("question3_model")
    if importlib.util.find_spec("question3_model") is not None
    else None
)
Solver = getattr(question3_model, "Solver", None)
from q3.data import HORIZONS
from q3.mechanistic import MechanisticExpert
from q3.moe import SoftmaxGate, _expand_gate_features, _gate_feature_matrix
from q3_fixtures import frame_fixture, supervised_fixture, upstream_fixture


def test_solver_and_solve_accept_no_configuration_arguments():
    assert Solver is not None
    assert list(inspect.signature(Solver).parameters) == []
    assert list(inspect.signature(Solver.solve).parameters) == ["self"]
    source = inspect.getsource(question3_model)
    assert "argparse" not in source
    assert "--quick" not in source
    assert "--output-dir" not in source


def test_solver_contains_only_approved_process_scenarios():
    assert Solver is not None
    solver = Solver()
    scenarios = solver._build_process_scenarios(
        frame_fixture(), pd.Timestamp("2026-02-01")
    )
    assert set(scenarios) == {
        "基准",
        "原水+20NTU",
        "原水+50NTU",
        "原水+100NTU",
        "矾量-0.01",
        "矾量+0.01",
        "原水+50NTU且矾量+0.01",
    }
    assert not any("sigma" in name.lower() or "标准差" in name for name in scenarios)


def test_target_prediction_origins_are_unique_and_chronologically_sorted():
    origins = Solver()._target_origins()

    assert origins.is_unique
    assert origins.is_monotonic_increasing
    assert list(origins.hour) == [5, 7, 5, 7, 5, 7]


def test_all_q3_production_modules_have_no_python_annotations():
    files = sorted((SRC_DIR / "q3").glob("*.py")) + [SRC_DIR / "question3_model.py"]
    violations = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                violations.append((path.name, node.lineno, "annotated assignment"))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = (
                    list(node.args.posonlyargs)
                    + list(node.args.args)
                    + list(node.args.kwonlyargs)
                )
                if node.args.vararg is not None:
                    arguments.append(node.args.vararg)
                if node.args.kwarg is not None:
                    arguments.append(node.args.kwarg)
                if node.returns is not None or any(arg.annotation is not None for arg in arguments):
                    violations.append((path.name, node.lineno, "function annotation"))
    assert violations == []


def test_solver_upstream_simulation_changes_filtered_path_after_raw_shock():
    solver = Solver()
    frame = upstream_fixture()
    solver._fit_upstream_arx(frame, frame.index[180])
    scenarios = solver._build_process_scenarios(frame, frame.index[181].normalize())
    base = solver._simulate_filtered(frame, scenarios["基准"])
    shocked = solver._simulate_filtered(frame, scenarios["原水+50NTU"])
    assert not np.allclose(base, shocked)


def test_zero_shock_and_raw_shock_share_recursive_arx_baseline():
    solver = Solver()
    frame = upstream_fixture()
    solver._fit_upstream_arx(frame, frame.index[180])
    date = frame.index[181].normalize()
    scenarios = solver._build_process_scenarios(frame, date)
    baseline = solver._scenario_filtered_frame(frame, scenarios["基准"])
    repeated = solver._scenario_filtered_frame(frame, scenarios["基准"])
    shocked = solver._scenario_filtered_frame(frame, scenarios["原水+50NTU"])

    np.testing.assert_allclose(baseline["filtered_ntu"], repeated["filtered_ntu"])
    affected = shocked.index >= scenarios["原水+50NTU"]["start"] + pd.Timedelta(hours=2)
    assert (shocked.loc[affected, "filtered_ntu"] - baseline.loc[affected, "filtered_ntu"]).max() > 0


def test_signed_peak_and_recovery_follow_maximum_absolute_response():
    solver = Solver()
    baseline = np.full(7, 0.4)
    difference = np.array([0.0, -0.01, -0.08, -0.04, -0.01, 0.0, 0.0])

    summary = solver._summarize_response(baseline, difference)

    assert summary["signed_peak"] == pytest.approx(-0.08)
    assert summary["peak_position"] == 2
    assert summary["recovery"] == 8


def test_upstream_block_bootstrap_refits_exactly_200_times_and_is_deterministic():
    solver = Solver()
    frame = upstream_fixture(periods=360)
    train_end = frame.index[300]
    solver._fit_upstream_arx(frame, train_end)

    first = solver._bootstrap_upstream_models(frame)
    second = solver._bootstrap_upstream_models(frame)

    assert len(first) == 200
    assert solver.bootstrap_refit_count_ == 200
    for left, right in zip(first, second):
        np.testing.assert_allclose(
            left["model"].named_steps["ridge"].coef_,
            right["model"].named_steps["ridge"].coef_,
            rtol=0,
            atol=0,
        )


def test_sensitivity_bootstrap_recomputes_paired_scenarios_without_residual_noise(monkeypatch):
    solver = Solver()
    states = [{"replicate": number} for number in range(200)]
    monkeypatch.setattr(solver, "_bootstrap_upstream_models", lambda frame: states)

    def paired_prediction(frame, scenario, upstream):
        value = upstream["replicate"]
        centered = (value - 99.5) / 99.5
        effect = scenario["raw_delta"] * 0.0002 * (1.0 + centered * 0.1)
        effect += scenario["alum_delta"] * -0.5 * (1.0 + centered * 0.05)
        path = np.linspace(0.0, effect, 7)
        return np.tile(0.3 + path, (3, 1))

    monkeypatch.setattr(solver, "_scenario_prediction_with_upstream", paired_prediction)
    records = solver._bootstrap_sensitivity_metrics(frame_fixture(periods=120))

    assert len(records) == 200 * 3 * 7
    assert records["bootstrap_replicate"].nunique() == 200
    assert np.isfinite(records[["signed_peak", "cumulative", "recovery"]]).all().all()
    baseline = records.loc[records["scenario"] == "基准"]
    assert (baseline[["signed_peak", "cumulative", "recovery"]] == 0).all().all()
    widths = (
        records.loc[records["date"] == pd.Timestamp("2026-02-10")]
        .groupby("scenario")["signed_peak"]
        .quantile(0.975)
        - records.loc[records["date"] == pd.Timestamp("2026-02-10")]
        .groupby("scenario")["signed_peak"]
        .quantile(0.025)
    )
    assert widths["原水+100NTU"] > widths["原水+20NTU"] > 0
    assert "residual" not in inspect.getsource(Solver._bootstrap_sensitivity_metrics)


def test_block_conformal_intervals_use_strict_fit_calibration_evaluation_chronology():
    solver = Solver()
    horizons = np.tile(np.array([2, 2, 4, 6, 8, 10, 12]), 3)
    blocks = []
    for number in range(24):
        start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=number * 8)
        prediction = np.full(21, 1.0)
        if number < 8:
            residual = np.full(21, number * 0.01)
        elif number < 16:
            residual = np.full(21, 0.40 + (number - 8) * 0.01)
        else:
            residual = np.full(21, 0.41 + (number - 16) * 0.005)
        blocks.append(
            {
                "start": start,
                "end": start + pd.Timedelta(days=6),
                "residual": residual,
                "prediction": prediction,
                "actual": prediction + residual,
                "horizon": horizons,
            }
        )

    calibration = solver._calibrate_intervals(blocks)
    coverage = calibration["coverage"]
    split = calibration["split"]

    assert coverage["预测步长/小时"].tolist() == [2, 4, 6, 8, 10, 12, "总体"]
    assert (coverage["评估坐标数"] > 0).all()
    assert max(block["end"] for block in split["fit"]) < min(
        block["start"] for block in split["calibration"]
    )
    assert max(block["end"] for block in split["calibration"]) < min(
        block["start"] for block in split["evaluation"]
    )
    for name in ("fit", "calibration", "evaluation"):
        ordered = split[name]
        assert all(
            ordered[position - 1]["end"] < ordered[position]["start"]
            for position in range(1, len(ordered))
        )
    assert set(id(block) for block in split["fit"]).isdisjoint(
        id(block) for block in split["calibration"] + split["evaluation"]
    )
    assert set(id(block) for block in split["calibration"]).isdisjoint(
        id(block) for block in split["evaluation"]
    )
    assert np.isfinite(calibration["expansion_80"])
    assert np.isfinite(calibration["expansion_95"])
    assert calibration["expansion_80"] >= 0
    assert calibration["expansion_95"] >= 0
    assert coverage.loc[6, "校准80%评估覆盖率"] >= 0.80
    assert coverage.loc[6, "校准95%评估覆盖率"] >= 0.95
    assert coverage.loc[6, "校准80%评估覆盖率"] > coverage.loc[6, "原始80%评估覆盖率"]
    assert coverage.loc[6, "校准95%评估覆盖率"] > coverage.loc[6, "原始95%评估覆盖率"]

    solver.interval_calibration_ = calibration
    point = np.full(21, 1.0)
    raw = solver._interval_bounds(
        point, [block["residual"] for block in split["fit"]]
    )
    exported = solver._prediction_intervals(point)
    assert np.all(exported[:, 0] <= raw[:, 0])
    assert np.all(exported[:, 1] >= raw[:, 1])
    assert np.all(exported[:, 2] <= exported[:, 0])
    assert np.all(exported[:, 3] >= exported[:, 1])


def test_forecast_coordinate_map_uses_exact_origins_and_horizon_models():
    coordinates = Solver()._forecast_coordinate_map()

    assert len(coordinates) == 21
    for date, group in pd.DataFrame(coordinates).groupby("date", sort=True):
        group = group.reset_index(drop=True)
        assert group.loc[0, "origin"] == pd.Timestamp(date) + pd.Timedelta(hours=5)
        assert group.loc[0, "horizon_hours"] == 2
        assert (group.loc[1:, "origin"] == pd.Timestamp(date) + pd.Timedelta(hours=7)).all()
        assert group.loc[1:, "horizon_hours"].tolist() == [2, 4, 6, 8, 10, 12]


def test_forecast_wording_distinguishes_the_05_and_07_origins():
    solver = Solver()

    assert "05:00" in solver._forecast_basis(0)
    assert "2小时" in solver._forecast_basis(0)
    assert "07:00" in solver._forecast_basis(1)
    assert "外生" in solver._forecast_basis(1)


def _fit_small_solver(monkeypatch):
    from q3.gru_expert import GRUExpert
    from q3.tree_expert import LightGBMExpert

    monkeypatch.setattr(LightGBMExpert, "NUM_LEAVES", (15,))
    monkeypatch.setattr(LightGBMExpert, "LEARNING_RATES", (0.05,))
    monkeypatch.setattr(LightGBMExpert, "MIN_CHILD_SAMPLES", (20,))
    monkeypatch.setattr(LightGBMExpert, "FEATURE_FRACTIONS", (1.0,))
    monkeypatch.setattr(LightGBMExpert, "MAX_ESTIMATORS", 8)
    monkeypatch.setattr(LightGBMExpert, "EARLY_STOPPING_ROUNDS", 2)
    monkeypatch.setattr(GRUExpert, "HIDDEN_SIZES", (32,))
    monkeypatch.setattr(GRUExpert, "DROPOUTS", (0.1,))
    monkeypatch.setattr(GRUExpert, "MAX_EPOCHS", 2)
    monkeypatch.setattr(GRUExpert, "EARLY_STOPPING_ROUNDS", 1)
    monkeypatch.setattr(SoftmaxGate, "EPOCHS", 3)

    frame, origins, targets = supervised_fixture(n=180)
    train_origins = origins[:-20]
    train_targets = targets[:-20]
    query_origins = origins[-10:]
    solver = Solver()
    solver.mechanistic.fit(frame, train_origins[-1])
    filled = solver.mechanistic.fill_target_history(frame)
    solver.tree.fit(frame, train_origins, train_targets)
    solver.gru.fit(frame, train_origins, train_targets, filled)
    expert = np.stack(
        (
            solver.mechanistic.predict(frame, query_origins),
            solver.tree.predict(frame, query_origins, filled),
            solver.gru.predict(frame, query_origins, filled),
        ),
        axis=2,
    )
    base = _gate_feature_matrix(frame, train_origins[-1], query_origins, filled)
    gate_x = _expand_gate_features(base, expert)
    solver.gate.fit(expert, gate_x, targets[-10:])
    solver.filled_target = filled
    solver.train_start = train_origins.min()
    solver.train_end = train_origins[-1]
    return solver, frame, query_origins[-3:]


def test_solver_model_round_trip_uses_relative_manifest_paths(tmp_path, monkeypatch):
    solver, frame, origins = _fit_small_solver(monkeypatch)
    monkeypatch.setattr(question3_model, "MODEL_DIR", tmp_path)
    expected = solver._predict_with_loaded_state(frame, origins)
    solver._save_models()

    loaded = Solver()
    loaded._load_models()
    actual = loaded._predict_with_loaded_state(frame, origins)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    manifest = json.loads((tmp_path / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["random_state"] == 2026
    assert len(manifest["files"]) >= 10
    assert all(not Path(item["path"]).is_absolute() for item in manifest["files"])


def test_solver_rejects_a_tampered_model_file(tmp_path, monkeypatch):
    solver, _, _ = _fit_small_solver(monkeypatch)
    monkeypatch.setattr(question3_model, "MODEL_DIR", tmp_path)
    solver._save_models()
    path = tmp_path / "mechanistic_expert.joblib"
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="模型文件哈希校验失败"):
        Solver()._load_models()


def test_solver_model_round_trip_supports_unicode_model_directory(tmp_path, monkeypatch):
    solver, frame, origins = _fit_small_solver(monkeypatch)
    model_dir = tmp_path / "模型目录"
    monkeypatch.setattr(question3_model, "MODEL_DIR", model_dir)
    expected = solver._predict_with_loaded_state(frame, origins)

    solver._save_models()
    loaded = Solver()._load_models()

    np.testing.assert_allclose(
        loaded._predict_with_loaded_state(frame, origins), expected, rtol=1e-6, atol=1e-7
    )


def test_fresh_process_saved_models_ignore_unavailable_target_injection(tmp_path, monkeypatch):
    solver, frame, origins = _fit_small_solver(monkeypatch)
    model_dir = tmp_path / "模型目录"
    monkeypatch.setattr(question3_model, "MODEL_DIR", model_dir)
    unavailable = origins[0] - pd.Timedelta(hours=2)
    baseline = frame.copy()
    baseline.loc[unavailable, "target_available"] = False
    baseline.loc[unavailable, "missing_treated_ntu"] = True
    baseline.loc[unavailable, "treated_ntu"] = np.nan
    injected = baseline.copy()
    injected.loc[unavailable, "treated_ntu"] = 9999.0
    payload_path = tmp_path / "availability.pkl"
    output_path = tmp_path / "predictions.npy"
    pd.to_pickle({"baseline": baseline, "injected": injected, "origins": origins}, payload_path)
    solver._save_models()
    program = (
        "import sys,numpy as np,pandas as pd; sys.path.insert(0,sys.argv[1]); "
        "import question3_model as q; q.MODEL_DIR=q.Path(sys.argv[2]); "
        "p=pd.read_pickle(sys.argv[3]); s=q.Solver()._load_models(); "
        "np.save(sys.argv[4],np.stack((s._predict_with_loaded_state(p['baseline'],p['origins']),"
        "s._predict_with_loaded_state(p['injected'],p['origins']))))"
    )

    subprocess.check_call(
        [sys.executable, "-c", program, str(SRC_DIR), str(model_dir), str(payload_path), str(output_path)]
    )

    predictions = np.load(output_path)
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=0, atol=0)


def _pipeline_result_fixture():
    dates = pd.to_datetime(["2026-02-01", "2026-02-10", "2026-02-20"])
    rows = []
    for date in dates:
        for number, hour in enumerate(range(7, 20, 2)):
            rows.append(
                {
                    "日期": date,
                    "时间": f"{hour:02d}:00",
                    "预测步长/小时": number * 2,
                    "机理专家预测NTU": 0.30,
                    "LightGBM预测NTU": 0.31,
                    "GRU预测NTU": 0.32,
                    "机理专家权重": 0.3,
                    "LightGBM权重": 0.4,
                    "GRU权重": 0.3,
                    "MoE预测NTU": 0.31,
                    "80%下限": 0.28,
                    "80%上限": 0.34,
                    "95%下限": 0.26,
                    "95%上限": 0.36,
                }
            )
    forecast = pd.DataFrame(rows)
    metric = pd.DataFrame(
        {"模型": ["MoE"], "预测步长/小时": [2], "RMSE": [0.01], "MAE": [0.01], "R2": [0.8], "样本数": [20]}
    )
    sensitivity = pd.DataFrame(
        {"目标日期": ["2026-02-01"], "情景": ["基准"], "预测峰值变化": [0.0], "峰值出现时间": ["07:00"], "12小时累计浊度增量": [0.0], "恢复时间/小时": [0], "恢复时间下界/小时": [0]}
    )
    quality = pd.DataFrame({"检查项": ["预测行数"], "结果": [21]})
    return {
        "forecast": forecast,
        "metrics": metric,
        "stratified_metrics": metric,
        "gap_backtest": metric,
        "gate_weights": pd.DataFrame({"预测步长/小时": [2], "机理专家权重": [0.3], "LightGBM权重": [0.4], "GRU权重": [0.3]}),
        "sensitivity": sensitivity,
        "quality": quality,
        "interval_coverage": pd.DataFrame({"预测步长/小时": [2], "80%验证覆盖率": [0.8], "95%验证覆盖率": [0.95], "样本数": [20]}),
        "feature_importance": pd.DataFrame({"horizon_hours": [2], "feature": ["raw_water_ntu"], "importance": [1.0]}),
        "shap_importance": pd.DataFrame({"预测步长/小时": [2], "特征": ["raw_water_ntu"], "平均绝对SHAP值": [0.1], "平均SHAP值": [0.01]}),
        "shap_dependence": pd.DataFrame({"预测步长/小时": [2], "特征": ["raw_water_ntu"], "特征值": [20.0], "SHAP值": [0.01]}),
        "shap_local": pd.DataFrame({"日期": ["2026-02-01"], "预测步长/小时": [2], "特征": ["raw_water_ntu"], "SHAP值": [0.01]}),
    }


def test_solver_writes_required_outputs_without_markdown(tmp_path, monkeypatch):
    solver = Solver()
    solver.result = _pipeline_result_fixture()
    monkeypatch.setattr(question3_model, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(question3_model, "MODEL_DIR", tmp_path / "models")

    solver._write_outputs()

    workbook = pd.ExcelFile(tmp_path / "表5_指定日期NTU预测结果.xlsx")
    forecast = pd.read_excel(workbook, sheet_name="指定日期预测")
    assert len(forecast) == 21
    assert forecast.groupby("日期").size().eq(7).all()
    assert set(workbook.sheet_names) == {
        "指定日期预测", "模型评价", "门控权重", "敏感性分析", "质量检查",
        "SHAP全局", "SHAP依赖", "SHAP局部", "区间覆盖率",
    }
    assert len(list(tmp_path.glob("表*.csv"))) == 5
    assert not list(tmp_path.rglob("*.md"))


def test_excel_forecast_weights_still_sum_to_one_after_four_decimal_formatting(tmp_path, monkeypatch):
    solver = Solver()
    solver.result = _pipeline_result_fixture()
    weight_columns = ["机理专家权重", "LightGBM权重", "GRU权重"]
    solver.result["forecast"].loc[:, weight_columns] = 1.0 / 3.0
    monkeypatch.setattr(question3_model, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(question3_model, "MODEL_DIR", tmp_path / "models")

    solver._write_outputs()
    forecast = pd.read_excel(
        tmp_path / "表5_指定日期NTU预测结果.xlsx", sheet_name="指定日期预测"
    )

    np.testing.assert_allclose(forecast[weight_columns].sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_solver_writes_exactly_six_png_figures_including_shap(tmp_path, monkeypatch):
    solver = Solver()
    solver.result = _pipeline_result_fixture()
    monkeypatch.setattr(question3_model, "OUTPUT_DIR", tmp_path)

    solver._plot_outputs()

    figures = list(tmp_path.glob("图*.png"))
    assert len(figures) == 6
    assert any("SHAP" in path.name for path in figures)


def test_plots_label_forecast_band_as_calibrated_and_keep_sensitivity_panels_title_free(
    tmp_path, monkeypatch
):
    solver = Solver()
    solver.result = _pipeline_result_fixture()
    figures = []
    monkeypatch.setattr(question3_model, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        question3_model,
        "save_figure",
        lambda figure, output_dir, name: figures.append((name, figure)),
    )

    solver._plot_outputs()

    forecast_figure = next(figure for name, figure in figures if name.startswith("图1"))
    assert "块保形校准95%区间" in forecast_figure.axes[0].get_legend_handles_labels()[1]
    sensitivity_figure = next(figure for name, figure in figures if name.startswith("图4"))
    assert len(sensitivity_figure.axes) == 2 * solver.result["sensitivity"]["目标日期"].nunique()
    assert all(not axis.get_title() for axis in sensitivity_figure.axes)
