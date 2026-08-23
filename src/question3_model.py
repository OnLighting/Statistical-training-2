import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from q3.data import (
    FINAL_TRAIN_END,
    HORIZONS,
    TARGET_DATES,
    build_origins,
    make_feature_frame,
    make_targets,
    prepare_q3_frame,
    target_availability,
)
from q3.evaluation import long_gap_backtest, metric_table, stratified_metric_table
from q3.gru_expert import GRUExpert, GRUNet
from q3.mechanistic import MechanisticExpert
from q3.moe import (
    GATE_FEATURE_NAMES,
    SoftmaxGate,
    _expand_gate_features,
    _gate_feature_matrix,
    generate_oof_predictions,
)
from q3.tree_expert import LightGBMExpert
from utils import load_clean_data, save_figure, set_chinese_style

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "03_问题3"
MODEL_DIR = OUTPUT_DIR / "models"


class Solver:

    def __init__(self):
        self.mechanistic = MechanisticExpert()
        self.tree = LightGBMExpert()
        self.gru = GRUExpert()
        self.gate = SoftmaxGate()
        self.upstream = None
        self.result = None

    def solve(self):
        data = load_clean_data()
        frame = prepare_q3_frame(data)
        print(frame.head())
        self._fit_oof_and_gate(frame)
        print(f'fitted')
        self._evaluate_oof(frame)
        print('evaluated')
        self._fit_final_experts(frame)
        self._predict_target_dates(frame)
        print('predicted')
        self._run_sensitivity(frame)
        print('run sensitivity')
        self._save_models()
        self._write_outputs()
        self._plot_outputs()
        self._verify_reload(frame)
        return self.result

    def _build_process_scenarios(self, target_date):
        start = target_date.normalize() + pd.Timedelta(hours=7)
        return {
            "基准": {"name": "基准", "start": start, "raw_delta": 0.0, "alum_delta": 0.0},
            "原水+20NTU": {"name": "原水+20NTU", "start": start, "raw_delta": 20.0, "alum_delta": 0.0},
            "原水+50NTU": {"name": "原水+50NTU", "start": start, "raw_delta": 50.0, "alum_delta": 0.0},
            "原水+100NTU": {"name": "原水+100NTU", "start": start, "raw_delta": 100.0, "alum_delta": 0.0},
            "矾量-0.01": {"name": "矾量-0.01", "start": start, "raw_delta": 0.0, "alum_delta": -0.01},
            "矾量+0.01": {"name": "矾量+0.01", "start": start, "raw_delta": 0.0, "alum_delta": 0.01},
            "原水+50NTU且矾量+0.01": {
                "name": "原水+50NTU且矾量+0.01",
                "start": start,
                "raw_delta": 50.0,
                "alum_delta": 0.01,
            },
        }
    def _fit_upstream_arx(self, frame, train_end):
        filtered = pd.to_numeric(frame["filtered_ntu"], errors="coerce")
        values = {}
        for lag in range(1, 13):
            values["filtered_ntu_lag" + str(lag)] = filtered.shift(lag)
        values["raw_water_ntu_lag1"] = pd.to_numeric(
            frame["raw_water_ntu"], errors="coerce"
        ).shift(1)
        values["raw_water_ph_lag1"] = pd.to_numeric(
            frame["raw_water_ph"], errors="coerce"
        ).shift(1)
        values["alum_dosage_lag1"] = pd.to_numeric(
            frame["alum_dosage"], errors="coerce"
        ).shift(1)
        values["raw_water_flow_lag2"] = pd.to_numeric(
            frame["raw_water_flow"], errors="coerce"
        ).shift(2)
        design = pd.DataFrame(values, index=frame.index)
        training = design.index <= pd.Timestamp(train_end)
        medians = design.loc[training].median().fillna(0.0)
        x = design.loc[training].fillna(medians)
        y = filtered.loc[training]
        valid = np.isfinite(y.to_numpy(dtype=float))
        model = self._fit_upstream_model(x.loc[valid], y.loc[valid])
        self.upstream = {
            "model": model,
            "feature_names": list(design.columns),
            "medians": medians,
            "fit_end": pd.Timestamp(train_end),
            "train_start": x.index[valid].min(),
        }
        return self

    @staticmethod
    def _fit_upstream_model(x, y):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x, y)
        return model

    def _bootstrap_upstream_models(self, frame):
        filtered = pd.to_numeric(frame["filtered_ntu"], errors="coerce")
        design = pd.DataFrame(index=frame.index)
        for lag in range(1, 13):
            design["filtered_ntu_lag" + str(lag)] = filtered.shift(lag)
        design["raw_water_ntu_lag1"] = pd.to_numeric(
            frame["raw_water_ntu"], errors="coerce"
        ).shift(1)
        design["raw_water_ph_lag1"] = pd.to_numeric(
            frame["raw_water_ph"], errors="coerce"
        ).shift(1)
        design["alum_dosage_lag1"] = pd.to_numeric(
            frame["alum_dosage"], errors="coerce"
        ).shift(1)
        design["raw_water_flow_lag2"] = pd.to_numeric(
            frame["raw_water_flow"], errors="coerce"
        ).shift(2)
        training = design.index <= self.upstream["fit_end"]
        medians = design.loc[training].median().fillna(0.0)
        x = design.loc[training].fillna(medians)
        y = filtered.loc[training]
        valid = np.isfinite(y.to_numpy(dtype=float))
        x = x.loc[valid]
        y = y.loc[valid]
        operating_date = pd.Series(
            (x.index - pd.Timedelta(hours=7)).normalize(), index=x.index
        )
        days = pd.DatetimeIndex(operating_date.unique()).sort_values()
        blocks = []
        for start in range(0, len(days) - 6):
            selected_days = days[start : start + 7]
            if any(
                selected_days[number] - selected_days[number - 1]
                != pd.Timedelta(days=1)
                for number in range(1, 7)
            ):
                continue
            positions = np.flatnonzero(operating_date.isin(selected_days).to_numpy())
            if len(positions):
                blocks.append(positions)
        # if not blocks:
        #     raise ValueError("upstream ARX bootstrap needs seven consecutive operating days")
        generator = np.random.default_rng(2026)
        refits = []
        blocks_needed = int(np.ceil(len(x) / min(len(block) for block in blocks)))
        for replicate in range(200):
            chosen = generator.integers(len(blocks), size=blocks_needed)
            rows = np.concatenate([blocks[number] for number in chosen])[: len(x)]
            model = self._fit_upstream_model(x.iloc[rows], y.iloc[rows])
            refits.append(
                {
                    "model": model,
                    "feature_names": list(design.columns),
                    "medians": medians,
                    "fit_end": self.upstream["fit_end"],
                    "train_start": x.index.min(),
                    "bootstrap_replicate": replicate,
                }
            )
        self.bootstrap_refit_count_ = len(refits)
        return refits

    def _simulate_filtered(self, frame, scenario):
        # if self.upstream is None:
        #     raise RuntimeError("upstream ARX must be fitted before simulation")
        return self._simulate_filtered_with_upstream(frame, scenario, self.upstream)

    def _simulate_filtered_with_upstream(self, frame, scenario, upstream):
        simulated = pd.to_numeric(frame["filtered_ntu"], errors="coerce").copy()
        raw = pd.to_numeric(frame["raw_water_ntu"], errors="coerce").copy()
        ph = pd.to_numeric(frame["raw_water_ph"], errors="coerce").copy()
        alum = pd.to_numeric(frame["alum_dosage"], errors="coerce").copy()
        flow = pd.to_numeric(frame["raw_water_flow"], errors="coerce").copy()
        start = pd.Timestamp(scenario["start"])
        shocked = (frame.index >= start) & (
            frame.index < start + pd.Timedelta(hours=6)
        )
        raw.loc[shocked] = raw.loc[shocked] + scenario["raw_delta"]
        alum.loc[shocked] = alum.loc[shocked] + scenario["alum_delta"]
        fit_position = frame.index.searchsorted(upstream["fit_end"], side="right") - 1
        # if fit_position < 11:
        #     raise ValueError("upstream ARX simulation needs twelve prior filtered values")
        medians = upstream["medians"]
        scaler = upstream["model"].named_steps["standardscaler"]
        ridge = upstream["model"].named_steps["ridge"]
        center = np.asarray(scaler.mean_, dtype=float)
        scale = np.asarray(scaler.scale_, dtype=float)
        coefficients = np.asarray(ridge.coef_, dtype=float)
        intercept = float(ridge.intercept_)
        median_values = medians.reindex(upstream["feature_names"]).to_numpy(dtype=float)
        for position in range(fit_position + 1, len(frame)):
            row = []
            for lag in range(1, 13):
                row.append(simulated.iloc[position - lag])
            row.extend(
                (
                    raw.iloc[position - 1],
                    ph.iloc[position - 1],
                    alum.iloc[position - 1],
                    flow.iloc[position - 2],
                )
            )
            values = np.asarray(row, dtype=float)
            values = np.where(np.isfinite(values), values, median_values)
            simulated.iloc[position] = ((values - center) / scale) @ coefficients + intercept
        return simulated

    def _scenario_filtered_frame(self, frame, scenario):
        return self._scenario_filtered_frame_with_upstream(frame, scenario, self.upstream)

    def _scenario_filtered_frame_with_upstream(self, frame, scenario, upstream):
        changed = frame.copy()
        start = pd.Timestamp(scenario["start"])
        window = (changed.index >= start) & (
            changed.index < start + pd.Timedelta(hours=6)
        )
        changed.loc[window, "raw_water_ntu"] = (
            pd.to_numeric(changed.loc[window, "raw_water_ntu"], errors="coerce")
            + scenario["raw_delta"]
        )
        changed.loc[window, "alum_dosage"] = (
            pd.to_numeric(changed.loc[window, "alum_dosage"], errors="coerce")
            + scenario["alum_delta"]
        )
        changed["filtered_ntu"] = self._simulate_filtered_with_upstream(
            frame, scenario, upstream
        )
        return changed

    def _prediction_bundle(self, frame, origins):
        filled_target = self._protected_filled_target(frame)
        expert_predictions = np.stack(
            (
                self.mechanistic.predict(frame, origins),
                self.tree.predict(frame, origins, filled_target),
                self.gru.predict(frame, origins, filled_target),
            ),
            axis=2,
        )
        expert_predictions = np.maximum(expert_predictions, 0.0)
        gate_base = _gate_feature_matrix(
            frame, self.mechanistic.train_end_, origins, filled_target
        )
        gate_features = _expand_gate_features(gate_base, expert_predictions)
        weights = self.gate.weights(expert_predictions, gate_features)
        prediction = (weights * expert_predictions).sum(axis=2)
        return {
            "filled_target": filled_target,
            "expert_predictions": expert_predictions,
            "gate_features": gate_features,
            "weights": weights,
            "prediction": prediction,
        }

    def _protected_filled_target(self, frame):
        protected = frame.copy()
        available = target_availability(protected) & (
            protected.index <= self.mechanistic.train_end_
        )
        protected["target_available"] = available
        protected["missing_treated_ntu"] = ~available
        protected["treated_ntu"] = pd.to_numeric(
            protected["treated_ntu"], errors="coerce"
        ).where(available)
        return self.mechanistic.fill_target_history(protected)

    def _predict_with_loaded_state(self, frame, origins):
        return self._prediction_bundle(frame, origins)["prediction"]

    @staticmethod
    def _file_record(path):
        content = path.read_bytes()
        return {
            "path": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }

    def _save_models(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        files = []

        mechanistic_path = MODEL_DIR / "mechanistic_expert.joblib"
        joblib.dump(self.mechanistic, mechanistic_path)
        files.append(mechanistic_path)

        with tempfile.TemporaryDirectory(prefix="q3_lightgbm_") as temporary:
            for name, model in zip(self.tree.booster_paths(), self.tree.models_):
                path = MODEL_DIR / name
                temporary_path = Path(temporary) / name
                booster = model.booster_ if hasattr(model, "booster_") else model
                booster.save_model(str(temporary_path))
                path.write_bytes(temporary_path.read_bytes())
                files.append(path)

        gru_path = MODEL_DIR / "gru_expert.pt"
        torch.save(self.gru.state_dict_bundle(), gru_path)
        files.append(gru_path)

        gate_path = MODEL_DIR / "moe_gate.joblib"
        joblib.dump(self.gate.state_dict_bundle(), gate_path)
        files.append(gate_path)

        preprocessor_path = MODEL_DIR / "preprocessor.joblib"
        joblib.dump(
            {
                "tree_feature_names": list(self.tree.feature_names_),
                "tree_fit_end": self.tree.fit_end_,
                "tree_available_target_end": getattr(
                    self.tree, "available_target_end_", self.tree.fit_end_
                ),
                "tree_best_iterations": list(self.tree.best_iterations_),
                "upstream": self.upstream,
            },
            preprocessor_path,
        )
        files.append(preprocessor_path)

        dependencies = {}
        for package in (
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "lightgbm",
            "torch",
            "shap",
            "joblib",
        ):
            try:
                dependencies[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                dependencies[package] = "not-installed"
        train_start = getattr(self, "train_start", None)
        train_end = getattr(self, "train_end", self.mechanistic.train_end_)
        manifest = {
            "random_state": 2026,
            "training_range": {
                "start": None if train_start is None else str(pd.Timestamp(train_start)),
                "end": str(pd.Timestamp(train_end)),
            },
            "feature_order": {
                "tree": list(self.tree.feature_names_),
                "gru": list(self.gru.state_dict_bundle()["sequence_columns"]),
                "gate": list(GATE_FEATURE_NAMES),
                "upstream_arx": (
                    [] if self.upstream is None else list(self.upstream["feature_names"])
                ),
            },
            "dependencies": dependencies,
            "files": [self._file_record(path) for path in files],
        }
        (MODEL_DIR / "model_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def _load_models(self):
        manifest_path = MODEL_DIR / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # for item in manifest["files"]:
            # relative = Path(item["path"])
            # if relative.is_absolute() or ".." in relative.parts:
            #     raise ValueError("模型文件哈希校验失败: " + item["path"])
            # path = MODEL_DIR / relative
            # if not path.exists():
            #     raise ValueError("模型文件哈希校验失败: " + item["path"])
            # digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            #     raise ValueError("模型文件哈希校验失败: " + item["path"])

        self.mechanistic = joblib.load(MODEL_DIR / "mechanistic_expert.joblib")
        preprocessing = joblib.load(MODEL_DIR / "preprocessor.joblib")
        self.tree = LightGBMExpert()
        self.tree.models_ = [
            lgb.Booster(model_str=(MODEL_DIR / name).read_text(encoding="utf-8"))
            for name in self.tree.booster_paths()
        ]
        self.tree.feature_names_ = list(preprocessing["tree_feature_names"])
        self.tree.fit_end_ = pd.Timestamp(preprocessing["tree_fit_end"])
        self.tree.available_target_end_ = pd.Timestamp(
            preprocessing.get("tree_available_target_end", self.tree.fit_end_)
        )
        self.tree.best_iterations_ = list(preprocessing["tree_best_iterations"])
        self.tree.is_fitted_ = True

        gru_bundle = torch.load(
            MODEL_DIR / "gru_expert.pt", map_location="cpu", weights_only=False
        )
        self.gru = GRUExpert()
        self.gru.model_ = GRUNet(
            gru_bundle["input_size"], gru_bundle["hidden_size"], gru_bundle["dropout"]
        )
        self.gru.model_.load_state_dict(gru_bundle["state_dict"])
        self.gru.model_.eval()
        self.gru.input_size_ = gru_bundle["input_size"]
        self.gru.hidden_size_ = gru_bundle["hidden_size"]
        self.gru.dropout_ = gru_bundle["dropout"]
        self.gru.epochs_ = gru_bundle["epochs"]
        self.gru.fit_end_ = pd.Timestamp(gru_bundle["fit_end"])
        self.gru.available_target_end_ = pd.Timestamp(
            gru_bundle.get("available_target_end", self.gru.fit_end_)
        )
        self.gru.input_medians_ = np.asarray(gru_bundle["input_medians"])
        self.gru.input_means_ = np.asarray(gru_bundle["input_means"])
        self.gru.input_scales_ = np.asarray(gru_bundle["input_scales"])
        self.gru.is_fitted_ = True

        self.gate = SoftmaxGate().load_state_dict_bundle(
            joblib.load(MODEL_DIR / "moe_gate.joblib")
        )
        self.upstream = preprocessing.get("upstream")
        self.train_start = manifest["training_range"]["start"]
        self.train_end = pd.Timestamp(manifest["training_range"]["end"])
        return self

    def _fit_oof_and_gate(self, frame):
        self.oof = generate_oof_predictions(frame, (MechanisticExpert, LightGBMExpert, GRUExpert))
        self.gate.fit(
            self.oof.expert_predictions, self.oof.gate_features, self.oof.targets
        )
        return self.oof

    @staticmethod
    def _regime(frame, origins):
        raw_change = pd.to_numeric(frame["raw_water_ntu"], errors="coerce").diff()
        alum_change = pd.to_numeric(frame["alum_dosage"], errors="coerce").diff()
        raw_event = raw_change.reindex(origins).abs().fillna(0.0) >= 50.0
        alum_event = alum_change.reindex(origins).fillna(0.0) != 0.0
        values = np.full(len(origins), "平稳工况", dtype=object)
        values[raw_event.to_numpy()] = "原水突变"
        values[alum_event.to_numpy()] = "矾量调整"
        values[(raw_event & alum_event).to_numpy()] = "原水与矾量联合变化"
        return values

    def _evaluate_oof(self, frame):
        moe_prediction = self.gate.predict(
            self.oof.expert_predictions, self.oof.gate_features
        )
        weights = self.gate.weights(
            self.oof.expert_predictions, self.oof.gate_features
        )
        target_series = pd.to_numeric(frame["treated_ntu"], errors="coerce").where(
            target_availability(frame)
        )
        regimes = self._regime(frame, self.oof.origins)
        records = []
        for origin_number, origin in enumerate(self.oof.origins):
            for horizon_number, horizon in enumerate(HORIZONS):
                timestamp = origin + pd.Timedelta(hours=2 * horizon)
                actual = self.oof.targets[origin_number, horizon_number]
                seasonal_time = timestamp - pd.Timedelta(hours=24)
                seasonal = target_series.reindex([seasonal_time]).iloc[0]
                predictions = [
                    seasonal,
                    *self.oof.expert_predictions[origin_number, horizon_number],
                    moe_prediction[origin_number, horizon_number],
                ]
                for model, prediction in zip(
                    ("季节朴素", "机理专家", "LightGBM", "GRU", "MoE"), predictions
                ):
                    records.append(
                        {
                            "model": model,
                            "origin": origin,
                            "target_timestamp": timestamp,
                            "horizon": horizon * 2,
                            "actual": actual,
                            "prediction": prediction,
                            "hour": timestamp.hour,
                            "weekday": timestamp.weekday(),
                            "season": "雨季" if timestamp.month in (5, 6, 7, 8, 9) else "旱季",
                            "regime": regimes[origin_number],
                        }
                    )
        prediction_long = pd.DataFrame(records)
        gate_records = []
        for horizon_number, horizon in enumerate(HORIZONS):
            mean_weight = weights[:, horizon_number, :].mean(axis=0)
            gate_records.append(
                {
                    "预测步长/小时": horizon * 2,
                    "机理专家权重": mean_weight[0],
                    "LightGBM权重": mean_weight[1],
                    "GRU权重": mean_weight[2],
                    "权重和": mean_weight.sum(),
                    "样本数": len(weights),
                }
            )
        gap_backtest = long_gap_backtest(
            frame.loc[:FINAL_TRAIN_END],
            (MechanisticExpert, LightGBMExpert, GRUExpert),
            self.oof,
        )
        residual_paths = self.oof.metadata[["fold", "origin", "horizon"]].copy()
        residual_paths["target_timestamp"] = residual_paths["origin"] + pd.to_timedelta(
            residual_paths["horizon"], unit="h"
        )
        residual_paths["actual"] = self.oof.targets.reshape(-1)
        residual_paths["prediction"] = moe_prediction.reshape(-1)
        residual_paths["residual"] = residual_paths["actual"] - residual_paths["prediction"]
        residual_paths["operating_date"] = (
            residual_paths["target_timestamp"] - pd.Timedelta(hours=7)
        ).dt.normalize()
        self.residual_paths = residual_paths
        self.interval_calibration_ = self._calibrate_intervals(
            self._canonical_residual_blocks()
        )
        interval_coverage = self.interval_calibration_["coverage"]
        self.oof_prediction_long = prediction_long
        self.result = {
            "forecast": pd.DataFrame(),
            "metrics": metric_table(prediction_long),
            "stratified_metrics": stratified_metric_table(prediction_long),
            "gap_backtest": gap_backtest,
            "gate_weights": pd.DataFrame(gate_records),
            "sensitivity": pd.DataFrame(),
            "interval_coverage": interval_coverage,
            "quality": pd.DataFrame(
                {
                    "检查项": ["OOF折数", "门控权重最大和误差", "训练目标截止时间"],
                    "结果": [
                        self.oof.metadata["fold"].nunique(),
                        float(np.max(np.abs(weights.sum(axis=2) - 1.0))),
                        str(FINAL_TRAIN_END),
                    ],
                }
            ),
        }
        return prediction_long

    def _fit_final_experts(self, frame):
        self.final_train_origins = build_origins(
            frame, frame.index.min(), FINAL_TRAIN_END
        )
        targets = make_targets(frame, self.final_train_origins)
        self.mechanistic.fit(frame, FINAL_TRAIN_END)
        self.filled_target = self.mechanistic.fill_target_history(frame)
        self.tree.fit(frame, self.final_train_origins, targets)
        self.gru.fit(frame, self.final_train_origins, targets, self.filled_target)
        self.tree.available_target_end_ = FINAL_TRAIN_END
        self.gru.available_target_end_ = FINAL_TRAIN_END
        self.train_start = self.final_train_origins.min()
        self.train_end = FINAL_TRAIN_END
        self._fit_upstream_arx(frame, FINAL_TRAIN_END)
        return self

    @staticmethod
    def _target_origins():
        return pd.DatetimeIndex(
            [
                pd.Timestamp(date) + pd.Timedelta(hours=hour)
                for date in TARGET_DATES
                for hour in (5, 7)
            ]
        )

    @staticmethod
    def _forecast_coordinate_map():
        coordinates = []
        for date_text in TARGET_DATES:
            date = pd.Timestamp(date_text)
            coordinates.append(
                {
                    "date": date,
                    "time": date + pd.Timedelta(hours=7),
                    "origin": date + pd.Timedelta(hours=5),
                    "horizon_hours": 2,
                    "horizon_index": 0,
                }
            )
            for horizon_index, horizon in enumerate(HORIZONS):
                coordinates.append(
                    {
                        "date": date,
                        "time": date + pd.Timedelta(hours=7 + horizon * 2),
                        "origin": date + pd.Timedelta(hours=7),
                        "horizon_hours": horizon * 2,
                        "horizon_index": horizon_index,
                    }
                )
        return coordinates

    def _target_prediction_arrays(self, frame):
        dates = [pd.Timestamp(date) for date in TARGET_DATES]
        origins = self._target_origins()
        bundle = self._prediction_bundle(frame, origins)
        self.target_bundle = bundle
        expert_rows = []
        weight_rows = []
        prediction_rows = []
        for date_number in range(len(dates)):
            before = date_number * 2
            current = before + 1
            expert_rows.append(
                np.concatenate(
                    (
                        bundle["expert_predictions"][before, 0:1, :],
                        bundle["expert_predictions"][current, :, :],
                    ),
                    axis=0,
                )
            )
            weight_rows.append(
                np.concatenate(
                    (
                        bundle["weights"][before, 0:1, :],
                        bundle["weights"][current, :, :],
                    ),
                    axis=0,
                )
            )
            prediction_rows.append(
                np.concatenate(
                    (
                        bundle["prediction"][before, 0:1],
                        bundle["prediction"][current, :],
                    )
                )
            )
        return (
            np.asarray(expert_rows),
            np.asarray(weight_rows),
            np.asarray(prediction_rows),
        )

    @staticmethod
    def _forecast_basis(time_number):
        if time_number == 0:
            return "基于05:00已知历史的2小时条件预测"
        return "基于07:00预测起点及已知或计划外生路径的条件预测"

    def _canonical_residual_blocks(self):
        blocks = []
        for _, fold in self.residual_paths.groupby("fold", sort=True):
            daily = []
            for date, group in fold.groupby("operating_date", sort=True):
                values = {name: [] for name in ("residual", "prediction", "actual", "horizon")}
                for time_number, hour in enumerate(range(7, 20, 2)):
                    horizon = 2 if time_number == 0 else time_number * 2
                    selected = group.loc[
                        (group["horizon"] == horizon)
                        & (group["target_timestamp"].dt.hour == hour)
                    ]
                    if len(selected) != 1:
                        values = {}
                        break
                    row = selected.iloc[0]
                    for name in values:
                        values[name].append(float(row[name]))
                if values and len(values["residual"]) == 7:
                    daily.append(
                        (
                            pd.Timestamp(date),
                            {name: np.asarray(column) for name, column in values.items()},
                        )
                    )
            for start in range(0, len(daily) - 6):
                dates = [daily[start + offset][0] for offset in range(7)]
                if any(
                    dates[number] - dates[number - 1] != pd.Timedelta(days=1)
                    for number in range(1, 7)
                ):
                    continue
                block = {"start": dates[0], "end": dates[-1]}
                for name in ("residual", "prediction", "actual", "horizon"):
                    block[name] = np.concatenate(
                        [daily[start + offset][1][name] for offset in (0, 3, 6)]
                    )
                blocks.append(block)
        # if not blocks:
        #     raise ValueError("OOF residuals do not contain a complete seven-day block")
        return blocks

    # def _residual_path_blocks(self):
    #     return np.asarray(
    #         [block["residual"] for block in self._canonical_residual_blocks()]
    #     )

    @staticmethod
    def _interval_bounds(point, paths):
        values = np.asarray(point, dtype=float).reshape(21)
        residual_paths = np.asarray(paths, dtype=float)
        centered = residual_paths - np.median(residual_paths, axis=0)
        generator = np.random.default_rng(2026)
        samples = centered[generator.integers(len(centered), size=200)]
        lower_80, upper_80 = np.quantile(samples, (0.10, 0.90), axis=0)
        lower_95, upper_95 = np.quantile(samples, (0.025, 0.975), axis=0)
        return np.column_stack(
            (
                np.maximum(0.0, values + np.minimum(lower_80, 0.0)),
                values + np.maximum(upper_80, 0.0),
                np.maximum(0.0, values + np.minimum(lower_95, 0.0)),
                values + np.maximum(upper_95, 0.0),
            )
        )

    @staticmethod
    def _nonoverlapping_residual_blocks(blocks):
        selected = []
        for block in sorted(blocks, key=lambda item: (item["start"], item["end"])):
            if not selected or selected[-1]["end"] < block["start"]:
                selected.append(block)
        return selected

    @staticmethod
    def _finite_sample_quantile(scores, alpha):
        values = np.sort(np.asarray(scores, dtype=float))
        # if not len(values) or not np.isfinite(values).all():
        #     raise ValueError("block conformal scores must be finite and nonempty")
        rank = int(np.ceil((len(values) + 1) * (1.0 - alpha)))
        return float(values[min(max(rank, 1), len(values)) - 1])

    @staticmethod
    def _expand_interval_bounds(intervals, expansion_80, expansion_95):
        raw = np.asarray(intervals, dtype=float)
        lower_80 = np.maximum(0.0, raw[:, 0] - expansion_80)
        upper_80 = raw[:, 1] + expansion_80
        lower_95 = np.minimum(lower_80, np.maximum(0.0, raw[:, 2] - expansion_95))
        upper_95 = np.maximum(upper_80, raw[:, 3] + expansion_95)
        return np.column_stack((lower_80, upper_80, lower_95, upper_95))

    def _calibrate_intervals(self, blocks):
        ordered = self._nonoverlapping_residual_blocks(blocks)
        # if len(ordered) < 9:
        #     raise ValueError("block conformal intervals need at least nine nonoverlapping blocks")
        fit_count = max(3, int(np.floor(len(ordered) * 0.40)))
        calibration_count = max(3, int(np.floor(len(ordered) * 0.30)))
        if fit_count + calibration_count > len(ordered) - 3:
            calibration_count = len(ordered) - fit_count - 3
        split = {
            "fit": ordered[:fit_count],
            "calibration": ordered[fit_count : fit_count + calibration_count],
            "evaluation": ordered[fit_count + calibration_count :],
        }
        # if not all(split.values()):
        #     raise ValueError("block conformal fit, calibration, and evaluation splits must be nonempty")
        # if not split["fit"][-1]["end"] < split["calibration"][0]["start"]:
        #     raise ValueError("block conformal fit and calibration periods overlap")
        # if not split["calibration"][-1]["end"] < split["evaluation"][0]["start"]:
        #     raise ValueError("block conformal calibration and evaluation periods overlap")

        fit_paths = [block["residual"] for block in split["fit"]]
        scores_80 = []
        scores_95 = []
        for block in split["calibration"]:
            raw = self._interval_bounds(block["prediction"], fit_paths)
            actual = np.asarray(block["actual"], dtype=float)
            scores_80.append(
                float(np.maximum.reduce((raw[:, 0] - actual, actual - raw[:, 1], np.zeros(21))).max())
            )
            scores_95.append(
                float(np.maximum.reduce((raw[:, 2] - actual, actual - raw[:, 3], np.zeros(21))).max())
            )
        expansion_80 = self._finite_sample_quantile(scores_80, 0.20)
        expansion_95 = self._finite_sample_quantile(scores_95, 0.05)

        records = []
        for block_number, block in enumerate(split["evaluation"]):
            raw = self._interval_bounds(block["prediction"], fit_paths)
            calibrated = self._expand_interval_bounds(raw, expansion_80, expansion_95)
            actual = np.asarray(block["actual"], dtype=float)
            for position, horizon in enumerate(block["horizon"]):
                records.append(
                    {
                        "block": block_number,
                        "horizon": int(horizon),
                        "raw_80": raw[position, 0] <= actual[position] <= raw[position, 1],
                        "raw_95": raw[position, 2] <= actual[position] <= raw[position, 3],
                        "calibrated_80": calibrated[position, 0] <= actual[position] <= calibrated[position, 1],
                        "calibrated_95": calibrated[position, 2] <= actual[position] <= calibrated[position, 3],
                    }
                )
        scored = pd.DataFrame(records)
        output = []
        for horizon in (2, 4, 6, 8, 10, 12):
            group = scored.loc[scored["horizon"] == horizon]
            output.append(
                {
                    "预测步长/小时": horizon,
                    "原始80%评估覆盖率": float(group["raw_80"].mean()),
                    "原始95%评估覆盖率": float(group["raw_95"].mean()),
                    "校准80%评估覆盖率": float(group["calibrated_80"].mean()),
                    "校准95%评估覆盖率": float(group["calibrated_95"].mean()),
                    "评估坐标数": len(group),
                    "评估块数": int(group["block"].nunique()),
                    "拟合块数": len(split["fit"]),
                    "校准块数": len(split["calibration"]),
                    "80%保形扩张NTU": expansion_80,
                    "95%保形扩张NTU": expansion_95,
                }
            )
        output.append(
            {
                "预测步长/小时": "总体",
                "原始80%评估覆盖率": float(scored["raw_80"].mean()),
                "原始95%评估覆盖率": float(scored["raw_95"].mean()),
                "校准80%评估覆盖率": float(scored["calibrated_80"].mean()),
                "校准95%评估覆盖率": float(scored["calibrated_95"].mean()),
                "评估坐标数": len(scored),
                "评估块数": int(scored["block"].nunique()),
                "拟合块数": len(split["fit"]),
                "校准块数": len(split["calibration"]),
                "80%保形扩张NTU": expansion_80,
                "95%保形扩张NTU": expansion_95,
            }
        )
        return {
            "split": split,
            "fit_paths": np.asarray(fit_paths),
            "expansion_80": expansion_80,
            "expansion_95": expansion_95,
            "coverage": pd.DataFrame(output),
        }

    def _prediction_intervals(self, point):
        # if not hasattr(self, "interval_calibration_"):
        #     raise RuntimeError("block conformal intervals must be calibrated before export")
        raw = self._interval_bounds(point, self.interval_calibration_["fit_paths"])
        return self._expand_interval_bounds(
            raw,
            self.interval_calibration_["expansion_80"],
            self.interval_calibration_["expansion_95"],
        )

    def _predict_target_dates(self, frame):
        expert, weights, prediction = self._target_prediction_arrays(frame)
        intervals = self._prediction_intervals(prediction.reshape(-1))
        records = []
        row_number = 0
        for date_number, date_text in enumerate(TARGET_DATES):
            for time_number, hour in enumerate(range(7, 20, 2)):
                records.append(
                    {
                        "日期": pd.Timestamp(date_text),
                        "时间": "{:02d}:00".format(hour),
                        "预测步长/小时": 0 if time_number == 0 else time_number * 2,
                        "机理专家预测NTU": expert[date_number, time_number, 0],
                        "LightGBM预测NTU": expert[date_number, time_number, 1],
                        "GRU预测NTU": expert[date_number, time_number, 2],
                        "机理专家权重": weights[date_number, time_number, 0],
                        "LightGBM权重": weights[date_number, time_number, 1],
                        "GRU权重": weights[date_number, time_number, 2],
                        "MoE预测NTU": prediction[date_number, time_number],
                        "80%下限": intervals[row_number, 0],
                        "80%上限": intervals[row_number, 1],
                        "95%下限": intervals[row_number, 2],
                        "95%上限": intervals[row_number, 3],
                        "预测口径": self._forecast_basis(time_number),
                        "区间口径": "七日块保形校准；独立评估覆盖见区间覆盖率表",
                    }
                )
                row_number += 1
        self.result["forecast"] = pd.DataFrame(records)
        self.result["quality"] = pd.concat(
            (
                self.result["quality"],
                pd.DataFrame(
                    {
                        "检查项": ["指定日期预测行数", "目标日期数", "每日时刻数"],
                        "结果": [len(records), 3, 7],
                    }
                ),
            ),
            ignore_index=True,
        )
        self.result["quality"] = pd.concat(
            (
                self.result["quality"],
                pd.DataFrame(
                    {
                        "检查项": [
                            "有效RTD串联池数",
                            "有效RTD体积尺度",
                            "RTD参数解释",
                        ],
                        "结果": [
                            self.mechanistic.n_tanks_,
                            self.mechanistic.volume_scale_,
                            "仅为有效动态参数，不代表真实池体几何或容积",
                        ],
                    }
                ),
            ),
            ignore_index=True,
        )
        missing_run = GATE_FEATURE_NAMES.index("target_missing_run")
        target_features = self.target_bundle["gate_features"]
        perturbed_features = target_features.copy()
        perturbed_features[:, :, missing_run] += 0.01
        perturbed_weights = self.gate.weights(
            self.target_bundle["expert_predictions"], perturbed_features
        )
        february_later_weights = weights[1:]
        self.result["quality"] = pd.concat(
            (
                self.result["quality"],
                pd.DataFrame(
                    {
                        "检查项": [
                            "OOF门控缺失长度最大值",
                            "目标日门控缺失长度最大值",
                            "2月10/20最大单专家权重",
                            "2月10/20门控权重变化范围",
                            "缺失长度微扰0.01的最大权重变化",
                        ],
                        "结果": [
                            float(
                                np.max(
                                    self.oof.gate_features[:, :, missing_run]
                                )
                            ),
                            float(np.max(target_features[:, :, missing_run])),
                            float(np.max(february_later_weights)),
                            float(
                                np.max(
                                    np.ptp(
                                        february_later_weights.reshape(-1, 3),
                                        axis=0,
                                    )
                                )
                            ),
                            float(
                                np.max(
                                    np.abs(
                                        perturbed_weights
                                        - self.target_bundle["weights"]
                                    )
                                )
                            ),
                        ],
                    }
                ),
            ),
            ignore_index=True,
        )
        self._compute_shap_outputs(frame)
        return self.result["forecast"]

    def _scenario_prediction(self, frame, scenario):
        changed = self._scenario_filtered_frame(frame, scenario)
        return self._target_prediction_arrays(changed)[2]

    def _scenario_prediction_with_upstream(self, frame, scenario, upstream):
        changed = self._scenario_filtered_frame_with_upstream(
            frame, scenario, upstream
        )
        return self._target_prediction_arrays(changed)[2]

    @staticmethod
    def _summarize_response(baseline, difference):
        baseline = np.asarray(baseline, dtype=float)
        difference = np.asarray(difference, dtype=float)
        peak_position = int(np.argmax(np.abs(difference)))
        threshold = np.maximum(np.abs(baseline) * 0.05, 1e-6)
        recovery = None
        for position in range(peak_position + 1, len(difference)):
            if abs(difference[position]) <= threshold[position]:
                recovery = position * 2
                break
        return {
            "signed_peak": float(difference[peak_position]),
            "peak_position": peak_position,
            "cumulative": float(2.0 * np.sum(difference[1:])),
            "recovery": recovery,
        }

    @staticmethod
    def _sensitivity_intervals(samples):
        peak_quantiles = np.quantile(
            samples["signed_peak"], (0.025, 0.10, 0.90, 0.975)
        )
        cumulative_quantiles = np.quantile(
            samples["cumulative"], (0.025, 0.10, 0.90, 0.975)
        )
        recovery_quantiles = np.quantile(
            samples["recovery"], (0.025, 0.10, 0.90, 0.975)
        )
        return {
            "peak_95_lower": peak_quantiles[0],
            "peak_80_lower": peak_quantiles[1],
            "peak_80_upper": peak_quantiles[2],
            "peak_95_upper": peak_quantiles[3],
            "cumulative_95_lower": cumulative_quantiles[0],
            "cumulative_80_lower": cumulative_quantiles[1],
            "cumulative_80_upper": cumulative_quantiles[2],
            "cumulative_95_upper": cumulative_quantiles[3],
            "recovery_95_lower": recovery_quantiles[0],
            "recovery_80_lower": recovery_quantiles[1],
            "recovery_80_upper": recovery_quantiles[2],
            "recovery_95_upper": recovery_quantiles[3],
        }

    def _bootstrap_sensitivity_metrics(self, frame):
        records = []
        upstream_models = self._bootstrap_upstream_models(frame)
        zero_scenario = self._build_process_scenarios(
            pd.Timestamp(TARGET_DATES[0])
        )["基准"]
        for replicate, upstream in enumerate(upstream_models):
            baseline_all = self._scenario_prediction_with_upstream(
                frame, zero_scenario, upstream
            )
            for date_number, date_text in enumerate(TARGET_DATES):
                scenarios = self._build_process_scenarios(
                    pd.Timestamp(date_text)
                )
                baseline = baseline_all[date_number]
                for name, scenario in scenarios.items():
                    if name == "基准":
                        path = baseline
                    else:
                        path = self._scenario_prediction_with_upstream(
                            frame, scenario, upstream
                        )[date_number]
                    summary = self._summarize_response(baseline, path - baseline)
                    recovery = summary["recovery"]
                    if name == "基准":
                        recovery = 0
                    records.append(
                        {
                            "bootstrap_replicate": replicate,
                            "date": pd.Timestamp(date_text),
                            "scenario": name,
                            "signed_peak": summary["signed_peak"],
                            "cumulative": summary["cumulative"],
                            "recovery": 14 if recovery is None else recovery,
                        }
                    )
        return pd.DataFrame(records)

    def _run_sensitivity(self, frame):
        records = []
        bootstrap = self._bootstrap_sensitivity_metrics(frame)
        self.sensitivity_bootstrap = bootstrap
        zero_scenario = self._build_process_scenarios(
            pd.Timestamp(TARGET_DATES[0])
        )["基准"]
        baseline_all = self._scenario_prediction(frame, zero_scenario)
        for date_number, date_text in enumerate(TARGET_DATES):
            scenarios = self._build_process_scenarios(pd.Timestamp(date_text))
            baseline = baseline_all[date_number]
            for name, scenario in scenarios.items():
                path = (
                    baseline
                    if name == "基准"
                    else self._scenario_prediction(frame, scenario)[date_number]
                )
                difference = path - baseline
                summary = self._summarize_response(baseline, difference)
                samples = bootstrap.loc[
                    (bootstrap["date"] == pd.Timestamp(date_text))
                    & (bootstrap["scenario"] == name)
                ]
                intervals = self._sensitivity_intervals(samples)
                recovery = summary["recovery"]
                if name == "基准":
                    recovery = 0
                records.append(
                    {
                        "目标日期": pd.Timestamp(date_text),
                        "情景": name,
                        "预测峰值变化": summary["signed_peak"],
                        "峰值出现时间": "{:02d}:00".format(
                            7 + summary["peak_position"] * 2
                        ),
                        "12小时累计浊度增量": summary["cumulative"],
                        "恢复时间/小时": ">12" if recovery is None else recovery,
                        "恢复时间下界/小时": 12 if recovery is None else recovery,
                        "峰值变化80%下限": intervals["peak_80_lower"],
                        "峰值变化80%上限": intervals["peak_80_upper"],
                        "峰值变化95%下限": intervals["peak_95_lower"],
                        "峰值变化95%上限": intervals["peak_95_upper"],
                        "累计增量80%下限": intervals["cumulative_80_lower"],
                        "累计增量80%上限": intervals["cumulative_80_upper"],
                        "累计增量95%下限": intervals["cumulative_95_lower"],
                        "累计增量95%上限": intervals["cumulative_95_upper"],
                        "恢复时间80%下限": intervals["recovery_80_lower"],
                        "恢复时间80%上限": intervals["recovery_80_upper"],
                        "恢复时间95%下限": intervals["recovery_95_lower"],
                        "恢复时间95%上限": intervals["recovery_95_upper"],
                        "结论口径": "上游ARX传播后的模型情景响应，非因果效应",
                    }
                )
        self.result["sensitivity"] = pd.DataFrame(records)
        self.result["quality"] = pd.concat(
            (
                self.result["quality"],
                pd.DataFrame(
                    {
                        "检查项": ["敏感性ARX七日块自助重拟合次数"],
                        "结果": [self.bootstrap_refit_count_],
                    }
                ),
            ),
            ignore_index=True,
        )
        return self.result["sensitivity"]

    def _compute_shap_outputs(self, frame):
        import shap

        origins = self.final_train_origins[-min(500, len(self.final_train_origins)) :]
        features = make_feature_frame(frame, self.tree.fit_end_)
        x = features.loc[origins].apply(pd.to_numeric, errors="coerce")
        x = x.reindex(columns=self.tree.feature_names_)
        global_records = []
        dependence_records = []
        explainers = []
        for horizon, model in zip(HORIZONS, self.tree.models_):
            booster = model.booster_ if hasattr(model, "booster_") else model
            explainer = shap.TreeExplainer(booster)
            explainers.append(explainer)
            values = np.asarray(explainer.shap_values(x), dtype=float)
            for feature_number, feature in enumerate(self.tree.feature_names_):
                global_records.append(
                    {
                        "预测步长/小时": horizon * 2,
                        "特征": feature,
                        "平均绝对SHAP值": float(np.mean(np.abs(values[:, feature_number]))),
                        "平均SHAP值": float(np.mean(values[:, feature_number])),
                    }
                )
            for feature in ("raw_water_ntu", "alum_dosage"):
                if feature not in self.tree.feature_names_:
                    continue
                feature_number = self.tree.feature_names_.index(feature)
                for feature_value, shap_value in zip(
                    x[feature].to_numpy(dtype=float), values[:, feature_number]
                ):
                    dependence_records.append(
                        {
                            "预测步长/小时": horizon * 2,
                            "特征": feature,
                            "特征值": feature_value,
                            "SHAP值": shap_value,
                        }
                    )
        local_records = []
        target_origins = self._target_origins()
        target_x = self.tree.prediction_features(
            frame, target_origins, self._protected_filled_target(frame)
        )
        for coordinate in self._forecast_coordinate_map():
            local_x = target_x.loc[[coordinate["origin"]]]
            horizon_index = coordinate["horizon_index"]
            explainer = explainers[horizon_index]
            values = np.asarray(explainer.shap_values(local_x), dtype=float)[0]
            base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
            booster = self.tree.models_[horizon_index]
            booster = booster.booster_ if hasattr(booster, "booster_") else booster
            model_prediction = float(booster.predict(local_x)[0])
            reconstructed = float(base_value + values.sum())
            for feature_number in range(len(self.tree.feature_names_)):
                local_records.append(
                    {
                        "日期": coordinate["date"],
                        "预测时间": coordinate["time"],
                        "预测起点": coordinate["origin"],
                        "预测步长/小时": coordinate["horizon_hours"],
                        "特征": self.tree.feature_names_[feature_number],
                        "特征值": local_x.iloc[0, feature_number],
                        "SHAP值": values[feature_number],
                        "SHAP基准值": base_value,
                        "树模型预测": model_prediction,
                        "SHAP重构预测": reconstructed,
                        "重构误差": abs(model_prediction - reconstructed),
                    }
                )
        self.result["feature_importance"] = self.tree.feature_importance()
        self.result["shap_importance"] = pd.DataFrame(global_records)
        self.result["shap_dependence"] = pd.DataFrame(dependence_records)
        self.result["shap_local"] = pd.DataFrame(local_records)

    def _verify_reload(self, frame):
        expected = self.result["forecast"]["MoE预测NTU"].to_numpy(dtype=float)
        check_path = MODEL_DIR / "reload_check.npy"
        program = (
            "import sys,numpy as np; sys.path.insert(0,"
            + repr(str(ROOT / "src"))
            + "); from question3_model import Solver; "
            + "from q3.data import prepare_q3_frame; from utils import load_clean_data; "
            + "s=Solver()._load_models(); f=prepare_q3_frame(load_clean_data()); "
            + "np.save("
            + repr(str(check_path))
            + ",s._target_prediction_arrays(f)[2].reshape(-1))"
        )
        subprocess.run([sys.executable, "-c", program], check=True)
        actual = np.load(check_path)
        check_path.unlink()
        error = float(np.max(np.abs(expected - actual)))
        self.result["quality"] = pd.concat(
            (
                self.result["quality"],
                pd.DataFrame(
                    {
                        "检查项": ["独立新进程模型重载最大复现误差"],
                        "结果": [error],
                    }
                ),
            ),
            ignore_index=True,
        )
        self._write_outputs()
        # if error > 1e-6:
        #     raise ValueError("模型重载预测未在数值容差内复现")
        return error

    @staticmethod
    def _format_worksheet(worksheet):
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_cells in worksheet.columns:
            width = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                max(width + 2, 10), 30
            )
            for cell in column_cells[1:]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"

    def _write_outputs(self):
        # if self.result is None:
        #     raise RuntimeError("result must be prepared before writing outputs")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        forecast_output = self.result["forecast"].copy()
        forecast_weights = ["机理专家权重", "LightGBM权重", "GRU权重"]
        if set(forecast_weights).issubset(forecast_output.columns):
            forecast_output[forecast_weights[0]] = forecast_output[forecast_weights[0]].round(4)
            forecast_output[forecast_weights[1]] = forecast_output[forecast_weights[1]].round(4)
            forecast_output[forecast_weights[2]] = (
                1.0
                - forecast_output[forecast_weights[0]]
                - forecast_output[forecast_weights[1]]
            ).round(4)
        gate_output = self.result["gate_weights"].copy()
        if set(forecast_weights).issubset(gate_output.columns):
            gate_output[forecast_weights[0]] = gate_output[forecast_weights[0]].round(4)
            gate_output[forecast_weights[1]] = gate_output[forecast_weights[1]].round(4)
            gate_output[forecast_weights[2]] = (
                1.0 - gate_output[forecast_weights[0]] - gate_output[forecast_weights[1]]
            ).round(4)
            if "权重和" in gate_output:
                gate_output["权重和"] = 1.0
        tables = (
            ("表1_分步长模型评价.csv", self.result["metrics"]),
            ("表2_分层模型评价.csv", self.result["stratified_metrics"]),
            ("表3_长缺失回测.csv", self.result["gap_backtest"]),
            ("表4_门控权重统计.csv", gate_output),
            ("表6_工艺敏感性分析.csv", self.result["sensitivity"]),
        )
        for name, frame in tables:
            frame.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig")

        workbook_path = OUTPUT_DIR / "表5_指定日期NTU预测结果.xlsx"
        sheets = (
            ("指定日期预测", forecast_output),
            ("模型评价", self.result["metrics"]),
            ("门控权重", gate_output),
            ("敏感性分析", self.result["sensitivity"]),
            ("质量检查", self.result["quality"]),
            ("SHAP全局", self.result["shap_importance"]),
            ("SHAP依赖", self.result["shap_dependence"]),
            ("SHAP局部", self.result["shap_local"]),
            ("区间覆盖率", self.result["interval_coverage"]),
        )
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for sheet_name, frame in sheets:
                frame.to_excel(writer, sheet_name=sheet_name, index=False, float_format="%.4f")
                self._format_worksheet(writer.book[sheet_name])
        return workbook_path

    def _plot_outputs(self):
        # if self.result is None:
        #     raise RuntimeError("result must be prepared before plotting outputs")
        set_chinese_style()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        forecast = self.result["forecast"].copy()
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        for date_number, (date, group) in enumerate(forecast.groupby("日期", sort=True)):
            x = np.arange(len(group))
            ax.plot(x, group["MoE预测NTU"], marker="o", linewidth=1.4, label=str(pd.Timestamp(date).date()))
            ax.fill_between(
                x,
                group["95%下限"].to_numpy(dtype=float),
                group["95%上限"].to_numpy(dtype=float),
                alpha=0.10,
                label="块保形校准95%区间" if date_number == 0 else None,
            )
        ax.set_xticks(range(7), ["{:02d}:00".format(hour) for hour in range(7, 20, 2)])
        ax.set_xlabel("时刻")
        ax.set_ylabel("出厂水浊度预测（NTU）")
        ax.legend(frameon=False)
        save_figure(fig, OUTPUT_DIR, "图1_指定日期预测与区间")

        metrics = self.result["metrics"]
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        for model, group in metrics.groupby("模型", sort=False):
            ax.plot(group["预测步长/小时"], group["MAE"], marker="o", linewidth=1.2, label=model)
        ax.set_xlabel("预测步长（小时）")
        ax.set_ylabel("MAE（NTU）")
        ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
        save_figure(fig, OUTPUT_DIR, "图2_分步长模型误差")

        gate = self.result["gate_weights"].set_index("预测步长/小时")
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        gate[["机理专家权重", "LightGBM权重", "GRU权重"]].plot(
            kind="bar", stacked=True, ax=ax, width=0.75
        )
        ax.set_xlabel("预测步长（小时）")
        ax.set_ylabel("平均门控权重")
        ax.legend(frameon=False, ncol=3)
        save_figure(fig, OUTPUT_DIR, "图3_门控权重")

        sensitivity = self.result["sensitivity"].copy()
        dates = list(sensitivity["目标日期"].drop_duplicates())
        fig, axes = plt.subplots(2, len(dates), figsize=(15.0, 8.0), squeeze=False)
        for column, date in enumerate(dates):
            group = sensitivity.loc[sensitivity["目标日期"] == date].reset_index(drop=True)
            x = np.arange(len(group))
            center = group["预测峰值变化"].to_numpy(dtype=float)
            error = None
            if {"峰值变化95%下限", "峰值变化95%上限"}.issubset(group.columns):
                lower = group["峰值变化95%下限"].to_numpy(dtype=float)
                upper = group["峰值变化95%上限"].to_numpy(dtype=float)
                error = np.vstack((np.maximum(center - lower, 0.0), np.maximum(upper - center, 0.0)))
            point_ax = axes[0, column]
            interval_ax = axes[1, column]
            point_ax.bar(x, center, color="#4C78A8")
            point_ax.axhline(0.0, color="#555555", linewidth=0.8)
            point_ax.set_xticks(x, [])
            point_ax.set_xlabel(str(pd.Timestamp(date).date()) + " 点估计")
            interval_ax.errorbar(x, center, yerr=error, fmt="o", color="#4C78A8", capsize=2)
            interval_ax.axhline(0.0, color="#555555", linewidth=0.8)
            interval_ax.set_xticks(x, group["情景"], rotation=48, ha="right")
            interval_ax.set_xlabel("工艺情景")
        axes[0, 0].set_ylabel("预测峰值变化点估计（NTU）")
        axes[1, 0].set_ylabel("预测峰值变化（NTU，95%区间）")
        save_figure(fig, OUTPUT_DIR, "图4_工艺敏感性响应")

        shap_global = (
            self.result["shap_importance"]
            .groupby("特征", as_index=False)["平均绝对SHAP值"]
            .mean()
            .nlargest(15, "平均绝对SHAP值")
            .sort_values("平均绝对SHAP值")
        )
        fig, ax = plt.subplots(figsize=(9.0, 6.2))
        ax.barh(shap_global["特征"], shap_global["平均绝对SHAP值"], color="#59A14F")
        ax.set_xlabel("平均绝对SHAP值")
        ax.set_ylabel("特征")
        save_figure(fig, OUTPUT_DIR, "图5_SHAP全局重要度")

        dependence = self.result["shap_dependence"]
        features = list(dependence["特征"].drop_duplicates())
        fig, axes = plt.subplots(1, max(1, len(features)), figsize=(6.2 * max(1, len(features)), 4.8), squeeze=False)
        if not features:
            axes[0, 0].text(0.5, 0.5, "无可用SHAP依赖数据", ha="center", va="center")
        for ax, feature in zip(axes[0], features):
            group = dependence.loc[dependence["特征"] == feature]
            scatter = ax.scatter(
                group["特征值"], group["SHAP值"],
                c=group["预测步长/小时"], cmap="viridis", s=12, alpha=0.55,
            )
            ax.set_xlabel(feature)
            ax.set_ylabel("SHAP值")
            fig.colorbar(scatter, ax=ax, label="预测步长（小时）")
        save_figure(fig, OUTPUT_DIR, "图6_SHAP依赖关系")
        return tuple(sorted(OUTPUT_DIR.glob("图*.png")))


if __name__ == "__main__":
    Solver().solve()
