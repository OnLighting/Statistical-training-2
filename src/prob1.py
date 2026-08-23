import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from plot1 import CANDIDATES
from utils import label, load_clean_data, save_figure, set_chinese_style

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "01_问题1"
TEST_DAYS = 14
FORECAST_DATES = pd.to_datetime(["2026-02-01", "2026-02-10", "2026-02-20"])
TRAINING_WINDOW_DAYS = 50
FINAL_MODEL = "稳健周期时序回归"

MODEL_FEATURES = [
    "river_level",
    "raw_water_flow",
    "raw_water_ntu",
    "raw_water_color",
    "raw_water_ph",
    "filtered_ntu",
    "clear_well_level",
    "treated_ph",
    "treated_color",
    "chlorine_residual",
    "alum_dosage",
    "treated_water_flow",
    "raw_water_pump_count",
    "treated_water_pump_count",
    "hour_sin",
    "hour_cos",
    "week_sin",
    "week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "time_trend",
]
TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "week_sin",
    "week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "time_trend",
]
DYNAMIC_LAGS = (1, 2, 6, 12, 84)
ROLLING_WINDOWS = (12, 84)
def chronological_split_by_day(frame, test_days=TEST_DAYS):
    if test_days < 1:
        raise ValueError("test_days 必须为正整数")
    ordered = frame.sort_values("timestamp").copy()
    operating_days = pd.to_datetime(ordered["operating_date"]).dt.normalize()
    unique_days = operating_days.drop_duplicates().sort_values().to_numpy()
    if len(unique_days) <= test_days:
        raise ValueError("可用运行日不足，无法划分训练集和测试集")
    test_start = pd.Timestamp(unique_days[-test_days])
    is_test = operating_days.ge(test_start)
    return ordered.loc[~is_test].copy(), ordered.loc[is_test].copy()


def add_model_features(data):
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["operating_date"] = pd.to_datetime(frame["operating_date"]).dt.normalize()
    hour = frame["timestamp"].dt.hour + frame["timestamp"].dt.minute / 60
    day_of_year = frame["timestamp"].dt.dayofyear
    elapsed_hours = (frame["timestamp"] - frame["timestamp"].min()).dt.total_seconds() / 3600
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["week_sin"] = np.sin(2 * np.pi * elapsed_hours / 168)
    frame["week_cos"] = np.cos(2 * np.pi * elapsed_hours / 168)
    frame["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    frame["time_trend"] = elapsed_hours / 24
    return frame.sort_values("timestamp").reset_index(drop=True)


def quantitative_feature_screening(frame, feature_columns, top_n=8, random_state=2026):
    columns = [column for column in feature_columns if frame[column].notna().sum() >= 30]
    if not columns:
        raise ValueError("没有足够数据用于因素筛选")
    x = frame[columns]
    y = frame["treated_ntu"].astype(float)
    transformed_x = SimpleImputer(strategy="median").fit_transform(x)
    standardized_x = StandardScaler().fit_transform(transformed_x)
    split_count = min(5, max(2, len(frame) // 48))
    lasso = LassoCV(
        alphas=np.logspace(-5, 0, 60),
        cv=TimeSeriesSplit(n_splits=split_count),
        max_iter=30000,
        random_state=random_state,
    ).fit(standardized_x, y)
    forest = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=4,
        max_features=0.75,
        random_state=random_state,
        n_jobs=-1,
    ).fit(transformed_x, y)
    differenced = frame[columns + ["treated_ntu"]].diff(12)
    spearman = differenced[columns].corrwith(
        differenced["treated_ntu"], method="spearman"
    ).fillna(0)
    result = pd.DataFrame({
        "变量": columns,
        "差分Spearman系数": spearman.reindex(columns).to_numpy(),
        "差分Spearman绝对值": spearman.abs().reindex(columns).to_numpy(),
        "LASSO标准化系数": lasso.coef_,
        "LASSO绝对系数": np.abs(lasso.coef_),
        "随机森林重要性": forest.feature_importances_,
    })
    for source, rank_name in [
        ("差分Spearman绝对值", "Spearman排名"),
        ("LASSO绝对系数", "LASSO排名"),
        ("随机森林重要性", "随机森林排名"),
    ]:
        result[rank_name] = result[source].rank(method="min", ascending=False)
    result["综合排名得分"] = result[["Spearman排名", "LASSO排名", "随机森林排名"]].mean(axis=1)
    result = result.sort_values(["综合排名得分", "随机森林重要性"], ascending=[True, False]).reset_index(drop=True)
    result["综合排名"] = np.arange(1, len(result) + 1)
    result["是否入选"] = result["综合排名"].le(min(top_n, len(result)))
    return result
def fit_and_predict_models(
    train,
    test,
    feature_columns=None,
    random_state=2026,
):
    # 拟合两个白盒时序模型和一个非线性模型
    if feature_columns is None:
        feature_columns = MODEL_FEATURES
    if train["treated_ntu"].isna().any():
        raise ValueError("训练集的出厂水浊度不能包含缺失值")
    if test.empty:
        raise ValueError("预测集不能为空")
    ordered_train = train.sort_values("timestamp").copy()
    ordered_test = test.sort_values("timestamp").copy()
    x_train = ordered_train[feature_columns]
    x_test = ordered_test[feature_columns]
    y_train = ordered_train["treated_ntu"].astype(float)
    robust_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", HuberRegressor(epsilon=1.5, alpha=0.0001, max_iter=5000)),
    ]).fit(x_train, y_train)
    forest_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        )),
    ]).fit(x_train, y_train)
    sarimax_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    sarimax_x_train = sarimax_transformer.fit_transform(x_train)
    sarimax_x_test = sarimax_transformer.transform(x_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        sarimax_model = SARIMAX(
            y_train.to_numpy(),
            exog=sarimax_x_train,
            order=(2, 0, 1),
            seasonal_order=(1, 0, 0, 12),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=100)

    predictions = {
        "稳健周期时序回归": np.maximum(robust_model.predict(x_test), 0),
        "SARIMAX": np.maximum(np.asarray(sarimax_model.get_forecast(
            steps=len(ordered_test), exog=sarimax_x_test
        ).predicted_mean), 0),
        "随机森林": np.maximum(forest_model.predict(x_test), 0),
    }
    training_fits = {
        "稳健周期时序回归": (
            y_train.to_numpy(), np.maximum(robust_model.predict(x_train), 0)
        ),
        "SARIMAX": (
            y_train.iloc[12:].to_numpy(),
            np.maximum(np.asarray(sarimax_model.fittedvalues)[12:], 0),
        ),
        "随机森林": (
            y_train.to_numpy(), np.maximum(forest_model.predict(x_train), 0)
        ),
    }
    artifacts = {
        "稳健周期时序回归": robust_model,
        "SARIMAX": sarimax_model,
        "SARIMAX特征转换": sarimax_transformer,
        "随机森林": forest_model,
        "特征名称": list(feature_columns),
        "训练拟合": training_fits,
    }
    return predictions, artifacts
def _metric_row(model_name, data_name, actual, predicted, candidate=True):
    model_type = {
        "稳健周期时序回归": "白盒时序模型",
        "SARIMAX": "白盒时序模型",
        "随机森林": "黑盒模型",
        "季节朴素基线": "基线",
    }[model_name]
    return {
        "模型": model_name,
        "模型类型": model_type,
        "数据集": data_name,
        "样本数": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "是否候选模型": candidate,
    }
def evaluate_predictions(evaluation,predictions,artifacts,baseline,evaluation_name="测试集",include_training=True,):
    rows = []
    if include_training:
        for model_name, (actual, fitted) in artifacts["训练拟合"].items():
            rows.append(_metric_row(model_name, "训练集", actual, fitted))
    evaluation_actual = evaluation["treated_ntu"].to_numpy(dtype=float)
    for model_name, predicted in predictions.items():
        rows.append(_metric_row(
            model_name, evaluation_name, evaluation_actual, predicted
        ))
    rows.append(_metric_row(
        "季节朴素基线",
        evaluation_name,
        evaluation_actual,
        baseline,
        candidate=False,
    ))
    metrics = pd.DataFrame(rows)
    sarimax = artifacts["SARIMAX"]
    metrics["AIC"] = np.where(metrics["模型"].eq("SARIMAX"), sarimax.aic, np.nan)
    metrics["BIC"] = np.where(metrics["模型"].eq("SARIMAX"), sarimax.bic, np.nan)
    return metrics.sort_values(["数据集", "RMSE"]).reset_index(drop=True)
def _seasonal_naive(train, test_length, period=12):
    last_cycle = train.sort_values("timestamp")["treated_ntu"].to_numpy(dtype=float)[-period:]
    return np.resize(last_cycle, test_length)
def _restrict_training_window(train, validation_start, window_days):
    start = pd.Timestamp(validation_start) - pd.Timedelta(days=window_days)
    return train.loc[pd.to_datetime(train["operating_date"]).ge(start)].copy()
def plot_model_comparison(test, predictions, baseline):
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(test["timestamp"], test["treated_ntu"], color="black", linewidth=1.8,
            label="Ground truth", zorder=5)
    colors = {"稳健周期时序回归": "tab:blue", "SARIMAX": "tab:orange", "随机森林": "tab:green"}
    for model_name, predicted in predictions.items():
        ax.plot(test["timestamp"], predicted, linewidth=1.05, alpha=0.9,color=colors[model_name], label=model_name)
    ax.plot(test["timestamp"], baseline, color="gray", linestyle="--", linewidth=0.9,alpha=0.8, label="季节朴素基线")
    ax.set_xlabel("日期")
    ax.set_ylabel("出厂水浊度/NTU")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, ncol=5,loc="upper center")
    save_figure(fig, OUTPUT_DIR, "图7_测试集真实值与不同模型预测值")
def _feature_label(feature):
    if feature.startswith("target_lag_"):
        lag = int(feature.rsplit("_", 1)[1])
        return f"出厂水浊度滞后{lag * 2}小时"
    if feature.startswith("target_roll_mean_"):
        window = int(feature.rsplit("_", 1)[1])
        return f"出厂水浊度过去{window * 2}小时均值"
    time_labels = {
        "hour_sin": "日周期正弦项", "hour_cos": "日周期余弦项",
        "week_sin": "周周期正弦项", "week_cos": "周周期余弦项",
        "day_of_year_sin": "年周期正弦项", "day_of_year_cos": "年周期余弦项",
        "time_trend": "长期时间趋势",
    }
    return time_labels.get(feature, label(feature))
def model_interpretation_tables(artifacts):
    feature_names = artifacts["特征名称"]
    robust = artifacts["稳健周期时序回归"]
    scaler = robust.named_steps["scaler"]
    model = robust.named_steps["model"]
    standardized_coefficients = np.asarray(model.coef_, dtype=float)
    original_coefficients = standardized_coefficients / scaler.scale_
    original_intercept = float(
        model.intercept_
        - np.sum(standardized_coefficients * scaler.mean_ / scaler.scale_)
    )
    robust_coefficients = pd.DataFrame({
        "变量名": ["截距"] + [_feature_label(column) for column in feature_names],
        "均值": np.r_[0.0, scaler.mean_],
        "标准差": np.r_[1.0, scaler.scale_],
        "标准化系数": np.r_[float(model.intercept_), standardized_coefficients],
        "原始尺度系数": np.r_[original_intercept, original_coefficients],
    })
    robust_coefficients["影响方向"] = np.where(
        robust_coefficients["原始尺度系数"].ge(0), "正向", "负向"
    )
    forest = artifacts["随机森林"].named_steps["model"]
    forest_importance = pd.DataFrame({
        "变量": [_feature_label(column) for column in feature_names],
        "随机森林重要性": forest.feature_importances_,
    }).sort_values("随机森林重要性", ascending=False)
    sarimax = artifacts["SARIMAX"]
    sarimax_parameters = pd.DataFrame({"参数": sarimax.param_names, "估计值": np.asarray(sarimax.params)})
    return robust_coefficients, forest_importance, sarimax_parameters
def build_forecast_output(future, selected, predictions):
    selected_positions = future.index.get_indexer(selected.index)
    output = selected[["operating_date", "timestamp"]].copy()
    output["日期"] = output["operating_date"].dt.strftime("%Y-%m-%d")
    output["时间"] = output["timestamp"].dt.strftime("%H:%M")
    output["预测NTU"] = np.asarray(predictions[FINAL_MODEL])[selected_positions]
    return output[["日期", "时间", "预测NTU"]]
def run_modeling(data):
    frame = add_model_features(data)
    forecast_start = FORECAST_DATES.min()
    forecast_end = FORECAST_DATES.max()
    historical = frame.loc[
        frame["operating_date"].lt(forecast_start) & frame["treated_ntu"].notna()
    ].copy()
    train, test = chronological_split_by_day(historical, test_days=TEST_DAYS)
    available_candidates = [column for column in CANDIDATES if train[column].notna().sum() >= 30]
    model_train = _restrict_training_window(train, test["operating_date"].min(), TRAINING_WINDOW_DAYS)
    screening = quantitative_feature_screening(model_train, available_candidates, top_n=12)
    selected_process = screening.loc[screening["是否入选"], "变量"].tolist()
    selected_features = selected_process + TIME_FEATURES
    test_predictions, test_artifacts = fit_and_predict_models(model_train, test, selected_features)
    baseline = _seasonal_naive(model_train, len(test))
    metrics = evaluate_predictions(test, test_predictions, test_artifacts, baseline)
    plot_model_comparison(test, test_predictions, baseline)

    test_output = test[["operating_date", "timestamp", "treated_ntu"]].copy()
    test_output = test_output.rename(columns={"treated_ntu": "Ground truth"})
    for model_name, predicted in test_predictions.items():
        test_output[model_name] = predicted
    test_output["季节朴素基线"] = baseline

    future = frame.loc[frame["operating_date"].between(forecast_start, forecast_end)].copy()
    final_train = _restrict_training_window(historical, forecast_start, TRAINING_WINDOW_DAYS)
    final_screening = quantitative_feature_screening(final_train, available_candidates, top_n=12)
    final_process = final_screening.loc[final_screening["是否入选"], "变量"].tolist()
    final_features = final_process + TIME_FEATURES
    future_predictions, final_artifacts = fit_and_predict_models(final_train, future, final_features)
    selected = future[future["operating_date"].isin(FORECAST_DATES)].copy()
    forecast_output = build_forecast_output(future, selected, future_predictions)
    robust_coefficients, _, _ = model_interpretation_tables(final_artifacts)

    final_screening.to_csv(OUTPUT_DIR / "表2_定量因素筛选.csv", index=False, encoding="utf-8")
    metrics.to_csv(OUTPUT_DIR / "表3_模型评价.csv", index=False, encoding="utf-8")
    forecast_output.to_excel(OUTPUT_DIR / "表4_预测结果.xlsx", index=False)
    robust_coefficients.to_csv(
        OUTPUT_DIR / "表5_稳健周期回归公式系数.csv",
        index=False,
        encoding="utf-8",
    )
    return metrics, forecast_output
def main():
    set_chinese_style()
    data = load_clean_data()
    run_modeling(data)
    print(f"问题1建模结果已保存至：{OUTPUT_DIR}")
if __name__ == "__main__":
    main()
