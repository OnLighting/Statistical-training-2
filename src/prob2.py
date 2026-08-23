from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from utils import label, load_clean_data, save_figure, set_chinese_style
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "02_问题2"
INPUTS = ["raw_water_ntu", "raw_water_ph", "alum_dosage", "raw_water_flow"]
TARGET = "filtered_ntu"
FREQUENCY_HOURS = 2
SEASONAL_PERIOD = 12  # 2小时采样下的24小时周期
PHYSICAL_LAG_BOUNDS = {
    "raw_water_ntu": (1, 2),  # 题面提示约2—4小时
    "alum_dosage": (1, 3),  # 题面提示约2—6小时
    "raw_water_ph": (0, 3),
    "raw_water_flow": (0, 3),
}
def prepare_dynamic_frame(data, frequency="2h", interpolation_limit=2):
    indexed = data.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    frame = indexed[INPUTS + [TARGET]].resample(frequency).median()
    frame = frame.interpolate(method="time", limit=interpolation_limit, limit_area="inside")
    if "is_backwash_event" in indexed:
        frame["is_backwash_event"] = (
            indexed["is_backwash_event"]
            .astype(float)
            .resample(frequency)
            .max()
            .fillna(0.0)
        )
    else:
        frame["is_backwash_event"] = 0.0
    return frame
def identify_input_lags(frame,input_columns=None,max_lag_steps=6,seasonal_period=SEASONAL_PERIOD,min_samples=50,lag_bounds=None):
    input_columns = list(input_columns or INPUTS)
    differenced = frame[input_columns + [TARGET]].diff(seasonal_period)
    ordinary_mask = frame.get("is_backwash_event", pd.Series(0.0, index=frame.index)).eq(0)
    rows = []
    for column in input_columns:
        for lag_steps in range(max_lag_steps + 1):
            pair = pd.concat(
                [differenced[column].shift(lag_steps), differenced[TARGET]],
                axis=1,
                keys=["input", "target"],
            ).loc[ordinary_mask]
            pair = pair.dropna()
            has_variation = (pair["input"].nunique() > 1 and pair["target"].nunique() > 1)
            correlation = (
                pair["input"].corr(pair["target"])
                if len(pair) >= min_samples and has_variation
                else np.nan
            )
            rows.append(
                {
                    "变量": column,
                    "滞后阶数": lag_steps,
                    "滞后小时": lag_steps * FREQUENCY_HOURS,
                    "季节差分后相关系数": correlation,
                    "有效样本数": len(pair),
                }
            )
    result = pd.DataFrame(rows)
    best_rows = []
    bounds = PHYSICAL_LAG_BOUNDS if lag_bounds is None else lag_bounds
    for column, group in result.groupby("变量", sort=False):
        lower, upper = bounds.get(column, (0, max_lag_steps))
        group = group.loc[group["滞后阶数"].between(lower, upper)]
        valid = group.dropna(subset=["季节差分后相关系数"])
        if not valid.empty:
            best_rows.append(valid.loc[valid["季节差分后相关系数"].abs().idxmax()].copy())
    return result, pd.DataFrame(best_rows).reset_index(drop=True)
def lag_correlations(data, max_lag=6):
    frame = prepare_dynamic_frame(data)
    result, best_lags = identify_input_lags(frame, max_lag_steps=max_lag)
    best_lags.to_csv(OUTPUT_DIR / "表2_初步时滞识别结果.csv", index=False, encoding="utf-8-sig")
    return result, best_lags
def build_dynamic_design(frame, input_lags, include_target_lag=False, target_lags=None):
    design = pd.DataFrame(index=frame.index)
    design[TARGET] = frame[TARGET]
    for column, lag_steps in input_lags.items():
        design[f"{column}_lag{int(lag_steps)}"] = frame[column].shift(int(lag_steps))
    if target_lags is None:
        target_lags = [1] if include_target_lag else []
    for lag_steps in target_lags:
        design[f"{TARGET}_lag{int(lag_steps)}"] = frame[TARGET].shift(int(lag_steps))
    design["is_backwash_event"] = frame.get("is_backwash_event", 0.0)
    return design.dropna()
def chronological_split(frame, test_fraction=0.2):
    split_at = int(np.floor(len(frame) * (1 - test_fraction)))
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()
def _fit_standardized_ols(train, feature_columns):
    means = train[feature_columns].mean()
    scales = train[feature_columns].std(ddof=0).replace(0, 1.0)
    x_train = sm.add_constant((train[feature_columns] - means) / scales, has_constant="add")
    model = sm.OLS(train[TARGET], x_train).fit(cov_type="HAC", cov_kwds={"maxlags": SEASONAL_PERIOD})
    return model, means, scales
def _direct_predict(model, frame, feature_columns, means, scales):
    x = sm.add_constant((frame[feature_columns] - means) / scales, has_constant="add")
    return np.asarray(model.predict(x), dtype=float)
def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = actual - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = (1 - float(np.sum(residual**2)) / denominator if denominator > 0 else np.nan)
    return rmse, mae, r_squared
def fit_dynamic_models(frame, input_lags, test_fraction=0.2):
    lag_zero = {column: 0 for column in input_lags}
    designs = {
        "无时滞多元回归": build_dynamic_design(frame, lag_zero),
        "分布滞后模型(DLM)": build_dynamic_design(frame, input_lags),
        "带外生输入自回归(ARX)": build_dynamic_design(
            frame, input_lags, target_lags=range(1, SEASONAL_PERIOD + 1)
        ),
    }
    common_index = next(iter(designs.values())).index
    for design in list(designs.values())[1:]:
        common_index = common_index.intersection(design.index, sort=False)
    designs = {name: design.loc[common_index] for name, design in designs.items()}
    metric_rows, parameter_rows, residual_rows = [], [], []
    prediction_table = pd.DataFrame(index=common_index)
    prediction_table["实测滤后水浊度"] = designs[next(iter(designs))][TARGET]
    artifacts = {}
    for model_name, design in designs.items():
        train, test = chronological_split(design, test_fraction=test_fraction)
        feature_columns = [column for column in design if column != TARGET]
        model, means, scales = _fit_standardized_ols(train, feature_columns)
        train_prediction = _direct_predict(model, train, feature_columns, means, scales)
        test_prediction = _direct_predict(model, test, feature_columns, means, scales)
        prediction_table.loc[test.index, model_name] = test_prediction
        train_rmse, train_mae, train_r2 = _metrics(train[TARGET], train_prediction)
        test_rmse, test_mae, test_r2 = _metrics(test[TARGET], test_prediction)
        metric_rows.append(
            {
                "模型": model_name,
                "训练样本数": len(train),
                "测试样本数": len(test),
                "训练RMSE": train_rmse,
                "训练MAE": train_mae,
                "训练R2": train_r2,
                "测试RMSE": test_rmse,
                "测试MAE": test_mae,
                "测试R2": test_r2,
                "AIC": model.aic,
                "BIC": model.bic,
            }
        )
        confidence = model.conf_int()
        for parameter in model.params.index:
            value = float(model.params[parameter])
            parameter_rows.append(
                {
                    "模型": model_name,
                    "参数": parameter,
                    "标准化系数": value,
                    "HAC标准误": float(model.bse[parameter]),
                    "t统计量": float(model.tvalues[parameter]),
                    "p值": float(model.pvalues[parameter]),
                    "95%置信区间下限": float(confidence.loc[parameter, 0]),
                    "95%置信区间上限": float(confidence.loc[parameter, 1]),
                    "作用方向": "正向" if value >= 0 else "负向",
                }
            )
        residual = np.asarray(model.resid, dtype=float)
        diagnostic_lag = min(SEASONAL_PERIOD, max(1, len(residual) // 5))
        ljung_box = acorr_ljungbox(residual, lags=[diagnostic_lag], return_df=True).iloc[0]
        residual_rows.append(
            {
                "模型": model_name,
                "检验滞后阶数": diagnostic_lag,
                "Ljung-Box统计量": float(ljung_box["lb_stat"]),
                "Ljung-Box_p值": float(ljung_box["lb_pvalue"]),
                "是否可视为白噪声(alpha=0.05)": bool(
                    ljung_box["lb_pvalue"] >= 0.05
                ),
            }
        )
        artifacts[model_name] = {
            "model": model,
            "means": means,
            "scales": scales,
            "features": feature_columns,
            "train_residual": residual,
            "train_index": train.index,
            "test_index": test.index,
        }
    prediction_table = prediction_table.dropna(subset=list(designs))
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(parameter_rows),
        pd.DataFrame(residual_rows),
        prediction_table,
        artifacts,
    )


def seasonal_lag_stability(frame, max_lag_steps=6):
    segments = {
        "旱季(11月至次年4月)": frame.index.month.isin([11, 12, 1, 2, 3, 4]),
        "雨季(5月至10月)": frame.index.month.isin([5, 6, 7, 8, 9, 10]),
    }
    rows = []
    for segment_name, mask in segments.items():
        _, best = identify_input_lags(frame.loc[mask], max_lag_steps=max_lag_steps)
        if best.empty:
            continue
        best.insert(0, "分段", segment_name)
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
def arx_parameter_table(parameters):
    return parameters.loc[
        parameters["模型"].eq("带外生输入自回归(ARX)")
    ].reset_index(drop=True)
def create_dynamic_overview_figure(data):
    columns = ["raw_water_ntu", "raw_water_ph", "raw_water_flow", TARGET]
    daily = data.set_index("timestamp")[columns].resample("D").median()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True)
    colors = sns.color_palette("tab10", len(columns))
    for ax, column, color in zip(axes.ravel(), columns, colors):
        ax.plot(
            daily.index,
            daily[column],
            linewidth=0.9,
            color=color,
        )
        ax.set_xlabel("日期")
        ax.set_ylabel(label(column))
        ax.grid(True, linestyle="--", alpha=0.4)
    return fig
def plot_dynamic_overview(data):
    fig = create_dynamic_overview_figure(data)
    save_figure(fig, OUTPUT_DIR, "图1_输入与滤后水浊度动态变化")
def plot_lag_correlations(result):
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for column, group in result.groupby("变量", sort=False):
        ax.plot(
            group["滞后小时"],
            group["季节差分后相关系数"],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label=label(column),
        )
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("输入变量领先滤后水浊度的小时数")
    ax.set_ylabel("24小时季节差分后的相关系数")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_figure(fig, OUTPUT_DIR, "图2_输入变量候选时滞")
def create_delayed_scatter_figure(frame, best_lags):
    selected = best_lags.sort_values("季节差分后相关系数", key=lambda x: x.abs(), ascending=False).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, selected.iterrows()):
        column, lag_steps = row["变量"], int(row["滞后阶数"])
        pair = pd.concat(
            [frame[column].shift(lag_steps), frame[TARGET]],
            axis=1,
            keys=[column, TARGET],
        ).dropna()
        if len(pair) > 2500:
            pair = pair.sample(2500, random_state=2026)
        sns.regplot(
            data=pair,
            x=column,
            y=TARGET,
            scatter_kws={"s": 9, "alpha": 0.22, "color": "#4d4d4d"},
            line_kws={"color": "#cb181d", "linewidth": 1.5},
            ax=ax,
        )
        ax.set_xlabel(f"{label(column)}（领先{lag_steps * FREQUENCY_HOURS}小时）")
        ax.set_ylabel("滤后水浊度/NTU")
    return fig
def plot_delayed_scatter(frame, best_lags):
    fig = create_delayed_scatter_figure(frame, best_lags)
    save_figure(fig, OUTPUT_DIR, "图3_时滞变量与滤后水浊度关系")
def plot_model_validation(predictions):
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.plot(
        predictions.index,
        predictions["实测滤后水浊度"],
        color="#222222",
        linewidth=1.2,
        label="实测值",
    )
    for column, color in zip(predictions.columns[1:], sns.color_palette("tab10", 3)):
        ax.plot(
            predictions.index,
            predictions[column],
            linewidth=0.9,
            alpha=0.9,
            color=color,
            label=column,
        )
    ax.set_xlabel("测试集时间")
    ax.set_ylabel("滤后水浊度/NTU")
    ax.legend(frameon=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(True, linestyle="--", alpha=0.35)
    save_figure(fig, OUTPUT_DIR, "图4_动态模型测试集拟合对比")
def plot_residual_acf(artifacts):
    fig, axes = plt.subplots(len(artifacts), 1, figsize=(10.5, 8.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (model_name, artifact) in zip(axes, artifacts.items()):
        values = acf(artifact["train_residual"], nlags=36, fft=True)
        ax.stem(
            range(len(values)),
            values,
            basefmt=" ",
            linefmt="#3182bd",
            markerfmt="o",
        )
        bound = 1.96 / np.sqrt(len(artifact["train_residual"]))
        ax.axhspan(-bound, bound, color="#9ecae1", alpha=0.3)
        ax.set_ylabel("ACF")
        ax.set_title(model_name, fontsize=10)
    axes[-1].set_xlabel("残差滞后阶数（每阶2小时）")
    save_figure(fig, OUTPUT_DIR, "图5_模型残差自相关检验")
def plot_backwash_comparison(data):
    frame = data.loc[
        data[TARGET].notna(), [TARGET, "is_backwash_event"]
    ].copy()
    frame["事件状态"] = np.where(frame["is_backwash_event"], "反冲洗相关记录", "普通记录")
    summary = frame.groupby("事件状态")[TARGET].agg(
        ["count", "mean", "median", "std", "max"])
    summary.to_csv(OUTPUT_DIR / "表3_反冲洗事件对比.csv", encoding="utf-8-sig")
def main():
    set_chinese_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_clean_data()
    frame = prepare_dynamic_frame(data)
    plot_dynamic_overview(data)
    result, best_lags = lag_correlations(data)
    plot_lag_correlations(result)
    plot_delayed_scatter(frame, best_lags)
    plot_backwash_comparison(data)
    input_lags = {row["变量"]: int(row["滞后阶数"]) for _, row in best_lags.iterrows()}
    metrics, parameters, residual_tests, predictions, artifacts = fit_dynamic_models(frame, input_lags)
    metrics.to_csv(OUTPUT_DIR / "表4_动态模型评价.csv", index=False, encoding="utf-8-sig")
    arx_parameter_table(parameters).to_csv(OUTPUT_DIR / "表5_动态模型参数估计.csv",index=False,encoding="utf-8-sig")
    residual_tests.to_csv(OUTPUT_DIR / "表6_残差白噪声检验.csv",index=False,encoding="utf-8-sig")
    stability = seasonal_lag_stability(frame)
    stability.to_csv(OUTPUT_DIR / "表7_分季节时滞稳定性.csv",index=False,encoding="utf-8-sig")
    plot_model_validation(predictions)
    plot_residual_acf(artifacts)
    print(f"问题2动态时滞分析已保存至：{OUTPUT_DIR}")
if __name__ == "__main__":
    main()
