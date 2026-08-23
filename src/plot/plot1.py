import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, norm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from utils import label, load_clean_data, regular_series, save_figure, set_chinese_style
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "01_问题1"
CANDIDATES = [
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
    "alum_feed_rate",
    "alum_dosage",
    "treated_water_flow",
    "raw_water_pump_count",
    "treated_water_pump_count",
]
def select_log1p_features(data, candidates=None):
    if candidates is None:
        candidates = CANDIDATES
    columns = [column for column in candidates if column in data.columns]
    skewness = data[columns].skew()
    kurtosis = data[columns].kurt()
    heavy_tail = (skewness.abs() > 1) | (kurtosis.abs() > 3)
    return [column for column in columns if bool(heavy_tail.get(column, False))]
def apply_log1p_features(data, columns):
    transformed = data.copy()
    transformed.loc[:, columns] = transformed[columns].apply(np.log1p)
    return transformed
def plot_treated_ntu_series(data):
    daily = data.set_index("timestamp")["treated_ntu"].resample("D").agg(["mean", "max"])
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(daily.index, daily["mean"], color="blue", linewidth=1.1, label="日均值")
    ax.plot(daily.index, daily["max"], color="red", linewidth=0.8, alpha=0.75, label="日最大值")
    ax.axhline(1, color="black", linestyle="--", linewidth=1, label="1 NTU限值")
    ax.set_xlabel("日期")
    ax.set_ylabel("出厂水浊度")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, ncol=3)
    save_figure(fig, OUTPUT_DIR, "图1_出厂水浊度日序列")
def plot_monthly_and_seasonal_distribution(data):
    frame = data.loc[data["treated_ntu"].notna(), ["timestamp", "treated_ntu"]].copy()
    frame["月份"] = frame["timestamp"].dt.month
    frame["季节分组"] = np.where(frame["月份"].isin([5, 6, 7, 8, 9, 10]), "暂定雨季（5—10月）", "暂定旱季（11—4月）")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.boxplot(data=frame, x="月份", y="treated_ntu", color="blue", showfliers=False, ax=axes[0])
    axes[0].set_xlabel("月份")
    axes[0].set_ylabel("出厂水浊度")
    sns.violinplot(data=frame, x="季节分组", y="treated_ntu", color="green", inner="quart", cut=0, ax=axes[1])
    axes[1].set_xlabel("季节分组")
    axes[1].set_ylabel("出厂水浊度")
    save_figure(fig, OUTPUT_DIR, "图2_月份与季节分组分布")
def plot_distribution_check(data,candidates=None,distribution_filename="图3_关键变量分布检验（3x3）",qq_filename="图4_关键变量QQ图（3x3）",color="steelblue",):
    if candidates is None:
        candidates = CANDIDATES
    columns = [column for column in candidates if data[column].notna().sum() >= 100]
    frame = data[columns + ["treated_ntu"]].dropna()
    correlation = frame.corr(method="spearman")
    ranking = correlation["treated_ntu"].abs().sort_values(ascending=False)
    selected = [column for column in ranking.index if column != "treated_ntu"][:8]
    selected = ["treated_ntu"] + selected
    samples = {}
    rows = []
    for column in selected:
        values = data[column].dropna().to_numpy()
        if len(values) > 5000:
            rng = np.random.default_rng(2026)
            values = rng.choice(values, size=5000, replace=False)
        if len(values) < 3:
            continue
        samples[column] = values
        stat, p_value = shapiro(values)
        skewness = pd.Series(values).skew()
        kurtosis = pd.Series(values).kurt()
        rows.append(
            {
                "字段名": column,
                "Shapiro_p": float(p_value),
                "偏度": float(skewness),
                "峰度": float(kurtosis),
            }
        )
    result = pd.DataFrame(rows)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, column in zip(axes.ravel(), selected):
        values = samples[column]
        sns.histplot(values, kde=True, stat="density", color=color,
                     edgecolor="white", bins=40, ax=ax)
        mu, sigma = np.mean(values), np.std(values, ddof=1)
        x_range = np.linspace(values.min(), values.max(), 200)
        ax.plot(x_range, norm.pdf(x_range, mu, sigma),color="red", linewidth=1.6, label="正态拟合")
        p_value = float(result.loc[result["字段名"] == column, "Shapiro_p"].iloc[0])
        skewness = float(result.loc[result["字段名"] == column, "偏度"].iloc[0])
        kurtosis = float(result.loc[result["字段名"] == column, "峰度"].iloc[0])
        p_label = f"p={p_value:.1e}" if p_value < 1e-3 else f"p={p_value:.3f}"
        ax.set_title(f"{label(column)}\n偏度={skewness:.2f}  峰度={kurtosis:.2f}  Shapiro {p_label}",fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("密度")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes.ravel()[len(selected):]:
        ax.set_visible(False)
    save_figure(fig, OUTPUT_DIR, distribution_filename)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, column in zip(axes.ravel(), selected):
        values = samples[column]
        sorted_values = np.sort(values)
        theoretical = norm.ppf((np.arange(1, len(values) + 1) - 0.5) / len(values))
        mu, sigma = np.mean(values), np.std(values, ddof=1)
        theoretical_scaled = theoretical * sigma + mu
        order = np.argsort(theoretical_scaled)
        ax.scatter(theoretical_scaled[order], sorted_values,s=8, color=color, alpha=0.55, edgecolor="none")
        lo = float(min(theoretical_scaled.min(), sorted_values.min()))
        hi = float(max(theoretical_scaled.max(), sorted_values.max()))
        ref = np.linspace(lo, hi, 200)
        ax.plot(ref, ref, color="red", linewidth=1.4, label="正态参考线")
        p_value = float(result.loc[result["字段名"] == column, "Shapiro_p"].iloc[0])
        p_label = f"p={p_value:.1e}" if p_value < 1e-3 else f"p={p_value:.3f}"
        ax.set_title(f"{label(column)}\nShapiro {p_label}",fontsize=10)
        ax.set_xlabel("正态分位")
        ax.set_ylabel("样本分位")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes.ravel()[len(selected):]:
        ax.set_visible(False)
    save_figure(fig, OUTPUT_DIR, qq_filename)
def plot_correlation_heatmap(data,candidates=None,filename="图8_Spearman相关性热力图",):
    if candidates is None:
        candidates = CANDIDATES
    columns = [column for column in candidates if data[column].notna().sum() >= 100]
    frame = data[columns + ["treated_ntu"]]
    correlation = frame.corr(method="spearman", min_periods=100)
    all_nan_columns = [
        column for column in correlation.columns if correlation[column].isna().all()
    ]
    if all_nan_columns:
        print("以下变量与其它列无共同非缺失时段，已从相关热力图中剔除：",
            [label(column) for column in all_nan_columns])
        correlation = correlation.drop(index=all_nan_columns, columns=all_nan_columns)
    order = correlation["treated_ntu"].abs().sort_values(ascending=False).index
    correlation = correlation.loc[order, order]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=[label(column) for column in correlation.columns],
        yticklabels=[label(column) for column in correlation.index],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, OUTPUT_DIR, filename)
    return [column for column in order if column != "treated_ntu"][:4]
def plot_main_scatter(data, columns):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, column in zip(axes.ravel(), columns):
        frame = data[[column, "treated_ntu"]].dropna()
        if len(frame) > 2500:
            frame = frame.sample(2500, random_state=2026)
        sns.regplot(
            data=frame,
            x=column,
            y="treated_ntu",
            lowess=True,
            scatter_kws={"s": 9, "alpha": 0.22, "color": "gray"},
            line_kws={"color": "red", "linewidth": 1.8},
            ax=ax,
        )
        ax.set_xlabel(label(column))
        ax.set_ylabel("出厂水浊度")
    save_figure(fig, OUTPUT_DIR, "图9_主要变量与出厂水浊度关系")
def plot_periodicity(data):
    series = regular_series(data, "treated_ntu")
    acf_series = series.dropna()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_acf(acf_series, lags=100, zero=False, alpha=0.05, ax=ax)
    for lag in (6, 12, 84):
        ax.axvline(lag, color="red", linestyle="--", linewidth=1)
    ax.set_xticks([6, 12, 84])
    ax.set_xticklabels(["12小时", "24小时", "168小时"],color = 'red')
    ax.set_xlabel("滞后阶数（每阶2小时）")
    ax.set_ylabel("自相关系数")
    save_figure(fig, OUTPUT_DIR, "图10_出厂水浊度自相关")
    complete = series.interpolate(method="time", limit_direction="both")
    decomposition = STL(complete, period=12, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(complete.index, complete, color="black", linewidth=0.65)
    axes[0].set_ylabel("原序列")
    axes[1].plot(complete.index, decomposition.trend, color="blue", linewidth=0.8)
    axes[1].set_ylabel("趋势项")
    axes[2].plot(complete.index, decomposition.seasonal, color="green", linewidth=0.7)
    axes[2].set_ylabel("日周期项")
    axes[3].plot(complete.index, decomposition.resid, color="orange", linewidth=0.55)
    axes[3].set_ylabel("残差项")
    axes[3].set_xlabel("日期")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
    save_figure(fig, OUTPUT_DIR, "图11_出厂水浊度STL分解")
def main():
    set_chinese_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_clean_data()
    plot_treated_ntu_series(data)
    plot_monthly_and_seasonal_distribution(data)
    plot_distribution_check(data)
    log1p_features = select_log1p_features(data)
    log1p_data = apply_log1p_features(data, log1p_features + ["treated_ntu"])
    plot_distribution_check(
        log1p_data,
        candidates=log1p_features,
        distribution_filename="图5_log1p分布检验（3x3）",
        qq_filename="图6_log1pQQ图（3x3）",
        color="seagreen",
    )
    plot_correlation_heatmap(
        log1p_data,
        candidates=log1p_features,
        filename="图7_log1p Spearman热力图",
    )
    main_columns = plot_correlation_heatmap(data)
    plot_main_scatter(data, main_columns)
    plot_periodicity(data)
if __name__ == "__main__":
    main()
