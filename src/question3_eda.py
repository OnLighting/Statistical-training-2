from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import label, load_clean_data, save_figure, set_chinese_style, zscore_frame
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "03_问题3"
DYNAMIC_INPUTS = [
    "raw_water_ntu",
    "filtered_ntu",
    "clear_well_level",
    "alum_dosage",
    "alum_feed_rate",
    "raw_water_flow",
    "treated_water_flow",
]
def plot_process_chain(data):
    columns = [column for column in DYNAMIC_INPUTS + ["treated_ntu"] if data[column].notna().sum() >= 100]
    daily = data.set_index("timestamp")[columns].resample("D").median()
    standardized = zscore_frame(daily, columns)
    fig, ax = plt.subplots(figsize=(13, 5.3))
    for column, color in zip(columns, sns.color_palette("tab10", len(columns))):
        ax.plot(standardized.index, standardized[column], linewidth=0.85, color=color, label=label(column))
    ax.set_xlabel("日期")
    ax.set_ylabel("标准化日中位数")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save_figure(fig, OUTPUT_DIR, "图1_工艺链主要变量动态变化")
def process_lag_correlations(data, max_lag=18):
    columns = [column for column in DYNAMIC_INPUTS if data[column].notna().sum() >= 100]
    frame = data.set_index("timestamp")[columns + ["treated_ntu"]].resample("2h").median()
    differenced = frame.diff(12)
    rows = []
    for column in columns:
        for lag in range(max_lag + 1):
            pair = pd.concat(
                [differenced[column].shift(lag), differenced["treated_ntu"]],
                axis=1,
                keys=["input", "target"],
            ).dropna()
            rows.append(
                {
                    "变量": column,
                    "领先小时": lag * 2,
                    "相关系数": pair["input"].corr(pair["target"]) if len(pair) >= 50 else np.nan,
                    "有效样本数": len(pair),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "表1_工艺链候选时滞.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for column, group in result.groupby("变量"):
        ax.plot(group["领先小时"], group["相关系数"], marker="o", markersize=3, linewidth=1.1, label=label(column))
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("输入变量领先出厂水浊度的小时数")
    ax.set_ylabel("24小时差分后的相关系数")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, OUTPUT_DIR, "图2_工艺链候选时滞")


def plot_multi_horizon_change(data):
    series = data.set_index("timestamp")["treated_ntu"].resample("2h").median()
    rows = []
    for hours in [2, 6, 12]:
        steps = hours // 2
        changes = series.shift(-steps) - series
        for value in changes.dropna():
            rows.append({"预测步长": f"{hours}小时", "浊度变化": value})
    frame = pd.DataFrame(rows)

    summary = frame.groupby("预测步长")["浊度变化"].agg(["count", "mean", "std", "median"])
    summary["平均绝对变化"] = frame.assign(绝对变化=frame["浊度变化"].abs()).groupby("预测步长")["绝对变化"].mean()
    summary.to_csv(OUTPUT_DIR / "表2_不同预测步长变化特征.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.violinplot(data=frame, x="预测步长", y="浊度变化", inner="quart", cut=0, color="#9ecae1", ax=ax)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("预测步长")
    ax.set_ylabel("未来浊度减当前浊度（NTU）")
    save_figure(fig, OUTPUT_DIR, "图3_不同预测步长浊度变化")


def plot_time_patterns(data):
    valid = data.loc[data["treated_ntu"].notna()].copy()
    hour_month = valid.pivot_table(index="month", columns="hour", values="treated_ntu", aggfunc="median")
    weekday_hour = valid.pivot_table(index="weekday", columns="hour", values="treated_ntu", aggfunc="median").reindex(range(7))
    weekday_hour.index = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(hour_month, cmap="YlOrRd", ax=axes[0], cbar_kws={"label": "中位数（NTU）"})
    axes[0].set_xlabel("小时")
    axes[0].set_ylabel("月份")
    sns.heatmap(weekday_hour, cmap="YlGnBu", ax=axes[1], cbar_kws={"label": "中位数（NTU）"})
    axes[1].set_xlabel("小时")
    axes[1].set_ylabel("星期")
    save_figure(fig, OUTPUT_DIR, "图4_月份星期与小时模式")


def plot_2026_target_days(data):
    dates = [pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10"), pd.Timestamp("2026-02-20")]
    variables = ["raw_water_ntu", "filtered_ntu", "alum_dosage", "raw_water_flow"]
    standardized = zscore_frame(data, variables)
    plot_data = data[["timestamp", "hour"]].join(standardized)
    plot_data = plot_data[plot_data["timestamp"].dt.normalize().isin(dates)]
    plot_data = plot_data[plot_data["hour"].between(7, 19)]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True, sharey=True)
    for ax, date in zip(axes, dates):
        day = plot_data[plot_data["timestamp"].dt.normalize().eq(date)]
        for column in variables:
            ax.plot(day["hour"], day[column], marker="o", linewidth=1.2, label=label(column))
        ax.axhline(0, color="#666666", linewidth=0.7)
        ax.set_ylabel(f"{date:%m月%d日}\n标准化值")
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xticks(range(7, 20, 2))
    axes[-1].set_xlabel("时刻")
    save_figure(fig, OUTPUT_DIR, "图5_指定日期可用输入变化")


def main():
    set_chinese_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_clean_data()
    plot_process_chain(data)
    process_lag_correlations(data)
    plot_multi_horizon_change(data)
    plot_time_patterns(data)
    plot_2026_target_days(data)
    print(f"问题3探索性分析已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
