from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import load_clean_data, longest_true_run, save_figure, set_chinese_style
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "04_问题4"
def build_daily_metrics(data):
    frame = data.loc[
        data["timestamp"].between("2026-01-01 07:00", "2026-04-01 05:00"),
        ["timestamp", "operating_date", "treated_ntu", "raw_water_ntu", "treated_water_flow"],
    ].copy()
    frame["operating_date"] = pd.to_datetime(frame["operating_date"])

    rows = []
    for operating_date, group in frame.groupby("operating_date"):
        group = group.sort_values("timestamp")
        values = group["treated_ntu"].dropna()
        exceedance = values.gt(1)
        exceedance_count = int(exceedance.sum()) if values.size else np.nan
        exceedance_hours = exceedance_count * 2 if values.size else np.nan
        longest_hours = longest_true_run(exceedance.tolist()) * 2 if values.size else np.nan
        rows.append(
            {
                "运行日期": operating_date,
                "有效观测数": values.size,
                "数据完整率": values.size / 12,
                "日均浊度": values.mean(),
                "日中位浊度": values.median(),
                "日最大浊度": values.max(),
                "日超标幅度": max(values.max() - 1, 0) if not values.empty else np.nan,
                "超标观测数": exceedance_count,
                "超标小时数": exceedance_hours,
                "最长连续超标小时数": longest_hours,
                "超标观测比例": exceedance.sum() / values.size if values.size else np.nan,
                "原水浊度日最大值": group["raw_water_ntu"].max(),
                "出厂水流量日均值": group["treated_water_flow"].mean(),
            }
        )
    daily = pd.DataFrame(rows).sort_values("运行日期")
    daily.to_csv(OUTPUT_DIR / "表1_2026年逐日风险基础指标.csv", index=False, encoding="utf-8-sig")
    return frame, daily


def plot_daily_metrics(daily):
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8), sharex=True)
    axes[0].plot(daily["运行日期"], daily["日最大浊度"], color="#2166ac", marker="o", markersize=2.8, linewidth=0.9)
    axes[0].axhline(1, color="#cb181d", linestyle="--", linewidth=1)
    axes[0].set_ylabel("日最大浊度（NTU）")
    axes[1].bar(daily["运行日期"], daily["日超标幅度"], color="#fc9272", width=0.85)
    axes[1].set_ylabel("超标幅度（NTU）")
    axes[2].bar(daily["运行日期"], daily["最长连续超标小时数"], color="#74c476", width=0.85)
    axes[2].set_ylabel("最长持续时间（小时）")
    axes[2].set_xlabel("运行日期")
    save_figure(fig, OUTPUT_DIR, "图1_超标幅度与持续时间")


def plot_monthly_exceedance(daily):
    frame = daily.copy()
    frame["月份"] = frame["运行日期"].dt.month.astype(str) + "月"
    monthly = frame.groupby("月份").agg(
        运行天数=("运行日期", "count"),
        有效运行天数=("有效观测数", lambda values: int((values > 0).sum())),
        有超标天数=("超标观测数", lambda values: int((values.dropna() > 0).sum())),
        平均日最大浊度=("日最大浊度", "mean"),
        平均超标小时数=("超标小时数", "mean"),
    )
    monthly["有超标天数比例"] = monthly["有超标天数"] / monthly["有效运行天数"].replace(0, np.nan)
    monthly.to_csv(OUTPUT_DIR / "表2_月度超标概况.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    sns.barplot(data=monthly.reset_index(), x="月份", y="有超标天数比例", color="#9ecae1", ax=axes[0])
    axes[0].set_xlabel("月份")
    axes[0].set_ylabel("有超标天数比例")
    axes[0].set_ylim(0, 1)
    sns.barplot(data=monthly.reset_index(), x="月份", y="平均超标小时数", color="#fdae6b", ax=axes[1])
    axes[1].set_xlabel("月份")
    axes[1].set_ylabel("平均超标小时数")
    save_figure(fig, OUTPUT_DIR, "图2_月度超标概况")


def threshold_sensitivity(frame):
    rows = []
    for threshold in [0.8, 1.0, 1.2]:
        for operating_date, group in frame.groupby("operating_date"):
            values = group["treated_ntu"].dropna()
            rows.append(
                {
                    "阈值": threshold,
                    "运行日期": operating_date,
                    "存在超标": bool((values > threshold).any()) if len(values) else np.nan,
                    "超标比例": (values > threshold).mean() if len(values) else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    summary = result.groupby("阈值").agg(
        有超标天数=("存在超标", lambda values: values.sum(min_count=1)),
        有效运行天数=("存在超标", "count"),
        平均超标观测比例=("超标比例", "mean"),
    )
    summary["有超标天数比例"] = summary["有超标天数"] / summary["有效运行天数"].replace(0, np.nan)
    summary.to_csv(OUTPUT_DIR / "表3_阈值敏感性.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=summary.reset_index(), x="阈值", y="有超标天数比例", color="#bcbddc", ax=ax)
    ax.set_xlabel("浊度阈值（NTU）")
    ax.set_ylabel("有超标天数比例")
    ax.set_ylim(0, 1)
    save_figure(fig, OUTPUT_DIR, "图3_阈值敏感性")


def plot_hour_date_heatmap(frame):
    heatmap_data = frame.copy()
    heatmap_data["运行日期"] = pd.to_datetime(heatmap_data["operating_date"])
    heatmap_data["运行小时"] = (heatmap_data["timestamp"].dt.hour - 7) % 24
    pivot = heatmap_data.pivot_table(index="运行日期", columns="运行小时", values="treated_ntu", aggfunc="median")
    pivot = pivot.reindex(columns=range(0, 24, 2))
    pivot.columns = [f"{(hour + 7) % 24:02d}:00" for hour in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(pivot, cmap="YlOrRd", vmin=0, center=1, ax=ax, cbar_kws={"label": "出厂水浊度（NTU）"})
    ax.set_xlabel("运行时刻")
    ax.set_ylabel("运行日期")
    save_figure(fig, OUTPUT_DIR, "图4_逐日逐时浊度热力图")


def main():
    set_chinese_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_clean_data()
    frame, daily = build_daily_metrics(data)
    plot_daily_metrics(daily)
    plot_monthly_exceedance(daily)
    threshold_sensitivity(frame)
    plot_hour_date_heatmap(frame)
    print(f"问题4探索性分析已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
