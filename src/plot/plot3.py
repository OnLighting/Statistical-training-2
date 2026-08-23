import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
from utils import label, load_clean_data, save_figure, set_chinese_style
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "03_问题3"
DYNAMIC_INPUTS = [
    "raw_water_ntu",
    "filtered_ntu",
    "clear_well_level",
    "alum_dosage",
    "alum_feed_rate",
    "raw_water_flow",
    "treated_water_flow",
]
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
    result.to_csv(OUTPUT_DIR / "表1_工艺链候选时滞.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for column, group in result.groupby("变量"):
        ax.plot(group["领先小时"], group["相关系数"], marker="o", markersize=3, linewidth=1.1, label=label(column))
    ax.set_xlabel("输入变量领先出厂水浊度的小时数")
    ax.set_ylabel("24小时差分后的相关系数")
    ax.legend(frameon=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_figure(fig, OUTPUT_DIR, "图1_工艺链候选时滞")
def plot_time_patterns(data):
    valid = data.loc[data["treated_ntu"].notna()].copy()
    hour_month = valid.pivot_table(index="month", columns="hour", values="treated_ntu", aggfunc="median")
    weekday_hour = valid.pivot_table(index="weekday", columns="hour", values="treated_ntu", aggfunc="median").reindex(range(7))
    weekday_hour.index = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    shared_vmin = float(min(hour_month.min().min(), weekday_hour.min().min()))
    shared_vmax = float(max(hour_month.max().max(), weekday_hour.max().max()))
    cmap = "YlGnBu"
    fig = plt.figure(figsize=(13, 6))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0,0.06],
        hspace=0.15,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cbar_ax = fig.add_subplot(grid[1, :])
    cell_style = {"linewidths": 0.05, "linecolor": "gray"}
    heatmaps = [
        sns.heatmap(
            hour_month,
            cmap=cmap,
            vmin=shared_vmin,
            vmax=shared_vmax,
            ax=axes[0],
            cbar=False,
            **cell_style,
        ),
        sns.heatmap(
            weekday_hour,
            cmap=cmap,
            vmin=shared_vmin,
            vmax=shared_vmax,
            ax=axes[1],
            cbar=False,
            **cell_style,
        ),
    ]
    axes[0].set_xlabel("小时")
    axes[0].set_ylabel("月份")
    axes[1].set_xlabel("小时")
    axes[1].set_ylabel("星期")
    cbar = fig.colorbar(
        heatmaps[0].collections[0],
        cax=cbar_ax,
        orientation="horizontal",
        label="中位数/NTU")
    cbar.outline.set_linewidth(0.1)
    save_figure(fig, OUTPUT_DIR, "图2_月份星期与小时模式")
def main():
    set_chinese_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_clean_data()
    process_lag_correlations(data)
    plot_time_patterns(data)
    print(f"问题3分析已保存至：{OUTPUT_DIR}")
if __name__ == "__main__":
    main()
