from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
from prob2 import prepare_dynamic_frame, lag_correlations, TARGET, FREQUENCY_HOURS
from utils import label, load_clean_data, save_figure, set_chinese_style
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "02_问题2"
def plot_dynamic_overview(data):
    columns = ["raw_water_ntu", "raw_water_ph", "raw_water_flow", TARGET]
    daily = data.set_index("timestamp")[columns].resample("D").median()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
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
        ax.set_ylabel("滤后水浊度")
    return fig
def plot_delayed_scatter(frame, best_lags):
    fig = create_delayed_scatter_figure(frame, best_lags)
    save_figure(fig, OUTPUT_DIR, "图3_时滞变量与滤后水浊度关系")
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
    result, best_lags = lag_correlations(data)
    best_lags.to_csv(OUTPUT_DIR / "表2_初步时滞识别结果.csv", index=False, encoding="utf-8-sig")
    plot_dynamic_overview(data)
    result, best_lags = lag_correlations(data)
    plot_lag_correlations(result)
    plot_delayed_scatter(frame, best_lags)
    plot_backwash_comparison(data)
if __name__ == '__main__':
    main()