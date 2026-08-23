"""tmp_plot.py — 临时脚本:绘制 表4_预测结果.xlsx 中三个日期的预测曲线(1行3列)"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from utils import save_figure, set_chinese_style  # noqa: E402

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "outputs/01_问题1/表4_预测结果.xlsx"
COEF_FILE = ROOT / "outputs/01_问题1/表5_稳健周期回归公式系数.csv"
OUTPUT_DIR = ROOT / "outputs/01_问题1"


def main() -> None:
    set_chinese_style()

    df = pd.read_excel(INPUT_FILE)
    df.columns = ["日期", "时间", "预测NTU"]
    # 把 "07:00" 等转成 0~23 的整数小时,这样 23:00 后面才是 01:00 跨日点
    df["小时"] = df["时间"].str.split(":").str[0].astype(int)
    dates = sorted(df["日期"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    colors = plt.get_cmap("tab10").colors

    for ax, date, color in zip(axes, dates, colors):
        sub = df[df["日期"] == date].sort_values("小时").reset_index(drop=True)
        ax.plot(sub["小时"], sub["预测NTU"], marker="o", linewidth=2, color=color)
        ax.set_xticks(sub["小时"])
        ax.set_xticklabels([f"{h:02d}" for h in sub["小时"]])
        ax.set_xlabel("小时")
        ax.set_title(f"{date}预测")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False)

    axes[0].set_ylabel("出厂水浊度/NTU")
    save_figure(fig, OUTPUT_DIR, "图12_三个日期预测曲线")

    plot_coefficient_scatter()


def plot_coefficient_scatter() -> None:
    """绘制 表5 中指定变量的标准化系数柱状图:正向=红色,负向=蓝色。"""
    keep_columns = [
        "滤后水浊度", "原水浊度", "河水水位", "清水池水位", "原水流量",
        "矾投加量", "出厂水流量", "原水色度", "余氯", "处理后水pH", "原水pH",
    ]

    coef = pd.read_csv(COEF_FILE, encoding="utf-8-sig")
    coef.columns = ["变量", "均值", "标准差", "标准化系数", "原始量纲系数", "影响方向"]

    # 仅保留用户指定的变量,并按 keep_columns 的顺序排列
    coef = (
        coef[coef["变量"].isin(keep_columns)]
        .set_index("变量")
        .reindex(keep_columns)
        .dropna(subset=["标准化系数"])
        .reset_index()
    )

    is_positive = coef["影响方向"].str.contains("正")
    colors = ["#d62728" if pos else "#1f77b4" for pos in is_positive]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(coef["变量"], coef["标准化系数"], color=colors,
           edgecolor="black", linewidth=0.5, zorder=3)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("变量")
    ax.set_ylabel("标准化系数")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    # 图例(用空矩形代理两类颜色)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d62728", edgecolor="black", label="正向影响"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="负向影响"),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True)

    save_figure(fig, OUTPUT_DIR, "图13_标准化系数柱状图")


if __name__ == "__main__":
    main()
