# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

本仓库是 2026 年数学建模竞赛统计培训第二阶段的自来水厂水质预测与评估项目，仅包含四问在建模前的**数据预处理与探索性分析 (EDA)** 部分。完整赛题见 `题目.pdf`，评分参考见 `参考评分.docx`。本题以分钟级运行日的逐日数据为对象，重点研究出厂水浊度的成因、时滞关系、多步预测特征和超标风险。

## 运行流程

所有脚本必须按顺序执行；EDA 脚本依赖 `data_preprocessing.py` 产出的 `outputs/00_数据预处理/水质监测数据_清洗后.pkl`。

```powershell
python -m pip install -r requirements.txt
python src/data_preprocessing.py
python src/prob1.py
python src/question2_eda.py
python src/question3_eda.py
python src/question4_eda.py
```

数据从 `附件/附件1  2025数据集/`（2025 年月度 `.xlsx`）和 `附件/附件2  2026数据集/`（2026 年逐日工作表 `.xls`）读取；处理中产生的所有图表与表格写入 `outputs/` 对应子目录。

## 目录与代码结构

```
src/
├── utils.py                # 列名/月份常量、Excel 日期解析、文件月份识别
├── eda_common.py           # 共享样式(黑体)、load_clean_data、save_figure、regular_series、zscore_frame
├── data_preprocessing.py   # 一站式清洗：日期修复、去重、缺失标记、Hampel 异常标记、反冲洗事件
├── prob1.py                # 出厂水浊度影响因素、季节性、STL 分解与多模型预测
├── question2_eda.py        # 滤后水浊度与输入变量的时滞识别（24h 差分后相关）
├── question3_eda.py        # 多步预测(2/6/12h)的工艺链时滞与日内模式
└── question4_eda.py        # 2026 年 1-3 月超标幅度、持续时间、阈值敏感性
```

### 数据口径（不可改动）

- 运行日从 07:00 到次日 05:00；01/03/05 点的时间戳归属下一自然日，同时保留 `operating_date` 字段。
- 2025 年部分工作簿前 12 天出现日月颠倒，使用文件名月份重建日期；同样通过 `factorize` 兜底填充。
- 短缺口插值���针对非目标数值变量在同源文件内、连续不超过 3 个采样点；**`filtered_ntu` 与 `treated_ntu` 不插补**。
- 异常点只标记 `outlier_*` 列，不删除；`B/W`、`BACKWASH`、`FILTER` 等备注生成 `is_backwash_event`。
- 2026 年 2 月 `treated_ntu` 整列为空，EDA 保留为待预测区间，**不要把缺测当 0 或"安全"**。
- 2025 年 5-8 月缺失的多列为整列缺失，不跨月插补；通过 `available_*` 字段记录字段可用性。

### 输出约定

- 图表同时保存 300 dpi PNG + SVG，背景白色，`bbox_inches="tight"`。
- 中文统一用 `SimHei`（黑体）；坐标轴去除上、右脊柱；按要求不设置 `title`。
- 表格 CSV 用 `utf-8-sig` 编码，保留中文表头。
- 季度差异检验、STL 分解、时滞相关系数、超标指标等关键结果直接落 CSV，便于在论文里复用。

## 关键设计要点

- **数据流**：所有 EDA 脚本都通过 `eda_common.load_clean_data()` 读取同一份清洗后数据，不要再次重复 ETL。
- **时区**：所有时间序列统一按本地时区解析；季节性差分固定为 `diff(12)`（24 小时步长）、STL 周期 `period=12`（24 小时）。
- **异常检测**：`mark_robust_outliers` 使用滚动中位数 + MAD（`scale = 1.4826 * MAD`），阈值 `6`；窗口 `37`（≈3 天）中心化。
- **图样式**：色板以 `tab10`、`RdBu_r`、`YlOrRd`、`YlGnBu` 为主；散点图超过 2500 行随机采样以保持可读性。
- **缺失值监控**：原始缺失用 `missing_*` 标记；插值后再统计 `*_missing_rate`，用于质量报告。

## 依赖

依赖在 `requirements.txt` 中：pandas、numpy、openpyxl（处理 2025 年 `.xlsx`）、xlrd（处理 2026 年 `.xls`）、matplotlib、seaborn、scipy、statsmodels。

## 数据放置

- 2025 月度文件必须命名为 `JBALB_<Month英文>2025.xlsx`，由 `utils.find_month_from_name` 解析月份。
- 2026 日度工作簿为 `.xls`（注意是 xlrd 2.0+），工作表名格式 `M.DD`（如 `2.01`）。
- 任何被 Office 锁定的临时文件以 `~$` 开头，会被自动跳过。

## 后续工作

仓库当前只完成 EDA；正式建模（多元回归、VAR/状态空间、LSTM 等）和论文写作将基于 `outputs/` 下生成的图表与表格展开，新增模型脚本时建议放到 `src/model/` 子目录下并复用 `eda_common` 的样式与读取函数。
