# 问题4物理约束模糊聚类风险评价 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`. 主代理只分派、审查和汇总；每个任务由一个新的实现子代理按 TDD 完成，并在任务间接受审查。

**Goal:** 用问题3的 MoE 补全 2026 年第一季度缺测出厂水浊度，以四项日风险指标、2025 年完整实测超标日的 K=3 模糊 C 均值和透明阈值下限，输出逐日四级风险评价及必要验证结果。

**Architecture:** 只改写 `src/question4_eda.py`：其中集中完成数据整理、MoE 补全、日指标、简洁的模糊 C 均值、物理约束融合、表格和图形输出。只新建 `tests/test_question4_eda.py`，以小型 DataFrame 和替身预测器覆盖核心规则；现有 `sources/` 文献记录保留，不新建 `src/q4/`、模型持久化、测试夹具或额外模块。

**Tech Stack:** Python、pandas、NumPy、matplotlib、openpyxl、pytest；复用问题3已有 MoE 加载与预测接口，不添加依赖。

## Global Constraints

- 主代理不得运行 Python；实现、测试和正式运行均由实施子代理执行。
- 仅修改 `src/question4_eda.py`，仅创建 `tests/test_question4_eda.py`；保留现有 `sources/` 记录。
- 不写类型注解、`typing`、`Protocol`、`dataclass` 或其他类型说明代码。
- 不写异常检测、离群点检测或相关图表、指标、筛除逻辑。
- `treated_ntu` 的缺测值只由问题3已保存的 MoE 预测补全；实测值优先，并标记为“实测”或“MoE预测”。
- 每个运行日为 07:00 至次日 05:00 的 12 个两小时时点；2025 年训练日只取 12 个完整实测值的超标日，2026 年不参与训练。
- 四项特征固定为峰值超标幅度 `M`、超标观测比例 `F`、最长连续超标时长 `D`、累计超标负荷 `L`；不做 PCA。
- 用固定种子 2026 的 K=3、`m=2.0` 模糊 C 均值。对 `M/D/L` 做 `log1p`，再以训练中位数/IQR 缩放；按四项中心的风险方向由低到高映射低、中、高风险。
- 透明阈值下限固定为：最大 NTU 不超过 1 为安全；否则最大 NTU 超过 3 或 `D>6h` 为高风险；否则最大 NTU 超过 2 或 `D>2h` 为中风险；其余为低风险。
- 1 NTU 是安全硬门：超标日绝不为安全。最终序号为 `max(阈值序号, 聚类序号)`，其中安全、低、中、高依次为 0、1、2、3。
- 输出 2026 年 3 月逐日 Excel、第一季度等级占比、必要图表、K=2/3/4 聚类有效性，以及阈值上下 20% 的简单敏感性；图形保存 PNG 和 SVG，中文黑体、无图内标题。

## File Structure

- Rewrite `src/question4_eda.py`: 唯一的计算、融合和输出入口。
- Create `tests/test_question4_eda.py`: 唯一的确定性单元与输出契约测试。

---

### Task 1: 核心计算

**Files:**
- Rewrite: `src/question4_eda.py`
- Create: `tests/test_question4_eda.py`

**Interfaces:**
- Consumes: 清洗后的逐时数据和问题3的已保存 MoE 模型。
- Produces: `build_daily_features`、`fit_fuzzy_clusters`、`classify_q1` 和 `fuse_grades`；逐日结果包含 M/F/D/L、来源统计、阈值等级、聚类等级、隶属度和最终等级。

- [ ] **Step 1: 实施子代理先写失败测试**

在 `tests/test_question4_eda.py` 用内嵌的 12 点 DataFrame 和简单替身预测器，写出下列断言：实测优先、缺测点标为 MoE预测；`M/F/D/L` 分别等于超标幅度最大值、超标比例、最长连续超标点数乘 2、小于或等于 1 之外的累计幅度乘 2；1 NTU 不超标为安全；阈值等级不会被聚类降级；聚类可将低风险超标日上调。

- [ ] **Step 2: 实施子代理运行单测并确认失败**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_question4_eda.py -v`

Expected: FAIL，因为问题4接口尚未实现或与测试契约不一致。

- [ ] **Step 3: 实施子代理在单一脚本实现最小流程**

在 `src/question4_eda.py` 中复用问题3 MoE 补全缺测点，按运行日汇总 M/F/D/L 和实测/预测计数。只用 NumPy 循环实现 K=3 模糊 C 均值，保存中心、缩放统计量和风险顺序于运行时对象即可；不保存模型文件。对安全日直接给出 0；对超标日计算阈值等级与聚类等级，并取二者最大值。

- [ ] **Step 4: 实施子代理运行单测并确认通过**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_question4_eda.py -v`

Expected: PASS。

### Task 2: 输出

**Files:**
- Modify: `src/question4_eda.py`
- Modify: `tests/test_question4_eda.py`

**Interfaces:**
- Consumes: Task 1 的第一季度逐时与逐日分类结果、聚类中心。
- Produces: `outputs/04_问题4/` 中的逐日表、占比表、3 月 Excel、聚类诊断/敏感性表和必要图形。

- [ ] **Step 1: 实施子代理先写失败输出测试**

测试 `write_outputs` 后读取结果：3 月 Excel 的“3月逐日分类”工作表恰有 31 行，包含日期、M/F/D/L、实测点数、预测补全点数、阈值等级、聚类等级、最终等级和分类依据；等级占比表四类天数之和为 90、占比之和为 1。

- [ ] **Step 2: 实施子代理运行输出测试并确认失败**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_question4_eda.py -k output -v`

Expected: FAIL，因为输出尚未生成。

- [ ] **Step 3: 实施子代理补齐最小且可审计的输出**

输出逐时来源 CSV、第一季度逐日分类 CSV、四级天数占比 CSV、3 月逐日 Excel，以及一个含“聚类中心”“K值比较”“阈值敏感性”工作表的 Excel。绘制三张必要图：四级风险天数占比、月度风险等级分布、逐日最大 NTU 与风险等级；每张保存 300 dpi PNG 和 SVG。分类依据只写“国标内”“阈值下限主导”“聚类预警升级”或“阈值与聚类一致”。

- [ ] **Step 4: 实施子代理运行输出测试并确认通过**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_question4_eda.py -v`

Expected: PASS。

### Task 3: 验证

**Files:**
- Modify only if a test exposes a defect: `src/question4_eda.py` or `tests/test_question4_eda.py`

**Interfaces:**
- Consumes: Tasks 1–2 的实现与问题3模型。
- Produces: 可复现的 `outputs/04_问题4/` 结果和简明验证记录。

- [ ] **Step 1: 实施子代理扩展并运行聚类与敏感性测试**

在同一测试文件断言：K=3 的中心风险顺序递增；K=2、3、4 均输出轮廓系数、Calinski–Harabasz、Davies–Bouldin 和 FPC；阈值倍率 0.8、0.9、1.0、1.1、1.2 均有四级占比，且超标日始终不是安全。

Run: `E:\Anaconda3\python.exe -m pytest tests/test_question4_eda.py -v`

Expected: PASS。

- [ ] **Step 2: 实施子代理执行正式问题4流程**

Run: `E:\Anaconda3\python.exe src/question4_eda.py`

Expected: 生成 `outputs/04_问题4/`，其中 3 月 Excel 有连续 31 日，第一季度占比覆盖 90 个运行日。

- [ ] **Step 3: 主代理审查子代理证据，不运行 Python**

审查子代理提供的测试输出、输出文件清单和关键断言结果：MoE 仅用于缺测、M/F/D/L 完整、K=3 在 2025 完整实测超标日训练、1 NTU 硬门有效、最终 `max` 融合有效、3 月 Excel/占比/图表/有效性/敏感性齐全。主代理不执行 Python、不中途重构、不创建额外文件。

## Plan Self-Review Checklist

- 文件范围只有 `src/question4_eda.py` 和 `tests/test_question4_eda.py`，并保留现有文献记录。
- 核心保留问题3 MoE、M/F/D/L、1 NTU 硬门、2025 完整实测超标日 K=3、阈值下限和 `max` 融合。
- 结果保留 3 月 Excel、季度占比、三张必要图、K 值有效性和简单阈值敏感性。
- 实现不含类型说明、异常/离群检测、模型持久化、额外包、额外夹具或额外测试文件。
- 全部实施和 Python 命令由 `superpowers:subagent-driven-development` 的实现子代理完成；主代理仅协调与审查。
