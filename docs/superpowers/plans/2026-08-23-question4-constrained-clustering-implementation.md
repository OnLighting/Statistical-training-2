# 问题4物理约束模糊聚类风险评价 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 1 NTU 国标硬约束为安全门槛，结合问题3 MoE 对缺测时点的预测、四个可解释日风险指标、模糊 C 均值聚类和工程阈值风险下限，完成 2026 年 1—3 月逐日四级风险评价，并输出等级占比、3 月逐日 Excel、稳健性检验和周期分布图。

**Architecture:** `q4.data` 负责生成完整且带来源标记的 2026 年第一季度逐时浊度序列；`q4.risk` 负责四项日指标和透明阈值基准等级；`q4.clustering` 在 2025 年历史超标日上拟合三簇模糊 C 均值，并将簇按风险方向排序。最终等级采用“1 NTU 安全硬门 + `max(阈值基准等级, 聚类等级)`”的物理约束融合，聚类只能提高预警等级、不能把工程规则识别的风险降级；`question4_eda.py` 只负责端到端编排、验证和输出。

**Tech Stack:** Python 3.12、pandas、NumPy、SciPy、scikit-learn、matplotlib、seaborn、openpyxl、joblib、pytest；复用 `src/prob3.py`、`src/q3/` 已训练 MoE 模型，不增加聚类依赖。

**Spec:** `题目.pdf` 问题4、`参考评分.docx` 第六部分，以及 `sources/papers_20260823_turbidity_risk_classification.md`。

## Global Constraints

- 国标硬条件固定为出厂水浊度 `NTU <= 1`；任一有效时点 `NTU > 1` 时，当日最终等级不得为“安全”。
- 运行日固定为当日 07:00 至次日 05:00，共 12 个 2 小时时点；所有持续时长按 2 小时采样间隔计算。
- `treated_ntu` 不做线性插值、不填 0；2026 年 2 月和 1/3 月零散缺测只能使用问题3保存的 MoE 模型补全。
- 问题3模型必须从 `outputs/03_问题3/models/` 加载，训练截止时间必须不晚于 `2026-02-01 05:00`；问题4不得重新拟合或改变问题3模型。
- 每个逐时值必须保留 `数值来源`：`实测` 或 `MoE预测`；每日必须报告实测点数、预测补全点数和预测占比。
- 2025 年风险原型训练只使用实测出厂水浊度完整日（每日恰好 12 个有效观测）；不使用 2026 年待评价数据更新聚类中心。
- 安全日不参与聚类；仅对 2025 年历史超标日拟合 `K=3` 的低、中、高风险原型。
- 聚类特征固定为峰值超标幅度 `M`、超标观测比例 `F`、最长连续超标时长 `D`、累计超标负荷 `L`；不做 PCA，保留原始物理含义。
- 模糊 C 均值参数固定为 `m=2.0`、`max_iter=500`、`tol=1e-7`、`random_state=2026`；训练前对 `M/D/L` 做 `log1p`，再对四项特征做中位数/IQR 稳健标准化。
- 阈值基准等级固定为：`max_NTU <= 1` 为安全；否则 `max_NTU > 3` 或 `D > 6h` 为高风险；否则 `max_NTU > 2` 或 `D > 2h` 为中风险；其余为低风险。
- 最终风险序号固定为 `安全=0、低风险=1、中风险=2、高风险=3`；超标日最终序号为 `max(阈值基准序号, 聚类序号)`。
- 缺少问题3模型、模型预测非有限、运行日无法形成 12 个有效值、2025 年可用超标训练日少于 30 天时必须显式报错，禁止静默降级或误判安全。
- 所有随机过程固定种子 2026；所有 CSV 使用 UTF-8-SIG；图形同时保存 300 dpi PNG 和 SVG，中文黑体，无图内标题。
- 保留当前 `src/prob3.py` 和 `src/prob3_plot.py` 的用户改动，不做无关重构。

## File Structure

- Create `src/q4/__init__.py`: 导出问题4公共数据、指标、聚类和等级接口。
- Create `src/q4/data.py`: 第一季度时间网格、问题3模型适配器、缺测预测补全和数值来源审计。
- Create `src/q4/risk.py`: 四项日风险指标、运行日完整性、阈值基准分类和等级名称映射。
- Create `src/q4/clustering.py`: 无额外依赖的模糊 C 均值、稳健变换、簇风险排序、预测与稳定性评价。
- Rewrite `src/question4_eda.py`: `Solver` 端到端编排、物理约束融合、阈值敏感性、表格和图形输出。
- Create `tests/q4_fixtures.py`: 问题4共享确定性逐时和逐日测试数据。
- Create `tests/test_q4_data.py`: 时间网格、实测优先、预测补全和来源标记测试。
- Create `tests/test_q4_risk.py`: 四项指标、边界条件、缺测保护和基准等级测试。
- Create `tests/test_q4_clustering.py`: 模糊隶属度、簇排序、确定性和稳定性测试。
- Create `tests/test_question4_model.py`: 融合规则、正式输出、3 月 31 行和端到端质量门测试。

---

### Task 1: 锁定逐时数据契约并接入问题3 MoE

**Files:**
- Create: `src/q4/__init__.py`
- Create: `src/q4/data.py`
- Create: `tests/q4_fixtures.py`
- Create: `tests/test_q4_data.py`

**Interfaces:**
- Consumes: `utils.load_clean_data() -> pandas.DataFrame`、`q3.data.prepare_q3_frame(data) -> pandas.DataFrame`、`prob3.Solver.load_models()`、`prob3.Solver.prediction_bundle(frame, origins)`。
- Produces: `EXPECTED_Q1_TIMESTAMPS`、`Q3TurbidityPredictor.predict(frame, target_timestamps) -> pandas.Series`、`assemble_q1_turbidity(data, predictor) -> pandas.DataFrame`。

- [ ] **Step 1: 编写共享逐时夹具**

在 `tests/q4_fixtures.py` 写入：

```python
import numpy as np
import pandas as pd


def operating_day_values(date="2026-03-01", values=None, observed=True):
    values = np.asarray(values if values is not None else np.full(12, 0.5), dtype=float)
    timestamp = pd.date_range(pd.Timestamp(date) + pd.Timedelta(hours=7), periods=12, freq="2h")
    return pd.DataFrame({
        "timestamp": timestamp,
        "operating_date": pd.Timestamp(date),
        "treated_ntu": values if observed else np.nan,
        "raw_water_ntu": 20.0,
        "treated_water_flow": 45.0,
    })


class ConstantPredictor:
    def __init__(self, value=0.8):
        self.value = float(value)
        self.calls = []

    def predict(self, frame, target_timestamps):
        index = pd.DatetimeIndex(target_timestamps)
        self.calls.append(index)
        return pd.Series(self.value, index=index, name="treated_ntu_prediction")
```

- [ ] **Step 2: 编写失败测试，锁定实测优先和预测补全**

```python
def test_assemble_q1_preserves_observed_and_predicts_only_missing():
    first = operating_day_values(values=[0.4] * 11 + [np.nan])
    predictor = ConstantPredictor(0.8)
    result = assemble_q1_turbidity(first, predictor, expected_timestamps=first["timestamp"])
    assert result.loc[result["timestamp"] == first["timestamp"].iloc[0], "risk_ntu"].iat[0] == 0.4
    assert result.loc[result["timestamp"] == first["timestamp"].iloc[-1], "risk_ntu"].iat[0] == 0.8
    assert result["数值来源"].tolist() == ["实测"] * 11 + ["MoE预测"]
    assert len(predictor.calls) == 1
    assert predictor.calls[0].tolist() == [first["timestamp"].iloc[-1]]


def test_assemble_q1_rejects_nonfinite_model_prediction():
    frame = operating_day_values(observed=False)
    predictor = ConstantPredictor(np.nan)
    with pytest.raises(ValueError, match="MoE预测包含非有限值"):
        assemble_q1_turbidity(frame, predictor, expected_timestamps=frame["timestamp"])
```

- [ ] **Step 3: 运行测试确认失败**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_q4_data.py -v`

Expected: FAIL，提示 `ModuleNotFoundError: No module named 'q4'`。

- [ ] **Step 4: 实现时间网格、模型适配器和补全逻辑**

`src/q4/data.py` 固定实现：

```python
Q1_START = pd.Timestamp("2026-01-01 07:00")
Q1_END = pd.Timestamp("2026-04-01 05:00")
EXPECTED_Q1_TIMESTAMPS = pd.date_range(Q1_START, Q1_END, freq="2h")


class Q3TurbidityPredictor:
    def __init__(self):
        self.solver = Q3Solver().load_models()
        if pd.Timestamp(self.solver.train_end) > pd.Timestamp("2026-02-01 05:00"):
            raise ValueError("问题3模型训练截止时间晚于允许边界")

    def predict(self, frame, target_timestamps):
        target_timestamps = pd.DatetimeIndex(target_timestamps)
        origins = target_timestamps - pd.Timedelta(hours=2)
        bundle = self.solver.prediction_bundle(frame, origins)
        return pd.Series(bundle["prediction"][:, 0], index=target_timestamps)
```

`assemble_q1_turbidity` 必须先用 `prepare_q3_frame(data)` 建立完整特征帧，再按目标时间戳对齐原始 `treated_ntu`；仅把缺测目标时间传给预测器。结果字段固定为 `timestamp`、`operating_date`、`observed_ntu`、`predicted_ntu`、`risk_ntu`、`数值来源`、`raw_water_ntu`、`treated_water_flow`，其中 `risk_ntu` 始终实测优先。

- [ ] **Step 5: 运行数据测试**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_q4_data.py -v`

Expected: PASS。

- [ ] **Step 6: Commit**

```powershell
git add src/q4/__init__.py src/q4/data.py tests/q4_fixtures.py tests/test_q4_data.py
git commit -m "feat(q4): assemble audited quarterly turbidity series"
```

---

### Task 2: 构造四项日风险指标和透明阈值基准模型

**Files:**
- Create: `src/q4/risk.py`
- Create: `tests/test_q4_risk.py`
- Modify: `src/q4/__init__.py`

**Interfaces:**
- Consumes: `assemble_q1_turbidity()` 或历史逐时表。
- Produces: `compute_daily_risk_features(hourly, value_column="risk_ntu") -> pandas.DataFrame`、`baseline_risk_grade(features) -> pandas.Series`、`GRADE_NAMES`。

- [ ] **Step 1: 编写四项指标失败测试**

```python
def test_daily_features_capture_magnitude_frequency_duration_and_load():
    values = [0.4, 1.2, 1.4, 0.8, 1.1, 1.3, 1.5, 0.7, 0.6, 0.5, 0.4, 0.3]
    hourly = operating_day_values(values=values).rename(columns={"treated_ntu": "risk_ntu"})
    hourly["数值来源"] = "实测"
    daily = compute_daily_risk_features(hourly).iloc[0]
    assert daily["峰值超标幅度M"] == pytest.approx(0.5)
    assert daily["超标观测比例F"] == pytest.approx(5 / 12)
    assert daily["最长连续超标时长D"] == pytest.approx(6.0)
    assert daily["累计超标负荷L"] == pytest.approx(2 * (0.2 + 0.4 + 0.1 + 0.3 + 0.5))
    assert daily["实测点数"] == 12
    assert daily["预测补全点数"] == 0


@pytest.mark.parametrize(
    ("maximum", "duration", "expected"),
    [(1.0, 0.0, 0), (1.01, 2.0, 1), (2.01, 2.0, 2), (1.5, 4.0, 2), (3.01, 2.0, 3), (1.5, 8.0, 3)],
)
def test_baseline_grade_obeys_engineering_boundaries(maximum, duration, expected):
    frame = pd.DataFrame({"日最大浊度": [maximum], "最长连续超标时长D": [duration]})
    assert baseline_risk_grade(frame).iat[0] == expected
```

- [ ] **Step 2: 编写缺测保护失败测试**

```python
def test_daily_features_require_all_twelve_risk_values():
    hourly = operating_day_values().rename(columns={"treated_ntu": "risk_ntu"}).iloc[:-1]
    hourly["数值来源"] = "实测"
    with pytest.raises(ValueError, match="运行日必须包含12个有效风险值"):
        compute_daily_risk_features(hourly)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_q4_risk.py -v`

Expected: FAIL，提示风险函数未定义。

- [ ] **Step 4: 实现日指标和基准等级**

对每个运行日按时间排序，固定计算：

```python
excess = np.maximum(values - 1.0, 0.0)
M = excess.max()
F = np.mean(values > 1.0)
D = longest_true_run(values > 1.0) * 2.0
L = excess.sum() * 2.0
```

同时输出日均、中位数、最大值、95% 分位数、超标点数、实测点数、预测补全点数、预测占比、原水日最大浊度和出厂水日均流量。`baseline_risk_grade` 按 Global Constraints 中的高风险优先顺序返回整数 0—3。

- [ ] **Step 5: 运行风险测试**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_q4_risk.py -v`

Expected: PASS。

- [ ] **Step 6: Commit**

```powershell
git add src/q4/risk.py src/q4/__init__.py tests/test_q4_risk.py
git commit -m "feat(q4): add interpretable daily risk indicators"
```

---

### Task 3: 实现三簇模糊 C 均值及可解释风险排序

**Files:**
- Create: `src/q4/clustering.py`
- Create: `tests/test_q4_clustering.py`
- Modify: `src/q4/__init__.py`

**Interfaces:**
- Consumes: 含 `峰值超标幅度M`、`超标观测比例F`、`最长连续超标时长D`、`累计超标负荷L` 的历史日表。
- Produces: `RiskFuzzyClusterer.fit(features) -> RiskFuzzyClusterer`、`membership(features) -> np.ndarray`、`predict_grade(features) -> np.ndarray`、`centroid_table() -> pandas.DataFrame`、`cluster_validity(features, k_values=(2,3,4)) -> pandas.DataFrame`、`bootstrap_stability(features, repeats=200) -> pandas.DataFrame`。

- [ ] **Step 1: 编写三个明确风险原型的夹具和失败测试**

```python
def three_risk_regimes(seed=2026, each=40):
    rng = np.random.default_rng(seed)
    centers = np.array([
        [0.20, 0.10, 2.0, 0.6],
        [1.20, 0.40, 6.0, 5.0],
        [3.00, 0.80, 14.0, 25.0],
    ])
    blocks = []
    for center in centers:
        noise = rng.normal(0, [0.03, 0.02, 0.20, 0.20], size=(each, 4))
        blocks.append(np.maximum(center + noise, 1e-4))
    return pd.DataFrame(np.vstack(blocks), columns=RISK_FEATURE_COLUMNS)


def test_fuzzy_memberships_sum_to_one_and_are_deterministic():
    data = three_risk_regimes()
    first = RiskFuzzyClusterer().fit(data)
    second = RiskFuzzyClusterer().fit(data)
    u1 = first.membership(data)
    u2 = second.membership(data)
    np.testing.assert_allclose(u1.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(u1, u2, atol=1e-8)


def test_cluster_grades_follow_increasing_physical_risk():
    data = three_risk_regimes()
    model = RiskFuzzyClusterer().fit(data)
    grade = model.predict_grade(data)
    assert np.median(grade[:40]) == 1
    assert np.median(grade[40:80]) == 2
    assert np.median(grade[80:]) == 3
    centers = model.centroid_table().sort_values("聚类风险序号")
    assert centers["综合风险方向得分"].is_monotonic_increasing
```

- [ ] **Step 2: 运行测试确认失败**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_q4_clustering.py -v`

Expected: FAIL，提示 `RiskFuzzyClusterer` 未定义。

- [ ] **Step 3: 实现稳健变换与模糊 C 均值**

训练变换固定为：

```python
RISK_FEATURE_COLUMNS = ["峰值超标幅度M", "超标观测比例F", "最长连续超标时长D", "累计超标负荷L"]
LOG_COLUMNS = ["峰值超标幅度M", "最长连续超标时长D", "累计超标负荷L"]
transformed[LOG_COLUMNS] = np.log1p(transformed[LOG_COLUMNS])
scaled = (transformed - training_median) / training_iqr
```

IQR 为 0 时替换为 1。模糊 C 均值迭代必须实现零距离保护：样本与某中心距离小于 `1e-12` 时，将其隶属度完全赋给该中心；否则按

```python
u_ij = 1.0 / sum((d_ij / d_ik) ** (2.0 / (m - 1.0)) for k in clusters)
center_j = sum((u_ij ** m) * x_i) / sum(u_ij ** m)
```

更新，直至中心最大变化小于 `tol`。

- [ ] **Step 4: 实现簇语义排序和概率输出**

将标准化空间中的每个中心在四个维度的训练经验百分位取平均，得到 `综合风险方向得分`；由低到高映射为低风险 1、中风险 2、高风险 3。`predict_grade` 返回最大隶属度所属簇的语义等级；另输出 `低风险隶属度`、`中风险隶属度`、`高风险隶属度` 和 `最大隶属度`。

- [ ] **Step 5: 实现聚类数和稳定性诊断**

`cluster_validity` 对 `K=2,3,4` 分别报告轮廓系数、Calinski–Harabasz 指数、Davies–Bouldin 指数和模糊划分系数 `FPC=sum(u**2)/n`。`bootstrap_stability` 固定 200 次有放回抽样，每次重拟合 K=3 后对完整训练集预测，再按风险方向排序标签，与原模型标签计算 adjusted Rand index；输出 ARI 均值、标准差、5% 分位数和最小值。

- [ ] **Step 6: 运行聚类测试**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_q4_clustering.py -v`

Expected: PASS。

- [ ] **Step 7: Commit**

```powershell
git add src/q4/clustering.py src/q4/__init__.py tests/test_q4_clustering.py
git commit -m "feat(q4): add interpretable fuzzy risk clustering"
```

---

### Task 4: 建立历史风险原型和物理约束融合分类

**Files:**
- Rewrite: `src/question4_eda.py`
- Create: `tests/test_question4_model.py`

**Interfaces:**
- Consumes: Tasks 1—3 公共接口。
- Produces: 无参 `Solver()`、`Solver.build_training_features(data)`、`Solver.classify_q1(data)`、`Solver.solve()`；分类结果包含阈值等级、聚类等级、三类隶属度和最终等级。

- [ ] **Step 1: 编写物理约束融合失败测试**

```python
def test_fusion_never_labels_exceedance_safe_or_below_baseline():
    baseline = np.array([0, 1, 2, 3])
    cluster = np.array([1, 1, 1, 2])
    maximum = np.array([0.8, 1.1, 2.4, 3.5])
    final = fuse_risk_grades(maximum, baseline, cluster)
    np.testing.assert_array_equal(final, [0, 1, 2, 3])


def test_fusion_allows_cluster_to_raise_early_warning():
    final = fuse_risk_grades(np.array([1.4]), np.array([1]), np.array([2]))
    np.testing.assert_array_equal(final, [2])
```

- [ ] **Step 2: 编写历史训练口径失败测试**

```python
def test_training_features_use_only_2025_observed_complete_days(monkeypatch):
    solver = Solver()
    data = historical_and_q1_fixture()
    training = solver.build_training_features(data)
    assert training["运行日期"].dt.year.eq(2025).all()
    assert training["预测补全点数"].eq(0).all()
    assert training["有效观测数"].eq(12).all()
    assert training["日最大浊度"].gt(1.0).all()
```

- [ ] **Step 3: 运行融合测试确认失败**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_question4_model.py -k "fusion or training" -v`

Expected: FAIL，提示 `Solver` 或 `fuse_risk_grades` 未定义。

- [ ] **Step 4: 实现 Solver 分类主流程**

`Solver.classify_q1` 固定执行：

```python
hourly = assemble_q1_turbidity(data, self.predictor)
daily = compute_daily_risk_features(hourly)
daily["阈值基准序号"] = baseline_risk_grade(daily)
membership = self.clusterer.membership(daily[RISK_FEATURE_COLUMNS])
daily["聚类风险序号"] = self.clusterer.predict_grade(daily[RISK_FEATURE_COLUMNS])
daily["最终风险序号"] = fuse_risk_grades(
    daily["日最大浊度"].to_numpy(),
    daily["阈值基准序号"].to_numpy(),
    daily["聚类风险序号"].to_numpy(),
)
daily["最终风险等级"] = daily["最终风险序号"].map(GRADE_NAMES)
```

安全日的聚类等级和三类隶属度写为 0，最终等级强制为安全；超标日取阈值与聚类等级最大值。另增加 `分类依据` 字段，分别写为 `国标内`、`阈值下限主导`、`聚类预警升级` 或 `阈值与聚类一致`。

- [ ] **Step 5: 实现对照和敏感性分析**

计算阈值模型与纯聚类模型、最终融合模型之间的一致率和 Cohen’s kappa。将幅度阈值 `(2,3)` 和时长阈值 `(2,6)` 同时乘以 `0.8、0.9、1.0、1.1、1.2`，聚类模型保持固定，重复计算融合等级占比；输出各方案四级天数及占比，不改变 1 NTU 安全硬门。

- [ ] **Step 6: 运行融合测试**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_question4_model.py -k "fusion or training" -v`

Expected: PASS。

- [ ] **Step 7: Commit**

```powershell
git add src/question4_eda.py tests/test_question4_model.py
git commit -m "feat(q4): fuse fuzzy clusters with physical risk floor"
```

---

### Task 5: 输出完整表格、3 月 Excel 和解释性图形

**Files:**
- Modify: `src/question4_eda.py`
- Modify: `tests/test_question4_model.py`

**Interfaces:**
- Consumes: `Solver.classify_q1()` 的逐时、逐日、聚类中心、诊断和敏感性结果。
- Produces: `outputs/04_问题4/` 下固定命名的 CSV、Excel、PNG、SVG 和模型审计文件。

- [ ] **Step 1: 编写输出契约失败测试**

```python
def test_write_outputs_contains_31_march_days_and_required_audit_columns(tmp_path, monkeypatch):
    solver = solved_q4_fixture()
    monkeypatch.setattr(question4_eda, "OUTPUT_DIR", tmp_path)
    solver.write_outputs()
    march = pd.read_excel(tmp_path / "表6_2026年3月逐日风险分类.xlsx", sheet_name="3月逐日分类")
    assert len(march) == 31
    assert set(march["风险等级"]).issubset({"安全", "低风险", "中风险", "高风险"})
    assert {
        "日期", "日最大浊度", "峰值超标幅度M", "超标观测比例F",
        "最长连续超标时长D", "累计超标负荷L", "实测点数", "预测补全点数",
        "阈值基准等级", "聚类等级", "低风险隶属度", "中风险隶属度",
        "高风险隶属度", "风险等级", "分类依据",
    }.issubset(march.columns)


def test_grade_share_counts_every_q1_operating_day(tmp_path, monkeypatch):
    solver = solved_q4_fixture()
    monkeypatch.setattr(question4_eda, "OUTPUT_DIR", tmp_path)
    solver.write_outputs()
    share = pd.read_csv(tmp_path / "表3_风险等级天数占比.csv")
    assert share["天数"].sum() == 90
    assert share["占比"].sum() == pytest.approx(1.0)
```

- [ ] **Step 2: 运行输出测试确认失败**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_question4_model.py -k "outputs or share" -v`

Expected: FAIL，提示输出文件不存在。

- [ ] **Step 3: 实现固定表格输出**

`write_outputs` 固定生成：

1. `表1_2026Q1逐时浊度与来源.csv`：每个时点的实测、预测、最终风险输入值和来源。
2. `表2_2026Q1逐日风险指标与分类.csv`：90 个运行日完整指标、隶属度和三级分类结果。
3. `表3_风险等级天数占比.csv`：安全、低、中、高的天数和占比，零天类别也保留。
4. `表4_月度与星期风险分布.csv`：月份/星期 × 风险等级的天数、组内占比。
5. `表5_聚类中心与模型诊断.xlsx`：`聚类中心`、`聚类数比较`、`稳定性`、`模型一致性`、`阈值敏感性`、`数据来源审计` 六个工作表。
6. `表6_2026年3月逐日风险分类.xlsx`：`3月逐日分类` 和 `判定规则` 两个工作表。
7. `models/fuzzy_risk_clusterer.joblib`：训练中位数、IQR、三个中心、中心风险顺序、参数和训练日期范围。

Excel 使用 openpyxl 冻结首行、启用自动筛选、日期格式 `yyyy-mm-dd`、数值保留四位小数；`判定规则` 工作表明确写出“1 NTU 为国标硬约束，2/3 NTU 与 2/6 小时为本研究工程分层阈值，不是新增国家标准”。

- [ ] **Step 4: 实现固定图形输出**

`plot_outputs` 固定生成 PNG 和 SVG：

1. `图1_超标幅度持续时间与聚类`：横轴 `D`、纵轴 `M`、点大小为 `L`、颜色为最终等级。
2. `图2_聚类中心风险画像`：四指标训练百分位的雷达图，展示低/中/高三个原型。
3. `图3_四级风险天数占比`：四级条形图并标注天数与占比。
4. `图4_月度风险等级分布`：1—3 月 100% 堆积柱状图。
5. `图5_星期小时超标率热力图`：星期 × 运行时刻的 `NTU>1` 比例，逐时使用 `risk_ntu`。
6. `图6_阈值敏感性`：阈值倍率 × 四级占比折线图。
7. `图7_逐日风险等级与数据来源`：逐日最大 NTU 曲线、1 NTU 线、风险背景色和 2 月预测区间阴影。

- [ ] **Step 5: 运行输出测试**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_question4_model.py -v`

Expected: PASS。

- [ ] **Step 6: Commit**

```powershell
git add src/question4_eda.py tests/test_question4_model.py
git commit -m "feat(q4): export risk classifications and diagnostics"
```

---

### Task 6: 全流程运行与验收

**Files:**
- Modify only if verification exposes a tested defect: `src/q4/*.py`、`src/question4_eda.py`、对应测试。

**Interfaces:**
- Consumes: Tasks 1—5 全部实现和 `outputs/03_问题3/models/`。
- Produces: 可复现的 `outputs/04_问题4/` 正式结果。

- [ ] **Step 1: 运行问题4单元测试**

Run: `E:\Anaconda3\python.exe -m pytest tests/test_q4_data.py tests/test_q4_risk.py tests/test_q4_clustering.py tests/test_question4_model.py -v`

Expected: PASS，0 failed。

- [ ] **Step 2: 运行全项目回归测试**

Run: `E:\Anaconda3\python.exe -m pytest tests -v`

Expected: PASS，问题1—3和预处理测试无回归。

- [ ] **Step 3: 无参数正式运行问题4**

Run: `E:\Anaconda3\python.exe src/question4_eda.py`

Expected: 从问题3模型补全缺测，写入 `outputs/04_问题4/`，进程退出码为 0。

- [ ] **Step 4: 执行机器可读验收**

Run:

```powershell
E:\Anaconda3\python.exe -c "from pathlib import Path; import pandas as pd; p=Path(r'outputs/04_问题4'); d=pd.read_csv(p/'表2_2026Q1逐日风险指标与分类.csv'); m=pd.read_excel(p/'表6_2026年3月逐日风险分类.xlsx',sheet_name='3月逐日分类'); s=pd.read_csv(p/'表3_风险等级天数占比.csv'); assert len(d)==90; assert len(m)==31; assert s['天数'].sum()==90; assert abs(s['占比'].sum()-1)<1e-9; assert (d.loc[d['日最大浊度']>1,'最终风险序号']>=1).all(); assert (d.loc[d['阈值基准序号']>d['聚类风险序号'],'最终风险序号']==d['阈值基准序号']).all(); assert d.loc[d['运行日期'].str.startswith('2026-02'),'预测补全点数'].gt(0).all(); print(s.to_string(index=False))"
```

Expected: 所有断言通过并打印四级天数占比。

- [ ] **Step 5: 检查聚类质量和可解释性门槛**

检查 `表5_聚类中心与模型诊断.xlsx`：

- K=3 的三个中心按综合风险方向得分严格递增；
- 每簇至少包含 5 个历史训练日；
- K=3 轮廓系数大于 0；
- Bootstrap ARI 均值至少 0.70；
- 阈值基准与最终融合 Cohen’s kappa 有限；
- 阈值上下浮动 20% 时，不得出现超标日被归为安全。

如任一门槛失败，先新增一个能够复现该失败的测试，再修复实现；不得仅通过修改输出绕过质量门。

- [ ] **Step 6: 人工检查全部图形和 Excel**

逐一打开 7 张 PNG：确认中文无乱码、图例和坐标轴完整、无图内标题、2 月预测区间标识清楚、雷达图的低/中/高原型顺序合理。打开 3 月 Excel，确认 31 日齐全、日期连续、四项指标单位明确、实测/预测来源可追溯、判定规则未将研究阈值表述为国标。

- [ ] **Step 7: 最终差异审计**

Run: `git diff --check`

Expected: 无空白错误；`git status --short` 中 `src/prob3.py` 和 `src/prob3_plot.py` 的既有用户改动未被本任务覆盖。

---

## Plan Self-Review Checklist

- Spec coverage: 1 NTU 硬约束、幅度/频率/时长/累计负荷、四级占比、3 月 Excel、月份/星期/小时分布、阈值稳健性和来源不确定性均有对应任务。
- Physical meaning: 聚类仅作用于四个原始可解释指标；安全门和阈值风险下限不可被聚类覆盖；簇中心通过风险方向排序并输出画像。
- Missing-data safety: 2 月及零散缺测只由预 2 月训练的 MoE 补全，逐点标注来源；任何非有限预测或不完整运行日都会失败而非判安全。
- Statistical validation: K=2/3/4 比较、四项内部指标、200 次 bootstrap ARI、阈值法一致性和 ±20% 敏感性全部落表。
- Type consistency: 日风险特征列名、等级序号、隶属度列和输出字段在 Tasks 2—6 中保持一致。
- Scope: 不重训问题3、不引入新依赖、不写论文正文，仅完成问题4模型、测试和可复用结果。
- Completeness scan: 所有接口、命令和验收结果均已具体定义。
