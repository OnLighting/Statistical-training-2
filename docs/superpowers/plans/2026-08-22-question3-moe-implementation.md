# 问题3机理约束混合专家模型 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建并验证 RTD/CSTR、LightGBM、GRU 三专家及 Softmax 门控组成的多步预测模型，输出三个指定日期的 21 条预测、工艺敏感性结果、图表、Excel 和可重新加载的训练模型包。

**Architecture:** 使用独立 `q3` 包隔离数据契约、三位专家、门控和评价。四个扩展时间折产生严格样本外专家预测来训练门控，最终仅用 2026 年 2 月以前数据重训模型；2 月目标历史缺失时由机理专家递推并携带缺失标记。上游 ARX 情景传播、工艺敏感性、模型保存/加载、报表输出和端到端编排全部集中在 `src/question3_model.py` 的单个 `Solver` 类中。

**Tech Stack:** Python 3.12、pandas、NumPy、SciPy、statsmodels、scikit-learn、LightGBM、PyTorch、SHAP、joblib、matplotlib、seaborn、openpyxl、pytest。

**Spec:** `docs/superpowers/specs/2026-08-22-question3-moe-design.md`

## Global Constraints

- 原始采样间隔为 2 小时；可验证预测步长固定为 2、4、6、8、10、12 小时。
- 模型选择和拟合不得使用 2026 年 2 月之后的出厂水浊度。
- 所有变换器只在当前训练折拟合，验证期目标变化不得影响预测。
- 四个验证窗口固定为 2025-07-01--07-28、2025-09-01--09-28、2025-11-01--11-28、2026-01-01--01-28。
- 所有随机过程固定种子 2026。
- 结构性缺失不跨月插补；目标缺失不得填 0。
- 最终敏感性仅使用工艺幅度：原水浊度 +20/+50/+100 NTU、矾投加量 +/-0.01 及联合扰动，不做标准差倍数测试。
- 不生成 `outputs/03_问题3/问题3_建模结果说明.md` 或其他结果说明 Markdown。
- 保存的模型不得包含绝对机器路径，必须能在新进程中加载并复现 21 条指定日期预测。
- 图表沿用项目中文样式、300 dpi PNG、无图内标题；CSV 使用 UTF-8-SIG。
- `python src/question3_model.py` 不接收任何命令行参数；固定读取清洗数据并写入 `outputs/03_问题3/`。
- 所有超参数硬编码在对应模型类中，不创建中央参数对象，不从 CLI、配置文件或环境变量覆盖。
- 当前 Git 根目录错误覆盖整个用户主目录，执行期间不得提交或暂存任何文件，除非先确认本工作区成为独立仓库。

## File Structure

- Create `src/q3/__init__.py`: 公共类型和主流程导出。
- Create `src/q3/data.py`: 2 小时规则化、固定时间折、时间特征、滞后特征和监督样本。
- Create `src/q3/mechanistic.py`: RTD/CSTR 状态、参数拟合、状态空间校正和缺失目标递推。
- Create `src/q3/tree_expert.py`: 六个 LightGBM 直接多步模型。
- Create `src/q3/gru_expert.py`: 序列张量、GRU 网络、训练和预测。
- Create `src/q3/moe.py`: 样本外专家预测拼接、Softmax 门控训练和组合预测。
- Create `src/q3/evaluation.py`: 指标、分层评价、长缺失回测和残差块区间。
- Create `src/question3_model.py`: 单个 `Solver` 类，负责敏感性、模型持久化、报表和无参数端到端运行。
- Modify `requirements.txt`: 增加 LightGBM、PyTorch、SHAP 和 joblib 版本下限。
- Create `tests/q3_fixtures.py`: 所有第三问测试共享的确定性合成时间序列。
- Create `tests/test_q3_data.py`, `tests/test_q3_mechanistic.py`, `tests/test_q3_experts.py`, `tests/test_q3_moe.py`, `tests/test_q3_evaluation.py`, `tests/test_question3_model.py`。

---

### Task 1: 固定常量、时间折与无泄漏样本构造

**Files:**
- Create: `src/q3/__init__.py`
- Create: `src/q3/data.py`
- Create: `tests/q3_fixtures.py`
- Create: `tests/test_q3_data.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `utils.load_clean_data() -> pandas.DataFrame`。
- Produces: `TemporalFold`, `DEFAULT_FOLDS`, `FINAL_TRAIN_END`, `HORIZONS`, `prepare_q3_frame(data)`, `build_origins(frame, start, end)`, `make_feature_frame(frame, fit_end)`, `make_targets(frame, origins)`；`make_targets` 始终使用模块内固定 `HORIZONS`。

- [ ] **Step 1: 写入依赖下限**

在 `requirements.txt` 追加：

```text
lightgbm>=4.3
torch>=2.2
shap>=0.45
joblib>=1.3
```

- [ ] **Step 2: 编写失败测试，锁定规则化、时间折与预测目标对齐**

先在 `tests/q3_fixtures.py` 定义共享夹具：

```python
def synthetic_q3_data(periods=720, start="2025-01-01 07:00", include_target_dates=False):
    rng = np.random.default_rng(2026)
    timestamp = pd.date_range(start, periods=periods, freq="2h")
    step = np.arange(periods)
    raw = 30 + 15 * np.sin(2 * np.pi * step / 12) + rng.normal(0, 2, periods)
    alum = np.where(raw > 35, 0.06, 0.05)
    filtered = np.maximum(0.02, 0.05 + 0.002 * raw - 0.4 * alum + rng.normal(0, 0.01, periods))
    treated = pd.Series(filtered).ewm(alpha=0.35, adjust=False).mean().to_numpy() + 0.20
    data = pd.DataFrame({
        "timestamp": timestamp,
        "raw_water_ntu": raw,
        "raw_water_ph": 7.0,
        "filtered_ntu": filtered,
        "clear_well_level": 3.8 + 0.02 * np.sin(2 * np.pi * step / 12),
        "treated_ntu": treated,
        "alum_feed_rate": 0.01,
        "alum_dosage": alum,
        "raw_water_flow": 50 + np.sin(2 * np.pi * step / 12),
        "treated_water_flow": 46 + np.cos(2 * np.pi * step / 12),
        "is_backwash_event": False,
    })
    if include_target_dates:
        mask = data["timestamp"].between("2026-02-01 07:00", "2026-02-28 23:00")
        data.loc[mask, "treated_ntu"] = np.nan
    return data

def synthetic_regular_frame(periods=720, start="2025-01-01 07:00"):
    return synthetic_q3_data(periods, start).set_index("timestamp")

def supervised_fixture(n=360):
    frame = synthetic_regular_frame(n)
    origins = frame.index[24:-6]
    y = np.column_stack([frame["treated_ntu"].shift(-h).loc[origins] for h in range(1, 7)])
    return frame, origins, y

def frame_fixture(periods=720):
    return synthetic_regular_frame(periods)

def upstream_fixture(periods=240):
    return synthetic_regular_frame(periods)
```

然后编写数据测试：

```python
def test_prepare_q3_frame_is_two_hourly_and_deduplicated():
    raw = synthetic_q3_data(periods=80)
    raw = pd.concat([raw, raw.iloc[[10]]], ignore_index=True)
    frame = prepare_q3_frame(raw)
    assert frame.index.is_unique
    assert frame.index.to_series().diff().dropna().eq(pd.Timedelta(hours=2)).all()

def test_default_folds_never_train_on_or_after_validation_start():
    assert len(DEFAULT_FOLDS) == 4
    assert all(f.train_end < f.valid_start <= f.valid_end for f in DEFAULT_FOLDS)
    assert FINAL_TRAIN_END == pd.Timestamp("2026-02-01 05:00")

def test_make_targets_maps_steps_to_two_hour_horizons():
    frame = synthetic_q3_data(periods=20).set_index("timestamp")
    origins = frame.index[[5]]
    y = make_targets(frame, origins)
    np.testing.assert_allclose(y[0], frame["treated_ntu"].iloc[6:12])
```

- [ ] **Step 3: 运行数据测试确认失败**

Run: `pytest tests/test_q3_data.py -v`

Expected: FAIL，提示 `ModuleNotFoundError: No module named 'q3'`。

- [ ] **Step 4: 实现配置和数据接口**

`data.py` 直接定义固定常量：

```python
@dataclass(frozen=True)
class TemporalFold:
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    @property
    def train_end(self) -> pd.Timestamp:
        return self.valid_start - pd.Timedelta(hours=2)

FREQUENCY = "2h"
HORIZONS = (1, 2, 3, 4, 5, 6)
HISTORY_STEPS = 24
RANDOM_STATE = 2026
FINAL_TRAIN_END = pd.Timestamp("2026-02-01 05:00")
TARGET_DATES = ("2026-02-01", "2026-02-10", "2026-02-20")
DEFAULT_FOLDS = (
    TemporalFold(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-28 23:00")),
    TemporalFold(pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-28 23:00")),
    TemporalFold(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-28 23:00")),
    TemporalFold(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-28 23:00")),
)
```

`data.py` 必须：去重后按 2 小时重采样；保留原始缺失标记；加入 Fourier 时间编码；所有滚动特征使用 `.shift(1)` 后再滚动，防止把预测时刻之后的信息混入历史特征；`make_targets` 返回 `(n_origins, 6)` 数组。

- [ ] **Step 5: 运行测试并回归现有测试**

Run: `pytest tests/test_q3_data.py tests/test_data_preprocessing.py -v`

Expected: PASS。

---

### Task 2: RTD/CSTR 机理状态空间专家

**Files:**
- Create: `src/q3/mechanistic.py`
- Create: `tests/test_q3_mechanistic.py`

**Interfaces:**
- Consumes: `prepare_q3_frame()` 输出。
- Produces: `MechanisticExpert.fit(frame, train_end)`, `MechanisticExpert.predict(frame, origins) -> np.ndarray`, `MechanisticExpert.fill_target_history(frame) -> pandas.Series`, `MechanisticExpert.to_state() -> dict`。

- [ ] **Step 1: 编写 CSTR 递推与输出形状失败测试**

```python
def test_cstr_state_moves_monotonically_toward_constant_inlet():
    inlet = np.full(12, 1.0)
    flow = np.full(12, 50.0)
    level = np.full(12, 3.8)
    state = cstr_cascade(inlet, flow, level, n_tanks=1, volume_scale=100.0, initial=0.0)
    assert np.all(np.diff(state) >= 0)
    assert 0 < state[-1] < 1

def test_mechanistic_expert_predicts_six_finite_horizons():
    frame = synthetic_regular_frame(periods=240)
    model = MechanisticExpert().fit(frame, frame.index[179])
    pred = model.predict(frame, frame.index[180:190])
    assert pred.shape == (10, 6)
    assert np.isfinite(pred).all()

def test_fill_target_history_does_not_replace_observed_values():
    frame = synthetic_regular_frame(periods=100)
    frame.loc[frame.index[70:80], "treated_ntu"] = np.nan
    model = MechanisticExpert().fit(frame, frame.index[69])
    filled = model.fill_target_history(frame)
    np.testing.assert_allclose(filled.loc[:frame.index[69]], frame.loc[:frame.index[69], "treated_ntu"])
    assert filled.loc[frame.index[70:80]].notna().all()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_q3_mechanistic.py -v`

Expected: FAIL，提示 `cstr_cascade` 或 `MechanisticExpert` 未定义。

- [ ] **Step 3: 实现 CSTR、有界参数选择和状态空间校正**

实现核心：

```python
def cstr_cascade(inlet, flow, level, n_tanks, volume_scale, initial):
    effective_volume = np.maximum(level * volume_scale, 1e-6)
    alpha = np.exp(-np.maximum(flow, 1e-6) * 2.0 / effective_volume)
    states = np.empty((len(inlet), n_tanks), dtype=float)
    previous = np.full(n_tanks, initial, dtype=float)
    for i in range(len(inlet)):
        previous[0] = alpha[i] * previous[0] + (1 - alpha[i]) * inlet[i]
        for j in range(1, n_tanks):
            previous[j] = alpha[i] * previous[j] + (1 - alpha[i]) * previous[j - 1]
        states[i] = previous
    return states[:, -1]
```

对 `n_tanks in (1, 2, 3)` 和 `volume_scale` 有界区间 `[1, 5000]` 使用训练期尾部滚动 MAE 选择。状态空间校正使用 `statsmodels` 的 SARIMAX/状态空间回归，包含机理状态、时间 Fourier 项、流量和反冲洗标记；只在 `train_end` 以前拟合。

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_q3_mechanistic.py -v`

Expected: PASS。

---

### Task 3: LightGBM 直接多步专家

**Files:**
- Create: `src/q3/tree_expert.py`
- Create: `tests/test_q3_experts.py`

**Interfaces:**
- Consumes: 特征 DataFrame、训练 origins、六步目标矩阵、机理填充目标。
- Produces: `LightGBMExpert.fit(frame, origins, y)`, `predict(frame, origins) -> np.ndarray`, `feature_importance() -> pandas.DataFrame`, `booster_paths()`。

- [ ] **Step 1: 编写六模型、目标隔离和确定性失败测试**

```python
def test_lightgbm_expert_returns_six_horizons_and_is_deterministic():
    frame, origins, y = supervised_fixture(n=360)
    first = LightGBMExpert().fit(frame, origins, y)
    second = LightGBMExpert().fit(frame, origins, y)
    p1 = first.predict(frame, origins[-20:])
    p2 = second.predict(frame, origins[-20:])
    assert p1.shape == (20, 6)
    np.testing.assert_allclose(p1, p2)

def test_lightgbm_prediction_ignores_future_ground_truth():
    frame, origins, y = supervised_fixture(n=360)
    model = LightGBMExpert().fit(frame, origins[:-30], y[:-30])
    altered = frame.copy()
    altered.loc[origins[-20:] + pd.Timedelta(hours=2), "treated_ntu"] = 99.0
    np.testing.assert_allclose(
        model.predict(frame, origins[-20:]),
        model.predict(altered, origins[-20:]),
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_q3_experts.py -k lightgbm -v`

Expected: FAIL，提示 `LightGBMExpert` 未定义。

- [ ] **Step 3: 实现六步直接模型和小网格选择**

实现一个步长一个 `lightgbm.LGBMRegressor(objective="huber")`。训练权重规则固定为：原水 2 小时变化绝对值 >=50 NTU 权重 2；矾投加变化非零权重 2；目标位于训练期 95% 分位以上权重 2；多条件同时满足时权重相乘但上限为 4。`LightGBMExpert` 不接收超参数；其类常量硬编码设计规定的网格、最多 800 轮和 50 轮早停。

- [ ] **Step 4: 运行 LightGBM 测试**

Run: `pytest tests/test_q3_experts.py -k lightgbm -v`

Expected: PASS。

---

### Task 4: GRU 序列到向量专家

**Files:**
- Create: `src/q3/gru_expert.py`
- Modify: `tests/test_q3_experts.py`

**Interfaces:**
- Consumes: 48 小时多变量窗口、机理填充目标、缺失掩码和六步目标。
- Produces: `build_sequence_tensors(frame, origins, targets, filled_target) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, `GRUNet`, `GRUExpert.fit(frame, origins, targets, filled_target) -> GRUExpert`, `GRUExpert.predict(frame, origins, filled_target) -> np.ndarray`, `GRUExpert.state_dict_bundle() -> dict`。

- [ ] **Step 1: 编写序列对齐、缺失掩码和确定性失败测试**

```python
def test_sequence_tensor_uses_exactly_24_past_steps():
    frame, origins, y = supervised_fixture(n=160)
    x, mask, targets = build_sequence_tensors(frame, origins[24:], y[24:], history_steps=24)
    assert x.shape[1] == 24
    assert mask.shape[:2] == x.shape[:2]
    assert targets.shape[1] == 6

def test_gru_expert_handles_mechanistically_filled_target():
    frame, origins, y = supervised_fixture(n=220)
    frame.loc[origins[120:150], "treated_ntu"] = np.nan
    model = GRUExpert().fit(frame, origins[:160], y[:160])
    pred = model.predict(frame, origins[160:180])
    assert pred.shape == (20, 6)
    assert np.isfinite(pred).all()
```

- [ ] **Step 2: 运行 GRU 测试确认失败**

Run: `pytest tests/test_q3_experts.py -k gru -v`

Expected: FAIL，提示 GRU 接口未定义。

- [ ] **Step 3: 实现张量构造、网络和早停训练**

实现：

```python
class GRUNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 6)
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])
```

损失为逐样本加权的 `SmoothL1Loss(reduction="none")`，六个步长等权。`GRUExpert` 不接收超参数；类常量硬编码隐藏维度候选 32/64、Dropout 候选 0.1/0.2、学习率 `1e-3`、批量大小 64、最多 200 轮和 20 轮早停。每次训练设置 NumPy、Python、PyTorch 随机种子 2026，并启用可用的确定性算法。

- [ ] **Step 4: 运行专家测试**

Run: `pytest tests/test_q3_experts.py -v`

Expected: PASS。

---

### Task 5: 四折样本外预测与 Softmax 门控

**Files:**
- Create: `src/q3/moe.py`
- Create: `tests/test_q3_moe.py`

**Interfaces:**
- Consumes: 三位专家工厂、`DEFAULT_FOLDS`、专家预测 `(n, 6, 3)`、门控特征 `(n, 6, p)`、真实目标 `(n, 6)`。
- Produces: `generate_oof_predictions(frame, expert_factories) -> OOFBundle`, `SoftmaxGate.fit(expert_predictions, gate_features, targets) -> SoftmaxGate`, `SoftmaxGate.weights(expert_predictions, gate_features) -> np.ndarray`, `SoftmaxGate.predict(expert_predictions, gate_features) -> np.ndarray`。

- [ ] **Step 1: 编写权重约束、动态分配和验证隔离失败测试**

```python
def test_softmax_gate_weights_are_nonnegative_and_sum_to_one():
    expert_pred, gate_x, y = gate_fixture()
    gate = SoftmaxGate().fit(expert_pred, gate_x, y)
    weights = gate.weights(expert_pred, gate_x)
    assert weights.shape == expert_pred.shape
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=2), 1.0, atol=1e-6)

def test_oof_predictions_train_each_fold_strictly_before_validation(monkeypatch):
    seen = []
    factories = recording_expert_factories(seen)
    generate_oof_predictions(frame_fixture(), factories)
    assert all(train_end < valid_start for train_end, valid_start in seen)
```

同一测试文件中定义 `gate_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]`，返回形状分别为 `(60, 6, 3)`、`(60, 6, 5)`、`(60, 6)` 的确定性数组；定义 `recording_expert_factories(seen)`，其三个桩专家在 `fit` 时把 `(train_end, valid_start)` 追加到 `seen`，在 `predict` 时返回有限的 `(n, 6)` 数组。`frame_fixture` 从 `tests/q3_fixtures.py` 导入。

- [ ] **Step 2: 运行门控测试确认失败**

Run: `pytest tests/test_q3_moe.py -v`

Expected: FAIL，提示门控接口未定义。

- [ ] **Step 3: 实现 OOF 编排和线性 Softmax 门控**

门控参数使用 PyTorch 单层 `nn.Linear(p + 3, 3)`；将门控特征与三个专家预测拼接，按步长展开为二维样本。损失：

```python
weights = torch.softmax(logits, dim=-1)
combined = (weights * expert_predictions).sum(dim=-1)
huber = F.smooth_l1_loss(combined, target)
balance = ((weights.mean(dim=0) - 1 / 3) ** 2).sum()
loss = huber + 1e-3 * balance
```

门控只在四折 OOF 预测上拟合。`generate_oof_predictions` 必须记录每条预测的 origin、horizon、fold、train_end、valid_start，便于泄漏审计。

- [ ] **Step 4: 运行门控测试**

Run: `pytest tests/test_q3_moe.py -v`

Expected: PASS。

---

### Task 6: 分步长评价、分层评价和长缺失回测

**Files:**
- Create: `src/q3/evaluation.py`
- Create: `tests/test_q3_evaluation.py`

**Interfaces:**
- Consumes: 带 origin、horizon、actual、prediction、hour、weekday、season、regime 的长表。
- Produces: `metric_table(long_predictions) -> pandas.DataFrame`, `stratified_metric_table(long_predictions) -> pandas.DataFrame`, `long_gap_backtest(frame, expert_factories) -> pandas.DataFrame`, `residual_block_intervals(oof_residuals, point) -> pandas.DataFrame`。

- [ ] **Step 1: 编写指标、分层与区间失败测试**

```python
def test_metric_table_reports_every_model_and_horizon():
    table = metric_table(prediction_long_fixture())
    assert set(table["预测步长/小时"]) == {2, 4, 6, 8, 10, 12}
    assert {"季节朴素", "机理专家", "LightGBM", "GRU", "MoE"}.issubset(table["模型"])
    assert {"RMSE", "MAE", "R2", "样本数"}.issubset(table.columns)

def test_residual_block_intervals_are_ordered():
    intervals = residual_block_intervals(oof_residual_fixture(), point=np.full(21, 0.3))
    assert np.all(intervals["95%下限"] <= intervals["80%下限"])
    assert np.all(intervals["80%下限"] <= intervals["预测值"])
    assert np.all(intervals["预测值"] <= intervals["80%上限"])
    assert np.all(intervals["80%上限"] <= intervals["95%上限"])
```

同一测试文件中定义 `prediction_long_fixture()`，构造五个模型、六个步长各 20 条带 `actual` 和 `prediction` 的记录；定义 `oof_residual_fixture()`，构造至少 40 个连续运行日、每天 12 条、包含 `operating_date` 和 `residual` 的确定性残差表。

- [ ] **Step 2: 运行评价测试确认失败**

Run: `pytest tests/test_q3_evaluation.py -v`

Expected: FAIL，提示评价函数未定义。

- [ ] **Step 3: 实现评价与 10/20/28 天遮挡回测**

`long_gap_backtest` 在历史数据中选择三个完整区间，复制数据后将区间内 `treated_ntu` 置空，使用区间开始前拟合的模型递推预测；不得在遮挡区间内重拟合。残差区间按运行日组织连续 7 日块，抽样 200 次，将抽样残差加到 21 个点预测上并取 10/90、2.5/97.5 分位。

- [ ] **Step 4: 运行评价测试**

Run: `pytest tests/test_q3_evaluation.py -v`

Expected: PASS。

---

### Task 7: 单一 Solver 类完成敏感性、持久化、报表和端到端运行

**Files:**
- Create: `src/question3_model.py`
- Modify: `src/q3/__init__.py`
- Create: `tests/test_question3_model.py`

**Interfaces:**
- Consumes: Tasks 1--6 的数据、专家、门控和评价接口。
- Produces: 无参 `Solver()`、`Solver.solve()`，以及类内 `_build_process_scenarios`、`_fit_upstream_arx`、`_simulate_filtered`、`_run_sensitivity`、`_save_models`、`_load_models`、`_write_outputs`、`_plot_outputs`。

- [ ] **Step 1: 编写 Solver 结构、无参数入口和固定场景失败测试**

```python
def test_solver_and_solve_accept_no_configuration_arguments():
    assert list(inspect.signature(Solver).parameters) == []
    assert list(inspect.signature(Solver.solve).parameters) == ["self"]
    source = inspect.getsource(question3_model)
    assert "argparse" not in source
    assert "--quick" not in source
    assert "--output-dir" not in source

def test_solver_contains_only_approved_process_scenarios():
    solver = Solver()
    scenarios = solver._build_process_scenarios(frame_fixture(), pd.Timestamp("2026-02-01"))
    assert set(scenarios) == {
        "基准", "原水+20NTU", "原水+50NTU", "原水+100NTU",
        "矾量-0.01", "矾量+0.01", "原水+50NTU且矾量+0.01",
    }
    assert not any("sigma" in name.lower() or "标准差" in name for name in scenarios)
```

- [ ] **Step 2: 运行 Solver 结构测试确认失败**

Run: `pytest tests/test_question3_model.py -k "configuration or scenarios" -v`

Expected: FAIL，提示 `Solver` 未定义。

- [ ] **Step 3: 建立无参 Solver 骨架和固定路径**

`src/question3_model.py` 顶层固定：

```python
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "03_问题3"
MODEL_DIR = OUTPUT_DIR / "models"

class Solver:
    def __init__(self):
        self.mechanistic = MechanisticExpert()
        self.tree = LightGBMExpert()
        self.gru = GRUExpert()
        self.gate = SoftmaxGate()
        self.upstream = None
        self.result = None

    def solve(self):
        data = load_clean_data()
        frame = prepare_q3_frame(data)
        self._fit_oof_and_gate(frame)
        self._evaluate_oof(frame)
        self._fit_final_experts(frame)
        self._predict_target_dates(frame)
        self._run_sensitivity(frame)
        self._save_models()
        self._write_outputs()
        self._plot_outputs()
        self._verify_reload(frame)
        return self.result

if __name__ == "__main__":
    Solver().solve()
```

这些私有方法在后续步骤中全部落地；不得增加构造参数或 CLI。

- [ ] **Step 4: 编写上游 ARX 与扰动传播失败测试**

```python
def test_solver_upstream_simulation_changes_filtered_path_after_raw_shock():
    solver = Solver()
    frame = upstream_fixture()
    solver._fit_upstream_arx(frame, frame.index[180])
    scenarios = solver._build_process_scenarios(frame, frame.index[181].normalize())
    base = solver._simulate_filtered(frame, scenarios["基准"])
    shocked = solver._simulate_filtered(frame, scenarios["原水+50NTU"])
    assert not np.allclose(base, shocked)
```

- [ ] **Step 5: 在 Solver 中实现上游 ARX 和敏感性**

`_fit_upstream_arx` 使用 `filtered_ntu` 1--12 阶滞后及 `raw_water_ntu_lag1`、`raw_water_ph_lag1`、`alum_dosage_lag1`、`raw_water_flow_lag2` 的 Ridge；正则强度固定为 `alpha=1.0`。`_simulate_filtered` 在 6 小时扰动窗口递归生成滤后水路径，不读取窗口内真实滤后水。`_run_sensitivity` 返回目标日期、情景、预测峰值变化、峰值出现时间、12 小时累计浊度增量、恢复时间和恢复时间下界；12 小时内未恢复时文本值为 `>12`。

- [ ] **Step 6: 编写 Solver 模型保存、哈希和重载失败测试**

```python
def test_solver_model_round_trip_uses_relative_manifest_paths(tmp_path, monkeypatch):
    solver = fitted_solver_fixture()
    monkeypatch.setattr(question3_model, "MODEL_DIR", tmp_path)
    expected = solver._predict_with_loaded_state(frame_fixture(), origins_fixture())
    solver._save_models()
    loaded = Solver()
    monkeypatch.setattr(question3_model, "MODEL_DIR", tmp_path)
    loaded._load_models()
    actual = loaded._predict_with_loaded_state(frame_fixture(), origins_fixture())
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    manifest = json.loads((tmp_path / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["random_state"] == 2026
    assert all(not Path(item["path"]).is_absolute() for item in manifest["files"])
```

- [ ] **Step 7: 在 Solver 中实现模型包保存与加载**

`_save_models` 写入 `mechanistic_expert.joblib`、六个 `lightgbm_h02.txt`--`lightgbm_h12.txt`、`gru_expert.pt`、`moe_gate.joblib`、`preprocessor.joblib`、`model_manifest.json`。清单记录相对路径、SHA-256、字节数、依赖版本、训练范围、特征顺序和种子。`_load_models` 先校验哈希，不一致时抛出 `ValueError("模型文件哈希校验失败: <relative path>")`。

- [ ] **Step 8: 编写 Solver 报表输出失败测试**

```python
def test_solver_writes_required_outputs_without_markdown(tmp_path, monkeypatch):
    solver = Solver()
    solver.result = pipeline_result_fixture()
    monkeypatch.setattr(question3_model, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(question3_model, "MODEL_DIR", tmp_path / "models")
    solver._write_outputs()
    workbook = pd.ExcelFile(tmp_path / "表5_指定日期NTU预测结果.xlsx")
    forecast = pd.read_excel(workbook, sheet_name="指定日期预测")
    assert len(forecast) == 21
    assert forecast.groupby("日期").size().eq(7).all()
    assert not list(tmp_path.rglob("*.md"))
```

- [ ] **Step 9: 在 Solver 中实现表格和图形输出**

`_write_outputs` 和 `_plot_outputs` 固定生成表1--表6、图1--图6及包含 `指定日期预测`、`模型评价`、`门控权重`、`敏感性分析`、`质量检查` 的 Excel。预测工作表固定 21 行；不创建 Markdown。Excel 冻结首行、启用筛选、数值保留四位小数。

- [ ] **Step 10: 实现 Solver.solve 的固定执行顺序**

`solve` 无参数并固定执行：加载清洗数据；规则化；生成四折 OOF；训练门控；计算评价和区间；用全部预 2 月数据重训专家；连续填充 2 月目标历史；生成三个日期 21 条预测；运行工艺敏感性；保存模型；输出表图；新建第二个 `Solver()` 加载模型并复算 21 条预测；把最大复现误差写入质量检查工作表。

- [ ] **Step 11: 运行第三问全部测试**

Run: `pytest tests/test_q3_data.py tests/test_q3_mechanistic.py tests/test_q3_experts.py tests/test_q3_moe.py tests/test_q3_evaluation.py tests/test_question3_model.py -v`

Expected: PASS。

- [ ] **Step 12: 运行全项目回归测试**

Run: `pytest tests -v`

Expected: PASS，问题1、问题2和预处理测试无回归。

- [ ] **Step 13: 无参数正式运行**

Run: `python src/question3_model.py`

Expected: 不需要也不接受任何参数；固定读取清洗数据并写入 `outputs/03_问题3/`，包含六张表/工作簿、六张 PNG 和 `models/`，不包含 Markdown。

- [ ] **Step 14: 验证最终输出**

Run:

```powershell
python -c "import json,pandas as pd; from pathlib import Path; p=Path(r'outputs/03_问题3'); x=pd.read_excel(p/'表5_指定日期NTU预测结果.xlsx',sheet_name='指定日期预测'); assert len(x)==21; assert x['MoE预测NTU'].notna().all(); assert not list(p.rglob('*.md')); m=json.loads((p/'models/model_manifest.json').read_text(encoding='utf-8')); assert len(m['files'])>=10; print(x[['日期','时间','MoE预测NTU']].to_string(index=False))"
```

Expected: 断言通过并打印 21 条预测。

- [ ] **Step 15: 人工检查**

打开六张 PNG 检查中文、坐标轴、图例和置信区间；确认评价包含五模型六步长，MoE 权重和为 1，敏感性只有七个工艺情景，模型清单路径均为相对路径且哈希通过。

---

## Plan Self-Review Checklist

- Spec coverage: 三专家、门控、无泄漏四折、六步预测、长缺失回测、目标日期 21 条、工艺敏感性、区间、SHAP、模型包和禁止结果 Markdown 均有对应任务。
- Type consistency: 所有专家统一返回 `(n_origins, 6)`；门控输入专家维度固定为 3；最终长表使用 2--12 小时步长。
- Leakage controls: Task 1、3、5、6、7 均有显式测试或执行规则。
- Artifact controls: Task 7 验证相对路径、SHA-256 和重载复现。
- Placeholder scan: 计划不包含待填实现项；所有命令、文件和验收结果已明确。
