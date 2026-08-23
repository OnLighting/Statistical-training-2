from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_clean_data, set_chinese_style
GRADE_NAMES = ["安全", "低风险", "中风险", "高风险"]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "04_问题4"
NTU_LIMIT = 1.0
NEAR_LIMIT = 0.8
TIME_WEIGHT = 0.45
def operate_dates(frame):
    return (frame["timestamp"] - pd.Timedelta(hours=7)).dt.normalize()
def longest_exceedance(exceedance):
    longest, current = 0, 0
    for value in exceedance:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
def moe_predict(frame):
    from prob3 import Solver
    from q3.data import prepare_q3_frame
    solver = Solver().load_models()
    prepared = prepare_q3_frame(frame)
    prediction = pd.Series(np.nan, index=prepared.index, dtype=float)
    missing = pd.to_numeric(prepared["treated_ntu"], errors="coerce").isna()
    positions = np.flatnonzero(missing.to_numpy())
    origins = []
    targets = []
    for position in positions:
        origin_position = position - 1
        if origin_position >= 24 and position + 5 < len(prepared):
            origins.append(prepared.index[origin_position])
            targets.append(prepared.index[position])
    if origins:
        bundle = solver.prediction_bundle(prepared, pd.DatetimeIndex(origins))
        prediction.loc[targets] = bundle["prediction"][:, 0]
    return prediction.reindex(pd.to_datetime(frame["timestamp"])).set_axis(frame.index)
def daily_grid(data):
    points = data.copy()
    if "timestamp" not in points:
        points = points.reset_index().rename(columns={points.index.name or "index": "timestamp"})
    points["timestamp"] = pd.to_datetime(points["timestamp"], errors="coerce")
    points = points.dropna(subset=["timestamp"]).sort_values("timestamp")
    points = points.drop_duplicates("timestamp", keep="last")
    dates = operate_dates(points).drop_duplicates().sort_values()
    timestamps = []
    for date in dates:
        timestamps.extend(pd.date_range(date + pd.Timedelta(hours=7), periods=12, freq="2h"))
    grid = pd.DataFrame({"timestamp": timestamps})
    grid["运行日期"] = operate_dates(grid)
    return grid.merge(points.drop(columns=["operating_date"], errors="ignore"), on="timestamp", how="left")
def build_daily_features(data, moe_predictor=None, target_data=None):
    points = daily_grid(data if target_data is None else target_data)
    context = daily_grid(data)
    observed = pd.to_numeric(points["treated_ntu"], errors="coerce")
    missing = observed.isna()
    if missing.any():
        predictor = moe_predictor or moe_predict
        context_prediction = predictor(context.copy())
        if not isinstance(context_prediction, pd.Series):
            context_prediction = pd.Series(context_prediction, index=context.index)
        context_prediction = pd.to_numeric(context_prediction.reindex(context.index), errors="coerce")
        by_timestamp = pd.Series(context_prediction.to_numpy(), index=context["timestamp"])
        predicted = pd.to_numeric(points["timestamp"].map(by_timestamp), errors="coerce")
        observed.loc[missing] = predicted.loc[missing]
    points["treated_ntu"] = observed
    rows = []
    for date, group in points.groupby("运行日期", sort=True):
        values = pd.to_numeric(group["treated_ntu"], errors="coerce")
        exceedance = values.gt(NTU_LIMIT) & values.notna()
        near_limit = values.ge(NEAR_LIMIT * NTU_LIMIT) & values.notna()
        count = int(exceedance.sum())
        excess = (values - NTU_LIMIT).clip(lower=0)
        rows.append(
            {
                "运行日期": date,
                "有效观测数": int(values.notna().sum()),
                "日最大NTU": values.max() if values.notna().any() else np.nan,
                "日均NTU": values.mean() if values.notna().any() else np.nan,
                "NTU标准差": values.std(ddof=0) if values.notna().any() else np.nan,
                "临界频率": float(near_limit.sum()) / 12,
                "临界持续率": longest_exceedance(near_limit.tolist()) / 12,
                "M": excess.max() if values.notna().any() else np.nan,
                "F": count / 12,
                "D": longest_exceedance(exceedance.tolist()) * 2,
                "L": excess.sum(skipna=True) * 2,
            }
        )
    daily = pd.DataFrame(rows).sort_values("运行日期").reset_index(drop=True)
    return daily
def time_features(frame):
    dates = pd.to_datetime(frame["运行日期"], errors="coerce")
    phase = 2 * np.pi * (dates.dt.dayofyear.to_numpy(dtype=float) - 1) / 365
    return np.column_stack([np.sin(phase), np.cos(phase)])
def risk_features(frame, regime):
    values, names = None, None
    if regime == "within_limit":
        columns = ["日最大NTU", "日均NTU", "NTU标准差", "临界频率", "临界持续率"]
        values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        values[:, :3] = values[:, :3] / NTU_LIMIT
        names = ["最大NTU比", "平均NTU比", "标准差比", "临界频率", "临界持续率"]
    elif regime == "exceedance":
        columns = ["M", "F", "D", "L"]
        values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        values[:, 0] = np.log1p(values[:, 0] / NTU_LIMIT)
        values[:, 2] = values[:, 2] / 24.0
        values[:, 3] = np.log1p(values[:, 3] / (24.0 * NTU_LIMIT))
        names = ["超标幅度", "超标频率", "超标持续率", "超标负荷"]
    return values, names
def transform_features(frame, regime, median=None, iqr=None):
    risk, names = risk_features(frame, regime)
    if median is None:
        median = np.nanmedian(risk, axis=0)
    if iqr is None:
        lower = np.nanquantile(risk, 0.25, axis=0)
        upper = np.nanquantile(risk, 0.75, axis=0)
        iqr = upper - lower
        iqr[iqr < 1e-12] = 1.0
    scaled_risk = (risk - median) / iqr
    time = time_features(frame) * TIME_WEIGHT
    transformed = np.column_stack([scaled_risk, time])
    return transformed, risk, median, iqr
def fuzzy_c_means(values, clusters, seed):
    generator = np.random.default_rng(seed)
    membership = generator.random((len(values), clusters))
    membership = membership / membership.sum(axis=1, keepdims=True)
    for _ in range(300):
        powered = membership ** 2.0
        centers = (powered.T @ values) / powered.sum(axis=0)[:, None]
        distance = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        distance = np.maximum(distance, 1e-12)
        updated = 1.0 / (distance[:, :, None] / distance[:, None, :]) ** 2.0
        updated = updated.sum(axis=2) ** -1
        if np.max(np.abs(updated - membership)) < 1e-8:
            membership = updated
            break
        membership = updated
    return centers, membership
def fit_two_clusters(training, regime, grades):
    transformed, risk, median, iqr = transform_features(training, regime)
    valid = np.isfinite(transformed).all(axis=1)
    transformed = transformed[valid]
    risk = risk[valid]
    centers, membership = fuzzy_c_means(transformed, 2, 2026)
    labels = membership.argmax(axis=1)
    if np.unique(labels).size < 2:
        severity = np.nanmean((risk - np.nanmedian(risk, axis=0)) / iqr, axis=1)
        order = np.argsort(severity)
        labels = np.zeros(len(order), dtype=int)
        labels[order[len(order) // 2:]] = 1
        centers = np.vstack([transformed[labels == idx].mean(axis=0) for idx in range(2)])
    risk_centers = np.vstack([risk[labels == idx].mean(axis=0) for idx in range(2)])
    severity = np.nanmean((risk_centers - np.nanmedian(risk, axis=0)) / iqr, axis=1)
    cluster_grades = np.empty(2, dtype=int)
    cluster_grades[np.argsort(severity)] = np.asarray(grades, dtype=int)
    return {
        "centers": centers,
        "median": median,
        "iqr": iqr,
        "cluster_grades": cluster_grades,
    }
def fit_fuzzy_clusters(daily):
    dates = pd.to_datetime(daily["运行日期"], errors="coerce")
    eligible = dates.dt.year.eq(2025)
    training = daily.loc[eligible].copy()
    maximum = pd.to_numeric(training["日最大NTU"], errors="coerce")
    within_limit = training.loc[maximum.le(NTU_LIMIT)].copy()
    exceedance = training.loc[maximum.gt(NTU_LIMIT)].copy()
    return {
        "within_limit": fit_two_clusters(within_limit, "within_limit", grades=[0, 1]),
        "exceedance": fit_two_clusters(exceedance, "exceedance", grades=[2, 3]),
    }
def classify_q1(daily, model):
    result = daily.copy()
    maximum = pd.to_numeric(result["日最大NTU"], errors="coerce")
    grades = np.zeros(len(result), dtype=int)
    confidence = np.full(len(result), np.nan, dtype=float)
    regimes = np.where(maximum.le(NTU_LIMIT), "NTU≤1", "NTU>1")
    for mask, key in [(maximum.le(NTU_LIMIT), "within_limit"), (maximum.gt(NTU_LIMIT), "exceedance")]:
        positions = np.flatnonzero(mask.fillna(False).to_numpy())
        if not len(positions):
            continue
        part = result.iloc[positions]
        branch = model[key]
        transformed, _, _, _ = transform_features(part, key, median=branch["median"], iqr=branch["iqr"])
        valid = np.isfinite(transformed).all(axis=1)
        distance = np.maximum(np.linalg.norm(transformed[:, None, :] - branch["centers"][None, :, :], axis=2),1e-12)
        membership = 1.0 / (distance[:, :, None] / distance[:, None, :]) ** 2.0
        membership = membership.sum(axis=2) ** -1
        cluster = membership.argmax(axis=1)
        branch_grade = branch["cluster_grades"][cluster]
        grades[positions[valid]] = branch_grade[valid]
        confidence[positions[valid]] = membership[valid].max(axis=1)
    result["风险判别分层"] = regimes
    result["聚类置信度"] = confidence
    result["聚类风险等级"] = grades
    return result
def baseline_grades(frame):
    maximum = pd.to_numeric(frame["日最大NTU"], errors="coerce")
    duration = pd.to_numeric(frame["D"], errors="coerce")
    return np.where(
        maximum.le(NTU_LIMIT),0,
        np.where(maximum.gt(3 * NTU_LIMIT) | duration.gt(6),3,2),
    )
def fused_grades(frame):
    baseline = baseline_grades(frame)
    maximum = pd.to_numeric(frame["日最大NTU"], errors="coerce")
    if "聚类风险等级" in frame:
        cluster = pd.to_numeric(frame["聚类风险等级"], errors="coerce").fillna(0).to_numpy(dtype=int)
    else:
        cluster = np.zeros(len(frame), dtype=int)
    within_limit = maximum.le(NTU_LIMIT).fillna(False).to_numpy()
    final = np.where(within_limit,
        np.clip(cluster, 0, 1),
        np.maximum(baseline, np.clip(cluster, 2, 3)),
    ).astype(int)
    return baseline.astype(int), final
def fuse_grades(daily):
    result = daily.copy()
    baseline, final = fused_grades(result)
    result["基准风险等级"] = baseline.astype(int)
    result["最终风险等级"] = final
    return result
def grade_shares(frame):
    grades = pd.to_numeric(frame["最终风险等级"], errors="coerce").fillna(0).astype(int)
    counts = pd.Series(grades).value_counts().reindex(range(4), fill_value=0).sort_index()
    result = pd.DataFrame({"风险等级": GRADE_NAMES, "天数": counts.to_numpy()})
    result["占比"] = result["天数"] / len(frame) if len(frame) else 0.0
    return result
def get_q1_daily(frame):
    result = frame.copy()
    result["运行日期"] = pd.to_datetime(result["运行日期"])
    result = result.loc[result["运行日期"].between("2026-01-01", "2026-03-31")].sort_values("运行日期")
    return result.reset_index(drop=True)
def plot_out(result, daily_ntu, shares, output_dir):
    set_chinese_style()
    colors = ["#91bfdb", "#fee090", "#fc8d59", "#d73027"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.pie(shares["天数"], labels=shares["风险等级"], autopct="%.1f%%", colors=colors, startangle=90)
    fig.tight_layout()
    fig.savefig(output_dir / f"图1_四级风险天数占比.png", dpi=300, bbox_inches="tight", facecolor="white")

    month_labels = result["运行日期"].dt.month.map(lambda value: f"{value}月")
    monthly = pd.crosstab(month_labels, result["风险等级"]).reindex(index=["1月", "2月", "3月"], columns=GRADE_NAMES, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(monthly))
    for grade, color in zip(GRADE_NAMES, colors):
        ax.bar(monthly.index, monthly[grade], bottom=bottom, label=grade, color=color)
        bottom = bottom + monthly[grade].to_numpy()
    ax.set_xlabel("月份")
    ax.set_ylabel("天数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"图2_月度风险等级分布.png", dpi=300, bbox_inches="tight", facecolor="white")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(daily_ntu.index, daily_ntu.values, color="#2166ac", linewidth=1, marker="o", markersize=2.5)
    ax.axhline(NTU_LIMIT, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlabel("运行日期")
    ax.set_ylabel("日最大NTU")
    right = ax.twinx()
    right.step(result["运行日期"], result["最终风险等级"], where="mid", color="#d73027", linewidth=1)
    right.set_ylabel("风险等级")
    right.set_yticks(range(4))
    fig.tight_layout()
    fig.savefig(output_dir / f"图3_逐日最大NTU与风险等级.png", dpi=300, bbox_inches="tight", facecolor="white")
def write_outputs(daily, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    q1_daily = get_q1_daily(daily)
    daily_ntu = q1_daily.set_index("运行日期")["日最大NTU"]
    result = q1_daily.drop(columns=["日最大NTU", "日均NTU", "NTU标准差", "有效观测数"], errors="ignore")
    result["风险等级"] = (pd.to_numeric(result["最终风险等级"], errors="coerce")
                          .fillna(0).astype(int).map(dict(enumerate(GRADE_NAMES))))
    result_columns = [
        "运行日期", "临界频率", "临界持续率", "M", "F", "D", "L",
        "风险判别分层", "聚类置信度", "聚类风险等级", "基准风险等级",
        "最终风险等级", "风险等级",
    ]
    result = result.reindex(columns=result_columns)
    result.to_csv(output_dir / "2026年第一季度逐日风险分类.csv", index=False, encoding="utf-8-sig")
    shares = grade_shares(result)
    shares.to_csv(output_dir / "四级风险天数占比.csv", index=False, encoding="utf-8-sig")
    march = result.loc[result["运行日期"].dt.month.eq(3)].copy()
    march = march.rename(columns={"运行日期": "日期"})
    march["日期"] = pd.to_datetime(march["日期"]).dt.strftime("%Y-%m-%d")
    march_columns = ["日期", "风险判别分层", "聚类置信度", "基准风险等级", "聚类风险等级", "最终风险等级"]
    with pd.ExcelWriter(output_dir / "2026年3月风险分类.xlsx") as writer:
        march.reindex(columns=march_columns).to_excel(writer, index=False)
    plot_out(result, daily_ntu, shares, output_dir)
def solve(moe_predictor=None, output_dir=OUTPUT_DIR):
    data = load_clean_data()
    full_data = data.copy()
    full_data["timestamp"] = pd.to_datetime(full_data["timestamp"])
    operating_dates = operate_dates(full_data)
    q1_data = full_data.loc[operating_dates.between("2026-01-01", "2026-03-31")].copy()
    training_data = full_data.loc[operating_dates.dt.year.eq(2025)].copy()
    training_predictor = lambda frame: pd.Series(0.0, index=frame.index)
    training_daily = build_daily_features(training_data, training_predictor)
    model = fit_fuzzy_clusters(training_daily)
    q1_daily = build_daily_features(full_data, moe_predictor, target_data=q1_data)
    classified = classify_q1(q1_daily, model)
    final_daily = fuse_grades(classified)
    write_outputs(final_daily, output_dir)
if __name__ == "__main__":
    solve()