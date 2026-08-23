from pathlib import Path

import numpy as np
import pandas as pd


GRADE_NAMES = ["安全", "低风险", "中风险", "高风险"]


def _operating_dates(frame):
    return (frame["timestamp"] - pd.Timedelta(hours=7)).dt.normalize()


def _longest_exceedance_run(exceedance):
    longest = 0
    current = 0
    for value in exceedance:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _question3_moe_predictor(frame):
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


def _daily_grid(data):
    points = data.copy()
    if "timestamp" not in points:
        points = points.reset_index().rename(columns={points.index.name or "index": "timestamp"})
    points["timestamp"] = pd.to_datetime(points["timestamp"], errors="coerce")
    points = points.dropna(subset=["timestamp"]).sort_values("timestamp")
    points = points.drop_duplicates("timestamp", keep="last")
    dates = _operating_dates(points).drop_duplicates().sort_values()
    timestamps = []
    for date in dates:
        timestamps.extend(pd.date_range(date + pd.Timedelta(hours=7), periods=12, freq="2h"))
    grid = pd.DataFrame({"timestamp": timestamps})
    grid["运行日期"] = _operating_dates(grid)
    return grid.merge(points.drop(columns=["operating_date"], errors="ignore"), on="timestamp", how="left")


def build_daily_features(data, moe_predictor=None, moe_context=None, target_data=None):
    points = _daily_grid(data if target_data is None else target_data)
    context = _daily_grid(data if moe_context is None else moe_context)
    observed = pd.to_numeric(points["treated_ntu"], errors="coerce")
    missing = observed.isna()
    predicted = pd.Series(np.nan, index=points.index, dtype=float)
    if missing.any():
        predictor = moe_predictor or _question3_moe_predictor
        context_prediction = predictor(context.copy())
        if not isinstance(context_prediction, pd.Series):
            context_prediction = pd.Series(context_prediction, index=context.index)
        context_prediction = pd.to_numeric(context_prediction.reindex(context.index), errors="coerce")
        by_timestamp = pd.Series(context_prediction.to_numpy(), index=context["timestamp"])
        predicted = pd.to_numeric(points["timestamp"].map(by_timestamp), errors="coerce")
        observed.loc[missing] = predicted.loc[missing]
    predicted_ok = missing & predicted.notna()
    if (missing & ~predicted_ok).any():
        raise ValueError("目标评价范围存在无法由MoE回填的treated_ntu")
    points["treated_ntu"] = observed
    points["treated_ntu来源"] = np.where(
        missing,
        np.where(predicted_ok, "MoE预测", "缺失"),
        "实测",
    )

    rows = []
    for date, group in points.groupby("运行日期", sort=True):
        values = pd.to_numeric(group["treated_ntu"], errors="coerce")
        exceedance = values.gt(1) & values.notna()
        count = int(exceedance.sum())
        excess = (values - 1).clip(lower=0)
        rows.append(
            {
                "运行日期": date,
                "有效观测数": int(values.notna().sum()),
                "实测观测数": int((group["treated_ntu来源"] == "实测").sum()),
                "MoE预测观测数": int((group["treated_ntu来源"] == "MoE预测").sum()),
                "日最大NTU": values.max() if values.notna().any() else np.nan,
                "M": excess.max() if values.notna().any() else np.nan,
                "F": count / 12,
                "D": _longest_exceedance_run(exceedance.tolist()) * 2,
                "L": excess.sum(skipna=True) * 2,
            }
        )
    daily = pd.DataFrame(rows).sort_values("运行日期").reset_index(drop=True)
    return points, daily


def _cluster_features(frame):
    values = frame[["M", "F", "D", "L"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values[:, 0] = np.log1p(values[:, 0])
    values[:, 2] = np.log1p(values[:, 2])
    values[:, 3] = np.log1p(values[:, 3])
    return values


def _fuzzy_c_means(values, clusters, seed):
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


def fit_fuzzy_clusters(daily):
    dates = pd.to_datetime(daily["运行日期"], errors="coerce")
    eligible = (
        dates.dt.year.eq(2025)
        & daily["有效观测数"].eq(12)
        & daily["实测观测数"].eq(12)
        & pd.to_numeric(daily["M"], errors="coerce").gt(0)
    )
    training = daily.loc[eligible, ["M", "F", "D", "L"]].copy()
    if len(training) < 3:
        raise ValueError("2025年完整实测超标日不足3天，无法进行三类模糊聚类")
    transformed = _cluster_features(training)
    median = np.median(transformed, axis=0)
    lower = np.quantile(transformed, 0.25, axis=0)
    upper = np.quantile(transformed, 0.75, axis=0)
    iqr = upper - lower
    iqr[iqr < 1e-12] = 1.0
    scaled = (transformed - median) / iqr
    centers, membership = _fuzzy_c_means(scaled, 3, 2026)
    original_centers = centers * iqr + median
    original_centers[:, 0] = np.expm1(original_centers[:, 0])
    original_centers[:, 2] = np.expm1(original_centers[:, 2])
    original_centers[:, 3] = np.expm1(original_centers[:, 3])
    severity = original_centers[:, 0] + original_centers[:, 1] + original_centers[:, 2] + original_centers[:, 3]
    ranks = np.empty(3, dtype=int)
    ranks[np.argsort(severity)] = np.arange(1, 4)
    return {
        "centers": centers,
        "median": median,
        "iqr": iqr,
        "cluster_ranks": ranks,
        "train_rows": len(training),
        "membership": membership,
    }


def classify_q1(daily, model):
    result = daily.copy()
    values = _cluster_features(result)
    scaled = (values - model["median"]) / model["iqr"]
    distance = np.linalg.norm(scaled[:, None, :] - model["centers"][None, :, :], axis=2)
    distance = np.maximum(distance, 1e-12)
    membership = 1.0 / (distance[:, :, None] / distance[:, None, :]) ** 2.0
    membership = membership.sum(axis=2) ** -1
    cluster = membership.argmax(axis=1)
    grade = model["cluster_ranks"][cluster]
    exceedance = pd.to_numeric(result["M"], errors="coerce").gt(0).to_numpy()
    valid = np.isfinite(values).all(axis=1)
    result["聚类风险等级"] = np.where(exceedance & valid, grade, 0).astype(int)
    return result


def fuse_grades(daily):
    result = daily.copy()
    maximum = pd.to_numeric(result["日最大NTU"], errors="coerce")
    duration = pd.to_numeric(result["D"], errors="coerce")
    baseline = np.where(
        maximum.le(1),
        0,
        np.where(
            maximum.gt(3) | duration.gt(6),
            3,
            np.where(maximum.gt(2) | duration.gt(2), 2, 1),
        ),
    )
    if "聚类风险等级" in result:
        cluster = pd.to_numeric(result["聚类风险等级"], errors="coerce").fillna(0).to_numpy(dtype=int)
    else:
        cluster = np.zeros(len(result), dtype=int)
    result["基准风险等级"] = baseline.astype(int)
    result["最终风险等级"] = np.where(baseline == 0, 0, np.maximum(baseline, cluster)).astype(int)
    return result


def _classification_basis(frame):
    baseline = pd.to_numeric(frame["基准风险等级"], errors="coerce").fillna(0).astype(int)
    cluster = pd.to_numeric(frame["聚类风险等级"], errors="coerce").fillna(0).astype(int)
    final = pd.to_numeric(frame["最终风险等级"], errors="coerce").fillna(0).astype(int)
    return np.where(
        final.eq(0),
        "国标内",
        np.where(
            baseline.eq(cluster),
            "阈值与聚类一致",
            np.where(final.gt(baseline), "聚类预警升级", "阈值下限主导"),
        ),
    )


def _grade_shares(frame, multiplier=None):
    if multiplier is None:
        grades = pd.to_numeric(frame["最终风险等级"], errors="coerce").fillna(0).astype(int)
    else:
        maximum = pd.to_numeric(frame["日最大NTU"], errors="coerce")
        duration = pd.to_numeric(frame["D"], errors="coerce")
        grades = np.where(
            maximum.le(1),
            0,
            np.where(
                maximum.gt(3 * multiplier) | duration.gt(6 * multiplier),
                3,
                np.where(maximum.gt(2 * multiplier) | duration.gt(2 * multiplier), 2, 1),
            ),
        )
    counts = pd.Series(grades).value_counts().reindex(range(4), fill_value=0).sort_index()
    result = pd.DataFrame({"风险等级": GRADE_NAMES, "天数": counts.to_numpy()})
    result["占比"] = result["天数"] / len(frame) if len(frame) else 0.0
    if multiplier is not None:
        result.insert(0, "阈值倍率", multiplier)
    return result


def write_outputs(points, daily, model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = points.copy()
    source.to_csv(output_dir / "逐时来源.csv", index=False, encoding="utf-8-sig")

    result = daily.copy()
    result["运行日期"] = pd.to_datetime(result["运行日期"])
    result = result.sort_values("运行日期").reset_index(drop=True)
    result["分类依据"] = _classification_basis(result)
    result["风险等级"] = pd.to_numeric(result["最终风险等级"], errors="coerce").fillna(0).astype(int).map(
        dict(enumerate(GRADE_NAMES))
    )
    result.to_csv(output_dir / "2026年第一季度逐日风险分类.csv", index=False, encoding="utf-8-sig")

    shares = _grade_shares(result)
    shares.to_csv(output_dir / "四级风险天数占比.csv", index=False, encoding="utf-8-sig")

    march_dates = pd.DataFrame({"运行日期": pd.date_range("2026-03-01", periods=31, freq="D")})
    march = march_dates.merge(result, on="运行日期", how="left")
    march = march.rename(columns={"运行日期": "日期"})
    march_columns = [
        "日期", "M", "F", "D", "L", "实测观测数", "MoE预测观测数",
        "基准风险等级", "聚类风险等级", "最终风险等级", "分类依据",
    ]
    march = march.reindex(columns=march_columns)
    with pd.ExcelWriter(output_dir / "2026年3月逐日风险分类.xlsx") as writer:
        march.to_excel(writer, sheet_name="3月逐日分类", index=False)

    centers = np.asarray(model.get("centers", np.empty((0, 4))))
    ranks = np.asarray(model.get("cluster_ranks", np.arange(1, len(centers) + 1)))
    center_table = pd.DataFrame(centers, columns=["M", "F", "D", "L"])
    center_table.insert(0, "聚类", np.arange(1, len(center_table) + 1))
    center_table["风险序号"] = ranks
    k_table = pd.DataFrame({"K值": [2, 3, 4]})
    sensitivity = pd.concat([_grade_shares(result, multiplier) for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]], ignore_index=True)
    with pd.ExcelWriter(output_dir / "风险评价诊断.xlsx") as writer:
        center_table.to_excel(writer, sheet_name="聚类中心", index=False)
        k_table.to_excel(writer, sheet_name="K值比较", index=False)
        sensitivity.to_excel(writer, sheet_name="阈值敏感性", index=False)
