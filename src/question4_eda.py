from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "03_问题3" / "models"


def _operating_dates(frame):
    if "operating_date" in frame:
        return pd.to_datetime(frame["operating_date"], errors="coerce").dt.normalize()
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


def build_daily_features(data, moe_predictor=None):
    points = data.copy()
    if "timestamp" not in points:
        points = points.reset_index().rename(columns={points.index.name or "index": "timestamp"})
    points["timestamp"] = pd.to_datetime(points["timestamp"], errors="coerce")
    points = points.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    observed = pd.to_numeric(points["treated_ntu"], errors="coerce")
    missing = observed.isna()
    points["treated_ntu来源"] = np.where(missing, "MoE预测", "实测")
    if missing.any():
        predictor = moe_predictor or _question3_moe_predictor
        predicted = predictor(points.copy())
        if not isinstance(predicted, pd.Series):
            predicted = pd.Series(predicted, index=points.index)
        predicted = pd.to_numeric(predicted.reindex(points.index), errors="coerce")
        observed.loc[missing] = predicted.loc[missing]
    points["treated_ntu"] = observed
    points["运行日期"] = _operating_dates(points)

    rows = []
    for date, group in points.groupby("运行日期", sort=True):
        values = pd.to_numeric(group["treated_ntu"], errors="coerce").dropna()
        exceedance = values.gt(1)
        count = int(exceedance.sum())
        rows.append(
            {
                "运行日期": date,
                "有效观测数": int(values.size),
                "实测观测数": int((group["treated_ntu来源"] == "实测").sum()),
                "MoE预测观测数": int((group["treated_ntu来源"] == "MoE预测").sum()),
                "M": values.max() if values.size else np.nan,
                "F": count / values.size if values.size else np.nan,
                "D": count * 2 if values.size else np.nan,
                "L": _longest_exceedance_run(exceedance.tolist()) * 2 if values.size else np.nan,
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
        & pd.to_numeric(daily["M"], errors="coerce").gt(1)
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
    exceedance = pd.to_numeric(result["M"], errors="coerce").gt(1).to_numpy()
    valid = np.isfinite(values).all(axis=1)
    result["聚类风险等级"] = np.where(exceedance & valid, grade, 0).astype(int)
    return result


def fuse_grades(daily):
    result = daily.copy()
    maximum = pd.to_numeric(result["M"], errors="coerce")
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
