import numpy as np
import pandas as pd
from .data import HORIZONS, RANDOM_STATE, build_origins, make_targets, target_availability
GAP_DAYS = (10, 20, 28)
BLOCK_DAYS = 7
BOOTSTRAP_REPEATS = 200
PATH_DAY_OFFSETS = (0, 3, 6)
PATH_SLOTS = tuple(range(7))
def _column(frame, names):
    for name in names:
        if name in frame:
            return name
    return None
def _prediction_values(long_predictions):
    model = _column(long_predictions, ("model", "模型"))
    horizon = _column(long_predictions, ("horizon", "预测步长/小时"))
    actual = _column(long_predictions, ("actual", "实测值"))
    prediction = _column(long_predictions, ("prediction", "预测值"))
    values = pd.DataFrame(
        {
            "model": long_predictions[model],
            "horizon": pd.to_numeric(long_predictions[horizon], errors="coerce"),
            "actual": pd.to_numeric(long_predictions[actual], errors="coerce"),
            "prediction": pd.to_numeric(long_predictions[prediction], errors="coerce"),
        }
    )
    values = values.loc[
        values["model"].notna()
        & np.isfinite(values["horizon"])
        & np.isfinite(values["actual"])
        & np.isfinite(values["prediction"])
    ].copy()
    horizons = set(values["horizon"].unique())
    ordinal_horizons = set(HORIZONS)
    if horizons.issubset(ordinal_horizons) and any(horizon % 2 for horizon in horizons):
        values["horizon"] = values["horizon"] * 2
    return values
def _metrics(group):
    actual = group["actual"].to_numpy(dtype=float)
    prediction = group["prediction"].to_numpy(dtype=float)
    residual = actual - prediction
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else np.nan,
        "样本数": int(len(group)),
    }
def metric_table(long_predictions):
    values = _prediction_values(long_predictions)
    records = []
    for (model, horizon), group in values.groupby(["model", "horizon"], sort=False):
        record = {"模型": model, "预测步长/小时": int(horizon)}
        record.update(_metrics(group))
        records.append(record)
    return pd.DataFrame(records, columns=["模型", "预测步长/小时", "RMSE", "MAE", "R2", "样本数"])
def _stratum_values(long_predictions, values):
    source = long_predictions.loc[values.index]
    timestamp_name = next((name for name in ("origin", "timestamp") if name in source), None)
    timestamps = pd.to_datetime(source[timestamp_name], errors="coerce") if timestamp_name else pd.Series(pd.NaT, index=source.index)
    if "hour" in source:
        hours = pd.to_numeric(source["hour"], errors="coerce")
    else:
        hours = timestamps.dt.hour
    if "weekday" in source:
        weekdays = pd.to_numeric(source["weekday"], errors="coerce")
    else:
        weekdays = timestamps.dt.weekday
    if "season" in source:
        seasons = source["season"].astype(str)
    else:
        months = timestamps.dt.month
        seasons = pd.Series(np.where(months.isin((5, 6, 7, 8, 9)), "雨季", "旱季"), index=source.index)
    regimes = source["regime"].astype(str)
    return pd.DataFrame(
        {
            "时段": np.where(hours.between(7, 19, inclusive="both"), "07:00--19:00", "夜间"),
            "工作日": np.where(weekdays.between(0, 4, inclusive="both"), "工作日", "周末"),
            "季节": seasons,
            "工况": regimes,
        },
        index=values.index,
    )
def stratified_metric_table(long_predictions):
    values = _prediction_values(long_predictions)
    strata = _stratum_values(long_predictions, values)
    records = []
    for dimension in ("时段", "工作日", "季节", "工况"):
        grouped = pd.concat((values, strata[[dimension]]), axis=1).groupby(
            ["model", "horizon", dimension], sort=False
        )
        for (model, horizon, label), group in grouped:
            record = {
                "分层维度": dimension,
                "分层": label,
                "模型": model,
                "预测步长/小时": int(horizon),
            }
            record.update(_metrics(group))
            records.append(record)
    return pd.DataFrame(
        records,
        columns=["分层维度", "分层", "模型", "预测步长/小时", "RMSE", "MAE", "R2", "样本数"],
    )
def _operating_dates(index):
    return pd.DatetimeIndex(index - pd.Timedelta(hours=7)).normalize()
def _complete_operating_days(frame):
    dates = _operating_dates(frame.index)
    expected_hours = pd.timedelta_range("0h", periods=12, freq="2h")
    complete = []
    target = pd.to_numeric(frame["treated_ntu"], errors="coerce").where(
        target_availability(frame)
    )
    for operating_date in pd.DatetimeIndex(dates.unique()).sort_values():
        expected = operating_date + pd.Timedelta(hours=7) + expected_hours
        if expected.isin(frame.index).all() and target.reindex(expected).notna().all():
            complete.append(operating_date)
    return pd.DatetimeIndex(complete)
def _gap_windows(frame, preferred_starts=None):
    complete = _complete_operating_days(frame)
    windows = []
    cursor = 0
    for gap_number, gap_days in enumerate(GAP_DAYS):
        selected = None
        for start in range(cursor, len(complete) - gap_days + 1):
            dates = complete[start : start + gap_days]
            if preferred_starts is not None and dates[0] < preferred_starts[gap_number]:
                continue
            if not (dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
                continue
            gap_start = dates[0] + pd.Timedelta(hours=7)
            fit_end = gap_start - pd.Timedelta(hours=2)
            if fit_end < frame.index.min():
                continue
            origins = build_origins(frame, frame.index.min(), fit_end)
            if len(origins) < 8:
                continue
            selected = (gap_days, dates, fit_end, origins)
            cursor = start + gap_days
            break
        windows.append(selected)
    return windows
def _mask_gap(frame, gap_dates):
    masked = frame.copy()
    dates = _operating_dates(masked.index)
    hidden = dates.isin(gap_dates)
    masked.loc[hidden, "treated_ntu"] = np.nan
    available = target_availability(masked)
    available.loc[hidden] = False
    masked["target_available"] = available
    masked["missing_treated_ntu"] = ~available
    return masked, hidden
def long_gap_backtest(frame, expert_factories, oof_bundle=None):
    original_target = pd.to_numeric(frame["treated_ntu"], errors="coerce").where(
        target_availability(frame)
    )
    mechanistic_factory, tree_factory, gru_factory = expert_factories
    records = []
    preferred_starts = None
    if oof_bundle is not None:
        preferred_starts = pd.DatetimeIndex(["2025-08-01", "2025-10-01", "2025-12-01"])
    for gap_number, (gap_days, gap_dates, fit_end, train_origins) in enumerate(
        _gap_windows(frame, preferred_starts), start=1
    ):
        masked, hidden = _mask_gap(frame, gap_dates)
        targets = make_targets(masked, train_origins)
        training_label_end = train_origins.max() + pd.Timedelta(hours=2 * max(HORIZONS))
        mechanistic = mechanistic_factory().fit(masked, fit_end)
        filled_target = mechanistic.fill_target_history(masked)
        tree = tree_factory().fit(masked, train_origins, targets)
        gru = gru_factory().fit(masked, train_origins, targets, filled_target)
        tree.available_target_end_ = fit_end
        gru.available_target_end_ = fit_end

        gap_start = masked.index[hidden].min()
        gap_end = masked.index[hidden].max()
        origins = masked.index[(masked.index >= gap_start) & (masked.index <= gap_end - pd.Timedelta(hours=12))]
        mechanistic_prediction = np.maximum(
            mechanistic.predict(masked, origins), 0.0
        )
        tree_prediction = np.maximum(
            tree.predict(masked, origins, filled_target), 0.0
        )
        gru_prediction = np.maximum(
            gru.predict(masked, origins, filled_target), 0.0
        )
        expert_predictions = np.stack(
            (mechanistic_prediction, tree_prediction, gru_prediction), axis=2
        )
        gate_source = "prior OOF"
        if oof_bundle is None:
            moe_prediction = expert_predictions.mean(axis=2)
            gate_source = "uniform fixture fallback"
        else:
            from .moe import SoftmaxGate, _expand_gate_features, _gate_feature_matrix

            eligible = oof_bundle.origins < gap_start
            gap_gate = SoftmaxGate().fit(
                oof_bundle.expert_predictions[eligible],
                oof_bundle.gate_features[eligible],
                oof_bundle.targets[eligible],
            )
            base_features = _gate_feature_matrix(
                masked, fit_end, origins, filled_target
            )
            gate_features = _expand_gate_features(base_features, expert_predictions)
            moe_prediction = np.maximum(
                gap_gate.predict(expert_predictions, gate_features), 0.0
            )

        seasonal_history = pd.to_numeric(masked["treated_ntu"], errors="coerce").where(
            target_availability(masked)
        ).copy()
        fallback = float(seasonal_history.loc[:fit_end].dropna().iloc[-1])
        for timestamp in masked.index[hidden]:
            prior = timestamp - pd.Timedelta(hours=24)
            value = seasonal_history.get(prior, np.nan)
            seasonal_history.loc[timestamp] = fallback if not np.isfinite(value) else value
        seasonal_prediction = np.empty((len(origins), len(HORIZONS)), dtype=float)
        for origin_number, origin in enumerate(origins):
            for horizon_number, horizon in enumerate(HORIZONS):
                seasonal_prediction[origin_number, horizon_number] = max(
                    0.0,
                    float(seasonal_history.loc[origin + pd.Timedelta(hours=2 * horizon)]),
                )
        prediction_sets = (
            ("季节朴素", seasonal_prediction),
            ("机理专家", mechanistic_prediction),
            ("LightGBM", tree_prediction),
            ("GRU", gru_prediction),
            ("MoE", moe_prediction),
        )
        for model, predictions in prediction_sets:
            values = np.asarray(predictions, dtype=float)
            for origin_number, origin in enumerate(origins):
                for horizon_number, horizon in enumerate(HORIZONS):
                    target_timestamp = origin + pd.Timedelta(hours=2 * horizon)
                    records.append(
                        {
                            "gap_id": gap_number,
                            "gap_days": gap_days,
                            "gap_start": gap_start,
                            "gap_end": gap_end,
                            "gap_start_operating_date": gap_dates[0],
                            "gap_end_operating_date": gap_dates[-1],
                            "masked_points": int(hidden.sum()),
                            "fit_end": fit_end,
                            "training_origin_count": int(len(train_origins)),
                            "training_label_end": training_label_end,
                            "gate_source": gate_source,
                            "model": model,
                            "origin": origin,
                            "horizon": horizon * 2,
                            "target_timestamp": target_timestamp,
                            "actual": float(original_target.loc[target_timestamp]),
                            "prediction": float(values[origin_number, horizon_number]),
                        }
                    )
    return pd.DataFrame(records)


def _residual_blocks(oof_residuals):
    date_name = _column(oof_residuals, ("operating_date", "运行日期"))
    residual_name = _column(oof_residuals, ("residual", "残差"))
    values = oof_residuals.copy()
    values["_input_order"] = np.arange(len(values))
    values["_operating_date"] = pd.to_datetime(values[date_name], errors="coerce").dt.normalize()
    values["_residual"] = pd.to_numeric(values[residual_name], errors="coerce")

    time_name = "target_timestamp" if "target_timestamp" in values else "timestamp"
    clock_name = "clock" if "clock" in values else ("time" if "time" in values else None)
    if time_name in values:
        timestamps = pd.to_datetime(values[time_name], errors="coerce")
        start = values["_operating_date"] + pd.Timedelta(hours=7)
        values["_slot"] = ((timestamps - start).dt.total_seconds() / 60).round()
    elif clock_name is not None:
        numeric = pd.to_numeric(values[clock_name], errors="coerce")
        if numeric.notna().all():
            minutes = numeric * 60 if numeric.abs().max() <= 24 else numeric
        else:
            converted = pd.to_timedelta(values[clock_name].astype(str), errors="coerce")
            minutes = converted.dt.total_seconds() / 60
        values["_slot"] = (np.asarray(minutes, dtype=float) - 420) % 1440
    else:
        values["_slot"] = values.groupby("_operating_date", sort=False).cumcount() * 120

    grouped = []
    for date, group in values.groupby("_operating_date", sort=True):
        group = group.sort_values("_slot")
        grouped.append((date, group["_residual"].to_numpy(dtype=float)))
    blocks = []
    run_start = 0
    for stop in range(1, len(grouped) + 1):
        continued = stop < len(grouped) and grouped[stop][0] - grouped[stop - 1][0] == pd.Timedelta(days=1)
        if continued:
            continue
        if stop - run_start >= 40:
            for start in range(run_start, stop - BLOCK_DAYS + 1):
                blocks.append(
                    np.concatenate(
                        [grouped[start + offset][1][list(PATH_SLOTS)] for offset in PATH_DAY_OFFSETS]
                    )
                )
        run_start = stop
    return blocks
def residual_block_intervals(oof_residuals, point):
    values = np.asarray(point, dtype=float).reshape(-1)
    blocks = _residual_blocks(oof_residuals)
    generator = np.random.default_rng(RANDOM_STATE)
    paths = np.asarray(blocks, dtype=float)
    centers = np.median(paths, axis=0)
    samples = np.empty((BOOTSTRAP_REPEATS, len(values)), dtype=float)
    for repeat in range(BOOTSTRAP_REPEATS):
        samples[repeat] = paths[int(generator.integers(len(paths)))] - centers
    lower_80, upper_80 = np.quantile(samples, (0.10, 0.90), axis=0)
    lower_95, upper_95 = np.quantile(samples, (0.025, 0.975), axis=0)
    lower_80, upper_80 = np.minimum(lower_80, 0.0), np.maximum(upper_80, 0.0)
    lower_95, upper_95 = np.minimum(lower_95, 0.0), np.maximum(upper_95, 0.0)
    return pd.DataFrame(
        {
            "序号": np.arange(1, len(values) + 1),
            "预测值": values,
            "80%下限": values + lower_80,
            "80%上限": values + upper_80,
            "95%下限": values + lower_95,
            "95%上限": values + upper_95,
            "重复次数": BOOTSTRAP_REPEATS,
        }
    )
