import numpy as np
import pandas as pd
class TemporalFold:
    def __init__(self, valid_start, valid_end):
        self.valid_start = valid_start
        self.valid_end = valid_end
    @property
    def train_end(self):
        return self.valid_start - pd.Timedelta(hours=2)
FREQUENCY = "2h"
RESAMPLE_OPTIONS = {"origin": "start_day", "offset": "1h", "label": "left", "closed": "left"}
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

CORE_COLUMNS = (
    "raw_water_ntu",
    "raw_water_ph",
    "filtered_ntu",
    "clear_well_level",
    "treated_ntu",
    "alum_feed_rate",
    "alum_dosage",
    "raw_water_flow",
    "treated_water_flow",
)
LAG_STEPS = (1, 2, 3, 6, 12, 24)
ROLLING_WINDOWS = (3, 6, 12, 24)
CHANGE_STEPS = (1, 3, 6)


def target_availability(frame):
    if "target_available" in frame:
        return frame["target_available"].fillna(False).astype(bool)
    target = pd.to_numeric(frame["treated_ntu"], errors="coerce")
    if "missing_treated_ntu" in frame:
        return (~frame["missing_treated_ntu"].fillna(True).astype(bool)) & target.notna()
    return target.notna()
def _resample_column(series):
    if pd.api.types.is_bool_dtype(series):
        return series.resample(FREQUENCY, **RESAMPLE_OPTIONS).max()
    if pd.api.types.is_numeric_dtype(series):
        return series.resample(FREQUENCY, **RESAMPLE_OPTIONS).median()
    return series.resample(FREQUENCY, **RESAMPLE_OPTIONS).last()
class FeatureFrame(pd.DataFrame):
    _metadata = ["fit_end"]
    @property
    def _constructor(self):
        return FeatureFrame
    def fit_rows(self, rows=None):
        boundary = pd.Timestamp(self.fit_end)
        if rows is None:
            return pd.DataFrame(self.loc[self.index <= boundary]).copy()
        requested = pd.DatetimeIndex(rows)
        # if (requested > boundary).any():
        #     raise ValueError("fitted-statistic rows must not be after fit_end")
        return pd.DataFrame(self.loc[requested]).copy()


def prepare_q3_frame(data):
    # if "timestamp" not in data.columns:
    #     raise KeyError("data must contain a timestamp column")

    prepared = data.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared = prepared.dropna(subset=["timestamp"]).sort_values("timestamp")
    prepared = prepared.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    prepared.index.name = "timestamp"

    if "treated_ntu" in prepared:
        numeric_target = pd.to_numeric(prepared["treated_ntu"], errors="coerce")
        if "target_available" in prepared:
            available = prepared["target_available"].fillna(False).astype(bool)
        elif "missing_treated_ntu" in prepared:
            available = ~prepared["missing_treated_ntu"].fillna(True).astype(bool)
        else:
            available = numeric_target.notna()
        prepared["target_available"] = (available & numeric_target.notna()).astype(bool)

    if prepared.empty:
        empty = prepared.copy()
        empty.index = pd.DatetimeIndex([], name="timestamp")
        return empty

    original_columns = list(prepared.columns)
    missing_sources = [
        column
        for column in original_columns
        if not column.startswith("missing_")
        and column not in ("is_backwash_event", "target_available")
    ]
    raw_missing = {
        f"missing_{column}": prepared[column].isna().astype(bool)
        for column in missing_sources
    }

    regular = pd.DataFrame(
        {column: _resample_column(prepared[column]) for column in original_columns}
    )
    for flag, values in raw_missing.items():
        was_missing = (
            values.resample(FREQUENCY, **RESAMPLE_OPTIONS)
            .max()
            .reindex(regular.index)
            .fillna(True)
        )
        if flag in regular:
            was_missing = was_missing | regular[flag].fillna(True).astype(bool)
        regular[flag] = was_missing.astype(bool)

    if "is_backwash_event" in regular:
        regular["is_backwash_event"] = regular["is_backwash_event"].fillna(False).astype(bool)
    if "target_available" in regular:
        regular["target_available"] = regular["target_available"].fillna(False).astype(bool)
        regular["missing_treated_ntu"] = ~regular["target_available"]

    regular.index.name = "timestamp"
    return regular.sort_index()


# def _require_datetime_index(frame):
#     if not isinstance(frame.index, pd.DatetimeIndex):
#         raise TypeError("frame must use a DatetimeIndex")
#     if not frame.index.is_unique:
#         raise ValueError("frame index must be unique")


def make_targets(frame, origins):
    # """Build six direct targets at fixed two-hour horizons for each origin."""
    # _require_datetime_index(frame)
    # if "treated_ntu" not in frame:
    #     raise KeyError("frame must contain treated_ntu")

    origin_index = pd.DatetimeIndex(origins)
    target_index = pd.DataFrame(
        {
            horizon: origin_index + pd.Timedelta(hours=2 * horizon)
            for horizon in HORIZONS
        },
        index=origin_index,
    )
    target = pd.to_numeric(frame["treated_ntu"], errors="coerce").where(
        target_availability(frame)
    )
    return np.column_stack(
        [target.reindex(target_index[horizon]).to_numpy() for horizon in HORIZONS]
    ) if len(origin_index) else np.empty((0, len(HORIZONS)), dtype=float)


def build_origins(frame, start, end):
    # """Select origins whose 48-hour history and six labels end by ``end``."""
    # _require_datetime_index(frame)
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    # if end_timestamp < start_timestamp:
    #     raise ValueError("end must not precede start")

    latest_origin = end_timestamp - pd.Timedelta(hours=2 * max(HORIZONS))
    candidates = frame.index[(frame.index >= start_timestamp) & (frame.index <= latest_origin)]
    if not len(candidates):
        return pd.DatetimeIndex([], name=frame.index.name)

    positions = frame.index.get_indexer(candidates)
    has_history = positions >= HISTORY_STEPS
    targets = make_targets(frame, candidates)
    complete_target = np.isfinite(targets).all(axis=1)
    return pd.DatetimeIndex(candidates[has_history & complete_target], name=frame.index.name)


def _prior_missing_run(missing):
    # """Count consecutive target gaps ending immediately before each timestamp."""
    result = np.empty(len(missing), dtype=int)
    run = 0
    for position, is_missing in enumerate(missing.shift(1, fill_value=False).to_numpy(bool)):
        if is_missing:
            run += 1
        else:
            run = 0
        result[position] = run
    return pd.Series(result, index=missing.index, name="target_missing_run")


def make_feature_frame(frame, fit_end):
    # """Create timestamp-causal features with a fitted-statistics boundary.
    #
    # This operation has no fitted imputer or scaler.  Missing values remain as
    # missing values and their indicators are emitted.  The returned
    # :class:`FeatureFrame` keeps all prediction rows, while ``fit_rows()`` is the
    # explicit, guarded interface for any fold-fitted statistic or transform.
    # """
    # _require_datetime_index(frame)
    fit_end = pd.Timestamp(fit_end)
    feature_values = {}

    hour = frame.index.hour + frame.index.minute / 60.0
    weekday = frame.index.weekday
    month = frame.index.month
    feature_values["hour"] = hour
    feature_values["weekday"] = weekday
    feature_values["month"] = month
    feature_values["is_rainy_season"] = month.isin((5, 6, 7, 8, 9)).astype(int)
    feature_values["day_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feature_values["day_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feature_values["week_sin"] = np.sin(2 * np.pi * (weekday * 24.0 + hour) / (7 * 24.0))
    feature_values["week_cos"] = np.cos(2 * np.pi * (weekday * 24.0 + hour) / (7 * 24.0))

    for column in CORE_COLUMNS:
        if column not in frame:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        missing_name = f"missing_{column}"
        if column == "treated_ntu":
            available = target_availability(frame)
            series = series.where(available)
            missing = ~available
        else:
            missing = frame[missing_name].astype(bool) if missing_name in frame else series.isna()
        feature_values[missing_name] = missing.astype(int)
        if column != "treated_ntu":
            feature_values[column] = series
        for lag in LAG_STEPS:
            feature_values[f"{column}_lag{lag}"] = series.shift(lag)
        for delta in CHANGE_STEPS:
            feature_values[f"{column}_change{delta}"] = series.diff(delta)
        historical = series.shift(1)
        for window in ROLLING_WINDOWS:
            feature_values[f"{column}_rollmean{window}"] = historical.rolling(
                window, min_periods=1
            ).mean()
            feature_values[f"{column}_rollstd{window}"] = historical.rolling(
                window, min_periods=2
            ).std()

    if "is_backwash_event" in frame:
        feature_values["is_backwash_event"] = frame["is_backwash_event"].fillna(False).astype(int)
    if "treated_ntu" in frame:
        target_missing = ~target_availability(frame)
        feature_values["target_missing_run"] = _prior_missing_run(target_missing)

    features = FeatureFrame(feature_values, index=frame.index)
    features.index.name = frame.index.name
    features.fit_end = fit_end

    # Attribute documents the only admissible fitting boundary for consumers that
    # subsequently fit an imputer/scaler to this feature matrix.
    features.attrs["fit_end"] = fit_end
    return features
