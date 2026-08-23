import calendar
import re

import numpy as np
import pandas as pd

from utils import (FULL_FIELDS, SHORT_2025_FIELDS, NUMERIC_COLUMNS,
                   find_month_from_name, excel_serial_to_date, ROOT, CLEAN_DATA)

DATA_DIR = ROOT / "附件"
FULL_2025_FIELDS = ["date_raw"] + FULL_FIELDS
TARGET_COLUMNS = ["filtered_ntu", "treated_ntu"]

def repair_2025_date(value, file_month):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (int, float, np.integer, np.floating)):
        parsed = excel_serial_to_date(value)
    elif isinstance(value, (pd.Timestamp, np.datetime64)):
        parsed = pd.Timestamp(value)
    else:
        text = str(value).strip()
        matched = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$", text)
        if matched:
            day = int(matched.group(1))
            return pd.Timestamp(year=2025, month=file_month, day=day)
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    if parsed.month == file_month:
        day = parsed.day
    elif parsed.day == file_month:
        day = parsed.month
    else:
        return pd.NaT
    last_day = calendar.monthrange(2025, file_month)[1]
    if not 1 <= day <= last_day:
        return pd.NaT
    return pd.Timestamp(year=2025, month=file_month, day=day)
def parse_clock(value):
    if pd.isna(value):
        return np.nan, np.nan
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return int(value.hour), int(value.minute)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if 0 <= float(value) < 1:
            total_minutes = int(round(float(value) * 24 * 60)) % (24 * 60)
            return total_minutes // 60, total_minutes % 60
        text = str(int(round(float(value))))
    else:
        text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        return int(float(parts[0])), int(float(parts[1]))
    digits = re.sub(r"\D", "", text)
    if not digits:
        return np.nan, np.nan
    digits = digits.zfill(4)[-4:]
    hour = int(digits[:2])
    minute = int(digits[2:])
    if minute >= 60:
        try:
            numeric_hour = float(text)
            if 0 <= numeric_hour < 24:
                return int(numeric_hour), 0
        except ValueError:
            pass
        return np.nan, np.nan
    if hour >= 24:
        return np.nan, np.nan
    return hour, minute
def build_timestamp(base_date, time_value):
    hour, minute = parse_clock(time_value)
    if pd.isna(base_date) or pd.isna(hour):
        return pd.NaT
    timestamp = pd.Timestamp(base_date) + pd.Timedelta(hours=int(hour), minutes=int(minute))
    if hour < 7:
        timestamp += pd.Timedelta(days=1)
    return timestamp
def build_group_timestamps(group):
    expected_hours = [7, 9, 11, 13, 15, 17, 19, 21, 23, 1, 3, 5]
    base_date = group["base_date"].iloc[0]
    if len(group) == 12 and pd.notna(base_date):
        timestamps = []
        for hour in expected_hours:
            timestamp = pd.Timestamp(base_date) + pd.Timedelta(hours=hour)
            if hour < 7:
                timestamp += pd.Timedelta(days=1)
            timestamps.append(timestamp)
        return pd.Series(timestamps, index=group.index)
    return pd.Series(
        [build_timestamp(date, time) for date, time in zip(group["base_date"], group["time_raw"])],
        index=group.index,
    )
def assign_group_timestamps(data):
    timestamps = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    for _, group in data.groupby("base_date", sort=False, dropna=False):
        timestamps.loc[group.index] = build_group_timestamps(group)
    return timestamps
def assign_schema(raw, schema, source_name, source_sheet=""):
    body = raw.iloc[1:, :len(schema)].copy()
    body.columns = schema
    body = body.dropna(how="all")
    body["source_file"] = source_name
    body["source_sheet"] = source_sheet
    available = set(schema) - {"date_raw", "time_raw"}
    for column in FULL_FIELDS:
        if column not in body.columns:
            body[column] = np.nan
        body[f"available_{column}"] = column in available
    return body
def read_2025_file(path):
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    if raw.shape[1] >= 21:
        schema = FULL_2025_FIELDS
    elif raw.shape[1] == 16:
        schema = SHORT_2025_FIELDS
    data = assign_schema(raw, schema, path.name)
    file_month = find_month_from_name(path)
    data["base_date"] = data["date_raw"].apply(repair_2025_date, file_month=file_month)
    date_keys = data["date_raw"].ffill().astype(str)
    day_order = pd.factorize(date_keys, sort=False)[0] + 1
    valid_day = day_order <= calendar.monthrange(2025, file_month)[1]
    fallback_dates = pd.to_datetime(
        {
            "year": np.full(len(data), 2025),
            "month": np.full(len(data), file_month),
            "day": np.where(valid_day, day_order, 1),
        },
        errors="coerce",
    )
    missing_date = data["base_date"].isna().to_numpy() & valid_day
    data.loc[missing_date, "base_date"] = fallback_dates.to_numpy()[missing_date]
    data["timestamp"] = assign_group_timestamps(data)
    return data
def read_2026_file(path):
    year_match = re.search(r"(20\d{2})", path.name)
    year = int(year_match.group(1))
    pieces = []
    workbook = pd.ExcelFile(path, engine="xlrd")
    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="xlrd")
        data = assign_schema(raw, FULL_FIELDS, path.name, sheet_name.strip())
        matched = re.search(r"(\d{1,2})\.(\d{1,2})", sheet_name)
        if not matched:
            continue
        day = int(matched.group(1))
        month = int(matched.group(2))
        base_date = pd.Timestamp(year=year, month=month, day=day)
        data["base_date"] = base_date
        data["timestamp"] = assign_group_timestamps(data)
        pieces.append(data)
    return pd.concat(pieces, ignore_index=True)
def count_running_pumps(value):
    if pd.isna(value):
        return np.nan
    numbers = re.findall(r"\d+", str(value))
    return len(set(numbers)) if numbers else np.nan
def join_text(values):
    cleaned = []
    for value in values.dropna():
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in cleaned:
            cleaned.append(text)
    return "；".join(cleaned) if cleaned else np.nan
def aggregate_duplicate_timestamps(data):
    data = data.copy()
    data = data.replace({"": np.nan, "-": np.nan, "--": np.nan, "N/A": np.nan, "n/a": np.nan})
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    duplicated = data["timestamp"].duplicated(keep=False)
    if not duplicated.any():
        return data, pd.DataFrame(columns=data.columns)
    duplicate_rows = data.loc[duplicated].copy()
    rules = {}
    for column in data.columns:
        if column == "timestamp":
            continue
        if column in NUMERIC_COLUMNS or column.startswith("available_"):
            rules[column] = "median" if column in NUMERIC_COLUMNS else "max"
        elif column == "base_date":
            rules[column] = "min"
        else:
            rules[column] = join_text
    data = data.groupby("timestamp", as_index=False).agg(rules)
    return data, duplicate_rows
def mark_robust_outliers(series, window=37, threshold=6):
    rolling_median = series.rolling(window, center=True, min_periods=9).median()
    absolute_deviation = (series - rolling_median).abs()
    rolling_mad = absolute_deviation.rolling(window, center=True, min_periods=9).median()
    scale = 1.4826 * rolling_mad
    return (absolute_deviation > threshold * scale) & scale.gt(0)
def clean_values(data):
    missing_tokens = {"": np.nan, "-": np.nan, "--": np.nan, "N/A": np.nan, "n/a": np.nan}
    data = data.replace(missing_tokens)
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["raw_water_pump_count"] = data["raw_water_pump_duty"].apply(count_running_pumps)
    data["treated_water_pump_count"] = data["treated_water_pump_duty"].apply(count_running_pumps)
    nonnegative = NUMERIC_COLUMNS + ["raw_water_pump_count", "treated_water_pump_count"]
    for column in nonnegative:
        data.loc[data[column] < 0, column] = np.nan
    for column in ["raw_water_ph", "treated_ph"]:
        data.loc[~data[column].between(0, 14) & data[column].notna(), column] = np.nan
    remarks = data["remarks"].fillna("").astype(str)
    event_pattern = r"B\s*/\s*W|BACK\s*WASH|BACKWASH|FILTER"
    data["is_backwash_event"] = remarks.str.contains(event_pattern, case=False, regex=True)
    data["is_treated_ntu_exceedance"] = data["treated_ntu"].gt(1)
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["operating_date"] = (data["timestamp"] - pd.Timedelta(hours=7)).dt.date
    data["year"] = data["timestamp"].dt.year
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["hour"] = data["timestamp"].dt.hour
    data["weekday"] = data["timestamp"].dt.weekday
    data["sampling_gap_hours"] = data["timestamp"].diff().dt.total_seconds().div(3600)
    for column in NUMERIC_COLUMNS:
        data[f"missing_{column}"] = data[column].isna()
    predictors = [column for column in NUMERIC_COLUMNS if column not in TARGET_COLUMNS]
    for column in predictors:
        pieces = []
        for _, group in data.groupby("source_file", sort=False, dropna=False):
            values = group.set_index("timestamp")[column]
            values = values.interpolate(method="time", limit=3, limit_area="inside")
            pieces.append(values.rename(column).reset_index())
        filled = pd.concat(pieces, ignore_index=True)
        value_map = filled.drop_duplicates("timestamp").set_index("timestamp")[column]
        data[column] = data["timestamp"].map(value_map).combine_first(data[column])
    outlier_columns = NUMERIC_COLUMNS + ["raw_water_pump_count", "treated_water_pump_count"]
    for column in outlier_columns:
        data[f"outlier_{column}"] = mark_robust_outliers(data[column])
    return data
def load_all_data():
    xlsx_files = sorted(
        path for path in DATA_DIR.rglob("*.xlsx") if not path.name.startswith("~$")
    )
    xls_files = sorted(
        path for path in DATA_DIR.rglob("*.xls") if not path.name.startswith("~$")
    )
    pieces = [read_2025_file(path) for path in xlsx_files]
    pieces.extend(read_2026_file(path) for path in xls_files)
    data = pd.concat(pieces, ignore_index=True, sort=False)
    data = data.dropna(subset=["timestamp"])
    data, duplicate_rows = aggregate_duplicate_timestamps(data)
    data = clean_values(data)
    return data, duplicate_rows

def normalize_export_fields(data):
    normalized = data.copy()
    normalized["date_raw"] = pd.to_datetime(normalized["base_date"]).dt.strftime("%Y-%m-%d")
    normalized["time_raw"] = pd.to_datetime(normalized["timestamp"]).dt.strftime("%H:%M")
    return normalized

def main():
    CLEAN_DATA.mkdir(parents=True, exist_ok=True)
    data, duplicate_rows = load_all_data()
    data = normalize_export_fields(data)
    data.to_csv(CLEAN_DATA / "水质监测数据_清洗后.csv", index=False, encoding="utf-8-sig")
    data.to_pickle(CLEAN_DATA / "水质监测数据_清洗后.pkl")
    print(f"已保存清洗数据：{CLEAN_DATA / '水质监测数据_清洗后.csv'}")
    print(f"时间范围：{data['timestamp'].min()} 至 {data['timestamp'].max()}")
    print(f"观测数：{len(data)}")
if __name__ == "__main__":
    main()
