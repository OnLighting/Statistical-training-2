import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA = ROOT / "outputs" / "00_数据预处理"
FULL_FIELDS = [
    "time_raw",
    "river_level",
    "raw_water_pump_duty",
    "raw_water_flow",
    "raw_water_ntu",
    "raw_water_color",
    "raw_water_ph",
    "filtered_ntu",
    "clear_well_level",
    "treated_ph",
    "treated_ntu",
    "treated_color",
    "chlorine_residual",
    "alum_feed_rate",
    "alum_dosage",
    "treated_water_pump_duty",
    "treated_water_flow",
    "tank_18ml_level",
    "tank_18ml_flow",
    "remarks",
]
NUMERIC_COLUMNS = [
    "river_level",
    "raw_water_flow",
    "raw_water_ntu",
    "raw_water_color",
    "raw_water_ph",
    "filtered_ntu",
    "clear_well_level",
    "treated_ph",
    "treated_ntu",
    "treated_color",
    "chlorine_residual",
    "alum_feed_rate",
    "alum_dosage",
    "treated_water_flow",
    "tank_18ml_level",
    "tank_18ml_flow",
]
SHORT_2025_FIELDS = [
    "date_raw",
    "time_raw",
    "river_level",
    "raw_water_pump_duty",
    "raw_water_flow",
    "raw_water_ntu",
    "raw_water_color",
    "raw_water_ph",
    "filtered_ntu",
    "clear_well_level",
    "treated_ph",
    "treated_ntu",
    "treated_color",
    "chlorine_residual",
    "treated_water_flow",
    "remarks",
]
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "July": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
VARIABLE_LABELS = {
    "river_level": "河水水位",
    "raw_water_flow": "原水流量",
    "raw_water_ntu": "原水浊度/NTU",
    "raw_water_color": "原水色度",
    "raw_water_ph": "原水pH",
    "filtered_ntu": "滤后水浊度/NTU",
    "clear_well_level": "清水池水位",
    "treated_ph": "处理后水pH",
    "treated_ntu": "出厂水浊度/NTU",
    "treated_color": "处理后水色度",
    "chlorine_residual": "余氯",
    "alum_feed_rate": "矾投加流量",
    "alum_dosage": "矾投加量",
    "treated_water_flow": "出厂水流量",
    "raw_water_pump_count": "原水泵运行台数",
    "treated_water_pump_count": "送水泵运行台数",
}
def find_month_from_name(path):
    for name, month in MONTHS.items():
        if name.lower() in path.stem.lower():
            return month
    raise ValueError(f"无法从文件名识别月份：{path.name}")
def excel_serial_to_date(value):
    return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
def set_chinese_style():
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    sns.set_theme(style="whitegrid", font="SimHei")
def load_clean_data():
    data = pd.read_pickle(CLEAN_DATA / "水质监测数据_清洗后.pkl")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data.sort_values("timestamp").reset_index(drop=True)
def label(column):
    return VARIABLE_LABELS.get(column, column)
def save_figure(fig, output_dir, file_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
def regular_series(data, column, frequency="2h", limit=3):
    series = data.drop_duplicates("timestamp").set_index("timestamp")[column].sort_index()
    series = series.resample(frequency).median()
    return series.interpolate(method="time", limit=limit, limit_area="inside")
def longest_true_run(values):
    longest, current = 0, 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
def zscore_frame(data, columns):
    frame = data[columns].copy()
    standard_deviation = frame.std(ddof=0).replace(0, np.nan)
    return (frame - frame.mean()) / standard_deviation