import numpy as np
import pandas as pd


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
    data["target_available"] = data["treated_ntu"].notna()
    data["missing_treated_ntu"] = ~data["target_available"]
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
