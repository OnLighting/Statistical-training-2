from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from q3.data import (
    DEFAULT_FOLDS,
    FINAL_TRAIN_END,
    build_origins,
    make_feature_frame,
    make_targets,
    prepare_q3_frame,
    target_availability,
)
from q3_fixtures import synthetic_q3_data


def test_prepare_q3_frame_is_two_hourly_and_deduplicated():
    raw = synthetic_q3_data(periods=80)
    raw = pd.concat([raw, raw.iloc[[10]]], ignore_index=True)
    frame = prepare_q3_frame(raw)
    assert frame.index.is_unique
    assert frame.index.to_series().diff().dropna().eq(pd.Timedelta(hours=2)).all()


def test_prepare_q3_frame_preserves_odd_hour_operating_grid():
    raw = synthetic_q3_data(periods=2, start="2026-02-01 05:00")
    frame = prepare_q3_frame(raw)
    expected = pd.DatetimeIndex(["2026-02-01 05:00", "2026-02-01 07:00"])
    assert expected.isin(frame.index).all()
    np.testing.assert_allclose(
        frame.loc[expected, "treated_ntu"].to_numpy(), raw["treated_ntu"].to_numpy()
    )
    assert FINAL_TRAIN_END in frame.index


def test_default_folds_never_train_on_or_after_validation_start():
    assert len(DEFAULT_FOLDS) == 4
    assert all(f.train_end < f.valid_start <= f.valid_end for f in DEFAULT_FOLDS)
    assert FINAL_TRAIN_END == pd.Timestamp("2026-02-01 05:00")


def test_make_targets_maps_steps_to_two_hour_horizons():
    frame = synthetic_q3_data(periods=20).set_index("timestamp")
    origins = frame.index[[5]]
    y = make_targets(frame, origins)
    np.testing.assert_allclose(y[0], frame["treated_ntu"].iloc[6:12])


def test_training_origins_keep_all_six_targets_at_or_before_fit_end():
    frame = prepare_q3_frame(synthetic_q3_data(periods=200))
    fit_end = frame.index[150]
    origins = build_origins(frame, frame.index[24], fit_end)
    assert len(origins) > 0
    assert origins.max() + pd.Timedelta(hours=12) <= fit_end


def test_rolling_features_ignore_current_and_future_raw_water_values():
    frame = prepare_q3_frame(synthetic_q3_data(periods=160))
    origin = frame.index[100]
    altered = frame.copy()
    altered.loc[origin:, "raw_water_ntu"] = 999.0

    baseline = make_feature_frame(frame, origin).loc[origin]
    changed = make_feature_frame(altered, origin).loc[origin]
    rolling_columns = [column for column in baseline.index if "_roll" in column]
    pd.testing.assert_series_equal(baseline[rolling_columns], changed[rolling_columns])


def test_feature_fit_rows_rejects_requests_after_fit_end():
    frame = prepare_q3_frame(synthetic_q3_data(periods=100))
    fit_end = frame.index[60]
    features = make_feature_frame(frame, fit_end)

    assert features.fit_rows().index.max() <= fit_end
    with pytest.raises(ValueError, match="fit_end"):
        features.fit_rows(frame.index[61:63])


def test_target_availability_survives_numeric_injection_and_drives_missing_run():
    data = synthetic_q3_data(periods=80)
    unavailable_time = data.loc[50, "timestamp"]
    data.loc[50, "treated_ntu"] = np.nan
    data.loc[50, "target_available"] = False
    data.loc[50, "missing_treated_ntu"] = True
    frame = prepare_q3_frame(data)
    injected = frame.copy()
    injected.loc[unavailable_time, "treated_ntu"] = 9999.0

    assert not target_availability(injected).loc[unavailable_time]
    features = make_feature_frame(injected, injected.index[60])
    assert features.loc[unavailable_time + pd.Timedelta(hours=2), "target_missing_run"] == 1


def test_unavailable_numeric_target_is_never_a_label_or_complete_origin():
    frame = synthetic_q3_data(periods=180).set_index("timestamp")
    unavailable_time = frame.index[100]
    missing = frame.copy()
    missing.loc[unavailable_time, "treated_ntu"] = np.nan
    missing.loc[unavailable_time, "target_available"] = False
    missing.loc[unavailable_time, "missing_treated_ntu"] = True
    injected = missing.copy()
    injected.loc[unavailable_time, "treated_ntu"] = 999999.0
    origin = unavailable_time - pd.Timedelta(hours=12)

    missing_target = make_targets(missing, pd.DatetimeIndex([origin]))
    injected_target = make_targets(injected, pd.DatetimeIndex([origin]))
    missing_origins = build_origins(missing, frame.index[24], frame.index[140])
    injected_origins = build_origins(injected, frame.index[24], frame.index[140])

    assert np.isnan(missing_target[0, 5])
    assert np.isnan(injected_target[0, 5])
    pd.testing.assert_index_equal(missing_origins, injected_origins)
    assert origin not in injected_origins
