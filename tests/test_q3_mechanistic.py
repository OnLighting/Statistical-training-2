import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q3.mechanistic import MechanisticExpert, cstr_cascade
from q3_fixtures import synthetic_regular_frame


def _non_collinear_frame(periods=220):
    """Regular process data that permits the SARIMAX likelihood to converge."""
    frame = synthetic_regular_frame(periods=periods)
    rng = np.random.default_rng(91)
    step = np.arange(periods)
    filtered = 0.20 + 0.04 * np.sin(step / 5.0) + rng.normal(0, 0.01, periods)
    flow = 46.0 + rng.normal(0, 1.5, periods)
    level = 3.8 + rng.normal(0, 0.08, periods)
    frame["filtered_ntu"] = filtered
    frame["treated_water_flow"] = flow
    frame["clear_well_level"] = level
    frame["treated_ntu"] = (
        0.15 + 0.55 * filtered + 0.001 * flow + rng.normal(0, 0.015, periods)
    )
    return frame


def test_cstr_state_moves_monotonically_toward_constant_inlet():
    inlet = np.full(12, 1.0)
    flow = np.full(12, 50.0)
    level = np.full(12, 3.8)

    state = cstr_cascade(
        inlet, flow, level, n_tanks=1, volume_scale=100.0, initial=0.0
    )

    assert np.all(np.diff(state) >= 0)
    assert 0 < state[-1] < 1


def test_mechanistic_expert_predicts_six_finite_horizons():
    frame = synthetic_regular_frame(periods=240)
    model = MechanisticExpert().fit(frame, frame.index[179])

    prediction = model.predict(frame, frame.index[180:190])

    assert prediction.shape == (10, 6)
    assert np.isfinite(prediction).all()


def test_mechanistic_expert_can_forecast_from_the_known_training_endpoint():
    frame = synthetic_regular_frame(periods=240)
    train_end = frame.index[179]
    model = MechanisticExpert().fit(frame, train_end)

    prediction = model.predict(frame, pd.DatetimeIndex([train_end]))

    assert prediction.shape == (1, 6)
    assert np.isfinite(prediction).all()


def test_fill_target_history_does_not_replace_observed_values():
    frame = synthetic_regular_frame(periods=100)
    frame.loc[frame.index[70:80], "treated_ntu"] = np.nan
    model = MechanisticExpert().fit(frame, frame.index[69])

    filled = model.fill_target_history(frame)

    np.testing.assert_allclose(
        filled.loc[:frame.index[69]], frame.loc[:frame.index[69], "treated_ntu"]
    )
    assert filled.loc[frame.index[70:80]].notna().all()


def test_fill_target_history_fills_internal_training_gap_with_converged_sarimax():
    frame = _non_collinear_frame()
    gap = frame.index[90:96]
    observed = frame["treated_ntu"].copy()
    frame.loc[gap, "treated_ntu"] = np.nan
    model = MechanisticExpert().fit(frame, frame.index[179])

    assert model.results_ is not None
    filled = model.fill_target_history(frame)

    assert filled.loc[gap].notna().all()
    np.testing.assert_allclose(filled.drop(gap), observed.drop(gap))


def test_mechanistic_fit_and_fill_ignore_unavailable_numeric_injection():
    frame = _non_collinear_frame()
    unavailable_time = frame.index[100]
    missing = frame.copy()
    missing.loc[unavailable_time, "treated_ntu"] = np.nan
    missing.loc[unavailable_time, "target_available"] = False
    missing.loc[unavailable_time, "missing_treated_ntu"] = True
    injected = missing.copy()
    injected.loc[unavailable_time, "treated_ntu"] = 999999.0
    train_end = frame.index[179]
    origins = frame.index[180:190]

    missing_model = MechanisticExpert().fit(missing, train_end)
    injected_model = MechanisticExpert().fit(injected, train_end)

    assert missing_model.n_tanks_ == injected_model.n_tanks_
    assert missing_model.volume_scale_ == injected_model.volume_scale_
    np.testing.assert_allclose(
        missing_model._ols_coefficients_, injected_model._ols_coefficients_, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        missing_model.predict(missing, origins),
        injected_model.predict(injected, origins),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        missing_model.fill_target_history(missing),
        injected_model.fill_target_history(injected),
        rtol=0,
        atol=0,
    )
    assert injected_model.fill_target_history(injected).loc[unavailable_time] != 999999.0


def test_mechanistic_expert_rejects_even_hour_grid_for_fit_and_predict():
    frame = synthetic_regular_frame(periods=240)
    shifted = frame.copy()
    shifted.index = shifted.index - pd.Timedelta(hours=1)

    with pytest.raises(ValueError, match="odd-hour"):
        MechanisticExpert().fit(shifted, shifted.index[179])

    model = MechanisticExpert().fit(frame, frame.index[179])
    with pytest.raises(ValueError, match="odd-hour"):
        model.predict(shifted, shifted.index[180:190])


def test_cstr_selection_records_three_chronological_tail_folds():
    frame = synthetic_regular_frame(periods=240)
    model = MechanisticExpert().fit(frame, frame.index[179])

    diagnostics = model.to_state()["selection_folds"]

    assert len(diagnostics) >= 3
    assert [fold["validation_start"] for fold in diagnostics] == sorted(
        fold["validation_start"] for fold in diagnostics
    )
    assert all(fold["train_end"] < fold["validation_start"] for fold in diagnostics)
