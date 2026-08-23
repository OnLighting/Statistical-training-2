from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from q3.data import DEFAULT_FOLDS, HORIZONS
from q3.moe import GATE_FEATURE_NAMES, SoftmaxGate, generate_oof_predictions
import q3.moe as moe
from q3_fixtures import frame_fixture


EXPECTED_GATE_FEATURES = (
    "raw_water_ntu",
    "raw_water_ntu_change1",
    "raw_water_ntu_change3",
    "raw_water_ntu_change6",
    "alum_dosage",
    "alum_dosage_change1",
    "alum_dosage_change3",
    "alum_dosage_change6",
    "filtered_ntu",
    "raw_water_flow",
    "treated_water_flow",
    "clear_well_level",
    "hour",
    "weekday",
    "is_rainy_season",
    "is_backwash_event",
    "target_missing_run",
    "mechanistic_fill_ratio",
    "expert_prediction_std",
    "horizon_step",
)


def gate_fixture():
    rng = np.random.default_rng(2026)
    expert_predictions = rng.normal(0.5, 0.05, size=(60, 6, 3))
    gate_features = rng.normal(size=(60, 6, 5))
    targets = expert_predictions[:, :, 1] + rng.normal(0, 0.01, size=(60, 6))
    return expert_predictions, gate_features, targets


def recording_expert_factories(calls):
    class RecordingMechanistic:
        def fit(self, frame, train_end):
            calls["mechanistic"].append(pd.Timestamp(train_end))
            return self

        def fill_target_history(self, frame):
            return frame["treated_ntu"].fillna(0.42)

        def predict(self, frame, origins):
            calls["mechanistic_predict"].append(pd.DatetimeIndex(origins))
            return np.full((len(origins), 6), 0.4)

    class RecordingTree:
        def fit(self, frame, origins, targets):
            calls["tree"].append(pd.DatetimeIndex(origins))
            calls["tree_targets"].append(np.asarray(targets).copy())
            return self

        def predict(self, frame, origins, filled_target=None):
            calls["tree_predict"].append(pd.DatetimeIndex(origins))
            calls["tree_available_end"].append(getattr(self, "available_target_end_", None))
            assert filled_target is not None
            return np.full((len(origins), 6), 0.5)

    class RecordingGru:
        def fit(self, frame, origins, targets, filled_target=None):
            calls["gru"].append(pd.DatetimeIndex(origins))
            calls["gru_targets"].append(np.asarray(targets).copy())
            assert filled_target is not None
            return self

        def predict(self, frame, origins, filled_target=None):
            calls["gru_predict"].append(pd.DatetimeIndex(origins))
            calls["gru_available_end"].append(getattr(self, "available_target_end_", None))
            assert filled_target is not None
            return np.full((len(origins), 6), 0.6)

    return RecordingMechanistic, RecordingTree, RecordingGru


def dynamic_expert_factories():
    class Mechanistic:
        def fit(self, frame, train_end):
            return self

        def fill_target_history(self, frame):
            return frame["treated_ntu"].fillna(0.0)

        def predict(self, frame, origins):
            return np.zeros((len(origins), 6))

    class Tree:
        def fit(self, frame, origins, targets):
            return self

        def predict(self, frame, origins, filled_target=None):
            return np.ones((len(origins), 6))

    class Gru:
        def fit(self, frame, origins, targets, filled_target=None):
            return self

        def predict(self, frame, origins, filled_target=None):
            return np.full((len(origins), 6), 0.5)

    return Mechanistic, Tree, Gru


def call_log():
    return {
        "mechanistic": [], "tree": [], "gru": [],
        "tree_targets": [], "gru_targets": [],
        "mechanistic_predict": [], "tree_predict": [], "gru_predict": [],
        "tree_available_end": [], "gru_available_end": [],
    }


def test_softmax_gate_weights_are_nonnegative_and_sum_to_one():
    expert_predictions, gate_features, targets = gate_fixture()

    gate = SoftmaxGate().fit(expert_predictions, gate_features, targets)
    weights = gate.weights(expert_predictions, gate_features)

    assert weights.shape == expert_predictions.shape
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=2), 1.0, atol=1e-6)


def test_generated_oof_gate_weights_are_dynamic_without_one_hot_collapse():
    frame = frame_fixture(periods=5000)
    frame.loc[:, "treated_ntu"] = (frame["raw_water_ntu"] >= 30.0).astype(float)
    bundle = generate_oof_predictions(frame, dynamic_expert_factories())

    weights = SoftmaxGate().fit(
        bundle.expert_predictions, bundle.gate_features, bundle.targets
    ).weights(bundle.expert_predictions, bundle.gate_features)

    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights.sum(axis=2), 1.0, atol=1e-6)
    assert np.max(weights.std(axis=(0, 1))) > 0.02
    assert len(np.unique(np.round(weights.reshape(-1, 3), 4), axis=0)) > 8
    assert not np.all(np.max(weights, axis=2) > 0.9999)


def test_oof_uses_all_experts_and_records_four_purged_folds_with_compact_features():
    calls = call_log()
    frame = frame_fixture(periods=5000)
    bundle = generate_oof_predictions(frame, recording_expert_factories(calls))

    assert all(len(calls[name]) == 4 for name in calls)
    expected_available_ends = [
        bundle.metadata.loc[bundle.metadata["fold"] == fold, "train_end"].iloc[0]
        for fold in range(4)
    ]
    assert calls["tree_available_end"] == expected_available_ends
    assert calls["gru_available_end"] == expected_available_ends
    assert bundle.expert_predictions.shape[1:] == (6, 3)
    assert bundle.targets.shape == bundle.expert_predictions.shape[:2]
    assert bundle.gate_features.shape == (len(bundle.origins), 6, len(EXPECTED_GATE_FEATURES))
    assert bundle.gate_feature_names == EXPECTED_GATE_FEATURES
    assert len(bundle.metadata) == len(bundle.origins) * len(HORIZONS)
    assert bundle.metadata["fold"].nunique() == 4
    assert (bundle.metadata["train_end"] < bundle.metadata["valid_start"]).all()
    assert (bundle.metadata["label_end"] < bundle.metadata["valid_start"]).all()
    assert set(bundle.metadata["horizon"].unique()) == {2, 4, 6, 8, 10, 12}


def test_compact_gate_features_capture_prior_missing_fill_ratio_and_expert_dispersion():
    calls = call_log()
    frame = frame_fixture(periods=5000)
    first_origin = pd.Timestamp("2026-01-01 01:00")
    prior_gap = pd.date_range(first_origin - pd.Timedelta(hours=12), periods=6, freq="2h")
    frame.loc[prior_gap, "treated_ntu"] = np.nan
    frame.loc[prior_gap, "target_available"] = False
    frame.loc[prior_gap, "missing_treated_ntu"] = True

    bundle = generate_oof_predictions(frame, recording_expert_factories(calls))
    position = bundle.origins.get_loc(first_origin)
    columns = {name: number for number, name in enumerate(bundle.gate_feature_names)}

    assert bundle.gate_features[position, 0, columns["target_missing_run"]] == 6
    assert bundle.gate_features[position, 0, columns["mechanistic_fill_ratio"]] == pytest.approx(6 / 24)
    assert bundle.gate_features[position, 0, columns["expert_prediction_std"]] == pytest.approx(np.std([0.4, 0.5, 0.6]))
    np.testing.assert_allclose(bundle.gate_features[position, :, columns["horizon_step"]], HORIZONS)


def test_final_fold_predictions_and_gate_features_ignore_future_target_mutation():
    frame = frame_fixture(periods=5000)
    baseline = generate_oof_predictions(frame, recording_expert_factories(call_log()))
    altered = frame.copy()
    altered.loc[altered.index >= DEFAULT_FOLDS[-1].valid_start, "treated_ntu"] += 10000.0
    changed = generate_oof_predictions(altered, recording_expert_factories(call_log()))
    final_origins = baseline.metadata.loc[baseline.metadata["fold"] == 3, "origin"].unique()
    indices = baseline.origins.get_indexer(final_origins)

    np.testing.assert_allclose(baseline.expert_predictions[indices], changed.expert_predictions[indices], rtol=0, atol=0)
    np.testing.assert_allclose(baseline.gate_features[indices], changed.gate_features[indices], rtol=0, atol=0)


def test_softmax_gate_state_bundle_reloads_in_a_fresh_process(tmp_path):
    expert_predictions, gate_features, targets = gate_fixture()
    fitted = SoftmaxGate().fit(expert_predictions, gate_features, targets)
    expected = fitted.predict(expert_predictions, gate_features)
    bundle_path = tmp_path / "moe_gate.pkl"
    input_path = tmp_path / "gate_inputs.npz"
    output_path = tmp_path / "reloaded.npy"
    with bundle_path.open("wb") as stream:
        pickle.dump(fitted.state_dict_bundle(), stream)
    np.savez(input_path, expert_predictions=expert_predictions, gate_features=gate_features)
    program = (
        "import pickle, sys, numpy as np; "
        "sys.path.insert(0, sys.argv[1]); "
        "from q3.moe import SoftmaxGate; "
        "bundle = pickle.load(open(sys.argv[2], 'rb')); "
        "values = np.load(sys.argv[3]); "
        "prediction = SoftmaxGate().load_state_dict_bundle(bundle).predict(values['expert_predictions'], values['gate_features']); "
        "np.save(sys.argv[4], prediction)"
    )
    subprocess.check_call([sys.executable, "-c", program, str(SRC_DIR), str(bundle_path), str(input_path), str(output_path)])

    np.testing.assert_allclose(expected, np.load(output_path), rtol=0, atol=0)


def test_softmax_gate_state_bundle_contains_oof_input_standardization():
    expert_predictions, gate_features, targets = gate_fixture()

    state = SoftmaxGate().fit(expert_predictions, gate_features, targets).state_dict_bundle()

    assert state["input_mean"].shape == (gate_features.shape[2] + 3,)
    assert state["input_scale"].shape == (gate_features.shape[2] + 3,)
    assert np.all(state["input_scale"] > 0)


def test_softmax_gate_rejects_non_oof_shape_mismatches():
    expert_predictions, gate_features, targets = gate_fixture()

    with pytest.raises(ValueError, match="same first two dimensions"):
        SoftmaxGate().fit(expert_predictions, gate_features[:-1], targets)


def test_oof_orchestration_rejects_any_fold_scheme_other_than_four(monkeypatch):
    monkeypatch.setattr(moe, "DEFAULT_FOLDS", DEFAULT_FOLDS[:3])

    with pytest.raises(ValueError, match="exactly four"):
        generate_oof_predictions(frame_fixture(periods=5000), recording_expert_factories(call_log()))


def test_oof_gate_features_model_validation_targets_as_unavailable():
    frame = frame_fixture(periods=5000)
    bundle = generate_oof_predictions(frame, recording_expert_factories(call_log()))
    missing_run = bundle.gate_feature_names.index("target_missing_run")

    for fold in range(4):
        indices = bundle.metadata.loc[bundle.metadata["fold"] == fold, "origin"].drop_duplicates()
        positions = bundle.origins.get_indexer(indices)
        values = bundle.gate_features[positions, 0, missing_run]
        assert values.min() == 0
        assert values.max() >= 100


def test_oof_downstream_training_ignores_unavailable_numeric_labels():
    missing = frame_fixture(periods=5000)
    unavailable_time = pd.Timestamp("2025-10-10 01:00")
    missing.loc[unavailable_time, "treated_ntu"] = np.nan
    missing.loc[unavailable_time, "target_available"] = False
    missing.loc[unavailable_time, "missing_treated_ntu"] = True
    injected = missing.copy()
    injected.loc[unavailable_time, "treated_ntu"] = 999999.0
    missing_calls = call_log()
    injected_calls = call_log()

    missing_bundle = generate_oof_predictions(
        missing, recording_expert_factories(missing_calls)
    )
    injected_bundle = generate_oof_predictions(
        injected, recording_expert_factories(injected_calls)
    )

    pd.testing.assert_index_equal(missing_bundle.origins, injected_bundle.origins)
    np.testing.assert_allclose(missing_bundle.targets, injected_bundle.targets, rtol=0, atol=0)
    for name in ("tree", "gru"):
        for missing_origins, injected_origins in zip(
            missing_calls[name], injected_calls[name]
        ):
            pd.testing.assert_index_equal(missing_origins, injected_origins)
    for name in ("tree_targets", "gru_targets"):
        for missing_targets, injected_targets in zip(
            missing_calls[name], injected_calls[name]
        ):
            np.testing.assert_allclose(missing_targets, injected_targets, rtol=0, atol=0)


def test_zero_variance_gate_inputs_have_no_random_extrapolation_effect():
    expert_predictions, gate_features, targets = gate_fixture()
    gate_features[:, :, 2] = 0.0
    gate = SoftmaxGate().fit(expert_predictions, gate_features, targets)
    baseline = gate.weights(expert_predictions, gate_features)
    extrapolated = gate_features.copy()
    extrapolated[:, :, 2] = 10000.0

    np.testing.assert_allclose(
        gate.weights(expert_predictions, extrapolated), baseline, rtol=0, atol=1e-7
    )


def test_gate_is_smooth_and_dynamic_at_supported_long_missing_runs():
    generator = np.random.default_rng(2026)
    runs = np.arange(360, dtype=float)
    expert_predictions = generator.uniform(0.15, 0.65, size=(len(runs), 6, 3))
    gate_features = generator.normal(size=(len(runs), 6, len(GATE_FEATURE_NAMES)))
    missing_run = GATE_FEATURE_NAMES.index("target_missing_run")
    gate_features[:, :, missing_run] = runs[:, None]
    desired_tree_weight = 0.25 + 0.35 * runs / runs.max()
    targets = (
        0.45 * expert_predictions[:, :, 0]
        + desired_tree_weight[:, None] * expert_predictions[:, :, 1]
        + (0.55 - desired_tree_weight)[:, None] * expert_predictions[:, :, 2]
    )
    gate = SoftmaxGate().fit(expert_predictions, gate_features, targets)
    positions = np.array([107, 108, 227, 228])
    selected_predictions = expert_predictions[positions]
    selected_features = gate_features[positions]
    weights = gate.weights(selected_predictions, selected_features)
    perturbed = selected_features.copy()
    perturbed[:, :, missing_run] += 0.01
    changed = gate.weights(selected_predictions, perturbed)

    assert np.max(weights) < 0.999
    assert np.ptp(weights[:, :, 1]) > 1e-5
    assert np.max(np.abs(changed - weights)) < 1e-3
