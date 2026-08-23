from pathlib import Path
import inspect
import sys

import numpy as np
import pandas as pd
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q3.tree_expert import LightGBMExpert
from q3.mechanistic import MechanisticExpert
from q3.gru_expert import GRUExpert, GRUNet, SEQUENCE_COLUMNS, build_sequence_tensors
from q3_fixtures import supervised_fixture


def test_lightgbm_expert_returns_six_horizons_and_is_deterministic():
    frame, origins, y = supervised_fixture(n=360)

    first = LightGBMExpert().fit(frame, origins, y)
    second = LightGBMExpert().fit(frame, origins, y)

    p1 = first.predict(frame, origins[-20:])
    p2 = second.predict(frame, origins[-20:])

    assert p1.shape == (20, 6)
    np.testing.assert_allclose(p1, p2)
    assert len(first.models_) == 6
    assert all(model.objective == "huber" for model in first.models_)
    assert set(first.feature_importance()) == {"horizon_hours", "feature", "importance"}
    assert first.booster_paths() == (
        "lightgbm_h02.txt",
        "lightgbm_h04.txt",
        "lightgbm_h06.txt",
        "lightgbm_h08.txt",
        "lightgbm_h10.txt",
        "lightgbm_h12.txt",
    )


def test_lightgbm_prediction_ignores_future_ground_truth():
    frame, origins, y = supervised_fixture(n=360)
    model = LightGBMExpert().fit(frame, origins[:-30], y[:-30])
    altered = frame.copy()
    altered.loc[origins[-20:] + pd.Timedelta(hours=2), "treated_ntu"] = 99.0

    np.testing.assert_allclose(
        model.predict(frame, origins[-20:]),
        model.predict(altered, origins[-20:]),
    )


def test_lightgbm_exposes_the_exact_transformed_prediction_rows():
    frame, origins, y = supervised_fixture(n=180)
    model = LightGBMExpert().fit(frame, origins[:-20], y[:-20])
    query = origins[-10:]
    filled = MechanisticExpert().fit(frame, origins[-21]).fill_target_history(frame)

    transformed = model.prediction_features(frame, query, filled)
    reconstructed = np.column_stack(
        [tree.predict(transformed) for tree in model.models_]
    )

    np.testing.assert_allclose(reconstructed, model.predict(frame, query, filled))
    assert transformed.index.equals(query)


def test_lightgbm_uses_actual_mechanistic_fills_but_ignores_post_fit_truth():
    frame, origins, _ = supervised_fixture(n=360)
    target_history = frame["treated_ntu"].shift(1).loc[origins].to_numpy()
    y = np.repeat(target_history[:, None], 6, axis=1)
    model = LightGBMExpert().fit(frame, origins[:-40], y[:-40])
    query_origins = origins[-1:]
    evaluation = frame.copy()
    post_fit = evaluation.index > model.fit_end_
    evaluation.loc[post_fit, "treated_ntu"] = np.nan
    evaluation.loc[post_fit, "target_available"] = False
    evaluation.loc[post_fit, "missing_treated_ntu"] = True
    observed_position = query_origins[0] - pd.Timedelta(hours=4)
    evaluation.loc[observed_position, "treated_ntu"] = frame.loc[observed_position, "treated_ntu"]

    mechanistic = MechanisticExpert().fit(evaluation, model.fit_end_)
    filled_target = mechanistic.fill_target_history(evaluation)
    with_fills = model.predict(evaluation, query_origins, filled_target=filled_target)
    altered_truth = evaluation.copy()
    altered_truth.loc[observed_position, "treated_ntu"] = 99.0
    altered_filled_target = mechanistic.fill_target_history(altered_truth)

    assert np.isfinite(filled_target.loc[observed_position])
    assert altered_filled_target.loc[observed_position] == pytest.approx(
        filled_target.loc[observed_position]
    )
    np.testing.assert_allclose(
        with_fills,
        model.predict(altered_truth, query_origins, filled_target=altered_filled_target),
    )


def test_tree_and_gru_retain_permitted_observed_history_through_available_boundary(monkeypatch):
    frame, origins, y = supervised_fixture(n=220)
    train_origins = origins[:-30]
    available_end = train_origins[-1] + pd.Timedelta(hours=12)
    query = origins[-10:]
    monkeypatch.setattr(GRUExpert, "HIDDEN_SIZES", (32,))
    monkeypatch.setattr(GRUExpert, "DROPOUTS", (0.1,))
    monkeypatch.setattr(GRUExpert, "MAX_EPOCHS", 2)
    monkeypatch.setattr(GRUExpert, "EARLY_STOPPING_ROUNDS", 1)
    monkeypatch.setattr(LightGBMExpert, "NUM_LEAVES", (15,))
    monkeypatch.setattr(LightGBMExpert, "LEARNING_RATES", (0.05,))
    monkeypatch.setattr(LightGBMExpert, "MIN_CHILD_SAMPLES", (20,))
    monkeypatch.setattr(LightGBMExpert, "FEATURE_FRACTIONS", (1.0,))
    monkeypatch.setattr(LightGBMExpert, "MAX_ESTIMATORS", 8)
    monkeypatch.setattr(LightGBMExpert, "EARLY_STOPPING_ROUNDS", 2)
    tree = LightGBMExpert().fit(frame, train_origins, y[:-30])
    gru = GRUExpert().fit(frame, train_origins, y[:-30])
    tree.available_target_end_ = available_end
    gru.available_target_end_ = available_end
    filled = MechanisticExpert().fit(frame, available_end).fill_target_history(frame)
    baseline_tree = tree.predict(frame, query, filled)
    baseline_gru = gru.predict(frame, query, filled)

    changed_permitted = frame.copy()
    changed_permitted.loc[tree.fit_end_ + pd.Timedelta(hours=2):available_end, "treated_ntu"] += 0.5
    changed_future = frame.copy()
    changed_future.loc[available_end + pd.Timedelta(hours=2):, "treated_ntu"] += 100.0

    assert not np.allclose(tree.predict(changed_permitted, query, filled), baseline_tree)
    assert not np.allclose(gru.predict(changed_permitted, query, filled), baseline_gru)
    np.testing.assert_allclose(tree.predict(changed_future, query, filled), baseline_tree)
    np.testing.assert_allclose(gru.predict(changed_future, query, filled), baseline_gru)


def test_all_expert_fill_paths_ignore_injected_structurally_unavailable_target(monkeypatch):
    frame, origins, y = supervised_fixture(n=220)
    train_origins = origins[:-30]
    query = origins[-10:]
    unavailable = query[0] - pd.Timedelta(hours=2)
    frame.loc[unavailable, "target_available"] = False
    frame.loc[unavailable, "missing_treated_ntu"] = True
    frame.loc[unavailable, "treated_ntu"] = np.nan
    monkeypatch.setattr(GRUExpert, "HIDDEN_SIZES", (32,))
    monkeypatch.setattr(GRUExpert, "DROPOUTS", (0.1,))
    monkeypatch.setattr(GRUExpert, "MAX_EPOCHS", 2)
    monkeypatch.setattr(GRUExpert, "EARLY_STOPPING_ROUNDS", 1)
    monkeypatch.setattr(LightGBMExpert, "NUM_LEAVES", (15,))
    monkeypatch.setattr(LightGBMExpert, "LEARNING_RATES", (0.05,))
    monkeypatch.setattr(LightGBMExpert, "MIN_CHILD_SAMPLES", (20,))
    monkeypatch.setattr(LightGBMExpert, "FEATURE_FRACTIONS", (1.0,))
    monkeypatch.setattr(LightGBMExpert, "MAX_ESTIMATORS", 8)
    monkeypatch.setattr(LightGBMExpert, "EARLY_STOPPING_ROUNDS", 2)
    available_end = train_origins[-1] + pd.Timedelta(hours=12)
    mechanistic = MechanisticExpert().fit(frame, available_end)
    tree = LightGBMExpert().fit(frame, train_origins, y[:-30])
    gru = GRUExpert().fit(frame, train_origins, y[:-30])
    tree.available_target_end_ = available_end
    gru.available_target_end_ = available_end
    baseline_fill = mechanistic.fill_target_history(frame)
    baseline_tree = tree.predict(frame, query, baseline_fill)
    baseline_gru = gru.predict(frame, query, baseline_fill)
    injected = frame.copy()
    injected.loc[unavailable, "treated_ntu"] = 9999.0
    injected_fill = mechanistic.fill_target_history(injected)

    assert injected_fill.loc[unavailable] == pytest.approx(baseline_fill.loc[unavailable])
    np.testing.assert_allclose(tree.predict(injected, query, injected_fill), baseline_tree)
    np.testing.assert_allclose(gru.predict(injected, query, injected_fill), baseline_gru)


def test_lightgbm_requires_sorted_origins_and_purges_validation_labels():
    frame, origins, y = supervised_fixture(n=360)

    with pytest.raises(ValueError, match="sorted"):
        LightGBMExpert().fit(frame, origins[::-1], y[::-1])

    model = LightGBMExpert().fit(frame, origins, y)

    assert model.validation_train_label_end_ < model.validation_start_
    assert (model.validation_train_origins_ + pd.Timedelta(hours=12) < model.validation_start_).all()


def test_lightgbm_hard_codes_the_required_grid_weights_and_subsampling():
    assert list(inspect.signature(LightGBMExpert).parameters) == []
    assert len(LightGBMExpert._grid()) == 16
    assert {item["num_leaves"] for item in LightGBMExpert._grid()} == {15, 31}
    assert {item["learning_rate"] for item in LightGBMExpert._grid()} == {0.03, 0.05}
    assert {item["min_child_samples"] for item in LightGBMExpert._grid()} == {20, 40}
    assert {item["feature_fraction"] for item in LightGBMExpert._grid()} == {0.8, 1.0}
    assert LightGBMExpert.MAX_ESTIMATORS == 800
    assert LightGBMExpert.EARLY_STOPPING_ROUNDS == 50
    assert LightGBMExpert._regressor(LightGBMExpert._grid()[0]).get_params()["subsample_freq"] == 1

    features = pd.DataFrame(
        {
            "raw_water_ntu_change1": [0.0, 50.0, 0.0, 50.0],
            "alum_dosage_change1": [0.0, 0.0, 0.01, 0.01],
        }
    )
    weights = LightGBMExpert._sample_weights(features, np.array([1.0, 2.0, 3.0, 100.0]))

    np.testing.assert_allclose(weights, [1.0, 2.0, 2.0, 4.0])


def test_sequence_tensor_uses_exactly_24_past_steps():
    frame, origins, y = supervised_fixture(n=160)
    frame.loc[:, "raw_water_ntu"] = np.arange(len(frame), dtype=float)

    x, mask, targets = build_sequence_tensors(frame, origins[24:], y[24:])

    assert x.shape[1] == 24
    assert mask.shape == x.shape
    assert targets.shape == (len(origins[24:]), 6)
    np.testing.assert_allclose(x[0, :, 0], np.arange(24, 48, dtype=float))


def test_gru_expert_uses_mechanistic_fills_without_post_fit_truth_leakage():
    frame, origins, y = supervised_fixture(n=220)
    frame.loc[origins[120:150], "treated_ntu"] = np.nan
    model = GRUExpert().fit(frame, origins[:160], y[:160])
    query_origins = origins[160:180]
    mechanistic = MechanisticExpert().fit(frame, model.fit_end_)
    filled_target = mechanistic.fill_target_history(frame)
    prediction = model.predict(frame, query_origins, filled_target)

    altered = frame.copy()
    altered.loc[origins[160:180], "treated_ntu"] = 99.0
    altered_fills = mechanistic.fill_target_history(altered)

    assert prediction.shape == (20, 6)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(
        prediction,
        model.predict(altered, query_origins, altered_fills),
    )


def test_gru_expert_uses_a_max_horizon_purged_validation_split():
    frame, origins, y = supervised_fixture(n=180)

    model = GRUExpert().fit(frame, origins, y)

    assert model.validation_train_label_end_ < model.validation_start_
    assert (model.validation_train_origins_ + pd.Timedelta(hours=12) < model.validation_start_).all()
    assert list(inspect.signature(GRUExpert).parameters) == []


def test_sequence_tensor_keeps_fill_provenance_and_required_causal_features():
    frame, origins, y = supervised_fixture(n=100)
    target_feature = SEQUENCE_COLUMNS.index("treated_ntu")
    missing_time = origins[0] - pd.Timedelta(hours=2)
    observed_time = origins[0] - pd.Timedelta(hours=4)
    frame.loc[missing_time, "treated_ntu"] = np.nan
    frame.loc[missing_time, "target_available"] = False
    frame.loc[missing_time, "missing_treated_ntu"] = True
    fills = frame["treated_ntu"].copy()
    fills.loc[missing_time] = 0.77
    fills.loc[observed_time] = 999.0

    x, mask, _ = build_sequence_tensors(frame, origins[:1], y[:1], fills)

    assert {"is_backwash_event", "hour", "weekday", "month", "day_sin", "week_cos"}.issubset(SEQUENCE_COLUMNS)
    assert {"raw_water_ntu_change1", "raw_water_ntu_change3", "raw_water_ntu_change6"}.issubset(SEQUENCE_COLUMNS)
    assert x[0, -1, target_feature] == pytest.approx(0.77)
    assert mask[0, -1, target_feature]
    assert x[0, -2, target_feature] == frame.loc[observed_time, "treated_ntu"]
    assert not mask[0, -2, target_feature]


def test_gru_validation_inputs_ignore_validation_target_mutation(monkeypatch):
    frame, origins, y = supervised_fixture(n=180)
    monkeypatch.setattr(GRUExpert, "MAX_EPOCHS", 2)
    monkeypatch.setattr(GRUExpert, "EARLY_STOPPING_ROUNDS", 1)
    first = GRUExpert().fit(frame, origins, y)
    altered = frame.copy()
    altered.loc[first.validation_start_:, "treated_ntu"] = 99.0
    second = GRUExpert().fit(altered, origins, y)

    np.testing.assert_allclose(first.validation_inputs_, second.validation_inputs_)
    assert (first.hidden_size_, first.dropout_) == (second.hidden_size_, second.dropout_)


def test_gru_event_weights_and_block_augmentation_use_process_shocks_only():
    frame, origins, y = supervised_fixture(n=200)
    weight_origins = origins[[0, 2, 4, 6]]
    prior = weight_origins - pd.Timedelta(hours=2)
    frame.loc[weight_origins, "raw_water_ntu"] = [10.0, 60.0, 10.0, 60.0]
    frame.loc[prior, "raw_water_ntu"] = 10.0
    frame.loc[weight_origins, "alum_dosage"] = [0.05, 0.05, 0.06, 0.06]
    frame.loc[prior, "alum_dosage"] = 0.05

    np.testing.assert_allclose(GRUExpert._sample_weights(frame, weight_origins), [1.0, 2.0, 2.0, 4.0])
    fills = MechanisticExpert().fit(frame, origins[-1]).fill_target_history(frame)
    x, mask, _ = build_sequence_tensors(frame, origins, y, fills)
    augmented_x, augmented_mask, block = GRUExpert._augment_training_history(
        x, mask, origins, fills
    )
    target_feature = SEQUENCE_COLUMNS.index("treated_ntu")
    artificial = augmented_mask[:, :, target_feature] & ~mask[:, :, target_feature]

    assert 24 <= block["steps"] <= 168
    assert artificial.any()
    assert np.max(np.abs(
        augmented_x[:, :, target_feature][artificial] - x[:, :, target_feature][artificial]
    )) > 0
    np.testing.assert_allclose(
        augmented_x[:, :, :target_feature], x[:, :, :target_feature]
    )

    altered = frame.copy()
    hidden = (altered.index >= block["start"]) & (
        altered.index < block["start"] + pd.Timedelta(hours=2 * block["steps"])
    )
    altered.loc[hidden, "treated_ntu"] = 99.0
    altered_fills = MechanisticExpert().fit(altered, origins[-1]).fill_target_history(altered)
    altered_x, altered_mask, _ = build_sequence_tensors(altered, origins, y, altered_fills)
    altered_x, altered_mask, _ = GRUExpert._augment_training_history(
        altered_x, altered_mask, origins, altered_fills
    )

    np.testing.assert_allclose(augmented_x, altered_x)
    np.testing.assert_array_equal(augmented_mask, altered_mask)


def test_gru_repeat_fit_and_state_bundle_reload_are_deterministic(monkeypatch):
    frame, origins, y = supervised_fixture(n=180)
    fills = pd.Series(0.5, index=frame.index)
    monkeypatch.setattr(GRUExpert, "MAX_EPOCHS", 3)
    monkeypatch.setattr(GRUExpert, "EARLY_STOPPING_ROUNDS", 1)
    first = GRUExpert().fit(frame, origins, y, fills)
    second = GRUExpert().fit(frame, origins, y, fills)
    query = origins[-10:]
    expected = first.predict(frame, query, fills)
    np.testing.assert_allclose(expected, second.predict(frame, query, fills), rtol=0, atol=0)

    bundle = first.state_dict_bundle()
    restored = GRUExpert()
    restored.model_ = GRUNet(bundle["input_size"], bundle["hidden_size"], bundle["dropout"])
    restored.model_.load_state_dict(bundle["state_dict"])
    restored.input_size_ = bundle["input_size"]
    restored.hidden_size_ = bundle["hidden_size"]
    restored.dropout_ = bundle["dropout"]
    restored.epochs_ = bundle["epochs"]
    restored.fit_end_ = bundle["fit_end"]
    restored.input_medians_ = bundle["input_medians"]
    restored.input_means_ = bundle["input_means"]
    restored.input_scales_ = bundle["input_scales"]
    restored.is_fitted_ = True

    np.testing.assert_allclose(expected, restored.predict(frame, query, fills), rtol=0, atol=0)


def test_gru_weighted_smooth_l1_averages_all_six_horizons_before_sample_weighting():
    prediction = torch.zeros((2, 6))
    target = torch.tensor([[1.0] * 6, [2.0] * 6])
    weights = torch.tensor([2.0, 1.0])

    loss = GRUExpert._weighted_smooth_l1(prediction, target, weights)

    assert loss.item() == pytest.approx(1.25)


def test_gru_candidate_uses_adamw_clipping_and_mean_validation_mae(monkeypatch):
    train_x = np.zeros((4, 24, 3), dtype=np.float32)
    train_y = np.zeros((4, 6), dtype=np.float32)
    valid_x = np.ones((2, 24, 3), dtype=np.float32)
    valid_y = np.full((2, 6), 0.5, dtype=np.float32)
    seen = []
    real_adamw = torch.optim.AdamW
    real_clip = torch.nn.utils.clip_grad_norm_

    def recording_adamw(parameters, **kwargs):
        seen.append(("adamw", kwargs["lr"]))
        return real_adamw(parameters, **kwargs)

    def recording_clip(parameters, maximum):
        seen.append(("clip", maximum))
        return real_clip(parameters, maximum)

    monkeypatch.setattr(GRUExpert, "MAX_EPOCHS", 1)
    monkeypatch.setattr(torch.optim, "AdamW", recording_adamw)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    model, mae, _ = GRUExpert._train_candidate(
        train_x, train_y, np.ones(4, dtype=np.float32), valid_x, valid_y, 32, 0.1
    )

    with torch.no_grad():
        expected = torch.abs(model(torch.from_numpy(valid_x)) - torch.from_numpy(valid_y)).mean().item()
    assert mae == pytest.approx(expected)
    assert ("adamw", 1e-3) in seen
    assert ("clip", GRUExpert.GRADIENT_CLIP_NORM) in seen
