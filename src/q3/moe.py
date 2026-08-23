
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .data import DEFAULT_FOLDS, HORIZONS, RANDOM_STATE, build_origins, make_feature_frame, make_targets, target_availability


GATE_FRAME_FEATURES = (
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
)
GATE_FEATURE_NAMES = GATE_FRAME_FEATURES + (
    "mechanistic_fill_ratio",
    "expert_prediction_std",
    "horizon_step",
)


class OOFBundle:
    """The frozen four-fold expert outputs used to train the gate."""

    def __init__(self, origins, expert_predictions, gate_features, targets, metadata, gate_feature_names):
        self.origins = pd.DatetimeIndex(origins)
        self.expert_predictions = np.asarray(expert_predictions, dtype=float)
        self.gate_features = np.asarray(gate_features, dtype=float)
        self.targets = np.asarray(targets, dtype=float)
        self.metadata = metadata.copy()
        self.gate_feature_names = tuple(gate_feature_names)


def _target_missing(frame):
    return ~target_availability(frame)


def _mechanistic_fill_ratio(frame, filled_target, origins):
    # if not isinstance(filled_target, pd.Series) or not filled_target.index.equals(frame.index):
    #     raise ValueError("mechanistic filled target must align with frame")
    missing = _target_missing(frame).to_numpy(dtype=bool)
    filled = np.isfinite(pd.to_numeric(filled_target, errors="coerce").to_numpy(dtype=float))
    positions = frame.index.get_indexer(pd.DatetimeIndex(origins))
    ratio = np.empty(len(positions), dtype=float)
    for number, position in enumerate(positions):
        start = max(0, position - 24)
        ratio[number] = np.mean(missing[start:position] & filled[start:position])
    return ratio


def _gate_feature_matrix(frame, train_end, origins, filled_target):
    """Return finite, origin-causal operating features for one validation fold."""
    features = make_feature_frame(frame, train_end)
    # missing_columns = [name for name in GATE_FRAME_FEATURES if name not in features]
    # if missing_columns:
    #     raise ValueError("gate feature frame is missing required causal features")
    fitting = features.fit_rows()
    medians = fitting[list(GATE_FRAME_FEATURES)].apply(pd.to_numeric, errors="coerce").median()
    medians = medians.fillna(0.0)
    values = features.loc[pd.DatetimeIndex(origins), list(GATE_FRAME_FEATURES)].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(medians).fillna(0.0).to_numpy(dtype=float)
    return np.column_stack((values, _mechanistic_fill_ratio(frame, filled_target, origins)))


def _expand_gate_features(values, expert_predictions):
    horizon = np.asarray(HORIZONS, dtype=float)
    repeated = np.repeat(values[:, None, :], len(horizon), axis=1)
    dispersion = np.std(expert_predictions, axis=2, keepdims=True)
    horizon_feature = np.broadcast_to(horizon[None, :, None], (len(values), len(horizon), 1))
    return np.concatenate((repeated, dispersion, horizon_feature), axis=2)


# def _require_three_factories(expert_factories):
    # if len(expert_factories) != 3:
    #     raise ValueError("expert_factories must contain mechanistic, tree, and GRU factories")
    # if not all(callable(factory) for factory in expert_factories):
    #     raise TypeError("each expert factory must be callable")


def generate_oof_predictions(frame, expert_factories):
    """Fit the three experts on each fixed, purged expanding time fold.

    The returned arrays are the only data intended for ``SoftmaxGate.fit``.
    Every metadata row represents one origin/horizon prediction and documents
    both the training and validation boundaries used to create it.
    """
    # _require_three_factories(expert_factories)
    # if len(DEFAULT_FOLDS) != 4:
    #     raise ValueError("OOF orchestration requires exactly four fixed expanding folds")
    # if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique:
    #     raise ValueError("frame must have a unique DatetimeIndex")

    all_origins = []
    all_predictions = []
    all_features = []
    all_targets = []
    metadata = []
    expected_names = None
    mechanistic_factory, tree_factory, gru_factory = expert_factories

    for fold_number, fold in enumerate(DEFAULT_FOLDS):
        valid_start = pd.Timestamp(fold.valid_start)
        valid_end = pd.Timestamp(fold.valid_end)
        prior_rows = frame.index[frame.index < valid_start]
        # if not len(prior_rows):
        #     raise ValueError("frame must contain training rows before every fixed validation start")
        train_end = prior_rows.max()
        train_origins = build_origins(frame, frame.index.min(), train_end)
        valid_origins = build_origins(frame, valid_start, valid_end)
        # if not len(train_origins) or not len(valid_origins):
        #     raise ValueError("frame does not contain complete purged samples for every fixed fold")

        train_targets = make_targets(frame, train_origins)
        valid_targets = make_targets(frame, valid_origins)
        # label_end = train_origins + pd.Timedelta(hours=2 * max(HORIZONS))
        # if not (label_end < valid_start).all():
        #     raise ValueError("training labels must end strictly before validation")

        validation_frame = frame.copy()
        validation_rows = validation_frame.index >= valid_start
        validation_frame.loc[validation_rows, "treated_ntu"] = np.nan
        available = target_availability(validation_frame)
        available.loc[validation_rows] = False
        validation_frame["target_available"] = available
        validation_frame["missing_treated_ntu"] = ~available

        mechanistic = mechanistic_factory().fit(frame, train_end)
        filled_target = mechanistic.fill_target_history(validation_frame)
        tree = tree_factory().fit(frame, train_origins, train_targets)
        gru = gru_factory().fit(frame, train_origins, train_targets, filled_target)
        tree.available_target_end_ = train_end
        gru.available_target_end_ = train_end
        predictions = np.stack(
            (
                mechanistic.predict(validation_frame, valid_origins),
                tree.predict(validation_frame, valid_origins, filled_target),
                gru.predict(validation_frame, valid_origins, filled_target),
            ),
            axis=2,
        )
        base_features = _gate_feature_matrix(
            validation_frame, train_end, valid_origins, filled_target
        )
        if expected_names is None:
            expected_names = GATE_FEATURE_NAMES
        gate_features = _expand_gate_features(base_features, predictions)
        all_origins.append(valid_origins)
        all_predictions.append(predictions)
        all_features.append(gate_features)
        all_targets.append(valid_targets)
        for origin in valid_origins:
            for horizon in HORIZONS:
                metadata.append(
                    {
                        "origin": origin,
                        "horizon": horizon * 2,
                        "fold": fold_number,
                        "train_end": train_end,
                        "valid_start": valid_start,
                        "valid_end": valid_end,
                        "label_end": train_origins.max() + pd.Timedelta(hours=2 * max(HORIZONS)),
                    }
                )

    return OOFBundle(
        all_origins[0].append(all_origins[1:]),
        np.concatenate(all_predictions, axis=0),
        np.concatenate(all_features, axis=0),
        np.concatenate(all_targets, axis=0),
        pd.DataFrame(metadata),
        expected_names,
    )


class SoftmaxGate:
    LEARNING_RATE = 1e-2
    EPOCHS = 300
    BALANCE_PENALTY = 1e-3

    def __init__(self):
        self.is_fitted_ = False

    @staticmethod
    def _set_seed():
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)

    @staticmethod
    def _validate_inputs(expert_predictions, gate_features, targets=None):
        predictions = np.asarray(expert_predictions, dtype=np.float32)
        features = np.asarray(gate_features, dtype=np.float32)
        if targets is None:
            return predictions, features
        target_values = np.asarray(targets, dtype=np.float32)
        return predictions, features, target_values

    @staticmethod
    def _flat_inputs(expert_predictions, gate_features):
        return np.concatenate((gate_features, expert_predictions), axis=2).reshape(-1, gate_features.shape[2] + 3)

    def _standardize(self, inputs):
        return (inputs - self.input_mean_[None, :]) / self.input_scale_[None, :]

    def fit(self, expert_predictions, gate_features, targets):
        predictions, features, target_values = self._validate_inputs(expert_predictions, gate_features, targets)
        self._set_seed()
        self.feature_count_ = features.shape[2]
        self.model_ = nn.Linear(self.feature_count_ + 3, 3)
        inputs = self._flat_inputs(predictions, features)
        self.input_mean_ = inputs.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.input_scale_ = inputs.std(axis=0, dtype=np.float64).astype(np.float32)
        constant_inputs = self.input_scale_ < 1e-6
        self.input_scale_[constant_inputs] = 1.0
        with torch.no_grad():
            self.model_.weight[:, constant_inputs] = 0.0
        x = torch.from_numpy(self._standardize(inputs).astype(np.float32))
        prediction_tensor = torch.from_numpy(predictions.reshape(-1, 3))
        target_tensor = torch.from_numpy(target_values.reshape(-1))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.LEARNING_RATE)
        self.model_.train()
        for _ in range(self.EPOCHS):
            logits = self.model_(x)
            weights = torch.softmax(logits, dim=-1)
            combined = (weights * prediction_tensor).sum(dim=-1)
            huber = F.smooth_l1_loss(combined, target_tensor)
            balance = ((weights.mean(dim=0) - 1 / 3) ** 2).sum()
            loss = huber + self.BALANCE_PENALTY * balance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.model_.weight[:, constant_inputs] = 0.0
        self.model_.eval()
        self.is_fitted_ = True
        return self

    def weights(self, expert_predictions, gate_features):
        predictions, features = self._validate_inputs(expert_predictions, gate_features)
        with torch.no_grad():
            inputs = self._standardize(self._flat_inputs(predictions, features)).astype(np.float32)
            weights = torch.softmax(self.model_(torch.from_numpy(inputs)), dim=-1)
        return weights.numpy().reshape(predictions.shape)

    def predict(self, expert_predictions, gate_features):
        predictions, _ = self._validate_inputs(expert_predictions, gate_features)
        return (self.weights(predictions, gate_features) * predictions).sum(axis=2)

    def state_dict_bundle(self):
        return {
            "feature_count": self.feature_count_,
            "input_mean": self.input_mean_.copy(),
            "input_scale": self.input_scale_.copy(),
            "state_dict": {name: value.detach().clone() for name, value in self.model_.state_dict().items()},
        }

    def load_state_dict_bundle(self, bundle):
        self.feature_count_ = int(bundle["feature_count"])
        self.input_mean_ = np.asarray(bundle["input_mean"], dtype=np.float32).copy()
        self.input_scale_ = np.asarray(bundle["input_scale"], dtype=np.float32).copy()
        self.model_ = nn.Linear(self.feature_count_ + 3, 3)
        self.model_.load_state_dict(bundle["state_dict"])
        self.model_.eval()
        self.is_fitted_ = True
        return self
