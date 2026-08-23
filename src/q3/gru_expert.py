import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import CORE_COLUMNS, HISTORY_STEPS, HORIZONS, RANDOM_STATE, target_availability
TIME_COLUMNS = ("hour", "weekday", "month", "is_rainy_season", "day_sin", "day_cos", "week_sin", "week_cos")
CHANGE_COLUMNS = ("raw_water_ntu", "filtered_ntu", "alum_dosage", "raw_water_flow", "treated_water_flow")
CHANGE_STEPS = (1, 3, 6)
SEQUENCE_COLUMNS = CORE_COLUMNS + ("is_backwash_event",) + TIME_COLUMNS + tuple(
    column + "_change" + str(step) for column in CHANGE_COLUMNS for step in CHANGE_STEPS
)
def _source_column(frame, column, filled_target=None):
    if column in TIME_COLUMNS:
        hour = frame.index.hour + frame.index.minute / 60.0
        weekday = frame.index.weekday
        month = frame.index.month
        values = {
            "hour": hour, "weekday": weekday, "month": month,
            "is_rainy_season": month.isin((5, 6, 7, 8, 9)).astype(float),
            "day_sin": np.sin(2 * np.pi * hour / 24.0),
            "day_cos": np.cos(2 * np.pi * hour / 24.0),
            "week_sin": np.sin(2 * np.pi * (weekday * 24.0 + hour) / (7 * 24.0)),
            "week_cos": np.cos(2 * np.pi * (weekday * 24.0 + hour) / (7 * 24.0)),
        }
        return pd.Series(values[column], index=frame.index), pd.Series(False, index=frame.index)
    if "_change" in column:
        base, step = column.rsplit("_change", 1)
        source, missing = _source_column(frame, base, filled_target)
        changed = source.diff(int(step))
        return changed, missing | missing.shift(int(step), fill_value=True) | changed.isna()
    if column == "is_backwash_event":
        if column not in frame:
            return pd.Series(np.nan, index=frame.index), pd.Series(True, index=frame.index)
        return frame[column].fillna(False).astype(float), frame[column].isna()

    source = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)
    original_missing = frame.get("missing_" + column, source.isna()).fillna(True).astype(bool)
    if column == "treated_ntu":
        available = target_availability(frame)
        source = source.where(available)
        original_missing = ~available
    if column == "treated_ntu" and filled_target is not None:
        source = source.where(~original_missing, pd.to_numeric(filled_target, errors="coerce"))
    return source, original_missing | ~np.isfinite(source.to_numpy(dtype=float))


def build_sequence_tensors(frame, origins, targets, filled_target=None):
    origin_index = pd.DatetimeIndex(origins)
    target_values = np.asarray(targets, dtype=float)
    positions = frame.index.get_indexer(origin_index)
    values, missing = [], []
    for column in SEQUENCE_COLUMNS:
        source, source_missing = _source_column(frame, column, filled_target)
        values.append(source.to_numpy(dtype=float))
        missing.append(source_missing.to_numpy(dtype=bool))
    history_positions = positions[:, None] - HISTORY_STEPS + np.arange(HISTORY_STEPS)[None, :]
    return np.column_stack(values)[history_positions], np.column_stack(missing)[history_positions], target_values


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, len(HORIZONS)))

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class GRUExpert:
    HIDDEN_SIZES = (32, 64)
    DROPOUTS = (0.1, 0.2)
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    MAX_EPOCHS = 200
    EARLY_STOPPING_ROUNDS = 20
    GRADIENT_CLIP_NORM = 1.0
    AUGMENTATION_DAYS = (2, 14)

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
    def _validate_training_data(origins, targets):
        origin_index = pd.DatetimeIndex(origins)
        target_values = np.asarray(targets, dtype=float)
        return origin_index, target_values

    @staticmethod
    def _sample_weights(frame, origins):
        origin_index = pd.DatetimeIndex(origins)
        raw = pd.to_numeric(frame.get("raw_water_ntu"), errors="coerce").diff().reindex(origin_index)
        alum = pd.to_numeric(frame.get("alum_dosage"), errors="coerce").diff().reindex(origin_index)
        weights = np.ones(len(origin_index), dtype=np.float32)
        weights[np.abs(raw.to_numpy(dtype=float)) >= 50.0] *= 2.0
        weights[alum.fillna(0.0).to_numpy(dtype=float) != 0.0] *= 2.0
        return np.minimum(weights, 4.0)

    @staticmethod
    def _weighted_smooth_l1(prediction, target, weights):
        sample_loss = nn.SmoothL1Loss(reduction="none")(prediction, target).mean(dim=1)
        return (sample_loss * weights).mean()

    @classmethod
    def _augment_training_history(cls, x, mask, origins):
        augmented_x, augmented_mask = x.copy(), mask.copy()
        minimum, maximum = cls.AUGMENTATION_DAYS[0] * 12, cls.AUGMENTATION_DAYS[1] * 12
        available_steps = len(origins) + HISTORY_STEPS - 1
        limit = min(maximum, available_steps - 1)
        generator = np.random.default_rng(RANDOM_STATE)
        steps = int(generator.integers(minimum, limit + 1))
        start_offset = int(generator.integers(1, available_steps - steps + 1))
        block_start = pd.Timestamp(origins[0]) - pd.Timedelta(hours=2 * HISTORY_STEPS) + pd.Timedelta(hours=2 * start_offset)
        block_end = block_start + pd.Timedelta(hours=2 * (steps - 1))
        history_times = (pd.DatetimeIndex(origins).to_numpy()[:, None] - pd.to_timedelta(2 * (HISTORY_STEPS - np.arange(HISTORY_STEPS)), unit="h").to_numpy()[None, :])
        artificial = (history_times >= block_start.to_datetime64()) & (history_times <= block_end.to_datetime64())
        target_feature = SEQUENCE_COLUMNS.index("treated_ntu")
        target_values = augmented_x[:, :, target_feature]
        before_block = (history_times < block_start.to_datetime64()) & np.isfinite(target_values)
        latest_time = history_times[before_block].max()
        anchor = target_values[(history_times == latest_time) & np.isfinite(target_values)][0]
        eligible = artificial & ~augmented_mask[:, :, target_feature]
        augmented_x[:, :, target_feature][eligible] = anchor
        augmented_mask[:, :, target_feature][eligible] = True
        return augmented_x, augmented_mask, {"steps": steps, "start": block_start, "anchor": anchor}

    @staticmethod
    def _fit_transform(x, mask):
        count = x.shape[-1]
        medians = np.zeros(count, dtype=np.float32)
        for feature in range(count):
            available = x[:, :, feature][np.isfinite(x[:, :, feature])]
            if len(available):
                medians[feature] = float(np.median(available))
        filled = np.where(np.isfinite(x), x, medians[None, None, :]).astype(np.float32)
        means = filled.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        scales = filled.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        scales[scales < 1e-6] = 1.0
        return np.concatenate(((filled - means[None, None, :]) / scales[None, None, :], mask.astype(np.float32)), axis=2), medians, means, scales

    @staticmethod
    def _transform(x, mask, medians, means, scales):
        filled = np.where(np.isfinite(x), x, medians[None, None, :]).astype(np.float32)
        return np.concatenate(((filled - means[None, None, :]) / scales[None, None, :], mask.astype(np.float32)), axis=2)

    @classmethod
    def _train_candidate(cls, train_x, train_y, train_weights, valid_x, valid_y, hidden_size, dropout):
        cls._set_seed()
        model = GRUNet(train_x.shape[-1], hidden_size, dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cls.LEARNING_RATE)
        train_x_tensor, train_y_tensor = torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))
        train_weight_tensor = torch.from_numpy(train_weights)
        valid_x_tensor, valid_y_tensor = torch.from_numpy(valid_x), torch.from_numpy(valid_y.astype(np.float32))
        best_mae, best_epoch, best_state = None, 0, None
        for epoch in range(cls.MAX_EPOCHS):
            model.train()
            for start in range(0, len(train_x_tensor), cls.BATCH_SIZE):
                stop = min(start + cls.BATCH_SIZE, len(train_x_tensor))
                loss = cls._weighted_smooth_l1(
                    model(train_x_tensor[start:stop]),
                    train_y_tensor[start:stop],
                    train_weight_tensor[start:stop],
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cls.GRADIENT_CLIP_NORM)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_mae = float(torch.abs(model(valid_x_tensor) - valid_y_tensor).mean().item())
            if best_mae is None or validation_mae < best_mae:
                best_mae, best_epoch = validation_mae, epoch
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            if epoch - best_epoch >= cls.EARLY_STOPPING_ROUNDS:
                break
        model.load_state_dict(best_state)
        return model, best_mae, best_epoch + 1

    @classmethod
    def _fit_final_model(cls, x, targets, weights, hidden_size, dropout, epochs):
        cls._set_seed()
        model = GRUNet(x.shape[-1], hidden_size, dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cls.LEARNING_RATE)
        x_tensor, target_tensor, weight_tensor = torch.from_numpy(x), torch.from_numpy(targets.astype(np.float32)), torch.from_numpy(weights)
        model.train()
        for _ in range(epochs):
            for start in range(0, len(x_tensor), cls.BATCH_SIZE):
                stop = min(start + cls.BATCH_SIZE, len(x_tensor))
                loss = cls._weighted_smooth_l1(
                    model(x_tensor[start:stop]), target_tensor[start:stop], weight_tensor[start:stop]
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cls.GRADIENT_CLIP_NORM)
                optimizer.step()
        model.eval()
        return model

    @staticmethod
    def _validation_frame(frame, validation_start):
        protected = frame.copy()
        if "treated_ntu" in protected:
            available = target_availability(protected)
            available.loc[protected.index >= validation_start] = False
            protected.loc[protected.index >= validation_start, "treated_ntu"] = np.nan
            protected["target_available"] = available
            protected["missing_treated_ntu"] = ~available
        return protected

    def fit(self, frame, origins, targets, filled_target=None):
        origin_index, target_values = self._validate_training_data(origins, targets)
        validation_size = max(1, int(np.ceil(len(origin_index) * 0.2)))
        offset = len(origin_index) - validation_size
        validation_start = origin_index[offset]
        candidates = origin_index[:offset]
        label_end = candidates + pd.Timedelta(hours=2 * max(HORIZONS))
        keep = label_end < validation_start
        train_origins, train_y = candidates[keep], target_values[:offset][keep]
        train_x_raw, train_mask, _ = build_sequence_tensors(frame, train_origins, train_y, filled_target)
        train_x_raw, train_mask, self.validation_augmentation_ = self._augment_training_history(train_x_raw, train_mask, train_origins)
        train_x, medians, means, scales = self._fit_transform(train_x_raw, train_mask)
        protected = self._validation_frame(frame, validation_start)
        valid_x_raw, valid_mask, _ = build_sequence_tensors(protected, origin_index[offset:], target_values[offset:], filled_target)
        valid_x = self._transform(valid_x_raw, valid_mask, medians, means, scales)
        valid_y = target_values[offset:]
        self.validation_start_, self.validation_train_origins_, self.validation_train_label_end_ = validation_start, train_origins, label_end[keep].max()
        self.validation_inputs_ = valid_x.copy()
        best = None
        for hidden_size in self.HIDDEN_SIZES:
            for dropout in self.DROPOUTS:
                _, mae, epochs = self._train_candidate(train_x, train_y, self._sample_weights(frame, train_origins), valid_x, valid_y, hidden_size, dropout)
                candidate = (mae, hidden_size, dropout, epochs)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        self.best_validation_mae_, self.hidden_size_, self.dropout_, self.epochs_ = best
        full_x_raw, full_mask, _ = build_sequence_tensors(frame, origin_index, target_values, filled_target)
        full_x_raw, full_mask, self.augmentation_block_ = self._augment_training_history(full_x_raw, full_mask, origin_index)
        full_x, self.input_medians_, self.input_means_, self.input_scales_ = self._fit_transform(full_x_raw, full_mask)
        self.model_ = self._fit_final_model(full_x, target_values, self._sample_weights(frame, origin_index), self.hidden_size_, self.dropout_, self.epochs_)
        self.fit_end_, self.input_size_, self.is_fitted_ = origin_index.max(), full_x.shape[-1], True
        self.available_target_end_ = self.fit_end_
        return self

    def predict(self, frame, origins, filled_target=None):
        available_target_end = getattr(self, "available_target_end_", self.fit_end_)
        safe_filled_target = filled_target
        if filled_target is not None:
            safe_filled_target = filled_target.copy()
            blocked_observed = target_availability(frame) & (
                frame.index > available_target_end
            )
            safe_filled_target.loc[blocked_observed] = np.nan
        protected = self._validation_frame(
            frame, available_target_end + pd.Timedelta(hours=2)
        )
        x, mask, _ = build_sequence_tensors(protected, origins, np.empty((len(origins), len(HORIZONS))), safe_filled_target)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy(self._transform(x, mask, self.input_medians_, self.input_means_, self.input_scales_))).numpy()

    def state_dict_bundle(self):
        return {"state_dict": {name: value.detach().clone() for name, value in self.model_.state_dict().items()}, "input_size": self.input_size_, "hidden_size": self.hidden_size_, "dropout": self.dropout_, "epochs": self.epochs_, "fit_end": self.fit_end_, "available_target_end": getattr(self, "available_target_end_", self.fit_end_), "input_medians": self.input_medians_.copy(), "input_means": self.input_means_.copy(), "input_scales": self.input_scales_.copy(), "sequence_columns": SEQUENCE_COLUMNS}
