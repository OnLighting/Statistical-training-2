from itertools import product
import lightgbm as lgb
import numpy as np
import pandas as pd
from .data import HORIZONS, RANDOM_STATE, make_feature_frame, target_availability
class LightGBMExpert:
    NUM_LEAVES = (15, 31)
    LEARNING_RATES = (0.03, 0.05)
    MIN_CHILD_SAMPLES = (20, 40)
    FEATURE_FRACTIONS = (0.8, 1.0)
    SUBSAMPLE = 0.8
    MAX_ESTIMATORS = 800
    EARLY_STOPPING_ROUNDS = 50
    def __init__(self):
        self.is_fitted_ = False
    @staticmethod
    def _validate_training_data(origins, y):
        return pd.DatetimeIndex(origins), np.asarray(y, dtype=float)

    @staticmethod
    def _numeric_features(features, origins):
        selected = features.loc[pd.DatetimeIndex(origins)]
        return selected.apply(pd.to_numeric, errors="coerce")

    @staticmethod
    def _sample_weights(features, targets):
        raw_change = features.get("raw_water_ntu_change1")
        alum_change = features.get("alum_dosage_change1")
        raw_event = (
            raw_change.abs().to_numpy(dtype=float) >= 50.0
            if raw_change is not None
            else np.zeros(len(features), dtype=bool)
        )
        alum_event = (
            alum_change.fillna(0.0).to_numpy(dtype=float) != 0.0
            if alum_change is not None
            else np.zeros(len(features), dtype=bool)
        )
        weights = np.ones(len(features), dtype=float)
        weights[raw_event] *= 2.0
        weights[alum_event] *= 2.0
        threshold = float(np.quantile(targets, 0.95))
        weights[targets >= threshold] *= 2.0
        return np.minimum(weights, 4.0)

    @classmethod
    def _grid(cls):
        return tuple(
            {
                "num_leaves": leaves,
                "learning_rate": learning_rate,
                "min_child_samples": min_child_samples,
                "feature_fraction": feature_fraction,
            }
            for leaves, learning_rate, min_child_samples, feature_fraction in product(
                cls.NUM_LEAVES,
                cls.LEARNING_RATES,
                cls.MIN_CHILD_SAMPLES,
                cls.FEATURE_FRACTIONS,
            )
        )

    @classmethod
    def _regressor(cls, parameters, n_estimators=None):
        return lgb.LGBMRegressor(
            objective="huber",
            n_estimators=cls.MAX_ESTIMATORS if n_estimators is None else n_estimators,
            subsample=cls.SUBSAMPLE,
            subsample_freq=1,
            random_state=RANDOM_STATE,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **parameters,
        )

    def _select_parameters(self, x, y, origins):
        validation_size = max(1, int(np.ceil(len(x) * 0.2)))
        train_size = len(x) - validation_size
        validation_start = origins[train_size]
        training_origins = origins[:train_size]
        label_end = training_origins + pd.Timedelta(hours=2 * max(HORIZONS))
        keep_training = label_end < validation_start
        # if keep_training.sum() < 4:
        #     raise ValueError("at least four origins are required before validation")
        train_x = x.iloc[:train_size].iloc[np.flatnonzero(keep_training)]
        valid_x = x.iloc[train_size:]
        train_y = y[:train_size][keep_training]
        valid_y = y[train_size:]
        self.validation_start_ = validation_start
        self.validation_train_origins_ = training_origins[keep_training]
        self.validation_train_label_end_ = label_end[keep_training].max()
        scores = []
        best_parameters = None
        best_score = None
        best_iterations = []

        for parameters in self._grid():
            errors = []
            iterations = []
            for horizon_index in range(len(HORIZONS)):
                model = self._regressor(parameters)
                model.fit(
                    train_x,
                    train_y[:, horizon_index],
                    sample_weight=self._sample_weights(train_x, train_y[:, horizon_index]),
                    eval_set=[(valid_x, valid_y[:, horizon_index])],
                    eval_metric="mae",
                    callbacks=[lgb.early_stopping(self.EARLY_STOPPING_ROUNDS, verbose=False)],
                )
                prediction = model.predict(valid_x, num_iteration=model.best_iteration_)
                errors.append(np.mean(np.abs(valid_y[:, horizon_index] - prediction)))
                iterations.append(model.best_iteration_ or self.MAX_ESTIMATORS)
            score = float(np.mean(errors))
            scores.append({"parameters": parameters.copy(), "mae": score})
            if best_score is None or score < best_score:
                best_parameters = parameters
                best_score = score
                best_iterations = iterations

        self.validation_scores_ = pd.DataFrame(scores)
        self.validation_train_size_ = len(train_x)
        self.best_params_ = best_parameters.copy()
        return best_parameters, best_iterations

    def fit(self, frame, origins, y):
        origin_index, targets = self._validate_training_data(origins, y)
        self.feature_names_ = None
        features = make_feature_frame(frame, origin_index.max())
        x = self._numeric_features(features, origin_index)
        self.feature_names_ = list(x.columns)
        parameters, best_iterations = self._select_parameters(x, targets, origin_index)
        self.models_ = []
        self.best_iterations_ = []

        for horizon_index in range(len(HORIZONS)):
            iterations = int(min(self.MAX_ESTIMATORS, max(1, best_iterations[horizon_index])))
            model = self._regressor(parameters, n_estimators=iterations)
            model.fit(
                x,
                targets[:, horizon_index],
                sample_weight=self._sample_weights(x, targets[:, horizon_index]),
            )
            self.models_.append(model)
            self.best_iterations_.append(iterations)

        self.fit_end_ = origin_index.max()
        self.available_target_end_ = self.fit_end_
        self.is_fitted_ = True
        return self

    def prediction_features(self, frame, origins, filled_target=None):
        # if not self.is_fitted_:
        #     raise RuntimeError("fit must be called before prediction")
        origin_index = pd.DatetimeIndex(origins)
        # if not origin_index.isin(frame.index).all():
        #     raise KeyError("all origins must be present in frame.index")
        prediction_frame = frame.copy()
        if "treated_ntu" in prediction_frame:
            structurally_available = target_availability(prediction_frame)
            available_target_end = getattr(
                self, "available_target_end_", self.fit_end_
            )
            post_fit = prediction_frame.index > available_target_end
            available = structurally_available & ~post_fit
            prediction_frame["treated_ntu"] = pd.to_numeric(
                prediction_frame["treated_ntu"], errors="coerce"
            ).where(available)
            if filled_target is not None:
                # if not isinstance(filled_target, pd.Series):
                #     raise TypeError("filled_target must be a pandas Series from the mechanistic expert")
                # if not filled_target.index.equals(prediction_frame.index):
                #     raise ValueError("filled_target index must match frame.index")
                use_mechanistic_fill = ~structurally_available
                prediction_frame.loc[use_mechanistic_fill, "treated_ntu"] = pd.to_numeric(
                    filled_target.loc[use_mechanistic_fill], errors="coerce"
                )
            prediction_frame["target_available"] = available
            prediction_frame["missing_treated_ntu"] = ~available
        features = make_feature_frame(prediction_frame, self.fit_end_)
        x = self._numeric_features(features, origin_index).reindex(columns=self.feature_names_)
        return x

    def predict(self, frame, origins, filled_target=None):
        # """Predict six horizons from the protected transformed feature rows."""
        x = self.prediction_features(frame, origins, filled_target)
        return np.column_stack([model.predict(x) for model in self.models_])

    def feature_importance(self):
        # """Return gain importance for every feature and direct horizon."""
        # if not self.is_fitted_:
        #     raise RuntimeError("fit must be called before feature_importance")
        records = []
        for horizon, model in zip(HORIZONS, self.models_):
            importance = model.booster_.feature_importance(importance_type="gain")
            for feature, value in zip(self.feature_names_, importance):
                records.append(
                    {"horizon_hours": horizon * 2, "feature": feature, "importance": value}
                )
        return pd.DataFrame(records)

    def booster_paths(self):
        # """Return the standard six relative artifact names for this expert."""
        return tuple("lightgbm_h{:02d}.txt".format(horizon * 2) for horizon in HORIZONS)
