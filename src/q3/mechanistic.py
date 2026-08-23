import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .data import target_availability
def cstr_cascade(inlet, flow, level, n_tanks, volume_scale, initial):
    inlet = np.asarray(inlet, dtype=float)
    flow = np.asarray(flow, dtype=float)
    level = np.asarray(level, dtype=float)
    effective_volume = np.maximum(level * volume_scale, 1e-6)
    alpha = np.exp(-np.maximum(flow, 1e-6) * 2.0 / effective_volume)
    states = np.empty((len(inlet), n_tanks), dtype=float)
    previous = np.full(n_tanks, initial, dtype=float)
    for i in range(len(inlet)):
        previous[0] = alpha[i] * previous[0] + (1 - alpha[i]) * inlet[i]
        for j in range(1, n_tanks):
            previous[j] = alpha[i] * previous[j] + (1 - alpha[i]) * previous[j - 1]
        states[i] = previous
    return states[:, -1]
class MechanisticExpert:
    _N_TANKS = (1, 2, 3)
    _VOLUME_SCALES = (1.0, 25.0, 100.0, 500.0, 1000.0, 5000.0)
    def __init__(self):
        self.is_fitted_ = False
    @staticmethod
    def _column(frame, names):
        for name in names:
            if name in frame:
                return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype=float)

    def _process_inputs(self, frame):
        inlet = self._column(frame, ("filtered_ntu", "raw_water_ntu"))
        flow = self._column(frame, ("treated_water_flow", "raw_water_flow"))
        level = self._column(frame, ("clear_well_level",))
        values = (inlet, flow, level)
        return tuple(
            series.fillna(self.input_medians_[name]).to_numpy(dtype=float)
            for name, series in zip(("inlet", "flow", "level"), values)
        )

    @staticmethod
    def _cstr_state(inlet, flow, level, n_tanks, volume_scale):
        initial = float(inlet[0]) if len(inlet) else 0.0
        return cstr_cascade(inlet, flow, level, n_tanks, volume_scale, initial)

    @staticmethod
    def _design_matrix(frame, state, flow):
        hour = frame.index.hour.to_numpy(dtype=float) + frame.index.minute.to_numpy(dtype=float) / 60.0
        weekday_hour = frame.index.weekday.to_numpy(dtype=float) * 24.0 + hour
        backwash = (
            frame["is_backwash_event"].fillna(False).astype(float).to_numpy()
            if "is_backwash_event" in frame
            else np.zeros(len(frame), dtype=float)
        )
        return np.column_stack(
            (
                state,
                np.sin(2.0 * np.pi * hour / 24.0),
                np.cos(2.0 * np.pi * hour / 24.0),
                np.sin(2.0 * np.pi * weekday_hour / (7.0 * 24.0)),
                np.cos(2.0 * np.pi * weekday_hour / (7.0 * 24.0)),
                flow,
                backwash,
            )
        )

    def _select_cstr_parameters(self, inlet, flow, level, target, index):
        if np.isfinite(target).sum() < 4:
            self.selection_folds_ = []
            return 1, 100.0

        validation_width = max(1, len(target) // 12)
        starts = [len(target) - 3 * validation_width + fold * validation_width for fold in range(3)]
        folds = []
        diagnostics = []
        for start in starts:
            stop = min(start + validation_width, len(target))
            if start <= 0 or stop <= start:
                continue
            fit_positions = np.flatnonzero(np.isfinite(target[:start]))
            score_positions = np.flatnonzero(np.isfinite(target[start:stop])) + start
            if len(fit_positions) < 2 or not len(score_positions):
                continue
            folds.append((fit_positions, score_positions))
            diagnostics.append(
                {
                    "train_end": index[start - 1],
                    "validation_start": index[start],
                    "validation_end": index[stop - 1],
                    "n_train": int(len(fit_positions)),
                    "n_validation": int(len(score_positions)),
                }
            )
        self.selection_folds_ = diagnostics
        if not folds:
            return 1, 100.0

        best = None
        for n_tanks in self._N_TANKS:
            for volume_scale in self._VOLUME_SCALES:
                state = self._cstr_state(inlet, flow, level, n_tanks, volume_scale)
                errors = []
                for fit_positions, score_positions in folds:
                    coefficients, *_ = np.linalg.lstsq(
                        np.column_stack((np.ones(len(fit_positions)), state[fit_positions])),
                        target[fit_positions],
                        rcond=None,
                    )
                    prediction = coefficients[0] + coefficients[1] * state[score_positions]
                    errors.extend(np.abs(target[score_positions] - prediction))
                candidate = (float(np.mean(errors)), n_tanks, volume_scale)
                if best is None or candidate < best:
                    best = candidate
        return best[1], best[2]

    def fit(self, frame, train_end):
        self.train_end_ = pd.Timestamp(train_end)
        training = frame.loc[: self.train_end_]
        inlet = self._column(training, ("filtered_ntu", "raw_water_ntu"))
        flow = self._column(training, ("treated_water_flow", "raw_water_flow"))
        level = self._column(training, ("clear_well_level",))
        self.input_medians_ = {
            "inlet": float(inlet.median()) if np.isfinite(inlet.median()) else 0.0,
            "flow": float(flow.median()) if np.isfinite(flow.median()) else 1.0,
            "level": float(level.median()) if np.isfinite(level.median()) else 1.0,
        }
        inlet_values, flow_values, level_values = self._process_inputs(training)
        target = pd.to_numeric(training["treated_ntu"], errors="coerce").where(
            target_availability(training)
        ).to_numpy(dtype=float)
        self.n_tanks_, self.volume_scale_ = self._select_cstr_parameters(
            inlet_values, flow_values, level_values, target, training.index
        )
        state = self._cstr_state(
            inlet_values, flow_values, level_values, self.n_tanks_, self.volume_scale_
        )
        exog = self._design_matrix(training, state, flow_values)
        valid = np.isfinite(target)
        self._ols_coefficients_, *_ = np.linalg.lstsq(
            np.column_stack((np.ones(valid.sum()), exog[valid])), target[valid], rcond=None
        )
        self.exog_center_ = exog[valid].mean(axis=0)
        self.exog_scale_ = exog[valid].std(axis=0)
        self.exog_scale_[self.exog_scale_ < 1e-8] = 1.0
        model_exog = (exog - self.exog_center_) / self.exog_scale_
        try:
            result = SARIMAX(
                target,
                exog=model_exog,
                order=(1, 0, 0),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200, warn_convergence=False)
            self.results_ = result if result.mle_retvals.get("converged", False) else None
        except (ValueError, np.linalg.LinAlgError):
            self.results_ = None
        self.is_fitted_ = True
        return self
    def _forecast_values(self, frame):
        train_position = frame.index.get_loc(self.train_end_)
        inlet, flow, level = self._process_inputs(frame)
        state = self._cstr_state(inlet, flow, level, self.n_tanks_, self.volume_scale_)
        exog = self._design_matrix(frame, state, flow)
        forecast = np.full(len(frame), np.nan, dtype=float)
        future_exog = exog[train_position + 1 :]
        if len(future_exog):
            if self.results_ is None:
                forecast[train_position + 1 :] = np.column_stack(
                    (np.ones(len(future_exog)), future_exog)
                ) @ self._ols_coefficients_
            else:
                forecast[train_position + 1 :] = np.asarray(
                    self.results_.get_forecast(
                        steps=len(future_exog),
                        exog=(future_exog - self.exog_center_) / self.exog_scale_,
                    ).predicted_mean,
                    dtype=float,
                )
        return forecast

    def predict(self, frame, origins):
        forecast = self._forecast_values(frame)
        origin_index = pd.DatetimeIndex(origins)
        positions = frame.index.get_indexer(origin_index)
        horizon_positions = positions[:, None] + np.arange(1, 7)[None, :]
        return forecast[horizon_positions]

    def fill_target_history(self, frame):
        observed = pd.to_numeric(frame["treated_ntu"], errors="coerce").where(
            target_availability(frame)
        )
        filled = observed.copy()
        train_position = frame.index.get_loc(self.train_end_)
        if self.results_ is None:
            inlet, flow, level = self._process_inputs(frame)
            state = self._cstr_state(inlet, flow, level, self.n_tanks_, self.volume_scale_)
            all_prediction = np.column_stack((np.ones(len(frame)), self._design_matrix(frame, state, flow))) @ self._ols_coefficients_
            fitted = all_prediction[: train_position + 1]
        else:
            fitted = np.asarray(self.results_.fittedvalues, dtype=float)
            if not np.isfinite(fitted).all():
                inlet, flow, level = self._process_inputs(frame)
                state = self._cstr_state(inlet, flow, level, self.n_tanks_, self.volume_scale_)
                fallback = np.column_stack(
                    (np.ones(len(frame)), self._design_matrix(frame, state, flow))
                ) @ self._ols_coefficients_
                fitted = np.where(np.isfinite(fitted), fitted, fallback[: train_position + 1])
        pre_missing = filled.iloc[: train_position + 1].isna().to_numpy()
        values = filled.to_numpy(copy=True)
        values[: train_position + 1][pre_missing] = fitted[pre_missing]
        forecast = self._forecast_values(frame)
        post_missing = filled.iloc[train_position + 1 :].isna().to_numpy()
        values[train_position + 1 :][post_missing] = forecast[train_position + 1 :][post_missing]
        return pd.Series(values, index=frame.index, name="treated_ntu")

    def to_state(self):
        return {
            "train_end": self.train_end_,
            "n_tanks": self.n_tanks_,
            "volume_scale": self.volume_scale_,
            "input_medians": self.input_medians_.copy(),
            "selection_folds": [fold.copy() for fold in self.selection_folds_],
            "state_space_parameters": (
                np.asarray(self.results_.params, dtype=float).tolist()
                if self.results_ is not None
                else self._ols_coefficients_.tolist()
            ),
        }